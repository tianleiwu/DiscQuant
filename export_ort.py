"""Export DiscQuant-quantized models to an ONNX Runtime friendly checkpoint.

DiscQuant's quantization grid (``quantutils.GroupFinite``) dequantizes as

    W = (q - zero) * scale,   q in [0, 2**wbits - 1]

quantized along the input-feature (K) dimension, either block-wise
(``groupsize > 0``) or per-output-channel (``groupsize == -1``). This is exactly
the affine integer grid that ONNX Runtime's ``MatMulNBits`` (and ``QMoE``)
operators consume, and it is byte-for-byte compatible with the AutoGPTQ weight
layout that the onnxruntime-genai model builder already knows how to read
(``GPTQModel`` in ``quantized_model.py``).

This module walks a trained DiscQuant model (whose quantized ``nn.Linear`` layers
are wrapped by ``linearutils.quantize_linearlayer_multimode``), recovers the exact
integer codes / scales / zero-points from each layer's grid, packs them into the
AutoGPTQ tensor layout, and writes a Hugging Face checkpoint whose
``config.json`` advertises ``quantization_config = {"quant_method": "discquant",
...}``.

The exported checkpoint can then be consumed directly by the onnxruntime-genai
model builder (``builder.py``), which dispatches ``quant_method == "discquant"``
to ``DiscQuantModel`` and produces ``MatMulNBits`` (dense) ONNX models.

Only the block-scaling (non-QuIP) path is supported: QuIP applies a random
Hadamard rotation to the weights that would require a matching rotation on the
activations at inference time, which ``MatMulNBits``/``QMoE`` do not perform.
"""

import os

import torch

# The wrapper class that DiscQuant uses for every quantized linear layer.
from linearutils import quantize_linearlayer_multimode


def grid_params(grid):
    """Extract normalized quantization parameters from a ``GroupFinite`` grid.

    Returns ``(scale, zero, group_size, num_groups, maxq, out_features,
    in_features)`` where ``scale`` and ``zero`` are ``float32`` CPU tensors of
    shape ``[out_features, num_groups]``.
    """
    out_features = int(grid.rows)
    in_features = int(grid.cols)
    maxq = int(grid.maxq)

    if grid.groupsize == -1:
        group_size = in_features
        num_groups = 1
    else:
        group_size = int(grid.groupsize)
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features={in_features} is not divisible by group_size={group_size}."
            )
        num_groups = in_features // group_size

    scale = grid.scales.detach().to(torch.float32).cpu().reshape(out_features, num_groups)
    zero = grid.zeros.detach().to(torch.float32).cpu().reshape(out_features, num_groups)
    return scale, zero, group_size, num_groups, maxq, out_features, in_features


def recover_int_codes(weight_q, scale, zero, group_size, maxq):
    """Recover integer codes ``q in [0, maxq]`` from on-grid dequantized weights.

    ``weight_q`` has shape ``[out_features, in_features]`` and is assumed to lie
    exactly on the quantization grid (the DiscQuant solution). ``scale`` / ``zero``
    have shape ``[out_features, num_groups]``.
    """
    out_features, in_features = weight_q.shape
    num_groups = scale.shape[1]
    wg = weight_q.reshape(out_features, num_groups, group_size)
    s = scale.reshape(out_features, num_groups, 1)
    z = zero.reshape(out_features, num_groups, 1)
    q = torch.round(wg / s) + z
    q = q.clamp(0, maxq).reshape(out_features, in_features)
    return q.to(torch.int64)


def pack_gptq(q, scale, zero, bits):
    """Pack integer codes into the AutoGPTQ tensor layout.

    Args:
        q: ``[out_features, in_features]`` integer codes in ``[0, 2**bits - 1]``.
        scale: ``[out_features, num_groups]`` float scales.
        zero: ``[out_features, num_groups]`` integer zero-points (the *true* zero,
            e.g. 8 for symmetric 4-bit). AutoGPTQ stores ``zero - 1`` on disk;
            this function applies that offset.
        bits: 2, 4, or 8.

    Returns a dict with AutoGPTQ-named tensors:
        ``qweight`` int32 ``[in_features // pack, out_features]``
        ``qzeros``  int32 ``[num_groups, out_features // pack]``
        ``scales``  float16 ``[num_groups, out_features]``
        ``g_idx``   int32 ``[in_features]``
    """
    if bits not in (2, 4, 8):
        raise NotImplementedError(f"Only 2/4/8-bit packing is supported, got {bits}.")

    out_features, in_features = q.shape
    num_groups = scale.shape[1]
    pack = 32 // bits
    group_size = in_features // num_groups

    if in_features % pack != 0:
        raise ValueError(f"in_features={in_features} must be divisible by {pack} for {bits}-bit packing.")
    if out_features % pack != 0:
        raise ValueError(f"out_features={out_features} must be divisible by {pack} for {bits}-bit packing.")

    shifts = (bits * torch.arange(pack, dtype=torch.int64)).reshape(1, 1, pack)

    # qweight: pack `pack` consecutive K values of each output column into one int32.
    qk = q.reshape(out_features, in_features // pack, pack)  # [N, K/pack, pack]
    packed_w = (qk << shifts).sum(dim=-1)  # [N, K/pack]
    qweight = packed_w.t().contiguous().to(torch.int32)  # [K/pack, N]

    # qzeros: store (zero - 1), pack `pack` consecutive N values per int32.
    maxv = (1 << bits) - 1
    zc = (zero.round().to(torch.int64) - 1).clamp(0, maxv)  # [N, num_groups]
    zcg = zc.t().contiguous().reshape(num_groups, out_features // pack, pack)  # [G, N/pack, pack]
    packed_z = (zcg << shifts).sum(dim=-1)  # [G, N/pack]
    qzeros = packed_z.to(torch.int32)  # [G, N/pack]

    scales = scale.t().contiguous().to(torch.float16)  # [G, N]
    g_idx = (torch.arange(in_features, dtype=torch.int32) // group_size).to(torch.int32)

    return {"qweight": qweight, "qzeros": qzeros, "scales": scales, "g_idx": g_idx}


def export_layer(module, bits=None, groupsize=None):
    """Pack a single DiscQuant-wrapped linear layer into AutoGPTQ tensors."""
    grid = module.grid
    bits = grid.wbits if bits is None else bits
    if groupsize is not None and groupsize != grid.groupsize:
        raise ValueError(
            f"Requested groupsize={groupsize} does not match the layer grid groupsize={grid.groupsize}."
        )

    # The DiscQuant solution: round the interpolation parameter and snap to the grid.
    with torch.no_grad():
        weight_q = module._unquant(module.x, rd=True).detach().to(torch.float32).cpu()

    scale, zero, group_size, _num_groups, maxq, _n, _k = grid_params(grid)
    q = recover_int_codes(weight_q, scale, zero, group_size, maxq)
    tensors = pack_gptq(q, scale, zero, bits)

    bias = getattr(module.linear, "bias", None)
    if bias is not None:
        tensors["bias"] = bias.detach().to(torch.float16).cpu()
    return tensors


def _collect_wrapped_layers(model):
    """Map dotted module name -> wrapper for every quantized linear layer."""
    return {
        name: m
        for name, m in model.named_modules()
        if isinstance(m, quantize_linearlayer_multimode)
    }


def save_discquant_gptq(model, tokenizer, model_id, savedir, dtype, args):
    """Write a ``quant_method="discquant"`` AutoGPTQ-format checkpoint.

    Args:
        model: the trained DiscQuant model (with wrapped quantized linears).
        tokenizer: tokenizer to persist alongside the checkpoint.
        model_id: original Hugging Face model id (used to load the float skeleton
            for the un-quantized tensors: norms, embeddings, lm_head, biases).
        savedir: output directory.
        dtype: torch dtype of the float skeleton (``torch.float16`` / ``bfloat16``).
        args: the DiscQuant args namespace (uses ``wbits``, ``groupsize``, ``quip``).
    """
    import transformers
    from safetensors.torch import save_file

    if getattr(args, "quip", False):
        raise NotImplementedError(
            "QuIP (Hadamard rotation) checkpoints are not exportable to MatMulNBits/QMoE: "
            "the operators do not apply the matching activation rotation. Re-run DiscQuant "
            "without --quip (block-scaling mode) to export an ORT-compatible checkpoint."
        )

    os.makedirs(savedir, exist_ok=True)
    wrapped = _collect_wrapped_layers(model)
    if not wrapped:
        raise ValueError("No quantized DiscQuant layers found in the model.")

    # Load a clean float skeleton to source the un-quantized tensors.
    skeleton = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )
    state = {k: v.detach().cpu() for k, v in skeleton.state_dict().items()}

    bits = args.wbits
    group_size = args.groupsize
    for name, module in wrapped.items():
        tensors = export_layer(module, bits=bits)
        # Replace the float weight with the packed AutoGPTQ tensors.
        state.pop(f"{name}.weight", None)
        for suffix, tensor in tensors.items():
            state[f"{name}.{suffix}"] = tensor.contiguous()

    # Drop any tied-weight duplicates that safetensors cannot serialize.
    state = {k: v.contiguous() for k, v in state.items()}
    save_file(state, os.path.join(savedir, "model.safetensors"), metadata={"format": "pt"})

    # Persist a config that advertises the DiscQuant quantization method.
    config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # DiscQuant's GroupFinite configures the GPTQ Quantizer with sym=True (its default),
    # so the exported grid is symmetric (zero == (2**bits)/2). This is informational for
    # the genai reader (it decodes zero-points from qzeros regardless).
    config.quantization_config = {
        "quant_method": "discquant",
        "bits": int(bits),
        "group_size": int(group_size),
        "sym": True,
        "desc_act": False,
        "checkpoint_format": "gptq",
    }
    config.save_pretrained(savedir)
    if tokenizer is not None:
        tokenizer.save_pretrained(savedir)
    print(f"Saved DiscQuant ORT checkpoint ({len(wrapped)} quantized layers) to {savedir}")
