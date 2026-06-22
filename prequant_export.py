"""Export DiscQuant-quantized attention as a *pre-quantized* sidecar that the
onnxruntime-genai model builder consumes **without re-quantizing**.

Motivation
----------
For a fused-MoE model (gpt-oss) DiscQuant quantizes only the dense attention
projections (q/k/v/o_proj) and hands genai a plain fp16 checkpoint. genai then
re-quantizes attention with RTN. That re-quant is the exact fixed point of an
*asymmetric* grid but **not** of a *symmetric* one (symmetric RTN recomputes the
scale as ``2*amax/15`` and dilates it by 16/15 -> ~6.7% error). Asymmetric is
therefore lossless but stores per-block zero-points (slower MatMulNBits).

To get **symmetric (fast) AND lossless**, we avoid re-quantization entirely:
genai's ``make_matmul_int4`` already emits ``MatMulNBits`` *directly* from a
module that carries ``.qweight``/``.scales`` (no RTN pass). This module packs
DiscQuant's exact symmetric integer codes + scales into ORT's ``MatMulNBits``
layout (byte-for-byte identical to ``quantized_model.pack_ort_format``) and
writes them to a sidecar. A small overlay in the genai builder attaches them to
the attention ``nn.Linear`` modules so the direct-read path fires. Symmetric
``MatMulNBits`` uses the grid ``value = (q - 2**(bits-1)) * scale`` -- identical
to DiscQuant's symmetric grid -- so the round-trip is exact, with **no stored
zero-points**.

Per-layer bit-width
-------------------
Each projection may be int4 or int8 (``bits`` is stored per tensor and flows
through to ``MatMulNBits`` via the module's ``.bits``). See ``resolve_layer_bits``
for the placement policy (uniform, or the llama.cpp "int8_mixed" sensitivity
set, mirrored from genai's ``int8_mixed_layers``).
"""

import json
import os

import torch

from export_ort import grid_params, recover_int_codes


# ---------------------------------------------------------------------------
# Per-layer bit-width placement
# ---------------------------------------------------------------------------
def int8_mixed_layer_ids(num_layers):
    """Layer ids promoted to int8 by the llama.cpp "mixed" strategy.

    Mirrors onnxruntime-genai ``make_bit_placement_config`` (``int8_mixed_layers``):
    the first and last eighth of layers, plus every third layer thereafter.
    Reference: llama.cpp ``llama-quant.cpp`` (mixed quant heuristic).
    """
    return [
        i
        for i in range(num_layers)
        if i < num_layers / 8
        or i >= 7 * num_layers / 8
        or (i - round(num_layers / 8)) % 3 == 2
    ]


# Projections promoted to int8 on an upgraded layer. genai's set is
# {qkv_proj, v_proj, down_proj}; for DiscQuant's attention-only scope we map that
# to the q/k/v projections that are actually quantized here (o_proj and the
# experts' down_proj are out of scope -- experts go through QMoE).
_INT8_MIXED_PROJS = ("q_proj", "k_proj", "v_proj")


def resolve_layer_bits(args, layer_id, proj_name, num_layers):
    """Return the bit-width (4 or 8) for one (layer, projection).

    Policy is selected by ``args.bit_placement``:
      * ``uniform``    -> ``args.wbits`` everywhere (default).
      * ``int8_mixed`` -> ``args.wbits`` except the llama.cpp sensitivity set,
                          which is promoted to int8.
    ``proj_name`` is the dotted leaf of the quantlist entry (e.g. ``q_proj`` from
    ``self_attn.q_proj``).
    """
    placement = getattr(args, "bit_placement", "uniform") or "uniform"
    base = int(args.wbits)
    if placement == "uniform":
        return base
    if placement == "int8_mixed":
        leaf = proj_name.split(".")[-1]
        if layer_id in set(int8_mixed_layer_ids(num_layers)) and leaf in _INT8_MIXED_PROJS:
            return 8
        return base
    raise ValueError(f"Unknown bit_placement '{placement}'. Expected 'uniform' or 'int8_mixed'.")


# ---------------------------------------------------------------------------
# ORT MatMulNBits packing (byte-for-byte match to quantized_model.pack_ort_format)
# ---------------------------------------------------------------------------
def _pack_on_row_uint8(mat, bits):
    """Pack integer codes along the column (K) axis into uint8 blobs, LSB-first.

    ``mat`` is ``[R, C]`` with codes in ``[0, 2**bits - 1]``. Within each group of
    ``8 // bits`` consecutive columns, the first column occupies the low bits.
    Matches ``pack_on_row_for_2_4_8_bits(..., packed_dtype=torch.uint8)``.
    """
    mat = mat.to(torch.int64)
    R, C = mat.shape
    vpp = 8 // bits
    pad = (vpp - (C % vpp)) % vpp
    if pad:
        mat = torch.nn.functional.pad(mat, (0, pad), "constant", 0)
    C2 = mat.shape[1]
    out = torch.zeros((R, C2 // vpp), dtype=torch.int64)
    for j in range(vpp):
        out |= (mat[:, j::vpp] & ((1 << bits) - 1)) << (j * bits)
    return out.to(torch.uint8)


def pack_ort_matmulnbits(q, scale, bits, group_size):
    """Pack codes ``q`` [N, K] and ``scale`` [N, G] into ORT ``MatMulNBits`` tensors.

    Returns ``(qweight_uint8 [N, k_blocks, blob_size], scales_flat [N*k_blocks])``.
    Symmetric only: no zero-points are produced (the kernel uses implicit
    ``2**(bits-1)``). This reproduces ``pack_ort_format`` exactly for the
    no-zeros case.
    """
    if bits not in (4, 8):
        raise NotImplementedError(f"Only 4/8-bit prequant packing is supported, got {bits}.")
    N, K = q.shape
    if K % group_size != 0:
        raise ValueError(f"in_features={K} must be divisible by group_size={group_size}.")
    kpack = 8 // bits
    blob_size = group_size // kpack
    k_blocks = K // group_size

    # Pack along K: intweight.T is [N, K] already (q is [N, K]).
    packed = _pack_on_row_uint8(q, bits)  # [N, K // kpack]
    qweight = packed.reshape(N, k_blocks, blob_size).contiguous()

    scales_flat = scale.to(torch.float32).reshape(-1).contiguous()  # N-major, matches scales.T.reshape(-1)
    return qweight, scales_flat


def recover_layer(module, bits):
    """Recover symmetric codes + scales for one DiscQuant-wrapped linear.

    Returns ``(q [N, K] uint-codes, scale [N, G], group_size, N, K)``.
    Requires a symmetric grid (zero == 2**(bits-1) everywhere); otherwise the
    direct-read passthrough cannot stay symmetric and lossless.
    """
    grid = module.grid
    with torch.no_grad():
        weight_q = module._unquant(module.x, rd=True).detach().to(torch.float32).cpu()
    scale, zero, group_size, _num_groups, maxq, N, K = grid_params(grid)

    if int(maxq) != (1 << bits) - 1:
        raise ValueError(
            f"Grid maxq={int(maxq)} does not match bits={bits} (expected {(1 << bits) - 1}). "
            "The grid must be quantized at the requested bit-width."
        )
    expected_zero = float(1 << (bits - 1))
    if not torch.allclose(zero, torch.full_like(zero, expected_zero)):
        raise ValueError(
            "Pre-quantized passthrough requires a SYMMETRIC grid "
            f"(zero == {expected_zero}); found asymmetric zero-points. "
            "Re-run DiscQuant with --symmetric for this export."
        )

    q = recover_int_codes(weight_q, scale, zero, group_size, maxq)  # [N, K] in [0, maxq]
    return q.to(torch.int64), scale, group_size, N, K


# ---------------------------------------------------------------------------
# Sidecar writer
# ---------------------------------------------------------------------------
SIDECAR_FORMAT = "ort_matmulnbits_prequant_v1"


def save_prequant_sidecar(model, args, savedir):
    """Write the pre-quantized attention sidecar (safetensors + manifest.json).

    The sidecar is keyed by ``"<layer_id>.<proj>"`` (e.g. ``"3.self_attn.q_proj"``)
    so the genai overlay can map entries onto the loaded model's attention
    submodules. ``proj`` is the quantlist leaf path within a decoder layer.
    """
    from safetensors.torch import save_file

    from linearutils import quantize_linearlayer_multimode

    if getattr(args, "quip", False):
        raise NotImplementedError("QuIP checkpoints cannot be exported to MatMulNBits (no activation rotation).")

    os.makedirs(savedir, exist_ok=True)
    num_layers = len(model.model.layers)

    tensors = {}
    manifest = {"format": SIDECAR_FORMAT, "symmetric": True, "tensors": {}}

    for layer_id, layer in enumerate(model.model.layers):
        for name, m in layer.named_modules():
            if not isinstance(m, quantize_linearlayer_multimode):
                continue
            bits = resolve_layer_bits(args, layer_id, name, num_layers)
            if int(m.grid.wbits) != bits:
                raise ValueError(
                    f"Layer {layer_id} '{name}' grid was quantized at {int(m.grid.wbits)} bits "
                    f"but bit placement resolved to {bits}. Quantize with matching bits "
                    "(quantize_model uses resolve_layer_bits) before exporting the sidecar."
                )
            q, scale, group_size, N, K = recover_layer(m, bits)
            qweight, scales_flat = pack_ort_matmulnbits(q, scale, bits, group_size)

            key = f"{layer_id}.{name}"
            tensors[f"{key}.qweight"] = qweight
            tensors[f"{key}.scales"] = scales_flat.to(torch.float16)
            bias = getattr(m.linear, "bias", None)
            if bias is not None:
                tensors[f"{key}.bias"] = bias.detach().to(torch.float16).cpu()
            manifest["tensors"][key] = {
                "layer_id": layer_id,
                "proj": name,
                "bits": int(bits),
                "group_size": int(group_size),
                "in_features": int(K),
                "out_features": int(N),
                "has_bias": bias is not None,
            }

    if not manifest["tensors"]:
        raise ValueError("No quantized DiscQuant layers found to export to the pre-quant sidecar.")

    save_file(tensors, os.path.join(savedir, "prequant_attention.safetensors"))
    with open(os.path.join(savedir, "prequant_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Saved pre-quant attention sidecar: {len(manifest['tensors'])} tensors -> {savedir} "
        f"(format={SIDECAR_FORMAT})"
    )
    return savedir
