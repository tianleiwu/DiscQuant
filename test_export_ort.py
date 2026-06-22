"""Standalone numerical validation for export_ort.py.

Validates that DiscQuant's exported AutoGPTQ tensors decode, via the exact
onnxruntime-genai GPTQ decode path (``GPTQModel.handle_qzeros`` + ``unpack`` +
``dequant_weight``), back to the original on-grid DiscQuant weights.

Runs on CPU with no GPU and no real model download. Run with:

    /home/tianlei/venv/bin/python test_export_ort.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
GENAI_MODELS = "/home/tianlei/onnxruntime-genai/src/python/py/models"
sys.path.insert(0, GENAI_MODELS)

import export_ort  # noqa: E402
from quantized_model import GPTQModel, QuantizedTensorModule  # noqa: E402


def make_grid_solution(n, k, bits, groupsize, seed=0):
    """Create a random weight snapped to an affine grid (the 'DiscQuant solution')."""
    torch.manual_seed(seed)
    maxq = (1 << bits) - 1
    zero_val = (maxq + 1) // 2  # symmetric zero-point (8 for 4-bit)

    gs = k if groupsize == -1 else groupsize
    num_groups = k // gs

    w = torch.randn(n, k, dtype=torch.float32)
    wg = w.reshape(n, num_groups, gs)
    # per-(row, group) symmetric scale
    scale = wg.abs().amax(dim=-1, keepdim=True) / (maxq / 2)
    scale = scale.clamp(min=1e-6)  # [n, num_groups, 1]
    zero = torch.full((n, num_groups, 1), float(zero_val))

    q = torch.clamp(torch.round(wg / scale) + zero, 0, maxq)
    wq = ((q - zero) * scale).reshape(n, k)

    scale2d = scale.reshape(n, num_groups)
    zero2d = zero.reshape(n, num_groups)
    return wq, scale2d, zero2d, gs, maxq


def genai_decode(tensors, n, k, bits, group_size):
    """Decode AutoGPTQ tensors exactly as onnxruntime-genai's GPTQModel does."""
    gm = GPTQModel.__new__(GPTQModel)  # bypass __init__; methods are stateless
    mod = QuantizedTensorModule()
    mod.qweight = tensors["qweight"].clone()
    mod.qzeros = tensors["qzeros"].clone()
    mod.scales = tensors["scales"].clone()
    mod.g_idx = tensors["g_idx"].clone()
    mod.bits = bits
    mod.in_features = k
    mod.out_features = n
    mod.group_size = group_size  # property setter; -1 => in_features

    gm.handle_qzeros(mod)  # AutoGPTQ stores zero-1; this restores +1
    gm.unpack(mod)  # unpack_qzeros, unpack_qweight, dequant_weight
    return mod.qweight  # now dequantized, shape [n, k]


def run_case(n, k, bits, groupsize):
    wq, scale, zero, gs, maxq = make_grid_solution(n, k, bits, groupsize)

    # 1. recover_int_codes must reproduce exact integer codes.
    q = export_ort.recover_int_codes(wq, scale, zero, gs, maxq)
    q_ref = torch.clamp(torch.round(wq.reshape(n, k // gs, gs) / scale.reshape(n, k // gs, 1)) + zero.reshape(n, k // gs, 1), 0, maxq).reshape(n, k).to(torch.int64)
    assert torch.equal(q, q_ref), "recover_int_codes mismatch"
    assert int(q.min()) >= 0 and int(q.max()) <= maxq, "codes out of range"

    # 2. pack -> genai decode must reproduce the on-grid weights.
    tensors = export_ort.pack_gptq(q, scale, zero, bits)
    group_size = -1 if groupsize == -1 else groupsize
    w_dec = genai_decode(tensors, n, k, bits, group_size)

    assert w_dec.shape == wq.shape, f"shape {w_dec.shape} != {wq.shape}"
    max_err = (w_dec.float() - wq.float()).abs().max().item()
    # fp16 scales introduce tiny error; weights are otherwise exact.
    tol = 1e-2 * (wq.abs().max().item() + 1e-6)
    ok = max_err <= tol
    print(f"  n={n:5d} k={k:5d} bits={bits} groupsize={groupsize:>4}  max_err={max_err:.3e} tol={tol:.3e}  {'PASS' if ok else 'FAIL'}")
    assert ok, f"decode mismatch: max_err={max_err} > tol={tol}"
    return ok


def main():
    print("DiscQuant -> AutoGPTQ -> onnxruntime-genai GPTQ decode round-trip:")
    cases = [
        # (n, k, bits, groupsize)
        (256, 512, 4, -1),    # per-channel 4-bit
        (256, 512, 4, 32),    # block-wise 4-bit, gs=32
        (256, 512, 4, 64),    # block-wise 4-bit, gs=64
        (512, 1024, 4, 128),  # block-wise 4-bit, gs=128
        (128, 256, 8, -1),    # per-channel 8-bit
        (128, 256, 8, 64),    # block-wise 8-bit
    ]
    all_ok = True
    for n, k, bits, gs in cases:
        all_ok &= run_case(n, k, bits, gs)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
