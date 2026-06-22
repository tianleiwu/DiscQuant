"""Empirical check: is genai's RTN re-quantization lossless on DiscQuant-rounded weights?

DiscQuant (attention-only gpt-oss path) saves *dequantized* fp16 weights that lie on
DiscQuant's symmetric 4-bit grid. genai then re-quantizes them with the ORT RTN
MatMulNBits quantizer. This script applies BOTH real implementations back-to-back and
measures whether the second pass reproduces the first pass bit-for-bit.

DiscQuant grid  (gptq/quant.py):              scale = 2*amax/15, zero = 8
genai RTN grid  (neural_compressor.quant_tensor, sym, uint): scale = 2*amax/15, zero = 8
"""

import numpy as np

# Real genai/ORT RTN tensor quantizer (the exact function genai's int4_algo=rtn calls).
import sys
sys.path.insert(0, "/home/tianlei/onnxruntime/onnxruntime/python/tools/quantization")
from neural_compressor.weight_only import quant_tensor  # noqa: E402


def discquant_sym_dequant(W, group_size):
    """DiscQuant symmetric 4-bit round-to-nearest dequantization (gptq/quant.py)."""
    maxq = 15
    rows, cols = W.shape
    if group_size == -1:
        group_size = cols
    assert cols % group_size == 0
    Wd = np.empty_like(W, dtype=np.float64)
    for j0 in range(0, cols, group_size):
        blk = W[:, j0:j0 + group_size].astype(np.float64)
        xmax = np.maximum(np.abs(blk).max(axis=1, keepdims=True), 1e-12)
        scale = (2.0 * xmax) / maxq            # (xmax - xmin)/maxq with xmin=-xmax
        zero = 8.0
        q = np.clip(np.round(blk / scale) + zero, 0, maxq)
        Wd[:, j0:j0 + group_size] = scale * (q - zero)
    return Wd


def genai_rtn_sym_dequant(W, group_size):
    """genai RTN symmetric dequantization via the REAL ORT quant_tensor (sym, uint)."""
    rows, cols = W.shape
    gs = cols if group_size == -1 else group_size
    # quant_tensor reshapes to (-1, group_size) row-major, matching our per-row blocks.
    q, scale, zp = quant_tensor(W.astype(np.float64), num_bits=4, group_size=gs,
                                scheme="sym", dtype="uint")
    # sym => zero_point is a constant 8 (1 << (bits-1)); dequant = scale*(q - 8)
    zp = np.asarray(zp).reshape(-1, 1)
    Wd = (scale * (q - zp)).reshape(rows, cols)
    return Wd


def discquant_asym_dequant(W, group_size):
    """DiscQuant asymmetric 4-bit RTN dequantization (gptq/quant.py, sym=False)."""
    maxq = 15
    rows, cols = W.shape
    gs = cols if group_size == -1 else group_size
    Wd = np.empty_like(W, dtype=np.float64)
    codes = np.empty_like(W, dtype=np.int64)
    for j0 in range(0, cols, gs):
        blk = W[:, j0:j0 + gs].astype(np.float64)
        xmin = np.minimum(blk.min(axis=1, keepdims=True), 0.0)
        xmax = np.maximum(blk.max(axis=1, keepdims=True), 0.0)
        same = (xmin == 0) & (xmax == 0)
        xmin = np.where(same, -1.0, xmin)
        xmax = np.where(same, 1.0, xmax)
        scale = (xmax - xmin) / maxq
        zero = np.round(-xmin / scale)
        q = np.clip(np.round(blk / scale) + zero, 0, maxq)
        codes[:, j0:j0 + gs] = q.astype(np.int64)
        Wd[:, j0:j0 + gs] = scale * (q - zero)
    return Wd, codes


def genai_rtn_asym_dequant(W, group_size):
    """genai RTN asymmetric dequantization via real ORT quant_tensor (asym, uint)."""
    rows, cols = W.shape
    gs = cols if group_size == -1 else group_size
    q, scale, zp = quant_tensor(W.astype(np.float64), num_bits=4, group_size=gs,
                                scheme="asym", dtype="uint")
    zp = np.asarray(zp).reshape(-1, 1)
    Wd = (scale * (q - zp)).reshape(rows, cols)
    return Wd, q.reshape(rows, cols).astype(np.int64)


def discquant_sym_codes(W, group_size):
    maxq = 15
    rows, cols = W.shape
    gs = cols if group_size == -1 else group_size
    codes = np.empty_like(W, dtype=np.int64)
    for j0 in range(0, cols, gs):
        blk = W[:, j0:j0 + gs].astype(np.float64)
        xmax = np.maximum(np.abs(blk).max(axis=1, keepdims=True), 1e-12)
        scale = (2.0 * xmax) / maxq
        codes[:, j0:j0 + gs] = np.clip(np.round(blk / scale) + 8, 0, maxq).astype(np.int64)
    return codes


def genai_rtn_sym_codes(W, group_size):
    rows, cols = W.shape
    gs = cols if group_size == -1 else group_size
    q, _, _ = quant_tensor(W.astype(np.float64), num_bits=4, group_size=gs,
                           scheme="sym", dtype="uint")
    return q.reshape(rows, cols).astype(np.int64)


def run(rows=128, cols=256, group_size=-1, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((rows, cols)).astype(np.float32)

    # --- SYMMETRIC (genai/DiscQuant default) ---
    Wd1 = discquant_sym_dequant(W, group_size)
    Wd2 = genai_rtn_sym_dequant(Wd1, group_size)
    diff = np.abs(Wd2 - Wd1)
    c1 = discquant_sym_codes(W, group_size)
    c2 = genai_rtn_sym_codes(Wd1, group_size)
    codes_same = np.array_equal(c1, c2)
    print(f"[SYM  gs={group_size:>4}] bit-exact={np.array_equal(Wd2, Wd1)}"
          f"  changed={int((diff>0).sum())}/{rows*cols}"
          f"  rel_max={diff.max()/(np.abs(Wd1).max()+1e-12):.4%}"
          f"  codes_preserved={codes_same}")

    # --- ASYMMETRIC ---
    Wa1, ca1 = discquant_asym_dequant(W, group_size)
    Wa2, ca2 = genai_rtn_asym_dequant(Wa1, group_size)
    diffa = np.abs(Wa2 - Wa1)
    print(f"[ASYM gs={group_size:>4}] bit-exact={np.array_equal(Wa2, Wa1)}"
          f"  changed={int((diffa>0).sum())}/{rows*cols}"
          f"  rel_max={diffa.max()/(np.abs(Wa1).max()+1e-12):.4%}"
          f"  codes_preserved={np.array_equal(ca1, ca2)}")
    return np.array_equal(Wd2, Wd1), np.array_equal(Wa2, Wa1)


if __name__ == "__main__":
    print("Round-trip: DiscQuant grid  ->  genai RTN re-quant (real impls)\n")
    sym_all = asym_all = True
    for gs in (-1, 32, 64, 128):
        s, a = run(group_size=gs)
        sym_all &= s
        asym_all &= a
    print(f"\nLOSSLESS sym (bit-exact all):  {sym_all}")
    print(f"LOSSLESS asym (bit-exact all): {asym_all}")
