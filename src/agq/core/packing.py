"""The EXP009/EXP018 row-major, little-bit-order W1--W4 format.

Every row is byte padded independently. Metadata is one BF16 scale and one
uint8 zero/reserved value per output channel. W1 codes represent signed scale.
"""
import numpy as np


def _bits(bits):
    if type(bits) is not int or bits not in (1, 2, 3, 4):
        raise ValueError("bits must be one of 1, 2, 3, 4")


def pack_codes(codes, bits):
    """Pack a nonempty [out_channels, in_channels] integer code matrix."""
    _bits(bits)
    q = np.asarray(codes)
    if q.ndim != 2 or 0 in q.shape or q.dtype.kind not in "ui":
        raise ValueError("codes must be a nonempty two-dimensional integer array")
    if np.any(q < 0) or np.any(q >= (1 << bits)):
        raise ValueError("code outside the requested bitwidth")
    q = q.astype(np.uint8)
    planes = ((q[:, :, None] >> np.arange(bits, dtype=np.uint8)) & 1)
    return np.packbits(planes.reshape(q.shape[0], -1), axis=1, bitorder="little")


def unpack_codes(packed, bits, columns):
    """Byte-address extraction, independent of the packbits implementation."""
    _bits(bits)
    p = np.asarray(packed)
    if type(columns) is not int or columns <= 0:
        raise ValueError("columns must be a positive integer")
    if p.dtype != np.uint8 or p.ndim != 2 or not p.shape[0]:
        raise ValueError("packed must be a nonempty two-dimensional uint8 array")
    if p.shape[1] != (columns * bits + 7) // 8:
        raise ValueError("packed row stride does not match shape and bitwidth")
    offsets = np.arange(columns, dtype=np.int64) * bits
    idx, shift = offsets // 8, offsets % 8
    padded = np.pad(p.astype(np.uint16), ((0, 0), (0, 1)))
    word = padded[:, idx] | (padded[:, idx + 1] << 8)
    return ((word >> shift) & ((1 << bits) - 1)).astype(np.uint8)


def dequantize_numpy(packed, bits, columns, scale, zero):
    """FP32 inspection decoder; the runtime decoder explicitly rounds to BF16."""
    q = unpack_codes(packed, bits, columns).astype(np.float32)
    s, z = np.asarray(scale, dtype=np.float32), np.asarray(zero, dtype=np.float32)
    if s.size != q.shape[0] or z.size != q.shape[0]:
        raise ValueError("one scale and zero/reserved value is required per row")
    s, z = s.reshape(-1, 1), z.reshape(-1, 1)
    if not np.isfinite(s).all() or np.any(s <= 0) or not np.isfinite(z).all():
        raise ValueError("finite positive scales and finite zeros are required")
    return (2 * q - 1) * s if bits == 1 else (q - z) * s
