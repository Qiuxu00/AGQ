import itertools
import math
import shutil
import subprocess

import numpy as np
import pytest

from agq.core.allocation import allocate, joint_scores, rowwise_bytes
from agq.core.packing import pack_codes, unpack_codes
from agq.core.codec import encode, decode, encode_ghc1, decode_ghc1, build_native


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("columns", [1, 7, 8, 9, 13, 32])
def test_row_padding_and_independent_unpack(bits, columns):
    rng = np.random.default_rng(123)
    q = rng.integers(0, 1 << bits, size=(5, columns), dtype=np.uint8)
    packed = pack_codes(q, bits)
    assert packed.shape == (5, (columns * bits + 7) // 8)
    assert np.array_equal(unpack_codes(packed, bits, columns), q)
    assert rowwise_bytes(5, columns, bits) == packed.nbytes + 15


def test_pack_known_little_bit_order_and_bad_codes():
    assert pack_codes(np.array([[0, 1, 2, 3]], dtype=np.uint8), 2).tolist() == [[228]]
    with pytest.raises(ValueError):
        pack_codes(np.array([[4]], dtype=np.uint8), 2)
    with pytest.raises(ValueError):
        pack_codes(np.array([[-1]], dtype=np.int8), 2)
    with pytest.raises(ValueError):
        unpack_codes(np.zeros((2, 1), dtype=np.uint8), 3, 9)


def test_allocation_matches_exhaustive_optimum():
    pytest.importorskip("scipy")
    records = [{"name": f"m{i}", "rows": r, "columns": c, "sensitivity": s, "overhead": o}
               for i, (r, c, s, o) in enumerate([(2, 9, .2, 1), (3, 8, .3, 4), (1, 13, .5, 2)])]
    scores = joint_scores([.2, .3, .5], [1, 4, 2], .5)
    feasible = [(sum(s * (4 - b) ** 2 for s, b in zip(scores, bits)), bits)
                for bits in itertools.product(range(1, 5), repeat=3)
                if sum(rowwise_bytes(r["rows"], r["columns"], b) for r, b in zip(records, bits)) <= 35]
    result = allocate(records, 35)
    assert result["optimal"] and result["selected_bytes"] <= 35
    assert math.isclose(result["objective"], min(feasible)[0], abs_tol=1e-10)
    assert result["independent_check_passed"]
    with pytest.raises(ValueError, match="minimum"):
        allocate(records, 1)


def test_explicit_candidate_storage_and_zero_measurements():
    pytest.importorskip("scipy")
    record = {"name": "large", "rows": 3, "columns": 9, "sensitivity": 1, "overhead": 0,
              "candidate_bytes": [2, 4, 6, 8]}
    result = allocate([record], 6)
    assert result["records"][0]["bits"] == 3
    assert result["records"][0]["export_rowwise_bytes"] != 6
    assert joint_scores([0, 0], [0, 0]) == [0, 0]
    with pytest.raises(ValueError):
        joint_scores([float("nan")], [1])


@pytest.mark.parametrize("width", [2, 4, 6, 8])
def test_ghc1_multiple_chunks_and_single_symbol(width):
    raw = b"\0" * 33000
    blob = encode_ghc1(raw, width)
    assert decode_ghc1(blob) == raw
    with pytest.raises(ValueError):
        decode_ghc1(blob[:-1])
    with pytest.raises(ValueError):
        decode_ghc1(blob, max_output_bytes=len(raw) - 1)


def test_codec_selection_and_corruption():
    rng = np.random.default_rng(5)
    for bits in (1, 2, 3, 4):
        for raw in (b"", b"\0" * 4098, rng.integers(0, 256, 4098, dtype=np.uint8).tobytes()):
            blob, report = encode(raw, bits)
            assert decode(blob) == raw
            assert len(blob) <= len(raw) + 1
            if bits == 1:
                assert report["mode"] == "raw"
    with pytest.raises(ValueError):
        decode(b"\x02junk")
    blob = bytearray(encode_ghc1(b"\0" * 33, 8))
    blob[-1] |= 128
    with pytest.raises(ValueError, match="padding"):
        decode_ghc1(blob)


@pytest.mark.skipif(shutil.which("g++") is None, reason="optional C++ compiler is unavailable")
@pytest.mark.parametrize("width", [2, 4, 6, 8])
def test_original_cpp_format_byte_identity(tmp_path, width):
    tool = build_native(tmp_path)
    rng = np.random.default_rng(7)
    raw = rng.choice(np.array([0, 0, 0, 1, 2, 3], dtype=np.uint8), size=33000).tobytes()
    native = subprocess.run([str(tool), "encode", "16384", str(width)], input=raw, capture_output=True, check=True).stdout
    assert encode_ghc1(raw, width) == native
    assert decode_ghc1(native) == raw
    restored = subprocess.run([str(tool), "decode"], input=native, capture_output=True, check=True).stdout
    assert restored == raw
