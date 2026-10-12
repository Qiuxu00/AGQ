import importlib.util

import numpy as np
import pytest

from agq.cli import main
from agq.integrations.matrix import compare_matrices, rtn_reconstruct


def test_exact_binary_and_row_padding():
    weights = np.array([[-2, 2, -2], [0, 0, 0]], dtype=np.float32)
    recovered, size = rtn_reconstruct(weights, 1)
    np.testing.assert_array_equal(recovered, weights)
    assert size == 2 + 8  # two padded code bytes and two float32 scales


def test_zero_constant_and_nonfinite_inputs():
    for bits in [1, 2, 3, 4, 8]:
        for constant in [0, -3, 4]:
            weights = np.full((3, 17), constant, dtype=np.float32)
            recovered, _ = rtn_reconstruct(weights, bits)
            np.testing.assert_allclose(recovered, weights, atol=1e-6)
    with pytest.raises(ValueError, match="finite"):
        compare_matrices([[float("nan")]], [[1]])
    with pytest.raises(ValueError, match="features"):
        compare_matrices([[1, 2]], [[1]])


def test_same_input_quantization_error_and_no_weight_serialization():
    rng = np.random.default_rng(17)
    weight = rng.standard_normal((12, 64)).astype(np.float32)
    inputs = rng.standard_normal((24, 64)).astype(np.float32)
    original_weight, original_inputs = weight.copy(), inputs.copy()
    low = compare_matrices(weight, inputs, bits=2)
    high = compare_matrices(weight, inputs, bits=8)
    assert high["results"][0]["weight_mse"] < low["results"][0]["weight_mse"]
    assert high["results"][0]["projection_mse"] < low["results"][0]["projection_mse"]
    assert high["weights_included"] is False
    assert "weight" not in high and "inputs" not in high
    np.testing.assert_array_equal(weight, original_weight)
    np.testing.assert_array_equal(inputs, original_inputs)


def test_cli_never_loads_pickle(tmp_path):
    path = tmp_path / "objects.npy"
    np.save(path, np.array([[{"arbitrary": "object"}]], dtype=object))
    with pytest.raises(SystemExit) as error:
        main(["compare", "--weights", str(path)])
    assert error.value.code == 2


@pytest.mark.optional
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
def test_hqq_calls_real_quantizer(bits):
    if importlib.util.find_spec("hqq") is None:
        pytest.skip("HQQ optional dependency not installed")
    import hqq.core.quantize  # Installed but broken imports must fail the test.
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((16, 128)).astype(np.float32)
    inputs = rng.standard_normal((12, 128)).astype(np.float32)
    result = compare_matrices(weight, inputs, methods=["hqq"], bits=bits)
    row = result["results"][0]
    assert "hqq" in row["packages"]
    # One-bit HQQ can be less accurate than RTN for these synthetic weights;
    # adapter validation must not assume a quality advantage.
    assert np.isfinite(row["weight_mse"]) and row["weight_mse"] > 0
    if bits == 8:
        assert row["weight_mse"] < 0.01
    assert row["packed_payload_bytes"] is None


@pytest.mark.optional
@pytest.mark.parametrize("bits", [8])
def test_quanto_calls_real_quantizer(bits):
    if importlib.util.find_spec("optimum") is None:
        pytest.skip("Quanto optional dependency not installed")
    if importlib.util.find_spec("optimum.quanto") is None:
        pytest.skip("Quanto optional dependency not installed")
    import optimum.quanto  # Installed but broken imports must fail the test.
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((16, 128)).astype(np.float32)
    inputs = rng.standard_normal((12, 128)).astype(np.float32)
    result = compare_matrices(weight, inputs, methods=["quanto"], bits=bits)
    row = result["results"][0]
    assert "optimum-quanto" in row["packages"]
    assert 0 < row["weight_mse"] < 1.0


@pytest.mark.parametrize("bits", [True, 4.0, -1, 0])
def test_invalid_bitwidth_rejected(bits):
    with pytest.raises(ValueError):
        compare_matrices(np.ones((2, 64)), np.ones((1, 64)), bits=bits)


def test_hqq_group_alignment_rejected():
    if importlib.util.find_spec("hqq") is None:
        pytest.skip("HQQ optional dependency not installed")
    with pytest.raises(ValueError, match="groups"):
        compare_matrices(np.ones((2, 64)), np.ones((1, 64)), methods=["hqq"], bits=1)


def test_quanto_unsupported_cpu_bitwidth_rejected():
    with pytest.raises(ValueError, match="does not support"):
        compare_matrices(np.ones((16, 128)), np.ones((1, 128)), methods=["quanto"], bits=4)
