# Get started

[Home](../README.md) · [Principles](PRINCIPLES.md) · [Cases](CASES.md) · [Integration status](INTEGRATION_STATUS.md)

Start with the small numerical examples and the included result audit. They require **no pretrained model weights, robot simulator, remote machine, or model download**. Installing Python dependencies may require network access.

All commands below run from the repository root. Python 3.10 or newer is required; the validated local environment used **Python 3.12.14 and PyTorch 2.6.0+cpu**. Use a separate environment and the supplied [CPU constraints](../requirements/cpu-constraints.txt) to reproduce these component checks. This environment is separate from the upstream OpenVLA-OFT GPU environment, which has its own dependencies.

## 1. Install and inspect the available methods

```bash
python -m pip install -c requirements/cpu-constraints.txt -e ".[test]"
agq methods
```

`agq methods` lists the matrix adapters, supported bitwidths, and whether each optional library is installed. An installed library is not itself evidence of a full-model test. Use `python -m agq.cli` in place of `agq` if your environment does not expose the command on its executable path.

For an explicitly CPU-only PyTorch build, first run `python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu`. The constraints file records the tested package versions; it is not a lockfile for full-policy GPU reproduction.

## 2. Exercise AGQ's quantization components

```bash
python scripts/agq_quantize_demo.py --bits 1 2 3 4 --epochs 4
```

The demo creates a small two-layer block and generated calibration inputs, then performs **learnable channel calibration → packed export → grouped lossless coding → block-wise execution** for W1, W2, W3, and W4. It checks that execution from the packed representation exactly matches the calibrated BF16 weights and reports reconstruction error on held-out generated inputs.

Add `--output outputs/quant_demo.json` to save the scalar report. This script creates new output files; use a fresh filename if one already exists. Its generated block is a functional example, not an OpenVLA policy or a LIBERO evaluation.

## 3. Run a small method comparison

```bash
agq compare --methods rtn --bits 4 --output outputs/compare_rtn_w4.json
```

This creates a fixed-seed synthetic weight matrix and matching inputs, quantizes and reconstructs the matrix, and compares its linear outputs with the original. RTN is the simple row-wise round-to-nearest reference. It is distinct from AGQ's learned quantizer.

The JSON output contains **scalar metrics and shapes**, including:

| Field | Meaning |
| --- | --- |
| `weight_mse` | Mean squared error between original and recovered weight values |
| `projection_mse` | Mean squared error between linear outputs on the same inputs |
| `projection_relative_l2` | Output error norm divided by the reference-output norm |
| `packed_payload_bytes` | Count for the local RTN format; unmeasured upstream formats use `null` |
| `packages` | Versions of the method libraries used |
| `data_origin` | Whether weights and inputs were generated or provided locally |

No weight arrays are written to this report. These are matrix reconstruction metrics, **not robot success rates or optimized inference timings**.

## 4. Try official HQQ and Quanto adapters

Install only the adapters you want to exercise:

```bash
python -m pip install -c requirements/cpu-constraints.txt -e ".[hqq,quanto]"
agq compare --methods rtn,hqq --bits 4 --output outputs/compare_w4.json
agq compare --methods rtn,quanto --bits 8 --output outputs/compare_w8.json
```

The adapters call **HQQ 0.2.8.post1** and **Optimum Quanto 0.2.7**. HQQ was exercised at **1, 2, 3, 4, and 8 bits**. The Quanto adapter exposes **8-bit weights only**: its lower-bit CPU paths require an upstream native extension and are not offered in this first adapter. Both methods use the same source matrix and inputs as RTN, then measure errors using a common dense projection. Their optimized GPU kernels and complete-model serialization are outside this comparison.

The HQQ adapter defaults to group size 64. For a custom matrix, choose a supported grouping compatible with its shape. See [baseline documentation](BASELINES.md) for the tested combinations and [integration status](INTEGRATION_STATUS.md) for the distinction between adapters and full-policy integration.

## 5. Supply your own matrix

```bash
agq compare --methods rtn,hqq --bits 4 --group-size 64 --weights local/matrix.npy --inputs local/inputs.npy --output outputs/local_comparison.json
```

`matrix.npy` must be a finite numeric array shaped `[output_features, input_features]`. `inputs.npy` must be shaped `[samples, input_features]`. These paths stand for files you create locally; they are not bundled examples or downloadable checkpoints. If `--inputs` is omitted, the command generates fixed-seed synthetic inputs with matching feature dimensions.

The command reads local arrays with pickle disabled and does not upload them. Keep private model data outside tracked files; the repository's no-weight policy applies to your generated artifacts too.

## 6. Inspect allocation and coding

```bash
python scripts/agq_allocate.py --profiles examples/profiles.json --budget-bytes 35 --lambda-value 0.5 --output outputs/allocation.json
```

The bundled profiles are a deliberately small **synthetic** example. Each record supplies a matrix name, shape, sensitivity, and recovery overhead. Candidate storage costs can be given explicitly. The solver checks the selected bytes against the budget and reports whether optimality was established. Real vision and language backbones require separate profiles, normalization, and budgets.

To work with a packed integer stream you already have locally:

```bash
python scripts/agq_codec.py encode local/packed.bin outputs/encoded.bin --bits 3
python scripts/agq_codec.py decode outputs/encoded.bin outputs/recovered.bin
```

These filenames stand for your local input and newly created outputs. The default Python codec needs no compiler; the optional native encoder uses the included C++ source. This utility compresses **integer codes**, not arbitrary floating-point model files. The quantization demo above already exercises coding without requiring you to supply a binary stream.

See [Reproducing AGQ](REPRODUCING_AGQ.md) for component interfaces, format details, and model adapter prerequisites.

## 7. Audit the paper results and run checks

```bash
python scripts/verify_results.py
python -m pytest -q
```

The result audit checks the included experiment records and denominators. **All 74 tests passed without skips** in the prepared environment with HQQ and Quanto installed: 52 AGQ core, 16 adapter, and 6 aggregate-result tests. Optional-library tests skip when their package is absent; an installed but broken adapter fails. Neither command starts a simulator or reruns the paper's 6,000 episodes. See [the validation record](VALIDATION.md).

Browse the [case guide](CASES.md) alongside [quality.json](../results/quality.json), [tasks.csv](../results/tasks.csv), and [performance.json](../results/performance.json). The precision, hardware, sample counts, and measurement definitions matter when comparing numbers.

## Full-model work

Full-policy calibration, inference, and LIBERO evaluation need additional assets: an appropriate OpenVLA-OFT model, observations and normalization metadata, the upstream environment, and compatible GPU runtime dependencies. The initial repository build has not rerun those workflows.

Use the official [OpenVLA-OFT setup guide](https://github.com/moojink/openvla-oft/blob/main/SETUP.md) and [LIBERO guide](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) for upstream prerequisites, then consult [the AGQ integration status](INTEGRATION_STATUS.md) for what has actually been packaged and checked. Do not interpret the numerical examples as a one-command reproduction of the complete paper.
