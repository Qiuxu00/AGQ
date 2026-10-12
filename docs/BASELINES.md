# Quantization comparisons

[Home](../README.md) · [Getting started](GETTING_STARTED.md) · [Implementation status](INTEGRATION_STATUS.md)

The matrix comparison is a small working extension of this repository. It calls real quantizers on identical source weights and held-out inputs, then measures reconstruction with the same dense projection. It does not rerun the paper's robot experiments.

| Method | Implemented path | Locally exercised bitwidths | Scope |
| --- | --- | --- | --- |
| RTN | Per-row nearest rounding; binary mean-magnitude reference | 1, 2, 3, 4, 8 | Inspectable numerical baseline with FP32 scales |
| HQQ | Official `HQQLinear` and its `dequantize()` method | 1, 2, 3, 4, 8 | CPU matrix quantization, group size 64 |
| Optimum Quanto | Official `quantize`, `freeze`, and tensor `dequantize()` | 8 | CPU int8 weight quantization |
| AWQ / GPTQ | Official workflows linked in [References](REFERENCES.md) | Not executed here | Literature comparison and future full-policy integration |

AGQ's allocation and learnable calibration are exercised separately by `scripts/agq_quantize_demo.py`. RTN is not relabeled as AGQ, AWQ, or GPTQ.

## Reproduce the included numerical reports

```bash
python -m pip install -c requirements/cpu-constraints.txt -e ".[test,hqq,quanto]"
agq compare --methods rtn,hqq --bits 4 --output outputs/matrix_w4.json
agq compare --methods rtn,hqq,quanto --bits 8 --output outputs/matrix_w8.json
```

Recorded reports: [W4](../examples/reports/matrix_w4.json) and [W8](../examples/reports/matrix_w8.json). Both use seed 7, a generated 32-by-128 matrix, and 64 generated input vectors. Only metrics are stored; no weight arrays are included.

| Experiment | Method | Weight MSE | Projection relative L2 |
| --- | --- | ---: | ---: |
| W4 generated matrix | RTN | 0.00985610 | 0.0984414 |
| W4 generated matrix | HQQ | 0.00742425 | 0.0862697 |
| W8 generated matrix | RTN | 0.00003469 | 0.00591354 |
| W8 generated matrix | HQQ | 0.00002532 | 0.00503970 |
| W8 generated matrix | Quanto | 0.00004087 | 0.00624215 |

These particular matrices illustrate the command interface, not an accuracy ranking for real models. Bitwidth alone does not match metadata formats: RTN uses row-wise FP32 scale/uint8 zero, HQQ uses its own grouped format, and Quanto uses its own quantized tensor representation. One-bit HQQ is not guaranteed to improve on RTN for arbitrary inputs.

## Input and measurement contract

- `--weights` accepts a finite real `.npy` matrix `[out_features, in_features]`; `--inputs` accepts `[samples, in_features]`. Loading disables pickle.
- Omitting either array generates that part of the experiment and records its origin in the result.
- Weight error and projected-output error are accumulated in FP64 from FP32 source arrays.
- RTN's byte count describes only packed matrix codes, row padding, and metadata. It is not an actual serialized model file size.
- HQQ/Quanto physical storage is left `null`; the adapter does not pretend dense tensor sizes measure their packed storage.
- No optimized-kernel latency, GPU memory, action-space accuracy, or task success is inferred from these tests.

## Compatibility

This repository's CPU checks used Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 2.2.6, SciPy 1.15.3, HQQ 0.2.8.post1, Optimum Quanto 0.2.7, and Transformers 4.48.3. The constrained environment is independent of the original OpenVLA-OFT training/evaluation environment.

HQQ's row-group count must fit its packing alignment: divisible by 8 for W1, 4 for W2, and 2 for W4. The adapter checks this explicitly. Quanto 2/4-bit CPU unpacking requires an upstream native extension; that path is deliberately not exposed by this first adapter. Quanto int8 works without it. The optional extras pin Transformers because a newer dependency caused a CPU import failure during validation.

Full-model comparisons need matched module coverage, calibration observations, action preprocessing and normalization, and an independently validated inference environment. Follow [the integration checklist](INTEGRATION_STATUS.md) before reporting new VLA results.

Upstream implementations: [HQQ](https://github.com/dropbox/hqq), [Optimum Quanto](https://github.com/huggingface/optimum-quanto), [AWQ](https://github.com/mit-han-lab/llm-awq), [GPTQ](https://github.com/IST-DASLab/gptq).
