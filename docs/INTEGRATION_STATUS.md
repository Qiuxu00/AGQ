# Implementation and integration status

[Home](../README.md) · [Get started](GETTING_STARTED.md) · [Principles](PRINCIPLES.md)

This page distinguishes **available components**, **local numerical validation**, and **full-policy reproduction**. The current repository preparation does not distribute model weights, calibration datasets, learned quantizer parameters, or a pretrained AGQ checkpoint.

## AGQ components

| Component | Repository scope | Limit of this scope |
| --- | --- | --- |
| Joint score and matrix allocation | W4 sensitivity/cost combination, explicit candidate byte costs, separate solver calls per backbone | Profiling inputs must be supplied; a toy allocation does not regenerate the paper's measured probes |
| Per-channel quantization | W1 signed amplitudes; W2–W4 affine quantizers with BF16 scales | Matrix examples are not full-policy calibration |
| Learnable calibration | Frozen weights and bitwidths, clipping parameters, block-output reconstruction | Full OpenVLA-OFT block inputs and task-balanced calibration data are external prerequisites |
| Integer packing | Row-wise W1–W4 packing and exact unpacking checks | Packed weight files are generated locally, not shipped |
| Grouped lossless coding | Python codec and optional native encoder, derived from the research implementation | GPU throughput needs its own hardware validation |
| Compressed execution | Packed linear layers, block lifetime controls, and OpenVLA adapter interfaces | This preparation stage has not rerun full-policy GPU inference |
| Results | Sanitized aggregates, case analysis, and consistency checks | Historical measurements, not new evaluation runs |

The [principles guide](PRINCIPLES.md) explains the research method. Numerical examples make individual components inspectable; they do not certify that an arbitrary full model can be quantized with one command.

Local CPU checks exercise W1–W4 packing and BF16 reconstruction, calibration with frozen source weights, allocation against a small exhaustive optimum, Python/native codec byte identity, and release of temporary block weights including error paths. Regression checks preserve independent normalized, unnormalized, and deployed action arrays, restore the upstream hook after prediction failure, and verify that inference retains no backward graph or saved tensors. The small allocation example selects **W2, W2, W3 in exactly 35 bytes**. Full-model and GPU validation remain separate steps.

## Other quantization methods

The comparison interface uses the same small matrices and fixed inputs for the methods it supports. Its reconstruction errors are useful for checking adapters and studying numerical behavior; they are not LIBERO success rates.

| Method | Integration scope | Relationship to paper results |
| --- | --- | --- |
| RTN | A local round-to-nearest reference quantizer | Numerical reference only; not a new paper baseline |
| HQQ | Official HQQ package; CPU matrix tests passed at W1/W2/W3/W4/W8 | New repository comparison functionality; no new VLA result claimed |
| Optimum Quanto | Official Quanto package; verified CPU int8 adapter only | New repository comparison functionality; no new VLA result claimed |
| AWQ | Official implementation and reproduction guidance linked below | Paper comparison values remain literature results |
| GPTQ | Official implementation and reproduction guidance linked below | Paper comparison values remain literature results |
| OmniQuant | Calibration design attribution and upstream instructions | AGQ borrows clipping/reconstruction ideas; it is not the complete OmniQuant algorithm |
| TorchAO | Official reference and future integration point | No AGQ-specific full-model adapter claim |

Check [Getting started](GETTING_STARTED.md) for installed dependency checks and runnable commands, [baseline documentation](BASELINES.md) for the tested adapters, and [Reproducing AGQ](REPRODUCING_AGQ.md) for the extracted research components.

The complete local suite passed **74 tests without skips**: 52 AGQ core, 16 adapter, and 6 aggregate-result checks. Use [the constrained CPU environment](../requirements/cpu-constraints.txt) for these checks; do not assume it is compatible with the separate full-policy GPU environment. Quanto 2/4-bit CPU integration is not exposed because that path needs an upstream native extension. Details are in [Validation](VALIDATION.md).

## Use the official AWQ and GPTQ workflows

The official [AWQ repository](https://github.com/mit-han-lab/llm-awq) provides its search, quantization, and inference workflows. The official [GPTQ repository](https://github.com/IST-DASLab/gptq) provides quantization and evaluation examples for supported language-model families. Their examples are the source of truth for those implementations.

For an OpenVLA-OFT comparison, an integration must additionally define:

1. Which vision and language matrices are quantized, and which modules retain original precision.
2. The exact calibration observations, preprocessing, action head, normalization, and precision settings.
3. How each exported quantized format is loaded and executed.
4. A same-input check before closed-loop evaluation and a clear timing window for performance measurements.
5. Separate result files with method, upstream revision, configuration, hardware, episode counts, and errors.

Running an upstream language-model example does not complete this VLA integration. This repository does not relabel RTN or its own channel calibration as AWQ, GPTQ, or full OmniQuant.

## Validation levels

| Level | What a passing check establishes |
| --- | --- |
| Import and command check | The package or command can be called in the tested environment |
| Matrix numerical check | Quantizer/packing/adapter behavior on the supplied small inputs |
| Result audit | Published aggregate fields agree with the included records and denominators |
| Full-model inference | The intended model loads and emits actions in a compatible environment |
| Closed-loop evaluation | A defined policy/configuration completes the specified episodes |
| Resource measurement | Memory and latency measured on the stated hardware and timing window |

Only the first three levels belong to this local repository build's checks. Historical EXP019 and performance results remain historical evidence at their respective levels; they are not upgraded into fresh execution tests by packaging them here.

## Repository visibility and license

The repository is publicly accessible; the author still needs to choose the project's license. Existing upstream licenses continue to apply to third-party components. The repository excludes model checkpoints and learned quantizer parameters; public access to the code does not make those weights available.
