# Validation record

Local repository preparation, 2026-10-12. The host was Windows with Python 3.12.14 and CPU PyTorch 2.6.0. No simulator, cloud machine, GPU experiment, or pretrained model download was started by these checks.

## Checks performed

- 52 AGQ core tests: allocation compared with exhaustive small solutions, W1–W4 packing, calibrated BF16 weight recovery, frozen original weights, codec framing and lossless restoration, Python/C++ byte equality, model-package hashes, action sensitivity, and block cleanup including exceptions. Regression checks keep normalized, unnormalized, and deployed action arrays independent, restore the upstream capture hook after prediction failure, and ensure block inference retains no backward graph or saved tensors even with trainable normalization parameters and gradient-enabled inputs.
- 16 matrix-adapter tests: real HQQ W1/W2/W3/W4/W8 calls, real Quanto int8 calls, RTN behavior, fixed-input error calculations, nonfinite/invalid input rejection, and local array loading with pickle disabled.
- 6 aggregate-result tests: episode/task consistency and rejection of altered denominators, duplicate tasks, inconsistent sums, or incompatible resource numbers.
- Four W1–W4 calibration/packing/coding/inference demonstrations, the synthetic 35-byte allocation, and a generated codec round trip.
- Included W4/W8 matrix comparisons and the separate audit of 10 configurations, 6,000 episodes, and 400 per-task rows.

The complete 74-test suite passed without skips in the prepared environment. These tests validate the released components and records, not a new execution of the historical experiments. Reproduction commands are in [Getting started](GETTING_STARTED.md); the CPU package constraints are in [requirements/cpu-constraints.txt](../requirements/cpu-constraints.txt). This constrained component environment is independent of the upstream OpenVLA-OFT GPU environment.

## Explicitly outside this local validation

CUDA/Triton compilation and inference, complete-model calibration, full OpenVLA-OFT loading, and LIBERO closed-loop evaluation have not been rerun in this release environment. The supplied OpenVLA adapter first loads a user-provided BF16 model; its startup-memory demand cannot be represented by the historical standalone compressed-loader numbers.

HQQ/Quanto integration is validated at the matrix reconstruction level. HQQ covers W1/W2/W3/W4/W8; the Quanto adapter exposes only the verified CPU int8 path. Quanto 2/4-bit CPU paths require an upstream native extension and are not provided by this adapter. AWQ/GPTQ references and instructions do not amount to full-policy integration. Optional tests skip only when the corresponding library is absent; an installed but broken adapter fails.

Historical code-stream recovery and resource measurements do not establish new closed-loop equality for the coded exports. The evidence fields in [performance.json](../results/performance.json), including strict pairing status, remain visible.
