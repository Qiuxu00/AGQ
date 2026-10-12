# Reproducing and extending AGQ

This release contains executable allocation, calibration, integer packing,
grouped lossless coding and block-lifetime components. It also contains explicit
integration points for a locally installed OpenVLA-OFT and LIBERO environment.
No original or quantized model weights are distributed.

The release was checked with CPU functional tests and generated tensors. The
CUDA kernels and full OpenVLA-OFT/LIBERO integration were **not rerun on a GPU in
this repository-preparation session**. The paper's 6,000 episodes are historical
experiment results, not results of the examples below. The frozen observations,
initial-state panel, calibrated weights and complete historical environment are
not included; therefore a fresh user cannot claim an exact replay of that panel
from this repository alone.

## 1. Run the complete small example

After the installation in [Getting started](GETTING_STARTED.md), from the repository root:

```bash
python scripts/agq_quantize_demo.py --bits 1 2 3 4 --epochs 4 --output outputs/quant_demo.json
python scripts/agq_allocate.py --profiles examples/profiles.json --budget-bytes 35 --lambda-value 0.5 --output outputs/allocation.json
python -m pytest tests/test_core_formats.py tests/test_core_torch.py -q
```

The first command fits actual quantizer parameters on a generated two-layer
block, exports packed integer weights, compares coded and calibrated inference
exactly, and confirms that dense weights are released after the block. Each of
W1, W2, W3 and W4 is exercised. The example's MSE values describe generated
tensors only. The second command solves a three-matrix synthetic allocation:
its optimum uses exactly 35 bytes. Example scripts refuse to overwrite output
files; choose a new output filename for another run.

## 2. Allocation: scores and real storage budgets

```python
from agq.core.allocation import allocate

profiles = [
    dict(name="block.attn.q_proj", rows=64, columns=64,
         sensitivity=0.2, overhead=0.8),
    dict(name="block.attn.k_proj", rows=64, columns=64,
         sensitivity=0.1, overhead=0.3),
]
result = allocate(profiles, budget_bytes=3000, lambda_value=0.5)
print(result["records"], result["mip_gap"], result["optimal"])
```

Run allocation independently for vision and language: each has its own budget
and normalization. `sensitivity` is task/trajectory-balanced MSE over the first
six normalized action components and all eight predicted steps. `overhead` is
`max(T_W4 - T_BF16, 0)`, measured on the same matrix input. Their normalized sum
weights the penalty `(4-bits)**2`.

If `candidate_bytes` is omitted, the allocator counts
`rows * ceil(columns * bits / 8) + 3 * rows`: row-padded codes, a BF16 scale and a
uint8 zero/reserved value per row. Supply the explicit list
`candidate_bytes=[W1_bytes, W2_bytes, W3_bytes, W4_bytes]` to reproduce another
candidate representation. The historical vision profiling/allocation format
and the learned row-wise export did not have identical metadata; the API
reports planned candidate bytes and final row-wise bytes separately. Average
bitwidth alone is not a byte budget. Time-limited feasible solutions are marked
`optimal=False` when the requested gap is not proven.

`agq.runtime.profiling.probe_action_sensitivity` temporarily changes only one
matrix to W4 and restores it even on a prediction error. Its default probe is
the release's unlearned row-wise quantizer; supply `quantize_weight` to use the
exact historical probe implementation. It does not pretend the new default
scan equals a paper scan. `balanced_action_mse` supplies the equal-task,
equal-trajectory reduction and excludes the gripper.

## 3. Learn fixed-bit channel quantizers

```python
import torch
from torch import nn
from agq.core.calibration import calibrate_block, export_matrix

teacher = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32)).eval()
inputs = [torch.randn(2, 32) for _ in range(8)]
student, history = calibrate_block(
    teacher, {"0": 2, "2": 3}, inputs, epochs=20, learning_rate=0.01
)
payloads = {name: export_matrix(student.get_submodule(name)) for name in ("0", "2")}
```

W2–W4 use learned lower/upper clipping, derived row scales and integer zeros,
straight-through rounding, and AdamW. W1 fixes signs and learns row magnitudes;
the W2–W4 affine formula does not apply to it. Scales and reconstructed weights
are explicitly rounded to BF16, matching the original export path. Original
weights and selected bitwidths are frozen. This is the AGQ adaptation of
OmniQuant's clipping/reconstruction ideas, **not all of OmniQuant**.

`calibrate_block` accepts `forward(block, sample)` for attention masks/tuple
outputs and `loss_fn(output, target, sample)` for valid-token masking. When
calibrating a sequence of blocks, maintain independent teacher and student
activations and supply `teacher_inputs=` for the unquantized path. Use genuine
held-out calibration observations with recorded task/trajectory splits; never
fit on evaluation observations. The historical setup used 480 calibration
frames and 20 epochs per block. The synthetic example does not replace that
dataset. For full-size models move only the active block and sample to the GPU,
retain cached inputs/targets on CPU, and avoid two whole-policy GPU copies.

## 4. Packed inference and target-only packages

```python
from agq.runtime.linear import replace_linears
from agq.runtime.blockwise import install_block_lifetime
from agq.runtime.package import save_payloads, load_payloads

# payloads is a dict of fully qualified model Linear names -> export_matrix result.
# This writes YOUR local generated weights, never uploads them.
save_payloads("private_outputs/my_targets", payloads)
payloads, encoded = load_payloads("private_outputs/my_targets")

# model is your already-loaded architecture with non-target weights and biases.
modules = replace_linears(model, payloads, backend="reference", encoded=encoded)
lifetime = install_block_lifetime(model, {
    "language_model.model.layers.0": list(payloads)
})
```

The last snippet illustrates a single block; use the correct owning block for
every target. `PackedLinear` reconstructs BF16 weights and calls a regular
matrix multiplication. `backend="triton"` requests the original fused unpack
and dequantization kernel on CUDA. With a coded payload, CUDA first reconstructs
the packed bytes using the included Huffman kernel. Dense weights for a block
are created together on entry and released together in `finally`, including
when inference fails. `lifetime.report()` verifies no dense weights survive
outside a block. These wrappers are for non-reentrant, single-threaded inference.

The NPZ package uses numeric arrays with `allow_pickle=False`, verifies SHA256,
and preserves the scale bit patterns. It includes **target matrices only**:
biases, action head, projector, token embeddings and all other non-target values
must come from the user-provided base model. The adapter currently loads that
base model before replacing its target matrices, so **initial loading can use
more memory than the paper's standalone empty-model loader**. Do not quote a
paper inference peak as this loader's startup memory requirement.

## 5. Grouped lossless storage

```bash
python scripts/agq_codec.py encode private_outputs/packed.bin private_outputs/coded.bin --bits 3
python scripts/agq_codec.py decode private_outputs/coded.bin private_outputs/restored.bin
```

Input is already quantized packed integer bytes. W2 combines four two-bit
codes, W3/W4 combine two codes, and W1 remains raw. The complete encoded header,
canonical-code lengths, block offsets and payload compete with raw packing;
the smaller representation is chosen, with raw winning ties. The selector
costs one byte for either choice. An unaligned W3 stream or a codeword longer
than the supported 24 bits falls back to raw. Blocks are 16 KiB nominally;
six-bit grouped symbols use a three-byte-aligned 16,383-byte block.

The inspectable Python encoder emits the original GHC1 format. CPU tests compare
its bytes and round trips against the **unmodified original C++ codec**. For
large models, add `--backend native` and install a C++17 compiler. Compilation
writes a user cache, not the installed source tree. Encoding introduces no
additional quantization error. Its selector/file bytes are separate from the
GPU runtime's lookup/checkpoint metadata and do not establish a VRAM figure.

The included CUDA source uses CUDA-enabled PyTorch, a CUDA compiler, OpenMP,
zlib development files and a supported Linux toolchain. The fused decoder also
requires Triton. CPU package tests do not validate these GPU build dependencies.

## 6. Integrate a local OpenVLA-OFT policy

Prepare a separate environment following the upstream OpenVLA-OFT and LIBERO
installation instructions, including their compatible Transformers version and
custom model code. The CPU test environment for this release is not a validated
full-policy environment. Put your upstream checkout on `PYTHONPATH` or install
it as upstream instructs. Obtain the BF16 checkpoint and auxiliary action-head
and proprioceptive-projector weights yourself, according to their licenses.
The adapter requires a local checkpoint directory; upstream helper code retains
its own file/network behavior and must be configured appropriately by the user.

```python
from experiments.robot.libero.run_libero_eval import GenerateConfig
from agq.runtime.openvla import OpenVLAAdapter
from agq.runtime.package import load_payloads

cfg = GenerateConfig(
    pretrained_checkpoint="/your/local/combined-libero-checkpoint",
    task_suite_name="libero_spatial",
    use_l1_regression=True, use_diffusion=False, use_film=False,
    use_proprio=True, num_images_in_input=2,
    load_in_4bit=False, load_in_8bit=False,
)
policy = OpenVLAAdapter.from_upstream(cfg)
payloads, encoded = load_payloads("/your/local/agq-targets")
policy.install(payloads, encoded=encoded, backend="triton")

# observation is prepared by upstream OFT: two images + eight-value robot state.
prediction = policy.predict_prepared(observation, instruction, "libero_spatial")
```

`target_groups` checks all 32×7 language matrices and 51×4 vision matrices by
name and block ownership. The L1 action head and proprioceptive projector are
retained. Predictions expose normalized actions for sensitivity, unnormalized
actions, and gripper-adjusted actions for the simulator separately.

`agq.runtime.evaluation.evaluate_libero_state(policy, task, initial_state,
suite="libero_spatial")` can execute a user-selected state via installed OFT
helpers. It restores the state, waits ten settling steps, then executes chunks
of eight actions. An optional `validate_prepared` callback runs before policy
execution and can enforce a frozen observation's exact image/state checks.
Simulator errors propagate; no states are retried, substituted or silently
skipped. Its output explicitly identifies a caller-supplied panel. This helper
does **not** reproduce the original EXP019 strict restoration/retry audit by
itself. Save state hashes, observation checks, seeds, environment versions,
device information and per-episode outcomes with any new evaluation.

For resources, `benchmark_inference(call, device="cuda", warmup=3, repeats=10)`
reports synchronized whole-call timing and PyTorch peak allocated memory after
warmup. It does not include loading and does not measure driver allocations or
reserved memory. Keep timing, resident storage, target storage and peak VRAM
as separate metrics; compare only compatible devices and measurement scopes.

## 7. Source lineage and release changes

Paths in this table identify the authors' historical experiment archive; they
are provenance labels, **not runtime dependencies** of this package.

| Release component | Original source | Release adaptation |
| --- | --- | --- |
| `core/allocation.py` | `EXP008/solver_worker.py`, `EXP010/allocation_core.py`, `EXP018/solve_formal_allocations.py` | Callable solver, explicit profiles/budgets, independent byte/objective checks |
| `core/packing.py`, `core/calibration.py` | `EXP018/release/learn_quantizer.py`, `train_language.py` | Preserved W1 and W2–W4 math/format; generic block callback and CPU support |
| `core/codec.py` | `CODEBOOK003/group_huffman.cpp`, `EXP018/compression/codec_runtime.py` | Compatible inspectable Python path, raw fallback, portable native compiler wrapper |
| `core/native/group_huffman.cpp` | `CODEBOOK003/group_huffman.cpp` | Byte-for-byte original source |
| `runtime/fast_decode.py` | `EXP018/release/fast_decode.py` | Byte-for-byte original fused kernel source |
| `runtime/entropy_cuda.cu` | `EXP018/compression/entropy_cuda.cu` | Byte-for-byte original source; package wrapper validates complete streams |
| `runtime/linear.py`, `blockwise.py` | `EXP018/evaluation/runtime/layerwise.py`, `vision_lifetime.py`, `EXP018/release/vision_runtime.py` | General model injection, bounded block lifetime, portable reference decoder |
| `runtime/openvla.py`, `evaluation.py` | `EXP018/compression/standalone_loader.py`, `EXP019/worker.py`, upstream OFT interfaces | User-local base-model adapter and explicit new-state evaluation; no private panel or orchestration |
| `runtime/package.py` | New release wrapper around the established payload | Numeric target-only format; no redistributed weights |

The release does not vendor OpenVLA-OFT, LIBERO, OmniQuant, HQQ or other upstream
model implementations. Their sources and licenses remain separate; see
[Third-party notices](../THIRD_PARTY_NOTICES.md). The project license remains
the repository owner's decision. No scheduling, remote-access, shutdown or
experiment-machine configuration is part of these algorithm entry points.
