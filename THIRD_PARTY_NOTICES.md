# Sources and third-party notices

The repository is a private preparation of the AGQ research release. The authors have not yet selected a project-wide open-source license. No license for the AGQ code is inferred from an upstream dependency's license.

## Research implementation

The extracted AGQ code derives from the authors' experiment implementations:

| Component | Research source lineage | Release form |
| --- | --- | --- |
| Joint scoring and byte-budget allocation | EXP008, EXP010, EXP018 | Parameterized solver with independent budget checks |
| Channel calibration and row packing | EXP018/release/learn_quantizer.py | Reusable block/matrix functions |
| Grouped Huffman coding | CODEBOOK003/group_huffman.cpp | Original C++ source plus a compatible Python reference |
| CUDA grouped decoder | EXP018/compression | Original CUDA source with validated input framing |
| Fused unpack/dequantization | EXP018/evaluation/runtime | Original Triton kernel and explicit runtime interface |
| Policy and evaluation integration | EXP018 and EXP019 | Explicit local model/environment adapters; no scheduling scripts |

The three original native/kernel source files retain their research contents. The independently organized wrappers and reference implementation are tested as described in [VALIDATION.md](docs/VALIDATION.md). The private workspace keeps source digests; infrastructure logs and machine credentials are not part of this repository.

## Upstream projects

OpenVLA-OFT and LIBERO model/environment implementations are not vendored. Users install them separately under their own licenses. HQQ and Optimum Quanto are optional package dependencies and are called through their upstream interfaces. AWQ, GPTQ, and TorchAO are linked reference projects, not bundled implementations.

AGQ's channel calibration follows OmniQuant's learnable clipping and block reconstruction ideas. It does not claim to implement the entire OmniQuant method or its additional transforms. See [the official source directory](docs/REFERENCES.md) and [method principles](docs/PRINCIPLES.md) for attribution.

Literature success-rate tables are attributed to ActQuant in [results/sources.json](results/sources.json). They are not new AGQ measurements. The manuscript and original framework figure were supplied by the authors. The other repository charts are regenerated from the included aggregate data.

No model weights, learned quantizer parameters, calibration observations, or third-party datasets are redistributed.
