# References and upstream projects

[Home](../README.md) · [Method](PRINCIPLES.md) · [Integration status](INTEGRATION_STATUS.md)

Official project pages below were consulted on **2026-10-12**. These links identify the source implementations and documentation; a link alone does not indicate that a method has been integrated or rerun in AGQ.

## Research and implementation foundations

| Project | Official source | Relevance to AGQ |
| --- | --- | --- |
| OpenVLA-OFT | [Repository](https://github.com/moojink/openvla-oft) · [Paper](https://arxiv.org/abs/2502.19645) | Base VLA policy and upstream setup, model loading, and LIBERO evaluation instructions |
| OpenVLA | [Repository](https://github.com/openvla/openvla) · [Paper](https://arxiv.org/abs/2406.09246) | Underlying VLA architecture |
| LIBERO | [Repository](https://github.com/Lifelong-Robot-Learning/LIBERO) · [Paper](https://arxiv.org/abs/2306.03310) | Four closed-loop robot task suites used by the paper |
| AWQ | [MIT Han Lab repository](https://github.com/mit-han-lab/llm-awq) · [Paper](https://arxiv.org/abs/2306.00978) | Activation-aware weight quantization; source to use for a full AWQ implementation |
| GPTQ | [IST-DASLab repository](https://github.com/IST-DASLab/gptq) · [Paper](https://arxiv.org/abs/2210.17323) | Second-order post-training weight quantization and official evaluation examples |
| OmniQuant | [OpenGVLab repository](https://github.com/OpenGVLab/OmniQuant) · [Paper](https://arxiv.org/abs/2308.13137) | Learnable clipping and block reconstruction underlying AGQ's calibration design |
| HQQ | [Official Dropbox repository](https://github.com/dropbox/hqq) · [Method article](https://dropbox.github.io/hqq_blog/) | Optional half-quadratic weight quantization adapter; works without calibration data |
| Optimum Quanto | [Hugging Face repository](https://github.com/huggingface/optimum-quanto) | Optional general PyTorch quantization adapter; separate from paper baselines |
| TorchAO | [PyTorch repository](https://github.com/pytorch/ao) · [Documentation](https://docs.pytorch.org/ao/stable/) | Official PyTorch quantization workflows and an optional integration reference |

## Baseline result provenance

The manuscript's AWQ, GPTQ, QVLA, and FP16 reference success rates are attributed to **ActQuant**, rather than claimed as new AGQ runs. AGQ's ten EXP019 configurations account for the 6,000 local closed-loop episodes.

- [ActQuant, manuscript version 3](https://arxiv.org/abs/2605.24011v3): comparison table provenance identified by the author-provided AGQ manuscript and local source records.
- [QVLA](https://arxiv.org/abs/2602.03782): the VLA-specific comparison method referenced in that manuscript.
- [VLAQuantBench](https://arxiv.org/abs/2609.25376): cross-device deployment context in the manuscript; its RTX4090 timings are not a matched speedup comparison with AGQ's RTX3090 measurements.

These three paper identifiers are transcribed from the supplied manuscript. The repository's numerical source records preserve which values are literature references and which are local measurements.

## Repository presentation references

AGQ's organization draws on the following practical patterns, with original text and project-specific examples:

- **OpenVLA-OFT:** keep a short entry page and move environment setup and benchmark instructions to dedicated guides. [Official README](https://github.com/moojink/openvla-oft)
- **AWQ:** make results visible early, maintain dated news, and provide concrete examples near the usage instructions. [Official README](https://github.com/mit-han-lab/llm-awq)
- **GPTQ:** distinguish quantization, accuracy evaluation, and kernel benchmarking, with explicit commands and hardware limits. [Official README](https://github.com/IST-DASLab/gptq)
- **OmniQuant:** state supported configurations and separate calibration from deployment. [Official README](https://github.com/OpenGVLab/OmniQuant)
- **Quanto:** explain the quantize, calibrate, freeze, and serialize lifecycle, including which steps are optional. [Official README](https://github.com/huggingface/optimum-quanto)
- **TorchAO:** expose focused examples, tests, and workflow-specific documentation. [Official README](https://github.com/pytorch/ao)
- **HQQ:** document supported precisions and backend restrictions separately from the basic quantizer. [Official README](https://github.com/dropbox/hqq)

Upstream speedups, hardware support, and model coverage do not transfer automatically to AGQ. Consult the upstream revision you install for current compatibility and license terms. AGQ's own project license remains pending an author decision during this private preparation stage.
