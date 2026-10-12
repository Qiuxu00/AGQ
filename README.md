<div align="center">

# AGQ

### Action-Guided Ultra-Low-Bit Quantization<br>for Vision-Language-Action Models

**Preserve actions. Allocate precision. Recover weights on demand.**

Xuhong Qiu · Zhuoyu Wu · Qiawei Zheng · Wenhui Ou · Wenqi Fang · Raphael C.W- Phan

[Paper](paper/AGQ_ISCAS_27.pdf) · [Principles](docs/PRINCIPLES.md) · [Get started](docs/GETTING_STARTED.md) · [Cases](docs/CASES.md) · [Results](results/) · [News](NEWS.md) · [中文](README.zh-CN.md)

**OpenVLA-OFT · LIBERO · 1–4 bit matrix allocation · No model weights included**

</div>

AGQ compresses vision-language-action policies by assigning precision to the matrices that matter for **action prediction and weight recovery cost**. Per-channel calibration restores accuracy; grouped lossless coding and block-wise recovery reduce the memory needed to store and execute the policy.

This repository brings together the method, quantization components, runnable numerical examples, and traceable results. It is being prepared privately; the manuscript has no public publication identifier recorded here, and the project license awaits an author decision.

| W4 success | W3 success | W2.5 success | Complete-model compression |
| :---: | :---: | :---: | :---: |
| **97.7%** | **92.7%** | **70.5%** | **6.10×** |
| 586 / 600 episodes | 556 / 600 episodes | 423 / 600 episodes | Coded W1.5* storage |

<p align="center">
  <img src="assets/overview.png" alt="AGQ success rates and configuration comparison, with 600 evaluated episodes per configuration" width="900">
</p>

The success rates above come from EXP019. The compression figure comes from a separate performance export; it is not a claim that the coded export was rerun through those 600 episodes. W1.5* means W4 vision and average-W1.5 language, approximately 1.75 bits over all target weights.

## News

- **2026-10-12 — Quantization tools.** Added a W1–W4 calibration-to-execution example and a common matrix comparison interface for RTN and optional official HQQ/Quanto adapters. [Try them](docs/GETTING_STARTED.md).
- **2026-10-12 — Method guide.** Added a readable walkthrough of action sensitivity, budgeted allocation, calibration, and compressed execution, aligned with the latest author-provided AGQ manuscript.
- **2026-10-12 — Results and cases.** Organized the audited episode counts, per-task results, allocation analysis, and resource measurements into a browsable case guide with machine-readable sources.
- **2026-10-11 — Figure provenance.** Completed the independent v9 teaser revision and documented that its 96.9% reference is the FP16 literature value.
- **2026-10-09 — Closed-loop evaluation.** Completed and audited 6,000 episodes across ten configurations and four LIBERO suites. W2.5 with $\lambda=0.5$ reached 423/600 successes.

See [NEWS.md](NEWS.md) for the fuller record. Dates reflect research and repository preparation milestones.

## The principle in three steps

<p align="center">
  <img src="assets/framework.png" alt="AGQ framework: VLA architecture, action- and cost-informed bit allocation, per-channel calibration, and grouped lossless coding" width="1100">
</p>

1. **Allocate with actions and cost.** Probe each matrix at W4. Combine its action deviation and measured recovery overhead, then allocate W1–W4 under separate vision and language storage budgets.
2. **Calibrate with regular layouts.** Keep one bitwidth per matrix and learn channel quantizers through block reconstruction. Original model weights and assigned bitwidths stay fixed.
3. **Encode and recover by block.** Group integer codes for lossless compression. Recover only the active block's BF16 weights, use them for matrix multiplication, and release them afterward.

The W4 overhead is an allocation proxy, not a predictor of every candidate precision's latency. See [the equations and design choices](docs/PRINCIPLES.md).

## Start with something useful

```bash
python -m pip install -c requirements/cpu-constraints.txt -e ".[test]"
python scripts/agq_quantize_demo.py --bits 1 2 3 4 --epochs 4
agq methods
agq compare --methods rtn --bits 4 --output outputs/compare_rtn_w4.json
python scripts/verify_results.py
```

The AGQ demo runs calibration, packing, lossless coding, and block-wise execution on generated tensors. Official HQQ adapters have been checked at 1/2/3/4/8 bits, and the Quanto adapter exposes verified CPU int8. Follow the [step-by-step guide](docs/GETTING_STARTED.md) to install them. The constrained CPU environment is separate from the full OpenVLA-OFT GPU environment. These examples write numerical metrics, not model weights.

**Validation:** all **74 local tests passed without skips**, covering 52 AGQ component checks, 16 matrix-adapter checks, and 6 result-audit checks. See [the validation record](docs/VALIDATION.md) for the environment and scope.

| Your goal | Entry point | What you can inspect |
| --- | --- | --- |
| Understand AGQ | [Principles](docs/PRINCIPLES.md) | Equations, assumptions, and storage/execution design |
| Run quantization examples | [Getting started](docs/GETTING_STARTED.md) | Small local inputs and commands with explicit prerequisites |
| Check the paper's numbers | [Case guide](docs/CASES.md) · [Results](results/) | Exact episode counts, suite/task breakdowns, and performance definitions |
| Compare or extend methods | [Integration status](docs/INTEGRATION_STATUS.md) | What is implemented, what is optional, and what has been tested |
| Trace an upstream idea | [References](docs/REFERENCES.md) | Official OpenVLA-OFT, AWQ, GPTQ, OmniQuant, Quanto, and TorchAO sources |

Model checkpoints and calibration datasets are not distributed. The small numerical examples do not need pretrained model weights. Full-policy inference and closed-loop reproduction require separately obtained model assets and the corresponding environment; see the implementation status before planning a run.

## Explore the cases

The [case guide](docs/CASES.md) contains **13 cases** addressing questions such as:

| Question | Case material |
| --- | --- |
| How much success remains at W4, W3, W2.5, and W2? | Bitwidth sweep with integer success counts |
| Does guided allocation help after calibration? | W2.5 guided and calibrated-random comparison |
| How does $\lambda$ affect the result? | Five settings, overall and per-suite outcomes |
| Which task suites remain difficult? | Spatial, Object, Goal, and Long breakdowns |
| What happens at W1.5*? | Asymmetric vision/language allocation and observed limitations |
| Where do additional compression savings come from? | Uncoded and coded model storage, peak memory, and latency |
| Which individual tasks fail? | A task-level map covering all 40 tasks |
| How do allocation counts change with the budget? | W1–W4 matrix counts for vision and language |

<details>
<summary><strong>More analysis views</strong></summary>

<p align="center">
  <img src="assets/lambda_sweep.png" alt="W2.5 success rates across five lambda settings" width="850">
</p>
<p align="center">
  <img src="assets/resource_tradeoff.png" alt="Storage and inference measurements with coded and uncoded configurations distinguished" width="850">
</p>

</details>

## Results you can trace

Each configuration contains **600 closed-loop episodes**: 150 per suite and 15 per task. The five main bitwidth configurations, four additional W2.5 lambda settings, and one calibrated-random configuration make **ten configurations and 6,000 episodes** in total.

| AGQ configuration | Successes | Overall success |
| --- | ---: | ---: |
| W4 | 586 / 600 | 97.7% |
| W3 | 556 / 600 | 92.7% |
| W2.5, $\lambda=0.5$ | 423 / 600 | 70.5% |
| W2 | 210 / 600 | 35.0% |
| W1.5* | 153 / 600 | 25.5% |
| Calibrated random W2.5 | 49 / 600 | 8.2% |

Sources: [quality.json](results/quality.json), [tasks.csv](results/tasks.csv), and [the case guide](docs/CASES.md). The 6,000 episodes cover AGQ study configurations; AWQ, GPTQ, QVLA, and the FP16 reference are literature values recorded separately in [published_baselines.json](results/published_baselines.json).

**Read the resource numbers with their scope.** The main performance table's coded W1.5* configuration occupies **2.534 GB**, gives **6.10× complete-model compression**, and measures **3.537 GB** peak allocated GPU memory and **310.4 ms** per inference call on RTX3090. Its storage and timing are recorded in [performance.json](results/performance.json). The original paper teaser's **2.78 GB** uses another coding implementation and must not be paired with 310.4 ms. The performance records also retain the failed strict closed-loop trajectory-pairing checks; lossless weight recovery does not guarantee identical trajectories.

## Repository map

```text
AGQ/
├── src/agq/          # Quantization components and command interface
├── scripts/          # Numerical checks and research utilities
├── examples/         # Small examples and case catalog
├── tests/            # Numerical and interface checks
├── results/          # Audited aggregates; no model parameters
├── assets/           # Figures for the method and case guide
├── docs/             # Principles, setup, cases, and integration status
├── paper/            # Author-provided manuscript
├── NEWS.md
└── CITATION.cff
```

## Citation and acknowledgments

The current citation describes an **unpublished manuscript**; acceptance, proceedings, and a public paper identifier are not claimed.

```bibtex
@unpublished{qiu2026agq,
  title  = {AGQ: Action-Guided Ultra-Low-Bit Quantization for Vision-Language-Action Models},
  author = {Qiu, Xuhong and Wu, Zhuoyu and Zheng, Qiawei and Ou, Wenhui and Fang, Wenqi and Phan, Raphael C.W-},
  year   = {2026},
  note   = {Unpublished manuscript}
}
```

AGQ builds on [OpenVLA-OFT](https://github.com/moojink/openvla-oft) and LIBERO. Its calibration design follows OmniQuant's clipping and block reconstruction ideas. We also thank the authors of AWQ, GPTQ, and the other projects listed in [References](docs/REFERENCES.md). Their licenses and attribution remain separate from this project's pending license decision.
