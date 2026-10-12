# News

[Home](README.md) · [Cases](docs/CASES.md) · [Integration status](docs/INTEGRATION_STATUS.md)

This record distinguishes research milestones from repository preparation. The repository is being prepared privately. Dates below do not imply a public release, paper acceptance, or availability of model weights.

## 2026-10-12 — Repository preparation

- **Updated manuscript basis.** The author-provided *AGQ: Action-Guided Ultra-Low-Bit Quantization for Vision-Language-Action Models* is the current reference for the project title, authors, and method description.
- **Runnable quantization components.** Added a CPU example that exercises W1–W4 channel calibration, packing, grouped lossless coding, and block-wise recovery on generated inputs. Numerical checks compare the recovered execution against calibrated BF16 weights.
- **Method comparison interface.** Added a shared matrix-comparison entry point for RTN and optional official HQQ/Quanto adapters. Outputs explicitly identify reconstruction metrics and do not claim new robot-task results; tested combinations are recorded in the [baseline guide](docs/BASELINES.md).
- **Method walkthrough.** Added a [principles guide](docs/PRINCIPLES.md) explaining the W4 probe, action-and-cost score, byte-budgeted allocation, channel calibration, and block-wise execution.
- **Thirteen experiment cases.** Organized bitwidth, lambda, task-suite, task-level, allocation, and resource questions in the [case guide](docs/CASES.md).
- **Traceable result tables.** Packaged audited aggregate results with explicit distinctions between local closed-loop evaluations, literature baselines, and separate performance measurements.
- **Bilingual entry points.** Added English and Chinese project introductions and a directory of [official upstream references](docs/REFERENCES.md).
- **Weights remain excluded.** This preparation stage covers code, documentation, examples, and aggregate results. No model weights or learned quantizer parameters are distributed.

## 2026-10-11 — Independent teaser revision

- Completed the independent v9 teaser revision using the requested **96.9%** full-precision reference. Source notes preserve that this value is the **FP16 literature reference**; the historical local BF16 result is a different measurement.
- Recorded the resource distinction between the teaser's minimum-memory coding implementation and the main performance table's fast grouped decoder. The **2.78 GB** and **310.4 ms** values are not presented as one configuration.

## 2026-10-09 — EXP019 completed and audited

- Finished **6,000 closed-loop episodes** across ten configurations: four suites, 150 episodes per suite, and 15 episodes per task for each configuration.
- Audited **6,044 result-file digests** in the original experiment archive; the repository contains selected, sanitized aggregates rather than the full private run archive.
- Verified **W4: 586/600**, **W3: 556/600**, **W2: 210/600**, and **W1.5*: 153/600**.
- Completed the five-setting W2.5 lambda comparison. **$\lambda=0.5$ reached 423/600 (70.5%)**, the highest overall success among the tested settings; calibrated random allocation reached 49/600 (8.2%).

## Next steps

The [integration status](docs/INTEGRATION_STATUS.md) records the remaining full-model and optional-backend checks. A public release and project license are author decisions. No date is promised for distributing weights.
