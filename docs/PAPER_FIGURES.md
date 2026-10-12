# Figures and tables from the paper

[Home](../README.md) · [中文首页](../README.zh-CN.md) · [Full paper](../paper/AGQ_ISCAS_27.pdf) · [13 cases](CASES.md) · [Measurement definitions](../results/README.md)

All six images below are high-resolution crops of the **author-provided six-page AGQ manuscript**. They preserve the paper's artwork, labels, and numerical values. Cropping removes the surrounding page text; it does not redraw the plots. Click an image to open its full-size PNG, or follow the page link to read the original caption and discussion.

| Paper item | Source page | Read it for | Machine-readable companion |
| --- | :---: | --- | --- |
| [Figure 1](#figure-1--compression-success-and-memory) | [1](../paper/AGQ_ISCAS_27.pdf#page=1) | Success, target-weight compression, and peak memory | [Quality](../results/quality.json), [performance](../results/performance.json), [literature](../results/published_baselines.json) |
| [Figure 2](#figure-2--the-method) | [2](../paper/AGQ_ISCAS_27.pdf#page=2) | Model architecture and the full compression pipeline | [Principles](PRINCIPLES.md), [implementation status](INTEGRATION_STATUS.md) |
| [Figure 3](#figure-3--allocation-and-matrix-differences) | [3](../paper/AGQ_ISCAS_27.pdf#page=3) | Matrix priorities and bitwidth assignments | [Allocation counts](../results/allocations.json) |
| [Figure 4](#figure-4--lambda-and-closed-loop-success) | [4](../paper/AGQ_ISCAS_27.pdf#page=4) | Five allocation settings and four task suites | [Quality](../results/quality.json), [task records](../results/tasks.csv) |
| [Task success table](#complete-success-table) | [4](../paper/AGQ_ISCAS_27.pdf#page=4) | All paper methods, precisions, and task suites | [Quality](../results/quality.json), [literature](../results/published_baselines.json) |
| [Performance table](#complete-performance-table) | [4](../paper/AGQ_ISCAS_27.pdf#page=4) | Whole-model size, peak memory, and latency | [Performance](../results/performance.json) |

The paper displays both result tables without numbered table titles; the image filenames provide stable repository identifiers. The [extraction manifest](../assets/paper/manifest.json) records source pages, crop coordinates, and file digests. The [extraction script](../assets/paper/extract_figures.py) documents the rendering process.

## Figure 1 — Compression, success, and memory

[![Paper Figure 1](../assets/paper/figure1_compression_success.png)](../assets/paper/figure1_compression_success.png)

The first figure places AGQ and QVLA success alongside compression of the **targeted weights**, from the original precision through W1.5*. The memory row gives inference peak allocation for the selected implementations. It summarizes three different quantities rather than claiming they are interchangeable.

- The **96.9%** reference comes from **FP16 reported by ActQuant**. The supplied paper retains the **BF16** axis label; this is not a newly measured BF16 success result.
- The QVLA W2.5 zero is a literature result. Its W2 and W1.5* zero extensions are illustrative; they are not additional reported measurements. Missing literature entries remain missing in the released data.
- The **9.12×** W1.5* bar measures target-weight compression. The **6.10×** value elsewhere includes the complete model after coding.
- W2 and W1.5* memory minima use CODEBOOK001; the main coded performance table uses CODEBOOK003. See [measurement notes](#measurement-notes).

Source: [paper page 1](../paper/AGQ_ISCAS_27.pdf#page=1). Data: [quality](../results/quality.json), [published baselines](../results/published_baselines.json), and [performance](../results/performance.json).

## Figure 2 — The method

[![Paper Figure 2](../assets/paper/figure2_framework.png)](../assets/paper/figure2_framework.png)

The four panels connect the OpenVLA-OFT architecture to action-and-cost allocation, per-channel quantizer learning, and grouped lossless coding with block-wise execution. Vision contains **204 target matrices** and language contains **224**, for **428 total**. Original weights and assigned bitwidths stay fixed during quantizer learning. At execution, recovered BF16 weights are used for matrix multiplication and released after the block.

Source: [paper page 2](../paper/AGQ_ISCAS_27.pdf#page=2). Read the [principles and equations](PRINCIPLES.md) and [reproduction guide](REPRODUCING_AGQ.md) alongside the figure.

## Figure 3 — Allocation and matrix differences

[![Paper Figure 3](../assets/paper/figure3_allocation_sensitivity.png)](../assets/paper/figure3_allocation_sensitivity.png)

The combined allocation table and three-dimensional views show how matrices differ in action sensitivity and recovery overhead, and how the two normalized measurements produce the final score. At W2.5 and $\lambda=0.5$, the combined W1/W2/W3/W4 counts are **19/150/199/60**. These are counts of matrices, not a parameter-weighted bit average.

The repository releases the aggregate allocation counts. The three-dimensional panels are preserved directly from the manuscript; their full underlying per-matrix profiling inputs are not distributed here.

Source: [paper page 3](../paper/AGQ_ISCAS_27.pdf#page=3). Data: [allocations.json](../results/allocations.json). Discussion: [allocation cases](CASES.md).

## Figure 4 — Lambda and closed-loop success

[![Paper Figure 4](../assets/paper/figure4_lambda_success.png)](../assets/paper/figure4_lambda_success.png)

For $\lambda=0,0.25,0.5,0.75,1$, W2.5 success counts are **395, 381, 423, 406, and 400 out of 600**. The tested default $\lambda=0.5$ has the highest overall score, **70.5%**, while some task suites favor other settings. Calibrated random allocation reaches **49/600, or 8.2%**.

All of these AGQ study configurations include calibration. The comparison supports guided allocation within that pipeline; it does not isolate calibration as an independent ablation.

Source: [paper page 4](../paper/AGQ_ISCAS_27.pdf#page=4). Data: [quality.json](../results/quality.json) and [tasks.csv](../results/tasks.csv).

## Complete success table

[![Complete paper success table](../assets/paper/table1_task_success.png)](../assets/paper/table1_task_success.png)

The table preserves all paper rows: the original-precision reference, AWQ, GPTQ, QVLA, and AGQ, including each task suite and the reported average. AGQ's W3 **92.7%** exceeds the reported AWQ and GPTQ averages by **1.2** and **3.7 percentage points**. W2.5 reaches **70.5%**, with substantial differences between suites.

AGQ rows use EXP019: **600 episodes per configuration**, **150 per suite**, and **15 per task**. Other methods and the **96.9% FP16 reference** come from ActQuant's reported comparison. Their source-reported rounded averages are retained; local episode counts are not invented for them. W1.5* means W4 vision and average-W1.5 language, approximately **1.74753 bits across target weights**.

Source: [paper page 4](../paper/AGQ_ISCAS_27.pdf#page=4). Data: [quality.json](../results/quality.json), [tasks.csv](../results/tasks.csv), and [published_baselines.json](../results/published_baselines.json).

## Complete performance table

[![Complete paper performance table](../assets/paper/table2_storage_performance.png)](../assets/paper/table2_storage_performance.png)

The table presents complete-model storage, compression relative to BF16, peak GPU memory, and inference latency on RTX3090. Uncoded configurations measure **243.6–254.4 ms**. Grouped-coded W1.5* records **2.534 GB model storage**, **6.10× complete-model compression**, **3.537 GB peak allocated memory**, and **310.4 ms latency**.

Source: [paper page 4](../paper/AGQ_ISCAS_27.pdf#page=4). Data: [performance.json](../results/performance.json).

## Measurement notes

| Scope | Interpretation |
| --- | --- |
| Target-weight compression | Compression of the selected vision/language matrices; Figure 1's W1.5* value is **9.12×**. |
| Complete-model compression | Includes remaining parameters and metadata; coded W1.5* is **6.10×**, from 15.457 to 2.534 GB. |
| Minimum-memory decoder | **CODEBOOK001:** W1.5* measures **2.784 GB / 1771.9 ms**; Figure 1 shows this lower peak. |
| Main table decoder | **CODEBOOK003:** W1.5* measures **3.537 GB / 310.4 ms**. Do not combine its latency with CODEBOOK001's peak. |
| Peak memory | PyTorch peak allocated memory across ten fixed-input calls after loading and warm-up; includes resident and temporary tensors, but not allocator reserve or driver/context memory. |
| Latency | Mean model-action latency of ten fixed-input calls; each call predicts eight actions. Observation preprocessing and action postprocessing are excluded. |
| Quality versus coded execution | EXP019 evaluates the original packed representation. Coded storage/timing are separate measurements; these coded states were not rerun through EXP019's 600 episodes per configuration. |
| Trajectory pairing | The selected coded W2.5 and W1.5* states passed their historical offline action checks (10/10 each), but each matched 9/10 closed-loop trajectories and failed the strict trajectory-pairing requirement. This is a trajectory check, not a failed weight-recovery check. |

The full [measurement guide](../results/README.md) and [case guide](CASES.md) retain the detailed distinctions. This gallery adds **no new GPU measurements, closed-loop evaluations, or model weights**.
