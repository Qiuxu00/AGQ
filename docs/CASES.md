# Cases: read the behavior behind the headline

Thirteen measured cases connect the method to concrete outcomes. All success rates come from the released EXP019 counts; storage and timing use their own explicitly labeled performance records. These examples need no weights, private logs, or GPU to inspect.

![Success by precision and suite](../assets/overview.png)

## Five operating points across four suites

Each suite has 150 episodes per configuration. The total is 600, with 15 episodes for each of 40 tasks. Rates are percentages; overall counts are shown directly.

| Configuration | Spatial | Object | Goal | Long | Overall count | Overall rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W4 | 97.3 | 100.0 | 98.7 | 94.7 | 586/600 | 97.7% |
| W3 | 96.7 | 100.0 | 83.3 | 90.7 | 556/600 | 92.7% |
| W2.5 / lambda=0.5 | 80.0 | 100.0 | 17.3 | 84.7 | 423/600 | 70.5% |
| W2 | 29.3 | 92.0 | 1.3 | 17.3 | 210/600 | 35.0% |
| W1.5* | 38.7 | 60.7 | 0.0 | 2.7 | 153/600 | 25.5% |

W1.5* is asymmetric: W4 vision and W1.5 language, approximately 1.75 bits across target weights. The default W2.5 setting uses lambda=0.5.

### 01. A high-retention W3 operating point

W3 retains 556/600 successes (92.7%), compared with 586/600 (97.7%) at W4.

Use as a quality/storage reference; the 5.0-point difference is descriptive, not a significance test.

### 02. Spatial: a useful W2.5 regime

W2.5 reaches 120/150 successes (80.0%) on Spatial.

The pooled 70.5% average hides meaningful differences between suites.

### 03. Object: retention at lower bitwidth

Object achieves 150/150 at W3 and W2.5, and 138/150 (92.0%) at W2.

A strong Object score does not imply the same retention on other suites.

### 04. Goal: an explicit low-bit failure case

Goal falls from 125/150 at W3 to 26/150 at W2.5, 2/150 at W2, and 0/150 at W1.5*.

Inspect failures as well as successes when choosing a budget.

### 05. Long: guided allocation versus random

Guided W2.5 reaches 127/150 successes (84.7%); calibrated random W2.5 reaches 0/150.

Both configurations include calibration; this comparison does not isolate calibration itself.

![Lambda sweep and matrix allocations](../assets/lambda_sweep.png)

### 06. The full lambda sweep

Success counts are 395, 381, 423, 406, and 400 out of 600 for lambda=0, 0.25, 0.5, 0.75, and 1.

The tested default lambda=0.5 is 4.7 percentage points above lambda=0 after rounding.

### 07. An overall winner is not a suite-wise winner

Lambda=0.75 reaches 30/150 on Goal versus 26/150 for lambda=0.5, while its overall total is 406/600 versus 423/600.

The best overall setting need not maximize every task family.

### 08. Calibration cannot rescue arbitrary allocation

Calibrated random W2.5 reaches 49/600 (8.2%); guided W2.5 reaches 423/600 (70.5%).

This supports informed allocation within the calibrated pipeline, not an independent calibration ablation.

### 09. Asymmetric extreme compression

W1.5* retains W4 vision and W1.5 language: 153/600 overall, with 58/150 Spatial versus 44/150 at W2.

W1.5* is approximately 1.75 bits across target weights; quality is not monotonic in every suite.

![Storage and codec tradeoffs](../assets/resource_tradeoff.png)

### 10. Whole-model storage versus latency

W1.5* changes from 2.656 GB / 250.5 ms without coding to 2.534 GB / 310.4 ms with grouped coding.

The coded record gives 6.10x complete-model compression and 3.537 GB peak allocation. Coding adds 59.9 ms.

### 11. Minimum memory and fast decoding are different points

W1.5* CODEBOOK001 measures 2.784 GB / 1771.9 ms; CODEBOOK003 measures 3.537 GB / 310.4 ms.

Do not combine the 2.784 GB minimum with the 310.4 ms latency.

### 12. Coding overhead depends on the configuration

The grouped-coded W2.5 record takes 473.9 ms; the grouped-coded W1.5* record takes 310.4 ms.

Fewer stored bits do not alone determine decoder or full model-action latency.

### 13. From priorities to 428 matrix assignments

At W2.5, changing lambda from 0 to 0.5 changes W1/W2/W3/W4 counts from 54/107/189/78 to 19/150/199/60.

These are matrix counts, not parameter-weighted bit averages.

## Inspect all tasks, including failures

![Success counts for every task at five operating points](../assets/task_heatmap.png)

The image displays all ten tasks in each suite for the five principal operating points. [tasks.csv](../results/tasks.csv) additionally includes all lambda settings and the calibrated random control, for 400 aggregate rows in total. Task IDs follow zero-based LIBERO suite indices. There are no selected successful videos standing in for the evaluation.

## Recompute and reuse

```bash
python scripts/verify_results.py
python assets/generate_figures.py
```

The second command requires Matplotlib and NumPy. The machine-readable [case catalog](../examples/cases/catalog.json) identifies the configurations and scopes used by each case. The [measurement guide](../results/README.md) defines denominators, storage, memory, latency, and literature provenance.

The 96.9% original-precision reference is a reported FP16 result, not a local BF16 rerun. Optional lossless coding is measured separately from EXP019 quality; bit-exact weight reconstruction does not establish strict closed-loop trajectory equivalence. These descriptive cases do not assert statistical significance.
