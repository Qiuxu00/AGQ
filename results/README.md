# Results and measurement definitions

These small files let readers recompute the reported results without downloading weights. They contain numerical summaries, not robot observations or model parameters.

| File | Contents | Scope |
| --- | --- | --- |
| [quality.json](quality.json) | Ten configurations, four suites each | 6,000 EXP019 episodes |
| [tasks.csv](tasks.csv) | 400 configuration/task records | 40 tasks, 15 episodes per configuration/task |
| [performance.json](performance.json) | Eight table rows and four codec comparisons | Existing RTX3090 measurements |
| [allocations.json](allocations.json) | W1-W4 counts by backbone | 428 target matrices |
| [published_baselines.json](published_baselines.json) | Selected FP16, AWQ, GPTQ, QVLA rows | Literature reports, not our reruns |
| [sources.json](sources.json) | Digests of original evidence | Traceability without internal paths |
| [manifest.json](manifest.json) | Digests of released data | Accidental-change detection |

Run from the repository root:

```bash
python scripts/verify_results.py
python -m unittest discover -s tests -p 'test_results*.py'
```

The verification recomputes task-to-suite-to-configuration totals, checks every denominator, checks whole-model compression ratios, checks codec identity, and detects changed data files. It does not rerun a policy or reconstruct private raw logs.

## Closed-loop success

Each configuration has 600 episodes: 150 in each of Spatial, Object, Goal, and Long. Each suite has ten tasks with 15 episodes per task. Task indices in the CSV are zero-based LIBERO suite indices; `long` corresponds to `libero_10`. All ten configurations underwent calibration, including random allocation. These measurements use the original packed representation without optional entropy coding.

The old 60-episode experiment is excluded. The 96.9% full-precision reference is FP16 reported by [ActQuant v3, Table 1](https://arxiv.org/html/2605.24011v3#S5.T1), not a newly measured BF16 success rate. Literature averages are retained as reported, including rounding; no local episode counts are invented for them. Missing AWQ/GPTQ W2.5 and QVLA W2/W1.5* entries remain missing. The selected references are not an exhaustive comparison of published methods.

W1.5* means W4 vision and W1.5 language, approximately 1.74753 bits across targeted weights. Non-target parameters retain their original precision. Matrix-count averages do not equal parameter-weighted bit averages.

## Storage and timing

- **Model size:** complete standalone model directory including metadata, in decimal GB (10^9 bytes). This differs from target-weight-only compression.
- **Peak GPU memory:** PyTorch peak allocated bytes across ten fixed-input inference calls, with the counter reset after loading and warm-up. Resident tensors, activations, and temporary reconstruction tensors are included; allocator reserve, driver/context memory, and non-PyTorch allocations are excluded.
- **Latency:** mean model-action latency for ten fixed-input calls. Each call produces eight action steps. Observation preprocessing and action postprocessing are excluded.
- **Device:** RTX3090. BF16 is a historical reference measurement; quantized rows come from the subsequent performance evaluation. They are not paired repetitions on one unchanged device.

The main coded table uses CODEBOOK003, referred to as grouped Huffman coding in the repository. At W1.5* it measures **3.537 GB / 310.4 ms**. The alternative CODEBOOK001 result is **2.784 GB / 1771.9 ms**. The two measurements must not be combined. Figure 1 of the manuscript uses the lower-memory alternative at W2 and W1.5*, while the main performance table uses CODEBOOK003.

Bit-exact weight reconstruction does not establish identical closed-loop trajectories. The selected coded W2.5 and W1.5* performance states failed the historical strict trajectory-pairing check. EXP019 did not rerun these coded states; its quality scores therefore remain separate from coded timing/storage results.

## Figures

Install the optional plotting dependencies, then regenerate repository charts:

```bash
python -m pip install matplotlib numpy
python assets/generate_figures.py
```

The charts use only released numerical files. [framework.png](../assets/framework.png) is a crop of Figure 2 from the user-supplied manuscript, preserving its original artwork. It is not recreated by the plotting command.
