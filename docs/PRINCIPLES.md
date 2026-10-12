# How AGQ works

[Home](../README.md) · [Getting started](GETTING_STARTED.md) · [Cases](CASES.md)

AGQ allocates a limited weight budget according to how quantization changes **robot actions** and how much **weight recovery costs**. Calibration compensates for the resulting error. Grouped lossless coding and block-wise recovery then reduce resident and temporary weight storage.

This explanation follows the author-provided manuscript, *AGQ: Action-Guided Ultra-Low-Bit Quantization for Vision-Language-Action Models*, received on October 12, 2026. It describes the research method; the [implementation status](INTEGRATION_STATUS.md) identifies which parts can currently be exercised in this repository.

## 1. Measure actions, not just weight error

A low weight reconstruction error does not necessarily imply a small action deviation. AGQ probes one matrix at W4 while keeping the remaining policy in BF16. It compares the resulting actions on the same inputs.

For matrix $i$, the sensitivity is

$$
S_i = \mathbb{E}_{\mathrm{bal}}\left[\frac{1}{6H}\sum_{h=0}^{H-1}\left\|a_h^{(i)}-a_h^{\mathrm{BF16}}\right\|_2^2\right].
$$

Each action includes three position and three rotation components. The gripper is excluded. The paper uses $H=8$ actions, with tasks and trajectories weighted equally so that long trajectories do not dominate the score.

The shared W4 probe is a practical prior for lower-bit allocation. It assumes some consistency in relative sensitivity across precisions; it does **not** establish identical rankings at every precision.

## 2. Include the cost of recovering weights

Low-bit weights must be recovered for execution. AGQ measures a nonnegative overhead proxy on the same inputs:

$$
C_i=\max(T_i^{\mathrm{W4}}-T_i^{\mathrm{BF16}},0),\qquad
\rho_i=\frac{S_i}{\sum_j S_j}+\lambda\frac{C_i}{\sum_j C_j}.
$$

Sensitivity and overhead are normalized separately within the vision and language backbones. A higher score protects a matrix more strongly against aggressive precision reduction. The paper uses $\lambda=0.5$ by default after comparing five settings.

The W4 measurement sets an allocation priority. It is **not a latency predictor** for W1, W2, or W3.

## 3. Allocate integer bitwidths under real storage budgets

Each target matrix receives one bitwidth from $\{1,2,3,4\}$. Separate vision and language budgets constrain the allocation:

$$
\min_{\{b_i\}}\sum_i\rho_i(4-b_i)^2
\quad\text{subject to}\quad
\sum_i A_i(b_i)\le B.
$$

$A_i(b_i)$ includes the candidate representation's weight codes, metadata, and padding. The reported compression ratios use actual exported bytes.

The policy has **428 target matrices**: 204 vision matrices and 224 language matrices. Parameters outside this target set remain at their original precision. An average such as W2.5 therefore denotes a mixture of integer matrix bitwidths, not a 2.5-bit scalar datatype.

The asymmetric **W1.5\*** configuration keeps vision at W4 and targets an average of W1.5 for language. Its parameter-weighted average over all targeted vision and language weights is approximately **1.75 bits**.

## 4. Learn channel quantizers with frozen weights

AGQ follows the learnable clipping and block reconstruction ideas of [OmniQuant](https://github.com/OpenGVLab/OmniQuant). Original weights and allocated matrix bitwidths remain fixed; calibration updates clipping coefficients.

For W2–W4, a channel's learned lower and upper bounds determine its scale and integer zero point:

$$
Q_i=2^{b_i}-1,\quad
s_j=\frac{u_j-\ell_j}{Q_i},\quad
z_j=\operatorname{clip}\!\left(\operatorname{round}(-\ell_j/s_j),0,Q_i\right).
$$

The integer codes and recovered weights are

$$
q_{jk}=\operatorname{clip}\!\left(\operatorname{round}(W_{jk}/s_j)+z_j,0,Q_i\right),\qquad
\widehat W_{jk}=s_j(q_{jk}-z_j).
$$

The scale is floored at a positive constant and rounded to BF16 before the zero point is computed. **W1 uses a separate fixed-sign, learned-amplitude path**, rather than applying the affine formula indiscriminately.

Frozen BF16 block outputs provide the reconstruction targets. AdamW updates the clipping parameters using block-output mean squared error. Each configuration starts from BF16 independently; the paper assigns a larger calibration share to long-horizon tasks. The small matrix examples in this repository demonstrate numerical components without replacing this full-policy calibration process.

## 5. Compress codes without adding quantization error

After quantization, grouped lossless coding represents multiple integer values as a joint symbol: four W2 values or two W3/W4 values. W1 keeps its raw packed form.

Independent segments support parallel decoding, and prefix lookup reduces serial bit traversal. Per-matrix selection compares complete representations, including codebooks and indexes, and falls back to raw packing when coding is not worthwhile. Scales and zero points are unchanged; recovering the code stream exactly adds no further weight quantization error.

This lossless stage changes the representation of **already quantized** weights. It does not recover information lost during quantization.

## 6. Recover one active block at a time

Encoded weights and decoding metadata stay on the GPU. On block entry, decoding recovers packed codes, and a fused unpacking/dequantization kernel produces the active block's BF16 weights. Matrix multiplication then uses BF16 weights. Temporary weights are released on block exit.

This is compressed storage with block-wise BF16 execution. It should not be described as end-to-end native W1–W4 integer matrix multiplication. Seven language matrices or four vision matrices are recovered per block in the paper's layout.

## Reading the measurements correctly

| Measurement | What it counts |
| --- | --- |
| Target-weight compression | Original bytes divided by representation bytes for targeted matrices only |
| Complete-model compression | Original complete-model bytes divided by exported complete-model bytes, including unquantized modules |
| Peak GPU memory | Peak PyTorch allocated memory during the defined inference window, after loading and warmup |
| Latency | Mean over ten fixed inference inputs; one call emits eight actions; preprocessing and postprocessing are excluded |
| Closed-loop success | Successful episodes divided by evaluated episodes; separate from the timing input count |

The main performance table's coded W1.5* result is **2.534 GB complete-model storage, 3.537 GB peak memory, and 310.4 ms**, using the fast grouped decoder. The teaser's **2.78 GB** point comes from a different, lower-memory coding implementation. They must not be combined into one performance claim.

Closed-loop success and coded-export performance are separate experimental records. The coded performance exports were not rerun through EXP019's 600-episode panel. Their earlier paired checks passed all 10 fixed-observation comparisons but only 9 of 10 closed-loop trajectory comparisons for each of W2.5 and W1.5*, so strict trajectory pairing failed. Exact code-stream and weight recovery does not imply identical closed-loop action sequences or step counts.

The teaser's **96.9%** full-precision success comes from the ActQuant FP16 literature reference even though the original graphic retains a BF16 x-axis label. The historical local BF16 run was 59/60; it is not the source of the 96.9% value and is not mixed into EXP019.

See [the case guide](CASES.md) and the machine-readable results for the measured configurations and their provenance.
