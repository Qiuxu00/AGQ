<div align="center">

# AGQ

### 面向视觉—语言—动作模型的动作引导超低位宽量化

**保留动作能力 · 分配矩阵精度 · 按需恢复权重**

Xuhong Qiu · Zhuoyu Wu · Qiawei Zheng · Wenhui Ou · Wenqi Fang · Raphael C.W- Phan

[论文](paper/AGQ_ISCAS_27.pdf) · [论文图表](docs/PAPER_FIGURES.md) · [方法原理](docs/PRINCIPLES.md) · [开始使用](docs/GETTING_STARTED.md) · [13 个案例](docs/CASES.md) · [English](README.md)

</div>

AGQ 根据**动作敏感度与权重恢复开销**分配矩阵精度，将矩阵级位宽分配、逐通道校准和成组无损编码结合起来，在压缩 OpenVLA-OFT 策略的同时保留实用的闭环任务能力。

| W4 成功率 | W3 成功率 | W2.5 成功率 | 整模型压缩率 |
| :---: | :---: | :---: | :---: |
| **97.7%** | **92.7%** | **70.5%** | **6.10 倍** |
| 586 / 600 条 | 556 / 600 条 | 423 / 600 条 | 编码后的 W1.5* 存储结果 |

<p align="center">
  <a href="assets/paper/figure1_compression_success.png"><img src="assets/paper/figure1_compression_success.png" alt="论文图 1：六个精度设置的成功率、目标权重压缩比与峰值显存" width="760"></a>
</p>

**图 1 · 压缩与成功率的取舍。** 使用论文原图，所有图表均可点击放大。柱形表示**目标权重**压缩率；96.9% 来自 **FP16 文献参考**，原图保留了 BF16 标签。低位宽的最小显存与主性能表使用不同的解码实现。[图表口径与数据](docs/PAPER_FIGURES.md#figure-1--compression-success-and-memory)。

## 更新记录

- **2026-10-12：仓库已公开。** 代码、文档与汇总结果现已可公开访问；模型权重与已学习的量化参数仍不分发，项目许可证待定。
- **2026-10-12：补全论文图表。** 首页直接展示四幅原图和两张完整结果表，并提供[来源与数据索引](docs/PAPER_FIGURES.md)。
- **2026-10-12：代码与示例。** 提供 W1–W4 从校准到执行的组件、官方 HQQ/Quanto 矩阵适配，**74 项本地测试通过**。[开始使用](docs/GETTING_STARTED.md)。
- **2026-10-12：十三个案例。** 整理精度、分配、任务类别、失败与资源分析，并提供结构化结果。[阅读案例](docs/CASES.md)。
- **2026-10-09：闭环评测完成。** 十配置、四套 LIBERO 任务、共 **6,000 条闭环**完成并核验；W2.5、$\lambda=0.5$ 达到 423/600。

[全部项目动态](NEWS.md)

## 方法框架

<p align="center">
  <a href="assets/paper/figure2_framework.png"><img src="assets/paper/figure2_framework.png" alt="论文图 2：模型架构、动作与开销引导分配、逐通道校准，以及成组编码和逐块执行" width="1100"></a>
</p>

**图 2 · 从原精度策略到压缩执行。** 方法面向 **428 个矩阵**：视觉部分 204 个，语言部分 224 个。每个矩阵只采用一种整数位宽，保持规则的数据布局。

| 1 · 测量与分配 | 2 · 校准 | 3 · 编码与执行 |
| --- | --- | --- |
| 通过 W4 探测动作偏差和恢复开销，在视觉与语言的独立预算内分配 W1–W4。 | 固定原权重和位宽，以块重构学习逐通道量化参数。 | 成组无损编码；恢复当前块所需的 BF16 权重，计算后释放临时权重。 |

W4 开销用作分配依据；矩阵乘法仍使用恢复后的 BF16 权重。[公式、假设与实现](docs/PRINCIPLES.md)。

## 闭环任务成功率

<p align="center">
  <a href="assets/paper/table1_task_success.png"><img src="assets/paper/table1_task_success.png" alt="论文完整成功率表：原精度参考、AWQ、GPTQ、QVLA 和 AGQ 在四套 LIBERO 任务上的结果" width="880"></a>
</p>

**论文完整成功率表 · 四类任务与实际能力边界。** AGQ 在 **W3 达到 92.7%**，在 **W2.5 达到 70.5%**。W3 比报告的 AWQ 和 GPTQ 结果分别高 1.2 和 3.7 个百分点。W2.5 的 Object 保留 100.0%，而 Goal 降至 17.3%；总体成功率之外，各任务类别的差异同样清晰可见。

每个 AGQ 研究配置包含 **600 条闭环**：每套 150 条、每任务 15 条，十配置合计 6,000 条。AWQ、GPTQ、QVLA 和 96.9% 原精度参考均为文献报告值。W1.5* 指视觉 W4、语言平均 W1.5，全部目标权重的参数加权平均约为 **1.75 位**。[成功次数](results/quality.json) · [全部 40 个任务](results/tasks.csv) · [文献结果](results/published_baselines.json)。

## 为什么同时考虑动作与开销？

<p align="center">
  <a href="assets/paper/figure3_allocation_sensitivity.png"><img src="assets/paper/figure3_allocation_sensitivity.png" alt="论文图 3：位宽分配表及矩阵敏感度、执行开销、综合评分的三维图" width="920"></a>
</p>

**图 3 · 矩阵差异对应不同的精度分配。** 论文将分配表与三维图合并，展示测量依据与分配结果。在 W2.5、$\lambda=0.5$ 下，**W1 / W2 / W3 / W4 的矩阵数为 19 / 150 / 199 / 60**。这里统计矩阵个数，不能直接作为参数加权平均位宽。[分配数据](results/allocations.json)。

<p align="center">
  <a href="assets/paper/figure4_lambda_success.png"><img src="assets/paper/figure4_lambda_success.png" alt="论文图 4：五个 lambda 设置的 W2.5 分套和总体成功率，以及校准后随机分配和 QVLA 参考" width="700"></a>
</p>

**图 4 · 在闭环中检验分配选择。** W2.5 的引导分配达到 **63.5–70.5%**，校准后随机分配为 **8.2%**。已测试的默认 $\lambda=0.5$ 比只考虑敏感度的设置高 **4.7 个百分点**，但并非每套任务均最优。[五组设置与案例分析](docs/CASES.md)。

## 存储与推理性能

<p align="center">
  <a href="assets/paper/table2_storage_performance.png"><img src="assets/paper/table2_storage_performance.png" alt="论文完整性能表：RTX3090 上的整模型存储、压缩率、峰值已分配显存和时延" width="880"></a>
</p>

**论文完整性能表 · 同时展示存储收益与执行代价。** 成组编码后的 W1.5* 占用 **2.534 GB**，对应 **6.10 倍整模型压缩**。同一性能记录在 RTX3090 上测得 **3.537 GB 峰值已分配显存**和 **310.4 ms 单次推理时延**；编码相较未编码版本增加 59.9 ms。

图 1 的 **2.784 GB** 最小显存来自 **CODEBOOK001**，性能表的 **3.537 GB / 310.4 ms** 来自 **CODEBOOK003**。EXP019 成功率与编码性能记录分别测量。[完整定义与闭环轨迹配对结果](docs/PAPER_FIGURES.md#measurement-notes) · [性能数据](results/performance.json)。

## 阅读 13 个案例

| 分析方向 | 可查看的内容 |
| --- | --- |
| **不同精度的任务能力** | W3 保留率、W2.5 的 Spatial/Object 结果、Goal 失败与 W1.5* 非对称配置 |
| **分配选择** | 引导与校准后随机分配、五个 $\lambda$ 设置，以及 428 个矩阵的分配 |
| **部署取舍** | 整模型存储、编码开销、最小显存与较快解码之间的差别 |
| **每个任务** | 分套结果与全部 40 个任务，包含失败情况 |

[打开图文案例指南](docs/CASES.md) · [浏览结果文件](results/) · [查看六项论文图表](docs/PAPER_FIGURES.md)

## 运行代码

```bash
python -m pip install -c requirements/cpu-constraints.txt -e ".[test]"
python scripts/agq_quantize_demo.py --bits 1 2 3 4 --epochs 4
agq methods
agq compare --methods rtn --bits 4 --output outputs/compare_rtn_w4.json
python scripts/verify_results.py
```

AGQ 示例在自动生成的张量上执行校准、打包、无损编码和逐块推理。可选的官方 **HQQ** 适配已核验 **1/2/3/4/8 位**，**Quanto** 提供已核验的 **CPU int8**。这些矩阵示例报告数值误差；完整策略推理使用独立的 OpenVLA-OFT 环境及用户自行准备的模型资产。

**验证情况：74 项本地测试全部通过，无跳过**，其中 AGQ 核心 52 项、矩阵适配 16 项、结果核验 6 项。本次图文更新未新增 GPU 或闭环实验。仓库不分发模型权重与已学习的量化参数。

[开始使用](docs/GETTING_STARTED.md) · [复现 AGQ](docs/REPRODUCING_AGQ.md) · [方法集成](docs/INTEGRATION_STATUS.md) · [其他量化方法](docs/BASELINES.md) · [验证记录](docs/VALIDATION.md)

## 引用与致谢

```bibtex
@unpublished{qiu2026agq,
  title  = {AGQ: Action-Guided Ultra-Low-Bit Quantization for Vision-Language-Action Models},
  author = {Qiu, Xuhong and Wu, Zhuoyu and Zheng, Qiawei and Ou, Wenhui and Fang, Wenqi and Phan, Raphael C.W-},
  year   = {2026},
  note   = {Unpublished manuscript}
}
```

AGQ 基于 [OpenVLA-OFT](https://github.com/moojink/openvla-oft) 和 LIBERO；逐通道校准借鉴 OmniQuant 的裁剪学习与块重构。上游工作与署名说明见[官方参考项目](docs/REFERENCES.md)和[第三方说明](THIRD_PARTY_NOTICES.md)。

仓库已可公开访问；论文按未发表稿件记录，项目许可证等待作者确定。
