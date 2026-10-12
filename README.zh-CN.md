<div align="center">

# AGQ

### 面向视觉—语言—动作模型的动作引导超低位宽量化

**保留动作能力 · 分配矩阵精度 · 按需恢复权重**

Xuhong Qiu · Zhuoyu Wu · Qiawei Zheng · Wenhui Ou · Wenqi Fang · Raphael C.W- Phan

[论文](paper/AGQ_ISCAS_27.pdf) · [方法原理](docs/PRINCIPLES.md) · [开始使用](docs/GETTING_STARTED.md) · [案例分析](docs/CASES.md) · [结果数据](results/) · [更新记录](NEWS.md) · [English](README.md)

**OpenVLA-OFT · LIBERO · 矩阵级 1–4 位分配 · 不提供模型权重**

</div>

AGQ 根据量化引起的**动作偏差**和**权重恢复开销**分配矩阵精度，在存储预算内尽量保留闭环任务能力。逐通道校准补偿量化误差，成组无损编码和逐块恢复进一步降低常驻权重与临时权重的存储需求。

本仓库集中提供方法说明、量化组件、可运行的数值示例，以及可追溯的实验结果。目前处于私有整理阶段；尚未填写公开论文标识，项目许可证等待作者确定。

| W4 成功率 | W3 成功率 | W2.5 成功率 | 整模型压缩率 |
| :---: | :---: | :---: | :---: |
| **97.7%** | **92.7%** | **70.5%** | **6.10 倍** |
| 586 / 600 条 | 556 / 600 条 | 423 / 600 条 | 编码后的 W1.5* 存储结果 |

<p align="center">
  <img src="assets/overview.png" alt="AGQ 各配置闭环成功率，每配置 600 条" width="900">
</p>

成功率来自 EXP019，压缩率来自单独的性能导出记录；两者并不表示编码后的导出模型又执行了相同的 600 条闭环。W1.5* 指视觉 W4、语言平均 W1.5，全部目标权重的实际参数加权平均约为 1.75 位。

## 更新记录

- **2026-10-12：量化工具。** 提供 W1–W4 从校准到执行的示例，以及 RTN 和可选官方 HQQ/Quanto 接口的统一矩阵比较入口。[开始使用](docs/GETTING_STARTED.md)。
- **2026-10-12：方法文档。** 根据作者提供的最新版 AGQ 论文整理动作敏感度、预算分配、校准与压缩执行的完整说明。
- **2026-10-12：结果与案例。** 整理已核验的成功次数、逐任务表现、分配数量和资源测量，提供案例入口及结构化数据。
- **2026-10-11：首图来源核验。** 独立首图 v9 修订完成，明确其中 96.9% 是文献 FP16 参考值。
- **2026-10-09：闭环评测完成。** 十配置、四套 LIBERO 任务、共 6,000 条评测完成并核验；W2.5、$\lambda=0.5$ 为 423/600。

更多记录见 [NEWS.md](NEWS.md)。以上为实际工作里程碑，不代表这些日期曾有公开版本发布。

## 三步理解方法

<p align="center">
  <img src="assets/framework.png" alt="AGQ 架构、分配、校准与成组编码执行框架" width="1100">
</p>

1. **用动作与开销指导分配。** 逐矩阵做 W4 探测，结合相同输入下的动作偏差与恢复开销，在视觉和语言各自预算内分配 W1–W4。
2. **保持规则布局并学习量化参数。** 每个矩阵使用一个整数位宽，固定原始权重与位宽，以块重构学习逐通道量化参数。
3. **编码存储、逐块恢复。** 成组无损编码整数权重，只恢复当前块所需的 BF16 权重；矩阵乘法结束后按块释放临时权重。

W4 开销只作为分配优先级依据，并不预测每种候选位宽的时延。公式与假设详见[方法原理](docs/PRINCIPLES.md)。

## 从这里开始

```bash
python -m pip install -c requirements/cpu-constraints.txt -e ".[test]"
python scripts/agq_quantize_demo.py --bits 1 2 3 4 --epochs 4
agq methods
agq compare --methods rtn --bits 4 --output outputs/compare_rtn_w4.json
python scripts/verify_results.py
```

AGQ 示例在自动生成的张量上执行校准、打包、无损编码和逐块推理。官方 HQQ 接口已核验 1/2/3/4/8 位，Quanto 接口提供已核验的 CPU int8。按[使用指南](docs/GETTING_STARTED.md)安装对应依赖。受约束的 CPU 环境与完整 OpenVLA-OFT GPU 环境分别准备；上述示例只输出数值指标，不输出模型权重。

**验证情况：74 项本地测试全部通过，无跳过**，其中 AGQ 核心 52 项、量化方法接口 16 项、结果核验 6 项。具体环境及验证范围见[验证记录](docs/VALIDATION.md)。

| 想做什么 | 入口 | 提供什么 |
| --- | --- | --- |
| 了解 AGQ | [方法原理](docs/PRINCIPLES.md) | 公式、设计动机、假设和执行方式 |
| 运行量化示例 | [开始使用](docs/GETTING_STARTED.md) | 明确输入与依赖的小规模数值示例 |
| 检查论文数据 | [案例分析](docs/CASES.md) · [结果目录](results/) | 成功次数、逐套逐任务结果和性能口径 |
| 尝试其他量化方法 | [集成状态](docs/INTEGRATION_STATUS.md) | 已实现内容、可选依赖和实际验证范围 |
| 追溯上游工作 | [参考项目](docs/REFERENCES.md) | OpenVLA-OFT、AWQ、GPTQ、OmniQuant、Quanto 与 TorchAO 官方入口 |

本仓库不分发模型权重或校准数据。小规模数值示例不需要预训练权重；完整政策推理和闭环复现需要自行准备模型资产及对应环境，执行前请阅读集成状态。

## 不止一个总成功率

[案例指南](docs/CASES.md)包含 **13 个案例**，分别回答下面的问题，并给出对应的可核对数据：

| 问题 | 对应内容 |
| --- | --- |
| W4、W3、W2.5 和 W2 各保留多少能力？ | 位宽扫描与原始成功次数 |
| 随机分配经过校准后能否接近 AGQ？ | W2.5 指导分配与校准后随机分配对比 |
| $\lambda$ 怎样影响结果？ | 五个设置的总体和逐套表现 |
| 哪类任务仍然困难？ | Spatial、Object、Goal、Long 分套结果 |
| W1.5* 意味着什么？ | 视觉与语言非对称精度及其实际能力限制 |
| 编码额外节省多少空间，代价是什么？ | 未编码与编码的模型存储、显存和时延 |
| 具体哪些任务容易失败？ | 40 个任务的结果分布 |
| 预算改变后如何分配各位宽？ | 视觉与语言 W1–W4 矩阵数量 |

<details>
<summary><strong>展开更多分析图</strong></summary>

<p align="center">
  <img src="assets/lambda_sweep.png" alt="五个 lambda 设置的成功率" width="850">
</p>
<p align="center">
  <img src="assets/resource_tradeoff.png" alt="区分编码与未编码的资源和时延测量" width="850">
</p>

</details>

## 可追溯的结果

每个配置包含 **600 条闭环**：每套 150 条、每任务 15 条。五个主要位宽配置、其余四个 W2.5 设置和一个校准后随机分配配置，共 **十配置、6,000 条**。

| AGQ 配置 | 成功次数 | 总体成功率 |
| --- | ---: | ---: |
| W4 | 586 / 600 | 97.7% |
| W3 | 556 / 600 | 92.7% |
| W2.5，$\lambda=0.5$ | 423 / 600 | 70.5% |
| W2 | 210 / 600 | 35.0% |
| W1.5* | 153 / 600 | 25.5% |
| 校准后随机 W2.5 | 49 / 600 | 8.2% |

来源见 [quality.json](results/quality.json)、[tasks.csv](results/tasks.csv) 和[案例指南](docs/CASES.md)。AWQ、GPTQ、QVLA 以及 FP16 参考值来自文献，单独记录在 [published_baselines.json](results/published_baselines.json)，不计作本轮新测。

**资源指标使用各自的测量口径。** 性能表中编码后的 W1.5* 为 **2.534 GB 整模型存储、6.10 倍整模型压缩、3.537 GB 推理峰值已分配显存、310.4 ms 单次推理时延**，评测设备为 RTX3090。数据见 [performance.json](results/performance.json)。原论文首图的 **2.78 GB** 来自另一种编码实现，不能与 310.4 ms 组合为同一配置。性能记录保留了严格闭环轨迹配对未通过的状态；权重无损恢复并不保证闭环轨迹完全一致。

## 引用与致谢

当前论文按**未发表稿件**记录，不宣称已经录用或发表。引用信息见 [CITATION.cff](CITATION.cff) 和[英文首页](README.md#citation-and-acknowledgments)。

AGQ 基于 OpenVLA-OFT 与 LIBERO，逐通道校准借鉴 OmniQuant 的裁剪学习和块重构。感谢 AWQ、GPTQ 及[参考项目](docs/REFERENCES.md)中的其他工作。上游代码各自的许可证及署名要求，与本项目尚待决定的许可证分别保留。
