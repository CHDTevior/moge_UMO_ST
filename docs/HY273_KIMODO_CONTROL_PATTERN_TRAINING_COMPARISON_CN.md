# HY273 与 Kimodo 控制 Pattern 训练对照

记录日期：2026-07-24

## 1. 目的

本文记录 Kimodo 和当前 HY273 unified model 在控制 pattern、采样分布及训练阶段上的差异，供后续控制能力实验、训练配比调整和结果解释使用。

需要特别区分：

- “支持哪些控制 pattern”描述能力集合。
- “训练时以什么概率看到这些 pattern”决定模型偏向。
- Kimodo 技术报告没有公开每一种 pattern 的精确采样权重和关键帧位置 sampler，不能把我们的具体实现反推成 Kimodo 官方实现。

## 2. Kimodo 公开训练策略

Kimodo 使用两个训练阶段：

- Phase 1，前 500K step：100% text-to-motion，不输入运动学约束。
- Phase 2，后 500K step：90% 样本带运动学约束，10% 样本不带约束。
- 两个阶段都独立执行 10% text dropout。
- Phase 2 中有 25% 的情况混合两个控制 pattern。
- sparse constraint 的最大关键帧数在 Phase 2 中从 1 线性增加到 20，并偏向采样较少的关键帧。

因此，“Phase 2 有 90% Control”不等于这些样本没有文本。若约束存在和 text dropout 独立，Phase 2 可以近似理解为：

| 条件组合 | 比例 |
|---|---:|
| text + control | 81% |
| control-only | 9% |
| text-only | 9% |
| unconditional | 1% |

Kimodo 的标准 T2M 能力主要来自：

1. Phase 1 的 500K pure T2M 预训练。
2. Phase 2 中 10% 的 no-control replay。
3. 大多数 Control 样本仍带文本，并以同一 clean motion 为监督目标。

## 3. 控制 Pattern 对照

| 能力 | Kimodo 公开描述 | 当前 HY273 实现 | 结论 |
|---|---|---|---|
| Sparse full-body | 稀疏关键帧的全身 joint positions | `fullpose`：root、heading 和 22 个 joint positions | 基本一致；两边实际输入都不控制全身 rotations |
| End-effector | 随机手脚子集的位置和 rotation | 左脚、右脚、左手腕、右手腕四组随机非空子集；脚的位置包含 ankle+foot，rotation 控 ankle | 语义基本对齐 |
| Sparse root | 稀疏 2D root position + heading | `root_sparse`；50% XZ-only，50% XZ+heading | 我们额外训练了 heading-absent 能力 |
| Dense root path | 稠密 2D root position + heading path | `root_dense`；覆盖完整有效序列，50% XZ-only，50% XZ+heading | 概念一致；我们的覆盖范围和 heading dropout 更明确 |
| Sparse contact | 稀疏 foot-contact configuration | 稀疏关键帧的四路 binary contact | 基本一致 |
| Pattern composition | 25% 混合两个 pattern | 每个 Control 样本 25% 混合两个不同 pattern | 原理一致；Kimodo 没公开 25% 的精确分母 |

当前 HY273 单 pattern 在五种类型中均匀抽取；mixed 使用无放回方式抽取两个不同 pattern。Kimodo 没有公开各 pattern 的精确权重，因此不能确认其也是严格均匀分布。

## 4. 关键帧采样差异

当前 HY273：

- sparse 最大关键帧数随 curriculum 从 1 增加到 20。
- 数量使用 `1 + floor(U^2 * K)` 形式，偏向较少关键帧。
- 关键帧位置先近似均匀铺开，再加入 `[-1, 1]` 帧 jitter。
- 当前 400K 之后的训练已经达到最大值 20。
- dense root path 的 `dense_min_fraction=1.0`，即覆盖整个有效动作长度。

Kimodo 公开了“1 到 20 的线性 curriculum”和“偏向较少关键帧”，但没有公开关键帧位置、单 pattern 权重以及 dense path 的精确 sampler。

## 5. 我们的训练阶段

当前 400K parent 的历史训练分布：

| Step | T2M | Control | Edit |
|---|---:|---:|---:|
| 0–200K | 100% | 0% | 0% |
| 200–250K | 10% | 90% | 0% |
| 250–260K | 从 10% 增至 18% | 从 90% 降至 80% | 从 0% 增至 2% |
| 260–400K | 18% | 80% | 2% |

当前 R15 400K–450K：

| Task | 全局比例 |
|---|---:|
| T2M | 30% |
| Pure Control | 40% |
| Edit | 30% |

R15 的 Edit 中有 5% 为 Edit+Control，因此全局 control-bearing 样本约为：

```text
40% + 30% * 5% = 41.5%
```

每个 pure Control 或 Edit+Control 样本内部：

- 75%：一个控制 pattern。
- 25%：两个不同控制 pattern。
- `none_prob=0`；无控制能力由独立 T2M/Edit 分支提供。

当前 R15 additive 与 source-token-block A/B 没有修改控制 pattern、控制采样器或控制 loss。两组只比较 Edit source conditioning 的注入方式。

## 6. Loss 差异

Kimodo 技术报告给出的目标是按运动表示分组的 clean-x0 Smooth L1 与 FK loss，没有公开额外的 control-mask 专用 loss。

当前 HY273 除主表示重建、contact、velocity、foot-lock 和 FK loss 外，还包含：

```yaml
control_continuous: 0.25
control_contact: 0.05
```

它们在受控位置提供额外监督。因此即使 Control 样本比例低于 Kimodo Phase 2，单个 HY273 Control 样本对受控区域的监督也更强。

## 7. 后续实验解释

后续比较控制能力时，应重点记录：

1. 总 Control exposure，而不只看模型是否支持相同 pattern。
2. 单 pattern 与双 pattern 的分别结果。
3. XZ-only 和 XZ+heading 分开报告，不能合并后与 Kimodo 的 heading 条件直接比较。
4. 当前 R15 后续若 Control 退化，优先判断是否由 400K 后 Control replay 从 80% 降至约 41.5% 引起。
5. 若要做最接近 Kimodo Phase 2 的复现实验，应保持 90% control-bearing、10% no-control、10% 独立 text dropout，并单独处理 Edit，不应把当前 30/40/30 多任务分布称为 Kimodo Phase 2。

## 8. 对应材料

- Kimodo 技术报告：`outside_doc/kimodo_tech_report.pdf`，第 10–11 页。
- 当前控制 sampler：`models/raw_motion/hy273_constraints.py`。
- 当前多任务 scheduler：`data/hy273_multitask_scheduler.py`。
- 当前 SamplePlan：`data/hy273_multitask_manifest_dataset.py`。
- 当前基础控制配置：`configs/hy273_multitask_base.yaml`。
- 当前 R15 配置：`configs/hy273_multitask_r13_stage_c1_decomposed_cfg_edit.yaml`。
- Kimodo 公开 constraint compiler：`external_repos/kimodo/kimodo/constraints.py`。
