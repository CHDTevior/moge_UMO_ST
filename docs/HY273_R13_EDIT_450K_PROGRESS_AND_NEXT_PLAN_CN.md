# HY273 R13 Motion Editing 450K 实验进度与下一步计划

更新时间：2026-07-23

## 1. 本文目的

本文记录从统一模型 Parent 400K 出发进行的两组 Motion Editing 续训实验，
说明这轮具体改了什么、获得了什么有效结论、哪些问题仍未解决，以及下一轮
物理时序监督实验如何保持单变量比较。

这轮实验研究的是模型训练与推理条件分解，不涉及 Demo 的
MotionCorrection 或任何采样后优化。

## 2. 实验起点

共同起点是：

```text
outputs/hy273_multitask/
  hy273_r13_contactflow_controlled_staged_ddp8_20260720_040507/
  model/step_00400000.pt
```

该 Parent 400K 已经具备：

- Text-to-Motion；
- Kimodo-like root/path/end-effector/full-pose/contact control；
- 初步的 source motion + relative instruction 编辑能力；
- 统一 HY273 clean-x0 输出和 unified 273D contact flow；
- 共享 root/body backbone、HYText 条件和 source context。

Parent 的主要问题是 Motion Editing 对 instruction 的依赖较弱。给定同一个
source 时，正确 instruction、空 instruction 或其他 instruction 的输出差异不足，
模型容易依赖高带宽 source context 并复制 source。

## 3. 这轮具体修改了什么

### 3.1 固定联合任务比例

400K 到 450K 的两组候选都使用相同的高层任务比例：

```text
T2M     30%
Control 40%
Edit    30%
```

这样 Edit 获得足够训练量，同时持续 replay T2M 和 Control，避免为了提升
Editing 而遗忘原有能力。

### 3.2 为 Edit 引入分解式 CFG 训练分支

Edit 样本内部使用：

```text
source + text             75%
source identity           10%
text only                  5%
unconditional              5%
source + text + control     5%
```

这些分支让推理时可以分别控制 source 和 relative instruction：

```text
x_guided
  = x_uncond
  + g_source * (x_source - x_uncond)
  + g_edit   * (x_source_text - x_source)
```

其中：

- `g_source` 控制 source motion 的保持强度；
- `g_edit` 控制 instruction 在已有 source 条件之上的增量；
- T2M 和 Kimodo-like Control 的原有推理接口不变。

这比把文本和 source 混成一个不可分离 CFG 条件更适合 Motion Editing。

### 3.3 Positive-only 候选

Positive-only 保留统一模型原有主 loss，并只增加正确 instruction 分支上的正向
clean-x0 监督：

```text
L_positive
  = 0.05 * L_edit_target_x0
  + 0.02 * L_edit_hard_x0
```

其中：

- `L_edit_target_x0` 是按 HY273 语义组平衡的 clean-x0 SmoothL1；
- `L_edit_hard_x0` 强调当前预测误差较大的部分；
- 不使用 shuffled instruction；
- 不使用 ranking/contrastive negative；
- 不通过故意推坏负分支来获得 instruction gap。

对应 checkpoint：

```text
outputs/hy273_multitask/
  hy273_r13_decompcfg_no_rank_positive_only_ddp4_400k_to450k_
  20260723_motionfix_decompcfg/model/step_00450000.pt
```

### 3.4 Changed-region 候选

Changed-region 候选包含 Positive-only 的全部设置，并额外在 same-source
MotionFix pair 上构造 source-target discrepancy 区域：

- root 和每个 body joint 独立选取差异最大的 20% 时间位置；
- velocity mask 按有限差分边界扩展；
- 在这些坐标上增加 `0.05 * clean-x0 SmoothL1`；
- 只在具有 same-source sibling 的样本上启用。

对应 checkpoint：

```text
outputs/hy273_multitask/
  hy273_r13_decompcfg_same_source_changed_positive_only_ddp4_400k_to450k_
  20260723_motionfix_decompcfg/model/step_00450000.pt
```

这个 changed-region 目标仍然是 normalized clean-x0 重建，并不直接监督物理
joint speed、节奏或动作频率。

## 4. 训练和评估协议

- 两组都从相同 Parent 400K 开始；
- 分别使用 4 卡 DDP；
- global batch size 为 128；
- 完整续训 50K step，到 450K；
- 两组使用相同数据调度、seed、优化器阶段和学习率；
- 主要评估使用 raw model、ODE32 和相同 seed/noise；
- Control 使用 `text_cfg=2.0`、`control_cfg=2.0`；
- Control 全量评估包含 8,084 个 case。

完整评估报告：

```text
outputs/hy273_multitask/diagnostics/
  r13_changed_positive_ab_450k_20260723/EVALUATION_REPORT_CN.md
```

## 5. 实验结果

### 5.1 Edit instruction sensitivity

目标不重叠子集上的 ODE32：

| 模型 | Correct MSE | Correct-vs-empty gap | Correct assignment |
|---|---:|---:|---:|
| Parent 400K | 1.0581 | -0.0277 | 62.1% |
| Positive-only 450K | 1.0197 | +0.1252 | 75.9% |
| Changed-region 450K | 1.0532 | +0.0690 | 69.0% |

关键结论：

1. Positive-only 将 correct assignment 从 62.1% 提升到 75.9%。
2. Parent 中空 instruction 比正确 instruction 更有利；Positive-only 中这个关系
   在完整 ODE 采样后反转，说明模型开始真正使用 relative instruction。
3. Changed-region 也比 Parent 更依赖 instruction，但明显弱于 Positive-only。
4. 当前 changed-region 定义没有提供额外收益，不应按原配置继续增加训练量。

Positive-only 在 fixed-t `t=0` 上仍有 `correct MSE - empty MSE = +0.0155`，
因此条件改进主要体现在完整 ODE 轨迹，clean endpoint map 尚未完全稳定。

### 5.2 T2M 保持性

| 模型 | FK jerk (m/s^3) | Contact consistency | Foot-skate ratio |
|---|---:|---:|---:|
| Parent 400K | 60.675 | 0.9453 | 0.3229 |
| Positive-only 450K | 49.702 | 0.9475 | 0.3028 |
| Changed-region 450K | 50.065 | 0.9630 | 0.2996 |

两组 450K 模型均未出现 T2M 遗忘，这说明固定 30/40/30 replay 策略有效。

### 5.3 Kimodo-like Control 保持性

以下几何控制指标都保持在 Parent 的 5% 非退化范围内：

- end-effector position；
- end-effector rotation；
- full-body keyframe；
- root 2D position；
- root 2D adherence。

无文本 Control 全部通过。with-text Control 的唯一门禁失败项是 controlled
contact Brier：

| 模型 | Contact Brier | Accuracy | F1 |
|---|---:|---:|---:|
| Parent 400K | 0.002374 | 0.997626 | 0.997369 |
| Positive-only 450K | 0.004148 | 0.995852 | 0.994321 |
| Changed-region 450K | 0.004028 | 0.995972 | 0.994783 |

Brier 的相对变化较大，但绝对误差仍小，几何控制、Accuracy 和 F1 没有整体
崩溃。下一轮应同时观察 Brier、F1、contact-foot velocity 和 foot skate，
不能只用接近零的 Brier 相对百分比做科研结论。

### 5.4 动态编辑仍然失败

专项 pair：

```text
instruction = "move feet faster and punch once faster"
```

| 动作 | 平均脚速 (m/s) | 强脚步峰频率 (Hz) | 腕部峰值速度 (m/s) |
|---|---:|---:|---:|
| Source | 0.585 | 1.75 | 5.51 |
| Target | 1.557 | 3.00 | 6.37 |
| Parent 400K | 1.013 | 1.50 | 4.15 |
| Positive-only 450K | 0.192 | 1.50 | 5.34 |
| Changed-region 450K | 0.210 | 1.50 | 5.65 |

两组新模型对 punch 有部分响应，但没有实现 feet faster。Positive-only 将
`edit_cfg` 从 2.0 提高到 3.0 后，脚速进一步降至 0.122 m/s，说明 guidance
放大的是已经学到的静态/平均姿态方向，而不是正确的 temporal edit direction。

因此继续提高 CFG 不是解决方案。

## 6. 本轮可以支持的科研结论

本轮支持：

1. 当前统一 backbone 并非完全无法读取 Edit instruction。
2. 分解式 CFG 所需的多条件训练分支是有效且必要的。
3. 无负样本的正向 clean-x0 强化可以显著提高 instruction sensitivity。
4. 30/40/30 联合 replay 可以在提升 Edit 的同时保留 T2M 和几何 Control。
5. 当前主要瓶颈已经从“文本通路是否工作”转为“速度、节奏和时序语义是否被
   正确监督”。

本轮不支持：

1. Positive-only 已经解决完整 Motion Editing；
2. 更高 Edit CFG 可以解决动态编辑；
3. 当前 changed-region normalized-x0 loss 能提升动态编辑；
4. 任一 450K 候选可以直接替代 Parent 400K 成为最终统一模型。

当前应保留：

- Parent 400K：原始 T2M/Control 能力基线；
- Positive-only 450K：当前 Edit instruction sensitivity 最好的研究候选；
- Changed-region 450K：负结果和消融记录，不继续续训。

## 7. 数据侧补充观察

MotionFix train 共 5,387 个 pair：

- 5,047 个 source/target 等长；
- 340 个不等长；
- 955 条 instruction 含显式 speed/pace 相关措辞。

速度文本并非极少数。但使用全身平均 joint speed 做粗粒度检查时，
`faster` 和 `slower` 分别只有约 70% 和 68% 的 pair 呈现预期全身速度方向。
原因包括：

- 指令只要求某个肢体更快；
- 同一句指令包含姿态、幅度和方向等多个编辑；
- 局部动作变快不一定提高全身平均速度；
- 频率变化和瞬时速度变化不是同一个物理量。

因此下一轮不能根据关键词给全身速度加伪标签，必须从每个 source-target pair
中提取逐关节、逐时间的物理监督。

## 8. 下一轮：物理时序监督单变量实验

### 8.1 不改变的内容

下一轮仍从 Parent 400K 开始，并保持：

- 共享 root/body backbone；
- HYText encoder/cache；
- clean-x0 model output；
- unified 273D flow/contact；
- 30/40/30 高层任务比例；
- Edit 内部 75/10/5/5/5 分解式 CFG 分支；
- Positive-only 的 `0.05 target-x0 + 0.02 hard-x0`；
- batch size、seed、优化器和学习率；
- 不使用 ranking、negative branch 或原 changed-region loss。

### 8.2 唯一新增项

新增只作用于 MotionFix Edit 的物理时序目标：

```text
L_total
  = L_unified_flow
  + 0.05 * L_edit_target_x0
  + 0.02 * L_edit_hard_x0
  + lambda_temporal * L_edit_temporal
```

`L_edit_temporal` 的候选定义：

1. 将预测 clean HY273 反归一化到物理空间；
2. 由 HY273 position channels 与 smooth root 重建 global joint positions；
3. 以 30 FPS 计算相邻帧 joint velocity；
4. 同时监督 velocity vector 和 speed magnitude；
5. 对等长 source-target pair，使用逐关节
   `target velocity - source velocity` 构造 stop-gradient soft importance；
6. 保留低权重背景监督，不使用硬 changed mask；
7. 不直接对完整 273D 做时间插值；
8. contact 继续使用现有 Kimodo 比例，不混入 temporal loss。

这里需要准确区分：新目标的 velocity-vector 分量与基础 loss 中已有的
`clean_joint_velocity` 使用同一物理残差；新增信息来自 Edit-only
作用域、source-target soft importance、较小的物理 SmoothL1 beta，以及
speed-magnitude 分量。因此它是“面向编辑任务的定向物理时序增强”，不是
首次向模型加入 joint-velocity 监督。soft importance 也只应解释为变化区域
的代理权重，因为 MotionFix 等长 pair 并不保证逐帧相位严格对齐。

Parent 400K 上的 200-step、4 卡、no-update 标定结果为：

```text
provisional temporal_scale:             0.01
temporal/base rank-mean output-gradient RMS: 2.813
scale for 15% gradient ratio:            0.000533
final temporal_scale:                    0.00053
equal-length usable fraction:            93.9%
active instruction-bearing Edit rows:    75.7%
mean source-target velocity delta:        0.351 m/s
mean source-target speed delta:           0.217 m/s
```

因此正式训练固定使用 `lambda_temporal=0.00053`，其预期 rank-mean
weighted output-gradient RMS 约为当前 Edit 主目标的 15%，不在训练中动态改变。

### 8.3 训练安排

```text
start: Parent 400K
stop:  450K
steps: 50K
DDP:   4 cards
global batch: 128
```

现有 Positive-only 450K 已经是无 temporal loss 的匹配对照，不需要重复训练。
新候选只增加 temporal loss，保持单变量比较。

### 8.4 成功标准

主要成功标准：

1. held-out faster/slower/repeat/timing 子集的逐关节速度、speed magnitude 和
   频率指标显著向 target 靠近；
2. `pair_000038` 的脚速与步频方向改善，而不只是静态 position MSE 下降；
3. ODE32 correct assignment 不应明显低于 Positive-only 的 75.9%；
4. correct-vs-empty gap 保持为正；
5. T2M 和几何 Control 保持；
6. contact Brier、F1、contact-foot velocity 和 foot skate 不出现实质退化。

如果足量 50K 训练后动态编辑仍无显著改善，下一步才考虑结构升级：

- 将当前按 target 时间对齐后直接加到 motion token 的 source context，
  改为独立 source token block；
- 通过 separator/self-attention 或显式 source-text fusion 读取 source；
- 保持统一模型和分解式 CFG，不引入独立 task adapter。

在 temporal supervision 结果出来前，不先更换 backbone。
