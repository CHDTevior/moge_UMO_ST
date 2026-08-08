# HY273 Reaction-v5.1 正交 Control 300K-400K 实验计划

## 1. 目标与 parent

本实验从 Reaction-v5.1 当前最佳的 300K checkpoint 连续训练 100K：

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_1_full_contact/
  hy273_unified_reaction_v5_1_full_contact_20260806_1750/model/step_00300000.pt
```

恢复内容包括 model、EMA、optimizer、任务 scheduler、三个数据流 cursor、sample ordinal 和 normalizer。模型结构、K273 表示、共享 mean/std、文本/source 条件通路和 Reaction-v5.1 loss 全部不变。

## 2. 二维训练分布

任务轴保持原比例：

| 任务 | 300K-400K update 比例 | 100K 中的 updates |
|---|---:|---:|
| T2M | 30% | 30,000 |
| Edit | 35% | 35,000 |
| Reaction | 35% | 35,000 |

Control 是与任务轴正交的第二个采样轴。每个任务内部都是：

```text
80% control-present + 20% control-absent
```

按实际 global batch 计算，完整 100K 的确定性曝光为：

| 任务 | 总 actor samples | Control | No-control |
|---|---:|---:|---:|
| T2M | 3,840,000 | 3,072,000 | 768,000 |
| Edit | 2,240,000 | 1,792,000 | 448,000 |
| Reaction | 2,240,000 | 1,792,000 | 448,000 |

比例由每个 task stream 的 global sample ordinal 确定，不依赖 batch 排序或 rank；完整阶段严格为 80/20。原有任务内条件分布保持不变：T2M 文本 dropout、Edit CFG 分支和 Reaction source/text 条件 pattern 不因 Control 改写。

## 3. Control 的目标语义

- T2M + Control：从要生成的 GT motion 合成 control。
- Edit + Control：从当前 CFG 分支的有效监督 target 合成 control。
  `SOURCE_TEXT`/`TEXT_ONLY`/`UNCONDITIONAL` 分支的有效 target 是
  MotionFix GT target；`SOURCE_IDENTITY` 是明确的 no-edit CFG 分支，其有效
  target 按既有合同就是 source，因此该分支的 control 也从 identity
  target 构造。不允许从 source 构造 control 却用 MotionFix GT target 做
  flow 监督，或反过来，因为两者会直接矛盾。
- Reaction + Control：从要生成的 reactor target 合成 control；另一个人的 source motion 仍是独立条件。

因此 Control 不新增 task id，也不把 Edit/Reaction 改成新任务。模型仍通过原
task id 区分 T2M、Edit、Reaction，并通过 546D 输入中的 motion mask
感知同一任务内是否带 Control。`capability_id` 只用于数据/推理路由和
合同校验，不是额外的模型 embedding。

## 4. Tensor flow

对一个带控制样本，先从有效 target 帧编译 `observed_motion [B,T,273]` 和 `motion_mask [B,T,273]`：

```text
x0_norm [B,T,273]
epsilon [B,T,273]
t [B]

z_t = t * x0_norm + (1-t) * epsilon
z_imputed = where(motion_mask, observed_norm, z_t)
model_input = concat(z_imputed, motion_mask.float())  # [B,T,546]
```

模型输出仍为 `x0_hat [B,T,273]`。No-control 样本的 observed 和 mask 全零，因此沿用原始 T2M/Edit/Reaction 路径。模型 shape 和参数量不变。

## 5. 控制类型与 curriculum

Control-present 样本在以下 Kimodo-like pattern 中采样：

- sparse root/path，可随机只给 XZ 或同时给 heading；
- dense root/path；
- 手脚末端 position + global rotation，并带 root reference；
- sparse full-pose；
- sparse foot-contact pattern；
- 两种 pattern 的 mixed control。

其中 mixed 概率为 25%。稀疏 keyframe 上限在 300K-400K 内从 1 逐步增加到 20；dense root 覆盖完整有效序列。这里没有 Ease、淡入淡出或额外 correction/refinement。

## 6. Loss

原始共享 loss、Edit loss 和 Reaction-v5.1 全序列 FK contact loss 均保持不变。Control 只改变被 observation mask 覆盖的维度：

- 未控制的连续维度继续使用原 `velocity-MSE`。
- 被控制的连续维度从 velocity-MSE 中排除，改用 clean-x0 Smooth-L1，权重 `0.25`。
- 被控制的 contact channel 使用 clean-x0 Smooth-L1，权重 `0.02857142857142857`。

该设计避免同一 hard-observed channel 同时承受 velocity target 与 clean overwrite target。Control loss 是按有效 masked element 做 ratio-of-sums，不会因 full-pose 的 channel 数更多而仅靠元素总数放大。

首个三任务 8 卡 smoke 的结果：

```text
loss/total = 0.02097
grad/preclip_norm = 0.521
max_memory = 10.16 GiB
NaN/Inf = 0
```

Reaction 的初始 control-continuous raw loss 高于 T2M/Edit，因此正式训练重点观察其前 1K-5K 是否正常回落；不通过降低 Control 比例或引入 Ease 来掩盖问题。

## 7. 保存与评估

- 继续每 10K 更新 `latest.pt`。
- 每 50K 保存正式 checkpoint，因此保留 350K 和 400K。
- 训练不中途改变比例、loss 或 Control pattern。
- 400K 后对 T2M/Edit/Reaction 三任务运行相同 Control gate，逐 subtype 比较 `controlled` 与 `control-zero`。
- 同时重复 no-control 的 T2M、Edit、Reaction 守门，确认旧能力没有被 80% Control 覆盖洗掉。

只有 Control 三任务有效且 no-control 能力可接受，才把 400K 作为进入 HOI 设计的 parent。
