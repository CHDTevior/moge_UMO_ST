# HY273 R14 物理时序监督科研对抗审核

> 审核范围仅限科研正确性：数据语义、表示、条件路径、loss、DDP、训练与评估。
> 本项目为科研环境，不将生产级 SHA、防篡改或供应链问题作为门禁。

## 1. 审核结论

```text
模型：gpt-5.6-sol
reasoning effort：max
Critical：0
High：0
Medium：1（不阻断）
Low：2（不阻断）
最终判定：GO
```

未发现正式 50K 训练前必须修复的科学 blocker。

## 2. 主要结论

### 2.1 新目标的准确归因

新 temporal loss 的 velocity-vector 分量，与基础 loss 中已有的
`clean_joint_velocity` 都采用：

```text
clean HY273 prediction
  -> denormalize
  -> position channels + smooth root 重建 global joints
  -> 相邻帧差分 * 30 FPS
  -> 对 target joint velocity 做 SmoothL1
```

因此它不是全新的物理量，而是对 instruction-bearing Edit 样本的定向强化。
真正新增的是：

- 只作用于 source+instruction 的 Edit 行；
- 只作用于原生等长 source-target pair；
- detached source-target velocity soft importance；
- `beta=0.1 m/s` 的物理 SmoothL1；
- speed magnitude 监督；
- 逐样本归约和全局 active-sample DDP 归一化。

这属于有意的额外监督，不是错误重复计数，单位和维度均正确。

### 2.2 数据、配对增强和时间合同

- 两套数据均为 30 FPS。
- MotionFix train 共 5,387 pair，其中 5,047 等长、340 不等长。
- temporal loss 仅启用等长且至少两帧的 pair。
- MotionFix 不做 source/target 独立 crop。
- source 和 target 各自做 root-origin shift，但共享同一个最终 yaw `phi`；
  两边实际 yaw delta 可以不同。
- valid mask、source-native length、source-present 和单 source slot 均被检查。
- 等长不代表逐帧语义严格对齐，所以 soft importance 只能解释为变化区域代理；
  它不应被宣称为真实 changed-region annotation。

### 2.3 DDP 正确性

设全局 active 样本数为 `N`、world size 为 `W`。每个 rank 的 temporal
分母使用 `N/W`，固定桶梯度同步再除以 `W`，最终严格等价于全局 active
样本均值。

额外构造了一个 rank 无 active 样本的检查：

```text
rank0 temporal loss：0
DDP loss mean：与单进程 reference 一致
DDP gradient：与单进程 reference 一致
```

因此某个 rank 没有有效 Edit 行不会造成梯度偏差或死锁。

### 2.4 target leakage、copy shortcut 与抖动

- source-target importance 在 `no_grad` 中计算并 detach。
- target 只用于构造训练标签权重，不进入模型条件，推理时不需要 target。
- copy source 在真实变化区域会得到更大的 target velocity error，不是捷径。
- vector 与 speed 的共同最优点都是 target velocity；额外 jitter 会被惩罚。
- 该项不直接约束 acceleration/jerk，因此不能预先声称一定改善 jerk。
- contact 不接收该项的直接梯度。

### 2.5 权重标定

```text
0.01 * 0.15 / 2.813 = 0.000533...
final temporal_scale = 0.00053
```

200-step 真更新 smoke：

```text
rank-mean temporal/base output-gradient RMS：15.62%
temporal loss 占比：0.452%
clip rate：0
loss：0.00251 -> 0.00227
throughput：174.3 samples/s
active instruction Edit：75.7%
equal-length usable：93.9%
```

该标定是 output-gradient RMS 的 rank 均值，不是全局参数梯度范数比；这只限制
指标解释，不影响反向传播或启动判断。

### 2.6 单变量对照

相对 Positive-only 450K：

- Parent checkpoint 相同；
- T2M/Control/Edit 比例相同，均为 30/40/30；
- Edit CFG 分支相同，均为 75/10/5/5/5；
- DDP4、每卡 32、global batch 128 相同；
- optimizer、LR、seed、normalizer、contact、T2M、Control 和 sampler 未改；
- 唯一有效 treatment 差异是 `temporal_scale: 0 -> 0.00053`。

前 200 step 的 plan trace、control modes 和 Edit branch pattern 与
Positive-only 对照一致。

## 3. 验证记录

```text
完整相关回归：88 passed
审核中复跑子集：68 passed, 1 deselected
DDP zero-active-rank 数学/梯度检查：passed
DDP4 no-update 200-step 标定：passed
DDP4 real-update 200-step smoke：passed
```

## 4. 正式实验

```text
Parent：400K
训练终点：450K
增量：50K
GPU：4
global batch：128
temporal_scale：0.00053
对照：现有 Positive-only 450K
```

正式评估必须同时检查：

- Edit assignment 与 correct-vs-empty；
- faster/slower/repeat/timing 动态子集；
- 已知失败样本 `pair_000038`；
- T2M 物理质量；
- Kimodo-like control benchmark；
- contact Brier/F1、foot skate 和 contact-foot velocity。

不能用单条 GIF 作为实验结论。
