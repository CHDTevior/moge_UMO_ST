# HY273 K-Encoder Stage-BC Ease/Control/Edit 启动记录

日期：2026-07-28

## 1. 当前训练阶段

已完成：

| 阶段 | Step | T2M / Control / Edit | 目的 |
|---|---:|---:|---|
| Stage-A | 0K -> 200K | 100 / 0 / 0 | 训练 K-Encoder T2M 基模 |
| Stage-BE | 200K -> 250K | 60 / 0 / 40 | 在保留 T2M 的同时建立 Motion Editing 能力 |

当前运行：

| 阶段 | Step | T2M / Control / Edit | 目的 |
|---|---:|---:|---|
| Stage-BC | 250K -> 400K | 10 / 70 / 20 | Control bootstrap、Ease 学习、Edit replay |

Stage-BC 不把 Edit 关掉。20% Edit 用于保持 Stage-BE 已获得的编辑能力；70%
Control 提供接近 Kimodo Stage-2 的控制训练剂量；10% T2M 用于抑制基础生成能力遗忘。

## 2. Ease 条件

物理标签为：

```text
ease_physical = [E_in_xyz, E_out_xyz]  # [B, 6], 单位：米
```

它从增强后的物理 K273 动作重建全局关节质心轨迹，并分别计算前半段和后半段相对端点严格线性轨迹的平均残差。

模型使用独立 Ease stats，将 6D 标签通过零初始化的 `6 -> H -> H` MLP 转成同一个 additive bias，分别加到 root/body target hidden：

```text
root_hidden += ease_bias
body_hidden += ease_bias
```

Ease 不进入文本 cross-attention 或 AdaLN，不注入 source token，也不改变现有 loss。首版不增加 Ease auxiliary loss，以便只检验 Ease 条件本身是否可学。

组合覆盖率：

```text
T2M:     Ease present 25%
Control: Ease present 50%
Edit:    Ease present 0%
```

按 10/70/20 总任务比例，Ease 的总体期望覆盖率为 37.5%。Edit 在条件合同、dataset 和采样入口三层都禁止携带 Ease。

## 3. 训练目标

主任务保持统一 273D clean-x0 prediction，训练损失在 velocity-MSE 空间计算，不改成 noise-pred 或 v-pred。

基础 reconstruction 权重：

```text
representation_scale = 0.09397019716051493
semantic channel weights = [10, 2, 10, 10, 3]
contact = 0.010739451104058849
clean_root_velocity = 0.01
clean_joint_velocity = 0.01
foot_lock = 0.01
fk_consistency = 0.07
```

Control 附加项：

```text
control_continuous = 0.25
control_contact = 0.02857142857142857
```

Edit 保留 Stage-BE 的 positive-only objective：

```text
target_x0_scale = 0.05
hard_x0_scale = 0.02
hard_fraction = 0.20
instruction_rank_scale = 0.0
```

Edit CFG 训练分支为：

```text
source + instruction  80%
source identity        10%
instruction only       5%
unconditional          5%
```

## 4. 审核与恢复验证

`gpt-5.6-sol max` 科研审核结论：

```text
GO（无 blocker）
```

完整审核：

```text
docs/HY273_EASE_GPT56_REVIEW.md
```

审核后补充了两项科学语义约束：

1. `ConditionBatch` 在合同层拒绝 Edit + Ease。
2. checkpoint 显式记录并恢复核对 Ease 的 source manifest、统计口径和 Mean/Std。

真实 8 卡恢复实验比较了：

```text
连续 10 step
vs.
5 step -> save -> reload -> 5 step
```

以下状态全部逐值完全一致：

```text
model
EMA
Adam state
batcher/scheduler
RNG
next_global_step
context_update_count
ease_update_count
ema_update_count
Ease stats identity
```

## 5. 正式运行

父 checkpoint：

```text
/mnt/afs/mogeflow-control/outputs/hy273_text_fusion/
hy273_kencoder_stageBE_t2m60_edit40_ddp8x16_20260728_131901/
model/step_00250000.pt
```

正式 run：

```text
/mnt/afs/mogeflow-control/outputs/hy273_text_fusion/
hy273_kencoder_stageBC_ease_t2m10_ctrl70_edit20_ddp8x16_20260728_201555
```

运行会话与日志：

```text
tmux: hy273_stagebc_ease
log:
/mnt/afs/mogeflow-control/run_logs/
hy273_kencoder_stageBC_ease_t2m10_ctrl70_edit20_ddp8x16_20260728_201555.log
```

checkpoint 规则：

```text
latest.pt: 每 10K step
归档 checkpoint: 每 50K step
阶段终点: 400K
```

## 6. 启动观测

在 `250180`：

```text
T2M / Control / Edit = 9.34% / 70.66% / 20.00%
Ease present = 37.38%
overall loss = 0.00443
T2M loss = 0.00376
Control loss = 0.00512
Edit loss = 0.00178
total grad = 0.0633
Ease grad = 0.0143
throughput = 415 samples/s
step time = 0.310 s
peak allocated memory = 14.27 GiB/GPU
```

当前无 NaN、OOM、cache miss、DDP rank 分叉或恢复语义异常。

按当前有效吞吐估计，剩余 150K step 约需 13 小时，另加 checkpoint 写盘和短时吞吐波动。

## 7. 后续评估

训练中按小时检查 loss、任务比例、Ease 覆盖、Ease/total gradient、显存、吞吐和进程存活。

在中间检查点和 400K 终点执行：

1. T2M 固定文本/固定 seed 非退化可视化。
2. MotionFix Edit 固定样例可视化，重点复查速度类指令。
3. Kimodo-like root/path/endpoint/full-pose/contact benchmark。
4. 固定 text/noise/control，仅改变 `E_in/E_out` 的 Ease 因果 sweep。
5. Ease normalized MAE、Spearman 单调性，以及 hard-control error 非退化检查。
