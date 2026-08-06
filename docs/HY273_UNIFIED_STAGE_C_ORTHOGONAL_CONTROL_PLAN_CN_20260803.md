# HY273 Unified Stage C 正交控制训练方案

日期：2026-08-03

## 1. 本轮目标

从已完成的 Unified Full-Text Reaction `350K` checkpoint 继续训练，不改变 backbone、任务定义、文本编码器或统一 normalization，只增加与任务正交的 Kimodo 风格控制能力。

基础 checkpoint：

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction/
hy273_unified_fulltext_reaction_v1_20260801_0315/model/step_00350000.pt
```

任务轴和控制轴分开采样：

```text
Task:     T2M 30% / Edit 35% / Reaction 35%
Control:  present 90% / absent 10%，在每个 Task 内分别成立
Ease:     0%，本轮完全关闭
```

Control 不是第四个任务。`TaskId` 仍是 Generate / Edit / Reaction；模型通过原有的 `[motion_state, motion_mask]` 输入判断是否存在控制，`CapabilityId` 只负责路由与合同校验，不增加模型参数。

## 2. 为什么从 350K 接续

`0–350K` 已经完成：

```text
0–100K:     T2M-only
100–350K:   T2M / Edit / Reaction = 30 / 35 / 35
```

350K 的 Reaction 和 T2M 分布指标最好，Edit 的响应较强但平滑性不如 300K。Stage C 的研究问题是“在完整三任务模型上加入正交控制后，原能力能否保留、三种任务能否都接受同一种控制合同”，因此以完整 350K 状态作为唯一 parent，不回退或从头更换 backbone。

旧 checkpoint 对比见：

```text
docs/HY273_UNIFIED_FULLTEXT_REACTION_200K_350K_FINAL_REPORT_CN_20260803.md
```

## 3. 模型输入和 Tensor Flow

所有任务的目标 actor 都使用同一 K273：

```text
x0 physical:       [B, T, 273]
x0 normalized:     [B, T, 273]
noise:             [B, T, 273]
t:                 [B]
z_t:               t*x0 + (1-t)*noise                   [B, T, 273]
observed target:   从当前任务的有效 target 编译           [B, T, 273]
hard mask:         对应被控制的 channel/frame            [B, T, 273] bool
z_imputed:         (1-mask)*z_t + mask*observed_norm     [B, T, 273]
model input:       concat(z_imputed, mask)               [B, T, 546]
model output:      clean x0 prediction                   [B, T, 273]
```

各任务的条件来源：

```text
T2M:
  source absent + absolute text + optional target control -> generated motion

Edit:
  source motion + relative instruction + optional target control -> edited target

Reaction:
  observed actor + interaction text + optional reactor-target control -> reactor
```

控制值必须从有效监督目标构造：

- T2M：从 HumanML3D target 构造。
- Edit `SOURCE_TEXT`：从 MotionFix target 构造。
- Edit `SOURCE_IDENTITY`：该 CFG 分支的有效 target 是 source，因此从 source 构造。
- Reaction：始终从 reactor target 构造，禁止从 observed actor/source 构造。

Edit 的 source/target 继续共享 yaw gauge；Reaction 的 source/target 继续共享 source-root centering 和 yaw gauge。控制在这些配对变换完成后构造，因此不会破坏相对轨迹语义。

## 4. 控制模式分布

Control-present 样本内部：

```text
mixed:                         25%
其余单一模式合计:              75%
单一模式候选:
  root_sparse
  root_dense
  endpoints
  fullpose
  contact
```

具体合同：

- `root_sparse`：1 到 20 个低数量偏置关键帧，随 350K→500K 逐步扩展上限。
- `root_dense`：完整有效时间区间的 root XZ；其中 50% 同时给 heading，50% 只给 XZ。
- `endpoints`：Kimodo 手脚末端组，随机非空子集，包含位置和 global 6D rotation，并带 root reference。
- `fullpose`：稀疏 root + 22 关节位置关键帧。
- `contact`：稀疏四路 foot-contact 配置。
- `mixed`：从上述模式随机取两个组合。

Ease 不参与数据采样、ConditionBatch、模型 forward 或 loss。

## 5. Loss 设计

模型仍然预测 normalized clean `x0`，基础监督保持 velocity-equivalent MSE：

```text
v_hat    = (x0_hat - z_imputed) / max(1-t, 0.05)
v_target = (x0     - z_imputed) / max(1-t, 0.05)
L_repr   = MSE(v_hat, v_target)，只在未控制 channel 上计算
```

K273 continuous semantic group：

| Group | 内部权重 |
|---|---:|
| root xyz | 10 |
| heading | 2 |
| joint positions | 10 |
| global rot6d | 10 |
| joint velocities | 3 |

上述组先按 `10/2/10/10/3` 归一，再整体乘 `0.09397019716051493`。contact 的基础权重为 `0.010739451104058849`。继续保留：

```text
clean root velocity:   0.01
clean joint velocity:  0.01
foot lock:             0.01
FK consistency:        0.07
```

控制 channel 不再重复进入基础 representation/contact loss，而使用 Kimodo 比例的 clean-state adherence：

```text
L_control_cont = SmoothL1(x0_hat, observed_norm) on continuous hard mask
weight = 0.25

L_control_contact = SmoothL1(contact_hat, observed_contact_norm) on contact mask
weight = 0.25 * 4 / 35 = 0.02857142857142857
```

这样保持 Kimodo 的 continuous/control 主权重 `0.25`，并按 contact semantic weight `4` 相对 continuous 总权重 `35` 缩放 contact。控制监督对 T2M、Edit、Reaction 使用同一组系数。

任务专属辅助 loss 保持不变：

```text
Edit:
  target x0 auxiliary       0.05
  top-20% hard-region x0    0.02
  instruction ranking       0.0

Reaction:
  relative root             0.02
  relative heading          0.01
  joint distance            0.01
  close-joint vector        0.01
```

Edit 的 `hard-region` 指未控制区域中误差最高的 20%，不是 hard control mask，因此不会重复奖励被 overwrite 的控制 channel。Reaction relation loss 同时约束生成 reactor 与 observed actor 的空间关系，避免单纯追控制点而破坏交互。

## 6. CFG 推理

默认 scale 均为 `2.0`。

T2M + control 保持 Kimodo 官方 separated CFG：

```text
E = empty text + no control
T = text + no control
C = empty text + control

pred = E + g_text*(T-E) + g_control*(C-E)
```

Edit + control：

```text
E = no source, empty text, no control
S = source only
J = source + instruction
A = source + instruction + control

pred = E
     + g_source*(S-E)
     + g_edit*(J-S)
     + g_control*(A-J)
```

Reaction + control：

```text
E = no source, empty text, no control
S = observed actor only
J = observed actor + interaction text
A = observed actor + interaction text + reactor control

pred = E
     + g_source*(S-E)
     + g_text*(J-S)
     + g_control*(A-J)
```

Reaction 的四个 scale 全为 1 时严格等于 `A`。ODE state 不做 persistent clamp；每个 denoising step 只在 denoiser 输入上 overwrite，最终同时输出 raw learned adherence 和 exact-clamped diagnostic。

## 7. 训练阶段

```text
Stage C1: 350K -> 400K，50K updates
Stage C2: 400K -> 450K，50K updates
Stage C3: 450K -> 500K，50K updates
```

每段完成后保存 `step_00400000.pt`、`step_00450000.pt`、`step_00500000.pt`。只有门检通过才进入下一段。

继续使用八卡 DDP：

```text
T2M batch/rank:       16
Edit batch/rank:       8
Reaction batch/rank:   8
base LR:               5e-5
adaptation LR:         1e-4
precision:             bf16
gradient clip:         1.0
EMA:                   0.995, every 10 steps
```

任务 scheduler 保持精确 `30/35/35` debt 调度；control 使用每个 stream 自己的 global sample ordinal，按有理数序列精确实现 `9/10`，可在 DDP/resume 后逐样本重放。

## 8. 50K 门检

每个 gate 分两部分。

### 8.1 原能力保留

- T2M：固定 HumanML3D test 协议的 FID、R-Precision、jerk、foot skate 与 GIF。
- Edit：MotionFix target/changed-region/rotation/jerk/contact、counterfactual 与重点动态样例 GIF。
- Reaction：Inter-X fixed-role relation、MPJPE、close-event、contact、jerk、text advantage 与 GIF。

对照 350K parent；若某项轻微波动但控制明显学习，先看 400K→450K 趋势，不因单点噪声停训。若出现 NaN、持续 loss/gradient 发散或三任务系统性崩坏，停止并查真因。

### 8.2 控制能力

固定 held-out case、subtype、文本、source 和初始噪声，配对运行：

```text
controlled:    control_cfg = 2.0
control-zero:  control_cfg = 0.0
```

主结果是 raw learned adherence；exact clamp 只报告诊断下限。覆盖 Kimodo root/path/waypoint/end-effector/full-pose/contact/mixed subtype，并分别在 T2M、Edit、Reaction 上汇总。

400K/450K 快速门检默认每个 subtype 8 条；500K 再跑全 held-out split。启动脚本：

```text
scripts/launch/eval_hy273_unified_stage_c_control_gate_8gpu.sh
```

成功信号：

1. `control_cfg=2` 相对同噪声 `control_cfg=0` 的 root/end-effector/fullpose/contact error 有一致正向改善。
2. Reaction relation、Edit identity/jerk、T2M motion quality 没有因控制显著崩坏。
3. 10% no-control 样本和标准无控制推理维持 350K 的主要能力。

## 9. 启动方式

首段只启动到 400K：

```bash
RUN_NAME=hy273_unified_fulltext_reaction_v1_20260801_0315 \
STOP_STEP=400000 \
bash scripts/launch/train_hy273_unified_reaction_stage_c_control_ddp8.sh
```

400K 门检通过后，分别把 `STOP_STEP` 改为 `450000`、`500000`。脚本会从前一个 50K checkpoint 续训，禁止跨过门检直接跳段。
