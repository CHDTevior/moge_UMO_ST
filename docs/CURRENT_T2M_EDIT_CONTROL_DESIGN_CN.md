# 当前 K-Encoder 的 T2M、Edit 与 Control 设计

更新时间：2026-07-28

## 1. 文档目的

本文给代码审查者说明当前共享模型的真实状态：

1. 当前 200K checkpoint 实际训练过什么；
2. T2M、Motion Editing、Kimodo-like Control 在数据和网络中如何区分；
3. 三类任务分别如何构造条件、loss 和 CFG；
4. 下一阶段为什么建议先做 Edit bootstrap；
5. 哪些 Control 设计仍等待审查，不应被当作已经冻结的方案。

当前是科研环境。本文关注数据语义、表示、条件路径、目标函数、训练调度、
采样和评估，不把生产级防篡改机制作为科研结论或训练门禁。

## 2. 当前 checkpoint 状态

当前选定基模：

```text
K-Encoder Stage-A 200K

checkpoint:
/mnt/afs/mogeflow-control/outputs/hy273_text_fusion/
hy273_kencoder_stageA_ddp4x32_20260727_0832/model/step_00200000.pt
```

它只训练过 T2M：

```text
step 0 -> 200K
T2M / Control / Edit = 100% / 0% / 0%
DDP4 x batch/rank 32
global batch = 128
```

Control、Edit 和 Edit+Control 的代码路径已经实现，但这个 200K checkpoint
尚未在这些任务上更新过。旧的 Control-first Stage-B 已停止。

## 3. 数据与 HY273 表示

### 3.1 数据

HumanML3D K273：

- 绝对动作文本 `caption -> target motion`；
- 用于 T2M；
- Control 条件从同一 target motion 合成。

MotionFix K273：

- `source motion + relative instruction -> target motion`；
- 用于 Motion Editing；
- instruction 不能当作 target 的普通绝对 caption。

两套数据都是 30 FPS，并共享当前 unified stats。原始 `.npy` 保持只读，
统一 manifest 保存绝对路径、任务、文本 profile、pair 元信息和时间对应关系。

### 3.2 表示

模型输入/输出的 motion tensor：

```text
x: [B,T,273]

[0:3]     smooth_root_pos
[3:5]     global_root_heading
[5:71]    local_joints_positions, 22x3
[71:203]  global_rot_data, 22x6
[203:269] global_joint_velocities, 22x3
[269:273] foot_contacts, 4
```

当前 R13 `unified_273_clean_flow_v1` 对完整 273D 使用同一个 Gaussian
rectified-flow state。Contact 使用联合 stats 归一化并参与同一 clean-x0 flow，
不再使用旧的独立 sigmoid/contact-feedback 流程。

### 3.3 数据增强

T2M/Control：

```text
select window
-> frame-0 root XZ origin shift
-> whole-sequence random yaw
-> c_dir = transformed frame-0 heading
-> normalize
```

MotionFix：

```text
paired crop/time map
-> source 和 target 各自做 frame-0 root XZ shift
-> source 和 target 使用同一个随机目标 yaw
-> controls 从变换后的 target 构造
-> source/target 使用同一组 stats
```

禁止 source 和 target 独立随机 yaw。

## 4. 当前模型架构

### 4.1 总体结构

模型类：

```text
HY273KimodoContextFlow
  extends HY273RedenoiseKimodoLike
```

核心尺寸：

```text
hidden_dim:          1024
heads:               8
root double/single:  3 / 6
body double/single:  3 / 6
MLP ratio:           2.0
dropout:             0
self-conditioning:   disabled
parameters:          387,212,049 total
```

模型先预测 clean root，再预测 body：

```text
model_in [B,T,546]
  = imputed/noisy HY273 [B,T,273]
  + observation mask [B,T,273]

Root DiT
  -> clean root [B,T,5]

predicted root
  -> KimodoRootConditioner
  -> local root [B,T,4]

Body DiT(
  local root [B,T,4],
  noisy body [B,T,268],
  full observation mask [B,T,273]
)
  -> clean body [B,T,268]

concat
  -> clean x0 [B,T,273]
```

训练时 root-to-body bridge 默认 detach，避免 body loss 反向改变 root denoiser。
推理时 body 始终消费同一次 root denoiser 给出的完整 root trajectory。

### 4.2 K-Encoder 文本路径

当前文本编码器不是原 HYText 的 `Qwen tokens + CLIP pooled AdaLN`，而是：

```text
LLM2Vec Meta-Llama-3-8B MNTP-supervised
-> offline profile-aware cache
-> one 4096D embedding per text
-> learned projection to H=1024
-> one text token
```

文本 profile 分开：

```text
hytext_absolute_motion_v1  用于 T2M/Control 的绝对动作描述
hytext_relative_edit_v1    用于 Edit 的相对编辑指令
```

当前 K-Encoder 设置：

```text
conditioning_architecture = llm2vec_flux
text_global_conditioning  = llm2vec_tokens_only
text_fusion_mode          = f00
```

`llm2vec_tokens_only` 表示文本不再加入 pooled-text AdaLN：

```text
AdaLN global condition = timestep embedding + c_dir embedding
```

文本只通过 Flux/MMDiT token interaction 影响 motion。`f00` 表示：

```text
shared Q/K/V projection
+ bidirectional motion-text attention
```

Timestep 与 `c_dir` 的 AdaLN 路径仍然保留。

## 5. 三类任务如何区分

统一条件对象是：

```text
ConditionBatch
```

关键字段包括：

```text
task_id
capability_id
source_motion/source_present/source_role_id
target_op_id
text_encoding_profile
observed motion
hard observation mask
target/source valid mask and time map
```

任务路由：

| 任务 | task_id | source | hard control mask | text profile |
|---|---|---:|---:|---|
| T2M | GENERATE | absent | empty | absolute |
| Control | GENERATE | absent | non-empty | absolute |
| Edit | EDIT | present | empty | relative |
| Edit+Control | EDIT | present | non-empty | relative |

T2M 与 Control 都属于生成任务。它们不依赖额外模式开关，而是由 observation
mask/values 区分。Edit 使用显式 `EDIT` task identity 和 source context，因此不会
和 source-free T2M/Control 混淆。

当前 source context 投影、task/op/role embedding 全部零初始化。Stage-A 中它们
保持零，因此 no-source T2M 路径与原生成模型一致。

## 6. T2M

### 6.1 训练输入

```text
target: HumanML3D K273 [B,T,273]
text: absolute caption
source: absent
observed/mask: all zero
task: GENERATE
capability: T2M
```

Rectified flow：

```text
epsilon ~ N(0,I)
t ~ LogitNormal(mean=-0.8, std=0.8)
z_t = t*x0 + (1-t)*epsilon
model([z_t, zero_mask], t, c_dir, text)
  -> x0_hat [B,T,273]
```

模型参数化是 clean-x0 prediction，但主表示误差按 velocity-space
`velocity_mse` 计算。

### 6.2 T2M 推理

无控制 T2M 使用两支 CFG：

```text
x_empty
x_text

x_guided
  = x_empty
  + g_text * (x_text - x_empty)
```

Stage-A 评估建议：

```text
g_text = 1.5  paraphrase 一致性较好
g_text = 2.0  语义区分较强，公共比较档
g_text = 3.0  当前不建议作为默认值
```

### 6.3 Stage-A 已观察结果

- breakdance、empty、slow walk 已有明确输出差异；
- `breakdances` OOD 同义表达比旧文本路径改善；
- 复合动作不再全部坍缩为 walk prior；
- 200K 的 jerk/foot skate 相比 150K 有小幅退化；
- fixed-t 单点上 correct-text MSE 仍未稳定优于 empty，文本对齐尚未完全解决。

因此当前 200K 被暂定为 T2M 基模，而不是最终三任务模型。

## 7. Kimodo-like Control

### 7.1 已实现控制 pattern

```text
root_sparse:
  sparse root XZ，50% 带 heading，50% XZ-only

root_dense:
  dense root path，50% 带 heading，50% XZ-only

end_effector:
  随机非空手脚子集的位置与末端 rotation

fullpose:
  sparse full-body joint positions

contact:
  sparse four-channel foot-contact configurations
```

每个 Control 样本：

```text
75% one pattern
25% two different patterns
```

Sparse keyframe 数量 curriculum 从 1 增至 20，并偏向较少关键帧。

### 7.2 Control 训练输入

控制由同一个 GT target 合成：

```text
observed [B,T,273]
mask     [B,T,273]
```

在每个 flow timestep：

```text
z_t = t*x0 + (1-t)*epsilon
z_t_imputed = where(mask, observed, z_t)
model_in = concat(z_t_imputed, mask)
```

Control 不使用独立 adapter。它和 T2M 共享全部 root/body backbone。

### 7.3 Control loss

Control 样本保留和 T2M 相同的基础生成 loss，并在受控坐标增加：

```text
control_continuous = 0.25
control_contact    = 0.02857142857142857
```

Control loss 监督模型自身的 raw adherence，不只监督最终 overwrite 后的结果。

### 7.4 Control 推理

带文本的 Control 构造四个分支：

```text
joint    = text + control
text     = text only
control  = control only
empty    = unconditional
```

当前 guidance：

```text
x_guided
  = x_empty
  + g_text    * (x_text - x_empty)
  + g_control * (x_control - x_empty)
```

Hard observations 只在每个 ODE step 的 denoiser input 上 overwrite；ODE state
本身不永久 clamp。采样器分别返回：

```text
raw learned sample
exact-clamped sample
```

这样 benchmark 可以测模型真实 adherence，也可以给调用者严格满足输入约束的输出。

### 7.5 当前未冻结的 Control 设计

旧 Stage-B：

```text
200K -> 400K
T2M / Control / Edit = 10 / 90 / 0
```

目前已停止，不应直接启动。下一版仍需审查：

1. Control 是否继续使用当前 input overwrite；
2. text/control CFG 是否保留当前加法分解；
3. pattern 权重是否仍均匀；
4. curriculum 应从哪个 global step 重新起算；
5. 引入 Edit 后 Control replay 应为 60%、70% 还是接近 Kimodo 的 90%；
6. Edit+Control 是否从联合阶段开始训练。

## 8. Motion Editing

### 8.1 训练输入

```text
source:      MotionFix source K273
instruction: relative edit text
target:      MotionFix target K273
task:        EDIT
capability:  MOTION_EDIT
```

Source 不写入 hard control mask。它通过独立 source context 路径进入模型：

```text
source [B,K,Ts,273]
+ source value mask
-> normalize/project
-> align to target time [B,K,Tt,H]
-> aggregate source slots
+ task/op/role/length embeddings
-> context_root/context_body [B,Tt,H]
```

当前默认 source fusion 是逐帧 additive：

```text
target_hidden + source_context
```

代码同时支持参数不新增的 token-block 研究模式：

```text
[source tokens, SEP, target tokens]
```

Stage-A 不含 source，两种模式在当前 T2M checkpoint 上没有行为差异。正式
Edit bootstrap 使用哪一种，需要在开训前明确冻结；不能训练后再切换。

### 8.2 Edit CFG

纯 Edit 使用三层条件分解：

```text
x_empty
x_source
x_source_text

x_guided
  = x_empty
  + g_source * (x_source - x_empty)
  + g_edit   * (x_source_text - x_source)
```

`g_source` 控制 source identity 保持，`g_edit` 控制 instruction 在给定 source
后的增量。它们不能合并成一个不可解释的 CFG 数值。

### 8.3 既有 Edit 实验结论

旧 HYText 模型中：

- decomposed CFG 是有效基础设施；
- `no_rank_positive_only` 把 correct assignment 从 62.1% 提升到 75.9%；
- ranking/negative 容易通过推坏 negative 分支获利；
- changed-region loss 未超过 positive-only；
- temporal velocity/speed loss 改善平均误差，但没有解决 `move feet faster`；
- 提高 Edit CFG 会放大错误的静态方向；
- 主要问题是 source-copy shortcut、文本路径弱和动态语义监督覆盖不足。

K-Encoder 已在 T2M 上改善文本路由，因此下一步先验证它能否改善 Edit，暂不同时
加入新的 temporal/ranking loss。

## 9. Loss 总览

所有任务共享基础目标：

```text
clean-x0 model prediction
+ group-balanced velocity-space MSE
+ contact
+ clean root velocity
+ clean joint velocity
+ FK consistency
+ foot lock
```

当前主要配置：

```text
semantic group weights: [10, 2, 10, 10, 3]
representation scale:   0.09397019716051493
contact:                 0.010739451104058849
clean root velocity:     0.01
clean joint velocity:    0.01
FK consistency:          0.07
foot lock:               0.01
```

Control 再增加受控位置 loss。拟议 Edit bootstrap 使用此前最有效且最简单的
positive-only 增强：

```text
+ 0.05 * target clean-x0 positive loss
+ 0.02 * hard-coordinate clean-x0 positive loss
```

第一轮不加入：

```text
ranking negative
changed-region auxiliary loss
temporal speed auxiliary loss
low-t oversampling
```

## 10. 下一阶段建议

### 10.1 Stage-BE：Edit bootstrap

建议：

```text
step: 200K -> 250K
T2M / Control / Edit = 60% / 0% / 40%
global batch = 128
```

为什么不是 50/50：

1. MotionFix train 只有约 5.4K pair，40% 已产生高密度 Edit exposure；
2. Source 与 target 通常相似，过高 Edit 比例会加强 source-copy shortcut；
3. K-Encoder 文本路由刚建立，60% T2M replay 能保护 absolute-text semantics；
4. 此阶段目标是启动 source/text/task 通路，不是立刻让小数据主导 backbone；
5. 若 40% Edit 在 50K 内仍学不会，根因更可能是条件/目标设计，不是再加 10% exposure。

拟议 optimizer：

```text
G0 existing backbone/text fusion: 5e-5
G1/G2 source context/task:        1e-4
```

当前旧 Stage-B 会把 source-context LR 保持为 0，因此不能直接复用。

拟议 Edit 内部分支：

```text
source + instruction  80%
source identity       10%
text only              5%
unconditional          5%
```

这一阶段不加入 Edit+Control。

### 10.2 250K 评估

在加入 Control 之前检查：

T2M：

- Stage-A 困难文本和 paraphrase；
- matched-noise text/empty routing；
- jerk、foot skate、contact consistency。

Edit：

- 同 source 的 correct/empty/opposite instruction；
- source preservation 与 target movement；
- `faster/slower`、方向、幅度、肢体指定等可视化；
- `move feet faster` 的脚速、峰频和 contact occupancy；
- raw 与 EMA 都检查，避免 EMA lag 误判。

如果 Edit 在这个较简单的两任务环境里失败，应先定位 Edit 真因，不应加入 Control
把条件竞争重新混在一起。

### 10.3 后续联合阶段

在 Stage-BE 通过后，初始候选是：

```text
T2M / Control / Edit = 10% / 70% / 20%
```

这不是最终冻结值。Control 的输入、CFG、pattern 和 curriculum 完成审查后再确定。
无论最终 Control 比例是多少，都不建议切回 `10/90/0`，因为 Edit 需要持续 replay。

## 11. 主要代码入口

```text
train_hy273_multitask.py
  model construction, optimizer groups, flow/loss, DDP, checkpoint/resume

sample_hy273_multitask.py
  task routing, layered CFG, ODE integration, per-step control overwrite

data/hy273_multitask_manifest_dataset.py
  HumanML3D/MotionFix materialization and paired augmentation

data/hy273_multitask_scheduler.py
  high-level task ratios and Edit condition patterns

models/raw_motion/kimodo_context_flow_dit.py
  source context, task/op/role encoding, additive/token-block source fusion

models/raw_motion/kimodo_like_flow_dit.py
  root-first/body-second denoiser and K-Encoder text path

models/raw_motion/hy273_constraints.py
  Kimodo-like Control compiler and curriculum

models/raw_motion/hy273_multitask_losses.py
  shared T2M/Control objectives

models/raw_motion/hy273_unified_edit_losses.py
  Edit positive/ranking/discrepancy/temporal research objectives
```
