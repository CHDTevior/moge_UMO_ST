# 当前 KV-Control 实现梳理

本文按“人能顺着张量走一遍”的方式，说明当前项目如何做 KV control：数据怎么进来，模型怎么接控制，训练怎么构造监督，推理/评估怎么把控制作用到采样过程里。文末附关键代码行号。

当前版本的核心设定：

- Backbone T2M CodeFlow 从已有检查点加载，并在 KV-control adapter-only 训练中冻结。
- KV/VQ tokenizer 与 decoder 冻结；decoder 支持连续 embedding 解码。
- 训练只更新 control adapter 参数。
- 当前优先跑的是 `encoder` clean target 版本：KV control 分支的 flow clean target 用 VQ encoder 的 pre-quant embedding，而不是量化后的 codebook embedding。
- 训练 loss 当前是 `0.1 * flow_loss + 0.9 * kv_control_loss`，terminal/clean auxiliary loss 为 0。
- 推理/评估时，B 是训练出来的 KV adapter 控制；C 是可选 test-time gradient guidance。

## 1. 一句话理解

人给模型一段文本和一些稀疏关节目标，例如“第 30 帧左手要在这里”。模型先看当前 noisy latent 会生成什么动作，再把“目标关节 - 当前预测关节”编码成 control condition。这个 control condition 不直接改 motion token，而是通过一个小 adapter 生成额外的 K/V，插入 DiT 的 attention，让 flow velocity 朝满足控制的位置偏过去。

整体张量流如下：

```text
HumanML3D motion + caption
  motion: [B, F, 272]
  text:   list[str]
        |
        v
VQ / KV tokenizer
  ids:                [B, T, P]
  codebook embedding: [B, T, P, D]
  encoder prequant:   [B, T, P, D]   <- 当前 KV 训练 clean target
        |
        +--------------------------+
        |                          |
        v                          v
flow training target          sparse joint controls
  x_clean: [B,T,P,D]            target_joints: [B,F,22,3]
                                target_mask:   [B,F,22,3]
                                      |
                                      v
base/no-control estimate at ODE time t
  z_t -> base velocity -> clean_base -> frozen decoder -> current_joints
                                      |
                                      v
control condition
  residual = (target_joints - current_joints) * mask
  target   = target_joints * mask
  control_cond = concat(residual, target)
  shape: [B, F, 22*3*2] = [B, F, 132]
                                      |
                                      v
KV control adapter
  control_cond -> control tokens -> per-layer extra K/V
                                      |
                                      v
DiT flow model
  normal self/cross attention K/V + control K/V
                                      |
                                      v
controlled velocity -> clean_pred -> frozen decoder -> joints
                                      |
                                      v
loss
  flow loss on embedding space
  control loss on selected joint positions
```

这里的关键点是：control loss 不在 272D raw motion 上直接算，而是把模型预测的 clean embedding 通过冻结 decoder 解回关节坐标，再对 sparse control mask 选中的关节位置算损失。

## 2. 数据与控制目标

训练数据仍然是 motion streamer 的 272D motion 表示和文本 caption。

```text
motion_272: [B, F, 272]
caption:   B 条文本
```

272D 会通过 `Mean.npy` / `Std.npy` 反归一化，然后恢复成 22 个关节的世界坐标：

```text
motion_272 normalized
  -> denormalize
  -> recover root translation / heading
  -> local joints + root transform
  -> joints: [B, F, 22, 3]
```

当前控制训练不是读取外部 control 标注，而是在每个 batch 内随机采样稀疏控制：

- 随机采样 1 到 5 个 keyframes。
- 随机采样 1 到 6 个 joints。
- 从 ground-truth motion 恢复出的 joints 作为 target。
- 生成 `target_joints` 和 `target_mask`，形状都是 `[B,F,22,3]`。
- `kv_control_dropout_prob=0.1` 会随机丢掉部分控制，用来增强鲁棒性。

代码位置：

- 272D layout 检查与恢复入口：`models/codeflow/motionstreamer272.py:25`
- raw 272D 恢复到 joints：`models/codeflow/motionstreamer272.py:50`
- normalized motion 恢复到 joints：`models/codeflow/motionstreamer272.py:107`
- 随机 joint/keyframe control 采样：`models/codeflow/kv_control.py:67`
- 训练循环里采样 control batch：`train_codeflow.py:2824`

## 3. Clean Target: 当前 B 训练到底学什么

KV control 训练里有一个很重要的选择：flow 的 clean target 是谁。

代码里支持三种：

```text
codebook:
  clean target = tokenizer quantize 后的 codebook embedding

encoder:
  clean target = tokenizer encoder 输出、quantize 之前的 continuous embedding

hybrid:
  clean target = alpha * encoder + (1-alpha) * codebook
```

当前启动脚本使用的是：

```text
--kv_control_clean_target encoder
```

所以当前 KV control 的 B 分支训练目标是：

```text
motion_272
  -> VQ encoder
  -> pre-quant continuous embedding
  -> 作为 flow clean target
```

这和原本 backbone T2M 使用 codebook embedding 作为目标不同。这样做的动机是：KV control adapter 在连续 latent 空间中学控制，和后续连续解码/连续 guidance 更一致；同时 backbone 冻结，普通 T2M 权重不被改坏。

代码位置：

- tokenizer 的 pre-quant encoder embedding：`models/codeflow/kv_vq.py:319`
- clean target 选择逻辑：`train_codeflow.py:1094`
- 训练循环里替换 `training_embeddings` 为 clean target：`train_codeflow.py:2839`
- 训练启动参数：`scripts/launch_kv_control_adapter_encoder_target_baseonly_ddp4_20260705.sh:71`

## 4. 模型架构

### 4.1 主体模型

当前用的是 part-structured CodeFlow。token 结构可以理解成：

```text
embedding: [B, T, P, D]

B = batch
T = temporal token length
P = body parts, 当前为 6
D = part embedding dim, 当前为 128
```

训练时 noisy latent `z_t` 和 clean target 都在这个 embedding 空间中。模型预测 flow velocity：

```text
z_t, text, time
  -> DiT
  -> velocity_pred: [B, T, P, D]
```

代码位置：

- part-structured 模型创建 DiT：`models/codeflow/part_structured_motion_code_flow.py:151`
- latent raw/model shape 映射：`models/codeflow/motion_code_flow.py:331`
- compute loss 里构造 `z_t` 和 velocity target：`models/codeflow/part_structured_motion_code_flow.py:583`

### 4.2 KV-Control Adapter

控制信息不是拼进 motion token，而是编码成额外 attention K/V。

```text
control_cond: [B, F, 132]
  -> control_encoder
  -> control_tokens: [B, T, hidden]
  -> low-rank K projection
  -> low-rank V projection
  -> per-layer extra K/V
```

其中 `132 = 22 joints * 3 xyz * 2`，两部分分别是：

```text
residual: target_joints - current_joints
target:   target_joints
```

二者都会乘上 sparse mask，只在被控制的帧和关节上非零。

代码位置：

- 构造 residual + target control condition：`models/codeflow/kv_control.py:13`
- DiT 中创建 control encoder / low-rank K/V：`models/codeflow/dit_blocks.py:641`
- control condition 编成 control tokens：`models/codeflow/dit_blocks.py:680`
- control tokens 生成 K/V：`models/codeflow/dit_blocks.py:704`
- MultiHeadAttention 拼接 extra K/V：`models/codeflow/dit_blocks.py:216`
- Double stream block 注入 control K/V：`models/codeflow/dit_blocks.py:286`
- Single stream block 注入 control K/V：`models/codeflow/dit_blocks.py:355`

### 4.3 Adapter-only 冻结策略

当前训练使用：

```text
--kv_control_train_adapter_only
```

这会冻结非 control adapter 参数，只训练这些前缀相关的参数：

```text
control_encoder
control_to_key
control_to_value
control_attention_bias
```

代码位置：

- adapter-only freeze 调用：`models/codeflow/part_structured_motion_code_flow.py:177`
- control 参数前缀定义：`models/codeflow/part_structured_motion_code_flow.py:201`
- 冻结非 control 参数：`models/codeflow/part_structured_motion_code_flow.py:219`
- init checkpoint 允许缺少 control adapter 参数：`train_codeflow.py:947`

## 5. 训练时一次 step 怎么算

一次训练 step 可以按下面理解：

```text
1. 读 batch
   motions: [B,F,272]
   captions

2. tokenizer 编码
   target_ids, code_embeddings, token_lengths

3. 如果 enable_kv_control:
   从 GT motion 恢复 joints
   随机采样 keyframes 和 joints
   得到 target_joints / target_mask

4. 选择 clean target
   当前为 encoder prequant embedding

5. flow match
   sample t
   z_t = interpolation(noise, clean_target, t)
   velocity_target = clean_target - noise

6. 先跑 base/no-control 分支
   z_t -> base velocity -> clean_base estimate
   clean_base -> frozen decoder -> current_joints

7. 构造 control condition
   concat(target-current, target) -> [B,F,132]

8. 跑 controlled 分支
   z_t + text + time + control_cond
     -> DiT with extra control K/V
     -> controlled velocity

9. 损失
   flow_loss: controlled velocity vs velocity_target
   clean_pred = z_t + f(t, velocity_pred)
   clean_pred -> frozen decoder -> predicted_joints
   kv_control_loss: predicted_joints vs target_joints on mask

10. 只更新 control adapter 参数
```

当前总 loss：

```text
total_loss =
    0.1 * flow_loss
  + 0.9 * kv_control_loss
  + 0.0 * terminal_loss
  + 0.0 * clean_loss
```

代码位置：

- tokenizer 编码得到 ids / embeddings / lengths：`train_codeflow.py:2818`
- KV control batch 采样：`train_codeflow.py:2824`
- training embeddings 替换成 clean target：`train_codeflow.py:2839`
- loss module 接收 control target/mask：`train_codeflow.py:2868`
- no-control base clean estimate：`models/codeflow/part_structured_motion_code_flow.py:621`
- base clean 解码到 current joints：`models/codeflow/part_structured_motion_code_flow.py:638`
- 构造 control condition：`models/codeflow/part_structured_motion_code_flow.py:644`
- controlled forward：`models/codeflow/part_structured_motion_code_flow.py:668`
- flow loss：`models/codeflow/part_structured_motion_code_flow.py:691`
- controlled clean prediction：`models/codeflow/part_structured_motion_code_flow.py:694`
- clean embedding 连续解码到 joints：`models/codeflow/part_structured_motion_code_flow.py:706`
- masked joint control loss：`models/codeflow/part_structured_motion_code_flow.py:721`
- 总 loss 公式：`models/codeflow/part_structured_motion_code_flow.py:822`
- optimizer step：`train_codeflow.py:2914`

## 6. 连续解码为什么重要

原始 VQ 路线通常是：

```text
ids -> codebook embedding -> decoder -> motion
```

但当前项目为了做连续控制，需要支持：

```text
continuous embedding [B,T,P,D]
  -> frozen KV decoder
  -> motion_272
  -> joints
```

这让两个事情变得可行：

- 训练时可以把 `encoder prequant embedding` 当成 clean target。
- 推理时 C guidance 可以直接对连续 clean embedding 或当前 ODE state 做梯度优化，然后再继续 flow update。

代码位置：

- 连续 embedding 解码：`models/codeflow/kv_vq.py:296`
- continuous embedding 解码到 joints：`models/codeflow/motionstreamer272.py:117`

## 7. 推理 / 评估协议

当前评估控制时，核心是每个 ODE step 都重新根据当前状态构造 control condition。

### 7.1 B-only: 使用训练好的 KV adapter

B-only 控制流程：

```text
for each ODE step:
  z_t
    -> base/no-control velocity
    -> clean_base
    -> decode clean_base to current_joints
    -> control_cond = concat(target-current, target)
    -> controlled velocity with KV adapter
    -> z_{t_next}
```

这表示 B 的控制是模型前向的一部分，不做额外梯度优化。

代码位置：

- 从当前 state 构造 control condition：`eval_codeflow_kv_control.py:260`
- sampling loop 的 ODE step：`eval_codeflow_kv_control.py:644`
- B-only/default control condition + forward update：`eval_codeflow_kv_control.py:728`
- eval batch 里构造 control targets：`eval_codeflow_kv_control.py:798`

### 7.2 C: Test-time Gradient Guidance

C 是推理时额外做的梯度优化，当前代码有两种 backend：

```text
clean guidance:
  优化 clean estimate 本身
  clean_raw -> decode -> joints -> control loss
  优化后的 clean 再换算回 velocity / ODE update

state guidance:
  优化当前 ODE state z_t
  z_t -> model/decode -> joints -> control loss
  优化后的 z_t 继续 flow update
```

更符合“每个 ODE step 对当前 flow update 对象做小步梯度”的是 state guidance；而 clean guidance 更像是对当前 step 的 clean estimate 做投影修正。两者代码都保留，评估时通过参数选择。

代码位置：

- guidance eta 调度：`eval_codeflow_kv_control.py:314`
- 每个 ODE step 的 guidance iter 调度：`eval_codeflow_kv_control.py:329`
- state guidance 主体：`eval_codeflow_kv_control.py:364`
- clean guidance 主体：`eval_codeflow_kv_control.py:498`
- sampling loop 中 clean guidance 分支：`eval_codeflow_kv_control.py:662`
- sampling loop 中 state guidance 分支：`eval_codeflow_kv_control.py:708`
- JSON metadata 记录 protocol / backend / schedule：`eval_codeflow_kv_control.py:1000`

### 7.3 当前建议的控制评估口径

之前讨论后，控制评估不要只看普通 generation quality，而要看 controlled joints 是否贴合目标。建议记录：

```text
control_mpjpe / masked joint error
trajectory error on selected joints
R-precision / FID / diversity 作为副指标
```

采样步数方面，之前 96 step 太慢；后续控制评估优先用 32 step。C guidance 的总迭代次数可以按 ODE step 分配，例如总计约 1000 次，并比较：

```text
linear_increase: 后期 step 做更多优化
linear_decrease: 前期 step 做更多优化
constant:        每步平均优化
```

## 8. 当前训练配置

当前 encoder-target KV-control 训练 launcher：

```text
scripts/launch_kv_control_adapter_encoder_target_baseonly_ddp4_20260705.sh
```

关键配置：

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
DDP world size = 4
batch_size per GPU = 32
global batch size = 128
max_epoch = 4000
max_steps = 1464000
lr = 1e-4
amp_dtype = bf16
flow_loss_weight = 0.1
kv_control_loss_weight = 0.9
terminal_loss_weight = 0.0
clean_loss_weight = 0.0
kv_control_loss_type = l1
kv_control_clean_target = encoder
kv_control_min_keyframes = 1
kv_control_max_keyframes = 5
kv_control_min_joints = 1
kv_control_max_joints = 6
kv_control_dropout_prob = 0.1
```

checkpoint / log：

```text
checkpoints/t2m/kv_control_adapter_encoder_target_baseonly_ddp4_20260705
logs/kv_control_adapter_encoder_target_baseonly_ddp4_20260705.log
tmux session: kvctrl_encoder_target_ddp4_20260705
```

代码位置：

- launcher checkpoint init/resume：`scripts/launch_kv_control_adapter_encoder_target_baseonly_ddp4_20260705.sh:6`
- DDP/GPU 设置：`scripts/launch_kv_control_adapter_encoder_target_baseonly_ddp4_20260705.sh:20`
- data/model 参数：`scripts/launch_kv_control_adapter_encoder_target_baseonly_ddp4_20260705.sh:27`
- batch/epoch/worker 设置：`scripts/launch_kv_control_adapter_encoder_target_baseonly_ddp4_20260705.sh:52`
- loss 与 KV flags：`scripts/launch_kv_control_adapter_encoder_target_baseonly_ddp4_20260705.sh:71`

## 9. 关键代码索引

| 模块 | 行号 | 作用 |
|---|---:|---|
| `options/codeflow_options.py` | 108 | KV control 训练参数入口 |
| `models/codeflow/kv_control.py` | 13 | residual + target control condition |
| `models/codeflow/kv_control.py` | 44 | masked joint position loss |
| `models/codeflow/kv_control.py` | 67 | 随机 keyframe/joint control 采样 |
| `models/codeflow/motionstreamer272.py` | 50 | 272D raw motion 恢复 joints |
| `models/codeflow/motionstreamer272.py` | 107 | normalized 272D 恢复 joints |
| `models/codeflow/motionstreamer272.py` | 117 | continuous embedding 解码到 joints |
| `models/codeflow/kv_vq.py` | 296 | continuous embedding 通过冻结 decoder 解码 |
| `models/codeflow/kv_vq.py` | 319 | VQ encoder pre-quant embedding |
| `models/codeflow/dit_blocks.py` | 216 | attention 中拼接 extra K/V |
| `models/codeflow/dit_blocks.py` | 641 | control encoder 与 low-rank K/V adapter |
| `models/codeflow/dit_blocks.py` | 680 | control condition 编码到 control tokens |
| `models/codeflow/part_structured_motion_code_flow.py` | 177 | adapter-only freeze |
| `models/codeflow/part_structured_motion_code_flow.py` | 621 | no-control base clean estimate |
| `models/codeflow/part_structured_motion_code_flow.py` | 644 | 从 current/target joints 构造 control condition |
| `models/codeflow/part_structured_motion_code_flow.py` | 668 | controlled forward |
| `models/codeflow/part_structured_motion_code_flow.py` | 721 | KV control joint loss |
| `models/codeflow/part_structured_motion_code_flow.py` | 822 | 总 loss 公式 |
| `train_codeflow.py` | 1094 | clean target 选择 codebook/encoder/hybrid |
| `train_codeflow.py` | 2818 | tokenizer 编码 batch |
| `train_codeflow.py` | 2824 | 训练时采样 sparse controls |
| `train_codeflow.py` | 2839 | 使用 encoder clean target 替换 training embeddings |
| `train_codeflow.py` | 2868 | loss 调用传入 control target/mask |
| `train_codeflow.py` | 2914 | backward/clip/optimizer step |
| `eval_codeflow_kv_control.py` | 260 | 推理时从当前 state 构造 control condition |
| `eval_codeflow_kv_control.py` | 329 | C guidance 每步迭代数调度 |
| `eval_codeflow_kv_control.py` | 364 | state guidance |
| `eval_codeflow_kv_control.py` | 498 | clean guidance |
| `eval_codeflow_kv_control.py` | 644 | ODE sampling loop |
| `eval_codeflow_kv_control.py` | 728 | B-only/default controlled forward |
| `scripts/launch_kv_control_adapter_encoder_target_baseonly_ddp4_20260705.sh` | 71 | 当前训练 loss 与 KV control 参数 |

## 10. 用人的视角再串一次

训练时，模型看到一条真实动作和文本。我们从真实动作里抽几个“你必须满足”的关节点作为控制目标。然后把同一条真实动作编码成 continuous clean embedding，作为 flow 要还原的目标。

在一个随机噪声时间点 `t`，模型先问自己：“如果没有控制，我现在会生成什么动作？”它把这个 no-control clean estimate 解码成 joints。接着我们把“目标 joints 和当前 joints 的差”喂给 control adapter。adapter 生成额外 K/V，插进 DiT attention，让模型重新预测一个 controlled velocity。

最后，训练有两个约束：第一，controlled velocity 仍然应该朝 clean embedding 走；第二，controlled clean prediction 解码出来后，被 mask 选中的关节要贴近目标位置。因为 backbone 和 decoder 都冻结，训练压力集中在 KV control adapter 上。

推理时没有 ground-truth full motion，只有文本和用户给的 sparse controls。每个 ODE step 都根据当前生成状态估计 current joints，再和目标 joints 做差，动态生成 control condition。B-only 直接靠训练好的 adapter 控制；如果开 C guidance，就在每个 ODE step 额外对 clean estimate 或 `z_t` 做若干步梯度优化，让控制误差进一步下降。
