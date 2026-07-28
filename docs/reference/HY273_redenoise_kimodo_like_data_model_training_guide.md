# HY273 `redenoise_kimodo_like`：数据、模型与训练说明

> 本文描述的是 2026-07-12 正在运行的真实实现。文中会明确区分当前行为和未来实验。

## 0. 一句话理解

我们直接在 Kimodo273 的 raw motion space 里训练 rectified flow。模型每次 forward 时，先用 Root DiT 预测 clean 全局根轨迹，再把根轨迹转成 local-root 动态条件，交给 Body DiT 预测其余身体通道。

```text
模型输出语义:       clean x0 prediction
主损失计算空间:     capped velocity-equivalent space
采样过程:           一条 ODE，每个 ODE step 内执行 root -> body
训练阶段:           同一个模型，Stage 1 文本生成 -> Stage 2 控制续训
动作空间:           normalized raw Kimodo273，不使用 VQ 或 motion autoencoder
```

它不是 VQ latent diffusion，也不是先完整生成 root、再单独开始第二次 body diffusion。

---

## 1. 当前数据是什么

### 1.1 数据来源与规模

当前训练使用 HumanML3D，由 HY201 转成官方 Kimodo273 通道语义：

```text
动作路径:
/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22

文本路径:
/mnt/afs/mogo_base/datasets/HumanML3D/texts

单个动作:
[T, 273], float32, 30 FPS
```

转换仓库报告总计 26,846 个 motion、5,945,004 帧。当前 loader 会额外过滤少于 16 帧的动作和已知 smooth-root fallback 短动作。

| split | 可用 motion | 帧数 | 中位长度 | 最大长度 | caption 行数 | 缺失文本 |
|---|---:|---:|---:|---:|---:|---:|
| train | 21,454 | 4,749,418 | 240 | 300 | 64,104 | 0 |
| val | 1,338 | 301,538 | 252 | 300 | 3,996 | 0 |
| test | 4,038 | 893,910 | 240 | 300 | 12,078 | 0 |

配置中 `max_frames=300`。当前可用 motion 本身最长就是 300 帧，所以虽然 loader 支持 `random_crop=True`，这批数据上实际上不会触发裁剪。

代码：`data/kimodo273_datasets.py:66-169`。

### 1.2 文本和动作如何配对

每个 motion 可以有多条 caption。每次读取样本时，只从这个 motion 自己的 caption 列表中选择一条。

当前 deterministic trace 模式下，caption 和 crop 的选择由下面这些值确定：

```text
seed + epoch + dataset_index
```

因此断点续训能够继续相同的数据随机序列。

HumanML3D 文本行可能是：

```text
caption # tokens # from_tag # to_tag
```

当前 parser 只保留 `caption`，会丢掉 span。训练集 64,104 条 caption 中有 2,360 条带非零 span，占约 3.68%。当前边界是：

- full-motion caption 配对正常；
- segment caption 目前配给整段 motion，而不是对应 segment；
- 当前 motion 不超过 300 帧，因此不是 random crop 引起的错配；
- 但这 3.68% segment caption 仍可能产生少量文本语义噪声。

这是后续可做的 ablation，不把它当成当前已解决问题。

代码：`data/kimodo273_datasets.py:36-54`。

---

## 2. 一帧 HY273 里有什么

```text
HY273 frame [273]
|
+-- [0:3]      smooth_root_pos [3]
|                Y-up 世界坐标
|                x/z 经过平滑，y 保留 root 高度
|
+-- [3:5]      global_root_heading [2]
|                [cos(yaw), sin(yaw)]
|
+-- [5:71]     joint_positions [22 * 3 = 66]
|                x/z 相对 smooth-root x/z
|                y 是世界高度
|                不按每帧 heading canonicalize
|
+-- [71:203]   global_joint_rot6d [22 * 6 = 132]
|                官方 Kimodo global cont6d
|
+-- [203:269]  global_joint_velocity [22 * 3 = 66]
|                世界坐标 joint velocity
|
+-- [269:273]  foot_contacts [4]
                 left ankle, left foot, right ankle, right foot
                 二值 0/1
```

定义：`models/raw_motion/hy273_slices.py:11-80`。

### 2.1 为什么 position 不按每帧 heading 转正

position 通道只减 smooth-root x/z，不会每帧都把人体旋转到固定朝向。

这样可以保留空翻、侧手翻等动作的时间连续性。人在倒立附近时，基于 hips 投影的 heading 可能突然跳 180 度。如果每帧都据此转正，物理上连续的动作会变成表示上的突变。

恢复全局关节位置：

```text
global_joint.x = stored_joint.x + smooth_root.x
global_joint.y = stored_joint.y
global_joint.z = stored_joint.z + smooth_root.z
```

代码：`models/raw_motion/hy273_slices.py:105-115`。

### 2.2 Contact 为什么单独处理

```text
[0:269] 连续通道:
(x - mean) / sqrt(std^2 + 1e-5)

[269:273] contact:
保持 0/1，不做 z-score
```

网络输出四个 contact logits，`sigmoid(logits)` 才是 contact probability。Contact 用 BCE 训练，不作为普通连续 ODE state。

代码：

- `models/raw_motion/hy273_normalizer.py:35-108`
- `models/raw_motion/flow_schedule.py:71-107`

---

## 3. 训练前的数据变换

几何增强在未归一化的 source domain 完成：

```text
raw HY273 [B,T,273]
       |
       | 1. 首帧 smooth-root x/z 平移到 (0, 0)
       v
origin-shifted motion
       |
       | 2. 在 [-pi, pi] 随机选择首帧目标朝向
       | 3. 整段 motion 统一绕 Y 轴旋转一次
       v
augmented source motion x0_un [B,T,273]
       |
       +--> c_dir = 旋转后的首帧 heading [B,2]
       |
       | 4. 连续通道 normalize，contact 保持 0/1
       v
clean target x0 [B,T,273]
```

这里是整段统一旋转，不是每帧 heading canonicalization。

Stats 只用 train split 构建，并使用 0、90、180、270 度四个 yaw 做 quadrature。原因是训练会随机首帧朝向，stats 也要覆盖宽朝向分布。

代码：`models/raw_motion/hy273_normalizer.py:111-167`。

```text
stats 路径:
/mnt/afs/mogo_base/datasets/HumanML3D/
  kimodo273_from_hy201_smplx22/derived_stats/redenoise_kimodo_like_v1/
  |-- full/Mean.npy, Std.npy
  |-- local_root/Mean.npy, Std.npy
  `-- training_assets.json
```

---

## 4. 文本条件怎么进入模型

Qwen3-8B 和 CLIP ViT-L/14 只在离线 cache 阶段运行。DDP 训练时不会在线加载大文本模型。

```text
caption
   |
   +--> memmap: Qwen token [B,128,4096]
   |                       |
   |                       `--> Linear 4096 -> 1024
   |                            text tokens [B,128,1024]
   |
   `--> memmap: CLIP pooled [B,1,768]
                           |
                           `--> MLP 768 -> 1024
                                pooled text [B,1024]
```

Qwen tokens 进入 motion-text cross-attention。CLIP pooled 加入全局条件：

```text
global_cond = timestep_embedding(t)
            + direction_embedding(c_dir)
            + pooled_text
```

训练有 10% text dropout。被 dropout 的样本查 empty-caption cache row，用来学习 CFG unconditional branch。

代码：

- `models/raw_motion/hytext_cache.py:124-204`
- `models/raw_motion/kimodo_like_flow_dit.py:148-205`

---

## 5. Rectified flow 状态怎么构造

### 5.1 时间方向

```text
t = 0: noise
t = 1: clean motion
```

对 269 个连续通道：

```text
epsilon ~ Normal(0, I)
z_t = t * x0 + (1 - t) * epsilon
v_target = x0 - epsilon
```

训练 timestep：

```text
t = sigmoid(N(-0.8, 0.8^2))
```

它比对称分布更偏向 noisy 到中间区域。

代码：`models/raw_motion/flow_schedule.py:21-35`。

### 5.2 控制 overwrite 和输入

控制编译器产生：

```text
observed_motion obs [B,T,273]
motion_mask      m   [B,T,273], bool
```

输入侧按 entry overwrite：

```text
z_imp = z_t * (1 - m) + obs * m
model_in = concat(z_imp, m.float())
model_in = [B,T,546]
```

显式 mask 用来区分：

```text
这个 entry 未被控制
vs.
这个 entry 被控制成数值 0
```

Stage 1 的 mask 全 0，Stage 2 才有 sparse/dense mask。

代码：`models/raw_motion/flow_schedule.py:71-107`。

---

## 6. Root/Body 两级模型

### 6.1 完整 tensor flow

一次 forward 内有两个独立的 FrameMotionTextDiT：

```text
                            same t, c_dir, HYText
                                      |
                                      v
model_in [B,T,546] --> ROOT DiT --> root x0_hat [B,T,5]
                                           |
                                           | 完整预测 root trajectory
                                           | 训练时 stopgrad
                                           v
                                  FP32 global -> local
                        [yaw_vel, vx_world, vz_world, root_y]
                                  local_root [B,T,4]
                                           |
                                           v
body_in = concat(
    local_root4,
    noisy/imputed body268,
    full mask273
) = [B,T,545]
                                           |
                                           v
                                       BODY DiT
                                           |
                                           v
                               body output [B,T,268]
                                           |
                                           v
final output [B,T,273]
  [0:269]   clean continuous x0
  [269:273] contact logits
```

代码：`models/raw_motion/kimodo_like_flow_dit.py:30-255`。

### 6.2 Root stage

Root stage 看完整 `[z_imp, mask]`，不只是五个 root channel。因此它能利用 noisy body、文本、方向和控制上下文推断根轨迹。

```text
input projection: 546 -> 1024
backbone:         3 double-stream + 6 single-stream blocks
output:           1024 -> 5
```

五个输出是 normalized clean estimate：

```text
[root_x, root_y, root_z, cos(yaw), sin(yaw)]
```

### 6.3 Global-to-local root bridge

Root prediction 在 FP32 反归一化并做 finite difference：

```text
local_root[...,0] = 相邻 heading 的角速度
local_root[...,1] = world root x velocity
local_root[...,2] = world root z velocity
local_root[...,3] = world root y height
```

然后使用独立的 train-only local-root stats 归一化。

Bridge 使用完整 `root_prediction_raw`，不会先把 sparse GT waypoint 拼进预测 root 再差分，否则 waypoint 两侧会产生人为 velocity spike。

训练时：

```text
root_for_body = stopgrad(root_prediction_raw)
```

因此 body loss 不通过 bridge 反向修改 Root DiT。Root 由自己的输出 loss 学习，Body 学习适应 Root 当前的预测分布。Inference 时 detach 不影响数值。

代码：

- `models/raw_motion/kimodo_like_flow_dit.py:223-247`
- `models/raw_motion/hy273_root_conditioning.py:22-103`

### 6.4 Body stage

```text
predicted local root: 4
state channels 5:273: 268
完整 control mask: 273
总输入: 545
```

```text
input projection: 545 -> 1024
backbone:         3 double-stream + 6 single-stream blocks
output:           1024 -> 268
```

Body 输出包含 264 个连续 channel 和 4 个 contact logits。

### 6.5 模型规模

```text
hidden dim:            1024
attention heads:       8
MLP ratio:             2.0
network dropout:       0.0
trainable parameters:  387,632,913
self-conditioning:     接口已实现，当前关闭
```

HYText projection 参与训练；离线 Qwen3-8B 和 CLIP-L 不计入模型参数。

### 6.6 它不是两条 diffusion

ODE32 实际执行：

```text
step 0:  Root -> local bridge -> Body -> flow update
step 1:  Root -> local bridge -> Body -> flow update
...
step 31: Root -> local bridge -> Body -> flow update
```

Root 和 Body 使用同一个 ODE time 和同一份演化中的 motion state。Root 是每个 denoising evaluation 内结构上先于 Body，不是先完成一条独立采样。

---

## 7. 网络预测什么，loss 比较什么

### 7.1 Head 直接预测 clean x0

```text
x0_hat = model(...)[..., 0:269]
```

`redenoise_kimodo_like` 会拒绝 velocity head，否则 bridge 会把 velocity 错当 clean global root。

保护代码：`train_hy273_raw_flow.py:464-505`。

### 7.2 主 loss 在 velocity-equivalent space

虽然 head 是 x0，主 loss 会把预测和 GT 都换成同一 imputed state 对应的 capped velocity：

```text
denom = max(1 - t, 0.05)
v_hat = (x0_hat - z_imp) / denom
v_gt  = (x0      - z_imp) / denom
```

每个语义 block 先独立求均值：

```text
L_repr = 0.093970197 / 35 * (
    10 * MSE(root_xyz)
  +  2 * MSE(heading_cos_sin)
  + 10 * MSE(joint_positions)
  + 10 * MSE(global_rot6d)
  +  3 * MSE(global_velocities)
)
```

这样设计的原因：

- x0 head 给 bridge、未来 self-conditioning 和 control loss 明确 clean estimate；
- velocity-equivalent loss 与采样时 ODE vector field 对齐；
- `denom >= 0.05` 防止近 clean timestep 权重发散；
- block 独立 reduce，避免 132 维 rotation 只因维度多就主导 loss。

代码：

- `train_hy273_raw_flow.py:541-577`
- `train_hy273_raw_flow.py:624-651`

### 7.3 与 Kimodo 官方 loss 的精确关系

Kimodo 技术报告 Eq. (1) 写的是直接作用于 clean `x0_hat` 的 component-wise SmoothL1：

```text
L_Kimodo = 10 * SmoothL1(root_position)
         +  2 * SmoothL1(root_heading)
         + 10 * SmoothL1(joint_position)
         +  3 * SmoothL1(joint_velocity)
         + 10 * SmoothL1(joint_rotation)
         +  4 * SmoothL1(foot_contact)
         +  5 * SmoothL1(FK(rotation), joint_position)
```

所以当前连续五个 block 的相对比例其实与 Kimodo 一致：

| 语义 block | Kimodo | 当前实现 |
|---|---:|---:|
| root position | 10 | 10 |
| root heading | 2 | 2 |
| joint position | 10 | 10 |
| joint velocity | 3 | 3 |
| joint rotation | 10 | 10 |

代码中 rotation 写在 velocity 前面，所以显示为 `10:2:10:10:3`；按 Kimodo 报告的语义顺序重排就是 `10:2:10:3:10`，两者相同。

但当前完整 loss 不是 Kimodo exact loss，而是 rectified-flow 适配版，主要差异为：

1. Kimodo 用 DDPM clean-x0 SmoothL1；当前用 clean-x0 head，但主项在 capped velocity-equivalent space 做 MSE。
2. Kimodo 直接使用上述 gamma；当前连续主项整体乘 `0.093970197 / 35`，这个尺度来自真实 batch 上的梯度预算校准，不来自 Kimodo。
3. Kimodo contact 是权重 4 的 SmoothL1；当前 contact 是权重 0.1 的 BCEWithLogits。
4. Kimodo FK 权重是 5；当前 FK 权重是校准后的 0.07，并有 5K warmup。
5. 当前额外加入 0.01 root velocity、0.01 joint velocity、0.01 foot-lock，以及 Stage-2 control SmoothL1；这些不在 Kimodo Eq. (1) 中。
6. Kimodo 公开仓库没有训练 loss 源码，能够确认的是技术报告公式，无法从公开代码进一步确认它内部对各 component 的精确 reduction 细节。

因此准确表述应是：**模型架构、motion representation 和连续 block 相对比例是 Kimodo-like；完整优化目标是为当前 rectified-flow 实验校准过的 Kimodo-ratio loss，不是 Kimodo loss 的逐项复刻。**

---

## 8. 完整 loss

```text
L_total = 1.00 * L_repr
        + 0.10 * L_contact_BCE
        + 0.01 * L_clean_root_velocity
        + 0.01 * L_clean_joint_velocity
        + 0.01 * L_foot_lock
        + w_fk  * L_FK_consistency
        + w_ctl * L_controlled_entries
```

```text
w_fk:
  前 5,000 global steps 从 0 warmup 到 0.07

w_ctl:
  Stage 1 = 0.00
  Stage 2 = 0.25
```

| loss | 作用 |
|---|---|
| representation | root、heading、position、rotation、velocity 的主 flow field |
| contact BCE | 四个脚接触状态 |
| clean root velocity | source space 根位移的正确和平滑 |
| clean joint velocity | 全局关节时间一致性 |
| foot lock | GT contact 帧抑制脚滑 |
| FK consistency | position channel 与 rotation + skeleton 推出的 FK position 一致 |
| controlled entries | Stage 2 在 mask entry 上复现 observation |

Stage 2 的 controlled entries 从主 `L_repr` 排除，改由 control SmoothL1 单独监督，避免同一 entry 接受两套目标。

当前 v1 不开放 contact control，controlled-contact loss 固定为 0。

代码：`train_hy273_raw_flow.py:1311-1412`。

---

## 9. Stage 1：文本到动作训练

当前正在运行 Stage 1：

```text
global steps:             0 -> 200,000
phase:                    text_only
control mask:             全 0
per-GPU batch:            16
GPU:                      8 * A100-80GB
effective global batch:   128
gradient accumulation:    1
optimizer steps/epoch:    167
optimizer:                AdamW
learning rate:            1e-4，固定
weight decay:             0.01
precision:                BF16
global grad clip:         1.0
EMA:                      decay 0.995，每 10 step 更新
text dropout:             0.10
network dropout:          0.0
self-conditioning:        关闭
checkpoint:               每 50,000 step + latest
```

`max_epochs=4000` 是安全上限。真正停止条件是 200,000 optimizer steps。每个 loader epoch 167 step，因此 Stage 1 大约经历 1,198 个 loader epoch。

配置：`configs/redenoise_kimodo_like_stage1.yaml`。

### 9.1 文档生成时的运行快照

截至 2026-07-12 06:28 +08:00：

```text
run:
hy273_redenoise_kimodo_like_stage1_ddp8_20260712_0538

step:                 9,540 / 200,000
total loss:           0.012299
representation flow: 0.006594
contact BCE:          0.049215
FK position error:    0.668 cm diagnostic
gradient norm:        0.057806，clip 前
```

此前连续测量：

```text
throughput:           约 3.24 optimizer steps/s
sample throughput:    约 415 motion clips/s
GPU memory:           每卡约 19.4-19.6 GiB
average GPU utility:  约 77-80%
```

训练会继续推进。实时日志：

```text
/mnt/afs/mogeflow-control/logs/
hy273_redenoise_kimodo_like_stage1_ddp8_20260712_0538.log
```

---

## 10. Stage 2：控制续训

Stage 2 不需要一套新的 motion 数据。它从同一个 clean motion 在 augmentation 后的 source domain 中抽 observation 和 mask。

严格从 Stage-1 step 200,000 checkpoint 续训：

```text
global steps:             200,000 -> 400,000
模型参数:                 继承
optimizer state:          继承
EMA:                      继承
dataloader cursor:        继承
normalization:            继承
resume checkpoint SHA:    必须提供

只允许改变:
  training_phase          text_only -> control
  control modes           none -> frozen v1 set
  control continuous loss 0.00 -> 0.25
```

配置：`configs/redenoise_kimodo_like_stage2_control.yaml`。

### 10.1 控制分布

```text
10%: 无控制
65%: 一个 control pattern
25%: 两个不同 pattern 的 union
```

四类 pattern：

```text
1. root_sparse
   - sparse keyframes
   - 控制 root x/z 和 heading [cos, sin]
   - 不控制 root y

2. root_dense
   - 一段至少占 motion 25% 的连续区间
   - 区间每帧控制 root x/z 和 heading

3. endpoints
   - sparse keyframes
   - 从五点中随机选择非空子集:
       head, left wrist, right wrist, left foot, right foot
   - 控制 3D position
   - 模型内部在同帧附带 full root reference

4. fullpose
   - sparse keyframes
   - 控制 22 个 joint position + full root reference
   - 不直接控制 rotation、velocity、contact
```

`mixed` 从四类中选择两个不同 pattern 并合并 mask。

Sparse K 上限在 Stage 2 从 1 增长到 20。K 用 squared-uniform 抽样，所以后期仍更常见少量关键帧。

代码：`models/raw_motion/hy273_constraints.py:120-238`。

### 10.2 未 mask 的帧为什么也会变化

只有选中的 frame/channel 接受显式 observation，但所有帧共同进入 temporal self-attention。因此受控关键帧会通过全局上下文影响其他帧。

```text
受控 frame/entry:
  尽量精确接近 observation

未直接受控 frame/entry:
  不做 hard overwrite，但根据受控帧、文本、root 和上下文自然过渡
```

未 mask 帧不是与控制无关，而是没有直接 hard target。这正是我们希望得到类似 motion-matching transition 的地方。

---

## 11. 推理与 guidance

默认 ODE32，使用 separated CFG：

```text
joint:    text + control
text:     text + no control
control:  empty text + control
empty:    empty text + no control
```

连续通道：

```text
D_guided = D_empty
         + 3.5 * (D_text    - D_empty)
         + 2.0 * (D_control - D_empty)
```

Contact 是 logits，默认使用 joint branch，不按连续 CFG 公式放大。

每个 ODE step：

```text
shared state
  -> 各 branch 输入侧 overwrite
  -> Root DiT
  -> global-to-local bridge
  -> Body DiT
  -> separated CFG clean estimate
  -> clean estimate 转 ODE velocity
  -> 更新一份不被永久 clamp 的 state
```

最终保存：

```text
raw output:
  exact overwrite 前的模型真实响应
  用于评估模型控制能力

exact-clamped output:
  最终把 observed entries 精确写回
  用于系统交付
```

实现：`sample_hy273_raw.py`。

---

## 12. 可复现性与门禁

```text
资产文件: 42,994
总字节数: 58,953,101,623

asset manifest SHA256:
ff8da22b41f440931c35a9c1a86291c07c134a25e55a677fc9e95534f3113e84

initial model SHA256:
efbdf65ce2adb26a71e87bcae9c9cc55e4260c5523d8495508a0bb669116302a
```

Manifest 包括 motions、captions、split、stats、HYText shards 和 FK skeleton。

Checkpoint 包含：

```text
model
optimizer
EMA
resolved args
global step
exact next epoch/batch cursor
normalizer mean/std/variance_eps
```

如果 Stage-2 checkpoint 不是精确 step 200,000、SHA 不匹配、架构/资产变化，或者 cursor/global batch 不一致，会直接失败，不会静默降级。

代码：`train_hy273_raw_flow.py:1534-1765`。

---

## 13. 关键代码地图

| 内容 | 文件与关键行 |
|---|---|
| dataset、caption、padding | `data/kimodo273_datasets.py:36-207` |
| HY273 slices 与 FK | `models/raw_motion/hy273_slices.py:11-236` |
| root shift、random yaw、normalization | `models/raw_motion/hy273_normalizer.py:35-167` |
| flow timestep 与 state | `models/raw_motion/flow_schedule.py:21-107` |
| cached HYText projection | `models/raw_motion/hytext_cache.py:124-204` |
| root/body backbone | `models/raw_motion/kimodo_like_flow_dit.py:30-255` |
| global-to-local bridge | `models/raw_motion/hy273_root_conditioning.py:22-103` |
| control curriculum | `models/raw_motion/hy273_constraints.py:120-238` |
| model 创建与 x0 guard | `train_hy273_raw_flow.py:464-505` |
| representation loss | `train_hy273_raw_flow.py:541-651` |
| 单步 data-to-loss | `train_hy273_raw_flow.py:1166-1455` |
| DDP、resume、EMA、checkpoint | `train_hy273_raw_flow.py:1534-1925` |
| Stage-1 config | `configs/redenoise_kimodo_like_stage1.yaml` |
| Stage-2 config | `configs/redenoise_kimodo_like_stage2_control.yaml` |
| inference | `sample_hy273_raw.py` |
| control evaluation | `eval_hy273_raw_control.py` |

---

## 14. 最后用人的视角串一次

```text
HumanML3D motion + caption
        |
        v
保留世界朝向与连续性的 Kimodo273 source feature
        |
        +--> 整段随机 heading + 首帧 root-origin augmentation
        |
        +--> normalized clean target x0
        |
        +--> 与 Gaussian noise 插值得到 z_t
        |
        +--> Stage 2 可写入 source-domain observation + mask
        |
        v
Root DiT 先决定人体整体往哪里、以什么朝向运动
        |
        v
预测 root 转成角速度、平移速度和高度
        |
        v
Body DiT 围绕预测 root 生成 pose、rotation、velocity、contact
        |
        v
head 输出 clean x0，主 loss 在 capped velocity-equivalent space 计算
        |
        v
Stage 1 先学习自然 text-to-motion
        |
        v
Stage 2 让同一模型学习 sparse/dense 空间控制
        |
        v
受控帧尽量准确，未直接受控帧通过上下文自然过渡
```
