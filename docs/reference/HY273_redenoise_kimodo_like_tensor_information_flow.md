# HY273 `redenoise_kimodo_like` Tensor Information Flow

> 当前生产配置的逐层 shape 图。训练时每个 DDP rank 使用 `B=16`，`T<=300`；文本长度 `L=128`，hidden size `H=1024`。

## 0. 符号

```text
B = 16                  每卡 batch
G = 8                   DDP ranks
B_global = 128
T <= 300                当前 rank 内 padding 后帧数
L = 128                 Qwen text tokens
D = 273                 HY273
D_cont = 269
D_contact = 4
D_root = 5
D_body = 268
D_local_root = 4
H = 1024
heads = 8
head_dim = 128
FFN_dim = 2048
```

不同 rank 的 `T` 可以不同。DDP 同步参数梯度，不会把各卡 motion tensor 拼成 `[128,T,273]`。

---

## 1. 训练总图

```text
磁盘 motion_i [Ti,273] + caption_i
        |
        | collate on one rank
        v
motion_un [B,T,273]     valid [B,T]     lengths [B]
        |
        | root-origin shift + whole-sequence random yaw
        v
x0_un [B,T,273]         c_dir [B,2]
        |
        +---------------------> control compiler
        |                         obs_un [B,T,273]
        |                         mask   [B,T,273]
        |
        | normalize
        v
x0 [B,T,273]            obs [B,T,273]
        |
        +--> x0_cont [B,T,269] + noise_cont [B,T,269]
        |                          |
        |                          v
        |                    z_cont [B,T,269]
        |
        +--> x0_contact [B,T,4] + contact_aux [B,T,4]
                                   |
                                   v
                             z_contact [B,T,4]
                                   |
             overwrite by obs/mask
                                   v
                             z_imp [B,T,273]
                                   |
                      concat mask [B,T,273]
                                   v
                          model_in [B,T,546]
                                   |
             HYText tokens [B,128,1024]
             pooled cond   [B,1024]
                                   |
                                   v
                             Root DiT
                                   |
                        root_hidden [B,T,1024]
                                   |
                         Linear 1024->5
                                   v
                         root_hat [B,T,5]
                                   |
                stopgrad + FP32 global-to-local
                                   v
                       local_root [B,T,4]
                                   |
          body state [B,T,268] + mask [B,T,273]
                                   |
                  concat 4+268+273=545
                                   v
                          body_in [B,T,545]
                                   |
                             Body DiT
                                   |
                        body_hidden [B,T,1024]
                                   |
                        Linear 1024->268
                                   v
                         body_pred [B,T,268]
                                   |
                    concat root5 + body268
                                   v
                              pred [B,T,273]
                   [0:269] clean continuous x0
                   [269:273] contact logits
                                   |
                            losses -> scalar
                                   |
                       backward + DDP all-reduce
```

---

## 2. Dataset 与 collate

单条数据：

```text
motion_i       [Ti,273], FP32
length_i       scalar
text_i         str
dataset_index  scalar
crop_start     scalar
caption_index  scalar
```

单卡 16 条样本 collate：

```text
T = max(T1...T16)

motion_un       [16,T,273], FP32
valid           [16,T], bool
lengths         [16], int64
texts           list[str], len=16
dataset_indices [16]
crop_starts     [16]
caption_indices [16]
```

Padding：

```text
motion_un[b,length[b]:] = 0
valid[b,:length[b]]     = True
valid[b,length[b]:]     = False
```

代码：`data/kimodo273_datasets.py:141-207`。

---

## 3. 几何增强与 normalization

```text
motion_un [B,T,273]
   |
   | first root x/z [B,1] -> broadcast over T
   v
origin shifted [B,T,273]
   |
   | first_heading [B,2]
   | current_angle [B]
   | target_angle  [B]
   | yaw_delta     [B]
   v
whole-sequence rotation:
  root xyz         [B,T,3]    -> [B,T,3]
  heading          [B,T,2]    -> [B,T,2]
  joint positions  [B,T,22,3] -> [B,T,22,3]
  global rot6d     [B,T,22,6] -> [B,T,22,6]
  joint velocity   [B,T,22,3] -> [B,T,22,3]
  contact          [B,T,4]    -> unchanged
   |
   v
x0_un [B,T,273]
c_dir [B,2]
yaw_delta [B]
```

Normalize：

```text
mean/std [1,1,273]
x0_un [B,T,273] -> x0 [B,T,273]

x0[...,0:269]   normalized continuous
x0[...,269:273] raw contact 0/1
```

代码：`models/raw_motion/hy273_normalizer.py:92-167`。

---

## 4. Control tensor

```text
input:  x0_un [B,T,273]
output: obs_un [B,T,273]
        mask   [B,T,273], bool

obs = normalize(obs_un) [B,T,273]
```

Stage 1：

```text
obs_un = zeros [B,T,273]
mask   = False [B,T,273]
```

Stage 2 只设置被控制的 frame/channel；shape 始终不变。

代码：`models/raw_motion/hy273_constraints.py:145-238`。

---

## 5. Flow state

Timestep：

```text
t_raw [B] ~ N(-0.8,0.8^2)
t = sigmoid(t_raw) [B]
t_view [B,1,1]
```

Continuous：

```text
x0_cont    [B,T,269]
noise_cont [B,T,269]

z_cont = t*x0_cont + (1-t)*noise_cont
z_cont [B,T,269]

v_target = x0_cont-noise_cont
v_target [B,T,269]
```

Contact auxiliary：

```text
x0_contact [B,T,4]
contact_aux [B,T,4] ~ Uniform(0,1)

z_contact = t*x0_contact + (1-t)*contact_aux
z_contact [B,T,4]
```

Overwrite 与拼接：

```text
mask_cont    [B,T,269]
mask_contact [B,T,4]

z_cont_imp    [B,T,269]
z_contact_imp [B,T,4]

concat -> z_imp [B,T,273]
concat z_imp + mask.float -> model_in [B,T,546]
```

代码：`models/raw_motion/flow_schedule.py:71-107`。

---

## 6. HYText flow

```text
texts list[str], len=B
   |
   v
Qwen cache ctxt_raw [B,128,4096]
ctxt_len            [B]
CLIP cache vtxt_raw [B,1,768]
```

Token projection：

```text
[B,128,4096]
 -> Linear(4096,1024)
text_tokens [B,128,1024]
```

Pooled projection：

```text
vtxt_raw[:,0] [B,768]
 -> Linear(768,1024)
 -> SiLU
 -> Linear(1024,1024)
text_pooled [B,1024]
```

Mask 与全局条件：

```text
padding_mask [B,128]

t [B]
 -> sinusoidal [B,256]
 -> MLP 256->1024
t_embed [B,1024]

c_dir [B,2]
 -> MLP 2->1024
dir_embed [B,1024]

cond = t_embed + dir_embed + text_pooled
cond [B,1024]

motion_pos_ids [B,T,1]
```

代码：

- `models/raw_motion/hytext_cache.py:124-204`
- `models/raw_motion/kimodo_like_flow_dit.py:184-206`

---

## 7. Root DiT

输入投影：

```text
model_in [B,T,546]
 -> Linear(546,1024)
root_motion [B,T,1024]
```

Self-conditioning 当前关闭。打开时：

```text
x_self_cond_root [B,T,5]
 -> Linear(5,1024)
 -> add root_motion
```

### 7.1 一个 double-stream block

输入：

```text
motion [B,T,1024]
text   [B,128,1024]
cond   [B,1024]
```

AdaLN：

```text
cond [B,1024]
 -> Linear(1024,1024*3*2)
 -> attention shift/scale/gate [B,1,1024]
 -> FFN shift/scale/gate       [B,1,1024]
```

Joint attention：

```text
concat motion + text along token axis
joint       [B,T+128,1024]
joint_valid [B,T+128]
joint_pos   [B,T+128,1]

Q [B,T+128,1024] -> reshape -> [B,8,T+128,128]
K [B,T+128,1024] -> reshape -> [B,8,T+128,128]
V [B,T+128,1024] -> reshape -> [B,8,T+128,128]

attention score logical shape [B,8,T+128,T+128]
attention out                 [B,8,T+128,128]
reshape                       [B,T+128,1024]

split:
motion_out [B,T,1024]
text_out   [B,128,1024]
```

Motion/Text 各自 SwiGLU：

```text
[B,N,1024]
 -> gate/up [B,N,2048]
 -> multiply [B,N,2048]
 -> down [B,N,1024]
```

Root 共执行 3 个 double-stream block，shape 不变。

### 7.2 Six single-stream blocks

```text
concat motion + text
x [B,T+128,1024]

每层:
Q/K/V [B,8,T+128,128]
attention output [B,T+128,1024]
SwiGLU 1024->2048->1024
output [B,T+128,1024]

共执行 6 层
```

最后只取 motion tokens：

```text
x[:,:T] [B,T,1024]
 -> Linear(1024,5)
root_prediction_raw [B,T,5]
```

代码：

- `models/codeflow/dit_blocks.py:261-381`
- `models/codeflow/dit_blocks.py:607-768`
- `models/raw_motion/kimodo_like_flow_dit.py:208-223`

---

## 8. Global-to-local root bridge

```text
root_prediction_raw [B,T,5], BF16
lengths             [B]
```

训练时：

```text
stopgrad -> bridge_root [B,T,5]
cast/unnormalize -> root [B,T,5], FP32

pos     [B,T,3]
heading [B,T,2]
```

相邻帧差分：

```text
heading pairs -> dot/cross [B,T-1]
atan2(cross,dot)*30       [B,T-1]

pos[:,1:]-pos[:,:-1]     [B,T-1,3]
*30 FPS                   [B,T-1,3]
```

组装与归一化：

```text
local[...,0] yaw velocity [B,T]
local[...,1] root vx      [B,T]
local[...,2] root vz      [B,T]
local[...,3] root y       [B,T]

local [B,T,4], FP32
 -> normalize with mean/std [1,1,4]
 -> cast model dtype
local_root [B,T,4], BF16
```

代码：`models/raw_motion/hy273_root_conditioning.py:63-103`。

---

## 9. Body DiT

```text
state       = model_in[...,0:273]   [B,T,273]
body_state  = state[...,5:273]      [B,T,268]
mask        = model_in[...,273:546] [B,T,273]
local_root                           [B,T,4]

concat 4+268+273
body_in [B,T,545]
 -> Linear(545,1024)
body_motion [B,T,1024]
```

Body 使用同样的 3 double + 6 single：

```text
double joint tokens [B,T+128,1024]
double Q/K/V        [B,8,T+128,128]

single tokens       [B,T+128,1024]
single Q/K/V        [B,8,T+128,128]

取前 T 个 motion tokens
body_hidden [B,T,1024]
```

Body 重新使用原始 projected text tokens，不接收 Root block 内更新后的 text stream。

输出：

```text
body_hidden [B,T,1024]
 -> Linear(1024,268)
body_prediction [B,T,268]

concat root5 + body268
prediction [B,T,273]

prediction[...,0:5]     root clean x0       [B,T,5]
prediction[...,5:269]   body continuous x0  [B,T,264]
prediction[...,269:273] contact logits      [B,T,4]
```

代码：`models/raw_motion/kimodo_like_flow_dit.py:228-255`。

---

## 10. Loss flow

```text
pred_cont      [B,T,269]
contact_logits [B,T,4]

x0_hat_cont = pred_cont [B,T,269]

t                  [B]
t_view             [B,1,1]
denom=clamp(1-t,0.05) [B,1,1]

v_pred = (x0_hat_cont-z_cont_imp)/denom [B,T,269]
v_gt   = x0_cont-noise_cont              [B,T,269]

valid               [B,T]
mask_cont           [B,T,269]
valid_cont_unmasked [B,T,269]
```

这里必须区分两个概念：

```text
model prediction type = clean x0
main representation loss space = velocity

也就是：网络直接输出 [B,T,269] clean continuous x0，
训练 loss 再把该输出换算成 [B,T,269] velocity 后与 v_gt 比较。
```

主 block：

```text
v_pred/v_gt[...,0:3]     root xyz    [B,T,3]   -> masked MSE -> scalar L_root
v_pred/v_gt[...,3:5]     heading     [B,T,2]   -> masked MSE -> scalar L_heading
v_pred/v_gt[...,5:71]    joint pos   [B,T,66]  -> masked MSE -> scalar L_joint_pos
v_pred/v_gt[...,71:203]  global rot  [B,T,132] -> masked MSE -> scalar L_rot6d
v_pred/v_gt[...,203:269] joint vel   [B,T,66]  -> masked MSE -> scalar L_velocity

representation_scale = 0.09397019716051493

L_flow = representation_scale * (
    10/35 * L_root
  +  2/35 * L_heading
  + 10/35 * L_joint_pos
  + 10/35 * L_rot6d
  +  3/35 * L_velocity
)

L_flow scalar []
```

Contact：

```text
contact_logits [B,T,4]
x0_contact     [B,T,4]
valid_contact  [B,T,4]
BCE            [B,T,4]
masked mean -> scalar
```

几何辅助项：

```text
contact_prob = sigmoid(logits) [B,T,4]
x0_hat = concat                [B,T,273]
denormalize                    [B,T,273], FP32

root velocity loss  -> scalar
joint velocity loss -> scalar
foot lock loss      -> scalar
FK loss             -> scalar
control loss        -> scalar
```

最终：

```text
L_total = 1.0*L_flow
        + 0.1*L_contact
        + 0.01*L_clean_root_velocity
        + 0.01*L_clean_joint_velocity
        + 0.01*L_foot_lock
        + warmup_to_0.07*L_FK

Stage 1 control weights are zero.

L_total scalar []
 -> backward
 -> local parameter gradients
 -> DDP all-reduce over 8 ranks
 -> AdamW step
 -> 每 10 step 更新 EMA
```

代码：`train_hy273_raw_flow.py:1166-1455`。

---

## 11. DDP 视角

```text
rank0: [16,T0,273] -> scalar loss0 -> grads0
rank1: [16,T1,273] -> scalar loss1 -> grads1
...
rank7: [16,T7,273] -> scalar loss7 -> grads7

DDP: grads = mean(grads0...grads7)

effective global batch = 16 * 8 * grad_accum1 = 128
```

当前 trainable parameters：`387,632,913`。

---

## 12. ODE32 + separated CFG 推理

推理 batch 记作 `S`。

### 12.1 初始化

```text
observed_un [S,T,273]
mask        [S,T,273]
lengths     [S]
c_dir       [S,2]
texts       list[str], len=S

z_cont       [S,T,269]
contact_noise[S,T,4]
contact_aux  [S,T,4]
state        [S,T,273]
ODE grid     [33]
```

### 12.2 一个有控制的 ODE step

```text
t  [S]
dt scalar

controlled_state [S,T,273]
zero_mask        [S,T,273]

input_free    = concat(state,zero_mask)        [S,T,546]
input_control = concat(controlled_state,mask)  [S,T,546]
```

四个 branch 一次拼 batch：

```text
joint, text, control, empty

branch_input [4S,T,546]
branch_t     [4S]
branch_c_dir [4S,2]
branch_valid [4S,T]
branch_text  list[str], len=4S
```

模型内所有 B 替换成 4S：

```text
pred_all [4S,T,273]

chunk ->
pred_joint   [S,T,273]
pred_text    [S,T,273]
pred_control [S,T,273]
pred_empty   [S,T,273]
```

CFG：

```text
pred_guided = pred_empty
            + 3.5*(pred_text-pred_empty)
            + 2.0*(pred_control-pred_empty)

guided continuous [S,T,269]
joint contact logit[S,T,4]
```

Update：

```text
x0_hat_cont [S,T,269]
v_cont      [S,T,269]
z_cont_next [S,T,269]

contact_prob     [S,T,4]
contact_aux_next [S,T,4]
```

重复 32 次。

无 control 但有 text CFG 时只扩成 `2S`；无 CFG 时保持 `S`。

### 12.3 输出

```text
raw_motion             [S,T,273]
exact_clamped_motion   [S,T,273]
final_clean_prediction [S,T,273]
branch diagnostics     dict[str,[S,T,273]]
```

代码：`sample_hy273_raw.py:115-322`。

---

## 13. Shape 速查

```text
dataset motion                   [B,T,273]
valid                            [B,T]
lengths                          [B]
c_dir                            [B,2]
t                                [B]
obs/mask                         [B,T,273]
noise continuous                 [B,T,269]
contact auxiliary                [B,T,4]
z_imp                            [B,T,273]
model_in                         [B,T,546]

Qwen raw                         [B,128,4096]
CLIP pooled raw                  [B,1,768]
text tokens                      [B,128,1024]
text pooled                      [B,1024]
global cond                      [B,1024]

root input hidden                [B,T,1024]
root/text joint tokens           [B,T+128,1024]
root attention Q/K/V             [B,8,T+128,128]
root output                      [B,T,5]
local root                       [B,T,4]

body input                       [B,T,545]
body hidden                      [B,T,1024]
body/text joint tokens           [B,T+128,1024]
body attention Q/K/V             [B,8,T+128,128]
body output                      [B,T,268]

final prediction                 [B,T,273]
continuous x0                    [B,T,269]
contact logits                   [B,T,4]
total loss                       scalar

controlled CFG model input       [4S,T,546]
controlled CFG model output      [4S,T,273]
generated motion                 [S,T,273]
```

---

## 14. 关键代码

| 流程 | 文件与关键行 |
|---|---|
| dataset/collate | `data/kimodo273_datasets.py:141-207` |
| augmentation/normalization | `models/raw_motion/hy273_normalizer.py:92-167` |
| flow state | `models/raw_motion/flow_schedule.py:71-107` |
| HYText | `models/raw_motion/hytext_cache.py:124-204` |
| Root/Body | `models/raw_motion/kimodo_like_flow_dit.py:168-255` |
| attention blocks | `models/codeflow/dit_blocks.py:171-381` |
| FrameMotionTextDiT | `models/codeflow/dit_blocks.py:607-768` |
| root bridge | `models/raw_motion/hy273_root_conditioning.py:63-103` |
| loss | `train_hy273_raw_flow.py:1166-1455` |
| ODE/CFG | `sample_hy273_raw.py:115-322` |
