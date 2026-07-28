# HY273 Kimodo-Context 多任务扩展方案

> 目标：在不改变现有 Kimodo-like 硬控制语义的前提下，从头训练一个同时支持 T2M、硬控制和 MotionFix 文本编辑，并能继续扩展 reaction 等任务的 HY273 raw-space rectified-flow 模型。
>
> 状态：`v2-design-r8-R11-frozen`，30 FPS/no-resample 合同和多任务设计均已通过 `gpt-5.6-sol / max` 对抗审核；R11 closure 为 `Critical 0 / High 0 / Medium 0`，可进入实现与 smoke。正式长训仍受 non-regression baseline artifact 门禁约束。详细数据合同见 [HY273_multitask_data_dataloader_protocol_CN.md](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_data_dataloader_protocol_CN.md)。基线代码固定为 `mogeflow_kimodo_like` commit `e85cd6ea2b8be5bba18091fe9a2a561a7c50c79a`；成功 Stage-1/complete Stage-2 的 resolved config SHA256 分别为 `da09bc3d0a7f1ee2559d75162c877f7c317d59ea4fa47bcf1b1b6e308628e2fa`、`b194ad7ff5c293d4022f1e4b2afff949adbafcc5c74683b34e7bbc4cc5510b2e`。
>
> 训练日程更新：本文件的总览已同步为 `A 0-200K -> B1 200-250K -> B2 250-400K -> C 400-500K`，editing 从 B2 开始低比例进入；精确概率、LR、resume 和 gate 仍以 [HY273_multitask_model_training_plan_CN.md](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_model_training_plan_CN.md) v1.3 为唯一权威来源。

## 1. 结论先行

### 1.1 `source motion` 不能直接当 `obs`

这是本方案最重要的边界。

```text
obs / hard_mask:
  含义 = 输出必须满足的硬约束
  用法 = 每个 ODE step 的 denoiser 输入前，在指定 frame/channel overwrite
  结果 = 被 mask 的值被要求精确复现

source_motion:
  含义 = 供模型理解、保留和修改的参考动作
  用法 = 独立的 noise-free context branch
  结果 = 模型可根据编辑文本保留一部分、修改另一部分
```

如果把完整 source motion 当成 `obs` 并令 `mask=1`，每个 ODE step 都会把 source 写回去，编辑模型只能复制 source；如果令 `mask=0`，source 又根本没有进入模型。因此 MotionFix 的 source 必须是第三条独立条件通路，不能复用 hard-control tensor。

### 1.2 推荐的统一条件系统

模型需要同时看到三类互不替代的信息：

```text
1. task intent       这次要生成、编辑还是 reaction
2. source context    参考谁、哪些 frame 是 source-aware 的
3. hard constraint   哪些输出 frame/channel 必须精确满足
```

具体采用：

- 全局 `task_id`：`GENERATE / EDIT / REACTION / ...`，表示 source-target 语义关系；
- 仅用于采样/评估路由的 `capability_id`：`T2M / KIMODO_CONTROL / MOTION_EDIT / ...`；
- 逐帧 meta-operation：`PRESERVE / GENERATE / EDIT`；
- `source_role_id`：`SELF_REFERENCE / OTHER_ACTOR / ...`；
- 独立 `source_motion` temporal-fusion branch；
- 现有 `observed_motion + motion_mask` overwrite 原样保留。

这样，即使 EDIT 和 REACTION 都有一条 source motion，模型也能通过 `task_id + source_role_id` 区分“修改自己”和“响应另一个人”；Kimodo control 只由 hard mask 和 overwrite 定义，不创建 `CONTROL` 或 `EDIT_CONTROL` task，因此 CFG residual 不混入 task-embedding 变化。

### 1.3 对 UMO 的取舍

采用：

- `[preserve]/[generate]/[edit]` 三种逐帧意图；
- noise-free source motion 独立编码；
-按时间对齐的 Temporal Fusion；
- 统一模型内的多任务 replay。

不采用：

- 不把 Kimodo 控制改成纯文本序列化控制；
- 不用 UMO 的 learned-preserve 代替现有 exact overwrite；
- 不把我们的 x0 prediction 改回 velocity prediction；
- 第一版不使用训练时由 target/source 差异得到、推理时又不可获得的 oracle edit mask；
- 不把 source motion 拼进 ODE state，也不覆盖 `root_for_body`。

## 2. 当前成功基线中不能破坏的契约

### 2.1 数据表示

每帧 HY273：

```text
[0:3]     smooth root xyz
[3:5]     global heading [cos(yaw), sin(yaw)]
[5:71]    22 joint positions
[71:203]  22 global rotations, cont6d
[203:269] 22 global joint velocities
[269:273] 4 foot contacts, raw 0/1
```

连续通道使用 HumanML3D K273 训练 stats；contact 永远不做 z-score。当前 normalizer 在 [hy273_normalizer.py](/mnt/afs/mogeflow_kimodo_like/models/raw_motion/hy273_normalizer.py:92) 明确保留 contact 原值。

### 2.2 Root/Body 两阶段去噪

当前模型主输入固定为：

```text
state [B,T,273] + hard_mask [B,T,273]
                   |
                   v
model_in [B,T,546]
```

Root DiT 先预测 target clean root `[B,T,5]`，再由预测 root 构造 local-root `[B,T,4]`，Body DiT 预测其余 `[B,T,268]`。Body 只能消费 predicted target root，不能改为 source root 或稀疏 GT root。实现见 [kimodo_like_flow_dit.py](/mnt/afs/mogeflow_kimodo_like/models/raw_motion/kimodo_like_flow_dit.py:168)，特别是 root/body bridge 的 [225 行](/mnt/afs/mogeflow_kimodo_like/models/raw_motion/kimodo_like_flow_dit.py:225)。

### 2.3 硬控制

```text
z_imp = z_t * (1 - hard_mask) + hard_obs * hard_mask
model_in = concat(z_imp, hard_mask)
```

现有 compiler 支持：

- sparse/dense root path；
- sparse full-body positions；
- 手脚末端 position + global rotation；
- sparse foot contact；
- 两种 pattern 的组合。

训练 compiler 见 [hy273_constraints.py](/mnt/afs/mogeflow_kimodo_like/models/raw_motion/hy273_constraints.py:186)。采样时 overwrite 只发生在 joint/control denoiser input，ODE state 本身保持 unclamped，见 [sample_hy273_raw.py](/mnt/afs/mogeflow_kimodo_like/sample_hy273_raw.py:158)。最终同时输出 raw sample 和 exact-clamped sample。

### 2.4 预测与损失

当前模型是：

```text
网络输出:
  [0:269]   clean x0 continuous prediction
  [269:273] contact logits

主 representation loss:
  在 velocity space 比较由 x0 prediction 换算得到的 v_pred 和 v_target
```

所以“x0 prediction”和“velocity-space loss”同时成立。不能因为 UMO 使用 velocity prediction 就改变这条已经验证成功的契约。当前 loss 组装见 [train_hy273_raw_flow.py](/mnt/afs/mogeflow_kimodo_like/train_hy273_raw_flow.py:1334)。

可复现基线不是 CLI 默认值，而是以下只读 resolved config：

```text
Stage-1:
  /mnt/afs/mogeflow-control/checkpoints/t2m/
  hy273_redenoise_kimodo_like_stage1_ddp8_20260712_0538/config_resolved.json
  sha256 da09bc3d0a7f1ee2559d75162c877f7c317d59ea4fa47bcf1b1b6e308628e2fa
  checkpoint model/step_00200000.pt
  sha256 5cc0413201a5652c119a822fc44c13833c11df79ec34e560eac5abd66a21a55d

complete Stage-2:
  /mnt/afs/mogeflow-control/checkpoints/t2m/
  hy273_redenoise_kimodo_complete_stage2_control_ddp8_20260713_0547/config_resolved.json
  sha256 b194ad7ff5c293d4022f1e4b2afff949adbafcc5c74683b34e7bbc4cc5510b2e
  checkpoint model/step_00400000.pt
  sha256 d5f00ec15888e1dc3ca9f8c38c8ef436ec6524397ae0257a01fc48ce3542b2f4

complete control source YAML（唯一允许的 Stage-B 基线）:
  /mnt/afs/mogeflow-control/configs/redenoise_kimodo_complete_stage2_control.yaml
  sha256 f53b22b779262119d93a9fd76e988897cd628e03cb84130e3d1afe1ee712be40
```

名字相近的 `configs/redenoise_kimodo_like_stage2_control.yaml` 缺少 complete endpoint-rotation/contact 控制集，不能用作 Stage B1/B2 合同。结构化门禁模板见 [HY273_multitask_nonregression_baseline_v1.json](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_nonregression_baseline_v1.json)，当前 SHA256 为 `2332badc52da7c603c8550e43e432be43dd0f858e7bc88e62df6160c59db47c0`；在其中逐 pattern 指标、raw/EMA 来源、阈值和 paired CI 全部填实并将 `status` 改为 `ready` 前，不启动正式 Stage A/B1/B2/C 长训。

新训练启动器必须把这两份合同复制到 run directory，并对 loss、control modes、normalizer、text cache、自条件和 EMA 字段逐项 fail-closed，而不是只比较代码 commit。

## 3. UMO 论文给出的有效证据

参考论文：Cong et al., **UMO: Unified In-Context Learning Unlocks Motion Foundation Model Priors**, arXiv:2603.15975v1, 2026-03-16，当前仍是 preprint。

### 3.1 三种原子操作

UMO 将每个 target frame 相对 source 的关系归纳为：

```text
PRESERVE: source frame 应保留
GENERATE: 没有 source，重新生成
EDIT:     基于 source 修改
```

其消融显示，把 EDIT 合并进 GENERATE 后，MotionFix full-set AvgR 从 `1.75` 变差到 `1.97`，并同时损害 inpainting；因此“有 source 的修改”和“无 source 的生成”不能只靠 source 是否为零让网络自行猜测。

### 3.2 Temporal Fusion

UMO 将 source context 投影成与 noisy motion token 同 shape 的 frame tokens，再逐帧相加。其 keyframe-infilling 消融中，Temporal Fusion 的额外参数仅 `0.207M`，并且在 preserve MPJPE、FID、延迟上优于 sequential concat、AdaLN 和 ControlNet。

这与我们的需求一致：source 和 target 在时间上对齐，不能先池化成一个 global vector；也没有必要把序列扩成 `2T`。

### 3.3 UMO 不能直接照搬的地方

- UMO 的 motion 是 201D；我们是带 contact、global position、global rotation 和 velocity 的 HY273。
- UMO 用 velocity prediction；我们当前成功模型用 x0 prediction。
- UMO 的 geometric control 主要序列化到文本；我们的 exact channel-wise overwrite 更适合精确控制。
- UMO 的 P/G/E 是 whole-frame 级，论文也明确承认不支持 part-level intent；我们的硬控制已经是 frame/channel 级。
- UMO 公开 GitHub `Oliver-Cong02/UMO` 在 commit `46a9378c2897b6f0411f5b98e4d3fe3940da0b0f` 仍只有 README 和 teaser，代码标记为 coming soon。因此本方案参考的是论文公式和消融，不声称核对了未发布实现。

## 4. 统一输入协议

### 4.1 逻辑接口

```text
ConditionBatch
  train_stream_id        [B]                int64, scheduler/replay only; never embedded
  task_id                [B]                int64
  capability_id          [B]                int64, routing/eval only
  text                   list[str]          len=B
  text_encoding_profile  list[str]          len=B
  frame_gauge_dir        [B,2]

  hard_obs               [B,Tt,273]         normalized after compiler
  hard_mask              [B,Tt,273]         bool

  source_motion          [B,K,Ts,273]        unnormalized at dataset boundary
  source_present         [B,K]               bool
  source_time_valid      [B,K,Ts]            bool
  source_value_mask      [B,K,Ts,273]        bool
  source_role_id         [B,K]               int64

  target_motion          [B,Tt,273]          only training/evaluation
  target_valid           [B,Tt]              bool
  target_op_id           [B,Tt]              P/G/E
  requested_target_len   [B]                 int64
  source_native_lengths  [B,K]               int64
  target_to_source_time_map optional [B,K,Tt], source-frame coordinates
```

第一版只实现 `K=1`，但 public dataclass/collate 保留 source slot 和独立 `Ts/Tt`。`source_present`、`source_time_valid`、`source_value_mask`、`target_valid` 不得合并：它们分别表示 slot 是否存在、source padding、source 哪些值可见、target loss/attention padding。

### 4.2 任务映射

| capability | `train_stream_id` | `task_id` | source | target op | hard mask | text | 目标 |
|---|---|---|---|---|---|---|---|
| T2M | HML_MIXED | GENERATE | 无 | G | 全 0 | motion description | 从头生成 |
| Kimodo control | HML_MIXED | GENERATE | 无 | G | 现有 compiler | description | 精确受控生成 |
| Motion editing | MOTION_EDIT | EDIT | source motion | E | 默认全 0 | edit instruction | 修改 source 得到 target |
| Editing + control | MOTION_EDIT | EDIT | source motion | E | 现有 compiler | edit instruction | 修改且满足硬约束 |
| Inpainting | 专用 stream | GENERATE 或专用 task | partial self | P/G | exact known values 可 hard clamp | description | 补全未知 frame |
| Reaction | REACTION | REACTION | other actor | G | 默认全 0 | interaction text | 生成 responder |

### 4.3 Fail-closed 校验

公开 sampler 必须拒绝以下组合：

```text
EDIT      + source_missing
REACTION  + source_missing
capability=T2M/KIMODO_CONTROL + unexpected source_present
同一 CFG 组合内 train_stream_id/task_id/capability_id/frame gauge 发生变化
hard_mask + hard_obs shape/domain 不一致
source/target 使用不同 normalizer 或不同 frame_policy/output gauge draw
independent_sequence_frame 被错误要求使用相同 yaw delta
```

CFG 内部为了计算 dropped-condition branch 产生的空 text/source 是内部状态，不经过公开任务校验。

## 5. MotionFix 数据准备

### 5.1 已有数据事实

当前可用 K273 文件：

```text
/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22
```

当前 conversion semantic audit 已证明每个单独 motion 的 273 个 channel 与 Kimodo 定义一致；总计 `13,460` 个 motion 文件、`1,489,149` 帧。

时间轴已经单独复核：HumanML3D K273 的 `26,846` 条资产和 MotionFix K273 的 `13,460` 条资产全部是 30 FPS，两个 `manifest.jsonl` 与 conversion summary 的文件数/总帧数一致；`272 -> HY201 -> K273` 全链路保持 `T`。因此联合训练直接使用一个 `dt=1/30s` 合同，不做跨数据 temporal resampling。传统 HumanML3D 263D 的 20 FPS 约定不属于当前 MotionStreamer272 lineage；13,423 条非镜像对照固定使用 `/mnt/afs/HumanML3D/index.csv`（SHA256 `80226910d7259655ab5f411fd807b7651037341b4806fde025946a74d62b1065`），避免误用残缺副本。完整证据见 [HY273_30fps_lineage_audit_v1.json](/mnt/afs/mogeflow-control/outside_doc/HY273_30fps_lineage_audit_v1.json)，对抗审核见 [HY273_30fps_plan_review_R6.md](/mnt/afs/mogeflow-control/outside_doc/HY273_30fps_plan_review_R6.md)，结论为 **GO**。

当前结论只适用于以上 pinned 资产。现有通用 HY201 -> K273 converter 的 `--fps` 不会认证源 FPS 或执行重采样；未来接入第三套数据时必须由 unified manifest preflight 验证 `source_fps`，非 30 FPS 默认拒绝，或先在 root translation + local SO(3) 物理域重采样并重新提取全部 K273 channel。

官方 split 对应的 pair 数：

| split | 总 pair | source/target 等长 | 不等长 | 空 instruction |
|---|---:|---:|---:|---:|
| train | 5,387 | 5,047 | 340 | 0 |
| val | 330 | 313 | 17 | 0 |
| test | 1,013 | 952 | 61 | 0 |

训练集中共有 `4,845` 条唯一 instruction。

### 5.2 新建 pair manifest

现有 K273 `manifest.jsonl` 是逐 motion 的，不包含 edit instruction。不要复制 motion 文件；从官方 annotation/split 和两个 K273 manifest 构造统一 JSONL row。最小示意如下，完整 schema 以数据协议第 4 节为准：

```json
{
  "uid": "motionfix:000362",
  "dataset": "motionfix_k273",
  "split": "train",
  "task_capabilities": ["motion_edit", "motion_edit_with_control"],
  "source_motion": {
    "path": "/abs/.../train/000362_source.npy",
    "hy201_path": "/abs/.../train/000362_source.npy",
    "base_motion_id": "007102",
    "timestamp_sec": [16.35, 20.35],
    "frames": 120,
    "coordinate_frame": "per_sequence_hml_canonical_then_raw_k273"
  },
  "target_motion": {
    "path": "/abs/.../train/000362_target.npy",
    "hy201_path": "/abs/.../train/000362_target.npy",
    "base_motion_id": "005808",
    "timestamp_sec": [15.95, 19.95],
    "frames": 120,
    "coordinate_frame": "per_sequence_hml_canonical_then_raw_k273"
  },
  "texts": [{
    "value": "raise your left hand higher and wave wider in the end",
    "kind": "relative_edit_instruction",
    "encoding_profile": "hytext_relative_edit_v1"
  }],
  "pair": {
    "frame_policy_id": "independent_sequence_frame_v1",
    "framewise_aligned": false,
    "shared_world_frame": false,
    "output_gauge_policy": "shared_target_yaw_phi_v1",
    "default_time_relation": "normalized_progress"
  },
  "provenance": {"conversion_commit": "...", "sha256": "..."}
}
```

这里只是模型侧摘要，完整 manifest 必须按数据协议把 K273 与 HY201 写成两个独立 AssetRef，并固定 source、target、instruction、split、base motion ID、timestamps、FPS、representation/frame contract、文件 SHA 和 converter provenance。builder 对 `000000`、`000362` 做正向 golden join；`splits.json` 中缺失的 `train/006722` 是拒绝 golden row，进入 `rejected.jsonl`，不能产生空训练 row。

### 5.3 当前 pair 的真实坐标 lineage 与几何增强

K273 extraction 最后一步的确是 `to_canonicalize=False`，但 MotionFix 上游已对 source/target **分别执行 HumanML3D canonicalization**：

```text
MotionFix raw pair
  -> source/target independent HML origin+heading canonicalization
  -> HY201
  -> K273 raw extraction without another canonicalization
```

所以当前 pair 不是 shared-world 数据，不能假装原始 source-target 世界位移仍然存在；也不能为两边独立采两个随机 yaw。正确 policy 是“各自去 origin，但共享一个最终 gauge”：

```text
source = root_origin_shift(source)
target = root_origin_shift(target)

sample one phi ~ Uniform(-pi, pi)
delta_source = phi - source_heading0
delta_target = phi - target_heading0

source = yaw_rotate(source, delta_source)
target = yaw_rotate(target, delta_target)
frame_gauge_dir = [cos(phi), sin(phi)]
```

两个 delta 可以不同，但随机变量 `phi` 只有一个。该操作保留各自动作相对自己首帧的轨迹/转向语义，并把两者放入同一个训练 gauge；它不声称恢复已经丢失的原始 world relation。未来 reaction/shared-scene 数据必须走另一条 `shared_world_frame` policy，对所有 actor/source/target 使用完全相同的 SE(2) 变换。

实现/校验词汇必须分离：当前 pair 的 `frame_policy_id` 相同，source/target 的 `physical_transform_group_id` 不同，`output_gauge_draw_id/phi` 相同，各 slot 的 `applied_yaw_delta` 可不同。只验证相同 policy、相同 `phi` 和变换后首帧 heading；不能用“same pair transform”要求两个 delta 相等。

### 5.4 Normalization

分两种情况：

```text
续训已有 checkpoint:
  必须继续使用成功 HumanML3D stats，不能中途换尺度

本方案从头训练 multitask-v2:
  从 step 0 起使用固定的 train-only / target-only / task-weighted joint stats
  统计 HumanML3D target windows + MotionFix target
  不把 MotionFix source 重复计入 output moments
```

source 使用同一 normalizer 输入 context branch，但单独报告 z-score coverage。contact 继续保持 raw 0/1。joint stats 的 domain 权重必须等于预注册的 steady-state target-domain/train-stream mix并写入 stats manifest；Stage A/B1/B2/C 全程固定同一 SHA。

### 5.5 原生不等长策略

第一版 public interface 原生保留 `Ts != Tt`，source 与 target 分开 pad/mask。这里的长度差发生在同一个 30 FPS timebase 上，表达 duration edit/端点取整，不是 source-target 或跨数据 FPS 不一致。MotionFix v1 最大约 150 帧，小于 `max_T=300`，因此不做 temporal crop。训练时 target length 已知；推理时必须由调用者显式提供 `requested_target_len`，或以后接独立 length predictor。

source context adapter 可在 hidden token 域按 normalized progress 映射到 target query 长度，并同时输入 `Ts/Tt` 或 `log(Tt/Ts)`；这不是改写 source K273。禁止：

```text
静默截断到 min(Ts,Tt)
source/target 独立 random crop
直接线性插值完整 273D/cont6d
```

若某个替代模型强制同 T，只能从 HY201 local rotation/root translation 或更早物理表示重采样：rotation 用 SO(3) SLERP，随后重新计算 FK、heading、smooth root、positions、global cont6d、velocity、contact。使用 GT `Tt` 的评估必须标为 `conditioned on requested target length`，不能声称模型自主学会 duration。

### 5.6 文本 cache

继续离线使用 Qwen3-8B token features + CLIP ViT-L/14 pooled features，但绝对 caption 与相对 instruction 使用独立、版本化的 encoding profile：

```text
T2M:  "Generate a motion: {description}"
EDIT: "Edit the source motion according to this instruction: {instruction}"
```

cache key 必须包含：

```text
encoder identity + prompt_template_version + text_encoding_profile + normalized text
```

不能只用裸文本 SHA1，否则同一句话在 description/instruction 语境下会错误共用。每个 profile 各有 empty row；task embedding 仍显式存在，不能只靠 prompt prefix 猜任务。

### 5.7 数据 P0 门禁

单 motion semantic audit 还不足以证明 edit pair 正确。训练前必须新增：

1. 6,730 pair coverage、`006722` reject、split/UID/hash 可复现；
2. `coordinate_frame=per_sequence_hml_canonical_then_raw_k273` 与 `pair_shared_world=false` fail-closed；
3. one-phi transform 后两边 `frame_gauge_dir` 一致，且无独立 yaw RNG；
4. native `Ts/Tt`、独立 padding mask、no-crop 路径通过 batch/replay 测试；
5. 32 条 instruction/source/target 人工可视化，foot contact/ground audit 通过；
6. official MotionFix evaluator 所需 conversion/gauge/meta 完成数值 roundtrip并钉住 evaluator commit/checkpoint/protocol。

前 1-5 项是训练数据门禁。第 6 项是发布 official MotionFix benchmark 数字的门禁；它不阻塞标注清楚的内部 K273 pilot，但未通过时不得把内部 retrieval/geometry metric 冒充 official result。

## 6. 模型设计

建议名称：`HY273KimodoContextFlow`。

### 6.1 不改主输入

```text
hard-controlled noisy state  [B,Tt,273]
hard mask                    [B,Tt,273]
concat                        [B,Tt,546]
```

保留 root/body 的现有 `input_proj` shape，source context 使用单独 projection。这既符合 UMO Temporal Fusion，也使 source 缺失时能严格退化为当前模型。

### 6.2 Source context encoder

第一版 `K=1`：

```text
source_un          [B,K,Ts,273]
source_value_mask  [B,K,Ts,273]
source_time_valid  [B,K,Ts]
source_present     [B,K]
       |
same frozen normalizer; unknown value -> 0 after normalize
       v
concat(source_value, source_value_mask.float()) [B,K,Ts,546]

target_op_id       [B,Tt]    -> Embedding(3,H)  -> op_token   [B,Tt,H]
source_role_id     [B,K]     -> Embedding(R,H)  -> role_token [B,K,1,H]
task_id            [B]       -> Embedding(Q,H)  -> task_token [B,1,H]
length_features    [B,K,3]   -> Linear(3,H)     -> length_token [B,K,1,H]
```

Root/body 各有一条 source projection，然后在 **hidden token 域** 对齐到 target query 长度：

```text
src_root [B,K,Ts,H] = RootSourceProj(546 -> H)(source+value-mask)
src_body [B,K,Ts,H] = BodySourceProj(546 -> H)(source+value-mask)

TemporalAlign(src_*, source_time_valid, Tt, optional time_map)
  equal Ts=Tt: identity
  unequal: normalized-progress interpolation in hidden context only
  output [B,K,Tt,H]

valid_gate[b,k,t] = source_present[b,k] AND aligned_source_valid[b,k,t]

token_ctx_root[b,t] =
  sum_k(where(valid_gate, aligned_root, 0)) / max(sum_k(valid_gate), 1)

token_ctx_body[b,t] =
  sum_k(where(valid_gate, aligned_body, 0)) / max(sum_k(valid_gate), 1)

slot_meta[b,t] =
  sum_k(where(source_present, role_token + length_token, 0))
  / max(sum_k(source_present), 1)

context_present[b] = any_k(source_present[b,k])

ctx_root = where(context_present AND target_valid,
                 token_ctx_root + slot_meta + op_token + task_token,
                 0)
ctx_body = where(context_present AND target_valid,
                 token_ctx_body + slot_meta + op_token + task_token,
                 0)                                      [B,Tt,H]
```

`source_present` 与 `source_time_valid` 必须在 projection 后 gate。projected token 只按逐帧 `valid_gate` 聚合；role/length 是 slot metadata，只按 `source_present` 聚合；task/op 是 target-side intent，不受 source-time/value gate。最后使用 finite `torch.where(context_present AND target_valid, ..., 0)`，保证 Linear bias、NULL role 或 invalid token 都不能产生伪 context，且 T2M/control 的新增分支严格为零。这一公式与模型训练专项文档第 4.5 节一致。

normalized-progress 只用于 context 对齐，不重采样 source K273，也不抹掉 `Ts/Tt`：模型同时接收 `source_native_lengths`、`requested_target_len` 和 `log(Tt/Ts)` embedding。若该简单 adapter 对局部 timing edit 不够，再消融 masked source cross-attention；第一版不直接把 backbone 序列扩成 `Ts+Tt`。

`normalized_progress_v1` 明确定义为 target query 到 source coordinate 的映射：

```text
source_coord(j) = j * (Ts - 1) / (Tt - 1),  Ts>1 and Tt>1
source_coord(j) = 0,                       otherwise
```

runtime map shape 为 `[B,K,Tt_max]`。只在 valid source span 内插值 hidden token，padding 不参与；target padding 由 `target_valid` gate。`Ts=Tt` 走严格 identity path，避免本来等长的 pair 被不必要插值。equal/off-by-one/material-duration 三层分别评估；若局部 timing instruction 在 material-duration 层失败，再切换到 masked source cross-attention，而不是重采样 raw K273。

### 6.3 Root stage

```text
model_in [B,Tt,546]
   |
RootInputProj(546 -> 1024)
   |
   + ctx_root [B,Tt,1024]
   |
Root DiT + HYText + time + frame_gauge_dir
   |
root_hat [B,Tt,5]
```

task/op/role 信息都在 source context 中，且只对 source-conditioned task 生效。T2M/Kimodo control 的新分支输出严格为 0，避免仅仅“加了接口”就改变旧任务函数。

### 6.4 Body stage

```text
root_hat [B,Tt,5]
   |
原有 target-root -> local-root 转换
   v
local_root [B,Tt,4]

concat(local_root, noisy target body, hard_mask) [B,Tt,545]
   |
BodyInputProj(545 -> 1024)
   |
   + ctx_body [B,Tt,1024]
   |
Body DiT
   |
body_hat [B,Tt,268]
```

禁止：

- 用 source root 替换 `root_hat`；
- 把 source 写进 `state`；
- 根据 source/target GT 差异在推理时构造不可获得的 oracle mask。

### 6.5 输出不变

```text
concat(root_hat, body_hat) [B,Tt,273]

[0:269]   target clean x0 continuous
[269:273] target contact logits
```

两个 `Linear(546,1024)` source/value-mask projections约增加 `1.12M` 参数；加上 op/task/role/length embeddings后约 `1.15M`。这是估算值，实施后必须由参数统计脚本给出精确总量和 trainable 参数量。

### 6.6 完整 Tensor Information Flow

```text
DATA
====

HumanML3D item                       MotionFix item
motion + description                 source + target + instruction
[Ti,273]                              [Ts,273] [Tt,273]
       |                                      |
       | window then transform                | no crop; one-phi pair transform
       v                                      v
target_un [B,Tt,273]                  source_un [B,Ts,273]
                                      target_un [B,Tt,273]
       |                                      |
       +------------------+-------------------+
                          |
                   task batch adapter
                          |
          +---------------+------------------+
          |                                  |
          v                                  v
target x0 [B,Tt,273]                  source [B,Ts,273]
                                            target_op_id [B,Tt]
                                            task_id [B]
                                            role_id [B]

FLOW TARGET PATH
================

target x0_cont [B,Tt,269] + eps [B,Tt,269] + t [B]
                   |
                   v
z_t = t*x0 + (1-t)*eps                     [B,Tt,269]

target contact [B,Tt,4] + contact_aux      [B,Tt,4]
                   |
                   v
state [B,Tt,273]
                   |
       hard overwrite only here
                   |
hard_obs [B,Tt,273] + hard_mask [B,Tt,273]
                   |
                   v
z_imp [B,Tt,273] + hard_mask [B,Tt,273]
                   |
                   v
model_in [B,Tt,546]

SOURCE CONTEXT PATH
===================

source/value mask [B,K,Ts,546]    role [B,K]    task [B]
       |                               |            |
       v                               v            v
RootSourceProj / BodySourceProj   role_embed   task_embed
       |
source tokens [B,K,Ts,1024]
       |
TemporalAlign(Ts -> Tt, masks, optional time_map)
       |
aligned source [B,K,Tt,1024] + target_op_embed [B,Tt,1024]
       |
source/time/target-valid exact gates
       |
       +-------------------+------------------+
       |                                      |
       v                                      v
ctx_root [B,Tt,1024]                 ctx_body [B,Tt,1024]

ROOT/BODY DENOISING
===================

model_in -> RootInputProj -> +ctx_root -> Root DiT -> root_hat [B,Tt,5]
                                                        |
                                                        v
                                              local_root [B,Tt,4]
                                                        |
noisy body [B,Tt,268] + hard_mask [B,Tt,273] + local_root
                                                        |
                                                        v
                                               body_in [B,Tt,545]
                                                        |
                         BodyInputProj -> +ctx_body -> Body DiT
                                                        |
                                                        v
                                               body_hat [B,Tt,268]
                                                        |
                              root_hat + body_hat -------+
                                                        |
                                                        v
                                                 pred [B,Tt,273]
                                                        |
                                                        v
                                            existing target losses
```

## 7. 各任务训练样本如何构造

### 7.1 T2M

```text
target = HumanML3D clean motion
source_present = 0
op = GENERATE
hard_mask = 0
task = GENERATE
capability = T2M
text = description
```

### 7.2 Kimodo-like control

```text
target = HumanML3D clean motion
source_present = 0
op = GENERATE
hard_obs_un, hard_mask = existing compiler(
  transformed_target_un, lengths=requested_target_len, keyed_generator)
assert hard_mask is zero on target padding
hard_obs = normalizer(hard_obs_un)
task = GENERATE
capability = KIMODO_CONTROL
text = description
```

control curriculum、mask channel、overwrite、loss 和 sampler 分支均保持当前实现。唯一合法数据顺序是：pair/window transform 后先在 unnormalized padded batch 上调用 `compiler(target_un, lengths=requested_target_len, keyed_generator)`，断言 padding 上 mask 全零，再分别 normalize target/hard_obs；不能对 normalized 273D 编译控制。

### 7.3 MotionFix editing

```text
target = MotionFix target K273
source = MotionFix source K273
op = EDIT for every valid frame
hard_mask = 0
task = EDIT
text = edit instruction
```

注意：第一版不需要自动猜“哪些关节被编辑”。UMO 的结果表明全帧 EDIT + frame-aligned source context 已可工作；而旧代码中训练用 source/target code 差异构造 preserve mask、推理用文本规则猜 mask，存在明显 train/inference mismatch，见 [edit_masks.py](/mnt/afs/UMO_debug/models/codeflow/edit_masks.py:35) 与 [edit_masks.py](/mnt/afs/UMO_debug/models/codeflow/edit_masks.py:179)。这条旧路径不能直接迁移。

### 7.4 Editing + control

在基本 editing 验证成功后，从同一 MotionFix target 合成 hard constraints：

```text
source = source motion
target = target motion
task = EDIT
capability = MOTION_EDIT_WITH_CONTROL
op = EDIT
hard_obs_un, hard_mask = compiler(
  transformed_target_un, lengths=requested_target_len,
  keyed_generator, none_prob=0)
assert hard_mask is nonempty on valid frames and zero on padding
hard_obs = normalizer(hard_obs_un)
```

这不需要新数据，可训练“按照文本修改 source，同时手/脚/root 必须经过给定点”。对于被 condition sampler 声明为 controlled 的 EDIT 样本，compiler 强制 `none_prob=0`、只采非空 mode，并逐样本断言 valid frame 内 `hard_mask.any()`；complete Stage-2 GENERATE curriculum 的 `none_prob=0.10` 不得继承进来。建议只占 editing batch 的 `10%-20%`，不能一开始压过纯 editing。

### 7.5 Reaction

未来 reaction 数据映射为：

```text
source = initiator/other actor motion
target = responder motion
source_role = OTHER_ACTOR
task = REACTION
op = GENERATE
hard_mask = optional target-actor control
```

两个人必须使用同一个 world transform，不能分别 root-origin-shift，否则互动距离和朝向会消失。EDIT 与 REACTION 都使用 source context，但 task/role/op 不同，因此不会混淆。这里只是接口预留；没有 actor/world transform、time map 和互动几何 QA 前，不宣称 reaction 已可直接训练。

## 8. Loss 设计

### 8.1 第一版保持一个 target objective

所有任务的输出都是 target motion，所以第一版不新增 output head，也不改变主 loss：

```text
L_target = 1.00 * L_repr_velocity_from_x0
         + 0.10 * L_contact_all
         + 0.01 * L_clean_root_velocity
         + 0.01 * L_clean_joint_velocity
         + 0.01 * L_foot_lock
         + 0.07 * L_FK_consistency
```

有 hard control 时再加：

```text
+ 0.25 * L_control_continuous
+ 0.05 * L_control_contact
```

Editing 的 target 就是 MotionFix target；其 source 只是条件，不是 loss target。

上述系数不是代码默认值，而是成功实验 resolved config 的合同。启动时必须同时 assert：prediction=`x0`、representation loss space=`velocity`、semantic block weights/scale、`velocity_t_eps`、所有 auxiliary/control loss、normalizer SHA、HYText manifest/template、control curriculum、self-conditioning、EMA weight source。Stage-1/complete Stage-2 resolved config SHA 分别为 `da09bc3d0a7f1ee2559d75162c877f7c317d59ea4fa47bcf1b1b6e308628e2fa` 与 `b194ad7ff5c293d4022f1e4b2afff949adbafcc5c74683b34e7bbc4cc5510b2e`。

### 8.2 第一版不加 source-copy loss

直接对全序列施加 `prediction ~= source` 会与有效编辑冲突。第一版依赖 target loss 学习“哪里保持、哪里修改”，与 UMO 的基本训练方式一致。

后续只能做受控消融：

- `L_preserve_soft`：仅在 source/target 物理差异非常小的保守区域加权；
- instruction-to-part auxiliary predictor；
- edit delta/changed-region reweighting。

任何利用 target 构造的 mask 都只能作为训练 supervision，推理必须由 source+instruction 预测；不能把 oracle mask直接送给 denoiser后再用规则 mask 做评估。

## 9. 从头训练策略

这里的“从头训练”指新实验从随机初始化开始；阶段之间只恢复同一条新实验自己的完整 checkpoint，不借用归档成功 checkpoint 作为主结果初始化。本节只保留总览，精确 step 边界、概率插值、optimizer/resume 合同和门禁以 [HY273_multitask_model_training_plan_CN.md](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_model_training_plan_CN.md) 第 7-16 节为唯一权威来源。

```text
Stage A     0 -> 200K
  100% HumanML3D T2M
  建立正常动作与绝对文本 prior

Stage B1  200 -> 250K
  复用旧版 per-sample none_prob=0.10 control curriculum
  单独确认 hard-mask/overwrite 路径在新数据合同下可学习

Stage B2  250 -> 400K
  前 10K: (T2M, control, edit)
           (10%,90%,0%) -> (18%,80%,2%)
  此后:    18% / 80% / 2%
  editing 从这里低比例进入，建立 source optimizer state 和依赖

Stage C   400 -> 500K
  前 25K: (18%,80%,2%) -> (35%,45%,20%)
  此后:    35% / 45% / 20%
  不首次引入新能力，只做三能力联合巩固与防遗忘
```

editing 既不从 step 0 加入，也不拖到最后 100K 才第一次出现。选择 B2 的原因是：200K 时已有 T2M prior，250K 时已有 control-alive 证据；此时可以把 source context 和相对指令作为唯一新增变量，而 Stage C 仍保留足够窗口做质量训练和旧能力回归。

MotionFix edit batch 内条件组合保持：

```text
70% source + edit text
10% source only
15% source + edit text + nonempty hard control
 5% source + nonempty hard control
```

四种组合都固定 `task_id=EDIT`、source present；condition-pattern sampler 是唯一 text-drop 决策源。T2M/control 都固定 `task_id=GENERATE`，是否受控只由 `hard_obs/hard_mask` 区分。

固定 optimizer groups 从 update 0 创建，不在阶段边界增删：

```text
                         Stage A/B1   Stage B2   Stage C
existing backbone G0       1e-4        1e-4       2e-5
new context G1/G2              0        1e-4       5e-5
```

load checkpoint 后必须按 `next_global_step` 重写并校验 phase LR/WD；所有 source-absent step 把 context grad 设为 `None`，避免 AdamW moment/weight-decay 污染。8 卡 DDP、stateless `SamplePlan`、weighted-deficit scheduler、50K archive/10K latest 和 matched non-regression gate 的细则不在本文件重复。

不建议冻结整个 backbone。source context 要适配已有 motion prior；首跑用低比例 editing、分组 LR 和逐阶段 gate 控制干扰，只有明确发生遗忘时才按顺序降低 edit ratio、降低 G0 LR、增加 replay，最后才考虑 teacher distillation。

## 10. CFG 与采样

### 10.1 T2M/control 保持当前 separated CFG

当前有控制时的四个 branch：

```text
joint   = text + control
text    = text only
control = control only
empty   = neither
```

四个 branch 始终固定 `task_id=GENERATE`；差异只来自 text/drop 与 hard_mask/obs。其组合和每步 input overwrite 不改。正式 GENERATE/control non-regression 保留归档协议：EMA、ODE32、text/control CFG=2.0、`contact_init=random`、`contact_feedback=blend`、`cfg_apply_contacts=true`。

### 10.2 Editing 第一版只做 text-over-source CFG

训练中显式覆盖 `source+text` 和 `source-only`，source、task、role、frame gauge 保留：

```text
pred_source = model(source, empty_text, task=EDIT)
pred_joint  = model(source, edit_text,  task=EDIT)

pred = pred_source + s_edit * (pred_joint - pred_source)
```

这让 guidance 表示“在同一个 source 基础上增强 instruction”，不会把 source 和 text 当作可以线性独立相加的两个无关条件。第一版不做 source dropout，也不引入 source CFG；先扫描 `s_edit = 1.0, 1.5, 2.0, 3.5`。

### 10.3 Editing + control 使用分层 guidance

```text
p_src      = source only
p_edit     = source + edit text
p_all      = source + edit text + hard control

p = p_src
  + s_edit * (p_edit - p_src)
  + s_ctrl * (p_all  - p_edit)
```

三个 branch 共享同一 unclamped ODE state；只有 `p_all` 的 denoiser input 做 hard overwrite。最终仍同时保存 raw 与 exact-clamped 输出。

EDIT 使用独立版本化的 `editing_selected_contact_v1`：continuous x0 使用上式，contact 取最完整条件 branch 的原始 logits，等价 `cfg_apply_contacts=false`；`s_edit` 默认 1.5、validation 扫描 `1.0/1.5/2.0/3.5`，`s_ctrl` 默认 2.0、validation 扫描 `1.0/2.0/3.5`。它不是归档 GENERATE/contact parity 协议，两者必须分开记录和评估。

训练中必须覆盖同一 `task_id=EDIT` 下的四种 source/text/control 组合，不能通过切换到 `EDIT_CONTROL` 制造 `p_all`。若以后启用 self-conditioning，每个 CFG branch 维护自己的 `x_self_cond` 历史；不能把 clamped `p_all` clean estimate喂给 `p_src/p_edit`，否则控制条件会跨分支泄漏。

## 11. 如何证明没有把任务搞混

### 11.1 接口级测试

- EDIT 无 source 必须报错；
- KIMODO_CONTROL capability 不得把 source 塞进 hard_obs；
- source 缺失时，新模型加载旧权重后在同 dtype/device 下与旧模型 `torch.equal`；
- source context 永远不改变 hard mask/value；
- 当前 independent-canonical MotionFix 使用一个共享目标 yaw `phi`，不使用两个随机 yaw；
- 当前 pair 共享 output gauge draw，但 source/target physical transform group 与 applied delta 分开记录；
- shared-world 数据才对所有 motion 使用同一个 SE(2) delta；
- `Ts != Tt` 时 source/target padding mask 与 hidden alignment 可重放；
- CFG drop text 后 task_id/op_id/source_role 不得被 drop；
- T2M/control 共用同一个 `HML_MIXED` stream cursor，在 batch 内由 per-sample capability plan 与 `hard_mask` 区分；进入模型后都保持 `task_id=GENERATE`；
- intended-controlled EDIT 不允许 compiler 再抽 `none`，且 padding 上 hard mask 必为零。

### 11.2 行为级反事实测试

对同一 noise seed 做：

```text
A: correct source + correct instruction + EDIT
B: shuffled source + correct instruction + EDIT
C: correct source + shuffled instruction + EDIT
D: no source + same relative instruction + GENERATE task/relative profile (OOD diagnostic)
E: correct source + hard control + EDIT
```

模型必须同时依赖 source 与 instruction：A 应优于 B/C；E 的 hard-control error 应与单独 CONTROL 同等级。B/C 的 derangement、length matching、relative-text profile 和 permutation SHA 以模型训练专项文档冻结的 counterfactual manifest 为准。D 不是普通 T2M，也不用于替代 A/B/C 的 source/text dependence 结论。

### 11.3 不允许的伪成功

- 输出几乎等于 source，却只报告 source similarity；
- 使用 GT target 生成 edit mask；
- 使用 GT target length/frame_gauge_dir 却不在协议中披露；
- exact-clamped output error 为零，却不报告 raw model control error；
- 只看 retrieval，不看脚滑、接触和自然度。

## 12. 评估协议

### 12.1 T2M 回归

- 固定 prompt/seed visual；
- foot skate、contact consistency、jerk；
- 同协议记录本次 200K Stage-A、250K Stage-B1 和 400K Stage-B2；Stage C 的遗忘主判断以本次 400K EMA 为 matched baseline，归档旧模型只提供 absolute floor。

当前不做 K273 到其他表示后的 FID、R-Precision/Top3 或 MM-Dist，也不把它们作为阶段门禁。完整 owner override 见 [HY273_current_evaluation_protocol_CN.md](/mnt/afs/mogeflow-control/outside_doc/HY273_current_evaluation_protocol_CN.md)。

### 12.2 Control 回归

完整保留当前 Kimodo-like test protocol：

- root sparse/dense error；
- endpoint position/rotation error；
- fullpose error；
- sparse contact adherence；
- FK consistency；
- raw output 与 exact-clamped output 分开；
- foot skate 为硬门禁。

non-regression 线必须在 Stage C 前按每个 control pattern 预注册：同时给绝对容差、相对本次 400K Stage-B2 matched baseline 的容差和 paired CI。`5%` 只可作为非近零指标的初始相对线，不能代替 sparse contact/rotation/near-zero exact error 的绝对阈值；超过即不宣布“保留了控制能力”，而是回退 task mix/LR。

### 12.3 MotionFix editing

当前第一阶段 protocol：

- source-copy baseline；
- relative-instruction-only OOD diagnostic（不能标为常规 T2M baseline）；
- same-noise source/instruction shuffle 或 drop 反事实；
- foot skate、contact、jerk 的可视化诊断；
- `source | target | prediction | source-target delta | source-pred delta` 五联可视化。

Editing 暂不发布 retrieval/FID/Top3 或 official evaluator 数字。先用固定 test pair 逐条确认模型确实依赖 source 与 instruction、编辑方向正确且动作自然；之后再单独敲定数值 benchmark。若只可视化等长 952 subset，source-copy、relative-instruction-only diagnostic 和 source+instruction model 仍必须使用同一 ID 列表。

官方 test 的 1,013 个 pair ID 均未出现在 MotionFix train，但并非所有底层内容未见：按数据协议的 `motionfix_instruction_nfkc_casefold_punctspace_ws_v1` 与 `motionfix_base_resolved30fps_span_v1`，`143` 条 normalized instruction、`508` 个 exact source clip、`59` 个 exact target clip 在 train 出现；`exact source+target clip pair` 与 `exact instruction+exact target clip` 都是 0。报告必须逐项给 `pair_id_seen/instruction_seen/source_clip_seen/target_clip_seen/source_base_seen/target_base_seen/exact_pair_clip_seen/exact_instruction_target_clip_seen`，并钉住 normalization implementation SHA，不能把 `pair_id_seen=false` 简化成 strict unseen-motion generalization。

### 12.4 组合能力

对 MotionFix test target 合成现有五类 Kimodo constraints，分别评估：

```text
edit success
+ raw hard-control adherence
+ exact-clamped adherence
+ foot skate/naturalness
```

这项验证能证明 source context 和 hard control 真正正交，而不是只在各自单任务中工作。

## 13. Reaction 及后续任务如何接入

以后新任务继续复用 hard-control core，但不保证“只加一个 task id 就够”。统一数据接口至少预留：

```text
New task dataset
   |
   +--> target HY273
   +--> zero or more source HY273 streams
   +--> task_id
   +--> capability_id
   +--> source_role_id
   +--> P/G/E op pattern
   +--> source native length / target length / time map
   +--> actor_id / optional world_from_asset
   +--> text/audio/etc condition
   +--> optional existing hard_obs/hard_mask
```

Reaction 的额外模型变化主要是多 source slot 聚合：

```text
source_i token + role_i token
        |
sum/attention over K sources at each target frame
        |
ctx_root / ctx_body
```

第一版 MotionFix 不实现多 source attention，但 public tensor schema 预留 K。Reaction v0 之前还必须补：共享世界坐标数据、actor/world transform、双人相对距离/朝向、碰撞与接触 QA，以及可能新增的 K-source aggregator 参数；当前方案只保证数据/API 不必推翻，不声称现有单人数据已足够训练 reaction。

## 14. 实施文件建议

```text
data/hy273_manifest_schema.py
data/hy273_multitask_dataset.py
data/hy273_temporal.py
data/hy273_pair_transforms.py
data/hy273_task_sampler.py
data/hy273_collate.py
  - HML motion/caption/window + MotionFix pair adapters
  - native Ts/Tt, independent masks, frame-policy dispatch

models/raw_motion/context_conditioning.py
  - task/op/role enums and embeddings
  - source/value-mask projections, hidden temporal alignment, exact-zero gating

models/raw_motion/kimodo_like_flow_dit.py
  - forward 增加 context batch
  - root/body temporal fusion
  - no-source identity path

train_hy273_multitask.py
  - synchronized batch-level train-stream scheduler
  - T2M/control/edit adapters
  - per-stream/per-task metrics and LR groups

sample_hy273_multitask.py
  - typed task validation
  - editing hierarchical CFG
  - reuse current control sampler core

tools/build_hy273_multitask_manifest.py
tools/build_hy273_multitask_stats.py
tools/audit_hy273_multitask_data.py
tools/cache_hy273_multitask_text.py
tools/compare_hy273_nonregression.py
eval_hy273_motionfix_edit.py

configs/hy273_kimodo_context_stageA_t2m.yaml
configs/hy273_kimodo_context_stageB_control.yaml
configs/hy273_kimodo_context_stageC_multitask.yaml

tests/test_multitask_condition_contract.py
tests/test_motionfix_k273_pairs.py
tests/test_context_no_source_identity.py
tests/test_edit_sampling.py
tests/test_edit_control_composition.py
tests/test_multitask_ddp_schedule.py
```

## 15. 实验顺序与 Go/No-Go

```text
P0 Data preflight
  unified manifest + HML segment re-extract + MotionFix frame policy
  + fixed 30 FPS/no-resample contract + authenticated source_fps/fail-closed
  + native Ts/Tt/collate + 32-pair visual
        |
        v
M0 Architecture identity
  新 context 接口在 source absent 时不改变旧模型输出
  equal/unequal source adapter + DDP/resume smoke
        |
        v
M1 From-scratch Stage A 200K
  T2M gate
        |
        v
M2 Stage B1 250K
  control-alive gate
        |
        v
M3 Stage B2 300K / 350K / 400K
  control-dominant joint adaptation
  + early source/text dependence
        |
        v
M4 Stage C 410K / 425K
  early edit pilot + transition-complete gate
        |
        v
M5 Stage C 450K / 500K
  full T2M/control/edit/edit+control claim gate
        |
        v
M6 Optional extension / length predictor / reaction data contract

Parallel benchmark gate
  official MotionFix evaluator conversion/gauge roundtrip
  未通过时只报告 internal K273 protocol
```

No-Go 条件：

- pair frame policy/lineage 未被 manifest 和 loader严格执行；
- source absent identity test 不通过；
- editing 只能复制 source；
- source/text shuffle 不改变结果；
- control raw error 或 foot skate 超过 non-regression 线；
- source/target 被静默截断或独立 crop/yaw；
- 缺少 shared-world/reaction 数据合同却提前宣称 reaction 能力。

## 16. 当前需要先验证、但不需要改变总体设计的事项

1. **MotionFix official evaluator**：当前 K273 单 motion semantic audit 已通过，但 prediction -> official evaluator input 的 conversion/gauge roundtrip 仍需实测；它是 official benchmark claim blocker，不是内部 K273 pilot blocker。
2. **Frame gauge**：当前 MotionFix 是 independent-canonical pair；两边各自 root shift，再映射到同一个随机 `phi`。`frame_gauge_dir` 不是 source 原始朝向，也不是 GT target heading。
3. **不等长 pair**：public interface 从第一版保留 native `Ts/Tt`；采样时 target length 由调用者给定。自动 duration 另需 length predictor。
4. **任务比例**：`35/45/20` 和 edit 条件组合 `70/10/15/5` 是预注册的首跑合同，不是先验真理。每 10K 只评估三任务指标，不在线改写正在运行的 main schedule；需要调整时必须停止并启动带新 config/version 的 run，单因素判断则使用 matched fork。
5. **Part-level edit intent**：保留接口扩展位，但第一版全帧 EDIT；不能复用旧的 oracle-vs-rule mask mismatch。
6. **Simple temporal alignment 上限**：normalized-progress hidden alignment 可能不足以表达局部提前/延后；以 unequal-duration 分层结果决定是否升级 masked source cross-attention。

## 17. 最终判断

这条路线可以在结构上保留现有 Kimodo-like 能力，并增加 editing/reaction 等 source-conditioned 能力，原因是两者不争用同一个条件通道：

```text
hard control 负责“输出必须等于什么”
source context 负责“基于什么进行修改/响应”
task/op/role 负责“为什么提供这个 source”
```

最小正确版本不是 `obs=source`，而是：

```text
existing [z_imp, hard_mask] denoising path
+ frame-aligned source temporal fusion
+ explicit EDIT task/meta-operation
+ frame-policy-aware paired geometry transform
+ old-task replay and non-regression gates
```

## 18. 资料与代码依据

- Cong et al., UMO, arXiv:2603.15975v1: <https://arxiv.org/html/2603.15975v1>
- UMO project: <https://oliver-cong02.github.io/UMO.github.io/>
- UMO public repository（截至上述 commit 尚未发布代码）: <https://github.com/Oliver-Cong02/UMO>
- Athanasiou et al., MotionFix, SIGGRAPH Asia 2024: <https://arxiv.org/abs/2408.00712>
- MotionFix official repository: <https://github.com/atnikos/motionfix>
- 当前成功归档: <https://github.com/CHDTevior/mogeflow_kimodo_like>
- MotionFix K273 conversion/data semantics: <https://github.com/CHDTevior/HY201_to_K273>

## 19. 对抗审核记录

R1 使用 `gpt-5.6-sol / max`，结论为 NO-GO，原始报告见 [HY273_kimodo_context_multitask_motion_editing_review_R1.md](/mnt/afs/mogeflow-control/outside_doc/HY273_kimodo_context_multitask_motion_editing_review_R1.md)。本版已处理其主要问题：

- 取消 `EDIT_CONTROL`，hard control 与 task 正交；
- 分离 source slot/time/value/target masks 和 target-side op；
- 钉住成功 resolved config hash；
- 增加 10K/25K/50K pilot checkpoint 与可执行 non-regression 门禁；
- 从 Stage A 固定 optimizer groups并明确 DDP branch 使用；
- `c_dir` 改称 `frame_gauge_dir`；
- 原生支持 `Ts != Tt` 且 MotionFix v1 不 crop；
- reaction 降格为有数据合同门禁的扩展方向。

R1 对 pair gauge 的 Critical 结论基于“source/target 应保留原始 shared-world relation”的假设。代码 lineage 复核表明当前 MotionFix source/target 在 HY201 之前已经分别 HumanML3D canonicalize；本版不尝试恢复不存在的 shared-world relation，而是明确标记 `pair_shared_world=false`，采用 independent-sequence frame policy。

R2 使用 `gpt-5.6-sol / max`，结论为 `CONDITIONAL GO`，原始报告见 [HY273_multitask_data_and_model_review_R2.md](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_data_and_model_review_R2.md)。本次 r3 修订关闭了其文档级问题：

- 修正 `000362` 被混入 `000000` 元数据的问题，并钉住三条 golden rows；
- HY201 独立 AssetRef、resolved frame span 和 derived-cache hash/audit 合同；
- span QA 修正为全量 90 个 clamped end、18 个 segment-too-short、16 个短 asset；
- stats 域内 `uniform motion -> uniform accepted caption -> valid-frame moments` 合同；
- target-to-source normalized-progress exact map；
- stateless DDP SamplePlan、optimizer-boundary checkpoint 和同拓扑 replay 范围；
- EDIT condition-pattern 是唯一 dropout 源；
- overlap 标签拆成 instruction/clip/base/pair 等精细维度；
- complete Stage-2 YAML/checkpoint SHA 与结构化 non-regression artifact。

仍需实验而不能靠文档关闭的门禁是：accepted-caption/derived-cache/pair manifest 实产物 QA、1/8-rank resume trace、T2M frozen eval、control paired bootstrap CI，以及 official MotionFix evaluator roundtrip。

R3 使用 `gpt-5.6-sol / max`，结论为 `CONDITIONAL GO`，报告见 [HY273_multitask_data_and_model_review_R3.md](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_data_and_model_review_R3.md)。本次 r4 设计修订关闭了其六个 Medium 和一个 Low：训练 stream 与模型 task 分离、EDIT compiler 非空合同、control 编译顺序、stats measure、pair gauge、overlap key 和 K>1 role gating。

R4 使用 `gpt-5.6-sol / max`，没有 Critical/High/Medium，结论为 `CONDITIONAL GO`，报告见 [HY273_multitask_data_and_model_review_R4.md](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_data_and_model_review_R4.md)。其两个 Low 已在当前版关闭：四类验收 gate artifact 获得 typed immutable binding；control per-case manifest 获得自描述的 case-key/digest 合同。设计/schema 已可实施和 smoke，但 baseline `status=incomplete` 与 official evaluator 门禁仍保持 fail closed。

R5 使用 `gpt-5.6-sol / max` 做定向收口审核，报告见 [HY273_multitask_data_and_model_review_R5.md](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_data_and_model_review_R5.md)。它复算了 8,084 个 case key、8 个 shard、ordered/sorted digest、sidecar 和所有 live cross-reference，确认两项 R4 Low 均关闭且没有残余 finding：design/schema implementation `GO`；正式长训与 official benchmark 仍保持 `NO-GO` 门禁，所以总判断仍是 `CONDITIONAL GO`。

R6 使用 `gpt-5.6-sol / max` 对 30 FPS lineage 做专项对抗审核，报告见 [HY273_30fps_plan_review_R6.md](/mnt/afs/mogeflow-control/outside_doc/HY273_30fps_plan_review_R6.md)。它独立复算了 40,306 条最终资产、两段转换的逐文件 `T`、MotionFix 13,460 条 timestamp residual，以及 13,423 条 HumanML3D 30/20 FPS 对照，结论为当前 pinned 数据 `GO`：`runtime_fps=30` 与 `merge_resampling=none` 成立。唯一 Low 是未来新数据的 source-FPS 认证/fail-closed 尚待 P0 代码机械实现，不影响当前两套资产，也不解除正式长训和 official benchmark 的既有门禁。
