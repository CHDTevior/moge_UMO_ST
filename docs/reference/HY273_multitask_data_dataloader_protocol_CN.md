# HY273 多任务统一数据与 DataLoader 协议

> 版本：`v1-design-r5`  
> 日期：2026-07-14  
> 范围：HumanML3D K273、MotionFix K273 的统一 manifest、时间窗口、成对增强、normalization、task sampling、collate 和评估数据边界。  
> 不在本文重新设计完整 denoiser；只固定模型必须接收的数据字段。  
> 当前状态：**数据协议可实施。统一 manifest、accepted-caption 表、derived segment cache、joint stats 和 DDP replay smoke 是正式训练门禁；official-evaluator roundtrip 是发布官方 benchmark 数字前的门禁，不阻塞内部 K273 pilot。**
> 训练调度更新：high-level stream 与 A/B1/B2/C 比例以 `HY273_multitask_model_training_plan_CN.md` v1.3 为权威；本文已同步为 `HML_MIXED/MOTION_EDIT`，HML 内逐样本区分 T2M/control。

## 1. 结论先行

### 1.1 推荐方案

1. 原始 K273 `.npy` 保持只读；生成一个确定性统一 JSONL manifest，不复制完整 6GB 以上数据。
2. HumanML3D 的一个 manifest row 仍对应一个 motion，row 内保留多条 caption；训练时先均匀采 motion，再在该 motion 内采 caption，避免 caption 多的 motion 被过采样。
3. HumanML3D 非零 `from_tag/to_tag` 必须按 30 FPS 切片，不能继续把 segment caption 配给完整 motion。
4. segment 不能只切现有 273D 数组。应从对应 HY201 的 root translation + local rotations 切片，再运行官方 Kimodo feature extraction，重新得到 smooth root、positions、velocity 和 contact。
5. MotionFix instruction 是 `relative_edit_instruction`，绝不能当 target 的普通绝对 caption。
6. MotionFix source 是独立 context，不是 Kimodo `hard_obs`；`hard_obs/hard_mask` 继续只表示逐 ODE step 精确 overwrite 的控制。
7. MotionFix source/target 分别保留原生长度。最终接口应原生支持 `Ts != Tt`；第一版 smoke 可只跑等长子集，但不能把该结果写成 full benchmark。
8. 当前 MotionFix 数据的上游已经对 source/target **分别做过 HumanML3D canonicalization**。K273 最后一步虽未再次 canonicalize，但并不处于原始共享世界坐标。manifest 必须如实记录这一 lineage。
9. 当前 MotionFix 的随机 yaw 只能采一个共享目标角 `phi`；禁止 source/target 各自采随机角。对于当前 independent-canonical 数据，可用各自初始 heading 计算不同 delta，但最终都映射到同一个 `phi`。
10. 从头训练的多任务 v2 推荐预先重算一套 train-only、target-only、task-weighted joint stats，并从 Stage A 起冻结；若续训现有成功 checkpoint，则必须继续用旧 HumanML3D stats，不能中途换 normalizer。
11. task 由显式 `task_id` 决定，hard control 是否存在是正交字段。T2M 与 Kimodo control 都属于 `task=GENERATE`；pure edit 与 edit+control 都属于 `task=EDIT`。不要新增会破坏 CFG 可解释性的 `CONTROL`/`EDIT_CONTROL` 条件分支。
12. 主训练按 owner 决定不做 HumanML3D/MotionFix 跨数据 motion 过滤；评估报告必须区分 pair-unseen 与 motion-content-unseen。

### 1.2 当前代码不能直接承担联合训练

当前 [kimodo273_datasets.py](/mnt/afs/mogeflow-control/data/kimodo273_datasets.py:199) 已解析 caption span，但 [__getitem__](/mnt/afs/mogeflow-control/data/kimodo273_datasets.py:231) 的实际顺序是：

```text
load full motion
  -> random crop full motion
  -> random choose caption
  -> 只返回 from_tag/to_tag metadata
  -> 不按 span 切动作
```

当前 [train_hy273_raw_flow.py](/mnt/afs/mogeflow-control/train_hy273_raw_flow.py:1619) 也只实例化一个 `Kimodo273TextDataset`，没有 MotionFix pair、task sampler、source tensor 或独立 source mask。当前 [stats builder](/mnt/afs/mogeflow-control/tools/build_hy273_redenoise_stats.py:90) 只遍历 HumanML3D `train.txt`；当前 [HYText cache builder](/mnt/afs/mogeflow-control/tools/cache_hy273_hytext_embeddings.py:62) 也只会从 HumanML3D split/text 文件收集绝对 caption。

因此不能把两个目录简单 `ConcatDataset` 后开训。

## 2. 已独立复核的数据合同

### 2.1 K273 channel layout

官方 `KimodoMotionRep` 的 `size_dict` 直接定义了 273D 顺序，见 [kimodo_motionrep.py](/mnt/afs/mogeflow-control/external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:34)：

```text
[0:3]     smooth_root_pos          3
[3:5]     global_root_heading      2 = [cos(theta), sin(theta)]
[5:71]    local_joints_positions  66 = 22 x 3
[71:203]  global_rot_data         132 = 22 x 6, global cont6d
[203:269] velocities               66 = 22 x 3, global m/s
[269:273] foot_contacts             4, binary
```

转换实现不是复制 6D channel，而是：HY201 local rotations -> rotation matrices -> SMPL-X22 FK -> Kimodo global cont6d。代码证据见 [convert.py](/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/convert.py:46) 和 [geometry.py](/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:143)。

两套全量 semantic audit 均确认：

```text
shape                     [T,273]
finite                    true
joint order               official SMPL-X22 body-22
coordinate system         right-handed, Y-up, XZ ground
forward reference         +Z when heading angle is zero
position unit             meter
fps                       30
contact                   exact 0/1
saved vs official feature max error  0.0
```

当前项目 slice 常量与官方一致，见 [hy273_slices.py](/mnt/afs/mogeflow-control/models/raw_motion/hy273_slices.py:11)。contact 在 normalizer 中被强制设为 mean=0/std=1，并在 normalize/denormalize 后恢复原值，见 [hy273_normalizer.py](/mnt/afs/mogeflow-control/models/raw_motion/hy273_normalizer.py:35)。

### 2.2 必须澄清的 canonicalization lineage

`HY201 -> K273` 最后一步确实没有调用 `KimodoMotionRep.canonicalize`，等价于官方 `to_canonicalize=False`。但这只描述**最后一步**，不能推导出原始 AMASS world frame 仍然存在。

MotionFix 当前 K273 的真实 lineage 是：

```text
official MotionFix source/target
  -> motionstreamer272_hml
       source 独立 origin/heading canonicalize
       target 独立 origin/heading canonicalize
  -> HY201 o6dp
  -> K273, to_canonicalize=False
```

上游代码在 [prepare_motionfix_272_hml.py](/mnt/afs/mogeflow-umo/tools/prepare_motionfix_272_hml.py:141) 对每个 motion 单独减 frame-0 origin、移除 initial heading；source 和 target 又在 [252-253 行](/mnt/afs/mogeflow-umo/tools/prepare_motionfix_272_hml.py:252) 分别调用转换。其 manifest 明确写 `canonical_frame=per_sequence_humanml3d`。

因此当前数据应声明：

```text
coordinate_frame = per_sequence_hml_canonical_then_raw_k273
pair_shared_world = false
```

这对当前 MotionFix semantic editing 并非必然错误，因为 pair 本来来自不同 AMASS clip，编辑通常在各自动作起点规范化后的语义空间学习；但它有三条边界：

1. 不能声称保留原始 source-target 世界平移/初始朝向关系。
2. 不能直接把该 frame contract 迁移到 scene-aware editing 或 reaction。
3. official MotionFix evaluator 的恢复 gauge 必须单独做 roundtrip，不能由单 motion K273 audit 代替。

### 2.3 FPS 复核

结论固定为：**当前 HumanML3D K273 与 MotionFix K273 都使用 30 Hz 时间轴，联合 manifest/dataloader 不做跨数据 FPS 重采样。** 传统 HumanML3D `263D/new_joint_vecs` 的 20 FPS 约定属于另一条数据 lineage，不能迁移到当前资产。

全量最终资产检查：

| dataset | manifest rows | frames | manifest `fps` | feature dim |
|---|---:|---:|---:|---:|
| HumanML3D K273 | 26,846 | 5,945,004 | 26,846 条全部为 30 | 全部 273 |
| MotionFix K273 | 13,460 | 1,489,149 | 13,460 条全部为 30 | 全部 273 |

代码与帧数 lineage 也一致：

```text
HumanML3D released MotionStreamer 272D, 30 FPS
  -> HY201: 5,945,004 frames
  -> K273:  5,945,004 frames

MotionFix raw pair clips, 30 FPS
  -> MotionStreamer-compatible 272D
  -> HY201: 1,489,149 frames
  -> K273:  1,489,149 frames
```

`272 -> HY201` 在 [geometry.py](/mnt/afs/UMO_debug/motion272_to_hymotion/motion272_to_hymotion/geometry.py:184) 只恢复/重编码每帧并创建相同 `T` 的输出；`HY201 -> K273` 在 [convert.py](/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/convert.py:84) 也保持 `T`，并用 `motion_rep.fps=30` 计算 m/s velocity 与 contact。两步都没有 temporal resampling。

仅看上游脚本中的 `ex_fps=30` 仍不够，因为参考 AMASS processor 在 [amass_process.py](/mnt/afs/UMO_debug/outside_material/272-dim-Motion-Representation/amass_process.py:129) 使用整数 stride。为排除名义 30 FPS 与实际约 33.3 Hz 的子域，额外做了两项实测：

1. MotionFix 的 13,460 条 source/target clip 与 annotation timestamp 对比：`13,203` 条严格等于 `duration*30`，`139/118` 条分别差 `-1/+1` 帧，没有任何条目超过 1 帧；因此不等长 pair 是 duration 语义或端点取整，不是 FPS mismatch。
2. HumanML3D 的 13,423 条非镜像 clip 与传统 20 FPS lineage 对比：对照源固定为 `/mnt/afs/HumanML3D/index.csv`，SHA256 为 `80226910d7259655ab5f411fd807b7651037341b4806fde025946a74d62b1065`，传统 263D feature 长度按 `end_frame-start_frame-1` 复算。帧数满足约 `30/20=1.5`；对传统长度至少 59 帧的 12,761 条 clip，endpoint ratio `(T30+1)/(T20+1)` 全部在 `[1.5,1.5167]`，没有接近 `5/3` 的长 clip 子群。因此实际发布资产中未发现 100 Hz 经 `stride=3` 形成约 33.3 Hz 的系统性子域。这里不能改用磁盘上其他残缺或软链接的 `new_joint_vecs` 副本复核。

完整机器可读证据见 [HY273_30fps_lineage_audit_v1.json](/mnt/afs/mogeflow-control/outside_doc/HY273_30fps_lineage_audit_v1.json)，SHA256 为 `5c394a575ef1f96466e4a3c2147200bcaa81c307c732964e087f8da5f7dba469`。由此运行时合同固定为：

```text
runtime_fps = 30.0
dt = 1 / 30 second
merge_resampling = none
caption seconds -> frames = round_half_up(seconds * 30)
K273 velocity/contact fps = 30
```

未来接入非 30 FPS 数据时，manifest builder 默认 fail closed；如确需统一到 30 Hz，必须在 root translation + local SO(3) rotation 的物理源域重采样后完整重提 K273，不能线性插值现成 273D/cont6d。特别注意：当前 [HY201 -> K273 converter](/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/convert.py:273) 的 `--fps` 只设置 Kimodo velocity/contact 的物理缩放并写入 manifest，它既不认证源 FPS，也不执行 temporal resampling；未来新数据不得把“传 `--fps 30`”当成重采样。

### 2.4 R6 对抗审核闭环

`gpt-5.6-sol / max` 已对最终 manifest、实际 NPY header、转换代码、MotionFix timestamp 和 HumanML3D 30/20 帧数关系做独立复算，结论为 **GO**。没有发现会推翻当前 30 Hz/no-resample 合同的问题；唯一 low finding 是上一段所述的未来新数据 fail-closed 机制尚未在通用 converter 内机械实现。完整报告见 [HY273_30fps_plan_review_R6.md](/mnt/afs/mogeflow-control/outside_doc/HY273_30fps_plan_review_R6.md)。

## 3. 两套数据的实测事实

### 3.1 HumanML3D K273

```text
motions             26,846
frames              5,945,004
train/val/test       21,466 / 1,338 / 4,042
frame range          3 .. 300
files > 300          0
```

四个 smooth-root fallback 文件与报告一致：

```text
motion_data/000990.npy
motion_data/005836.npy
motion_data/M000990.npy
motion_data/M005836.npy
```

但当前 `min_frames=16` 不只排除这 4 个 fallback。实测总共会排除 16 个短 clip：train 12、test 4。新 manifest summary 必须分别报告：

```text
excluded_by_min_frames
excluded_by_smooth_root_fallback
```

不能把两者合并描述成“只排除 4 个”。

#### Caption/span 审计

| split | motions | captions | candidate segments | invalid nonzero spans | clamped end | segment `<16` | 当前每 epoch 误配 segment 的期望比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 21,466 | 64,140 | 2,220 | 4 | 82 | 16 | 3.463% |
| val | 1,338 | 3,996 | 130 | 0 | 2 | 2 | 3.239% |
| test | 4,042 | 12,090 | 458 | 2 | 6 | 0 | 3.802% |
| total | 26,846 | 80,226 | 2,808 | 6 | 90 | 18 | - |

这里“当前误配”指 loader 随机抽到 segment caption，却仍返回完整 motion。6 条 invalid span 是 `end <= start`，应丢弃该 caption 并写入 QA，不应丢弃仍有其他合法 caption 的 motion。

另有 90 条 segment 的 tag end 在 30 FPS 下超过当前 clip 末尾，其中 train/val/test 分别为 `82/2/6`；它们多为旧 annotation 的宽松上界。策略是 clamp 到 `[0,T]` 并记录 `span_status=clamped_end`，而不是越界或静默改成 full caption。clamp 后另有 `16/2/0` 条 segment 短于 16 帧，应拒绝该 caption。这与完整 asset 本身 `<16` 的 `12/0/4` 条是两个互不合并的 QA 计数。

### 3.2 MotionFix K273

```text
motion files         13,460
valid pairs           6,730
frames              1,489,149
train pairs           5,387
val pairs               330
test pairs            1,013
missing split row         1 = train/006722
empty instruction         0
```

`splits.json` 有 6,731 个 ID，但 annotation 和 source/target 文件均缺 `006722`。manifest builder 应把它写进 rejected report，不得生成可训练 row。

#### 长度差不是单一问题

| split | equal | off-by-one | material difference (`abs(Tt-Ts)>1`) |
|---|---:|---:|---:|
| train | 5,047 | 208 | 132 |
| val | 313 | 12 | 5 |
| test | 952 | 26 | 35 |

完整差值分布主要是 `-30/-1/0/+1/+30`，test 另有一条 `+60`。所有不等长 case 的文件长度都能由 annotation source/target timestamp 在 30 FPS 下解释到正负 1 帧。

这说明：

- `+/-1` 多为采样/边界取整；
- `+/-30`、`+60` 是 1-2 秒的真实 duration 差；
- instruction 中确实包含 `slower`、`faster`、`do it twice`、`get up later` 等时序编辑。

因此不能把 418 个 pair 统一当作 conversion bug。

### 3.3 跨数据与跨 split motion overlap

按 MotionFix annotation 的底层 HumanML3D motion ID 统计：

```text
MotionFix test 1,013 pairs:
  974 pairs 的 source 或 target base motion 出现在 HML train
  657 pairs 的 source 和 target base motion 都出现在 HML train
   39 pairs 的两边 base motion 均未出现在 HML train
```

再把 MotionFix train 的 base motion 也算作 seen 后，官方 test 中只有 1 个 pair 的两边 base ID 都完全未见。MotionFix official test 与 MotionFix train 的直接重叠还必须按对象精确定义：

```text
official test pairs                                  1,013
pair ID seen in train                                    0
normalized instruction seen in train                   143
exact source clip (base ID + resolved frame span) seen 508
exact target clip (base ID + resolved frame span) seen  59
source-target base-ID combination seen                 148
instruction + target base-ID seen                        8
exact source+target clip pair seen                        0
exact instruction + exact target clip seen               0
```

上述 `143/508/59` 使用以下可复现 key 合同：

```text
instruction_normalization_version = motionfix_instruction_nfkc_casefold_punctspace_ws_v1
instruction key:
  Unicode NFKC
  -> Unicode casefold
  -> 每个 Unicode punctuation category 替换为空格
  -> whitespace collapse + trim
seen instruction corpus:
  accepted MotionFix train pair instructions only

clip_identity_version = motionfix_base_resolved30fps_span_v1
clip key:
  (base_motion_id,
   floor(timestamp_start_sec * 30 + 0.5),
   floor(timestamp_end_sec * 30 + 0.5),
   fps=30)
```

overlap report 必须写 policy version、实现文件 SHA256 和 train/test source manifest SHA。原始 timestamp 仍保留在 row 中供审计，但 seen 判定使用 resolved frame-span key。

由此得到明确评估边界：

```text
可以声称:
  unseen MotionFix pair ID
  unseen exact instruction + exact target clip
  official pair-level benchmark protocol

不能直接声称:
  strict unseen-motion generalization
```

主训练按 owner 决定保留原始 split、不做跨数据过滤。可另建一个小型 component-disjoint sanity experiment，但它必须是独立训练/消融，不能把主 benchmark 偷换成过滤后的数据。

## 4. 统一 manifest 设计

### 4.1 两层结构

采用：

```text
AssetRef       描述一个只读 tensor 文件及其 payload hash/representation
MotionAsset    关联同一 motion 的 K273 与 HY201 AssetRef，并记录 frame lineage
TrainingRow    描述一个语义训练单元及其 text/pair 关系
```

不要把每条 HumanML3D caption 展开成独立 row；否则 caption 多的 motion 会被重复采样。

### 4.2 建议 JSONL schema

HumanML3D row 示例：

```json
{
  "schema_version": "hy273_multitask_manifest_v1",
  "uid": "humanml3d:000006",
  "dataset": "humanml3d_k273",
  "split": "train",
  "task_capabilities": ["t2m", "kimodo_control_synth"],
  "source_motion": null,
  "target_motion": {
    "motion_uid": "humanml3d:000006:target",
    "base_motion_id": "000006",
    "timestamp_sec": null,
    "coordinate_frame": "per_sequence_hml_canonical_then_raw_k273",
    "smooth_root_fallback": false,
    "k273_asset": {
      "path": "/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/motion_data/000006.npy",
      "sha256": "...",
      "frames": 300,
      "fps": 30.0,
      "feature_dim": 273,
      "representation_version": "kimodo273_smplx22_v1"
    },
    "hy201_asset": {
      "path": "/mnt/afs/mogo_base/datasets/HumanML3D/hymotion201_o6dp_hml272/motion_data/000006.npy",
      "sha256": "...",
      "frames": 300,
      "fps": 30.0,
      "feature_dim": 201,
      "representation_version": "hymotion201_o6dp_hml272_v1"
    }
  },
  "texts": [
    {
      "text_id": "humanml3d:000006:line1",
      "value": "...",
      "kind": "absolute_motion_caption",
      "encoding_profile": "hytext_absolute_motion_v1",
      "span": {
        "kind": "segment",
        "from_sec": 0.0,
        "to_sec": 5.0,
        "start_frame": 0,
        "end_frame_exclusive": 150,
        "span_status": "accepted",
        "span_policy_version": "hml_caption_span_30fps_round_half_up_v1"
      },
      "source_line": 1
    }
  ],
  "provenance": {
    "conversion_commit": "ea668b7073de3d86894b17fa84cb8b456e06a9ed",
    "kimodo_commit": "6bb58488037dd65360ff0c5d1692b403a23309f7",
    "source_manifest_sha256": "bc1bb0e8df4e139a99dbe326956b8a16a7f1b58c66a3d9c6509b6a2b6cec3e8c"
  }
}
```

MotionFix row 示例：

```json
{
  "schema_version": "hy273_multitask_manifest_v1",
  "uid": "motionfix:000362",
  "dataset": "motionfix_k273",
  "split": "train",
  "task_capabilities": ["motion_edit", "motion_edit_with_control"],
  "source_motion": {
    "motion_uid": "motionfix:000362:source",
    "base_motion_id": "007102",
    "timestamp_sec": [16.35, 20.35],
    "coordinate_frame": "per_sequence_hml_canonical_then_raw_k273",
    "physical_transform_group_id": "motionfix:000362:source_independent",
    "k273_asset": {
      "path": "/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22/train/000362_source.npy",
      "sha256": "...",
      "frames": 120,
      "fps": 30.0,
      "feature_dim": 273,
      "representation_version": "kimodo273_smplx22_v1"
    },
    "hy201_asset": {
      "path": "/mnt/afs/mogo_base/datasets/MotionFix/hymotion201_o6dp_hml272/train/000362_source.npy",
      "sha256": "...",
      "frames": 120,
      "fps": 30.0,
      "feature_dim": 201,
      "representation_version": "hymotion201_o6dp_hml272_v1"
    }
  },
  "target_motion": {
    "motion_uid": "motionfix:000362:target",
    "base_motion_id": "005808",
    "timestamp_sec": [15.95, 19.95],
    "coordinate_frame": "per_sequence_hml_canonical_then_raw_k273",
    "physical_transform_group_id": "motionfix:000362:target_independent",
    "k273_asset": {
      "path": "/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22/train/000362_target.npy",
      "sha256": "...",
      "frames": 120,
      "fps": 30.0,
      "feature_dim": 273,
      "representation_version": "kimodo273_smplx22_v1"
    },
    "hy201_asset": {
      "path": "/mnt/afs/mogo_base/datasets/MotionFix/hymotion201_o6dp_hml272/train/000362_target.npy",
      "sha256": "...",
      "frames": 120,
      "fps": 30.0,
      "feature_dim": 201,
      "representation_version": "hymotion201_o6dp_hml272_v1"
    }
  },
  "texts": [
    {
      "text_id": "motionfix:000362:instruction",
      "value": "raise your left hand higher and wave wider in the end",
      "kind": "relative_edit_instruction",
      "encoding_profile": "hytext_relative_edit_v1",
      "from_sec": null,
      "to_sec": null,
      "span_kind": "pair_instruction"
    }
  ],
  "pair": {
    "frame_policy_id": "independent_sequence_frame_v1",
    "framewise_aligned": false,
    "shared_world_frame": false,
    "output_gauge_policy": "shared_target_yaw_phi_v1",
    "source_frames": 120,
    "target_frames": 120,
    "length_relation": "equal",
    "default_time_relation": "normalized_progress",
    "official_pair_id": "000362"
  },
  "provenance": {
    "annotation_sha256": "8d999be2bffe086cfe2816aefdba944782dc3534e1c45e3fc4dcd5ac8755d203",
    "split_sha256": "1f91f945b76c85b2780fd9dbdce32bde1f14988a06ceecf16cb2eb4e02d0de45",
    "upstream_hml_converter_commit": "49f17a3b7c63835a2bc0b48e15f0b408dfece870",
    "hy201_converter_commit": "4bf40fe269478886712ef4fa7c37edf193416ce3",
    "k273_converter_commit": "ea668b7073de3d86894b17fa84cb8b456e06a9ed"
  }
}
```

`k273_asset` 与 `hy201_asset` 是两个独立 `AssetRef`，二者都必须有 payload SHA、shape、FPS 和 representation version。manifest builder 必须逐行执行 annotation join，而不是只按文件名拼路径：

```text
official pair uid
  -> annotation source/target base ID + timestamps + instruction
  -> split membership
  -> exact source/target K273 and HY201 assets
  -> shape/hash/frame-policy checks
```

`000000`、`000362` 是正向 golden rows；缺 annotation/asset 的 `006722` 是拒绝 golden row。任意字段串行、错 join 或路径与 UID 不一致都必须 fail closed。

### 4.3 为后续数据留出的字段

不要把 schema 固定成永远只有一个 source。逻辑上使用：

```text
target_motion
context_motions[]:
  role                 self_reference / other_actor / scene_reference / ...
  actor_id
  asset
  slot_present
  source_timebase
  target_to_source_time_map  # target query frame -> source frame coordinate
  world_from_asset     optional SE(3)
```

v1 可把单个 `source_motion` 解析成 `context_motions[0]`。未来 reaction 必须额外提供 actor/world transform 和相对空间 QA；只预留字段不等于已经解决 reaction。

### 4.4 Manifest 产物

```text
manifests/hy273_multitask_v1/train.jsonl
manifests/hy273_multitask_v1/val.jsonl
manifests/hy273_multitask_v1/test.jsonl
manifests/hy273_multitask_v1/schema.json
manifests/hy273_multitask_v1/summary.json
manifests/hy273_multitask_v1/rejected.jsonl
manifests/hy273_multitask_v1/accepted_captions.jsonl
manifests/hy273_multitask_v1/overlap_report.json
manifests/hy273_multitask_v1/manifest.sha256
```

JSONL 必须按 `(dataset, split, uid)` 稳定排序。大数组的 per-file SHA 只做一次并缓存到 asset index；之后所有 smoke/训练只校验 manifest hash 和抽样 payload hash，避免每次启动重读 6GB。

## 5. HumanML3D caption/window 流程

### 5.1 两级采样，避免 caption multiplicity bias

```text
uniform motion row
      |
      +--> uniform valid caption within this motion
                  |
                  +--> full caption    -> full target clip
                  +--> segment caption -> caption span clip
```

不要把 64,140 条 train caption 展开后均匀采样；那会让有更多 caption 的 motion 获得更高权重。

### 5.2 Span 解析规则

```text
NaN -> 0.0
(from=0, to=0)                 full motion
to > from >= 0                 candidate segment
其他非零组合                   invalid caption -> reject caption

start = floor(from_sec * 30 + 0.5)
end   = floor(to_sec   * 30 + 0.5)   # exclusive
start/end clamp to [0,T]
end-start < min_frames          reject caption for training
```

所有 clamp/reject 都写入 manifest summary。若一个 motion 的 segment caption 无效，但仍有 full caption，不应丢整条 motion。

resolver 必须输出一张稳定排序的 `accepted_captions.jsonl`，每行包含 `uid/text_id/start_frame/end_frame_exclusive/span_status/span_policy_version`；loader 与 stats builder 只消费这张表，禁止各自重新解释秒数。其 SHA 写入 unified manifest、stats manifest 和 checkpoint。summary 分 split 固定报告：

```text
asset_too_short
smooth_root_fallback_excluded
invalid_nonzero_span
clamped_end
segment_too_short_after_clamp
accepted_full / accepted_segment
```

### 5.3 为什么 segment 后必须重新提取 K273

Kimodo smooth root 是整段轨迹上的优化结果；velocity/contact 也依赖相邻帧。直接做：

```python
segment = full_k273[start:end]
```

会让 segment 继续携带 full clip 的 smooth-root 边界条件，末帧 velocity 还可能指向窗口外下一帧。

已对 8 个真实 train segment 做 `slice full K273` 与 `slice HY201 -> re-extract K273` 比较。各自做 frame-0 smooth-root shift 后，部分样本仍出现：

```text
smooth-root max difference       17.05 cm
joint-position max difference     9.79 cm
velocity max difference          96.08 cm/s
contact terminal difference       1.0
heading/global rotation difference 0
```

因此正确流程是：

```text
caption seconds
  -> HY201 frame window
  -> decode local rotations + root translation
  -> official SMPL-X22 FK/Kimodo extraction
  -> new self-consistent K273 segment
  -> root-origin/random-heading transform
```

原始 K273/HY201 均保持只读。实现可先 on-the-fly + worker LRU；正式长训建议生成可丢弃的 content-addressed derived-window cache，key 为：

```text
HY201 payload SHA256
+ [start_frame,end_frame_exclusive)
+ K273 converter commit
+ Kimodo commit
+ skeleton/config SHA256
+ fps
```

该 cache 只覆盖约 2.8K 条 segment annotation，不复制完整数据集，也不是新的 source of truth。

每个 derived tensor sidecar 必须保存 cache key、derived K273 payload SHA256、shape、extractor/config provenance，以及 official 273-channel audit 的 `passed` 和最大误差。命中 cache 时仍校验 sidecar；payload/hash/audit 任一不一致即拒绝，不能仅凭文件名复用。

segment 重提取也必须 fail closed：若某个短窗口触发 smooth-root solver singular、产生非有限值或未通过 official channel audit，v1 直接拒绝该 caption 并写入 `rejected.jsonl`，不能静默退回“直接切 K273”。若以后引入 raw-root fallback，必须换 extractor/version key 并单独报告数量。

### 5.4 Root-origin shift 的时机

必须先选 window 并重新提取 K273，再以该 window 的 frame 0 做 root-origin shift。若先 shift full clip 再切 segment，segment 的起点仍携带 full clip 的全局位移。

## 6. MotionFix 时间对齐策略

本节处理的是**同为 30 FPS 时 source/target 的原生 duration 不同**，不是 HumanML3D/MotionFix 之间的 FPS 对齐。当前两套数据合并不经过 temporal resampling；normalized-progress 只用于把不等长 source hidden context 映射到 target query。

### 6.1 对五种方案的判断

| 方案 | 判断 | 原因 |
|---|---|---|
| 过滤不等长 pair | 只允许 bring-up/pilot | 丢掉 6%左右 pair，且系统性丢 faster/slower/repetition |
| 截断到 common length | 禁止作为主方案 | 会直接删掉 instruction 指向的末尾事件或改变速度语义 |
| normalized-time 对齐 | 可作为 context adapter | 保留动作进度对应，但会弱化原生 duration；必须同时输入原始长度/时长比 |
| 物理域重采样后重提 K273 | 仅在模型强制同 T 时采用 | 数学正确，但会主动 time-warp source；不能冒充原始 pair |
| 模型原生支持不同长度 | **推荐主方案** | 完整保留 Ts/Tt 和 duration edit，也适合未来 continuation/reaction |

### 6.2 推荐 runtime contract

```text
source_motion       [B,K,Ts_max,273]
source_time_valid   [B,K,Ts_max]
source_present      [B,K]

target_motion       [B,Tt_max,273]
target_valid        [B,Tt_max]
requested_Tt        [B]
source_native_Ts    [B,K]
target_to_source_time_map  optional [B,K,Tt_max], source-frame coordinates
```

不要为了 collate 把 source 强制 pad/crop 成 target length。模型需要时可以让 target query 对 source tokens 做 masked cross-attention，或在 hidden token 域按 normalized time 插值；原始 K273 不被改写。

`normalized_progress_v1` 的方向和端点合同固定为 **target query -> source coordinate**。对每个 valid target frame `j`：

```text
if Ts > 1 and Tt > 1:
    u = j / (Tt - 1)
    source_coord[j] = u * (Ts - 1)
else:
    source_coord[j] = 0
```

`target_to_source_time_map` 的 runtime shape 是 `[B,K,Tt_max]`，单位是 source frame index。只允许在 `[0, Ts-1]` 的 valid source span 内对 hidden source tokens做线性插值；source padding 永远不能作为端点。target padding 的 map 值统一填 0，并由 `target_valid=false` gate。`Ts=Tt` 必须走严格 identity gather path，不经过浮点插值；off-by-one 与 material-duration pair 分层记指标。该映射是 v1 对齐 heuristic，不代表 pair 被声明为 framewise aligned。

### 6.3 如果必须重采样

优先起点：

```text
official MotionFix raw rotations/root translation
  或现有 upstream HY201 local rotations/root translation
```

变换：

```text
root translation      linear/cubic interpolation in meters
local joint rotation  SO(3) SLERP per joint
                       不能对 cont6d 直接 lerp
```

之后完整重新运行：

```text
FK
heading
smooth root
local joint positions
global cont6d
global velocities
foot contacts
```

不推荐从 273D 直接线性插值。若上游 HY201 不可用，可从 K273 的 global rotations 和 reconstructed root 逆出 local rotations/root，但必须先通过 roundtrip threshold；它是 fallback，不是首选 source。

### 6.4 Crop 策略

当前 MotionFix source/target 最大 151/150 帧，小于模型 `max_T=300`，所以 v1 **完全不做 MotionFix temporal crop**。这同时避免把 `in the end`、`get up later` 等 instruction 的关键区域裁掉。

未来长数据必须裁剪时：

- shared-world pair：使用同一物理时间窗口；
- independent-duration pair：采同一 normalized-progress 窗口，再分别映射到 source/target；
- manifest/runtime 显式记录两个窗口和 time map；
- 不允许两个 dataset object 各自随机 crop。

## 7. 成对几何增强

### 7.1 两种 frame policy 必须分开

```text
Policy A: independent_sequence_frame
  当前 HumanML3D / 当前 MotionFix
  每条 sequence 已有自己的起点规范化 frame

Policy B: shared_world_frame
  未来真实 before/after 同世界数据、reaction、scene motion
  所有 actor/source/target 共享一个 world frame
```

### 7.2 当前 MotionFix 的正确处理

当前 pair 不是 shared-world。推荐：

```text
source -> 自己的 frame-0 smooth-root XZ shift
target -> 自己的 frame-0 smooth-root XZ shift

sample one phi ~ Uniform(-pi, pi)

delta_source = phi - heading_source[0]
delta_target = phi - heading_target[0]

rotate source by delta_source
rotate target by delta_target

frame_gauge_dir = [cos(phi), sin(phi)]
```

这里 source/target delta 可以不同，但随机变量只有一个 `phi`，最终 frame 一致。它与“分别随机 heading”有本质区别。

为避免把“共享最终 gauge”误写成“共享物理变换”，字段必须拆开：

```text
frame_policy_id             pair 两边一致 = independent_sequence_frame_v1
physical_transform_group_id 当前 source/target 不同
output_gauge_draw_id        pair 两边一致，对应同一个 phi draw
applied_yaw_delta           每个 slot 单独记录，可不同
```

校验的是相同 `frame_policy_id`、相同 `output_gauge_draw_id/phi`，以及两边变换后首帧 heading 都等于 `phi`；绝不能校验 source/target delta 相等。未来 `shared_world_frame` 才要求相同 SE(2) physical transform group 和相同 delta。

为什么不继续使用旧方案“从 source 取一个 origin/delta，再原样施加到 target”：因为上游 source/target 已分别 canonicalize，当前两者之间没有可信的原始 world offset；把残余 smooth-root/heading 差强行当作世界关系反而会学习 preprocessing residual。

### 7.3 未来 shared-world 数据

若 manifest 声明 `shared_world_frame=true`：

```text
one translation delta
one yaw delta
apply exactly the same SE(2) transform to every motion slot and target
```

两类 policy 必须通过 manifest 字段分发，不能按 dataset 名硬编码。

### 7.4 `c_dir` 语义修正

当前代码的 `c_dir` 来自 target augmentation。多任务协议中应更名或至少定义为：

```text
frame_gauge_dir = 这次 sample 被放进哪个全局 yaw frame
```

它不是“source 原始朝向”，也不是泄漏的“GT target desired heading”。如果用户还提供目标 heading，应通过 Kimodo hard control 表达，或使用单独的 `desired_target_dir + valid` 字段。

## 8. 统一 normalization stats

### 8.1 两种训练场景

#### 场景 A：续训现有成功 checkpoint

必须继续使用现有 HumanML3D stats：

```text
/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/
derived_stats/redenoise_kimodo_like_v1
```

checkpoint 中 input/output projection 已经适应该尺度；中途换 stats 等价于改变模型输入输出定义。

#### 场景 B：按当前计划从头训练多任务 v2

推荐重算 `hy273_multitask_stats_v1`，并从 Stage A 第 0 step 起一直冻结。

### 8.2 Stats 应统计谁

只统计生成 target：

```text
HumanML3D target windows
MotionFix target motions
```

不要把 MotionFix source 再计入 output normalizer。source context 使用同一 normalizer 映射即可，但它不是生成分布；应单独报告 source 在该 normalizer 下的 z-score coverage。

### 8.3 Task-weighted moments

若 steady-state target-domain/train-stream mix 是 HML 80%、MotionFix 20%，先分别算 domain moments，再组合：

```text
mu = 0.8 * mu_hml + 0.2 * mu_mf_target

E[x^2] = 0.8 * (sigma_hml^2 + mu_hml^2)
       + 0.2 * (sigma_mf^2  + mu_mf^2)

sigma = sqrt(E[x^2] - mu^2)
```

不要简单把所有 frame 拼起来，否则 HML 约 4.75M train frames 会按文件体量主导，而不是按实际 task batch 比例主导。task weights 必须进入 stats manifest，后续改 sampling ratio 时要记录 domain-shift，而不是偷偷覆盖 stats。

HML 域内 moments 必须复现训练采样合同，而不是展开全部 caption 后均匀统计：

```text
uniform accepted motion m
  -> uniform accepted caption c within m
  -> resolve c 对应的 full/segment window
  -> window 中每个 valid frame 进入 moments

window sampling mass:
  q(m,c) = 1 / N_accepted_motions / N_accepted_captions(m)

frame-population moments:
  mu_num = sum_(m,c) q(m,c) * sum_f x[m,c,f]
  mu_den = sum_(m,c) q(m,c) * T[m,c]
  mu = mu_num / mu_den

  E2_num 同理把 x 换成 x^2
```

因此同一被选 window 内 frame 权重相同，长 window 对 normalizer moments 的总贡献更大。这里明确选择的是 **sampled-window frame-population measure**，不是声称它等于每个 optimizer step 内的 masked-mean loss measure；后者受 batch/bucket 组成影响，二者用途不同。实现可以解析加权枚举所有 accepted `(motion, caption, frame)`，或使用与 scheduler 完全相同且带固定样本数/seed 的 Monte Carlo；不能让 caption 数量或 length bucket 额外改变 row marginal probability。

Length bucket 只负责 grouping。bucket 被调度的长期概率必须与其中尚可采 row 的原始质量成比例；不得“每个 bucket 均匀”而把短/长动作重新加权。stats manifest 至少保存：

```text
accepted_caption_table_sha256
caption/span policy version
frame reduction = valid_frame_weighted
bucket marginal policy
domain/task weights
root/yaw transform policy
min-frame/fallback policy
stats builder commit and payload SHA256
```

需要用固定 `SamplePlan` trace 做 stats audit：跨所有 sampled windows 累积全局 `sum(x)/valid_frame_count` 与 `sum(x^2)/valid_frame_count`，不能先算每 batch mean 再等权平均。该 global ratio-of-sums 应与 stats builder moments 在预注册容差内一致；训练 loss 仍单独记录其 per-batch reduction 合同。

### 8.4 与训练 transform 一致

stats builder 必须复用与 loader 相同的：

```text
caption span policy
root-origin policy
yaw quadrature/random-heading 等价分布
fallback/min-frame filter
target-only domain mix
```

当前 HML stats builder 包含 train-only + root shift + 四个 yaw target，这是正确基础；但它只跳 4 个 fallback，未完全复用 loader 的 `min_frames=16`，联合版应修正。

### 8.5 实测 domain shift

对 MotionFix train 随机 1,024 pair 的 source/target 做相同 root shift + 四方向 yaw quadrature，并用现有 HML stats 对比：

- joint positions/global rotations 的大多数 std ratio 接近 1；
- MotionFix planar root 与许多 velocity channel 更窄，std ratio 中位数约 `0.38-0.66`；
- normalized mean shift 大多小于 `0.4 sigma`；
- contact mean：HumanML3D 全量约 `0.6643`，MotionFix 全量约 `0.8274`。

结论：旧 HML stats 用于 checkpoint-compatible editing 不会立刻越界；但从头联合训练仍应重算 task-weighted stats。contact 继续保持 raw 0/1，不参与 z-score；需按 task 单独监控 BCE 和 contact prior。

## 9. Runtime sample schema

### 9.1 Dataset 输出，仍在 unnormalized physical K273 域

```text
SampleSpec
  uid                    str
  dataset_id             int64
  train_stream_id        int64  # high-level scheduler/replay: HML_MIXED/MOTION_EDIT
  task_id                int64
  capability_id          int64  # sampler/eval routing; model embedding 默认不用

  text                   str
  text_kind              int64
  text_encoding_profile  str

  target_motion          [Tt,273] float32, unnormalized
  target_valid           [Tt] bool
  target_op_id           [Tt] int64  # PRESERVE/GENERATE/EDIT

  source_motion          [K,Ts,273] float32, unnormalized
  source_time_valid      [K,Ts] bool
  source_value_mask      [K,Ts,273] bool
  source_present         [K] bool
  source_role_id         [K] int64

  frame_gauge_dir        [2]
  frame_policy_id        int64
  output_gauge_draw_id   int64  # pair 内共享随机 phi 的 provenance ID
  target_applied_yaw_delta scalar
  source_applied_yaw_deltas [K]
  requested_target_len   int64
  source_native_lengths  [K]
  target_to_source_time_map optional [K,Tt], source-frame coordinates

  provenance             compact IDs/hashes
```

`target_op_id` 是 target-side 意图，不应被 source mask gate。未来 inpainting 的 known/unknown source value mask 与 source slot 是否存在也必须分开。

### 9.2 Task 映射

这里区分两个概念：

```text
train_stream_id = high-level 训练调度流，决定 HML_MIXED/MOTION_EDIT、dataset cursor 和 source-presence topology，不进入模型
task_id       = source 与 target 的语义关系，进入 task embedding
capability_id = 数据/评估路由标签，不必进入模型
```

| capability | `train_stream_id` | runtime `task_id` | source | target op | hard mask | text kind |
|---|---|---|---|---|---|---|
| T2M | HML_MIXED | GENERATE | absent | GENERATE | zero | absolute caption |
| Kimodo control | HML_MIXED | GENERATE | absent | GENERATE | compiler | absolute caption |
| MotionFix edit | MOTION_EDIT | EDIT | self reference | EDIT | zero | relative edit instruction |
| MotionFix edit + control | MOTION_EDIT | EDIT | self reference | EDIT | compiler | relative edit instruction |
| continuation/inbetween | 专用 stream | GENERATE 或后续专用 task | partial self context | PRESERVE/GENERATE | exact known frames 可同时 hard clamp | absolute/transition text |
| reaction | REACTION | REACTION | other actor | GENERATE | optional | interaction instruction |

硬控制不创建新 task。由此 T2M/control 的 separated CFG 始终固定 `task_id=GENERATE`，editing 的 source-only/source+text/source+control/all-condition 分支始终固定 `task_id=EDIT`；分支差只来自真实被 drop 的条件。

## 10. 完整 DataLoader / Tensor Information Flow

```text
                           Unified JSONL manifest
                                     |
                    +----------------+----------------+
                    |                                 |
             HumanML3D row                     MotionFix pair row
             motion + captions                 source + instruction + target
                    |                                 |
          sample motion once                 sample pair once
          sample caption inside row           keep native Ts / Tt
                    |                                 |
          resolve full/segment window          no temporal crop in v1
                    |                                 |
          segment: HY201 re-extract K273        source [Ts,273]
          full: raw K273                        target [Tt,273]
                    |                                 |
                    +---------------+-----------------+
                                    |
                       frame-policy-aware transform
                       - root origin after window
                       - one shared target yaw phi
                                    |
                     target/source unnormalized K273
                                    |
                  stream-homogeneous physical-domain collate
                                     |
       target_un [B,Tt_max,273]      source_un [B,K,Ts_max,273]
       target_valid [B,Tt_max]       source_time_valid [B,K,Ts_max]
       requested_target_len [B]
                                     |
                if intended control: compiler(
                  target_un, lengths=requested_target_len, keyed seed)
                                     |
                hard_obs_un [B,Tt,273], hard_mask [B,Tt,273]
                assert mask is empty on target padding
                                     |
                 +-------------------+-------------------+
                 |                   |                   |
          target normalizer    source normalizer   hard_obs normalizer
          contact stays 0/1    contact stays 0/1   contact stays 0/1
                 |                   |                   |
       target [B,Tt,273]    source [B,K,Ts,273]   hard_obs [B,Tt,273]
                 |                   |                   |
                 +-------------------+-------------------+
                                     |
                  hard_mask + source context + text cache
                                    |
                    build flow/noise state in target length Tt
                                    |
                  overwrite only where hard_mask is true
                                    |
                       existing root -> body denoiser
                                    |
                             loss against target
```

关键顺序：

```text
select text/window
  -> physically consistent re-extraction/resampling
  -> geometric transform
  -> physical-domain collate with lengths/valid masks
  -> build controls from transformed target
  -> assert no control on padding
  -> normalize target/source/hard_obs
  -> noise/flow
```

不能先构造 control obs 再旋转 target，也不能在 normalized 273D 上做 SO(3) 或时间重采样。

## 11. Task sampler、batching 与 DDP

### 11.1 不使用裸 `ConcatDataset`

推荐 high-level stream-first scheduler：

```text
global optimizer step
  -> deterministic/synchronized choose train_stream_id
  -> choose corresponding dataset sampler
  -> emit one source-topology-homogeneous global batch
       HML_MIXED: source absent; per-sample T2M/control
       MOTION_EDIT: source present; per-sample edit condition pattern
```

原因：

- HumanML3D 与 MotionFix grain 不同；
- source/target length 和字段不同；
- 同 stream batch 更容易做 bucketing；
- DDP 所有 rank 必须走相同模型 branch。

### 11.2 DDP 状态

随机语义不能由 worker 调用顺序或 prefetch 时机决定。采用 stateless keyed RNG；每个 sample 的派生随机流固定为：

```text
sample_key = hash(
  manifest_sha256,
  run_seed,
  global_sample_ordinal,
  train_stream_id,
  task_id,
  uid,
  random_stream_id
)

random_stream_id in:
  caption / window / yaw_phi / control_pattern / condition_pattern / flow_t / noise
```

状态机合同：

1. `global_optimizer_step` 按模型/训练专项文档的整数 weighted-deficit curriculum 选择 `HML_MIXED` 或 `MOTION_EDIT`；只有这一 high-level 选择有小于一个 optimizer step 的 prefix discrepancy 保证。该 optimizer step 的所有 gradient-accumulation microbatch 和所有 rank 使用同一 high-level stream。HML batch 内再由 stateless SamplePlan 做独立逐样本 Bernoulli，选择 T2M 或非空 Kimodo control；它只保证期望比例和预注册 binomial 容差，不做 quota balancing。两者都映射为 `task_id=GENERATE`。
2. 每个 train stream 有独立 `(cycle, batch_cursor, per_bucket_cursor, carry_queue)`。每个 cycle 对各 length bucket 内 row 做由 `manifest SHA + run_seed + stream ID + cycle` 驱动的确定性 permutation；bucket permutation 明确受 `run_seed` 控制，不采用 seed-invariant 顺序。
3. bucket schedule 按剩余 row/batch 质量做确定性 weighted-without-replacement 调度；尾部 row 确定性 carry 到下一 cycle，不静默 drop/duplicate。这样 bucketing 只改变 grouping，不改变 row marginal。
4. 一个 global batch 先在主进程生成完整 `SamplePlan`：固定 UID、caption、resolved window、`phi`、control pattern 和 edit condition pattern；之后才按固定规则分片给 rank/worker。worker 只 materialize 已决定的 plan。
5. `global_sample_ordinal` 是同一 world size/global-batch 合同下单调递增的 plan 序号，不由 worker 局部计数生成。
6. checkpoint 只允许写在完整 optimizer step、optimizer/EMA 更新完成且所有 rank barrier 后；保存“下一步”状态，不能写在 gradient accumulation 中间。

checkpoint 至少保存/校验：

```text
run seed, next global_optimizer_step, next global_sample_ordinal
per-stream cycle/batch_cursor/per-bucket cursor/carry queue
high-level debt_hml/debt_edit int64, realized stream counts,
PROB_SCALE=5_000_000,
high_level_schedule_version=hy273_multitask_weighted_deficit_v1,
hml_inner_schedule_version=hy273_hml_stateless_bernoulli_v1,
current target probability units
HML inner E_none/V_none/N_none accumulators per stage and full run
deterministic bucket-plan version and bucket-plan SHA256
global batch / microbatch / grad-accumulation / world size
manifest, accepted-caption, stats and text-cache SHA256
sampler/RNG contract version
```

保证范围是：**相同 world size、global batch 和 accumulation 合同下，uninterrupted 与 resume 的 UID/train-stream/task/caption/window/yaw/control/condition trace 逐项相等**。v1 不承诺跨 world-size bitwise replay；world size 变化必须启动新 run lineage，或以后实现显式 global-plan repartition converter。

新增 source/task projection 时，要么每 step 都执行 projection 再用 zero gate，要么显式验证 `find_unused_parameters=True`；仅同步 task choice 不能自动解决未使用参数和 optimizer group 恢复问题。

### 11.3 Length bucketing

按 `(train_stream_id, target_len_bucket, source_len_bucket)` 组 batch；bucket schedule 必须遵守 11.2 的 row-marginal 合同。padding mask 分开：

```text
target_valid != source_time_valid
```

不允许用 target mask 代替 source mask。MotionFix v1 最大约 150 帧，保留完整 pair 的吞吐成本可控。

### 11.4 EDIT condition-pattern 是唯一 dropout 决策源

Stage B2/C 的 EDIT batch 先且只采一次：

```text
70% source + text
10% source only
15% source + text + hard control
 5% source + hard control
```

EDIT adapter 的通用 `text_dropout_prob` 固定为 `0`，source v1 也不做独立 dropout；空文本和 hard control 是否存在完全由上述 sampler 决定，禁止之后再套用 GENERATE 的 0.1 text dropout。GENERATE/T2M/control 继续使用成功基线的 `text_dropout_prob=0.1`。训练日志和 checkpoint 必须按 task 累计四种 EDIT pattern 的目标计数与实际计数，以验证 resume 前后总体比例没有漂移。

对两种声明含 hard control 的 EDIT pattern，compiler 必须使用 `none_prob=0` 并只从非空 control modes 采样；每个样本编译后断言 `hard_mask[target_valid].any()`。声明无 control 的两种 pattern 直接构造全零 mask，不调用可能产生非空 mask 的 compiler。日志分开记录 `intended_condition_pattern` 与 `realized_hard_mask_nonempty`，二者不一致立即 fail closed。complete Stage-2 的 `none_prob=0.10` 只属于 GENERATE control curriculum，不能继承到 EDIT 条件组合内部。

## 12. HYText cache 合并

当前 cache 的 key 仅为 normalized text 的 SHA1，见 [hytext_cache.py](/mnt/afs/mogeflow-control/models/raw_motion/hytext_cache.py:23)；当前 builder 的 system prompt 只要求总结绝对 human motion，见 [cache_hy273_hytext_embeddings.py](/mnt/afs/mogeflow-control/tools/cache_hy273_hytext_embeddings.py:25)。这不足以安全加入 relative edit instruction。

统一 cache 应从 manifest 收集 text，并使用：

```text
cache_key = hash(
  encoder_identity,
  prompt_template_version,
  text_encoding_profile,
  normalized_text
)
```

至少两个 profile：

```text
hytext_absolute_motion_v1
hytext_relative_edit_v1
```

两者分别包含 empty row，用于各自 task 下的 text dropout/CFG。task embedding 仍在模型中显式提供；不要只靠 prompt prefix 猜任务。

## 13. Control compiler 与数据的边界

manifest 不存随机 hard mask。运行时从**变换后的 unnormalized target** 合成：

```text
hard_obs_un, hard_mask = compiler(
  transformed_target_un,
  lengths=requested_target_len,
  generator=keyed_control_generator,
  none_prob=0 for an intended-controlled EDIT branch
)
assert not (hard_mask & ~target_valid[..., None]).any()
hard_obs = normalizer(hard_obs_un)
```

这样 root/path/end-effector/full-pose/contact 均与最终 frame gauge 一致。MotionFix pure edit 默认 `hard_mask=0`；editing+control 才从同一 target 合成现有 Kimodo constraints。

## 14. 评估数据协议

### 14.1 主报告必须给精细 overlap 标签

```text
pair_protocol             official pair-level / custom subset
pair_id_seen              bool
instruction_seen          bool, versioned normalized exact text
source_clip_seen          bool, base ID + resolved 30-FPS frame span
target_clip_seen          bool, base ID + resolved 30-FPS frame span
source_base_seen          bool
target_base_seen          bool
source_target_base_pair_seen bool
exact_pair_clip_seen      bool
exact_instruction_target_clip_seen bool
target_length_protocol    GT-requested / source-length / predicted-length
frame_contract            independent-canonical / shared-world
```

主表至少同时给全量计数和按这些标签分层的结果。`pair_id_seen=false` 不能替代 motion-content overlap；“instruction-target unseen”只有在明确指 `exact_instruction_target_clip_seen=false` 时才允许使用。

### 14.2 不等长 benchmark

如果 sampler 使用 GT target length，结果必须命名为：

```text
MotionFix editing conditioned on requested target length
```

不能声称模型从 instruction 自主预测了 duration。若只评估等长 952 test：

```text
MotionFix equal-length-952 subset
```

并在同一 952 ID 上重跑 `source-copy`、`relative-instruction-only OOD diagnostic` 和 `source+instruction model`，禁止直接与 full 1,013 数字比较；relative instruction 不能冒充普通 T2M text-only baseline。

### 14.3 Motion-disjoint sanity

当前 official test 几乎没有严格 base-motion-disjoint 样本。可额外构建一个小型 component-disjoint sanity split：

1. 以 base motion ID 为节点、MotionFix pair 为边；
2. 选低连接度 component 作为 64-128 pair holdout；
3. 只在该辅助实验中，从 HML/MotionFix train 删除 holdout component；
4. 主训练和官方 split 不变；
5. 结果只用于 strict generalization sanity，不替代官方 benchmark。

## 15. 实施文件清单

```text
data/hy273_manifest_schema.py
  AssetRef / TextAnnotation / PairMeta / TrainingRow dataclass
  schema version + fail-closed validation

tools/build_hy273_multitask_manifest.py
  合并 HML split/text、MotionFix annotation/split、K273 manifests
  provenance/hash/overlap/rejected reports

data/hy273_multitask_dataset.py
  HML row adapter
  MotionFix pair adapter
  runtime SampleSpec

data/hy273_temporal.py
  caption span resolver
  HY201/K273 window re-extraction
  optional SO(3) resampling

data/hy273_pair_transforms.py
  independent_sequence_frame
  shared_world_frame
  one-phi paired yaw

data/hy273_task_sampler.py
  synchronized stream-first DDP scheduler
  per-stream length bucketing/cursors/carry queues

data/hy273_collate.py
  separate target/source padding and masks

tools/build_hy273_multitask_stats.py
  train-only target moments
  task-weighted domain combine
  full/global-root/local-root stats

tools/cache_hy273_hytext_embeddings.py
  增加 --manifest 和 encoding profiles

tools/audit_hy273_multitask_data.py
  schema/coverage/span/pair/time/gauge/overlap/stats audit

tools/compare_hy273_nonregression.py
  verify baseline/per-case manifest hashes
  paired bootstrap + regime/subtype/metric gate matrix
```

现有 [Kimodo273TextDataset](/mnt/afs/mogeflow-control/data/kimodo273_datasets.py:120) 保留给成功基线复现，不在原类里继续堆 pair/edit 分支。

## 16. 测试与 Go/No-Go 门禁

### 16.1 Manifest

- train/val/test UID 唯一且稳定；
- HML `26,846` assets coverage；
- MotionFix `6,730` valid pair coverage，`006722` 进入 rejected；
- 所有 path 为绝对只读 path；
- representation/fps/dim/frame contract 一致；
- 两套最终 manifest 的每条资产均为 `fps=30`，联合 loader 不触发 resampling；
- 30 FPS lineage audit 的 summary/manifest SHA、MotionFix timestamp 误差和 HumanML3D 30/20 帧数比均复现；
- future asset 若 `fps != 30` 默认拒绝，不能只改 metadata 或直接插值 K273；
- future asset 的 `source_fps` 必须来自可验证 provenance；当前通用 converter 的 `--fps` 不得被当作 source-FPS 认证或重采样步骤；
- conversion/file/manifest hash 可复现；
- overlap report 数字固定。
- `000000`、`000362` annotation/asset join golden tests 通过，`006722` 是唯一缺失拒绝 golden row；
- K273 与 HY201 AssetRef 均有独立 payload hash/shape/version。

### 16.2 HumanML3D span

- full caption 返回 full clip；
- segment seconds 正确映射到 30 FPS；
- invalid/clamped/too-short 分类固定；
- summary 固定为 90 个 clamped end、18 个 segment-too-short 和 16 个完整短 asset，并按 split 拆分；
- segment re-extract 后 official channel audit 通过；
- accepted-caption table 同时被 loader 与 stats builder消费且 SHA 一致；
- caption sampling 不改变每 motion 的期望采样权重；
- mirrored `M*` caption/span 处理一致。

### 16.3 MotionFix pair

- source/target 不独立抽随机 yaw；
- one-phi transform 后两边 `frame_gauge_dir` 一致；
- independent frame policy 下 physical transform group/delta 不同但 output gauge draw 相同；
- native Ts/Tt 和 masks 均保留；
- no-crop v1 保留完整 instruction 语义；
- 32 条 pair 的 source/target/instruction 人工可视化；
- current independent-canonical lineage 与 official evaluator roundtrip 被明确验证。

### 16.4 Normalization

- stats 只读 train split；
- target-only、task weights、segment policy 写入 manifest；
- contact mean=0/std=1 且 normalize 后仍为 0/1；
- source/target z-score coverage 无非有限值和极端越界；
- Stage A/B1/B2/C 使用同一 stats SHA。
- 固定 SamplePlan 上 global ratio-of-sums moments 与 builder 一致，不拿 per-batch loss mean 冒充 stats measure；

### 16.5 DDP/replay

- 同 world-size 的 1-rank、8-rank 各自 uninterrupted/resume trace 完全一致；v1 不宣称 1-rank 与 8-rank 跨拓扑序列相同；
- resume 后 UID/train-stream/task/caption/window/yaw/control/condition pattern trace 连续且逐项相等；
- high-level `HML_MIXED/MOTION_EDIT` 的整数 weighted-deficit state 与逐阶段概率单位一致，prefix discrepancy 小于一个 optimizer step；HML 内逐样本 T2M/nonempty-control 保持独立 stateless Bernoulli，realized count 只要求落在 `abs(N-E)<=max(1,6*sqrt(V))` 的预注册容差内，两者 model task 都是 GENERATE；
- intended-controlled EDIT 的 compiler `none_prob=0` 且 valid hard mask 必非空；
- 所有 compiler hard mask 在 target padding 上为零；
- HML_MIXED 中所有 T2M/control 样本的 source branch zero-gate identity；
- EDIT batch 不能把 source 塞进 `hard_obs`；
- task/text/source shuffle 反事实测试能区分条件作用。

## 17. 当前 Go/No-Go 判断

```text
GO:
  实施统一 manifest/schema
  修 HumanML3D segment pipeline
  实施 pair-aware loader/collate
  构建 joint stats 与 profile-aware text cache
  equal-length + variable-length fake/real data smoke
  通过 pair coverage/frame-policy/32-pair visual 后运行内部 K273 editing pilot

NO-GO for formal long training:
  source/target frame policy 未写入 manifest
  仍把 segment caption 配 full motion
  仍把 source 当 hard_obs
  仍静默 truncate material unequal pairs
  accepted-caption/derived-cache/joint-stats artifact 未通过 QA
  同 topology uninterrupted/resume trace 不一致
  non-regression baseline artifact status != ready

NO-GO for official MotionFix benchmark claim:
  official evaluator conversion/gauge roundtrip 未通过
  evaluator commit/checkpoint/protocol hash 未固定
  equal-length subset 被冒充 full 1,013-pair benchmark
```

## 18. 固定的代码/数据依据

```text
HY201 -> K273 repo commit
  ea668b7073de3d86894b17fa84cb8b456e06a9ed

Kimodo reference commit
  6bb58488037dd65360ff0c5d1692b403a23309f7

MotionFix HML272 upstream repo HEAD
  49f17a3b7c63835a2bc0b48e15f0b408dfece870

272 -> HY201 repo HEAD
  4bf40fe269478886712ef4fa7c37edf193416ce3

MotionStreamer 272D HumanML3D dataset commit
  a8787eaac0b2247e26f1e58ef824cd8c0c3351fc

272D representation processing repo commit
  69750e94887800f0bcc84111e8a198fccc11af64

MotionStreamer repo commit
  9ecbb2569c3cfa9b91034f17cb6478153b9e65c7

Current MotionFix -> MotionStreamer272 converter commit
  efdb65f583095a2fa601cb7671ea9f3a99aa3894

HumanML3D K273 manifest SHA256
  bc1bb0e8df4e139a99dbe326956b8a16a7f1b58c66a3d9c6509b6a2b6cec3e8c

MotionFix K273 manifest SHA256
  b8bcf986be93a4f8a65cd55de69a14aef77c420f4a7c1d653d9ab5314c03a43b

MotionFix annotation SHA256
  8d999be2bffe086cfe2816aefdba944782dc3534e1c45e3fc4dcd5ac8755d203

MotionFix split SHA256
  1f91f945b76c85b2780fd9dbdce32bde1f14988a06ceecf16cb2eb4e02d0de45

Traditional HumanML3D 20 FPS comparison index
  /mnt/afs/HumanML3D/index.csv
  80226910d7259655ab5f411fd807b7651037341b4806fde025946a74d62b1065

30 FPS adversarial review
  /mnt/afs/mogeflow-control/outside_doc/HY273_30fps_plan_review_R6.md
  gpt-5.6-sol / max / GO

成功 Stage-1 resolved config SHA256
  da09bc3d0a7f1ee2559d75162c877f7c317d59ea4fa47bcf1b1b6e308628e2fa

成功 Stage-1 200K checkpoint SHA256
  5cc0413201a5652c119a822fc44c13833c11df79ec34e560eac5abd66a21a55d

成功 complete Stage-2 resolved config SHA256
  b194ad7ff5c293d4022f1e4b2afff949adbafcc5c74683b34e7bbc4cc5510b2e

成功 complete Stage-2 400K checkpoint SHA256
  d5f00ec15888e1dc3ca9f8c38c8ef436ec6524397ae0257a01fc48ce3542b2f4

成功 complete Stage-2 source YAML SHA256
  configs/redenoise_kimodo_complete_stage2_control.yaml
  f53b22b779262119d93a9fd76e988897cd628e03cb84130e3d1afe1ee712be40
```

这些 hash 与 [HY273_multitask_nonregression_baseline_v1.json](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_nonregression_baseline_v1.json)（当前 SHA256 `2332badc52da7c603c8550e43e432be43dd0f858e7bc88e62df6160c59db47c0`）用于钉住当前 non-regression baseline；不能只引用训练脚本默认值。名字相近的 `redenoise_kimodo_like_stage2_control.yaml` 不是 complete control 合同。

## 19. 对抗审核修订状态

R2 使用 `gpt-5.6-sol / max`，结论为 `CONDITIONAL GO`，报告见 [HY273_multitask_data_and_model_review_R2.md](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_data_and_model_review_R2.md)。r3 已关闭其文档级 High/Medium findings：错误 pair 示例、HY201/span provenance、stats 域内权重、time-map 方向、DDP replay 状态机、EDIT 二次 dropout、overlap 标签和 span split QA。

R3 使用 `gpt-5.6-sol / max`，结论为 `CONDITIONAL GO`，报告见 [HY273_multitask_data_and_model_review_R3.md](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_data_and_model_review_R3.md)。r4 修订关闭了其六个 Medium 和一个 Low：`train_stream_id/task_id`、受控 EDIT 的 `none_prob=0`、compiler/normalizer/padding 顺序、stats measure 与 batch loss 区分、独立 physical transform 与共享 output gauge、版本化 overlap key，以及 K>1 role slot gating。

R4 使用 `gpt-5.6-sol / max`，没有 Critical/High/Medium，结论为 `CONDITIONAL GO`，报告见 [HY273_multitask_data_and_model_review_R4.md](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_data_and_model_review_R4.md)。其两个 Low 已在当前版关闭：non-regression baseline 增加四类 typed gate artifact 的 path/SHA/count/protocol identity 和 `ready` 校验规则；per-case manifest 明确定义 record/case-key schema、UTF-8+末尾 LF digest preimage 以及 ordered/sorted 顺序。

R5 使用 `gpt-5.6-sol / max` 对上述两个修复做收口审核，没有任何残余 Critical/High/Medium/Low，报告见 [HY273_multitask_data_and_model_review_R5.md](/mnt/afs/mogeflow-control/outside_doc/HY273_multitask_data_and_model_review_R5.md)。结论为：design/schema implementation `GO`，整体因正式长训和 official benchmark 门禁仍为 `CONDITIONAL GO`。

R6 使用 `gpt-5.6-sol / max` 专项复核 30 FPS lineage，报告见 [HY273_30fps_plan_review_R6.md](/mnt/afs/mogeflow-control/outside_doc/HY273_30fps_plan_review_R6.md)。它独立复算最终 manifest/NPY、`272 -> HY201 -> K273` 的逐文件 `T`、MotionFix timestamp residual 和 HumanML3D 30/20 帧数比，确认当前 pinned 数据的 `runtime_fps=30`、`merge_resampling=none` 为 `GO`。其唯一 Low 已转成 P0 实施门禁：未来数据必须认证 `source_fps` 并 fail closed，不能把 converter 的 `--fps` 当成重采样。

尚未关闭的是需要代码/产物/实验的门禁：真实 manifest 与 derived-window cache、accepted-caption/stats 一致性、同拓扑 1/8-rank resume trace、T2M/控制 non-regression CI、official MotionFix evaluator roundtrip。它们不能由设计文档代替。
