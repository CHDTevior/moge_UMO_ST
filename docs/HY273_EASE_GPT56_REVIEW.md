结论：`GO（无 blocker）`。

当前实现的核心科学语义正确，可以进入正式 8 卡训练准备。但在直接启动 150K 长跑前，应先补一次真实 save→reload 恢复门禁。

## Blocker

无。

## 应修复

1. `ConditionBatch` 没有在类型合同层禁止 Edit 携带 Ease。

   [hy273_multitask_condition.py:224](/mnt/afs/mogeflow-control/models/raw_motion/hy273_multitask_condition.py:224)只检查 absent Ease 必须为零；Edit 分支 [hy273_multitask_condition.py:300](/mnt/afs/mogeflow-control/models/raw_motion/hy273_multitask_condition.py:300) 到 [hy273_multitask_condition.py:327](/mnt/afs/mogeflow-control/models/raw_motion/hy273_multitask_condition.py:327)没有要求 `ease_present=False`。只读构造探针确认 Edit+Ease 会通过 `validate()`。

   当前正式路径仍安全：dataset 在 [hy273_multitask_manifest_dataset.py:271](/mnt/afs/mogeflow-control/data/hy273_multitask_manifest_dataset.py:271)拒绝，采样 CLI 在 [sample_hy273_multitask.py:911](/mnt/afs/mogeflow-control/sample_hy273_multitask.py:911)拒绝。因此不是 blocker，但应在合同层 fail-closed，并增加负例测试，防止程序化调用改变 Edit replay 语义。

2. Ease stats 尚未纳入科学资产/恢复语义校验。

   `validate_assets()` 的必需资产列表 [train_hy273_multitask.py:1454](/mnt/afs/mogeflow-control/train_hy273_multitask.py:1454) 和返回的 asset identity [train_hy273_multitask.py:1548](/mnt/afs/mogeflow-control/train_hy273_multitask.py:1548)均不包含 Ease stats；跨阶段 base contract 还显式排除了整个 Ease section [train_hy273_multitask.py:876](/mnt/afs/mogeflow-control/train_hy273_multitask.py:876)。启动器仅检查 `metadata.json` 存在 [train_hy273_kencoder_stage_bc_ease_control_ddp8.sh:38](/mnt/afs/mogeflow-control/scripts/launch/train_hy273_kencoder_stage_bc_ease_control_ddp8.sh:38)。

   实际默认资产是正确的：train split、21,454 行、64,084 caption occurrence、23,626 个唯一片段，见 [metadata.json:2](/mnt/afs/mogo_base/datasets/HY273_multitask_v1/derived_stats/hy273_ease_stats_v1/metadata.json:2)。但 fresh fork 可以换成任意格式兼容的统计；in-stage resume 又会在 strict load 时用 checkpoint buffer 覆盖当前统计 [train_hy273_multitask.py:4401](/mnt/afs/mogeflow-control/train_hy273_multitask.py:4401)，造成配置路径与实际 buffer 语义不一致。

   建议绑定并比较 source manifest、split、row count、Mean/Std 数值即可；这是科研复现语义，不是 SHA/防篡改要求。

3. Stage-BC 的真实保存恢复证据尚未闭环，且现有通用 resume gate 不能替代。

   给定 smoke 的 [metrics.jsonl:1](/mnt/afs/mogeflow-control/outputs/hy273_text_fusion/hy273_kencoder_stageBC_ease_smoke10_fix_20260728_1925/metrics.jsonl:1)完整跑到 250010，但目录中没有保存 checkpoint，因此没有实证覆盖 Ease Adam、EMA、batcher 和下一批次的 reload continuation。

   同时，通用 DDP resume gate 在计算 `context_active` 后 [gate_hy273_multitask_phase_resume.py:461](/mnt/afs/mogeflow-control/tools/gate_hy273_multitask_phase_resume.py:461)，却在 [gate_hy273_multitask_phase_resume.py:488](/mnt/afs/mogeflow-control/tools/gate_hy273_multitask_phase_resume.py:488)引用未定义的 `source_present`，运行会报错；它也未覆盖 Ease 梯度屏蔽和 `ease_update_count`。相关测试只覆盖哈希/比较辅助函数 [test_multitask_phase_resume_gate.py:20](/mnt/afs/mogeflow-control/tests/test_multitask_phase_resume_gate.py:20)。

   正式长跑前应完成一次“连续 10 步”对比“5 步保存＋重载 5 步”的逐状态一致性门禁。

## 可接受风险

- Ease 与稠密 Control 存在信息冗余。覆盖率实现正确：T2M 25%、Control 50%、Edit 0% [hy273_multitask_scheduler.py:540](/mnt/afs/mogeflow-control/data/hy273_multitask_scheduler.py:540)。150K 更新、global batch 128 下，期望有约 480K 个 T2M+Ease 和 6.72M 个 Control+Ease 样本，剂量充足；但 dense root/full-pose control 已暴露大量轨迹信息，非零 Ease gradient 不等于模型学会可控 Ease。必须做固定 noise/text/control 的因果 sweep。

- Control curriculum 沿用全局 200K→400K 设置 [hy273_multitask_base.yaml:62](/mnt/afs/mogeflow-control/configs/hy273_multitask_base.yaml:62)。因此首次 250K Control 更新的 curriculum progress≈25%，`kmax=5`，由 [train_hy273_multitask.py:2294](/mnt/afs/mogeflow-control/train_hy273_multitask.py:2294) 和 [hy273_constraints.py:203](/mnt/afs/mogeflow-control/models/raw_motion/hy273_constraints.py:203)决定。它仍会采到 1–4 个关键帧，smoke 也稳定，故可接受；但应明确这是有意 warm start，而非真正从 `kmax=1` 冷启动。

- 全 train 只读扫描全部有限，但归一化 Ease 为重尾分布：caption-occurrence 口径下 `|z|` p99=4.26、p99.9=6.88、max=12.02。当前 conditioner 直接标准化后送入 MLP、没有裁剪 [hy273_ease.py:191](/mnt/afs/mogeflow-control/models/raw_motion/hy273_ease.py:191)。暂不建议随意裁剪；先记录输入 p99/max、Ease 梯度和 clip 触发率。

- 测试覆盖弱于实现：现有不变性测试的 joint-position 通道全零 [test_hy273_ease.py:16](/mnt/afs/mogeflow-control/tests/test_hy273_ease.py:16)，尚缺非零局部关节、所有 CFG 分支 Ease bit-identical、token-block 仅 target 注入、已训练 conditioner 的 absent identity，以及真实 Stage-BC resume 测试。本次只读关键测试为 5/5 通过，另做的 trained-absent 与 CFG 重复探针均通过。

## 八项核查结论

| 核查项 | 结论 |
|---|---|
| 物理标签/FK/有效长度 | PASS。端点严格 half-mean residual [hy273_ease.py:19](/mnt/afs/mogeflow-control/models/raw_motion/hy273_ease.py:19)，按有效前缀重建全局关节 [hy273_ease.py:90](/mnt/afs/mogeflow-control/models/raw_motion/hy273_ease.py:90)、[hy273_slices.py:111](/mnt/afs/mogeflow-control/models/raw_motion/hy273_slices.py:111)。 |
| paired yaw/crop、Control/Edit | PASS。先选 caption target，再增强、再算 Ease [hy273_multitask_manifest_dataset.py:188](/mnt/afs/mogeflow-control/data/hy273_multitask_manifest_dataset.py:188)。MotionFix 独立 root-center、共享最终 `phi` [hy273_multitask_manifest_dataset.py:228](/mnt/afs/mogeflow-control/data/hy273_multitask_manifest_dataset.py:228)；无运行期 crop，所有实际片段≤300。 |
| 6→H→H additive/identity | PASS。末层精确零初始化 [hy273_ease.py:158](/mnt/afs/mogeflow-control/models/raw_motion/hy273_ease.py:158)，同一 bias 只加 root/body target hidden [kimodo_context_flow_dit.py:627](/mnt/afs/mogeflow-control/models/raw_motion/kimodo_context_flow_dit.py:627)、[kimodo_context_flow_dit.py:655](/mnt/afs/mogeflow-control/models/raw_motion/kimodo_context_flow_dit.py:655)。 |
| CFG 训练/推理一致 | PASS。完整 ConditionBatch 被逐分支复制 [sample_hy273_multitask.py:240](/mnt/afs/mogeflow-control/sample_hy273_multitask.py:240)，模型调用统一使用 repeated condition [sample_hy273_multitask.py:639](/mnt/afs/mogeflow-control/sample_hy273_multitask.py:639)。 |
| 10/70/20 与覆盖率 | PASS。固定任务单位在 [hy273_multitask_scheduler.py:166](/mnt/afs/mogeflow-control/data/hy273_multitask_scheduler.py:166)，Ease 使用独立 keyed draw [hy273_multitask_manifest_dataset.py:341](/mnt/afs/mogeflow-control/data/hy273_multitask_manifest_dataset.py:341)。 |
| Loss/no Ease auxiliary | PASS。总 loss 只有既有 unified flow 和 Edit 项 [train_hy273_multitask.py:4953](/mnt/afs/mogeflow-control/train_hy273_multitask.py:4953)，直接 backward [train_hy273_multitask.py:5169](/mnt/afs/mogeflow-control/train_hy273_multitask.py:5169)，没有 Ease auxiliary。 |
| 250K model/EMA/Adam/DDP | 代码 PASS、实证待补。名称迁移 [train_hy273_multitask.py:1989](/mnt/afs/mogeflow-control/train_hy273_multitask.py:1989)、EMA 初始化 [train_hy273_multitask.py:2056](/mnt/afs/mogeflow-control/train_hy273_multitask.py:2056)、跨 rank Ease-active OR [train_hy273_multitask.py:5238](/mnt/afs/mogeflow-control/train_hy273_multitask.py:5238)均正确。 |
| 真实 smoke | PASS plumbing。8 个 HML update、2 个 Edit update；119/905/256 个 T2M/Control/Edit 样本，即 9.30/70.70/20.00%；`context_update_count=20002`、`ease_update_count=8`，Ease grad=0.0212，最大显存14.27 GiB，全部数值有限。 |

## 正式 8 卡训练前最小检查清单

- [ ] 确认父 checkpoint 为 `next_global_step=250000`、Stage-BE 60/0/40、8×16，且 raw/EMA Edit 与 T2M 门禁已通过；先运行启动器的 `PREFLIGHT_ONLY=1`。

- [ ] 核对 Ease metadata 指向当前 train manifest，并逐值确认 Mean/Std 与已审核资产一致。

- [ ] 修正或绕开通用 resume gate，完成 uninterrupted-10 对 save-5→reload-5：比较 model、EMA、全部 Adam state、batcher/scheduler、下一批 UID/plan、`context_update_count`、`ease_update_count` 和各 rank 同步。

- [ ] 预注册固定 seed/text/length/control 的 `E_in/E_out` scale sweep，报告 normalized MAE、Spearman、root/path/end-effector/control error，以及 T2M/Edit/contact 非退化指标。

- [ ] 前 100–500 update 高频检查：任务比例、Ease present≈37.5%、Ease/total grad、clip、loss、显存、NaN/OOM；同时确认 250K 的 `kmax=5` warm start 是有意设置。

- [ ] fresh run 使用新 `RUN_NAME`；任何 Stage-BC continuation 必须复用原 `RUN_NAME`，因为启动器默认生成时间戳名称 [train_hy273_kencoder_stage_bc_ease_control_ddp8.sh:10](/mnt/afs/mogeflow-control/scripts/launch/train_hy273_kencoder_stage_bc_ease_control_ddp8.sh:10)，trainer 会要求原 run identity [train_hy273_multitask.py:3711](/mnt/afs/mogeflow-control/train_hy273_multitask.py:3711)。

审核全程只读，未修改任何项目文件。