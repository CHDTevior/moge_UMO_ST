# HY273 Unified Full-Text Reaction 200K–350K 最终评估报告

日期：2026-08-03

## 1. 结论先行

本轮训练已按计划完成到 `350K`，350K 之后不建议在相同数据、loss 和 `30/35/35` 任务比例下继续训练。

- **Reaction**：350K 最好，但 300K→350K 已明显进入平台期。
- **T2M**：350K 的 FID 最好；300K 的 R@1 最好，250K 的 R@2/R@3 略高。350K 没有灾难性遗忘。
- **Edit**：350K test 位置误差略有改善，但 val 回落，rotation 仍未恢复，jerk 连续恶化；继续训练增强了编辑响应，也放大了过驱和不平滑。
- **统一模型默认 checkpoint**：建议使用 **300K**，它是文本检索、Reaction、Edit 动态响应和平滑性之间更合理的 Pareto 折中。
- **Reaction 或 T2M 分布质量优先**：使用 **350K**。
- **Edit 几何和平滑性基线优先**：保留 **200K**；对速度/时序编辑的统一模型比较优先看 **300K + edit CFG2**。

所有 50K checkpoint 均保留，`latest.pt` 与 350K checkpoint 是同一 inode。

## 2. 实验合同

- Run：`hy273_unified_fulltext_reaction_v1_20260801_0315`
- 0–100K：T2M-only
- 100K–350K：T2M / Edit / Reaction update 比例 = `30 / 35 / 35`
- 250K→300K→350K 期间保持模型、数据、normalizer、loss、LR、采样步数和任务比例不变
- T2M CFG：2.0
- Edit source CFG / edit CFG：2.0 / 3.0（动态面板同时比较 edit CFG2）
- Reaction source CFG / text CFG：2.0 / 2.0
- ODE steps：32
- benchmark 均使用 EMA 权重和固定 seed

350K checkpoint 状态：

- `next_global_step = 350000`
- 实际累计 update：T2M / Edit / Reaction = `175000 / 87500 / 87500`
- scheduler debt：`0 / 0 / 0`
- EMA update count：`35000`

## 3. 训练稳定性

300K→350K 八卡 DDP 用时约 4 小时 8 分钟。

- 无 NaN/Inf、梯度爆炸、掉卡或进程重启。
- grad norm 主要位于 `0.065–0.110`。
- step time 主要位于 `0.28–0.32 s`。
- 吞吐主要位于 `45K–55K actor-frames/s`。
- 末段少量 `0.36–0.42 s` 的 step 为短时 IO/同步波动，没有对应 loss 尖峰。
- 350K checkpoint、scheduler 和任务 update 计数严格一致。

## 4. T2M：HumanML3D test4042

| Step | FID ↓ | R@1 ↑ | R@2 ↑ | R@3 ↑ | FK jerk ↓ | Foot skate ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 200K | 8.1828 | 0.5896 | 0.7412 | 0.8107 | 57.225 | 0.2089 |
| 250K | 8.5002 | 0.5950 | **0.7583** | **0.8253** | 56.899 | **0.2079** |
| 300K | 8.5859 | **0.6000** | 0.7531 | 0.8241 | **56.116** | 0.2122 |
| 350K | **8.0807** | 0.5923 | 0.7496 | 0.8216 | 56.354 | 0.2083 |

观察：

1. 350K 的 FID 从 300K 的 `8.586` 恢复到 `8.081`，是四个 checkpoint 中最好。
2. 文本检索在 250K–300K 达峰；350K 的 R@1 小幅下降，但仍高于 200K。
3. jerk 和 foot skate 均无持续退化，350K 视觉样例没有静止或动作爆炸。

解释：继续多任务训练没有抹掉 T2M，但文本检索和动作分布指标的最优 step 不一致。350K 不能被表述成“全面优于 300K”。

## 5. Reaction：Inter-X fixed-role actor → reactor

### Val522

| Step | Relation MAE ↓ | Close-event ↓ | MPJPE ↓ | Jerk ↓ | Contact F1 ↑ | Text advantage ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 200K | 28.49 cm | 0.250 | 71.73 cm | 96.30 | 0.509 | 13.03 cm |
| 250K | 27.10 cm | 0.234 | 71.81 cm | 100.33 | 0.480 | 14.79 cm |
| 300K | 25.81 cm | 0.216 | 66.77 cm | 97.31 | 0.513 | 15.61 cm |
| 350K | **25.65 cm** | **0.215** | **66.67 cm** | 97.39 | **0.515** | **15.81 cm** |

### Test1579

| Step | Relation MAE ↓ | Close-event ↓ | MPJPE ↓ | Jerk ↓ | Contact F1 ↑ | Text advantage ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 200K | 26.66 cm | 0.258 | 70.91 cm | 91.10 | 0.493 | 12.64 cm |
| 250K | 26.07 cm | 0.249 | 71.04 cm | 94.95 | 0.490 | 13.47 cm |
| 300K | 24.62 cm | 0.233 | 66.16 cm | 90.82 | 0.516 | 14.06 cm |
| 350K | **24.52 cm** | **0.229** | **66.12 cm** | **90.73** | **0.524** | **14.38 cm** |

观察：Reaction 从 200K 到 350K 的整体趋势最清楚。350K 在 val/test 上仍有一致改善，但 300K→350K 的 relation/MPJPE 增益已经很小，主要剩下 contact F1 和文本因果优势的提升。

结论：350K 是当前 Reaction 最优 checkpoint；若继续相同 recipe，边际收益预计很低。

## 6. MotionFix Edit

### Val330

| Step | Target error ↓ | Changed-region ↓ | Rotation ↓ | Jerk ↓ | Skate ↓ | Edit gain ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 200K | **0.5880 m** | **0.6115 m** | **54.64 deg** | **216.5** | **0.3367** | **-2.003** |
| 250K | 0.6457 m | 0.6688 m | 58.47 deg | 259.8 | 0.3723 | -2.383 |
| 300K | 0.6276 m | 0.6506 m | 58.42 deg | 279.7 | 0.3615 | -2.299 |
| 350K | 0.6410 m | 0.6672 m | 58.07 deg | 339.7 | 0.3767 | -2.285 |

350K val 上：

- 正确 instruction 相对 shuffled instruction 优势为 `+0.0448 m`，95% CI `[-0.0364, 0.1253]`。
- 正确 source 相对 shuffled source 优势为 `-0.0181 m`，95% CI `[-0.1044, 0.0672]`。

两者 CI 都跨 0，不能据此声称 350K 在 val 上有稳定因果优势。

### Test1013

| Step | Target error ↓ | Changed-region ↓ | Rotation ↓ | Jerk ↓ | Skate ↓ | Contact acc ↑ | Edit gain ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200K | **0.6403 m** | 0.6428 m | **58.01 deg** | **231.6** | **0.3810** | 0.8211 | **-2.272** |
| 250K | 0.6591 m | 0.6627 m | 60.95 deg | 284.5 | 0.4219 | 0.8235 | -2.483 |
| 300K | 0.6423 m | 0.6432 m | 61.01 deg | 294.9 | 0.4074 | 0.8286 | -2.363 |
| 350K | 0.6406 m | **0.6406 m** | 60.57 deg | 325.1 | 0.4137 | **0.8315** | -2.313 |

350K test 上：

- 正确 instruction 相对 shuffled instruction 优势为 `+0.0538 m`，95% CI `[0.0122, 0.0955]`。
- 正确 source 相对 shuffled source 优势为 `+0.0494 m`，95% CI `[0.0022, 0.0973]`。

因此 test 上 source 和 instruction 通路均有显著因果贡献，不是条件断路。但整体 edit gain 仍为负，说明模型仍未超过 source-copy baseline；同时 jerk 从 200K 的 `231.6` 单调升到 350K 的 `325.1`。

### 速度/时序编辑动态面板

在 CFG2 下，16 条 `faster` 样例的方向正确率：

| Step | Faster direction accuracy ↑ | Faster speed MAE ↓ | Faster target error ↓ |
|---:|---:|---:|---:|
| 250K | 0.566 | 0.2049 m/s | **0.1159 m** |
| 300K | **0.645** | **0.1915 m/s** | 0.1170 m |
| 350K | 0.505 | 0.2024 m/s | 0.1257 m |

`pair 000038: move feet faster and punch once faster`：

- source 脚速：`0.585 m/s`
- target 脚速：`1.557 m/s`
- 300K CFG2：`0.576 m/s`
- 350K CFG2：`0.761 m/s`
- 350K CFG3：`4.293 m/s`

350K CFG2 对该单例的编辑强度更明显，但总体 faster 集合不如 300K；CFG3 已经严重过驱。训练到 350K 增强的是“条件响应幅度”，并没有同步解决速度方向稳定性和平滑性。

## 7. 视觉结论

- T2M：抽查慢速原地走、弯腰拾取和 paraphrase 样例，均有完整动态，无新增静止、全局漂移或姿态爆炸。
- Reaction：抽查接近、抓手、拥抱、坐腿等长短样例，source-text 分支维持双人空间关系；没有新增塌缩。
- Edit：CFG2 仍是合理主协议；CFG3 在多个样例上出现幅度过大、速度过高和姿态夸张。350K 的高 jerk 与视觉上的强响应一致，不是 contact 全饱和复发。

## 8. Checkpoint 选择

### 默认统一模型

推荐：`step_00300000.pt`

理由：

- T2M R@1 最高；
- Reaction 已完成主要跃升，与 350K 差距很小；
- Edit test 位置接近 350K，但 jerk 更低；
- faster 动态面板优于 250K/350K。

### 专项选择

- Reaction 最优：`step_00350000.pt`
- T2M FID 最优：`step_00350000.pt`
- T2M 检索最优：R@1 用 `step_00300000.pt`；R@2/R@3 用 `step_00250000.pt`
- Edit 几何/平滑性基线：`step_00200000.pt`
- Edit 速度/时序的统一模型折中：`step_00300000.pt`，推理使用 source CFG2 / edit CFG2

## 9. 下一步科研判断

不继续把相同 recipe 机械训练到 400K。当前瓶颈已经不是优化不稳定或训练步数不足，而是 Edit 的监督/采样目标让更强条件响应伴随更高 jerk 和 CFG 过驱。

下一轮应控制变量地解决：

1. Edit 推理先固定 CFG2，避免把 CFG3 过驱当成模型能力。
2. 训练目标加入针对 changed-region 的速度/加速度或时间结构监督，同时保留未编辑区域 identity 权重。
3. 用 300K 作为统一模型 parent 做短程 ablation；成功后再决定是否从头重训。
4. Reaction 不需要继续堆训练步数，下一步应检查更多动作类别的分层指标和失败样例。

## 10. 资产位置

- 300K checkpoint：`model/step_00300000.pt`
- 350K checkpoint：`model/step_00350000.pt`
- latest：`model/latest.pt`（与 350K 为同一文件内容/inode）
- 300K 门检：`reports/STEP300K_GATE_REPORT_CN.md`
- 350K 全量评估：`eval_stage_b_350k`
- T2M GIF：`eval_stage_b_350k/t2m/visual16_ema_cfg2/gifs`
- T2M paraphrase GIF：`eval_stage_b_350k/t2m/paraphrase_ema_cfg2/unified_reaction_stage_b_350k_cfg2/gifs`
- Edit 动态 GIF：`eval_stage_b_350k/edit/dynamic/visuals`
- Edit same-source GIF：`eval_stage_b_350k/edit/same_source/gifs`
- Reaction GIF：`eval_stage_b_350k/reaction/test/gifs_action_balanced`

## 11. 解释边界

- 所有 checkpoint 使用同一固定评估 seed，适合做配对趋势比较，但本轮没有多训练 seed，不能把很小的 checkpoint 差异外推成稳定的总体提升。
- T2M benchmark 是当前暂定的内部统一协议；不在本报告中宣称与其他表示或外部论文的绝对 FID/R-Precision 可直接横比。
- Edit 的 aggregate target error 会被 source≈target 稀释，因此必须和 changed-region、counterfactual、动态物理量及 GIF 一起解释。
