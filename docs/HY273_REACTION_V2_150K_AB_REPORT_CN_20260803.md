# HY273 Reaction-v2 150K A/B 评估报告

## 1. 结论

**状态：NO-GO，不将当前 Reaction-v2 原样续训到 200K。**

Reaction-v2 的新关系损失确实改变了模型行为，但方向不完整：

- `false-close@20cm` 显著下降，模型不再像 v1 那样频繁错误贴近 source actor。
- `missed-close@20cm` 显著上升，拥抱、握手、踢、推、摔跤、亲吻等真实接触也更容易被避开。
- mean per-frame closest-joint distance 由 `41.18cm` 增至 `48.14cm`，几乎等于 GT 的 `48.29cm`；这说明 v2 的平均预测距离变得更保守，但不能单凭该均值断言每个文本和场景都采用同一先验。
- root radius、bearing、facing、relation-distance 没有获得统计显著的整体改善。
- T2M 小幅、方向一致地退化；Edit 的位置编辑反而改善，但旋转保持和 foot skate 略退化。

因此当前结果不是训练崩溃，也不是 Reaction 条件通路失效。结果与 **fine relation loss 的吸引/排斥不对称造成保守接触偏置** 的假设一致，但 v1/v2 同时改变了 coarse/fine 多个项，当前 A/B 不能把退化唯一归因到其中某一项。没有 checkpoint 趋势或 200K 证据证明继续一定更坏；NO-GO 是因为 150K 已违反预注册 gate，没有理由继续为原配方投入计算。

## 2. 实验协议

比较对象均从同一个 100K T2M parent 开始，在相同任务序列下训练到 150K：

| 项目 | v1 | v2 |
|---|---|---|
| 100K 后任务比例 | T2M/Edit/Reaction = 30/35/35 | 相同 |
| backbone / 文本 / source 融合 | full-text、source token block | 相同 |
| flow | x0 prediction + velocity-MSE | 相同 |
| 优化器、LR、batch、EMA | 相同 | 相同 |
| 训练变量 | 原 v1 relation loss | 新 coarse/fine relation loss |

Reaction 评估使用：

- Inter-X K273 fixed-role Reaction。
- 原始 split 为 train/val/test = `9110/570/1708`；长度超过 300 帧分别排除 `782/48/127` 条。
- train 另有 `2` 条无可用文本；test 另有 `1` 条无文本和 `1` 条已知异常 `G046T007A038R019`。最终可用 train/val/test = `8326/522/1579`。
- EMA、ODE32、`source_cfg=2.0`、`text_cfg=2.0`。
- 同 UID、同初始噪声，禁止 source/reactor 交换身份。
- 主分支为 `source+text`；同时评估 source-only、shuffled-text、unrelated-source 和 empty。

### 2.1 Inter-X 官方源任务核对

本报告完成后重新核对了 Inter-X 官方仓库 `liangxuy/Inter-X`（commit `7902f35`）与论文：

- `interaction_order.pkl` 的官方定义为 `0=P1 actor`、`1=P2 actor`。
- 官方 `preprocess/4_reaction_generation.py` 只按该标签交换两个人，使输出排列为 `[reactor, actor]`；它不包含 Reaction 网络、关系 loss、噪声时间门或采样器。
- 论文中的 Reaction 任务 I/O 是“actor motion 作为条件，生成 reactor motion”。我们的 loader 直接按标签取 actor 为 source、另一人为 target，与该角色语义一致。
- 论文 Table 4 标题同时写有“based on action labels”，但缺少发布代码来确认 action label 是否还是额外条件。因此 `source-only` 只视为最小 I/O 对齐，不宣称逐项复现其完整 conditioning recipe。
- 官方仓库发布的是双人 text-to-motion 训练/评估代码；没有发布 Reaction baseline 的训练实现。论文中的 Reaction 指标为 FID、action accuracy、diversity、multimodality，而非本报告的 fixed-role 物理误差。
- 当前 `source+text -> reactor` 是统一模型的扩展；`source-only -> reactor` 只在任务 I/O 层面与官方定义对齐。由于表示、预处理、训练和指标不同，本报告结果不声明为官方 Inter-X benchmark 数值。
- `tr3e/InterGen` 是相关双人生成方法参考，不是 Inter-X 官方 Reaction 实现。v2 的 relation loss 应表述为受 InterGen 启发的自研实验，不能归因于 Inter-X 官方 recipe。

因此，仓库参考曾发生错位，但数据集、actor/reactor 角色和 Reaction 目标方向没有反。官方脚本先输出 target、再输出 condition；我们的内部 schema 则显式存成 source actor 与 target reactor。

训练到 150K 时状态正常：

- `next_global_step=150000`，EMA 完整。
- 100K--150K 实际任务数为 `T2M 15000 / Edit 17500 / Reaction 17500`；加上 Stage A 后总计 `115000 / 17500 / 17500`。
- Reaction loss 的最近 2K 均值为 `0.00933`，起始 2K 为 `0.02331`。
- 最近 2K 平均 step time `0.296s`，约 `282 scenes/s`；无 NaN、OOM 或 DDP 异常。

训练稳定只说明优化正常，不代表所优化的关系目标正确。

## 3. Reaction 主结果

### 3.1 Test 全量结果

| 指标 | v1 150K | v2 150K | 趋势 |
|---|---:|---:|---|
| Reactor FK MPJPE (cm) ↓ | 76.72 | 81.19 | 退化 |
| Root error (cm) ↓ | 75.88 | 80.33 | 退化 |
| Root-radius error (cm) ↓ | 30.10 | 29.87 | 基本不变 |
| Root-bearing error (deg) ↓ | 51.44 | 52.55 | 基本不变/略退 |
| Partner-facing error (deg) ↓ | 52.36 | 51.88 | 基本不变 |
| Relation-distance MAE (cm) ↓ | 28.38 | 28.71 | 基本不变 |
| Close@20 precision ↑ | 0.655 | 0.684 | 改善 |
| Close@20 recall ↑ | 0.696 | 0.605 | 退化 |
| Close@20 F1 ↑ | 0.675 | 0.642 | 退化 |
| False-close@20 ↓ | 0.260 | 0.197 | 改善 |
| Missed-close@20 ↓ | 0.304 | 0.395 | 明显退化 |
| Mean per-frame closest-joint distance, pred (cm) | 41.18 | 48.14 | 更保守 |
| Mean per-frame closest-joint distance, target (cm) | 48.29 | 48.29 | 参考值 |

Val 上出现同方向结果：F1 `0.673 -> 0.636`，false-close `0.257 -> 0.190`，missed-close `0.305 -> 0.407`。因此不是 test 偶然波动。

### 3.2 Matched test bootstrap

以下为 `v2 - v1`，按同 UID 配对重采样；区间为 95% CI：

| 指标 | 差值 | 95% CI | 判断 |
|---|---:|---:|---|
| Reactor MPJPE (cm) | +4.46 | [3.28, 5.66] | 显著退化 |
| Root error (cm) | +4.45 | [3.21, 5.69] | 显著退化 |
| Root-radius error (cm) | -0.23 | [-0.95, 0.48] | 无显著变化 |
| Root-bearing error (deg) | +1.11 | [-0.04, 2.28] | 无显著变化 |
| Partner-facing error (deg) | -0.48 | [-1.57, 0.62] | 无显著变化 |
| Relation-distance MAE (cm) | +0.33 | [-0.19, 0.85] | 无显著变化 |
| Close@20 F1 | -0.0324 | [-0.0446, -0.0206] | 显著退化 |
| False-close@20 | -0.0624 | [-0.0743, -0.0507] | 显著改善 |
| Missed-close@20 | +0.0910 | [0.0740, 0.1084] | 显著退化 |
| Mean per-frame closest-joint distance (cm) | +6.96 | [6.14, 7.78] | 显著增大 |
| FK jerk error | +1.83 | [1.14, 2.50] | 显著退化 |

F1、false-close 和 missed-close 使用 TP/FP/FN 的 matched cluster micro-bootstrap，而不是简单平均每条样本的 F1；它估计的是帧加权 micro 指标对 clip 抽样的不确定性。

Close precision 的配对 micro-bootstrap 差值为 `+0.0297`，95% CI `[0.0183, 0.0414]`；precision 的提高不足以抵消 recall 的显著下降。

上述 CI 不覆盖训练 seed、生成 noise seed、EMA/raw 或 checkpoint 选择方差。当前结论严格限于一个配对训练 seed 和一个配对生成 noise；更粗的 group/action 聚类重采样不改变方向，但仍不能替代多训练 seed。

### 3.3 条件通路没有断

| Test 指标 | source+text | source-only | shuffled-text | unrelated-source | empty |
|---|---:|---:|---:|---:|---:|
| v2 MPJPE (cm) ↓ | 81.19 | 95.14 | 95.84 | 106.95 | 117.53 |
| v2 relation MAE (cm) ↓ | 28.71 | 39.82 | 42.33 | 42.24 | 47.50 |
| v2 Close@20 F1 ↑ | 0.642 | 0.470 | 0.462 | 0.477 | 0.377 |

正确 source 和正确 text 仍分别提供明显增益。问题不是 source/text 被完全忽略；观测结果表明当前 v2 更偏向“不要错误接触”，但仅凭这次多项联合 A/B 不能确定是哪一个 loss 项造成。

## 4. 可视化核查

同 seed 的 12 类 action-balanced GIF：

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v2/
  hy273_unified_reaction_v2_20260803_082413/eval_reaction_v2_150k_ab/
    v1_150k/reaction/test/gifs_action_balanced
    v2_150k/reaction/test/gifs_action_balanced
```

额外渲染了 missed-close 退化最大的摔跤和亲吻样本：

```text
v1_150k/reaction/test/gifs_targeted_worst
v2_150k/reaction/test/gifs_targeted_worst
```

观察结果：

- `G001T001A005R002`（踢）中，v2 反应者更常停在踢击范围外。
- `G031T006A028R016`（摔跤）中，v1 的 source+text 分支会进入缠抱区域；v2 多数时刻保持明显横向间隔。
- `G034T006A034R023`（亲吻）中，v2 的接近时序和最终贴近不稳定。
- 拥抱、握手、推、坐腿等样本也没有呈现稳定的 radius/bearing 改善。

这与 mean per-frame closest-joint distance `+6.96cm`、missed-close 上升以及 kick/wrestling/kiss 类别退化一致，不是播放器造成的假象。但当前 close-event 由 K273 position channel 计算，FK 路径只报告 relation MAE；在补齐全量 FK close@10/20/30 前，“物理接触退化”仍需保留这一表示路径限定。

## 5. T2M 非退化结果

全量 HumanML3D test `4042` 条，EMA、ODE32、text CFG 2.0。FID/R-precision 是当前 K273 经过同一 MotionStreamer272 bridge 的内部可比协议，不声明为官方 HumanML3D benchmark。

| 指标 | v1 150K | v2 150K | 变化 |
|---|---:|---:|---:|
| FID ↓ | 9.398 | 9.594 | +0.196 |
| R@1 ↑ | 0.586 | 0.571 | -0.016 |
| R@2 ↑ | 0.738 | 0.726 | -0.012 |
| R@3 ↑ | 0.810 | 0.795 | -0.015 |
| FK jerk ↓ | 57.29 | 60.60 | +3.30 |
| Foot-skate ratio ↓ | 0.2129 | 0.2194 | +0.0065 |

这是小幅但方向一致的退化；其中 R@1 的配对 case-bootstrap 差值约为 `-0.0158`，95% CI `[-0.0289, -0.0027]`。不能单独据此判定模型失败，但它不支持继续延长当前 v2。

## 6. Edit 非退化结果

MotionFix test `1013` 对，EMA、ODE32、`source_cfg=2.0`、`edit_cfg=3.0`。全局指标使用全部 `1013` 对；changed-region 指标有 `952` 个有效样本，unchanged-region 指标有 `291` 个有效样本：

| 指标 | v1 150K | v2 150K | 变化 |
|---|---:|---:|---:|
| Target position error (m) ↓ | 0.6721 | 0.6295 | 改善 |
| Changed-region position (m) ↓ | 0.6768 | 0.6350 | 改善 |
| Root target error (m) ↓ | 0.5736 | 0.5223 | 改善 |
| Target rotation error (deg) ↓ | 56.93 | 58.50 | 略退 |
| Unchanged source error (m) ↓ | 0.0968 | 0.1037 | 略退 |
| Contact accuracy ↑ | 0.8068 | 0.8111 | 略升 |
| Foot-skate ratio ↓ | 0.3537 | 0.3752 | 退化 |

Same-source instruction diagnostic：

- ODE correct assignment：`0.500 -> 0.625`。
- `t=0` instruction margin：`0.0135 -> 0.0272`。

因此不能笼统写成“Edit 完全未退化”：位置编辑改善，但 rotation `+1.57deg` 与 foot-skate `+0.0215` 在当前配对协议下均呈显著退化。Edit 出现了指标间 trade-off，该收益不能抵消 Reaction 主目标的退化。

## 7. v2 保守解的可检验解释

v2 的 fine geometry 同时包含：

- union joint-distance map；
- 正负各占一半权重的 soft proximity；
- target-aware false-close penalty。

训练日志显示：

- fine gate 实际覆盖约 `10.0%` Reaction scenes，符合 `t>=0.55` 的设计。
- GT-close pair 只占约 `0.4%` 的全部 22x22 pair entries。由于正负项各自按 mask 分母归一化，这个稀有比例不会直接按 `0.4%` 缩小其 loss 权重；真正风险是零分母 batch、饱和区和有效梯度覆盖。
- false-close mask fraction 从起始 2K 的 `1.10e-4` 降到末尾 2K 的 `2.55e-6`。

以下机制可以解释现象，但仍是待单变量实验验证的假设：

1. GT-far/pred-near 的排斥在靠近阈值时梯度明确。训练项使用 pair-level `GT>=20cm 且 pred<8cm`，评测 false-close@20 使用 frame-level close event；二者相关但不是同一个事件。
2. 当前 sigmoid proximity 对 GT-near/pred-far 样本会在远端饱和，吸引梯度反而变弱。
3. v2 移除了 v1 的 close-joint-vector 重加权。该项虽然不是独立的“关系恒等式”，但它会对 GT-close 关节提供不饱和、带方向的 target reconstruction 梯度。
4. coarse radius/bearing/facing 全时段生效，但对应指标尚未改善，不能补偿 fine contact attraction 的不足。
5. 前 500 updates 的 relation/base output-gradient RMS 比值均值约为 v1 `4.02`、v2 `7.45`，v2 峰值约 `18.08`。v2 并非“关系监督太弱”，而是关系梯度显著增大后仍未改善目标指标，说明方向、时间覆盖与构成比单纯增大总权重更值得怀疑。

模型最终快速消除了错误贴近，却没有同等程度地学会在需要接触时进入准确接触区域。这个终态事实成立；其具体因果归属必须由下一轮单变量实验确定。

## 8. 下一步单变量实验

原 v2.1 草案会同时恢复 positive-close、拆分 proximity 正负权重并降低 false-close，无法判断哪个机制有效，因此不直接启动。所有候选都从同一个 100K parent 重新训练 50K，不从 v2 150K 续训，并保持数据、backbone、任务比例、采样和其他 loss 不变。

建议按以下顺序推进：

1. **P-only**：在 v2 上只恢复 `d_gt<0.20m` 的 `close_joint_vector: 0 -> 0.01`；其余 v2 项逐字不变。该实验检验缺少远端梯度不消失的 GT-target-directed close-region reconstruction reweighting 是否导致 missed-close。它不是新的 source-relative relation identity，也不等同于直接吸向 source。固定筛选协议见 `docs/HY273_REACTION_V2_P_ONLY_SCREEN_PROTOCOL_CN_20260803.md`。
2. **G-only**：在原 v2 上只将 `fine_min_flow_t: 0.55 -> 0.20`。该实验检验 fine 监督覆盖由约 `76.8%` 降至 `10.5%` 是否才是主因。
3. 若 P-only 与 G-only 各有部分互补收益，再做 **P+G** 估计交互；否则不组合。
4. 只有以上结果仍指向排斥过强，才分别做 **N-only**（只降低 soft-negative）或 **F-only**（只降低 false-close），首轮不同时降低两种负项。
5. source-only 分支单独报告，用来覆盖 Inter-X 官方任务 I/O；source+text 继续作为统一模型扩展主分支。两者不混成一个“官方分数”。

预注册判据：

- false-close@20 不回到 v1 水平；
- missed-close@20 至少显著低于 v2，并优先恢复到不差于 v1；
- Close@20 F1 高于 v1；
- radius/bearing/relation MAE 至少一项出现配对显著改善；
- Reaction FK MPJPE/root、T2M R@1/FID、Edit rotation/foot-skate/changed-region 均纳入非退化门。

arm、checkpoint、CFG 和 EMA/raw 的选择只使用 Val。现有 Test 已参与 v2 机制与下一步设计，后续 Test 结果只能标为 exploratory；最终确认需要另留 holdout 或重新定义一次不再调参的 final split。数值非退化 margin 在开训前用 v1 Val 多 noise 重复的波动锁定，不在看到候选结果后决定。

在满足这些判据前，不进入 200K，也不叠加 Control/ease 变量。上述候选仍需完成代码差分审核，并补齐 position/FK close 指标后再启动。

## 9. 评估效率说明

本次 Reaction 评估慢于 T2M 的原因不是没有 batch，而是并行粒度不同：

- T2M：8 个数据 shard，8 张 GPU，`batch_size=16`，每条只跑主生成分支。
- Reaction：单 split 只用 1 张 GPU，`batch_size=8`，每批串行跑 5 个 ODE32 因果分支。

因此 Reaction 每条样本至少有 5 倍采样分支，并且缺少 8 卡数据并行。下一轮 evaluator 应加入 `shard_id/num_shards` 和最终聚合；这只改变调度，不改变已有科学协议。
