# HY273 Reaction-v5 Event-Layout 200K 归档报告

## 1. 结论

Reaction-v5 相比最初的 Reaction-v1 有明确且较大的进步，不只是主观可视化变好：

- 同为 200K，Test reactor FK MPJPE 从 `70.91 cm` 降到 `48.45 cm`。
- relation-distance MAE 从 `26.66 cm` 降到 `20.47 cm`。
- 正确文本相对 source-only 的 FK MPJPE 优势从约 `12.64 cm` 增至 `20.56 cm`。
- v5 自身从 150K 继续到 200K 后，root、bearing、partner-facing、heading、Close@20 F1 和首次接触时序均继续改善。

因此，这一版给出了两个关键判断的强探索性证据：

1. Reaction 的主要问题不是 source/text 条件通路断开。
2. 由低 t 混合采样、source-local 相对几何和接触事件监督组成的 v5 组合方案方向有效。

这里不能把收益独立归因到某一个组件：v5 同时改变了 low-t mixture、layout 权重、scene proximity、pre-contact 和 first-contact CDF，尚未完成同预算逐组件移除实验。

当前仍未达到高精度 Interaction。视觉上的剩余问题主要是接触部位、左右/前后布局 mode、首次接触帧和接触后的精细姿态不够准确。下一步不应只继续堆普通 273D 重建 loss，也不建议无依据地整体放大所有 relation 权重。

## 2. 实验身份

### 2.1 代码与数据

- 代码仓库：`https://github.com/CHDTevior/moge_UMO_ST`
- 归档分支：`agent/reaction-v5-event-layout-200k`
- 数据：Inter-X K273 fixed-role actor-to-reactor
- 表示：Kimodo273，30 FPS，SMPL-X22
- source actor：作为逐帧高带宽条件输入
- target reactor：模型唯一需要生成的动作
- 文本：LLM2Vec sentence token + full context tokens
- source 融合：独立 source token block
- backbone：root/body 双路径 Unified Actor Flow DiT，结构未因 v5 改变

### 2.2 训练

- 0K-100K：T2M 100%
- 100K-200K：T2M/Edit/Reaction = `30/35/35`
- 上述是 update 比例；由于每 rank batch 分别为 `16/8/8`，Stage-B 样本曝光比例约为 `46.15/26.92/26.92`
- flow：`x0 prediction + velocity-MSE`
- 基础 timestep：`sigmoid(N(-0.8, 0.8^2))`
- Reaction-v5 额外 timestep：30% 样本从 `U(eps, 0.15)` 采样
- optimizer、LR、batch、EMA、文本和 source 融合均保持原 Unified Full-Text 配方
- Control/ease：本轮关闭；仓库中的正交 Control 入口不属于 v5 收益来源

配置与启动入口：

```text
configs/hy273_unified_fulltext_reaction_v5_event_layout.yaml
scripts/launch/train_hy273_unified_reaction_v5_event_layout_stage_b_ddp8.sh
```

本地 checkpoint：

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_event_layout/
  hy273_unified_reaction_v5_event_layout_20260805_1345/model/step_00200000.pt
```

## 3. 相比最初版本增加了什么

### 3.1 最初 Reaction-v1

v1 在完整 273D 共享重建 loss 之外增加四项关系 loss：

- relative root
- relative heading
- 22x22 joint distance
- GT-close joint vector

其中 relative root、relative heading 和 close vector 在 prediction/target 两边使用同一个 observed source 时，残差会退化为 target reactor 的单人重建误差或其重加权。它们不能充分告诉模型“reactor 应该位于 source 的哪一侧、朝向谁、何时接触”。真正非线性的跨人关系主要来自 joint-distance，但距离只能约束径向，不能唯一决定方位。

### 3.2 Reaction-v2：真实 source-local 几何

v2 增加：

- reactor root 相对 source heading 坐标系的 radius/bearing。
- reactor heading 相对“朝向 partner 方向”的 partner-facing descriptor。
- coarse/fine timestep gate。
- predicted-near union mask、soft proximity 和 false-close 项。
- frame-0、初始 15 帧、pre-contact、10/20/30cm close event 等评估指标。

v2 证明了负接触约束确实会改变行为，但过强地把模型推向保守分离：false-close 下降，missed-close 上升，因此没有继续使用原配方。

### 3.3 Reaction-v3：恢复正接触吸引并扩大 fine 覆盖

v3 增加或调整：

- `adaptive_gt_inverse` 的完整 22x22 joint-distance weighting。
- `joint_distance=0.0273`。
- 小权重 GT-close directional vector：`0.00191`。
- fine gate 从 `0.55` 降到 `0.20`。

这一步改善了 Close@20/30、MPJPE 和 relation MAE，说明 fine geometry 需要同时覆盖“应该靠近”和“不要错误靠近”，而且不能只在极少数 timestep 上生效。

### 3.4 Reaction-v4：初始与接触前布局加权

v4 对逐帧 source-local root/heading 加入事件阶段权重：

- 初始 15 帧：3x。
- 首次 GT close event 之前：2x。
- 两个阶段重叠时取最大值，不相乘。

该版本针对“最终接触姿态提前铺满整段”和“初始相对位置错误”设计。不过 source 在 prediction/target 两侧相同的 signed-root/heading 直接残差仍存在抵消问题，v5 不再把预算放在这两个冗余项上。

### 3.5 Reaction-v5：从纯噪声选择布局 mode，并显式监督事件

v5 的有效新增项为：

1. **低 t 混合采样**
   - 30% Reaction 样本强制采样 `t in [eps, 0.15]`。
   - 在当前 flow 定义中，低 t 更接近纯噪声端。
   - 目标是在轨迹和姿态尚未由 noisy target 泄露前，让 source+text 决定全局布局 mode。

2. **更强的真实布局监督**
   - initial 15-frame multiplier：`4x`。
   - pre-contact multiplier：`3x`。
   - radius：`0.02`。
   - source-local 有向 bearing 单位向量：`0.05`。
   - partner-facing：`0.04`。

3. **scene-level 接近状态**
   - 不再使用被 22x22 pair 数量稀释的旧 8cm collision 主项。
   - 增加 benchmark-aligned scene proximity，正/负 scene 分开归一化。

4. **接触时序**
   - pre-contact false-close penalty：避免过早进入接触区。
   - first-contact CDF loss：监督首次接触事件的时间分布，而不只监督逐帧距离。

5. **更完整的诊断评估**
   - frame-0、初始 15 帧、pre-contact 的 root/heading error。
   - radius、bearing、partner-facing。
   - 10/20/30cm precision、recall、F1、false-close、missed-close。
   - position/FK 两条几何路径。
   - source-only、shuffled-text、unrelated-source、empty 四种因果消融。

## 4. 200K 结果

评估协议：Inter-X Test 1579 条，EMA，ODE32，source CFG=2.0，text CFG=2.0，fixed role，同 UID/同初始噪声；排除已知异常 `G046T007A038R019`。

聚合口径不是全部相同：Close@10/20/30、false-close、missed-close 和 pre-contact 帧统计先汇总计数再计算 pooled micro ratio；MPJPE、bearing、partner-facing、timing 和 jerk 采用 clip 等权 macro mean。因果消融保持 matched UID，并对 ratio 指标使用 matched micro aggregation。

### 4.1 最初 v1 200K 与当前 v5 200K

下表只比较两个报告中定义一致的重叠指标。它是同 step 的 endpoint comparison，不等价于多训练 seed 的严格因果估计。

| Test 指标 | v1 200K | v5 200K | 变化 |
|---|---:|---:|---:|
| Reactor FK MPJPE | 70.91 cm | 48.45 cm | -22.46 cm |
| Relation-distance MAE | 26.66 cm | 20.47 cm | -6.19 cm |
| Reactor contact F1 | 0.493 | 0.541 | +0.048 |
| FK jerk error | 91.10 | 92.46 | +1.36，略退 |
| 正确文本相对 source-only 的 MPJPE 优势 | 12.64 cm | 20.56 cm | +7.92 cm |

相同 200K 预算下，v5 的空间误差下降约 23%-32%，同时正确文本的因果贡献增大。平滑性没有同步改善，说明剩余问题已经从“整体站错位置”逐步转向“精细接触和局部时间连续性”。

### 4.2 v5 150K 到 200K

| Test 指标 | v5 150K | v5 200K | 变化 |
|---|---:|---:|---:|
| Reactor FK MPJPE | 53.49 cm | 48.45 cm | -5.04 cm |
| Root error | 51.82 cm | 46.45 cm | -5.37 cm |
| Frame-0 relative-root error | 67.12 cm | 60.01 cm | -7.11 cm |
| Initial-15f relative-root error | 63.85 cm | 56.73 cm | -7.12 cm |
| Root-radius error | 22.87 cm | 21.29 cm | -1.58 cm |
| Root-bearing error | 31.04 deg | 26.57 deg | -4.47 deg |
| Partner-facing error | 34.13 deg | 29.92 deg | -4.21 deg |
| Relative-heading error | 44.25 deg | 39.04 deg | -5.21 deg |
| Relation-distance MAE | 21.84 cm | 20.47 cm | -1.37 cm |
| Close@10 F1 | 0.320 | 0.342 | +0.023 |
| Close@20 F1 | 0.747 | 0.773 | +0.026 |
| Missed-close@20 | 0.258 | 0.219 | -0.039 |
| First-close timing error | 0.788 s | 0.729 s | -0.059 s |
| FK jerk error | 91.58 | 92.46 | +0.88，略退 |

这不是“几乎没降”。多数核心布局指标在额外 50K 内下降约 8%-14%，而且方向一致。新增 loss 的绝对数值下降看起来不大，部分原因是：

- 指标已经从 v1 的大误差区进入更难的精细对齐区。
- bearing/heading 是角度，Close F1 是有上限的比例，不能和厘米误差按绝对值比较。
- Reaction 只占总 update 的 35%，150K 到 200K 实际只有 17.5K 个 Reaction updates。
- 多模态接触无法仅靠继续降低逐帧均方误差线性解决。

### 4.3 条件通路

v5 200K 正确 `source+text` 相对消融分支的优势均为正：

| 消融分支 | FK MPJPE 优势 | Bearing 优势 | Partner-facing 优势 | Close@20 F1 优势 |
|---|---:|---:|---:|---:|
| source-only | +20.56 cm | +12.55 deg | +9.45 deg | +0.122 |
| shuffled-text | +33.70 cm | +18.76 deg | +14.83 deg | +0.253 |
| unrelated-source | +43.28 cm | +26.73 deg | +18.77 deg | +0.234 |
| empty | +51.95 cm | +27.73 deg | +21.15 deg | +0.433 |

这些优势的 95% bootstrap CI 均不跨 0。模型确实同时使用 source 和文本；当前瓶颈不是条件完全失效。

## 5. 其他能力守门

### 5.1 T2M

v5 150K 到 200K：

- FID：`10.63 -> 10.43`。
- R@1：`0.588 -> 0.603`。
- R@2：`0.740 -> 0.751`。
- R@3：`0.815 -> 0.824`。
- foot-skate ratio：`0.197 -> 0.205`，略退。

T2M 没有出现能力坍塌，文本检索继续改善，物理平滑性有轻微 trade-off。

### 5.2 Edit

200K 的 Edit 对 CFG 很敏感：

- CFG=3.0 已出现明显过引导和 jerk 放大。
- CFG=2.0 的全量结果更稳定：target position error `0.245 m`、changed-region position error `0.243 m`、rotation error `27.83 deg`、foot-skate ratio `0.0955`。
- `000038 move feet faster` 的视觉响应相对 100K parent 已改善，但强度仍未完全达到 GT。

但 CFG=2.0 仍未通过 Edit target-improvement/non-regression gate：source-copy 的 target position error 为 `0.232 m`，changed-region position error 为 `0.235 m`，rotation error 为 `27.65 deg`，均略优于当前模型；per-case normalized edit gain 为 `-0.0813`，95% CI `[-0.1294, -0.0375]`。正确 instruction 相对 shuffled instruction 显著更好，说明存在 instruction sensitivity；但正确 instruction 相对 source-only 的位置与旋转优势尚不显著。

因此当前默认建议仍是 source CFG=2.0、Edit CFG=2.0，因为它比 CFG=3.0 稳定；准确结论是 **Edit 通路未断、视觉上有响应，但尚未证明优于复制 source，也没有通过能力非退化门**。

## 6. 可视化结论

统一查看目录：

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_event_layout/
  hy273_unified_reaction_v5_event_layout_20260805_1345/
    eval_v5_200k_final/VISUAL_REVIEW_200K/
```

主要观察：

- 粗粒度初始站位、相对侧向、朝向 partner 和接近范围明显优于最初版本。
- 模型没有系统性地把最终接触姿态提前铺满整段。
- source+text 分支明显优于 source-only 和 shuffled-text。
- 握手、拍肩、拥抱等仍经常出现“动作类别对、接触部位或帧不准”。
- 侧后方布局、坐姿接触、长复合文本的精确对齐仍不稳定。
- 少量样本仍有局部高频抖动，但没有观察到普遍单帧爆点。

## 7. 为什么仍然不够准确

### 7.1 离散布局 mode 仍由连续 MSE 隐式选择

给定同一类 source 和文本，reactor 可能位于左、右、前、后等多个合理 mode。当前网络用单个连续 x0/velocity objective 直接回归 273D target。低 t mixture 能迫使条件更早参与，但 MSE 仍倾向于在多 mode 间取条件均值，造成“距离大致对、侧向或朝向不够准”。

### 7.2 关系监督相对完整 273D 目标仍是辅助项

每帧主目标覆盖 273 个通道，relation state 只有少量 root/heading/event 标量。即使权重数值看起来不小，模型容量和梯度的大部分仍用于单人姿态与速度重建。继续整体放大 relation loss 容易换来 jerk、姿态或 T2M/Edit 退化。

### 7.3 22x22 距离图缺少语义对应

joint-distance 可以告诉模型两人是否接近，却不直接指定“右手应在这一帧接触对方左肩”。min-distance 和 pooled F1 也可能被错误关节对满足。因此模型会出现动作语义正确但接触部位错误。

### 7.4 首次接触不等于完整事件结构

当前 first-contact CDF 主要约束首次进入 20cm 区域。它没有完整描述接触关节对、保持时长、释放时刻以及接触后的相对姿态，所以复杂拥抱、拉拽、坐腿仍不够精确。

## 8. 下一步优化优先级

### P0：先做 200K 到 250K 的原配方剂量对照

保持 v5 所有配置不变，只续训 50K，并只用 Val 决定是否保留。它的作用不是期待解决根因，而是测量当前 objective 是否已经平台：

- 若 FK MPJPE 还能下降至少 2 cm、bearing 至少 1 deg、Close@20 F1 至少 0.01，则说明同配方仍有可利用空间。
- 若三项均低于上述量级，停止堆训练步数，进入结构改动。

### P1：增加显式 coarse relation state，而不是继续堆派生 loss

在共享 backbone 内加入紧凑的 source-local relation trajectory/state：

- radius、source-local 有向 bearing 单位向量、relative heading。
- first-contact phase。
- active contact body-part pair。

先由 source+text/global token 预测 coarse relation state，再把该 state 回注 root/body denoising blocks。仍然是一套共享模型和共享 backbone，不使用任务 adapter。目标是把“选哪个布局 mode”从 273D 连续回归中显式拆出来。

### P2：监督时空 contact correspondence

从现有 GT FK 自动构造，不需要新数据：

- `[T,22,22]` soft contact map。
- first-contact joint pair、接触保持区间和 release event。
- 对激活 joint pair 使用带方向 vector loss；非激活 pair 使用较弱 separation loss。
- 对握手、拥抱、拍肩、踢等类别分层报告，而不是只看 pooled min-distance。

### P3：针对难例重平衡数据

- 对接触稀有、侧后方、坐姿和长复合描述做 curriculum/oversampling。
- 保持 source/target 共享 crop、共享 yaw 和相对轨迹语义。
- 不修改 K273 mean/std，也不需要复制原始 motion 文件。

### P4：推理端低成本增强

- 对 source CFG/text CFG 做 Val sweep，当前中心点为 2.0/2.0。
- 对高精度 Interaction 允许多 layout seed 采样，并用独立的 source-text relation scorer 排序。
- 该项只能提升选样命中率，不能替代训练端 coarse relation 建模。

## 9. 科学解释边界

- 当前结果来自一个训练 seed，bootstrap CI 只覆盖测试 clip 抽样，不覆盖训练 seed 方差。
- Test 已参与多轮机制分析，因此 v2-v5 结果应标为 exploratory development evidence。
- v5 是多组件组合实验，缺少同预算 low-t/layout/event loss 的逐项消融，不能把 endpoint 改善独立归因到单个组件。
- 无 GT contact 的 clip 在训练中整段视为 pre-contact；首帧已 contact 的 clip 没有 pre-contact 区间。这是当前自洽协议，不是唯一事件定义。
- event loss 使用 K273 position-joint 路径，而 headline close/MPJPE 使用 FK 路径；二者改善的一致性属于间接证据。
- 下一版结构选择应只看 Val；最终结论需保留新的 motion-disjoint 或未参与调参的 holdout。
- Inter-X 官方仓库没有发布与本实现完全一致的 Reaction 训练 loss。本项目是 fixed-role actor-to-reactor 的统一模型扩展，不声明为官方 Inter-X benchmark 复现。
