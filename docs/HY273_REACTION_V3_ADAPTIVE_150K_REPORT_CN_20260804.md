# HY273 Reaction-v3 Adaptive Geometry 150K 评估报告

## 1. 结论

**Reaction 主结论：v3 是当前三组 150K 模型里更好的中等距离交互模型，但还不是精确接触模型。**

- 相对 v2，v3 的 FK Close@20/30 F1、接近召回、Reactor MPJPE、root error 和跨人关系距离均取得配对显著改善。
- 相对更强的 P-only，v3 的 Close@20/30 recall 和 Close@30 F1 仍改善；MPJPE、root、relation MAE 只有较小点估计收益，95% CI 跨 0。
- FK Close@10 F1 没有改善：相对 v2 为 `-0.0071`，相对 P-only 为 `-0.0216`，两者 CI 都跨 0。当前改动主要让 reactor 更会进入对方附近，而没有稳定解决握手、拍肩、拥抱等精确接触点。
- 正确 `source+text` 显著优于 source-only、shuffled-text、unrelated-source 和 empty，说明 Reaction 确实同时使用了观察动作和文本，不是靠单一姿态/距离先验完成指标。
- 可视化与数字一致：正确条件分支通常能选择更合理的接近范围，但相对侧向、手/肩/身体接触部位及接触时序仍不稳定，样本间方差较大。部分 Close@20 改善只是进入距离阈值，不等价于语义正确的 Reaction。

因此建议把 v3 归档为当前 **coarse/mid-range Reaction best checkpoint**。它可以作为下一轮精接触实验的 parent，但不能宣称 Reaction 已解决，也不建议仅凭当前结果直接把同一配方长训很多步来替代精接触设计。

## 2. 实验设计

三组模型均从同一个 100K T2M parent 出发，使用相同数据顺序、任务比例、backbone、文本/source 条件通路和优化配置训练到 150K：

| 100K--150K 配置 | v2 | P-only | v3 adaptive |
|---|---:|---:|---:|
| T2M / Edit / Reaction | 30% / 35% / 35% | 相同 | 相同 |
| `joint_distance` | `0.01` | `0.01` | `0.0273` |
| distance 形式 | 原距离图 | 相同 | `1/(d_gt+0.1)` adaptive weighting |
| SmoothL1 beta | 原配置 | 相同 | `0.05m` |
| GT-close directional vector | `0` | `0.01` | `0.00191` |
| fine geometry gate | `t>=0.55` | `t>=0.55` | `t>=0.20` |

v3 的 close vector 在 observed-source heading 的局部坐标系中计算；保留 v2 中不会因共享 source 而代数抵消的 radius、bearing、partner-facing、soft-proximity 和 false-close 项。没有恢复 legacy no-op relative-root/heading 项。

训练正常完成：

- checkpoint：`step_00150000.pt`，`next_global_step=150000`，EMA update 完整；
- 100K--150K 实际任务数：T2M `15000`、Edit `17500`、Reaction `17500`；
- 八卡 DDP 无 NaN、OOM、通信异常或调度债务。

## 3. Reaction 固定评估协议

- 数据：Inter-X K273 Val，过滤后 `522/522` 条；fixed actor->reactor role，禁止 swap。
- 模型：EMA 150K。
- 采样：ODE32，`source_cfg=2.0`，`text_cfg=2.0`，seed `20260801`。
- 每个 UID 的 v2、P-only、v3 使用相同初始噪声。
- 主分支：正确 `source+text`。
- 因果分支：source-only、shuffled-text、unrelated-source、empty；每个分支均为 `522` 条。
- 统计：按 UID 配对的 10,000 次 cluster bootstrap；Close 指标由 TP/FP/FN 做 pooled micro 统计。

完整产物：

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v3_adaptive/
  hy273_unified_reaction_v3_adaptive_20260804_0628_smoke/
    eval_v3_150k_screen/
      v3_150k/reaction/val/reaction_val.json
      matched/v3_vs_v2.json
      matched/v3_vs_p_only.json
```

## 4. Reaction 主结果

### 4.1 三模型绝对值

| 指标 | v2 150K | P-only 150K | v3 150K | 最优方向 |
|---|---:|---:|---:|---|
| FK Close@10 F1 | 0.2932 | **0.3077** | 0.2861 | 高 |
| FK Close@20 F1 | 0.6360 | 0.6435 | **0.6630** | 高 |
| FK Close@30 F1 | 0.7232 | 0.7332 | **0.7483** | 高 |
| FK missed-close@20 | 0.4074 | 0.4002 | **0.3694** | 低 |
| FK false-close@20 | 0.1893 | **0.1847** | 0.1898 | 低 |
| FK relation-distance MAE (cm) | 29.95 | 29.65 | **29.06** | 低 |
| Reactor FK MPJPE (cm) | 81.28 | 79.17 | **78.20** | 低 |
| Reactor root error (cm) | 81.07 | 78.81 | **77.68** | 低 |
| Partner-facing error (deg) | 50.79 | 50.64 | **49.37** | 低 |
| Root-bearing error (deg) | 51.18 | **49.70** | 50.44 | 低 |
| Root-radius error (cm) | 31.37 | 31.38 | **30.68** | 低 |

v3 没有通过制造更多错误贴近来换 recall：相对 v2，false-close@20 基本不变；相对 P-only 的小幅上升也不显著。

### 4.2 v3 相对 v2 的配对差值

| 指标 | v3 - v2 | 95% CI | 判断 |
|---|---:|---:|---|
| FK Close@10 F1 | -0.0071 | [-0.0371, 0.0230] | 无显著变化 |
| FK Close@20 F1 | +0.0270 | [0.0040, 0.0499] | 显著改善 |
| FK Close@30 F1 | +0.0251 | [0.0099, 0.0408] | 显著改善 |
| FK Close@20 recall | +0.0380 | [0.0061, 0.0693] | 显著改善 |
| FK missed-close@20 | -0.0380 | [-0.0693, -0.0061] | 显著改善 |
| FK false-close@20 | +0.0005 | [-0.0161, 0.0176] | 基本不变 |
| FK relation MAE (cm) | -0.885 | [-1.732, -0.021] | 小幅显著改善 |
| Reactor FK MPJPE (cm) | -3.086 | [-4.882, -1.329] | 显著改善 |
| Reactor root error (cm) | -3.397 | [-5.263, -1.478] | 显著改善 |
| Partner-facing (deg) | -1.418 | [-3.266, 0.373] | 点估计改善，未显著 |

### 4.3 v3 相对 P-only 的配对差值

| 指标 | v3 - P-only | 95% CI | 判断 |
|---|---:|---:|---|
| FK Close@10 F1 | -0.0216 | [-0.0519, 0.0101] | 无显著变化 |
| FK Close@20 F1 | +0.0195 | [-0.0004, 0.0399] | 边界改善，CI 略跨 0 |
| FK Close@30 F1 | +0.0151 | [0.0009, 0.0296] | 显著改善 |
| FK Close@20 recall | +0.0308 | [0.0033, 0.0585] | 显著改善 |
| FK missed-close@20 | -0.0308 | [-0.0585, -0.0033] | 显著改善 |
| FK false-close@20 | +0.0051 | [-0.0123, 0.0228] | 无显著变化 |
| FK relation MAE (cm) | -0.592 | [-1.491, 0.288] | 未显著 |
| Reactor FK MPJPE (cm) | -0.970 | [-2.915, 0.910] | 未显著 |
| Reactor root error (cm) | -1.129 | [-3.172, 0.875] | 未显著 |

这说明 v3 的主要确定性增益是恢复更多应该发生的 20/30cm 接近事件。P-only 已经吸收了部分 directional close supervision 的收益，所以 v3 对它的整体姿态误差优势较弱。

## 5. 条件是否真的被使用

以下为 v3 正确 `source+text` 相对错误/缺失条件的优势；正数表示正确条件更好：

| 对照分支 | FK Close@20 F1 优势 | Relation MAE 优势 (cm) | Reactor MPJPE 优势 (cm) |
|---|---:|---:|---:|
| source-only | +0.1923 | +13.81 | +18.46 |
| shuffled-text | +0.1788 | +14.85 | +16.96 |
| unrelated-source | +0.1837 | +12.55 | +21.31 |
| empty | +0.3171 | +21.48 | +42.11 |

以上 12 个 95% CI 均不跨 0。结论是：

1. source motion 是必要条件；换成无关 source 会明显破坏相对几何。
2. 文本在 source 已给定时仍提供显著增益；正确文本优于 source-only 和 shuffled-text。
3. 当前问题不是条件通路断开，而是精确空间/时间接触仍难。

## 6. Reaction 可视化

### 6.1 同 UID action-balanced 12 条

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v3_adaptive/
  hy273_unified_reaction_v3_adaptive_20260804_0628_smoke/
    eval_v3_150k_screen/reaction_visuals/
      v3_action_balanced_12/
      v2_same_action_balanced_12/
      p_only_same_action_balanced_12/
```

三个目录文件名和 UID 一一对应。每条 GIF 有四个面板：

- 绿色：GT reactor；
- 红色：正确 source+text；
- 橙色：source-only；
- 紫色：shuffled-text；
- 蓝色：所有面板共享的 observed source actor。

这 12 条覆盖握手、挥手、拉拽、击打、踢、推、坐腿、拍肩等交互。肉眼可见的稳定趋势是：正确 source+text 分支通常比 source-only/shuffled-text 更接近 GT 的活动范围，但精确相对侧向、手部、肩部和身体接触经常没有在正确帧落到正确位置。

### 6.2 提升/退化诊断 6 条

```text
reaction_visuals/
  v3_diagnostic_best_worst_6/
  v2_diagnostic_best_worst_6/
```

前 3 条按 Close@20/MPJPE 改善挑选，后 3 条按退化挑选；它们只用于解释模型方差，不用于估计总体性能。重点 UID：

- 改善：`G041T007A020R002`、`G043T002A011R010`、`G023T006A031R006`；
- 退化：`G054T000A003R023`、`G021T001A006R008`、`G002T009A039R009`。

其中：

- `G041T007A020R002` 的 FK Close@20 F1 从 `0` 升到 `1`，但 v3 reactor 只是更靠近坐姿 source，站位方向和 GT 仍有明显差异；该例说明 close-event 不能单独代表 Reaction 语义正确。
- `G021T001A006R008` 中，GT reactor 位于 source 的一侧，v2 生成在同侧，而 v3 生成到镜像侧；其 MPJPE 从 `28.7cm` 退化到 `110.8cm`，Close@20 F1 从 `0.875` 降到 `0.116`。这与总体 bearing 指标未显著改善相符。
- 握手和踢的 action-balanced 样本中，v3 正确条件分支明显区别于 source-only/shuffled-text，但接触点和反应时序仍只达到粗对齐。

总体指标改善并不意味着每条样本单调改善。当前模型在长组合描述、左右/前后相对方位、罕见姿态和接触时序上仍有较大的 sample 方差。

## 7. 其他任务仅作守门参考

用户当前关注 Reaction，因此以下不作为选模型主依据：

- T2M：v3 R@1 `0.5933`，高于 v2 `0.5705`；FID `9.959`，差于 v2 `9.594`。属于混合结果，没有观察到整体能力坍塌。
- Edit：v3 相对 v2 的 changed-position `0.6217 -> 0.5935m`、rotation `54.61 -> 51.31deg`、foot-skate `0.3220 -> 0.3111`，均未退化。

## 8. 下一步建议

1. 保留 v3 的 adaptive distance 和低 gate，它们已经改善 20/30cm 接近、MPJPE 与 relation MAE。
2. 下一轮只针对 10cm 精接触做单变量实验，不再继续堆粗距离项。首选轴是提高/重构带方向的 close-vector 或 contact-phase supervision，同时保留非接触区 identity anchor。
3. 对握手、拍肩、拥抱等动作单独报告接触关节对与时间窗口；全身 22x22 pooled distance 会掩盖“距离对、部位或帧不对”。
4. 在决定是否从 v3 继续到 200K 前先看本报告的同 UID GIF。若只是动作更接近但接触仍错，延长同一 objective 不足以回答根因；若接触时序已呈持续改善，再用 150K->200K 作为 training-dose 实验。

当前证据来自一个训练 seed 和一个 matched sampling seed。配对 CI 只覆盖 Val UID 抽样不确定性，不覆盖训练 seed 方差。
