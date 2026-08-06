# HY273 Reaction-v2 P-only 150K 探索性筛选协议

## 1. 实验定位

本轮是机制筛选，不是最终确认性实验。它只回答：在 Reaction-v2 上恢复
GT-close 区域的 target-directed reconstruction reweighting，能否减少 v2 的
missed-close，同时保留 false-close 收益。

现有 Inter-X Test 已参与 v1/v2 问题发现，因此本轮只用 Val 做 arm 决策。
若 P-only 通过，最终论文级结论仍需独立 final holdout、至少两个额外训练
seed 和多个生成 noise；本轮 Test 即使补跑，也只能标为 exploratory。

## 2. 控制变量与假设

P-only 和已完成的 v2-150K 都从同一个 100K T2M parent 重启，训练区间均为
`100K -> 150K`，任务顺序、数据、模型、优化器、LR、EMA 和采样协议相同。
两臂唯一可执行配置差异是：

```yaml
reaction_loss:
  close_joint_vector: 0.0 -> 0.01
```

该项不是新的 source-relative relation identity。因为 prediction pair 和 target
pair 中的 source 相同，其残差化简为 reactor prediction 与 GT target 的残差；
`d_gt < 0.20m` 只负责选择并重加权 GT-close joint pairs。同一 reactor joint
若同时靠近多个 source joints，会被重复计权。

准确假设为：`scale=0.05m` 的逐坐标 SmoothL1 在 5cm 以外提供幅值恒定而非
消失的 GT-target-directed 梯度，可能补回 sigmoid proximity 在远端的弱吸引。
base reconstruction 原本已有正监督，因此不能表述为“首次加入正监督”或
“直接把 reactor 吸向 source”。逐坐标 SmoothL1 的线性尾部也不是连续 yaw
旋转不变量；共享随机 yaw 只让其期望覆盖各朝向。

## 3. 固定评估协议

- split：Inter-X Val，固定 role，不允许 actor swap。
- checkpoint：150K EMA。
- sampling：ODE32，`source_cfg=2.0`，`text_cfg=2.0`。
- case/noise：与 v2-150K、v1-150K 使用同 UID、caption 和初始 noise。
- 主比较：`P-only 150K - v2 150K`；v1 只作为性能目标，不作为单变量基线。
- close 统计：先汇总 TP/FP/FN，再做 UID-cluster matched micro-bootstrap。
- 表示路径：K273 position 与 rotation-FK 均报告 10/20/30cm confusion。
- 因果分支：source+text、source-only、shuffled-text、unrelated-source、empty。

## 4. 预注册决策门

按以下固定层级判断；前一层失败即停止，不用后续显著性挑选结果：

1. 主机制门：position `missed-close@20` 相对 v2 至少下降 `0.03`，且配对
   95% CI 上界 `< 0`。
2. 整体接触门：position `Close@20 F1` 相对 v2 至少增加 `0.015`，且配对
   95% CI 下界 `> 0`。
3. 排斥收益保留门：相对 v2 的 `false-close@20` 增量不超过 `0.025`，其
   95% CI 上界不超过 `0.04`，且 P-only 的绝对值 `< 0.235`。
4. v1 恢复目标：P-only 的 position `Close@20 F1 >= 0.675`，并且
   `missed-close@20 <= 0.334`。这是筛选性能门，不解释为独立显著性检验。
5. 双路径一致性：FK `Close@20 F1` 相对 v2 为正、FK missed-close@20
   相对 v2 为负。只有 position 改善时，只能称 position-channel 改善。

`0.03` missed-close 是 v1->v2 已观察退化 `0.091` 的约三分之一；`0.015`
F1 是已观察退化 `0.032` 的约一半。这些是开训前锁定的最小有意义效应，
不是看到 P-only 结果后再调整的阈值。层级检验避免在 10/20/30cm 和多个
relation 指标中择优报告；10/30cm 仅作稳健性诊断。

## 5. Source 因果门与非退化门

source+text 相对 source-only、shuffled-text、unrelated-source 和 empty 的
`reactor_fk_mpjpe_cm`、`fk_relation_distance_mae_cm`、`fk_close_20cm_f1`
advantage 必须保持为正。核心的 unrelated-source 与 shuffled-text 分支要求
95% CI 下界 `> 0`；否则不能把接触改善归因于正确 source/text 条件。

相对 v2-150K Val，允许的最大退化固定为：

| 能力 | 指标 | 最大退化 |
|---|---|---:|
| Reaction | FK MPJPE / root error | `+2.0cm / +2.0cm` |
| T2M | R@1 / FID | `-0.010 / +0.25` |
| Edit | rotation / foot-skate / changed-region position | `+1.0deg / +0.015 / +0.02m` |

这些 margin 用于一次科研筛选，不声称已经由多 noise 方差完成正式
non-inferiority 标定。

## 6. 训练期诊断与后续

训练期检查 close-mask 非零率、fine active scene coverage、close raw/weighted
loss、relation/base output-gradient ratio、global preclip norm、clip frequency、
NaN/OOM、8-rank step 同步和吞吐。`weight=0.01` 不应解释为只占 1% 梯度预算。

P-only 通过全部门后，才考虑进入 200K 或补做 P+G。P-only 未通过时，从同一
100K parent 做 G-only：只将 `fine_min_flow_t: 0.55 -> 0.20`，不从任何 150K
候选续训。
