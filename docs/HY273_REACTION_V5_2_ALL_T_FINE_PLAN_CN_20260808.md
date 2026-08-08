# HY273 Reaction-v5.2 全 timestep 关系监督实验计划

## 1. 问题与单变量

Reaction-v5.1 将布局、方位和接触事件等 coarse 项用于全部 sampled timestep，
但 joint-distance、close-vector 和四项 FK contact loss 只在 `t>=0.20`
启用。v5.2 检验：精细关系监督从纯噪声端开始参与，是否能进一步改善
Reaction 的初始相对布局、朝向和精确接触。

相对 v5.1 的唯一有效训练变量是：

```text
fine_min_flow_t: 0.20 -> 0.0
```

`min_flow_t` 同步写成 `0.0`，使配置明确表达所有 Reaction gate 均从
`t>=eps` 开启；当前非零项中只有 `fine_min_flow_t` 的变化会改变 loss。
数据、模型、任务比例、随机种子、低 t 混合、loss 权重、优化器、EMA 和
推理协议全部保持不变。

## 2. Parent 与训练预算

使用与 v5/v5.1 相同的 100K 纯 T2M parent：

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction/
  hy273_unified_fulltext_reaction_v1_20260801_0315/model/step_00100000.pt
```

从 `100K` 训练到 `200K`，8 卡 DDP：

| 任务 | update 比例 | 100K 阶段 updates |
|---|---:|---:|
| T2M | 30% | 30,000 |
| Edit | 35% | 35,000 |
| Reaction | 35% | 35,000 |

保存 `150K` 和 `200K`。200K 完成后先做同预算 matched comparison；只有
结果有继续训练价值，才按相同配方续到 250K/300K。

## 3. v5.2 Reaction loss

每个有效 Reaction sample 的 sampled `t` 都位于 `[eps,1-eps]`。以下全部
Reaction 专项项均启用：

- root radius、source-local bearing、partner-facing；
- scene proximity、pre-contact false-close、first-contact CDF；
- adaptive 22x22 joint distance、GT-close joint vector；
- FK contact-map positive/negative、FK contact vector、FK contact transition。

共享 K273 主 loss 始终不变。低 t 混合仍为 30% 样本均匀采
`t in [eps,0.15]`；其余 70% logit-normal 样本中约 23.2% 也落在 `t<0.2`。
因此全部 Reaction scene 中约 46.2% 新增 fine supervision。再考虑 95%
source-present，fine-active scene fraction 预计由 v5.1 的约 0.511 提高到
v5.2 的约 0.95，而不是只覆盖显式 low-t mixture。

## 4. 预期与风险

预期收益：低 t 下具体 joint/contact 关系更早影响生成轨迹，可能提高
relative bearing、partner-facing、pair-contact 和 contact transition。

主要风险：给定 source+text 时精确接触仍可能多模态；在纯噪声端对单个 GT
接触 realization 施加全强度 fine loss，可能导致条件均值、提前接触、错误
贴近或 jerk 增大。因此训练重点监控：

- source-present 有效 Reaction scene 的 fine active fraction 必须为 1.0；整体
  batch 预期约为 0.95，因为约 5% 的 Reaction 样本按训练协议移除了 source，
  这类 scene 没有合法双人关系项；
- fine/full-contact gradient RMS 及其与 base gradient 的 cosine；
- total loss、preclip gradient、NaN/OOM；
- pre-contact false-close、first-contact timing 和 FK jerk。

## 5. 200K 比较

使用同 UID、同初始噪声、EMA、ODE32、source/text CFG=2.0，对比：

```text
Reaction-v5.1 200K  vs  Reaction-v5.2 200K
```

主要查看 Reaction：frame-0/initial-15f root、relative radius/bearing、
partner-facing、Close@20、pair-contact F1@15cm、contact-vector error、
transition F1、pre-contact false-close、first-contact timing、FK MPJPE 和 jerk。
T2M/Edit 只做相同协议的能力保持守门。

固定 Test 样本此前已参与多轮开发观察，因此 v5.1/v5.2 的 Test 对比只能解释为
探索性结果，不能包装成未见过的确认性结论。训练决策先看 matched validation 与
预先固定的可视化 case；最终结论需要另留未参与调参的 holdout。
本实验只有一个训练 seed；UID paired bootstrap 不包含训练 seed 方差，多指标的
pointwise 区间也未做多重比较校正。T2M/Edit 未预注册非劣界时，只能报告“未见
明显退化”，不能声明正式能力保持。
