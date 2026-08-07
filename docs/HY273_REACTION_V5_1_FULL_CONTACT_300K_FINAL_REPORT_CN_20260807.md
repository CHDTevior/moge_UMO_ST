# HY273 Reaction-v5.1 全序列 FK 接触实验最终报告（300K）

## 1. 最终结论

Reaction-v5.1 相对 Reaction-v5 的唯一训练变量，是新增四项基于 global-rotation FK 的全序列接触损失：contact-map positive、contact-map negative、contact-vector 和 contact-transition。数据、模型、文本/source 条件通路、任务比例、timestep 采样、优化器和推理协议保持不变。

本实验必须分成两个不同的科学问题解释：

1. `Reaction-v5 200K -> Reaction-v5.1 200K` 是同预算的 loss-family 机制对照。
2. `Reaction-v5.1 200K -> Reaction-v5.1 300K` 是同一 run 的额外训练剂量对照，不是 full-contact 机制消融。

最终判断如下：

- **200K 的机制对照未支持主要假设。** Pair-contact F1、contact-vector error 和 transition F1 的 matched 95% CI 均跨 0，不能声称新增 full-contact family 在相同预算下显著改善了接触生命周期。
- **继续同一配方到 300K 后，Reaction 出现了明确而广泛的剂量收益。** Pair-contact F1 从 `0.12138` 提高到 `0.15058`，contact-vector error 从 `31.87 cm` 降到 `29.38 cm`，transition F1 从 `0.00712` 提高到 `0.00948`；三项 paired 95% CI 均不跨 0。FK MPJPE、root error、相对距离和朝向也同步改善。
- **300K 结果不能单独证明 full-contact loss 是收益来源。** 这 100K 同时增加了 T2M/Edit/Reaction 的训练量；若要估计 full-contact family 在 300K 的净机制效应，还需要把无 full-contact 的 Reaction-v5 用同一配方续到 300K。
- **绝对接触精度仍不高。** 300K 的 pair-contact precision/recall/F1 仅为 `0.12263/0.19504/0.15058`，transition F1 仅 `0.00948`。模型已明显更会靠近、面对和形成大致接触，但精确 joint-pair、onset/hold/release 时序仍是主要短板。
- **T2M 未见灾难性遗忘，但也没有正式 non-inferiority 结论。** FID 点估计从 `10.645` 降到 `8.887`，R@1/R@2/R@3 基本稳定；foot-skate ratio 从 `0.2063` 升到 `0.2114`，paired CI 不跨 0，是需要保留的轻微质量退化。
- **Edit 的 CFG=2 描述性指标继续改善，但不能用普通 target error 宣称语义编辑成功。** 正确 source+instruction 的 target joint error 从 `25.09 cm` 降到 `22.34 cm`，changed-region error 从 `25.09 cm` 降到 `22.15 cm`；然而 MotionFix 的 target 与 source 高度相似，source-copy 仍是很强的基线，最终判断仍需可视化和 instruction-specific 指标。

因此，**Reaction-v5.1 建议停在 300K，不再继续单纯堆同配方训练。** 300K 是当前 v5.1 系列中更好的 Reaction checkpoint，可以作为后续 Control 实验的候选起点。本次单 seed、探索性 Test 证据应表述为“200K 机制效应未证实；同一 v5.1 run 继续到 300K 后，多项 endpoint 在该固定 run 上改善”，不能外推成训练分布上的无条件论文级剂量结论。

## 2. 实验身份

代码仓库和分支：

```text
repo:   https://github.com/CHDTevior/moge_UMO_ST
branch: agent/reaction-v5-1-full-contact
```

本地 run：

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_1_full_contact/
  hy273_unified_reaction_v5_1_full_contact_20260806_1750/
```

关键 checkpoint：

```text
model/step_00150000.pt
model/step_00200000.pt
model/step_00250000.pt
model/step_00300000.pt
```

300K checkpoint：

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_1_full_contact/
  hy273_unified_reaction_v5_1_full_contact_20260806_1750/model/step_00300000.pt
```

checkpoint 和数据仅保留在本地，不上传 GitHub。

## 3. 训练合同与稳定性

### 3.1 分阶段任务比例

```text
0K-100K:     T2M/Edit/Reaction = 100/0/0
100K-300K:   T2M/Edit/Reaction = 30/35/35
```

300K checkpoint 的实际 deterministic scheduler 计数：

```text
T2M realized updates       160000
Edit realized updates       70000
Reaction realized updates   70000
next global step           300000
task debts                      0
```

200K 到 300K 的新增曝光严格为：

```text
T2M +30000, Edit +35000, Reaction +35000
```

训练使用 8 卡 DDP。每 rank batch 为 T2M `16`、Edit `8`、Reaction `8`；EMA update count 从 200K 的 `20000` 增至 300K 的 `30000`。200K 与 300K checkpoint 的 stream cursor 可按保存状态精确续接，额外 100K 没有重置 task debt、数据游标或 EMA。

同预算 v5/v5.1 对照的共同 100K T2M parent 已由 200K 报告中两条真实 launch record 与 launcher 默认值共同核实；matched comparator 本身明确标记 `external_launch_record_required`，不再声称能从 child checkpoint 自动证明 parent。另需注意 v5 在 150K 有一次主动停止和续训，而 v5.1 没有相同边界。v5.1 200K -> 300K 比较器只验证相同 run metadata、checkpoint step、task exposure 和三个 stream cursor 的精确连续性；它不从权重内容独立证明 model/EMA/optimizer resume ancestry。真实 continuation 由 `continue300k_watch.log` 中 200K、250K 两次 resume 启动记录和 300K stage-complete 记录支持。

### 3.2 数值稳定性

`metrics.jsonl` 共 `10,000` 条记录，覆盖 100K-300K，每 20 step 一条；没有 JSON 损坏、NaN 或 Inf。除 100K 刚切入新 loss family 的首个记录外，梯度持续稳定：

| 区间 | pre-clip grad 中位数 | 最大值 | Reaction total loss 中位数 | 显存中位数 |
|---|---:|---:|---:|---:|
| 100K-150K | 0.1216 | 15.19（首步瞬态） | 0.02203 | 15.08 GiB |
| 150K-200K | 0.0989 | 0.188 | 0.01504 | 15.07 GiB |
| 200K-250K | 0.0917 | 0.143 | 0.01175 | 15.08 GiB |
| 250K-300K | 0.0865 | 0.156 | 0.00952 | 15.08 GiB |

首个 Stage-B 梯度尖峰在下一条记录即降至 `0.592`，之后继续收敛，没有 NaN、OOM 或 NCCL 错误。典型中位吞吐为 `42K-44K actor frames/s`，中位 step time 为 `0.317-0.339 s`。

训练 loss 下降只用于稳定性判断，不作为生成质量结论；最终决策以固定协议的生成评估为准。

## 4. 评估协议

Reaction 主协议：

- 数据：Inter-X K273 Test。
- 样本：1,579 条，排除已知异常 `G046T007A038R019`。
- fixed-role source actor -> target reactor，不允许角色交换。
- EMA，ODE32，source CFG=`2.0`，text CFG=`2.0`。
- `uid_balanced` caption policy。
- 同 UID、同 caption、同 source/target、同 donor、同初始噪声。
- 10,000 次 UID-cluster paired bootstrap。
- headline 几何从 global rotations 做 FK；position-channel 同时作为一致性旁证。

关键结果：

```text
eval_v5_1_200k_final/reaction/test/reaction_test.json
eval_v5_1_200k_final/reaction/test/matched_v5_vs_v5_1_200000.json
eval_v5_1_300k_final/reaction/test/reaction_test.json
eval_v5_1_300k_final/reaction/test/matched_v5_1_200k_vs_300k.json
eval_v5_1_300k_final/guardrails/matched_t2m_edit_200k_vs_300k.json
```

matched comparator 已逐 UID 核对 1,579 组 `source.npy`、`target.npy`、文本、caption index、actor role、长度和 negative donor，并从保存 prediction 重新计算指标。
T2M/Edit matched guardrail 另逐案核对 4,042 条 T2M 和 1,013 对 MotionFix 的 case/pair identity、seed、source/target 和采样协议，并从保存的 case/shard rows 复算跨 checkpoint 差值与 paired CI。

默认 wrapper 是完整复跑入口：它固定 300K T2M/Edit 生成为 7 shards，若 200K Edit CFG=2 基线不存在则先自动用 7 shards 补齐，然后运行 300K 评估并生成两份 matched JSON：

```bash
RUN_NAME=hy273_unified_reaction_v5_1_full_contact_20260806_1750 \
  EVAL_PHASE=all \
  bash scripts/launch/eval_hy273_unified_reaction_v5_1_300k_final.sh
```

`EVAL_PHASE=finalize` 只用于已完成生成后重算 comparison/render；`benchmarks`、`edit_benchmarks`、`diagnostics` 和 `postprocess` 是定点分段入口，不代替上述默认完整入口。

## 5. 200K 机制对照回顾

Reaction-v5.1 相对 v5 只增加以下 loss family：

| 新增项 | 权重 |
|---|---:|
| FK contact-map positive | 0.001 |
| FK contact-map negative | 0.005 |
| FK contact vector | 0.002 |
| FK contact transition | 0.003 |

这些 fine 项使用 `fine_min_flow_t=0.20`；训练日志中的实际 scene 激活比例均值约为 `51.1%`。transition 项只监督相邻帧 soft contact probability 的变化，hold 段的绝对接触状态由 contact-map 项监督。

200K 主要结果：

| 指标 | v5 200K | v5.1 200K | 差值 | 95% CI |
|---|---:|---:|---:|---:|
| Pair-contact F1@15cm | 0.11818 | 0.12138 | +0.00320 | [-0.00567, 0.01218] |
| Contact-vector error@15cm | 32.56 cm | 31.87 cm | -0.69 cm | [-1.50, 0.09] |
| Transition F1@15cm | 0.00735 | 0.00712 | -0.00023 | [-0.00168, 0.00121] |
| FK MPJPE | 48.45 cm | 47.87 cm | -0.58 cm | [-1.33, 0.17] |
| FK relation MAE | 20.47 cm | 20.40 cm | -0.07 cm | [-0.35, 0.21] |

三项接触 headline 指标均未显著改善。frame-0/initial-15f layout 有探索性改善，但 jerk fidelity error 增加。完整解释见：

```text
docs/HY273_REACTION_V5_1_FULL_CONTACT_200K_REPORT_CN_20260807.md
```

## 6. 300K 同配方剂量结果

### 6.1 全局几何与布局

负差值表示 300K 更好：

| 指标 | v5.1 200K | v5.1 300K | 300K-200K | 95% CI |
|---|---:|---:|---:|---:|
| FK MPJPE | 47.87 cm | 45.11 cm | -2.76 cm | [-3.60, -1.92] |
| Reactor root error | 45.83 cm | 42.88 cm | -2.95 cm | [-3.86, -2.05] |
| FK relation MAE | 20.40 cm | 19.17 cm | -1.23 cm | [-1.48, -0.97] |
| Relative heading | 38.27 deg | 36.72 deg | -1.55 deg | [-2.62, -0.49] |
| Relative bearing | 25.99 deg | 23.44 deg | -2.55 deg | [-3.52, -1.56] |
| Relative radius | 21.03 cm | 19.77 cm | -1.27 cm | [-1.60, -0.92] |
| Frame-0 root | 57.94 cm | 55.23 cm | -2.71 cm | [-4.10, -1.33] |
| Initial-15f root | 54.72 cm | 52.13 cm | -2.59 cm | [-3.92, -1.32] |
| Partner-facing | 29.77 deg | 28.87 deg | -0.90 deg | [-1.69, -0.15] |

frame-0 和 initial-15f heading 的均值也下降，但 CI 跨 0；不把它们列为显著收益。pre-contact root/heading 的 pooled CI 同样跨 0。

frame-0/initial/pre-contact relative-root 与 relative-heading 在两侧共享同一个 observed source，因此应解释为“共享 source gauge 下的 reactor target fidelity”，不是独立的跨人关系项。relative radius/bearing、partner-facing 与跨人 joint-distance/contact 才是本表中额外的非线性关系证据。

### 6.2 接触与生命周期

| 指标 | v5.1 200K | v5.1 300K | 300K-200K | 95% CI |
|---|---:|---:|---:|---:|
| Pair-contact precision@15cm | 0.09411 | 0.12263 | +0.02852 | [0.01880, 0.03844] |
| Pair-contact recall@15cm | 0.17091 | 0.19504 | +0.02413 | [0.01059, 0.03740] |
| Pair-contact F1@15cm | 0.12138 | 0.15058 | +0.02920 | [0.01834, 0.03999] |
| Contact-vector error@15cm | 31.87 cm | 29.38 cm | -2.48 cm | [-3.35, -1.57] |
| Transition F1@15cm | 0.00712 | 0.00948 | +0.00236 | [0.00070, 0.00409] |

300K 同时减少了 pair-level false close，并改善 precision 和 recall，不是只靠增加预测接触数量洗高 F1。scene-level Close@20 F1 从 `0.77623` 到 `0.78345`，其 CI 略跨 0；precision 改善、false-close 降低，但 recall 基本不变。

需要强调绝对量级：pair-contact missed rate 仍为 `0.80496`，transition missed rate 仍为 `0.98667`。模型尚未达到精确接触生成器的水平。

### 6.3 时间平滑性

| 指标 | v5.1 200K | v5.1 300K | 300K-200K | 95% CI |
|---|---:|---:|---:|---:|
| FK jerk fidelity error | 93.61 | 92.18 | -1.44 | [-2.04, -0.82] |
| Prediction FK jerk magnitude | 57.38 | 55.67 | -1.71 | [-2.46, -0.95] |

200K 相对 v5 出现的 jerk 退化，在 300K 剂量下部分回落。该指标不等同于主观平滑度，但与本轮 GIF 中较少的局部急跳方向一致。

### 6.4 条件通路仍然有效

300K 的正确 source+text 相对消融分支优势：

| 消融分支 | FK MPJPE 优势 | Relation MAE 优势 | Close@20 F1 优势 |
|---|---:|---:|---:|
| source-only | 18.13 cm | 10.11 cm | 0.0900 |
| shuffled-text | 34.72 cm | 18.66 cm | 0.2427 |
| unrelated-source | 44.97 cm | 16.93 cm | 0.2473 |
| empty | 53.53 cm | 23.37 cm | 0.4459 |

对应 CI 均不跨 0。300K 的收益不是通过彻底忽略 source 或文本获得；但 source-only gap 相对 200K 有所缩小，应继续把文本条件敏感性作为后续 Control 训练的 guardrail。

## 7. T2M 描述性守门

HumanML3D Test 4,042 条，EMA，ODE32，text CFG=`2.0`：

| 指标 | v5.1 200K | v5.1 300K | 变化 |
|---|---:|---:|---:|
| FID | 10.645 | 8.887 | -1.759 |
| R@1 | 0.5962 | 0.6039 | +0.0077 |
| R@2 | 0.7580 | 0.7580 | 0.0000 |
| R@3 | 0.8266 | 0.8308 | +0.0042 |
| MM distance | 17.325 | 17.209 | -0.116 |
| FK jerk | 53.91 | 54.36 | +0.46 |
| Foot-skate ratio | 0.2063 | 0.2114 | +0.0051 |

4,042 条逐案 identity 和 seed 完全匹配。200K 使用 8 个生成 shard、300K 使用 7 个，但 case plan、逐案 seed/长度与科学采样协议一致，shard 不是统计抽样单位。R@1/R@2/R@3 与 FK jerk 的 paired CI 均跨 0；MM distance 改善的 95% CI 为 `[-0.213, -0.019]`，foot-skate ratio 退化的 95% CI 为 `[0.0023, 0.0079]`。FID 是同协议总体点估计，没有 paired CI。

因此只能说 T2M 语义检索能力没有观察到明显遗忘，同时存在轻微 foot-skate 退化；由于没有预注册 non-inferiority margin，不能写成正式“能力完全保持”。

## 8. Edit 描述性守门

为避免 CFG 不一致，本报告使用新补齐的 200K/300K CFG=2 严格同协议对照：MotionFix Test 总计 1,013 对，EMA，ODE32，source CFG=`2.0`，Edit CFG=`2.0`，同 pair、instruction、source/target、长度和 seed。Target error、jerk 和 foot-skate 使用全部 `N=1013`；changed-region 指标在 61 个 pair 上未定义，因此使用 `N=952`。

| 正确 source+instruction 分支 | N | v5.1 200K | v5.1 300K | 变化 | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Target joint error | 1013 | 25.09 cm | 22.34 cm | -2.75 cm | [-3.69, -1.82] |
| Target rotation error | 1013 | 27.99 deg | 26.69 deg | -1.30 deg | [-1.78, -0.85] |
| Changed-region joint error | 952 | 25.09 cm | 22.15 cm | -2.94 cm | [-3.94, -1.99] |
| Changed-region rotation error | 952 | 28.37 deg | 27.26 deg | -1.12 deg | [-1.61, -0.64] |
| Prediction jerk | 1013 | 46.28 | 43.53 | -2.75 | [-4.24, -1.28] |
| Foot-skate ratio | 1013 | 0.1013 | 0.0920 | -0.0092 | [-0.0145, -0.0041] |

条件反事实结果：

- 300K 在同 source 下，将无 instruction 的 `source_only_model` 分支换回正确 instruction 后，target position error 平均改善 `1.18 cm`，95% CI `[-0.10, 2.46] cm`；rotation error 改善 `1.09 deg`，95% CI `[0.29, 1.93] deg`。这项衡量“是否提供文本”的贡献；方向变正，但位置证据仍弱。
- 300K 把 shuffled source 换回正确 source 后，target position error 平均改善 `14.27 cm`（`N=1012`），表明 source 条件贡献很强。
- 同 source、shuffled instruction 相对正确 instruction 的 position degradation 为 `4.72 cm`，95% CI `[3.43, 6.02] cm`；rotation degradation 为 `5.91 deg`，说明 instruction 内容而不只是文本是否存在会影响输出，instruction 通路并未断开。
- normalized edit-gain 从 200K 的 `-0.093` 提高到 300K 的 `-0.026`，但 300K CI `[-0.072, 0.017]` 仍跨 0；source-copy 仍是不能忽略的强基线。

跨 checkpoint matched 差分进一步显示：text-presence position/rotation advantage 分别增加 `2.22 cm`（95% CI `[1.33, 3.18]`）和 `1.24 deg`（`[0.78, 1.71]`）；instruction-content advantage 分别增加 `1.97 cm`（`[0.75, 3.27]`）和 `0.59 deg`（`[0.03, 1.16]`）。source-content advantage 的跨 checkpoint CI 则跨 0。normalized edit-gain 的变化量为 `+0.067`，95% CI `[0.043, 0.093]`，但 300K 的绝对 edit-gain 仍未显著高于 0。

这些数字支持“Edit 的几何、changed-region、文本存在性与 instruction 内容敏感性在该固定 run 上改善”，不支持“语义编辑问题已经解决”。`move feet faster` 等高阶时序指令仍应使用定向物理指标和 GIF 判断，而不是 aggregate target error。

## 9. 可视化

300K Reaction Test 的 16 类 action-balanced FK GIF：

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_1_full_contact/
  hy273_unified_reaction_v5_1_full_contact_20260806_1750/
  eval_v5_1_300k_final/reaction/test/gifs_action_balanced_fk/
```

同一选择规则的 200K 对照：

```text
eval_v5_1_200k_final/reaction/test/gifs_action_balanced_fk/
```

对 hug、handshake、push、hit 四类样本在相同时间点做 spot check，300K 通常表现为：

- reactor 与 source 的相对距离更接近 GT；
- 面对方向和接近路径更稳定；
- source+text 相比 source-only/shuffled-text 更容易形成正确的交互类别；
- 仍存在手部落点不准、具体接触 joint-pair 错位和接触持续时间不准确。

这与定量结果一致：布局和粗接触明显改善，精确 lifecycle 仍远未饱和。

额外生成的 InterHuman `4313` 是跨数据集 OOD 个例，不属于主 benchmark。原始条目合同为 `p1_to_p2`，因此正确版本明确使用 P1 source -> P2 reactor；此前 `p2_to_p1` 版本是反向诊断，不作为该请求的结果：

```text
eval_v5_1_300k_final/reaction/
  ood_interhuman_4313_correct_p1_source_p2_reactor_retry_seed20260807/
  review_gifs_fk/000_interhuman__4313::p1_to_p2_retry_seed20260807.gif
```

该个例中正确文本比 source-only 更接近目标，但绝对布局误差仍很大，不能外推为 InterHuman 泛化结论。

## 10. 科学解释边界

- 当前只有一个训练 seed；bootstrap 只覆盖 test UID 抽样，不覆盖训练 seed 方差。
- Test 已参与多轮机制开发，属于 exploratory development evidence。
- 因此 300K-vs-200K 的显著 CI 只说明固定 run 上的 test-case endpoint 差异，不估计跨训练 seed 方差，也不消除 checkpoint-selection bias。
- v5.1 同时加入四项 loss，200K 对照只能估计整个 loss family，不能归因到单项。
- 300K 没有无 full-contact 的同预算 baseline，不能把额外训练收益独立归因于 full-contact family。
- pair-contact 与 transition 的绝对 F1 仍低，不能只报告相对提升。
- T2M FID 使用 K273 到现有 evaluator domain 的 bridge，只作同协议相对参考。
- MotionFix aggregate target error 会偏爱 source copy；Edit 必须结合 counterfactual、changed-region、定向物理指标和可视化。
- 同预算 v5/v5.1 child checkpoint 没有内嵌 parent checkpoint 路径；本轮由真实 launch record 和 launcher 默认值支持共同 parent，比较器不独立验证该事实。200K->300K 的 model/EMA/optimizer resume ancestry 同样由训练日志而非 comparator 内部证明。后续应把 immediate parent checkpoint 路径和 step 写入 checkpoint metadata。
- checkpoint 和数据不上传 GitHub，代码、配置、启动脚本和报告上传。

## 11. 后续建议

1. 停止继续堆 v5.1 同配方剂量，保留 300K 作为当前 Reaction 候选基模。
2. 若论文需要 full-contact 机制结论，将 Reaction-v5 无 full-contact baseline 用完全相同的 30/35/35 配方训练到 300K，再做同 UID、同 seed 比较。
3. 若应用目标优先，进入后续正交 Control 训练；在每个 50K checkpoint 同时守住 Reaction pair-contact/layout、T2M retrieval 和 Edit counterfactual。
4. 后续接触改进优先针对 transition/onset/hold/release 的绝对低召回，而不是继续增大全局几何损失。
5. 新训练 checkpoint 增加 `parent_checkpoint_path`、parent step 和 resume lineage 字段，解决当前实验身份只能由 launcher 旁证的问题。
