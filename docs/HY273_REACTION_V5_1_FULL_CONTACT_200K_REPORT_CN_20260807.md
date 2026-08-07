# HY273 Reaction-v5.1 全序列 FK 接触 200K 归档报告

## 1. 结论

Reaction-v5.1 是相对 Reaction-v5 的单变量 loss-family 实验：保持数据、backbone、条件通路、任务比例、timestep、优化器和推理协议不变，只增加基于 global-rotation FK 的全序列 joint-pair contact-map、contact-vector 和 transition 监督。

截至 200K，结果不能表述为“全序列接触监督已经训练成功”：

- 预注册主要指标没有显著改善。FK pair-contact F1 从 `0.11818` 到 `0.12138`，差值 `+0.00320`，95% CI `[-0.00567, 0.01218]`；contact-vector error 从 `32.56 cm` 到 `31.87 cm`，差值 `-0.69 cm`，95% CI `[-1.50, 0.09]`；transition F1 从 `0.00735` 到 `0.00712`，差值 `-0.00023`，95% CI `[-0.00168, 0.00121]`。
- 初始布局出现了探索性的正向 signal。frame-0 relative-root error 下降 `2.07 cm`，initial-15f relative-root error 下降 `2.01 cm`，initial-15f relative-heading error 下降 `1.54 deg`，三项未经多重比较校正的 pointwise 95% CI 均不跨 0。
- Reaction 的 GT jerk fidelity 显著变差：`||jerk(pred)-jerk(GT)||` 从 `92.46` 上升到 `93.61 m/s^3`，差值 `+1.16`，95% CI `[0.62, 1.71]`。预测 FK jerk magnitude 也从 `55.80` 上升到 `57.38 m/s^3`，差值 `+1.58`，95% CI `[0.87, 2.27]`；后者表示高阶时间变化增大，但不能单独等同于主观抖动。
- 全局 FK MPJPE 和 relation-distance MAE 仅小幅改善，95% CI 均跨 0，不能据此声称整体几何显著提升。
- T2M 描述性 guardrail 未见灾难性退化，但没有预注册 non-inferiority margin，不能作正式“保持”结论；Edit 的平均几何误差、prediction jerk magnitude 和 foot-skate 相对 v5 改善，但聚合 target error 仍明显差于 source-copy/source-only，因此 Edit 仍不能用普通 target MSE 宣称成功。

当前科学判断是：**v5.1 在 200K 未支持“全序列 FK 接触项能显著提高精确 joint-pair contact lifecycle”的主要假设；它带来了 post-hoc 的初始布局 signal，并增加了 jerk fidelity error 与预测 jerk magnitude。** 因为 100K-200K 只有 `35K` 个 Reaction updates，已按控制变量原则继续同一 run 到 300K；300K 前不改变权重、不加入 Control。

## 2. 实验身份

### 2.1 代码与分支

- 仓库：`https://github.com/CHDTevior/moge_UMO_ST`
- 分支：`agent/reaction-v5-1-full-contact`
- v5.1 训练实现：`9a2f8f0`
- v5.1 matched evaluator：`20d0b98`
- 配置：`configs/hy273_unified_fulltext_reaction_v5_1_full_contact.yaml`
- 启动脚本：`scripts/launch/train_hy273_unified_reaction_v5_1_full_contact_stage_b_ddp8.sh`
- 200K 最终评估：`scripts/launch/eval_hy273_unified_reaction_v5_1_200k_final.sh`

### 2.2 Run 与 checkpoint

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_1_full_contact/
  hy273_unified_reaction_v5_1_full_contact_20260806_1750/
```

关键 checkpoint：

```text
model/step_00150000.pt
model/step_00200000.pt
```

训练合同：

- 0K-100K：同一个 Unified Full-Text 纯 T2M parent。
- 100K-200K：T2M/Edit/Reaction update 比例为 `30/35/35`。
- 每 rank batch：T2M `16`，Edit `8`，Reaction `8`。
- 模型：root/body 双路径 Unified Actor Flow DiT。
- 文本：LLM2Vec sentence token + full context tokens。
- source：独立逐帧 source token block。
- flow：`x0 prediction + velocity-MSE`。
- Reaction timestep：基础 `sigmoid(N(-0.8, 0.8^2))`，其中 30% 改采 `U(eps, 0.15)`。
- EMA、LR、optimizer、normalizer、数据、随机种子与 v5 相同。
- Control/ease：关闭。

共同 parent 由外部真实启动记录核实，而不是由 child checkpoint 自证：

- Codex rollout `rollout-2026-07-23T05-10-28-019f8baa-6662-7903-8320-087d4ae139d7.jsonl:165646` 记录了 v5 在 `2026-08-05T05:42:17.774Z` 的启动命令；命令只覆盖 `RUN_NAME/STOP_STEP/GPU_IDS/MASTER_PORT`，没有覆盖 `CHECKPOINT/PARENT_CHECKPOINT`。
- 同一 rollout 的 `:170095` 记录了 v5.1 在 `2026-08-06T09:50:43.895Z` 的启动命令；命令只覆盖 `RUN_NAME/MASTER_PORT`，同样没有覆盖 parent。
- 两个 launcher 的默认值都解析到 `/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction/hy273_unified_fulltext_reaction_v1_20260801_0315/model/step_00100000.pt`。

当前 child checkpoint 没有保存 resume path，因此 matched comparator 只能验证 checkpoint 配置、RNG 和最终 batcher state；它不能独立验证共同 parent。机器结果相应标记为 `parent_lineage_status=external_launch_record_required`。

checkpoint 中的 deterministic task scheduler state 在 200K 为：

```text
T2M realized updates      130000
Edit realized updates      35000
Reaction realized updates  35000
next global step          200000
task debts                     0
```

因此 v5 与 v5.1 的最终任务曝光和数据游标一致。需要保留一个实际差异：v5 在 150K 为评估主动停止，并从 `step_00150000.pt` 续到 200K；v5.1 没有这次相同的运行边界。该 resume 没有改变最终 scheduler debt、realized task counts 或 stream cursor，但不能写成“两边没有 resume 边界差异”。

## 3. v5.1 改了什么

v5.1 保留 v5 的 low-t layout、source-local radius/bearing、partner-facing、scene proximity、pre-contact 和 first-contact CDF，只增加以下 FK contact family：

| 新增项 | 权重 | 作用 |
|---|---:|---|
| FK contact-map positive | 0.001 | 在 GT `<15 cm` joint-pair 上提高接触召回 |
| FK contact-map negative | 0.005 | 抑制 GT 非接触 joint-pair 的错误接触 |
| FK contact vector | 0.002 | 在 GT 接触 pair 上精修 FK 相对向量 |
| FK contact transition | 0.003 | 监督相邻帧接触概率变化，直接对应 onset/release；hold 的绝对接触水平由 map 项锚定 |

这些项使用 `[B,T,22,22]` joint-pair contact map 和 `[B,T,22,22,3]` pair vector，仅参与训练 loss，不进入模型输入，不增加推理计算或模型参数。

四项都服从 `fine_min_flow_t=0.20`。训练日志中 `fine_active_scene_fraction` 在 100K-300K 的 10,000 条记录上均值为 `0.51075`，即约 `51.1%` 的 Reaction scene/update 激活这些 fine 项，并非全量 scene 都激活。transition 项比较的是相邻帧 soft contact probability 的差分；它不单独约束 hold 段的概率绝对值。

需要保留两个解释边界：

1. 这是一个 loss-family 实验，map positive/negative、vector、transition 同时加入，不能把任何 endpoint 变化独立归因到其中一项。
2. FK contact vector 在 prediction/target 两侧共享同一个 observed source，source 会在向量残差中抵消；它本质上是 GT-contact-mask 选择的 reactor FK 精修。真正非线性的跨人关系新增监督主要来自 contact-map。

## 4. 评估协议

Reaction 固定协议：

- 数据：Inter-X K273 fixed-role actor-to-reactor。
- Test：1,579 条，排除已知异常 `G046T007A038R019`。
- EMA，ODE32。
- source CFG=`2.0`，text CFG=`2.0`。
- fixed role，不允许 actor/reactor swap。
- v5 与 v5.1 使用同 UID、同初始噪声。
- 10,000 次 UID-cluster bootstrap。

200K 结果目录：

```text
eval_v5_1_200k_final/reaction/test/reaction_test.json
eval_v5_1_200k_final/reaction/test/matched_v5_vs_v5_1_200000.json
eval_v5_1_200k_final/t2m/full_test4042_ema_cfg2/summary.json
eval_v5_1_200k_final/edit/full_test1013_ema_sourcecfg2_editcfg3/summary.json
eval_v5_1_200k_final/guardrails/matched_t2m_edit_v5_vs_v5_1_200k.json
```

## 5. Reaction 结果

### 5.1 150K Val 早期门

150K 在 522 条 Val 上没有出现主要 contact 收益，而且三个朝向指标显著变差：

| 指标 | v5 150K | v5.1 150K | v5.1-v5 | 95% CI |
|---|---:|---:|---:|---:|
| FK MPJPE | 53.30 cm | 53.78 cm | +0.48 cm | [-0.67, 1.61] |
| FK relation MAE | 22.50 cm | 22.58 cm | +0.08 cm | [-0.45, 0.61] |
| Overall relative heading | 41.26 deg | 42.96 deg | +1.70 deg | [0.45, 2.97] |
| Initial-15f heading | 42.90 deg | 45.15 deg | +2.25 deg | [0.51, 4.00] |
| Partner-facing | 31.22 deg | 32.92 deg | +1.69 deg | [0.58, 2.91] |
| FK pair-contact F1 | 0.09408 | 0.09111 | -0.00297 | [-0.01843, 0.01275] |
| Contact-vector error | 36.81 cm | 36.71 cm | -0.10 cm | [-1.53, 1.28] |
| FK transition F1 | 0.00582 | 0.00453 | -0.00129 | [-0.00324, 0.00062] |

因此训练继续到 200K 的依据是预注册训练预算，而不是 150K 已经显示成功。

### 5.2 200K Test matched comparison

下表的负误差差值表示 v5.1 更好，正 jerk 差值表示更差；“显著”仅表示未经多重比较校正的 pointwise 95% CI 不跨 0：

| 指标 | v5 200K | v5.1 200K | v5.1-v5 | 95% CI | 判断 |
|---|---:|---:|---:|---:|---|
| Frame-0 relative-root | 60.01 cm | 57.94 cm | -2.07 cm | [-3.36, -0.81] | 显著改善 |
| Initial-15f relative-root | 56.73 cm | 54.72 cm | -2.01 cm | [-3.26, -0.80] | 显著改善 |
| Initial-15f relative-heading | 41.12 deg | 39.58 deg | -1.54 deg | [-2.75, -0.36] | 显著改善 |
| Pre-contact relative-root | 56.30 cm | 54.69 cm | -1.61 cm | [-2.97, -0.30] | 显著改善 |
| Pre-contact relative-heading | 36.26 deg | 34.64 deg | -1.61 deg | [-2.88, -0.30] | 显著改善 |
| FK MPJPE | 48.45 cm | 47.87 cm | -0.58 cm | [-1.33, 0.17] | 未显著 |
| FK relation MAE | 20.47 cm | 20.40 cm | -0.07 cm | [-0.35, 0.21] | 未显著 |
| Partner-facing | 29.92 deg | 29.77 deg | -0.15 deg | [-0.86, 0.55] | 未显著 |
| FK jerk fidelity error | 92.46 | 93.61 | +1.16 | [0.62, 1.71] | 显著变差 |
| Prediction FK jerk magnitude | 55.80 | 57.38 | +1.58 | [0.87, 2.27] | 显著增加 |

这里的 frame-0/initial-15f/pre-contact relative-root 与 relative-heading 都在 prediction 和 GT 两侧使用同一个 observed source。其残差在代数上等价于共享 source gauge 下的 reactor target fidelity，不能把它们单独解释成独立的跨人关系建模收益；radius、bearing、partner-facing 和跨人 joint-distance/contact 指标才提供额外的非线性关系证据。

### 5.3 主要 contact lifecycle 指标

| 指标 | v5 200K | v5.1 200K | v5.1-v5 | 95% CI | 判断 |
|---|---:|---:|---:|---:|---|
| FK pair-contact precision@15 | 0.09247 | 0.09411 | +0.00164 | [-0.00659, 0.00993] | 未显著 |
| FK pair-contact recall@15 | 0.16368 | 0.17091 | +0.00722 | [-0.00373, 0.01832] | 未显著 |
| FK pair-contact F1@15 | 0.11818 | 0.12138 | +0.00320 | [-0.00567, 0.01218] | 未显著 |
| FK contact-vector error@15 | 32.56 cm | 31.87 cm | -0.69 cm | [-1.50, 0.09] | 未显著 |
| FK transition F1@15 | 0.00735 | 0.00712 | -0.00023 | [-0.00168, 0.00121] | 未显著 |

scene-level Close@20 F1 从 `0.77289` 到 `0.77623`，差值 `+0.00334`，95% CI `[-0.00387, 0.01042]`，同样未显著。结果没有证据表明 v5.1 只是通过 scene min-distance 洗高主要指标。

### 5.4 条件因果消融

v5.1 200K 的正确 source+text 分支明显优于所有消融分支：

| 消融分支 | 正确条件的 FK MPJPE 优势 | Relation MAE 优势 | Close@20 F1 优势 |
|---|---:|---:|---:|
| source-only | 20.58 cm | 11.42 cm | 0.1230 |
| shuffled-text | 34.32 cm | 18.57 cm | 0.2439 |
| unrelated-source | 43.82 cm | 16.38 cm | 0.2276 |
| empty | 53.26 cm | 24.55 cm | 0.4538 |

对应 95% CI 均不跨 0。v5.1 的失败点不是 source/text 通路断开，而是精确 contact-map 与 transition 没有在当前剂量下转化为显著 endpoint 收益。

## 6. T2M 描述性守门

HumanML3D Test 4,042 条，EMA，ODE32，text CFG=2.0：

| 指标 | v5 200K | v5.1 200K | 变化 |
|---|---:|---:|---:|
| FID | 10.435 | 10.645 | +0.210，略退 |
| R@1 | 0.6027 | 0.5962 | -0.0064 |
| R@2 | 0.7511 | 0.7580 | +0.0069 |
| R@3 | 0.8236 | 0.8266 | +0.0030 |
| FK jerk | 53.64 | 53.91 | +0.26 |
| Foot-skate ratio | 0.2046 | 0.2063 | +0.0017 |

matched guardrail 已核对 4,042 条 case key、长度、seed 和 GT reference identity。R@1/R@2/R@3、FK jerk 和 foot-skate 的 paired 95% CI 均跨 0。描述性 guardrail 未见灾难性遗忘；由于没有预注册非劣界，这不构成正式 non-inferiority 结论，FID 也只作同协议点估计参考。

## 7. Edit 描述性守门

MotionFix Test 1,013 对，EMA，ODE32，source CFG=2.0，Edit CFG=3.0：

| 正确 source+instruction 分支 | v5 200K | v5.1 200K | 变化 | Paired 95% CI |
|---|---:|---:|---:|---:|
| Target joint error | 51.83 cm | 48.03 cm | -3.80 cm | [-5.42, -2.22] |
| Target rotation error | 49.26 deg | 46.70 deg | -2.56 deg | [-3.18, -1.91] |
| Prediction jerk magnitude | 197.37 | 139.94 | -57.43 | [-114.73, -25.55] |
| Foot-skate ratio | 0.3019 | 0.2744 | -0.0275 | [-0.0352, -0.0199] |

source 与 instruction 各自仍有条件贡献：

- shuffled-source/correct-instruction 的 target position degradation 为 `+4.18 cm`，95% CI `[0.60, 7.63]`，衡量正确 source 的贡献。
- same-source/shuffled-instruction 的 target position degradation 为 `+4.61 cm`，95% CI `[1.00, 8.17]`，衡量正确 instruction 的贡献。

matched guardrail 已逐 pair/system 核对 1,013 对的 instruction、source/target、长度、seed、gauge 和采样协议，并验证 checkpoint-independent source-copy 分支逐案不变。仍需避免错误解释：MotionFix target 与 source 高度相似，source-copy 的平均 target joint error 只有 `23.21 cm`，source-only 为 `24.05 cm`，仍显著低于正确编辑分支的 `48.03 cm`。普通 aggregate target MSE 奖励复制 source，不能作为 Edit 语义成功的主指标。当前能支持的结论仅是：v5.1 没有破坏 Edit 条件敏感性，并在 v5 CFG=3 的四项描述性指标上改善；语义编辑成功仍需视觉与 instruction-specific 指标确认。

## 8. 可视化

Reaction Test 共生成两套 action-balanced GIF：

```text
eval_v5_1_200k_final/reaction/test/gifs_action_balanced/
eval_v5_1_200k_final/reaction/test/gifs_action_balanced_fk/
```

后者使用 global-rotation FK，与 headline MPJPE/contact 指标的几何路径一致，共 16 条。

观察结论：

- 初始站位、接近过程和接触前朝向相对 v5 有可见改善，与 matched 指标一致。
- 接触动作类别通常可识别，但具体手-手、手-肩等 joint-pair、保持时长和释放时刻仍粗糙。
- 没有观察到系统性条件失效；source-only、shuffled-text 的失败明显更多。
- 部分样本主观上仍有局部抖动；定量上 GT jerk fidelity error 与 prediction jerk magnitude 均增加。两类证据方向一致，但 jerk 指标本身不等同于主观平滑性评分。

## 9. 200K 到 300K 剂量验证

用户决定保持控制变量，将同一 run 从 200K 续到 300K：

```text
configs/hy273_unified_fulltext_reaction_v5_1_full_contact_continue300k.yaml
scripts/launch/train_hy273_unified_reaction_v5_1_continue300k_ddp8.sh
```

仅有两项计划变化：

- 末段 schedule end：`200000 -> 300000`。
- max global step：`200000 -> 300000`。

保持不变：

- T2M/Edit/Reaction=`30/35/35`。
- 8 卡 DDP。
- 同一 model/optimizer/EMA/task-batcher state，并保持 stateless RNG contract、seed、stream cursor 与 sample ordinal。
- v5.1 全部 loss 和权重。
- Control/ease 关闭。
- 250K、300K 各保存 checkpoint。

300K 的主比较是**同一 v5.1 run 的剂量效应**：在同 UID、同初始噪声下比较 v5.1 300K 与 v5.1 200K，并做 paired bootstrap。它只能回答“额外 100K 同配方训练带来什么变化”，不能把跨训练时长变化单独归因于 full-contact loss。若要在 300K 再估计机制效应，必须另将 v5 按相同 `30/35/35` 配方续到 300K；本轮不安排该对照。

300K 的决策标准不是只看训练 loss。必须重复 Reaction Test/GIF，并检查：

1. pair-contact F1、contact-vector error、transition F1 的 300K-vs-200K paired CI。
2. 初始布局收益是否保持。
3. jerk fidelity error 与 prediction jerk magnitude 是否继续增加。
4. T2M/Edit 描述性 guardrail 是否仍稳定。

如果 300K 仍只改善 layout、而精确 contact lifecycle 不改善，应停止继续堆 v5.1 剂量，进入新的 Control 阶段或重新设计 contact state/监督；不能因为训练 loss 继续下降或 300K 相对 200K 变化，就扩大 200K 已有的机制因果结论。

## 10. 科学解释边界

- 当前只有一个训练 seed；bootstrap CI 覆盖测试 UID 抽样，不覆盖训练 seed 方差。
- Test 已参与多轮机制开发，结果属于 exploratory development evidence。
- 除预注册 contact family 外，layout 与其他守门指标使用未经多重比较校正的 pointwise CI，只能作为探索性 signal。
- v5.1 是四项 contact loss 的组合实验，缺少同预算逐项 ablation。
- 150K 使用 Val，200K 使用完整 Test，二者不能直接当同一分布 learning curve。
- T2M FID 使用 K273 到现有 evaluator domain 的 bridge，报告中只用于同协议相对守门，不声明为其他表示的官方绝对 benchmark。
- Edit 的 aggregate target MSE 对 source-copy 有结构性偏好；视觉和 instruction counterfactual 必须与其共同解释。
- checkpoint 和数据保留在本地，不上传 GitHub。
