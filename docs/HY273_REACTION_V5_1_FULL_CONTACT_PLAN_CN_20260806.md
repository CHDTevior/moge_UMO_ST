# HY273 Reaction-v5.1 全序列接触实验计划

## 1. 实验问题

Reaction-v5 已明显改善初始布局、相对方位、partner-facing 和首次接触时间，但仍常出现：动作类别正确，实际接触关节、保持时间或释放时间不正确。

v5 主要监督初始 15 帧、接触前阶段和 first-contact CDF；完整 22x22 joint-distance 使用 position channel，而最终渲染和 headline MPJPE 主要依赖 global-rotation FK。v5.1 只补这一处，不改变 backbone、参数量、条件输入或 `[B,T,273]` 输出。

## 2. 不变合同

- 100K parent：同一个 Unified Full-Text T2M checkpoint。
- 100K-200K update 比例：T2M/Edit/Reaction = `30/35/35`。
- 模型：root/body 双路径 Unified Actor Flow DiT，source token block，LLM2Vec sentence + context tokens。
- flow：`x0 prediction + velocity-MSE`。
- timestep：基础 `sigmoid(N(-0.8,0.8^2))`，Reaction 30% 额外采样 `t in [eps,0.15]`。
- optimizer、LR、batch、EMA、normalizer、数据与随机种子保持 v5 一致。
- Control/ease 保持关闭。
- 50K 存一个归档 checkpoint，即 150K 和 200K。

## 3. 新增监督

阈值统一为 FK joint-pair distance `< 0.15m`，仅在 v5 原有 fine gate `t>=0.20` 生效。

| 项 | 权重 | 含义 |
|---|---:|---|
| FK contact-map positive | 0.001 | 对 GT 接触 joint-pair 强化接触召回 |
| FK contact-map negative | 0.005 | 对 GT 非接触 joint-pair 抑制错误接触 |
| FK contact vector | 0.002 | 在 GT 接触 pair 上精修 FK 接触位置和方向 |
| FK contact transition | 0.003 | 监督相邻帧 contact probability 变化，覆盖 onset/hold/release |

contact-map 使用由距离构造的 soft Bernoulli 分布，并计算 `KL(target || prediction)`。GT 正 pair 与负 pair 分开归一化，避免正接触约 `0.1%` 的稀疏度把召回梯度淹没。

欧氏距离在预测 joint-pair 精确重叠时没有径向梯度。对“GT 非接触、预测距离小于 `1e-4m`”的退化点，negative map 保持原前向数值，但反向沿 GT pair 分离方向仅路由到 FK 根平移载体：`smooth_root XZ + pelvis-position Y`；普通非重叠样本及其他局部关节梯度不变。

FK contact vector 使用 source-heading local frame 的 coordinate-wise SmoothL1。需要准确解释：prediction 和 target 使用同一个 observed source 时，vector 残差中的 source 会代数抵消；它不是独立的新关系状态，而是由 GT 接触 pair mask 选择的 reactor FK 精修。真正非线性的双人关系新增项是 contact-map。

transition 比较每个 pair 相邻帧 soft contact probability 的差值。逐帧 contact-map 负责 hold 状态，transition 额外强调接触建立和释放边界。

## 4. L1、L2 与 SmoothL1

- 主 flow `velocity-MSE` 是 L2 loss：误差按平方惩罚。
- `torch.linalg.vector_norm(...,2)` 只是计算三维欧氏距离，不代表后续使用 L2 loss。
- Reaction root、heading、distance、vector 和 transition 多数采用 SmoothL1/Huber：小误差区近似 L2，用于精修；大误差区近似 L1，降低离群 pair 对梯度的支配。
- v5.1 不修改主 flow loss，只新增上述辅助几何项。

## 5. 真实梯度标定

从同一 100K parent 运行 8 卡、20 update、7 个真实 Reaction update 的无 checkpoint smoke。最终权重下，各组 output-gradient RMS 相对主 flow 为：

| 梯度组 | 与 base 的比值 |
|---|---:|
| v5 true layout | 1.080 |
| v5 event timing | 0.896 |
| v5 fine geometry | 0.312 |
| FK contact-map positive | 0.572 |
| FK contact-map negative | 0.014 |
| FK contact vector | 0.294 |
| FK contact transition | 0.091 |
| v5.1 full-contact 合计 | 0.850 |
| 全部 relation 合计 | 2.078 |

20-update smoke 无 NaN，pre-clip gradient norm `15.19`，峰值显存约 `14.74 GiB/GPU`，step time 约 `0.40s`。初始 `positive=0.004` 曾使 full-contact 达到约 `2.18x base`、全部 relation 达到 `3.38x base`，因此已否决。

## 6. Tensor Information Flow

```text
source K273 [B,T,273] -----------+
                                   +--> shared model --> reactor x0 [B,T,273]
noise/noisy reactor [B,T,273] ----+
text/task/timestep conditions -----+

training only:
source + predicted reactor --> [B,2,T,273] --> global-rot FK --> [B,2,T,22,3]
source + target reactor ----> [B,2,T,273] --> global-rot FK --> [B,2,T,22,3]
                                                    |
                                                    +--> pair vectors [B,T,22,22,3]
                                                    +--> pair distances [B,T,22,22]
                                                    +--> soft contact map [B,T,22,22]
                                                    +--> temporal delta [B,T-1,22,22]
```

这些张量只参与训练 loss，不进入模型输入，不增加推理计算。

## 7. 评估

固定 EMA、ODE32、source CFG=2.0、text CFG=2.0、fixed-role、同 UID 和确定性噪声。

新增 headline 指标：

- `fk_pair_close_15cm_precision/recall/f1`：是否碰对具体 joint-pair。
- `fk_contact_vector_error_cm_15cm`：GT 接触 pair 的 FK 向量误差。
- `fk_pair_transition_15cm_precision/recall/f1`：onset/release 是否以正确方向发生在正确 pair 和相邻帧；`0->1` 与 `1->0` 是两个不同事件。

使用当前实现重算 v5 200K Test 的 1,579 条已保存 source+text 预测，得到预注册基线：

| 指标 | v5 200K 基线 |
|---|---:|
| FK pair-close@15 precision | 0.0925 |
| FK pair-close@15 recall | 0.1637 |
| FK pair-close@15 F1 | 0.1182 |
| FK contact-vector error | 32.56 cm |
| FK pair-transition precision | 0.0054 |
| FK pair-transition recall | 0.0114 |
| FK pair-transition F1 | 0.00735 |
| FK MPJPE | 48.45 cm |
| FK relation-distance MAE | 20.47 cm |
| FK jerk error | 92.46 m/s^3 |

这说明 v5 的 scene-level Close@20 虽然较好，但具体关节 pair 和精确 transition 仍很弱，正是 v5.1 要检验的缺口。基线文件保存在本地：

```text
/mnt/afs/mogeflow-control/outputs/_smoke_hy273_reaction_v5_1/
  v5_200k_full_contact_metric_baseline.json
```

继续保留 v5 的 FK MPJPE、relation MAE、root/bearing/partner-facing、Close@10/20/30、first-close timing、jerk 和四种条件消融。接触改进不能以明显恶化 layout、jerk、T2M 或 Edit 为代价。

150K 先跑 Val 和 GIF；训练不中断并保存 checkpoint。200K 跑 Val/Test、GIF、T2M/Edit guardrail，再与 v5 同 step 的已保存预测做 matched comparison。

## 8. 启动入口

```bash
bash scripts/launch/train_hy273_unified_reaction_v5_1_full_contact_stage_b_ddp8.sh
```

评估示例：

```bash
RUN_NAME=<run> STEP=150000 SPLITS=val \
  bash scripts/launch/eval_hy273_unified_reaction_v5_1_checkpoint.sh

RUN_NAME=<run> STEP=200000 SPLITS=val,test \
  bash scripts/launch/eval_hy273_unified_reaction_v5_1_checkpoint.sh
```

## 9. 结论边界

- v5.1 是相对 v5 的单变量 loss-family 实验，但 family 内同时包含 map、vector 和 transition，不能把最终收益归因到其中单项。
- 当前只有一个训练 seed；clip bootstrap 不覆盖训练 seed 方差。
- Test 已用于早期机制分析，最终结果仍属于探索性开发证据。
- 若 pair-contact F1 和 transition F1 不提高，不能仅凭 scene min-distance 或视觉上“靠得更近”判定成功。
