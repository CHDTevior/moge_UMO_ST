# HY273 K-Encoder 统一模型 400K 最终评估

日期：2026-07-29
性质：科研实验记录
评估对象：K-Encoder T2M + Motion Editing + Kimodo-like Control + Ease 统一模型

## 1. 核心结论

最终 checkpoint：

```text
/mnt/afs/mogeflow-control/outputs/hy273_text_fusion/
hy273_kencoder_stageBC_ease_t2m10_ctrl70_edit20_ddp8x16_20260728_201555/
model/step_00400000.pt
```

本轮训练在数值、DDP、resume 和吞吐层面正常完成。下列能力结论的证据强度不同，当前没有预定义的跨任务 utility 可以证明某个 checkpoint 对所有用途最优：

1. **Control 是 400K 统一模型证据最完整的能力项。**
   - HumanML3D test 全量 4,042 motions、with-text/notext 共 8,084 cases 已全部完成。
   - 19 种 Kimodo-like control subtype 全覆盖。
   - 相比上一轮 R13 400K，本模型在 with-text 条件下改善 full-pose、endpoint rotation、contact calibration/consistency 和足滑。
   - 主要回退是 endpoint position：with-text 平均误差从 5.53 cm 增至 6.30 cm。
   - 因此这是明确的 Pareto trade-off，不应压缩成无条件的 “best Control checkpoint”。

2. **固定 T2M probe 显示 400K 仍有文本条件敏感性。**
   - 在一个 concept、两个 paraphrase、一个 counterfactual 和 3 个 seeds 上，400K 对 `break dancing` 与 `walking slowly` 产生不同输出。
   - 相比 250K，两个 paraphrase routing 的 3-seed 均值提高，但每项都只有 2/3 seeds 改善。
   - 相比纯 T2M 的 200K，`breaking dance` 更接近 canonical，但 `breakdances` 仍明显更差。
   - 这是输出间距离 probe，不证明动作本身语义正确，也不是完整 T2M benchmark。

3. **Stage C 的 Edit 证据是混合的，尚不能排序 250K 与 400K。**
   - target reconstruction、joint error 和 speed MAE 的点估计略有改善。
   - faster/slower 的 mean-speed proxy、repeat 的 peak-count direction 和 timing 的 activity-center proxy 多数下降。
   - repeat/timing 的若干匹配 progress descriptor 持平或改善。
   - 62-pair 配对敏感性区间均跨 0；250K 可作为偏 Edit 的保守候选，但未被统计证明优于 400K。

4. **Ease 条件路径已连通，但尚未成为可靠能力。**
   - 一部分无控制样例呈正确单调响应。
   - 响应斜率总体偏小，部分方向反号。
   - 在当前唯一的 hands+feet 强控制样例中，Ease 两半均呈近零反向响应；该现象尚不能一般化为所有 hard control 都会压过 Ease。
   - 不能仅凭 Spearman 的绝对值把 Ease 宣称为成功。

综合判断：

```text
Training health: PASS
Control evidence: COMPLETE，存在 endpoint 与其余指标的 trade-off
T2M probe:       EXPLORATORY，保留部分条件敏感性
Motion Editing:  INCONCLUSIVE，250K/400K 各有优劣
Ease:            NOT YET RELIABLE
Unified model:   COMPLETED CANDIDATE，未建立跨任务最优性
```

## 2. 训练阶段

| 阶段 | Step | T2M / Control / Edit | 目标 |
|---|---:|---:|---|
| Stage A | 0K -> 200K | 100 / 0 / 0 | K-Encoder T2M 基模 |
| Stage BE | 200K -> 250K | 60 / 0 / 40 | 保留 T2M，建立 Motion Editing |
| Stage C | 250K -> 400K | 10 / 70 / 20 | Control bootstrap、Ease、Edit replay |

Stage C 的 Ease 覆盖：

```text
T2M:     25%
Control: 50%
Edit:     0%
总体期望: 37.5%
```

Edit 条件分支：

```text
source + instruction  80%
source identity        10%
instruction only        5%
unconditional           5%
```

## 3. 表示、预测目标和 Loss

### 3.1 表示与预测

输出仍为完整 Kimodo273：

```text
[0:3]     smooth_root_pos
[3:5]     global_root_heading
[5:71]    local_joints_positions
[71:203]  global_rot6d
[203:269] global_joint_velocities
[269:273] foot_contacts
```

模型参数化保持为：

```text
network output: normalized clean x0_hat over all 273 channels
base training objective: group-balanced velocity-space MSE
contact: z-normalized unified clean-flow, separate weighted velocity-MSE group
```

实际 velocity-equivalent 目标为：

```text
d = max(1 - t, 0.05)
v_hat = (x0_hat - z_imputed) / d
v_gt  = (x0     - z_imputed) / d
MSE(v_hat, v_gt) = MSE(x0_hat, x0) / d^2
```

Contact 与 continuous 一起使用 unified-273 stats。400K checkpoint 的 contact stats 为：

```text
mean = [0.660750, 0.729053, 0.670056, 0.730754]
std  = [0.473455, 0.444449, 0.470192, 0.443568]
```

采样后先 denormalize，再以 `0.5` 阈值二值化。代码中的局部变量名 `contact_logits` 是历史命名，在本协议中不是独立 BCE-logit 预测头。

文本使用 K-Encoder 路线：

```text
LLM2Vec Llama-3-8B token cache
conditioning_architecture = llm2vec_flux
text_global_conditioning = llm2vec_tokens_only
```

本路线不再把 CLIP pooled vector 加入全局 AdaLN 条件。

实际生效的 K-Encoder 配置由 YAML 与 launch CLI override 共同决定。基础 YAML 中仍保留旧 `hy_cache` 字段，不能单独用作本 run 的最终文本配置；复现时必须使用启动脚本，并以 checkpoint `runtime_identity` / `research_overrides` 中的以下字段为准：

```text
conditioning_architecture = llm2vec_flux
llm2vec_cache_dir = .../llm2vec_llama3_8b_profile_v1
text_global_conditioning = llm2vec_tokens_only
text_fusion_mode = f00
base_representation_loss_space = velocity_mse
base_contact_loss_space = velocity_mse
```

### 3.2 基础 Loss

```text
representation_scale = 0.09397019716051493
semantic channel weights = [10, 2, 10, 10, 3]
contact = 0.010739451104058849
clean_root_velocity = 0.01
clean_joint_velocity = 0.01
foot_lock = 0.01
fk_consistency = 0.07
```

Control 附加项：

```text
control_continuous = 0.25
control_contact = 0.02857142857142857
```

Edit positive-only 附加项：

```text
target_x0_scale = 0.05
hard_x0_scale = 0.02
hard_fraction = 0.20
instruction_rank_scale = 0.0
```

主要 auxiliary loss 的空间为：

```text
control_continuous/contact: normalized clean-x0 SmoothL1
clean_root_velocity:        physical m/s SmoothL1
clean_joint_velocity:       FK physical m/s SmoothL1
foot_lock:                  GT-contact frame physical foot velocity -> 0
fk_consistency:             physical position residual / 0.05 m, then SmoothL1
```

Edit 的 `target_x0/hard_x0` 只作用于 269 个 continuous channels，并排除 hard-observed mask；`hard_fraction=0.20` 表示每个 semantic block 中当前误差最大的 20% 元素，不是 hard-control 区域。`instruction_rank_scale=0`，因此本轮没有负指令排序监督。

Ease 使用独立 stats 和 `6 -> H -> SiLU -> H` MLP；只有 output projection 的 weight/bias 为零初始化，input projection 不是。相同 Ease bias 加到 root/body hidden，且没有独立 auxiliary Ease loss。

## 4. 训练稳定性

以下数值是每个 10K 区间内的日志均值：

| 区间 | Overall | T2M | Control | Edit | Total grad | Ease grad | samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 250K-260K | 0.003223 | 0.002978 | 0.003690 | 0.001471 | 0.04635 | 0.004276 | 405.9 |
| 300K-310K | 0.002283 | 0.002426 | 0.002539 | 0.001255 | 0.02470 | 0.000288 | 408.9 |
| 350K-360K | 0.002013 | 0.002230 | 0.002234 | 0.001083 | 0.02227 | 0.000274 | 411.2 |
| 390K-400K | 0.001865 | 0.002074 | 0.002064 | 0.001013 | 0.02211 | 0.000263 | 402.4 |

终点 step 400K：

```text
overall loss = 0.002032
T2M loss     = 0.002285
Control loss = 0.002189
Edit loss    = 0.001256
total grad   = 0.02796
step time    = 0.343 s
throughput   = 402.1 samples/s
```

训练期间未观察到：

```text
NaN
OOM
DDP rank divergence
cache lookup failure
resume state mismatch
```

注意：Ease grad 从早期约 `4.28e-3` 降到末期约 `2.63e-4`，同时最终 Ease 响应偏弱。两者是并存现象，本评估没有识别其因果关系。

## 5. T2M 评估

### 5.1 协议

```text
EMA
ODE32
text CFG = 2.0
matched noise
seeds = [3407, 12345, 20260725]
length = 150
```

文本：

```text
canonical:       a person is break dancing.
gerund variant:  a person is breaking dance.
verb variant:    a person breakdances.
counterfactual:  a person is walking slowly.
empty:           ""
```

`routing = distance(paraphrase, empty) - distance(paraphrase, canonical)`，越大表示 paraphrase 更靠近 canonical。
`variant vs canonical` 越低越好。

### 5.2 结果

下表使用 global-joint MPJPE 距离，单位为米：

| Checkpoint | Canonical vs counterfactual | Gerund routing | Verb routing | Gerund vs canonical | Verb vs canonical |
|---|---:|---:|---:|---:|---:|
| Stage A 200K | 0.822 | 0.466 | 0.695 | 0.239 | **0.061** |
| Stage BE 250K | 0.831 | 0.182 | 0.415 | 0.411 | 0.347 |
| Stage C 400K | **0.972** | 0.330 | 0.507 | **0.207** | 0.215 |

判断：

1. 在该固定 probe 上，400K 的 canonical/counterfactual 输出距离均值最大，说明文本改变仍能影响输出。
2. 400K 相比 250K 的两个 routing 均值更高，但两项都只有 2/3 seeds 改善，seed 波动较大。
3. Gerund variant 在 400K 的 3/3 seeds 都比 200K 更接近 canonical。
4. Verb variant 在 400K 的 3/3 seeds 都比 200K 更远离 canonical。
5. 这些结果不能证明 canonical/counterfactual 动作本身正确，也不能单独证明整体 T2M retention 或 “10% replay 阻止灾难性遗忘”；后者需要同起点、同预算的 no-replay continuation。

该 panel 只有一个 concept、两个 paraphrase、一个 counterfactual 和 3 seeds，只定位为探索性条件敏感性 probe，不扩展为完整 T2M quality、FID 或 R-precision 结论。

## 6. Motion Editing 评估

### 6.1 协议

```text
MotionFix test
62 unique selected pairs
64 category slots: faster/slower/repeat/timing 每类 16
002680/001470 各跨两个类别，因此只有 62 个 unique pairs
EMA
ODE32
source CFG = 2.0
edit CFG = 2.0
matched source/instruction/noise
one sampling seed per pair
```

### 6.2 250K 与 400K

下表先在每个 category 内做 pair-macro，再对四类等权。`Continuous target MSE` 是 269 个 continuous channels 在 normalizer 空间中的无量纲 MSE：

| Metric | Stage BE 250K | Stage C 400K | 变化 |
|---|---:|---:|---:|
| Normalized continuous target MSE | 0.5432 | **0.5028** | -7.4% |
| Global joint target error | 11.96 cm | **11.73 cm** | -0.23 cm |
| Velocity target MAE | 0.1419 m/s | **0.1367 m/s** | -3.7% |
| Speed target MAE | 0.1940 m/s | **0.1843 m/s** | -5.0% |
| Mean-speed direction accuracy | **70.18%** | 66.94% | -3.24 pp |
| Mean-speed target progress | **47.81%** | 33.67% | -14.14 pp |

Mean-speed alignment 只统计 source-target mean-speed 差超过阈值的 active joint dimensions；progress 被裁剪到 `[-2, 3]`。各类有效 pair 数分别为 faster `14`、slower `15`、repeat `16`、timing `13`，因此这两项不是通用自然语言编辑正确率。

Mean-speed 分类型点估计：

| 类别 | Target MSE 250K -> 400K | Mean-speed direction | Mean-speed progress |
|---|---:|---:|---:|
| faster | 0.702 -> **0.612** | **67.46% -> 55.69%** | **44.20% -> 32.74%** |
| slower | 0.567 -> **0.535** | 68.38% -> 66.52% | **42.40% -> 33.71%** |
| repeat | 0.686 -> **0.629** | 65.59% -> 64.57% | **47.93% -> 34.56%** |
| timing | **0.218 -> 0.234** | 79.29% -> **80.96%** | **56.72% -> 33.68%** |

对 repeat/timing 使用更匹配的物理 descriptor 后，结果有升有降：

| Category | Descriptor | 250K -> 400K |
|---|---|---:|
| faster | mean-speed direction / progress | 67.46% -> 55.69% / 44.20% -> 32.74% |
| slower | mean-speed direction / progress | 68.38% -> 66.52% / 42.40% -> 33.71% |
| repeat | peak-count direction / progress | 65.91% -> 59.68% / **59.54% -> 61.11%** |
| timing | activity-center direction / progress | 78.35% -> 68.17% / 50.81% -> 39.39% |
| timing | activity-onset direction / progress | 74.31% -> 74.08% / **43.10% -> 47.50%** |
| timing | activity-offset direction / progress | 67.49% -> 64.93% / **34.40% -> 34.85%** |

62 个 unique pair 的 post-hoc 配对 cluster bootstrap 敏感性分析（100,000 resamples，同一 pair 的跨类别 membership 共用权重，四类等权）为：

| Delta = 400K - 250K | 点估计 | 约 95% percentile 区间 |
|---|---:|---:|
| Normalized target MSE | -0.0404 | [-0.1057, +0.0123] |
| Global joint error | -0.225 cm | [-1.244, +0.688] cm |
| Mean-speed direction | -3.240 pp | [-9.246, +2.315] pp |
| Mean-speed progress | -14.141 pp | [-35.249, +3.616] pp |

所有区间跨 0，且不包含 sampling-seed/training-seed 不确定性。因此当前只能说：

```text
400K 的重建类点估计略好；
faster/slower 与部分 repeat/timing descriptor 的点估计变差；
另一些 repeat/timing progress descriptor 持平或改善；
现有 panel 不足以证明 250K 或 400K 是总体更好的 Edit checkpoint。
```

这些变化与 Stage C 多任务 continuation 同时发生，但没有 edit-only、no-Control 或 no-Ease 的同起点反事实，不能归因为 `70% Control / 20% Edit` 本身造成了任务干扰。

### 6.3 Edit CFG

两个 CFG sweep 共覆盖 7 个固定 pair、单 seed。该有限 panel 表明：

1. `edit CFG=2` 是当前样例上的保守 fidelity 默认值。
2. `CFG=3` 偶尔提高某个方向指标，但 target MSE 和动作幅度开始明显增大。
3. `CFG=4/5/6` 在多个样例上快速外推，不能作为修复 Edit 语义的通用方法。
4. `000038 move feet faster` 在 CFG 3 以上出现明显误差放大。

因此当前推荐：

```text
source CFG = 2.0
edit CFG   = 2.0
```

这不是全 MotionFix test distribution 的 CFG 最优性结论；它只说明盲目提高 CFG 没有在当前 7 个样例上稳定解决 Edit 问题。

## 7. Kimodo-like Control 全量 Benchmark

### 7.1 协议与完整性

```text
HumanML3D test motions: 4,042
Control subtypes:       19
Text regimes:           with-text + notext
Total cases:            8,084
Weights:                EMA
Sampler:                ODE32
Text CFG:               2.0
Control CFG:            2.0
Seed:                   3407
Primary output:         generated_raw
```

`generated_raw` 的准确含义是 **pre-terminal-exact-overwrite conditional sampler output**：ODE state 没有 persistent exact clamp，但每个去噪 step 的 controlled CFG branch 会在模型输入中看到 observation overwrite。它不是完全无条件的裸网络输出，也不是最后一个 clean-x0 branch prediction。

聚合结果：

```text
status: validated
cases:  8084 / 8084
duplicate case keys: 0
failed cases:        0
```

持久 text cache 缺少 2 条唯一 caption。这 2 条 caption 在 3 个 shard 中涉及 4 个 case，已按同一 LLM2Vec profile runtime encode：

```text
runtime shard rows: 4
runtime unique text/profile pairs: 2
runtime cases: 4
```

四个 case 均纳入最终 summary，没有删除或以空文本替代。

### 7.2 400K 总体指标

距离单位为 cm，旋转为 degree，足部速度为 m/s。报告的是 `generated_raw`，不是 terminal exact-clamp。

本节和 7.3/7.4 的主表均为 **case-macro**：先在单个 case 内对适用 observation 求均值，再对适用 case 等权平均。`N` 是 case 数，不是 frame/keyframe/endpoint/contact-entry 数。采用 case-macro 是为了让每个生成请求等权，避免长序列或 dense constraint 自动获得更大权重。

| Regime | Root cm | Root@10cm | EE cm | EE rot | Full cm | Contact acc | Contact F1 | Contact BCE | FK gap cm | Contact consistency | Contact skate | Max skate | Skate ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| with-text | 4.095 | 97.482% | 6.296 | 4.471 | 2.576 | **99.976%** | 99.891% | **0.00332** | 0.158 | 92.752% | 0.0805 | 0.7204 | 25.086% |
| notext | **4.031** | 97.620% | **5.999** | **4.288** | **2.533** | 99.924% | **99.912%** | 0.01046 | **0.147** | **93.225%** | **0.0722** | **0.6427** | **23.200%** |
| GT | 3.691* | 99.851%* | 0.000 | 0.000* | 0.000 | 100.000% | 100.000% | 0.000001 | 0.000 | 100.000% | 0.0416 | 0.1325 | 21.740% |

`*`：root metric 使用 rotation-FK root，而 observation 是 smooth-root position，因此 GT 也存在约 3.69 cm 的表示口径下限。

### 7.3 相比旧 R13 control-best 400K

旧基线：

```text
outputs/hy273_multitask/
hy273_r13_contactflow_controlled_staged_ddp8_20260720_040507/
model/step_00400000.pt
```

两者使用相同 8,084 case plan、初始噪声、ODE32、CFG 2/2、benchmark protocol 和 metric schema。新 summary format 为 v4、旧 summary format 为 v3，评估器保存版本并非逐字相同；逐 case 核对确认 case key、constraint payload、GT metric dict 和 generated metric schema 完全配对，因此可以比较。

下面的 `delta = K-Encoder 400K - R13 400K` 使用 case 为配对单位，并已重新生成可复现统计产物：

```text
.../eval_step400k/
control_vs_r13_case_paired_bootstrap_10k_seed20260729.json
```

固定协议为：

```text
method: paired percentile, two-sided
resampling unit: case_key
resamples: 10,000
confidence: 0.95
base seed: 20260729
row seed: sha256(base_seed, text_regime, metric)
indices shared across metrics: false
quantile method: numpy linear
estimand: generated_raw case-macro, candidate minus baseline
```

因此下表区间可由 `tools/summarize_hy273_control_paired_bootstrap.py` 逐位复现。它们仍只表示 **固定 checkpoint、固定 sampling seed 下的 case variation**，不覆盖 sampling seed、training seed 或 checkpoint-selection 不确定性。

#### With-text

| Metric | N | Old -> New | Delta | 95% CI | 判断 |
|---|---:|---:|---:|---:|---|
| Root error | 2,340 | 4.440 -> 4.095 cm | -0.345 | [-1.161, +0.355] | 无明确变化 |
| Root@10cm | 2,340 | 96.366 -> 97.482% | +1.116 pp | [+0.579, +1.662] | 改善 |
| Endpoint position | 1,702 | 5.532 -> 6.296 cm | +0.764 | [+0.512, +1.010] | **回退** |
| Endpoint rotation | 1,702 | 5.016 -> 4.471 deg | -0.545 | [-0.805, -0.330] | 改善 |
| Full-pose | 1,489 | 2.995 -> 2.576 cm | -0.419 | [-0.526, -0.321] | 改善 |
| Contact accuracy | 1,273 | 99.835 -> 99.976% | +0.141 pp | [+0.106, +0.180] | 改善 |
| Contact BCE | 1,273 | 0.02279 -> 0.00332 | -0.01947 | [-0.02471, -0.01469] | 改善 |
| FK gap | 1,273 | 0.202 -> 0.158 cm | -0.043 | [-0.047, -0.040] | 改善 |
| Contact consistency | 4,042 | 90.452 -> 92.752% | +2.300 pp | [+2.108, +2.491] | 改善 |
| Contact-foot velocity | 4,042 | 0.1000 -> 0.0805 | -0.0196 | [-0.0227, -0.0165] | 改善 |
| Max foot velocity | 4,042 | 1.0530 -> 0.7204 | -0.3326 | [-0.3655, -0.3002] | 改善 |
| Foot-skate ratio | 4,042 | 26.780 -> 25.086% | -1.694 pp | [-2.005, -1.378] | 改善 |

#### Notext

| Metric | N | Old -> New | Delta | 95% CI | 判断 |
|---|---:|---:|---:|---:|---|
| Root error | 2,340 | 4.228 -> 4.031 cm | -0.197 | [-0.853, +0.293] | 无明确变化 |
| Endpoint position | 1,702 | 4.083 -> 5.999 cm | +1.916 | [+1.699, +2.138] | **回退** |
| Endpoint rotation | 1,702 | 4.016 -> 4.288 deg | +0.272 | [+0.175, +0.366] | 回退 |
| Full-pose | 1,489 | 2.263 -> 2.533 cm | +0.270 | [+0.206, +0.334] | 回退 |
| Contact accuracy | 1,273 | 99.973 -> 99.924% | -0.049 pp | [-0.094, -0.012] | 小幅回退 |
| Contact BCE | 1,273 | 0.00369 -> 0.01046 | +0.00676 | [+0.00151, +0.01274] | 回退 |
| Contact consistency | 4,042 | 92.742 -> 93.225% | +0.483 pp | [+0.342, +0.621] | 改善 |
| Contact-foot velocity | 4,042 | 0.0816 -> 0.0722 | -0.0093 | [-0.0110, -0.0077] | 改善 |
| Max foot velocity | 4,042 | 0.7687 -> 0.6427 | -0.1260 | [-0.1463, -0.1047] | 改善 |
| Foot-skate ratio | 4,042 | 25.832 -> 23.200% | -2.632 pp | [-3.102, -2.182] | 改善 |

科学解释：

1. 新统一模型不是简单复制旧 control-best。
2. with-text Control 的 full-pose、rotation、contact 和足部动力学明显更强。
3. endpoint position 是一致且显著的主要回退项。
4. notext 下 endpoint/full-pose/contact observation fidelity 回退，但足滑仍改善。
5. 在当前固定评估协议内，`text CFG=2.0, control CFG=2.0` 是已完整跑完的标准模式；不能从本表推出它是所有 CFG 的总体最优值，也不能把 notext 当作同等成熟模式。

Observation-micro 敏感性检查没有翻转主要方向，但效应大小会改变：

| With-text metric | Case-macro old -> new | Observation-micro old -> new |
|---|---:|---:|
| Root error | 4.440 -> 4.095 cm | 3.967 -> 3.789 cm |
| Endpoint position | 5.532 -> 6.296 cm | 4.631 -> 4.926 cm |
| Full-pose | 2.995 -> 2.576 cm | 2.859 -> 2.312 cm |

因此这些结果回答的是“一个随机 case 的平均表现”，不是“一个随机 observation entry 的平均表现”。

### 7.4 19 种 With-text Subtype

| Subtype | N | Root cm | EE cm | Rot deg | Full cm | Contact acc | Contact skate | Skate ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| path_2dpos | 213 | 3.739 |  |  |  |  | 0.0605 | 25.52% |
| path_2dposrot | 213 | 3.742 |  |  |  |  | 0.0609 | 21.75% |
| waypoint_2dpos | 213 | 6.794 |  |  |  |  | 0.0661 | 24.57% |
| waypoint_2dposrot | 213 | 4.306 |  |  |  |  | 0.0604 | 21.68% |
| inbetweening | 213 |  |  |  | 2.402 |  | 0.0622 | 17.98% |
| random | 213 |  |  |  | 2.853 |  | 0.0755 | 24.61% |
| feet_posrot | 213 |  | 6.207 | 7.844 |  |  | 0.0919 | 29.65% |
| hands_posrot | 213 |  | 10.187 | 3.415 |  |  | 0.0693 | 22.59% |
| hands_feet_posrot | 213 |  | 6.123 | 4.728 |  |  | 0.0929 | 28.01% |
| root_ee_hands_feet_posrot_fullbody | 213 | 3.852 | 3.903 | 4.268 | 2.557 |  | 0.1051 | 32.68% |
| root_ee_hands_posrot | 213 | 3.907 | 9.614 | 3.346 |  |  | 0.0713 | 23.19% |
| root_ee_hands_posrot_fullbody | 213 | 3.806 | 4.824 | 3.134 | 2.401 |  | 0.0964 | 28.14% |
| root_path_fullbody | 213 | 3.786 |  |  | 2.770 |  | 0.0890 | 27.99% |
| contact_only_sparse | 213 |  |  |  |  | 99.940% | 0.0915 | 21.40% |
| root_sparse_contact | 212 | 3.867 |  |  |  | 99.978% | 0.0712 | 22.36% |
| root_dense_contact | 212 | 3.622 |  |  |  | 99.989% | 0.0617 | 19.67% |
| endpoints_contact | 212 |  | 5.805 | 4.629 |  | 100.000% | 0.1060 | 27.15% |
| fullpose_contact | 212 |  |  |  | 2.657 | 99.949% | 0.0965 | 28.90% |
| mixed_contact | 212 | 3.619 | 3.690 | 4.404 | 2.393 | 100.000% | 0.1006 | 28.82% |

难点仍集中在：

```text
hands-only endpoint position
feet rotation
root + endpoint + full-pose composite
endpoint/full-pose/contact composite 的足滑
```

## 8. Ease 400K 因果 Sweep

### 8.1 协议

固定：

```text
checkpoint / text / initial noise / sampler / control
```

只沿 reference Ease direction 改变一个 half：

```text
scale = [0, 0.25, 0.5, 0.75, 1.0]
```

另一个 half 保持 stats mean。正确响应应当同时满足：

```text
Spearman 为正
linear response slope 为正且有可见幅度
active-half MAE 随目标合理变化
cross-half drift 小
```

`linear response slope` 是 normalized output projection 对 unit normalized scale 的无量纲斜率；`active-half MAE` 位于独立 Ease stats 的标准化空间，不是米。每个样例的五个 scale 点共享 checkpoint、文本、初始噪声和 control，不是五个独立样本。

### 8.2 结果

| 样例 | Control | Ease-in rho / slope | Ease-out rho / slope | Active MAE in/out | 判断 |
|---|---|---:|---:|---:|---|
| 000708 | none | +1.0 / +0.00330 | -1.0 / -0.00397 | 1.80 / 1.94 | 一半反号，幅度弱 |
| 003800 | none | +1.0 / +0.01191 | +1.0 / +0.09470 | 1.07 / 1.21 | 最明确的正向样例 |
| 005818 | none | +0.9 / +0.00915 | +0.9 / +0.01124 | 2.11 / 1.91 | 单调但幅度弱 |
| 003800 | hands+feet posrot | -1.0 / -0.00094 | -1.0 / -0.00096 | 1.31 / 1.25 | hard control 下近似失效 |

hands+feet 控制本身从 340K 到 400K 改善：

```text
endpoint position: 5.31 cm -> 3.66 cm
endpoint rotation: 4.04 deg -> 3.54 deg
```

但该单一样例中的 Ease response 同时接近零并反号。这与模型在该样例中优先满足硬约束的解释一致，但不足以建立一般化机制结论。

当前 Ease 结论：

```text
condition path is active
causal response is sample-dependent
response magnitude is usually weak
composition with hard control is not established
```

下一轮不能只增加训练步数。至少需要：

1. 更明确的 Ease 监督或可观测 response loss。
2. 提高包含 Ease 的有效梯度比例。
3. 单独评估 Ease 与 hard-control 的冲突。
4. 在更多 held-out 样例上定义成功阈值。

## 9. 可视化资产

### 9.1 T2M

```text
/mnt/afs/mogeflow-control/outputs/hy273_text_fusion/
hy273_kencoder_stageBC_ease_t2m10_ctrl70_edit20_ddp8x16_20260728_201555/
eval_step400k/t2m_visual16_ema_cfg2/gifs
```

Paraphrase panel：

```text
.../eval_step400k/t2m_paraphrase_ema_cfg2/stageBC400k_ema/gifs
```

### 9.2 Edit

250K 与 400K 同 pair 对照：

```text
.../eval_step400k/dynamic_edit_parent250k_vs_final400k/visuals
```

`000038` CFG sweep：

```text
.../eval_step400k/edit_cfg_sweep_000038/visuals
```

指定样例 `004884 / 002493 / 001866`：

```text
.../eval_step400k/edit_cfg_sweep_requested_004884_002493_001866/visuals
```

### 9.3 Control

19 个 subtype，每类 25%/50%/75% 误差分位各 1 条，共 57 条 GIF：

```text
/mnt/afs/mogeflow-control/outputs/hy273_text_fusion/
hy273_kencoder_stageBC_ease_t2m10_ctrl70_edit20_ddp8x16_20260728_201555/
eval_step400k/kimodo_v5_gallery_withtext_q25_q50_q75
```

每个 case 的 GIF：

```text
cases/<index>_<subtype>_<quantile>_<motion_id>/media/control_raw.gif
```

可视化直接使用 benchmark 的 pre-terminal-exact-overwrite conditional sampler output，并显示：

```text
generated skeleton
motion trail
root path / waypoints
endpoint position and rotation targets
full-pose ghost targets
contact targets
per-case constraint and foot-skate metrics
```

已抽查 root path、hands+feet 和 mixed-contact 的首帧与中间帧；目标可见、动画非空、布局无重叠。复杂 mixed-contact 的 q50 样例仍可见约 49% foot-skate ratio，符合 benchmark 对复杂组合仍有物理质量余量的判断。

## 10. Checkpoint 使用建议

当前没有预定义跨任务 utility，以下是按已观察 trade-off 给出的候选选择，而不是通用 “best checkpoint” 排名：

| 用途 | 候选 checkpoint | 证据边界 |
|---|---|---|
| T2M-only 后续实验起点 | Stage A 200K | 只在当前单 concept probe 的 verb variant 上优于 400K；未做完整 benchmark |
| Edit-heavy 后续实验起点 | Stage BE 250K | 更接近纯 Edit 训练阶段，部分 Edit proxy 点估计更好；未证明总体优于 400K |
| 统一 T2M/Edit/Control | Stage C 400K | 当前唯一完成三任务联训的 checkpoint |
| with-text full-pose/rotation/contact/foot dynamics | Stage C 400K | 对旧 R13 的相关 case-macro 指标改善，但 endpoint position 回退 |
| notext endpoint/full-pose control | 旧 R13 control-best 400K | 新模型在该模式下 endpoint/full-pose 明确回退 |

Stage C 标准推理参数：

```text
T2M:
  text CFG = 2.0

Edit:
  source CFG = 2.0
  edit CFG = 2.0

Control:
  text CFG = 2.0
  control CFG = 2.0
  primary output = generated_raw (pre-terminal exact overwrite)
```

## 11. 下一步实验建议

本轮结果不支持在没有新对照和门禁的情况下继续用完全相同的 `10/70/20` 配比续训。原因：

1. Control 已出现明确的指标 trade-off，继续同类剂量的收益方向未知。
2. Edit reconstruction 点估计下降，但 instruction-matched physical descriptors 有升有降。
3. Ease 路径可影响输出，但当前响应弱且样本依赖；仅延长训练是否有效未知。

下一轮应保持单一统一 backbone，不引入 task adapter，但改进多任务训练本身：

1. 以 Stage C 400K 的 Control 作为 retention baseline。
2. Edit success metric 按 instruction category 预定义：speed、peak count、activity timing、changed-region 和 identity preservation，不能用单一 mean-speed proxy 或 target MSE 统摄。
3. 在更新前测量 T2M/Control/Edit 梯度夹角和各条件分支梯度量级，确认实际冲突位置。
4. Edit 使用 source-motion-only 与 source+instruction 的分解式 CFG，不靠提高总 CFG。
5. 优先试 changed-region reweight 的 positive reconstruction，未编辑区保留小权重维持 identity。
6. Control replay 继续保留 contact/foot dynamics 指标，尤其 endpoint composite。
7. Ease 单独做有成功阈值的小规模 ablation；在其单独成立前，不把它作为统一模型已具备的正式能力。
8. 若要识别多任务干扰，应从同一 250K 起点做 edit-only、Control/Ease on/off 或不同 replay ratio 的等预算 continuation。

建议的下一轮决策门槛：

```text
Edit:
  speed/peak-count/timing 各自 primary descriptor 不低于预设基线
  changed-region improvement 与 unedited-region identity 同时达标
  target MSE 不显著回退

Control:
  endpoint position 不再高于旧 R13 400K
  contact-foot velocity / max velocity / skate ratio 不回退

T2M:
  扩展 prompt/paraphrase panel，多 seed，并预设保留阈值

Ease:
  held-out samples 正向 Spearman + 正向最小 slope
  hard-control composition 不反号
```

## 12. 最终判断

这次实验完成了 K-Encoder 统一模型的一条训练轨迹，并验证了文本、source、Control 和 Ease 条件路径可以共同运行。现有证据支持训练健康和若干 Control 指标变化；T2M、Edit 与 Ease 仍是探索性证据，不能据此证明统一训练中的具体因果干扰：

```text
Control: with-text full-pose/rotation/contact/foot dynamics 改善，endpoint 回退；
T2M: 固定 probe 仍有条件敏感性，整体 retention 未建立；
Edit: 重建与 instruction-matched descriptors 呈混合变化；
Ease: 条件能影响输出，但缺少稳定、可组合的响应证据。
```

因此本轮不是训练失败，但也不能把 200K、250K 或 400K 描述为各能力的统计最优解。400K 是当前唯一完成三任务联训并完成全量 Control benchmark 的 **Control 主导统一候选**；250K 是 Edit-heavy 后续实验的保守起点，而不是已证明优于 400K 的 Edit 最优 checkpoint。
