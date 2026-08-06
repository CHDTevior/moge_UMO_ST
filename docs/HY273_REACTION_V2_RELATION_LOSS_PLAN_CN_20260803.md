# HY273 Reaction-v2 关系监督实验说明

## 1. 实验目标

本实验从同一个 100K T2M parent 重新训练 Reaction，目标是让固定 source actor 条件真正影响 reactor 的相对位置、朝向和近距离事件，同时保持 T2M 与 Motion Editing 能力。

控制变量保持不变：

- 数据：Inter-X Reaction、HumanML3D、MotionFix。
- 表示与归一化：同一套 K273、Mean/Std。
- 模型：同一 backbone、full-text token stream、source token block。
- 任务比例：T2M/Edit/Reaction = 30/35/35。
- flow：x0 prediction、velocity-MSE、logit-normal(-0.8, 0.8)。
- 优化器、LR、batch size、EMA 和保存间隔均不变。

唯一实验变量是 `reaction_loss`。

## 2. v1 问题

Reaction 固定 source，只预测 reactor：

```text
prediction_pair = [source, prediction]
target_pair     = [source, target]
```

因此线性相对位置残差严格化简：

```text
(prediction - source) - (target - source) = prediction - target
```

即便先乘 source heading 的旋转矩阵也仍然成立：

```text
R_source^T(prediction - source) - R_source^T(target - source)
= R_source^T(prediction - target)
```

所以“只投影到 source 局部坐标再比较完整向量”并不会产生新的关系监督。v1 的 `close_joint_vector` 同样只是根据 GT proximity 对 reactor joint reconstruction 重加权。

## 3. Reaction-v2 coarse relation

首先把 reactor root 的 XZ 位移投影到 source 每帧 heading 局部系：

```text
delta_local = R_source^T(root_reactor - root_source)
```

随后不比较完整线性向量，而比较非线性物理描述量：

```text
radius  = ||delta_local||
bearing = delta_local / max(||delta_local||, eps)
```

另构造 reactor 面向 source 的 descriptor：

```text
to_source = normalize(-delta_local)
facing    = [dot(heading_reactor_local, to_source),
             cross_2d(heading_reactor_local, to_source)]
```

对应三项：

```text
relative_root_radius   weight 0.02, scale 0.25m
relative_root_bearing  weight 0.01
partner_facing         weight 0.01
```

这些 coarse 项在全部 flow time 生效，负责从高噪声阶段规划大体距离、方位和面向关系。

## 4. Reaction-v2 fine geometry

### 4.1 Union distance map

v1 只在 `d_gt < 1m` 时比较 22x22 joint distance，因此不能约束 `GT far / prediction near`。

v2 使用：

```text
mask_distance = (d_gt < 1m) OR (d_pred < 1m)
```

在该集合内匹配预测与 GT 的完整 joint distance map，权重 0.01，距离尺度 0.10m。

### 4.2 Balanced soft proximity

使用连续 proximity：

```text
q(d) = sigmoid((0.20m - d) / 0.03m)
```

GT-near 与 relevant GT-far 分别求均值，再各承担总权重的一半，避免 22x22 大量负例淹没真实近距离事件。总权重为 0.02。

### 4.3 Target-aware false close

不采用无条件 `relu(0.08-d_pred)^2`，因为它会惩罚握手、拥抱等真实接触。

只在下面条件成立时激活：

```text
d_gt >= 0.20m AND d_pred < 0.08m
```

纯距离 hinge 在 `d_pred=0` 时没有可选方向，`torch.cdist` 会返回零梯度。因此实际损失沿 GT 分离方向计算。令：

```text
u_gt = vector_gt / ||vector_gt||
s_pred = dot(vector_pred, u_gt)
L_radial = [relu(0.08m - d_pred) / 0.08m]^2
L_direction = [relu(0.08m - s_pred) / 0.08m]^2
L_false_close = L_radial + 0.025 * L_direction
```

权重 0.005。mask 仍只选择 `d_gt>=0.20m AND d_pred<0.08m`；0.025 倍方向项只用于补足完全重合处的逃逸方向，避免取代原径向 penalty 后显著抬高梯度预算。它只处理 GT 明确较远但预测错误贴合的 joint pair。

## 5. 时间门

base reconstruction 与 coarse relation 覆盖全部时间。

22x22 distance、soft proximity 和 false-close 仅在：

```text
t >= 0.55
```

时生效。当前 logit-normal(-0.8, 0.8) 下约有 10.5% 的 Reaction 样本满足该条件。未选择 0.65，因为其覆盖率只有约 3.8%，第一轮实验监督过稀。

## 6. 初始 loss 预算标定

从同一 100K parent、同一 deterministic task sequence 分别运行 20 updates：

| 指标 | Reaction-v1 | Reaction-v2 |
|---|---:|---:|
| reaction total | 0.1816 | 0.1748 |
| relation weighted sum | 0.0816 | 0.0753 |
| relation/base output-gradient ratio | 1.21x | 1.25x |
| global preclip grad norm | 15.19 | 15.16 |
| step time | 0.364s | 0.368s |

当时 20 updates 的短标定显示 loss/gradient/throughput 量级接近；该结论不能外推到正式训练。事后前 500 updates 的 relation/base output-gradient ratio 均值约为 v1 `4.02x`、v2 `7.45x`，说明 v2 的有效关系梯度明显更大，不能再表述为“只改变方向而预算不变”。

## 7. Benchmark

继续保留：

- fixed source-to-reactor role，不允许 actor swap。
- source/text causal ablation。
- position/FK 两条重建路径一致性。

新增：

- relative root radius error。
- relative root bearing error。
- partner-facing error 与 bearing 有效帧比例。
- close precision/recall/F1 @ 10/20/30cm。
- false-close rate @ 10/20/30cm。
- missed-close rate @ 10/20/30cm。

所有 close-event aggregate/stratified 以及 causal advantage 均先在测试样本上汇总 TP/FP/FN，再做 matched cluster micro-bootstrap。训练中的 fine mask、false-close 等 coverage 日志同样汇总各 DDP rank 的 numerator/denominator 后求比，不使用 rank-local ratio 的等权平均。

事后口径修正：上述开发期 `precision 0.702 / recall 0.373` 不是最终锁定的同协议 v1-150K 全量结果，且其 evaluator/checkpoint 口径未被完整记录，不应作为可复现基线。最终固定协议 v1-150K Test micro 指标为 precision `0.6547`、recall `0.6964`、F1 `0.6749`、false-close `0.2598`、missed-close `0.3036`；后续设计以最终报告中的配对 A/B 为准。

## 8. 训练与决策

从 `step_00100000.pt` 启动八卡训练，先停在 150K。

150K 比较对象是原 v1 的同阶段 checkpoint，使用相同 EMA、32-step ODE、source/text CFG=2.0 和 fixed-role test protocol。

继续到 200K 的条件：

- root radius/bearing 与 heading 明显改善。
- relation distance MAE 改善。
- 20cm recall 提升且 precision 不发生明显坍塌。
- source/text causal advantage 保持为正。
- T2M 与 Edit benchmark/可视化无明显退化。
