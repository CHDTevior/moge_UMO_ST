# HY273 Ease 控制实施修订稿

更新时间：2026-07-28
项目性质：科研实验
输入材料：`outside_doc/EASE_INJECTION_DESIGN_CN.md`

## 1. 当前执行边界

Stage-BE `200K -> 250K` 保持原实验不变：

```text
T2M / Control / Edit = 60 / 0 / 40
```

该阶段不启用 Ease、不加入 Ease 参数、不修改 loss。只有 250K 的 raw/EMA Edit
门禁和 T2M 非退化评估完成后，才从冻结的 250K checkpoint 分叉 Control
bootstrap。

## 2. 原设计中必须修正的三点

### 2.1 不能在归一化 local-joint 通道上直接计算

Ease 标签必须由增强后的、未归一化物理 K273 重建全局关节：

```python
global_joints = reconstruct_global_joints_from_features(target_physical)
centroid = global_joints.mean(dim=-2)
```

`[5:71]` 的 X/Z 是相对 smooth root 的局部量；直接对它求均值会丢失全局
轨迹，不能表示整段运动的 ease-in/out。

### 2.2 文档公式的端点和分母不一致

原文档伪代码使用末端 `p[k-1]`，却除以 `k`。这样即便质心严格匀速直线
运动，Ease 也不为零，而且误差随序列长度增长。

本项目采用端点严格、可单测的离散定义。对任意一段长度为 `M >= 2` 的轨迹
`q[0:M]`：

```text
u_m = m / (M - 1)
linear_m = (1 - u_m) * q_0 + u_m * q_(M-1)
E(q) = mean_m(q_m - linear_m)
```

长度为 `L >= 4` 的有效序列拆成：

```text
k = floor(L / 2)
E_in  = E(p[0:k])
E_out = E(p[k:L])
ease_physical = concat(E_in, E_out)  # [6], unit: meter
```

这个定义满足：

- 任意平移不变；
- 任意匀速直线轨迹严格为零；
- yaw 旋转后，两个 3D 向量按同一个 yaw 等变旋转；
- 使用均值而不是求和，避免标签仅因有效帧数增加而机械放大。

它是对 IBMM 离散指标的变长序列适配，不声称逐下标复刻论文实现。

### 2.3 Ease 必须有独立统计，且水平轴统计要保持 yaw 等变

不能复用 HY273 的 273D mean/std。训练集统计只使用 Ease 自身：

```text
mean  : [6]
std   : [6]
format: hy273_ease_stats_v1
```

由于训练采用均匀随机全局 yaw，水平 X/Z 不能分别拟合不同尺度，否则归一化
会破坏旋转等变性。对 `E_in`、`E_out` 分别使用：

```text
mean_x = mean_z = 0
std_x = std_z = sqrt(E[x^2 + z^2] / 2)
```

Y 轴单独拟合 mean/std。统计只由 train split 构建；HML 行按训练 sampler 的
“motion row 等概率、行内 caption 等概率”权重累计，避免 caption 多的 motion
被无意放大。

## 3. 数据合同

在 `ConditionBatch` 中增加：

```text
ease_physical: [B, 6] float32
ease_present : [B] bool
```

约束：

- `ease_present=False` 时，`ease_physical` 必须为有限零 sentinel；
- Ease 标签由已经完成 root-origin shift 和随机共享 yaw 的
  `target_physical` 计算，因此与 `frame_gauge_dir` 在同一坐标系；
- 初版仅对 `TaskId.GENERATE` 使用 Ease；
- Motion Editing replay 初版 `ease_present=False`，不让新条件改变 Edit
  语义；
- padding 帧不得进入 Ease 计算。

显式 `ease_present` 是必要的。归一化后的零向量表示“训练分布均值附近的
Ease”，不能同时拿它表示“用户未提供 Ease”。

## 4. 模型接入

配置：

```yaml
ease:
  enabled: false
  stats_dir: ""
```

默认关闭。启用时，模型持有 Ease normalizer buffer，并增加：

```text
6 -> H -> H MLP
```

末层 weight/bias 全零初始化。前向中始终计算：

```python
ease_norm = normalize(ease_physical)
ease_bias = ease_embed(ease_norm)
ease_bias = ease_bias * ease_present[:, None]
```

然后分别加到 root/body 的 target frame hidden：

```text
root_hidden += ease_bias[:, None, :]
body_hidden += ease_bias[:, None, :]
```

接入原则：

- 不进入 text cross-attention；
- 不与 timestep/direction 共用 AdaLN；
- token-block source fusion 时只加到 target tokens，不加到 source tokens；
- root/body 使用同一 Ease bias；
- 即使一个 batch 全部 absent，也保留 Ease 参数在 autograd graph 中，满足
  DDP 的一致梯度拓扑；
- 从 250K checkpoint 迁移时，仅允许新增 Ease 参数缺失，旧参数必须全部
  严格命中；保留旧 optimizer moment，并为新增参数初始化空状态。

## 5. Control bootstrap 训练安排

候选主比例：

```text
T2M / Control / Edit = 10 / 70 / 20
```

不能回到 `10/90/0`，否则刚学到的 Edit 会被遗忘。Ease 使用组合覆盖，而不是
所有 GENERATE 样本都强制提供：

```text
T2M:
  Ease absent  75%
  Ease present 25%

Control:
  Ease absent  50%
  Ease present 50%

Edit:
  Ease absent 100%
```

折算到全局更新：

```text
T2M only          7.5%
T2M + Ease        2.5%
Control only     35.0%
Control + Ease   35.0%
Edit replay      20.0%
```

这样同时覆盖原始 T2M、原始 Control、T2M+Ease、Control+Ease 和 Edit replay，
不会把“有 Control”与“有 Ease”绑定成一个条件。

初版保持原 reconstruction/control/contact loss，不增加 Ease auxiliary loss，
用单一变量判断 additive 条件本身是否可学。若足剂量训练后 Ease
assignment/monotonicity 仍为 null，再单独实验可微 Ease consistency loss。

## 6. 推理与 CFG

初版不增加独立 `g_ease` 外推分支。用户提供 Ease 时，把同一 Ease 条件放入
现有 empty/text/control/joint 各分支：

```text
x_empty(ease)
+ g_text    * [x_text(ease)    - x_empty(ease)]
+ g_control * [x_control(ease) - x_empty(ease)]
```

这样 Ease 以直接条件强度 `1.0` 生效，现有 text/control CFG 差分仍只表示各自
增量。先验证模型确实学习 Ease，再决定是否增加独立 Ease CFG。

## 7. 启动门禁与评估

实现门禁：

1. 物理平移不变；
2. yaw 等变；
3. 匀速直线严格零；
4. padding/变长正确；
5. `ease_present=False` 时贡献严格零；
6. Ease 末层零初始化后，250K old/new model 输出一致；
7. checkpoint 迁移与 DDP resume 一致。

训练评估：

- 固定 checkpoint、text、hard control、noise，只改变 Ease；
- 对 `E_in/E_out` 分别做 target-direction scale sweep；
- 报告生成 Ease 对请求 Ease 的 normalized MAE、Spearman 单调性；
- 同时报 root/path/end-effector/control error，防止通过破坏硬控制实现 Ease；
- 复查 T2M 困难文本、Edit same-source gate、contact/foot-lock 非退化。

## 8. 执行顺序

```text
Stage-BE 到 250K
-> raw/EMA Edit + T2M 评估
-> 实现/验证 Ease 标签与 stats builder
-> 实现默认关闭的 ConditionBatch/模型/采样接口
-> 科研正确性审核
-> 从冻结 250K 分叉 Control bootstrap
-> Control/Ease/Edit/T2M 联合评估
```
