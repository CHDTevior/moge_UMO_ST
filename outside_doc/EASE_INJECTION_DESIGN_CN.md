# 在 moge_UMO_ST 上引入 Ease-in/Out 连续时间控制：背景、论文依据、推理过程与实验方案

更新时间：2026-07-28
文档性质：科研设计与决策记录（供 team review）
代码仓库：`github.com/CHDTevior/moge_UMO_ST`

---

## 0. 一页速览（TL;DR）

- **起点问题**：能否用我们已有的 Kimodo-like 系统复现 Disney 的 Generative Motion Rig (GMR)？GMR 是一个"以生成式运动模型为引擎"的创作系统，引擎可替换，我们的模型可以充当它的 ML-Betweener 引擎。
- **逐 gap 逆向后**，结合我们自己的取舍，当前唯一要动模型的增量是 **Gap 6：ease-in/out 连续时间控制**（缓入缓出旋钮）。其余 gap 要么已被我们的选择消解（Gap 3 骨架对齐），要么暂不考虑（Gap 2/4/5/7），要么是纯工程（Gap 1 闭环）。
- **ease 信号的本质**：整段动作的**全局静态属性**（两个向量 E_i / E_o），不是逐帧时序信号。
- **融合方式的最终结论（经更正）**：对全局静态条件，**additive 注入到 per-frame 隐藏态**最稳健、零额外参数、有 NanoWM 消融数据背书；**明确避开 cross-attention**（RT-1 崩盘）；adaLN-fuse 在 NanoWM PushT 上恰是最差，不采用。FiLM 作为"愿加参数换保真"的备选。
- **训练代价**：**不重训 T2M，不单开 ease 训练轮**。靠末层零初始化保证起点与现有 200K 逐比特等价，ease 标签搭 control bootstrap 阶段的顺风车一起学，由 T2M replay 保护语义。
- **落点时序**：不要在 Stage-BE (Edit bootstrap, 200K→250K) 加 ease；正确落点是其后的 control bootstrap 阶段，和 control 标签一起上。

---

## 1. 背景与目标

### 1.1 起点

从 Disney Research 的一篇 SIGGRAPH Talks '26 短文出发：

> **A Generative Motion Rig for Artist-Driven Motion Authoring**
> Buhmann, Agrawal, Borer, Vögeli, Sumner, Guay（DisneyResearch|Studios + ETH Zürich）
> DOI `10.1145/3799818.3812088`

核心问题（用户提出）：**如果已经训好一个类似 Kimodo 的系统，能不能复现这篇论文的系统？能的话怎么做，不能的话有哪些 gap；对有 gap 的部分做逆向技术拆解，为后续复现做准备。**

### 1.2 关键判断

GMR **不是一个模型，而是一个创作系统**（Blender/Maya 插件 + GPU 服务端）。论文明确说其框架"与其他生成式运动引擎兼容"，并直接引用 CondMDI（Cohan et al. 2024 的扩散 in-betweening）作为可替换引擎的例子。

含义：**我们的 Kimodo-like 模型可以插进去当引擎**，但 GMR ≠ 引擎，它在引擎外面还包了创作循环、表示层与工程基础设施。

### 1.3 我们的真实起点（现状对账）

| 能力 | 对应 GMR 部件 | 我们的状态 |
|---|---|---|
| ML-Betweener（稀疏约束→整段运动 + inpainting 编辑 + 多样性采样） | 运动引擎 | ✅ 已有 Kimodo-like 系统（moge_UMO_ST，t2m 已训到 200K，control/edit 待训） |
| ML-Poser（稀疏效应器→单帧全身 pose 的神经 IK） | 交互摆 pose | ✅ 已自训 SMPL-IK |

**两个核心引擎件（最难的模型侧）已就位。** 剩余工作是把它们缝进创作循环的那层，以及若干增量能力。

---

## 2. GMR 系统拆解（六大部件）

| # | 部件 | 论文里的实现 | 作用 |
|---|------|-------------|------|
| 1 | ML-Betweener（核心引擎） | IBMM = Implicit Bézier Motion Model [Vögeli 2025] | 从稀疏约束生成整段运动 |
| 2 | ML-Poser（神经 IK） | ProtoRes [Oreshkin 2022] | 从稀疏关节约束补全**单帧**全身 pose |
| 3 | NMC（Neural Motion Curves） | SKEL-Betweener [Agrawal 2024] | 可点击拖拽的"神经运动曲线" |
| 4 | Client-Server 基础设施 | 自研 | 模型跑 GPU 服务端，客户端 Blender 插件 |
| 5 | Motion Editing | 扩散过程 inpainting 引导 | 保留 base motion、局部重生成 |
| 6 | 图层 + Rig 切换 + 时间控制 | 自研 UI + ease-in/out | 图层混合、FK/IK/GMR 互转、缓入缓出 |

创作范式：**"generative keyframing"**——艺术家给稀疏姿态/handle/窗口长度/噪声种子，采样出整段运动，再 click-and-drag 微调。

### 2.1 逆向发现：NMC 不是独立训练的表示

SKEL-Betweener 原文：neural motion curves 就是"特定关节被生成出来的轨迹"，"曲线上任意一点都能被选中、变成模型里的约束，从而交互式重生成"。

即：**拖曲线上第 t 帧的点 = 在 (关节 j, 时刻 t) 加一个稀疏约束 = 重跑推理**。这把"NMC 需要重训一个表示模型"降级为"引擎本来就支持稀疏关节约束，只差一个曲线↔约束的闭环 UI + 速度"。

---

## 3. 参考文献与代码材料清单

### 3.1 核心论文（均已拉全文精读，非仅摘要）

| 论文 | 标识 | 在本调研中的角色 |
|---|---|---|
| **GMR**（Generative Motion Rig） | DOI `10.1145/3799818.3812088`，SIGGRAPH Talks '26 | 复现目标系统；3 页概述 |
| **SKEL-Betweener** | DOI `10.1145/3687941`，TOG 43(6) 2024 | NMC 表示层原理；18 层 skeletal graph transformer |
| **IBMM**（Implicit Bézier Motion Model） | DOI `10.1145/3769047.3769052`，MIG 2025 | GMR 引擎原型；**ease-in/out 公式来源（§3.3）** |
| **CondMDI**（Flexible Motion In-betweening） | arXiv `2405.11126` | 扩散 in-betweening 对照（与 Kimodo 同族） |
| **ProtoRes** | arXiv `2106.01981`，ICLR 2022 | ML-Poser 原型；effector sampling 数据增强 |
| **Kimodo 技术报告** | NVIDIA（nv-tlabs/kimodo） | 我们引擎的参照系；两阶段去噪器、约束注入机制 |
| **SMPL-IK** | arXiv `2208.08274`，SIGGRAPH Asia 2022 | 我们 ML-Poser 的实际实现（ProtoRes 的 SMPL 版） |
| **IK-GAT / Amortized IK** | arXiv `2604.16629`，2026 | 前馈 IK 对照（但任务是稠密→朝向，非稀疏补全） |
| **NanoWM**（Nano World Models） | arXiv `2605.23993` | **融合方式消融的数据依据（Table 3，五种 action-injection）** |

### 3.2 代码材料

- **我们的仓库**：`CHDTevior/moge_UMO_ST`（316 文件）。关键文件：
  - `docs/CURRENT_T2M_EDIT_CONTROL_DESIGN_CN.md` — 当前系统真实状态的设计文档
  - `models/raw_motion/kimodo_like_flow_dit.py` — root-first/body-second 双 DiT + K-Encoder 文本路径 + `_compose_global_condition`（**ease 注入相关**）
  - `models/codeflow/dit_blocks.py` — flux backbone（`FrameMotionTextDiT` / `DoubleStreamBlock` / `AdaLNModulation` = AdaLN-Zero）
  - `models/raw_motion/hy273_constraints.py` — Kimodo-like Control compiler + curriculum（**ease 标签宜归此处**）
  - `train_hy273_multitask.py` / `sample_hy273_multitask.py` — 训练与分层 CFG 采样入口
  - `data/hy273_multitask_scheduler.py` — 任务比例与 Control 条件 pattern
- **ProtoRes 权重核实**：`boreshkinai/protores` 是空壳仓库（仅 README+gif），代码已迁移到 `Unity-Technologies/Labs/.../ProtoRes/code`（122 文件，附 minimixamo/miniunity 数据 zip，**无预训练权重**）。
- **SMPL-IK 核实**：`boreshkinai/smpl-ik`（95★），**只有训练代码、无预训练权重**，且是 SMPL（body）非 SMPL-X。

---

## 4. 我们系统的真实架构（moge_UMO_ST，与原始 Kimodo 的差异）

读代码 + 设计文档后确认的关键差异——这些差异**直接决定 ease 的改法**：

| 维度 | 原始 Kimodo | 我们的 moge_UMO_ST |
|---|---|---|
| 生成范式 | DDPM(ε) + DDIM 100 步 | **Rectified flow（clean-x0 预测）** |
| 表示 | SOMA 77 关节 | **HY273（273D，22 关节，30 FPS）** |
| 双阶段去噪 | root→body | 同样 root-first/body-second 双 DiT ✅ |
| 文本路径 | LLM2Vec pooled | **K-Encoder：LLM2Vec token-only（进 attention），文本不进 AdaLN** |
| 全局条件(AdaLN) | timestep + dir | **`cond = timestep_token + direction_token`**（`_compose_global_condition`） |
| backbone block | — | **AdaLN-Zero**（`AdaLNModulation` 返回 shift/scale/**gate**，gate 零初始化） |
| 约束注入 | imputation + mask 拼特征维 | **完全一致**：`model_in = concat(z_t_imputed, mask)` → [B,T,546] ✅ |
| CFG | separated text/constraint | **分层加法 CFG**（empty / text / control / joint 四分支） |

### 4.1 HY273 273D 表示（逐通道）

```text
x: [B,T,273]
[0:3]     smooth_root_pos
[3:5]     global_root_heading
[5:71]    local_joints_positions, 22x3     ← 算 COM 用这段
[71:203]  global_rot_data, 22x6
[203:269] global_joint_velocities, 22x3
[269:273] foot_contacts, 4
```

### 4.2 当前 checkpoint 状态

```text
K-Encoder Stage-A 200K：T2M / Control / Edit = 100% / 0% / 0%
Control/Edit 代码路径已实现，但 200K 尚未在这些任务上更新。
```

T2M 与 Control 都是 `GENERATE` 任务、**共享全部 root/body backbone**，仅由 observation mask/values 区分（不靠额外模式开关）。这是 ease 能"搭 control 顺风车"的结构基础。

---

## 5. Gap 分析与我们的取舍

| Gap | 内容 | 我们的决定 |
|---|---|---|
| Gap 1 | NMC 曲线↔约束闭环（招牌交互） | 纯工程/UI，后续做；引擎侧已支持稀疏关节约束 |
| Gap 2 | 交互级推理速度（DDIM 慢） | **暂不考虑时效性**（先用 Kimodo，不上 ARDY） |
| Gap 3 | SMPL-IK ↔ 引擎骨架对齐 | **已消解**：统一用 SMPL/SMPL-X（仅 main body），同骨架无对齐问题 |
| Gap 4 | client-server + Blender 插件 | **暂不考虑** |
| Gap 5 | 图层 + FK/IK/GMR rig 切换 | **暂不考虑** |
| **Gap 6** | **ease-in/out 连续时间控制** | **本文档主题——现在要做的模型侧增量** |
| Gap 7 | 重生成 pop / 加约束稳定性 | **暂不考虑**（GMR 作者亦未解决，扩散固有） |

**结论：当前唯一要动模型的增量是 Gap 6。** 下面是它的完整技术拆解与实验方案。

---

## 6. Gap 6 深入：Ease-in/Out 连续时间控制

### 6.1 它解决什么

动画十二原理之一：真实运动很少匀速。ease-in = 起步慢、逐渐加速；ease-out = 结尾逐渐减速。相比 text-to-motion 只能用"慢/快"粗标签，IBMM 的 ease 信号是**连续的**，可细粒度控制"这段怎么起、怎么收"。

### 6.2 核心洞察：用质心(COM)偏离匀速的程度量化 easing —— 可全自动打标签

画"质心已走距离 / 总距离"随时间曲线：ease-out 前陡后平，ease-in 前平后陡，匀速为直线。所以 **easing 强度 = 真实轨迹相对"匀速直线运动"的累计偏差**，可从任意 mocap 片段自动算出，**无需人工标注**。

### 6.3 数学定义（IBMM 原文 Eq.1-2，已核对原文）

设 $\bar{p}_n$ = 第 $n$ 帧所有 $J$ 个关节位置的均值（COM 代理），序列长 $N$，分割点 $k=N/2$：

$$
E_i = \sum_{n=0}^{k-1}\left(\bar{p}_n - \left[\bar{p}_0 + \frac{n}{k}\left(\bar{p}_{k-1}-\bar{p}_0\right)\right]\right)
$$

$$
E_o = \sum_{n=k}^{N}\left(\bar{p}_n - \left[\bar{p}_k + \frac{n-k}{N-k}\left(\bar{p}_N-\bar{p}_k\right)\right]\right)
$$

中括号内是"若这半段匀速直线运动、第 $n$ 帧质心应在哪"（端点线性插值）。$E$ = 真实质心逐帧减匀速参考、累加。$E_i/E_o$ 分开算 → 一条动画独立控制起与收。注意 $\bar p_n$ 是 3D，故 $E_i,E_o$ 各是 **3D 向量**，拼成 6D。

### 6.4 IBMM 原始接法 vs 我们该怎么接

- IBMM（transformer encoder 扩散）：加两个 token $E_i,E_o$ append 到输入序列尾，各过 MLP 嵌入，并加 attention mask 让每帧按索引只 attend 其中之一。
- **我们不该照搬 append**：我们的 flux backbone 有清晰的**全局条件通路**（AdaLN），ease 是全局静态属性，应走全局条件而非 pose 序列 token。**具体注入方式在第 7 节由数学+数据论证后确定。**

---

## 7. 融合方式的思考过程（本调研的核心推理，含更正）

用户提出的关键质疑：多模态 condition DiT 有多种融合方式，AdaLN 相加未必最优。参照 **NanoWM（arXiv 2605.23993）** 专门做的 action-injection 消融（五种机制）。

### 7.1 NanoWM Table 3 真实数值（已核实，FID↓ 越低越好）

| 机制 | RT-1 FID | PushT FID | 额外参数 |
|---|---|---|---|
| **additive**（逐元素加） | 42.27（2nd） | **23.89（best）** | **0** |
| **FiLM** | **40.62（best）** | 25.45（2nd） | +14.4M |
| adaLN（专用头） | 43.62 | 26.32 | +42.5M |
| cross-attention | **51.12（worst）** | 28.64 | +28.3M |
| adaLN-fuse（与 timestep 融合） | 43.03 | **30.28（worst）** | 0 |

论文 **Finding #3**：action 注入**任务相关**——FiLM 在 RT-1 视觉保真最好，但简单 additive 在 PushT 最强且质量/参数性价比最优。

> **⚠️ 更正记录（重要）**：本调研过程中，我曾一度推荐 **adaLN-fuse** 并称其"零额外参数且性能与 additive 持平"。**这是错的**：如上表，adaLN-fuse 在 PushT 上 FID=30.28，是五种里**最差**的（比 cross-attention 的 28.64 还差）。cross-attention 只是 RT-1 最差、并非"两环境都最差"。更正后的数据指向 **additive**（两环境都强 + 零额外参数），而非 adaLN-fuse。

### 7.2 但 ease ≠ NanoWM 的 action：一个关键数学差异

NanoWM 的 action 是**每帧一个** $a_t$（时序对齐的密集信号）；我们的 ease 是**整段两个向量**（全局、静态）。这个差异要求我们从数学上重新审视每种机制在"全局静态条件"下退化成什么。设隐藏态 $h\in\mathbb{R}^{B\times T\times H}$，ease 嵌入 $e\in\mathbb{R}^{B\times H}$：

- **additive**：$h \leftarrow h + e$（broadcast 到所有帧）。给每帧加同一偏置向量。紧跟的 LayerNorm 会去掉均值分量、削弱一部分信号——这是理论上的小顾虑。但 NanoWM 经验显示 additive 仍最稳。
- **AdaLN（专用头）**：$h \leftarrow (1+\text{scale}(e))\odot\text{LN}(h)+\text{shift}(e)$。逐通道重标定，作用在 LN 之后不被去均值吃掉。数学上"全局属性调制统计量"很自然，但 NanoWM 里要 +42.5M 参数且不占优。
- **adaLN-fuse**：ease 加进 timestep embedding 共享 AdaLN 投影，零参数。**但 NanoWM PushT 上最差**，且会让 ease 与 timestep 语义纠缠、破坏 CFG 可分离性。**不采用。**
- **FiLM**：$h \leftarrow \gamma(e)\odot h+\beta(e)$，无 LN。NanoWM RT-1 最好，但要新增专用头、优势不跨环境。作为备选。
- **cross-attention**：ease 作 K/V。对两个全局向量，query-key 匹配退化、softmax 近似常权，等于用巨大参数学一个加权平均。**数据（两环境偏弱、RT-1 崩盘）与数学（无时序结构 → attention 归纳偏置失效）一致否定。明确避开。**

### 7.3 最终结论

对 ease 这种**全局静态条件**，排序（综合数据 + 数学 + 我们的架构）：

**additive ≳ FiLM > adaLN（专用头） > adaLN-fuse ；cross-attention 明确排除**

**推荐：additive 注入到 per-frame 隐藏态。** 理由：(1) NanoWM 两环境都强、零额外参数、Finding #3 直接背书；(2) 只多一个 6→H 的 MLP，工程最省；(3) 避开了 cross-attention 的坑和 adaLN-fuse 的弱项。**备选：FiLM**（愿花个位数 M 参数换保真时）。

---

## 8. 实验方案（精确到文件与函数）

设计原则：**默认关闭、零初始化、可分离、搭车训练**——加了代码但不影响现有 200K 基模与 Stage-BE，想开再开。

### 改动点 1 — 模型：加 ease embedder + additive 注入
**文件**：`models/raw_motion/kimodo_like_flow_dit.py`

**(a) `__init__`（约 line 114，紧挨 `self.direction_embed` 之后）**：
```python
self.use_ease = bool(use_ease)   # 从 config 传入，默认 False → 向后兼容
if self.use_ease:
    self.ease_embed = nn.Sequential(
        nn.Linear(6, self.hidden_dim), nn.SiLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
    )
    # 关键：零初始化末层 → t=0 时 ease 贡献恒为 0，不扰动已训好的 T2M
    nn.init.zeros_(self.ease_embed[-1].weight)
    nn.init.zeros_(self.ease_embed[-1].bias)
```

**(b) `forward`：在 backbone 消费隐藏态前，把 ease 嵌入 broadcast 加到 root/body 隐藏态**（additive，注入点是 `h` 而非 `cond`）：
```python
def forward(self, model_in, t, c_dir=None, text=None, ..., ease=None, ...):
    ...
    ease_bias = None
    if self.use_ease and ease is not None:
        ease = ease.to(device=device, dtype=dtype).view(bsz, 6)
        ease_bias = self.ease_embed(ease).unsqueeze(1)   # [B,1,H] broadcast over T

    root_hidden = self.root_input_proj(model_in)
    if ease_bias is not None:
        root_hidden = root_hidden + ease_bias            # additive 注入
    root_hidden = self.root_backbone(motion=root_hidden, ...)
    ...
    body_hidden = self.body_input_proj(body_in)
    if ease_bias is not None:
        body_hidden = body_hidden + ease_bias            # 同一 bias 注入 body
    body_hidden = self.body_backbone(motion=body_hidden, ...)
```

> 注意：**不改** `_compose_global_condition`（那条路会变成被否决的 adaLN-fuse）。ease 走 additive 到 `h`，`cond` 保持 `timestep + c_dir` 原样。

### 改动点 2 — 数据：算 E_i / E_o 标签
**文件**：`models/raw_motion/hy273_constraints.py`（标签由 dataset/batcher 注入 `ConditionBatch`）
```python
def compute_ease(x, k=None):
    # x: [B,T,273]，已在 unified-stats 归一化空间
    B, T, _ = x.shape
    joints = x[..., 5:71].reshape(B, T, 22, 3)   # local_joints_positions
    pbar = joints.mean(dim=2)                     # COM 代理 [B,T,3]
    k = T // 2 if k is None else k
    n_in = torch.arange(k, device=x.device).float()
    lin_in = pbar[:, 0:1] + (n_in / k).view(1, k, 1) * (pbar[:, k-1:k] - pbar[:, 0:1])
    E_i = (pbar[:, :k] - lin_in).sum(dim=1)       # [B,3]
    n_out = torch.arange(k, T, device=x.device).float()
    lin_out = pbar[:, k:k+1] + ((n_out - k) / (T - k)).view(1, T-k, 1) * (pbar[:, T-1:T] - pbar[:, k:k+1])
    E_o = (pbar[:, k:] - lin_out).sum(dim=1)      # [B,3]
    return torch.cat([E_i, E_o], dim=-1)          # [B,6]
```
**关键坑（我们系统特有）**：必须在**和 backbone 同一套 unified stats 的归一化空间**里算 `pbar`，否则 ease 尺度与条件对不上、微调学不动。设计文档已约定两套数据共享 unified stats。

### 改动点 3 — 训练：传标签 + drop 支持 CFG
**文件**：`train_hy273_multitask.py`、`data/hy273_multitask_scheduler.py`
- GENERATE 样本（T2M + Control）都算 ease 标签，喂进 `forward(ease=...)`。
- **按 `ease_drop_prob ≈ 0.1~0.2` 置零**（零初始化 embedder 输出本就是 0，drop 等价"无 ease 条件"），使推理可对 ease 单开一路 CFG。
- **loss 不改**——ease 是纯条件输入，靠 clean-x0 velocity MSE 自然学会"条件→节奏"对应。

### 改动点 4 — 采样：给 ease 一路分层 CFG
**文件**：`sample_hy273_multitask.py`
```python
x_guided = x_empty \
    + g_text    * (x_text    - x_empty) \
    + g_control * (x_control - x_empty) \
    + g_ease    * (x_ease    - x_empty)   # x_ease: 仅 ease 条件、其余置空的分支
```
起步可 `g_ease=0`（等于把 ease 当普通条件直接喂），验证有效后再加权。

---

## 9. 训练时序与代价：需不需要重训 T2M？

**不需要重训 T2M，也不需要单开 ease 训练轮。** 三层含义分清：

1. **从零重训 T2M？——不需要，零风险。** 末层零初始化的 `ease_embed` 使 `ease_embed(ease)≡0`，additive 加到 `h` = 加 0，前向输出**与现有 200K 逐比特一致**。用 `strict=False` 加载：旧权重全命中，只有 `ease_embed` 是新的。
2. **一次专门的 ease 微调？——不需要，合进 control 阶段。** ease 与 control 同属 `GENERATE`、共享 backbone。在 roadmap 本就存在的 **control bootstrap / 联合阶段**给 GENERATE 样本**同时打 control + ease 标签**即可。ease 边际成本 ≈ 0（多算个标签 + 一个 6→H MLP），**不新增训练阶段**。
3. **那个阶段 backbone 会更新吗？——会，但本就要发生，且 T2M 受保护。** control 训练同样共享并更新 backbone；ease 只是搭车，未引入新更新源。设计文档 §10 的 **T2M replay 比例**（如 60% replay / 联合阶段 10% T2M）正是为保护 T2M 文本路由不退化，ease 自动享受同一保护。

**时序纪律**：**不要在 Stage-BE（Edit bootstrap, 200K→250K）加 ease**——该阶段只碰 Edit/source 通路，ease 是干扰。ease 正确落点 = Stage-BE 之后的 control bootstrap 阶段，与 control 标签一起上。

（可选的更保守做法：该阶段冻结 backbone、只训 `ease_embed`。**不建议**——backbone 需要学会响应新信号，冻结会让 ease 学不动；交给 T2M replay 兜底、让 backbone 正常更新更好，与 control 完全同一套逻辑。）

---

## 10. 验证方法（sanity checks）

1. **向后兼容验证**：`use_ease=True` 但 ease 全零输入，前向输出应与 `use_ease=False` / 现有 200K 逐比特一致（零初始化保证）。
2. **ease 有效性**：固定其他条件，只扫 $E_i$ 从小到大，观察生成运动的**质心速度曲线是否单调从"匀速"变到"强缓入"**。$E_o$ 同理。
3. **CFG 可分离性**：改 `g_ease` 时 text/control 语义不应漂移。
4. **标签分布**：对训练集批量算 $E_i/E_o$，检查分布合理（匀速片段接近 0，明显加减速片段幅值大且符号正确）。
5. **T2M 不退化**：control+ease 阶段后，复查 Stage-A 困难文本/paraphrase、jerk/foot skate，确认 replay 保护住语义。

---

## 11. 更正记录（本调研过程中被修正的结论，如实保留）

| # | 曾经的说法 | 更正 | 依据 |
|---|---|---|---|
| 1 | Kimodo "自带用户级通用重定向" | 错。G1 模型是用 SOMA retargeter 对**训练数据**离线重定向训出来的，非用户级工具 | Kimodo 技术报告 |
| 2 | Kimodo 主力骨架是 SMPL-X | 错。主力是 **SOMA(77 关节)**；SMPL-X 是 R&D-license 变体 | Kimodo README |
| 3 | ProtoRes 有开箱即用预训练权重（路线 A 依据） | 错。`boreshkinai/protores` 是空壳；Unity 官方 repo 有代码+数据但**无权重**。SMPL-IK 同样无权重。**无论哪条路都要自训** | GitHub API 核实 122 文件 |
| 4 | 推荐 adaLN-fuse，称"零参数且性能与 additive 持平" | 错。NanoWM PushT 上 adaLN-fuse FID=30.28 是五种**最差**。更正为推荐 **additive** | NanoWM Table 3 |
| 5 | "cross-attention 两环境都最差" | 不准确。cross-attention 仅 RT-1 最差(51.12)；PushT 最差是 adaLN-fuse(30.28)。但 cross-attention 仍应避开（RT-1 崩盘 + 数学上 attention 对全局条件失效） | NanoWM Table 3 |

---

## 12. 结论与下一步

- **能复现 GMR 的核心创作范式吗？** 能。引擎可替换，我们的 Kimodo-like 模型 + SMPL-IK 已覆盖两个最难的模型件。
- **当前唯一要动模型的增量** = Gap 6 ease-in/out，且已有精确到函数的实验方案。
- **推荐融合方式** = additive 注入隐藏态（数据+数学双背书），避开 cross-attention 与 adaLN-fuse。
- **训练代价** = 不重训 T2M、不单开训练轮，搭 control bootstrap 顺风车，由 T2M replay 保护。
- **落点** = Stage-BE 之后的 control bootstrap 阶段。

**建议的下一步执行项**：
1. 实现 `compute_ease`（含 unified-stats 对齐）+ 一个仿 `tests/` 风格的单测，先在现有数据上验证标签分布合理。
2. 按第 8 节 4 处改动落 `use_ease` 开关（默认 False），跑向后兼容验证（第 10.1）。
3. 在 control bootstrap 阶段开 `use_ease=True`，同打 control+ease 标签训练。
4. 按第 10 节做 ease 有效性与 CFG 可分离性验证。

（可选备选实验：若 additive 的保真在 ease 上不足，再对照试 FiLM 专用头——有 NanoWM RT-1 数据支持，代价个位数 M 参数。）
