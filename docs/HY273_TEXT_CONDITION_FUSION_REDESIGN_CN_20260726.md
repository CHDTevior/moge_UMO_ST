# HY273 文本条件融合重设计与文本分布实验计划

> 状态：设计与实验决策材料。当前不要启动 R16 Stage B。
>
> 范围：只讨论文本编码、文本注入和文本训练分布。HY273 表示、root/body
> 双阶段预测、Kimodo 风格 overwrite control、现有 loss、采样器和任务比例在
> 第一轮实验中保持不变。

## 1. 结论

当前文本遵循问题不能只归因于一个因素。已有证据同时指向：

1. **条件融合存在需要验证的结构路径**：文本 token 会反向读取 noisy motion，且
   motion/text 共用 QKV；文本 pooled 向量又与 timestep、heading 先相加后进入同一
   AdaLN，语义路径不可独立测量和门控。代码确认这些路径存在，但尚未证明它们就是
   文本失败的因果来源。
2. **当前 encoder 输出与 denoiser 的接口不理想**：直接把 causal Qwen3-8B 的
   128 个上下文化隐藏状态交给较小 denoiser，却没有定义明确的句级 pooling
   contract；R16 删除 CLIP pooled 后，这个问题被放大。
3. **文本训练分布确实偏窄且模板化**：HumanML3D 的动作语义覆盖和措辞多样性都
   远小于 Kimodo。当前 system prompt 只改变 embedding 计算过程，并没有生成
   新 caption，不能等价于文本增强。

R16 的结论不是“CLIP 一定更好”，而是：

```text
只保留 raw Qwen token joint-attention
```

不足以替代：

```text
一个稳定的全局句义路径 + 一个不会被 noisy motion 污染的 token 路径
```

推荐按以下顺序做正交实验：

```text
Fusion 轴 -> Encoder 轴 -> Text-data 轴
```

第一轮只改 fusion。不能同时替换 encoder、增加 paraphrase、改 loss 或改任务比例。

## 2. Kimodo 是否使用 LLM 文本增强

答案是 **明确使用了**。

Kimodo 技术报告第 3 节说明：

- 使用 `Qwen3-32B` 离线改写动作描述；
- 将提示统一为总是以 `A [subject]...` 开头的结构；
- 生成不同细节层级的 paraphrase；
- 同时使用 full-clip overview caption 和 atomic-action subclip caption；
- 随机拼接两段动作，并将 stitched motion 与组合动作训练样本混入训练；
- 训练时从原始文本、LLM paraphrase、完整 clip、单个或组合 atomic subclip、
  stitched clip 中按预设分布采样。

本地依据：

```text
/tmp/kimodo_tech_report.txt:193-222
```

需要区分 Kimodo 的两个 LLM 用途：

```text
训练数据增强：
  Qwen3-32B -> 离线生成 paraphrase

模型文本条件：
  LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
  -> 单个 [B,1,4096] 句向量
```

公开仓库包含后者的 runtime/inference conditioning 代码，但没有发布 Qwen3-32B
paraphrase 的完整 prompt、增强脚本、stitched clip 对应复合文本的具体构造算法和
精确混合比例。因此我们可以复现其原则，不能声称复现了其未公开的数据配方。

Kimodo 的优势也不只来自 paraphrase。其训练集约 700 小时，覆盖大量舞蹈、格斗、
风格和同动作多演员多 take。我们的 HumanML3D K273 约 55 小时。文本改写可以提高
同一动作的词法鲁棒性，但不能创造新的动作物理分布。

## 3. 我们当前文本分布

当前统一 manifest 的实际规模约为：

```text
HumanML3D:
  caption rows       80,154
  unique captions    51,115
  motions            26,830
  median length      10 words
  常见开头            "A person" / "The person" / "A man"

MotionFix:
  edit instructions   6,730
  unique instructions 5,991
  median length       8 words
```

稀有表达示例：

```text
"break dance"     6
"breakdance"     20
"breakdancing"   12
"break dancing"  16
```

但训练集内确实存在：

```text
a person is break dancing.
a person breakdances ...
```

因此文本稀疏能解释一部分 OOD 泛化差，却不能单独解释 R16 在已有表达上从 100K
到 200K 继续退化。至少还存在融合或优化上的条件饥饿。

MotionFix 官方 loader 也支持：

```text
text-augmentations/paraphrases_dict.json
```

并在 train split 从原 instruction 与 paraphrases 中随机选一条。本地 MotionFix
数据目录目前没有该文件，所以我们现在只训练原始 instruction。

## 4. 当前 Tensor Information Flow

### 4.1 文本缓存

```text
caption/instruction
  |
  +-> Qwen3-8B hidden states
  |     [B,128,4096]
  |       -> Linear(4096,H)
  |       -> text tokens [B,128,H]
  |
  +-> CLIP ViT-L/14 pooled
        [B,1,768]
          -> MLP(768,H)
          -> pooled text [B,H]
```

这里的 Qwen system prompt 是编码模板，不会把原 caption 改写成新的训练样本。

### 4.2 R13

```text
c_dyn = TimeMLP(t) + DirectionMLP(c_dir) + CLIPPooledMLP(text)

motion tokens [B,T,H] --+
                         +-> shared-QKV bidirectional joint attention
Qwen tokens [B,128,H] ---+
                         -> double-stream blocks
                         -> concatenate again
                         -> single-stream blocks
                         -> x0 prediction
```

root denoiser 与 body denoiser各执行一套上述流程。

### 4.3 R16

R16 只做了：

```text
c_dyn = TimeMLP(t) + DirectionMLP(c_dir)
```

Qwen token joint-attention 保持不变。结果是物理质量接近 R13，但文本解析没有改善，
且 breakdance matched-noise routing advantage 在 200K 变成负值。

## 5. 当前融合的待验证假设

### 5.1 文本会读取 noisy motion

当前 `DoubleStreamBlock` 将 motion/text 拼接后送入同一个
`MultiHeadAttention`。text query 可以读取最多 300 个 noisy motion key/value。
single-stream 阶段仍然如此。

可能的后果，也是 F 轴要检验的假设：

- 文本状态不再是稳定条件，而会随 noise、timestep 和 motion sample 改变；
- noisy motion token 数量多于有效文本 token，可能支配 attention；
- CFG 的 conditional/unconditional 差分中混入 motion-dependent text state；
- root 与 body 两个 denoiser 会各自污染一次文本表示。

### 5.2 motion/text 共用 QKV 和输出投影

当前所谓 double stream 只有 modulation 和 FFN 分开，attention Q/K/V 与 output
projection 实际共享。动作 token 与语言 token 的统计分布不同，共用投影可能增加
两种模态争夺同一子空间的难度；它同时减少了参数量。必须通过投影共享与 attention
方向的 2x2 实验，才能区分结构作用和额外容量作用。

### 5.3 pooled text 与 timestep 过早融合

当前：

```text
cond = timestep + direction + pooled_text
AdaLN_params = MLP(cond)
```

这对应 NanoWM 的 `adaLN-fuse` 思路。它的问题不是数学上错误，而是：

- 无法分别测量 timestep、direction、text 对每层 shift/scale/gate 的贡献；
- timestep 是每步必需的强条件，可能压制较弱的文本向量；
- 删除 pooled 路径后，没有等价的全局句义替代。

## 6. 参考工作对比

| 工作 | 文本表示 | 注入方式 | 对我们的含义 |
|---|---|---|---|
| Kimodo | LLM2Vec 单句向量 `[B,1,4096]` | 作为 prefix token 与 time、heading、49 个 zero tokens、motion 一起进 TransformerEncoder | 证明高质量句向量和 in-context prefix 可行，但其大数据规模不可忽略 |
| HY-Motion | Qwen token + CLIP pooled | double-stream 使用独立 motion/text QKV/out；single-stream 仍共享投影；两部分都屏蔽 `text query -> motion key`；有 timestep-aware text refiner；CLIP pooled 与 time 形成 adapter | 与我们现有双流最接近，也是最低风险的结构修正来源 |
| MotionFix | CLIP token + 独立 source token block | `[time,text,source,SEP,target]` self-attention；编辑使用分解式 3-way CFG | 证明 source 高带宽并不必然导致文本失效，但需要独立 dropout/CFG |
| DiT | class vector | in-context、cross-attn、AdaLN、AdaLN-Zero 消融 | AdaLN-Zero 对单个 class vector 有效，不能直接外推到语言 token |
| NanoWM | frame-aligned action | additive、AdaLN、AdaLN-fuse、FiLM、cross-attn | 注入方式与任务结构相关；不能因 RT-1 上 FiLM 最好就直接用于 motion text |

NanoWM 的关键定量结论：

```text
RT-1 FID:
  FiLM             40.62  (best)
  additive         42.27
  adaLN-fuse       43.03
  cross-attention  51.12

PushT FID:
  additive         23.89  (best)
  FiLM             25.45
  cross-attention  28.64
```

这说明“哪种 injection 普遍最好”不是有效问题。对我们的语言条件，优先验证的
结构假设是：**motion 可以读取 text，而禁止 text 读取 noisy motion 是否更好**。

## 7. 推荐架构

### 7.1 第一优先：HY-Motion-compatible asymmetric MMDiT

在不改 root/body 分解和模型宽度的前提下：

```text
double-stream:
  motion QKV / out / FFN
  text QKV / out / FFN

single-stream:
  保留拼接序列的 shared projection

attention permission:
  motion query -> motion key  allowed
  motion query -> text key    allowed
  text query   -> text key    allowed
  text query   -> motion key  blocked
```

上述非对称 mask 必须同时覆盖 double-stream 和 single-stream。HY-Motion 的 text
refiner 会接收 timestep，但不读取 motion；是否增加 refiner 要作为独立实验，不能
与 asymmetric mask 或 QKV 拆分捆绑。

这是比直接切换到全新 backbone 更稳妥的第一版，因为：

- 保留当前 frame-token root/body 主干；
- 保留 token-level 细粒度语义；
- 直接移除已确认存在、但尚未证明有害的 text-to-noisy-motion 路径；
- 与 HY-Motion 的实现证据一致；
- 可以通过 `attention direction x projection sharing` 的 2x2 实验定位作用。

### 7.2 推荐的最终信息流

```text
raw text
  +-> global sentence encoder -> g_text [B,H]
  +-> token encoder -> U_text [B,L,H]
                         |
                         +-> optional timestep-aware text refiner
                                      |
                                      +---------------------------+
                                      |                           |
timestep + direction -> c_dyn        |                           |
                    |                 v                           v
imputed HY273+mask -> root asymmetric MMDiT -> root x0   original body state
                                                |                 |
                                                v                 |
                               local-root transform / detach       |
                                                |                 |
                                                +-> body asymmetric MMDiT
                                                            |
                                                            v
                                                         body x0
```

`c_dyn` 只包含 timestep 与 direction。全局 text 不再和 timestep 在输入处直接
相加。`独立 text modulation` 与 `global prefix token` 是两个不同实验，分别记为
F3a 和 F3b，不能在同一个 run 中任选其一。

若 text memory 包含：

```text
[LLM2Vec global token, token-level text sequence]
```

则 gated cross-attention 可以产生 token-selective alignment。反之，若 memory
只有一个 LLM2Vec token，标准 cross-attention 中：

```text
softmax(q_i k^T) = 1
```

条件增量退化为与 query 无关的 global broadcast。它不是完全无效，residual 后的
motion tokens 仍然不同，但这条路径不能提供 token-selective alignment。单 token
条件应使用带非对称 mask 的 prefix/joint attention、FiLM/AdaLN，或与多 token
memory 组合。

### 7.3 更明确但改动更大的备选

如果 asymmetric MMDiT 仍然无法形成稳定 text routing，再切到：

```text
motion self-attention
-> gated motion-query/text-KV cross-attention
-> motion FFN
```

text memory 在所有 motion block 中保持静态，完全取消 text query。这是最明确的
条件路径，但相对当前 checkpoint 结构改动更大，因此放在第二顺位。

## 8. Encoder 实验轴

只有选出 fusion 后，才允许比较 encoder。主矩阵必须是完整 2x2：

```text
                         global encoder
                   CLIP-L pooled   LLM2Vec
token Qwen3              E00          E01
encoder CLIP-L           E10          E11
```

必要的路径控制：

```text
E-Q-only:  Qwen token only，在最佳 fusion 下重测 R16 假设
E-L-only:  LLM2Vec global only，确认 global broadcast 单独能做多少
E-Q-pool:  Qwen token + Qwen masked-mean 或 attention pooling
```

优先测试 LLM2Vec 的理由：

- Kimodo 直接采用并报告早期实验优于 CLIP/T5；
- 本地已有完整 supervised LLM2Vec 权重；
- 小型语义测试中，breakdance paraphrase 和 faster/more quickly 的句向量相似度
  明显比当前 Qwen masked-mean 更合理；
- 可离线缓存，不增加 DDP 训练显存。

不能只用 LLM2Vec cosine 判断最终优劣。它只能说明句向量几何更合适，最终仍需
matched-noise 生成路由和动作物理量验证。

Qwen 与 CLIP 的 token 长度也会混杂结果。严格 2x2 先统一为 77 个有效 token，
随后再单独做 `Qwen@77 vs Qwen@128` length sweep。所有 bridge 使用相同 hidden
dimension、normalization 约定，并报告新增参数量。

Fusion 与 encoder 不能只做贪心选择。最终至少补一个小型交叉确认：

```text
F00/Fbest x E00/Ebest
```

## 9. Text-data 实验轴

### 9.1 HumanML3D absolute caption

当前 loader 已经是：

```text
sample motion row
-> sample caption/span
-> materialize that caption-specific motion
```

因此不重写 sampler，只在 caption 选定后新增 stateless `variant draw`。保持完全相同的
motion row、caption index、crop、yaw 和 text-drop trace。

HumanML3D 数据实验拆成：

```text
D-H0 original only

D-HC original vs canonical
     canonical 统一为 "A person ..."，不增加事实

D-HP original vs lexical paraphrase
     改变动词/句式，保留动作、顺序、方向、次数、速度和身体部位

D-HS optional concise
     只在 D-HC/D-HP 之后独立测试
```

不能把每条 paraphrase 展开为新的 motion row，否则 caption 多的 motion 会被
无意过采样。

segment caption 的 paraphrase 必须继承同一个 frame span，不能重新配给完整 motion。
Stage A 只评估这一组 absolute-caption 数据实验。

### 9.2 MotionFix relative instruction

MotionFix 必须使用独立的 relative-edit prompt。不能把 instruction 标准化成 target
的绝对 caption。增强时必须保持：

```text
source -> target 的方向性
left/right
faster/slower
earlier/later
more/less
次数
身体部位
否定
```

训练 pair 的 source/target、paired crop 和 paired yaw augmentation 不变，只替换
同义 instruction。这一轴只在 source-present Stage C editing 中评估：

```text
D-M0 original instruction only
D-MC original vs canonicalized relative instruction
D-MP original vs lexical relative paraphrase
```

其中 `faster/slower` 必须给定同一个 source motion 才是语义完备的评估，不能在
Stage A 的无 source T2M 中据此判断 editing。

### 9.3 实施模型

Kimodo 使用的是 Qwen3-32B。当前标准缓存目录未发现其权重，但已有：

```text
/mnt/afs/HY-Motion-1.0/ckpts/Qwen3-8B
```

第一轮数据实验可以用 Qwen3-8B 生成候选，再用小规模人工检查和规则检查筛除
方向、速度、次数、否定被改写的样本。这里的目标是科研对照，不把复杂防篡改机制
引入训练主线。

建议每个原文先生成 canonical 和 lexical 各 1 个有效变体，但训练时把两类放在
不同实验臂。更多 paraphrase 不会增加动作多样性，反而会放大同一 motion 的重复
监督。

## 10. 正交实验矩阵

### 10.1 Fusion 轴：固定现有 HYText、loss 和数据

```text
                               double-stream projection
attention direction       shared QKV/out       separate QKV/out
bidirectional                  F00                  F01
asymmetric                     F10                  F11
```

定义：

```text
F00 = R13 current
F10 = 只增加 double+single 的 text-query -> motion-key mask
F01 = 保留 bidirectional，只拆 double-stream motion/text QKV/out
F11 = HY-Motion double-stream-compatible 组合
```

分离投影从共享权重复制初始化，并增加 tied-weight 数值等价测试。F01/F11 会增加
约 25.2M 参数；若不做参数匹配，结论只能写成
`modality-specific capacity bundle` 有效，不能直接声称“共享 QKV 是真因”。

附加实验必须在上述 2x2 之后独立进行：

```text
F11b fully modality-specific
     single-stream 也拆 projection；这不是 HY-Motion faithful 复现

F-R0  no text refiner
F-R1  timestep-aware text refiner

F3a   separate global-text modulation
F3b   global prefix token

F4    static text memory + explicit gated cross-attention
      作为 fallback backbone，不用于估计 F00-F11 的单因素效应
```

### 10.2 Encoder 轴：固定最佳 fusion

执行第 8 节的完整 2x2、token-only/global-only 控制和 token-length sweep。

### 10.3 Data 轴：固定最佳 fusion + encoder

```text
Stage A: D-H0 / D-HC / D-HP
Stage C: D-M0 / D-MC / D-MP
```

不把 encoder、fusion、data 三轴合并成一个 run，否则成功或失败都无法解释。

## 11. 训练与门禁

Fusion 2x2 的第一轮只做 Stage A T2M：

```text
0 -> 200K
每个 run: DDP4 x batch/rank 32
每波并行两个 run，共两波完成 F00/F10/F01/F11
global batch=128、LR、EMA、t schedule、x0 parameterization、loss 全部沿用 R13
```

四组必须全部从头按这个新 DDP4 协议训练，使用相同 seed、全局 sample plan、
noise、增强和 rank-local 切分；新训练的 F00 是 2x2 唯一因果基线。历史
R13-200K 使用 DDP8 x batch/rank 16。虽然 global batch 同为 128，但现有
backward 是 rank-local ratio-of-sums 再做 DDP 等权平均，因此 DDP4 与历史 DDP8
并不严格优化同一个目标。历史 R13 只能作为背景物理质量参考，不能替代本轮 F00
或用于宣称单因素提升。

正式启动器固定：

```text
config = configs/hy273_multitask_r13_stage_a_t2m.yaml
text encoder = Qwen3 token cache + CLIP-L pooled cache
text global conditioning = pooled_adaln
world size = 4
batch/rank = 32
```

检查点与评估：

```text
50K / 100K / 150K / 200K
```

R16 曾在 100K 短暂变好、到 200K 退化，所以不能用 50K 或 100K 单点决定最终方案。
筛选 run 先用一个训练 seed；进入最终候选比较的方案至少跑 3 个训练 seed。

Stage A loss 保持现有合同：

```text
x0 prediction parameterization
+ existing representation group-balanced velocity-space MSE
+ contact loss
+ clean root/joint velocity
+ FK consistency
+ foot lock
```

第一轮不加新的 text contrastive loss，不改 semantic channel weights。先确认条件信息
是否能稳定到达 motion token；否则改 loss 只会混淆真因。

Stage A 不能证明统一能力保留。Fusion/encoder 各自的前两名要做同协议短 transfer
gate：

```text
short Stage B:
  T2M 10%
  Kimodo control 90%
  Editing 0%
  检查 root/path/endpoint/full-pose/contact

short Stage C:
  T2M + control + editing
  检查 source-shuffle、instruction-shuffle、source-only 和 3-way CFG
```

通过 transfer gate 后，最终 winner 才进入完整 Stage B/Stage C。验收时显式核对
root/body bridge detach、overwrite mask、source/task/dropout 分支不变；接口能运行
不等于能力已保留。

不要启动 R16 Stage B。R16-200K 只作为负对照和物理质量基线。

### 11.1 Null condition 与 CFG

Fusion 2x2 固定沿用现有 empty-cache-row 合同。进入 encoder 比较前，统一采用一个
encoder-independent null contract：

```text
force_drop_text:
  projected token values = 0
  projected global value = 0
  保留一个有效的 zero sentinel token
```

所有 encoder 固定相同 text/source/task dropout 概率。评估同时报告公共
`text_cfg=2.0` 和小范围 CFG curve，避免把 embedding 范数差异误判为语义能力差异。

## 12. 评估

### 12.1 文本语义

固定 noise、长度、heading、采样步数和 CFG，比较：

```text
canonical
paraphrase
close negative
empty
```

重点 panel：

```text
Stage A / absolute T2M:
break dancing / breakdances / breaking dance
vs ballroom dancing / walking / empty

walk quickly / walk rapidly / walk at a fast pace
vs walk slowly / stand still / empty

Stage C / source-present editing:
固定同一个 source motion
move the feet faster / increase the stepping pace
vs move the feet slower / leave the speed unchanged / empty
```

对每个动作物理量或语义 judge `m`，matched-noise routing diagnostic 定义为：

```text
A_m(p) = d_m(y_p, y_empty) - d_m(y_p, y_canonical)
```

其中 `y_p` 是 prompt `p` 的生成结果，`y_empty` 是同 noise 的空文本结果，
`y_canonical` 是 canonical prompt 的同 noise 结果，`d_m` 是在动作物理量或冻结
语义 judge 上的距离。`A_m(p) > 0` 只说明输出在该测量下更接近 canonical，而不等于
动作语义本身正确；必须与可视化和动作特定物理量共同解释。

主判据不是 embedding cosine，而是：

- correct prompt 相对 close negative/empty 的 matched-noise motion routing advantage；
- breakdance 等动作的可视化与动作特定物理量；
- paraphrase 间生成一致性；
- 训练表达和 OOD 表达都不能在 100K 到 200K 反向退化。

所有模型先在公共 `text_cfg=2.0` 下比较，再报告
`text_cfg in {1.0, 1.5, 2.0, 3.0}` 的小范围曲线。不能为每个模型单独挑一个最好看的
CFG 后再横向排名。

### 12.2 保留能力

- 现有 Kimodo 类 root/path/endpoint/full-pose/contact benchmark；
- T2M 物理质量与 R13-200K 对照；
- contact、foot skate、root/body continuity；
- 后续 Stage C 再做 pure edit 可视化和 MotionFix 物理量。

当前不引入其他表示的 FID/Top3。

### 12.3 统计单位

筛选实验可以先使用一个训练 seed，但不得把同一 motion 的多条 caption 或同一
MotionFix pair 的 paraphrase 当成独立样本扩大显著性。置信区间按以下单位做
hierarchical bootstrap：

```text
HumanML3D: motion -> caption/paraphrase -> matched-noise seeds
MotionFix: source-target pair -> instruction/paraphrase -> matched-noise seeds
```

进入最终候选的方案至少训练 3 个独立 seed，并同时报告：

```text
跨训练 seed 的均值与离散度
按 motion/pair bootstrap 的置信区间
每个 checkpoint 的 CFG curve
```

## 13. 对未来统一任务的兼容性

推荐接口仍保持任务解耦：

```text
text memory:
  绝对动作描述或相对编辑指令

task/source context:
  区分 GENERATE / EDIT / REACTION / CONTINUATION

observed_motion + mask:
  Kimodo overwrite control
```

因此：

- T2M：无 source，绝对文本；
- control：无 source 或 target source，控制仍走 overwrite；
- editing：source-present + relative instruction；
- edit+control：source-present + relative instruction + target-side control；
- reaction：新增角色化 source slot 与 task id，不需要重新改变文本融合主干。

文本融合不负责区分任务；`task_id/source_role/target_op` 负责。这样增加新任务时不会
把 control 语义塞进文本路径，也不会破坏现有 overwrite control。

## 14. 当前决策

1. R16 Stage B 保持停止。
2. 按新 DDP4 协议从头并行训练 `F00/F10`，单独测 asymmetric mask；不能用历史
   DDP8 R13 代替新 F00。
3. 第二波并行训练 `F01/F11`，检查 double-stream modality-specific projection
   及其与 asymmetric attention 的交互。
4. `F01/F11` 从共享投影复制同值但非共享的初始化；由于训练时增加约 25.2M
   独立参数，仍只能解释为 modality-specific capacity bundle。
5. 固定 fusion 后执行 encoder 2x2、token/global-only、Qwen pooling 和 token-length
   控制，不把 encoder 更换与 fusion 更换放进同一个 run。
6. Fusion/encoder 各自前两名先做短 Stage B control transfer 和短 Stage C edit
   transfer；Stage A 本身不能证明统一能力保留。
7. 只有 encoder 选定后才生成 LLM paraphrase，并分别做 HumanML3D absolute-caption
   与 MotionFix relative-instruction 数据实验。
8. 每一轴都在 50K/100K/150K/200K 做同协议评估；最终候选至少训练 3 个 seed。

这套顺序能回答三个不同问题：

```text
F: 文本是否因融合方式而失效？
E: 哪个预训练表示最适合我们的动作语义？
D: 在架构通路可用后，文本分布扩充能增加多少 OOD 鲁棒性？
```
