# Job_For_AIInfra_and_Inference_and_PostTraining

## 你的转型目标（3 个月）
你倾向于 **模型后训练 / 推理加速（Inference Acceleration）** 岗位。结合你已有的 **C++ 性能/系统架构经验** 与 **KV Cache 方向论文（ACL Findings / ECCV）**，这 3 个月的目标是：

1. 用一个“主项目”把 **推理加速能力落到可复现实验**（可压测、可分析瓶颈、可量化收益）。
2. 用一个“小闭环”把 **后训练（SFT + DPO/GRPO 选其一）** 做到工程可跑，并把训练影响映射到推理侧指标（性能/稳定性/一致性）。
3. 准备一套面试可复述的“证据链”：问题 → 定位 → 方案 → 指标 → trade-off → 回归与风险控制。

---

## 建议的学习/实践路线总览

### 共同底座（前 1/3 期间完成 70%）
- Transformer 推理侧关键机制：Prefill / Decode、Attention 计算与 KV Cache 的角色
- KV Cache 体系化理解：显存占用、带宽瓶颈、cache 管理（分页/块/回收/复用）、长上下文场景的显存曲线
- 推理服务工程化：batching、队列与调度、可观测性（延迟分位、吞吐、显存、GPU 利用率）

### 主线：推理加速（第 3–8 周）
- 以 **vLLM**（优先）或 **TensorRT-LLM**（备选）为 baseline
- 选 2–3 个“可落地优化点”，每个点都有明确变量与对照实验
- 输出：压测工具、图表与结论、失败场景与回归开关

### 支线：后训练闭环（第 7–12 周）
- 跑通 SFT +（DPO 或 GRPO 二选一）
- 做推理回归：相同推理服务/工作负载下对比性能与输出质量
- 输出：训练日志/吞吐、训练稳定性要点、对齐到推理侧指标的分析

---

## 技术栈学习清单（按“必须/建议/加分”）

### 必须（面试与项目都会用到）
- `Python` / `PyTorch`（数据管线、训练/推理脚本理解）
- `CUDA` 性能基础：memory hierarchy、kernel launch、带宽与延迟直觉
- `Nsight Systems` / `Nsight Compute`（时间线与热点定位）
- 推理服务关键指标与分解：
  - `TTFT`（首 token 延迟）
  - `TPOT`（逐 token 时间）
  - `吞吐`（tokens/s）
  - `P95/P99` tail latency
  - `显存峰值` 与 `显存随并发/上下文长度变化曲线`
- KV Cache 相关概念（你已有论文优势，这里做“工程化复述”）：
  - paged attention / block manager / eviction
  - prefill vs decode 的资源差异
  - 长上下文对显存/带宽的影响

### 建议（提升你落到“岗位能力”的程度）
- vLLM 核心机制（或 TRT-LLM 对应机制）：
  - continuous batching / scheduler
  - KV cache block 分配策略
  - 中间状态与内存复用
- 并行推理（只要会讲清 trade-off 就足够）：
  - tensor parallel 在推理侧的通信开销直觉
- 量化/精度工程：
  - FP16/FP8/INT8/4bit 的收益边界与风险点（特别是长上下文/尾延迟）

### 加分（你 C++ 与系统能力的“展示位”）
- 为推理服务做 micro-benchmark / profiling 自动化
- 把“优化点”落到代码结构层面的可解释性（模块边界、开关、回归策略）

---

## 算法理论基础（深入清单）

> 岗位 JD 常列的基础项：**Attention、RoPE/ALiBi、PPO/DPO/GRPO**。以下按「推理加速 / 后训练」方向补充**额外需要关注**的理论，并按优先级分层。你已有 KV Cache 论文背景，标有 ⭐ 的项建议结合论文做「工程化复述」。

### 一、Transformer 架构（JD 基础之外的延伸）

#### 已列基础（快速过一遍，确保能讲清）
| 主题 | 需要掌握到什么程度 |
|------|-------------------|
| Attention 机制 | 能手推 Q/K/V 计算；理解 scaled dot-product、因果 mask |
| RoPE | 旋转位置编码原理；相对位置如何编码；外推（extrapolation）问题 |
| ALiBi | 线性 bias 替代位置编码；长上下文外推优势 |

#### 额外关注（推理加速岗高频）
| 主题 | 为什么重要 | 掌握深度 |
|------|-----------|---------|
| **Multi-Head / GQA / MQA** ⭐ | GQA/MQA 直接决定 KV Cache 大小与带宽压力，是你论文方向的工程落点 | 能算 KV 显存公式；能解释 GQA 为何在质量-效率间折中 |
| **Flash Attention** | 现代推理/训练默认算子；理解 IO-aware 优化思路 | 知道 tiling + online softmax 直觉；prefill 阶段收益来源 |
| **Prefill vs Decode 计算特性** ⭐ | TTFT vs TPOT 瓶颈分解的核心 | prefill 是 compute-bound；decode 常是 memory-bandwidth-bound |
| **KV Cache 显存建模** ⭐ | 推理容量规划、优化收益估算的基础 | 能手算：`2 × layers × heads × head_dim × seq_len × bytes`（注意 GQA 的 head 数） |
| **LayerNorm / RMSNorm** | 推理融合、量化敏感点 | 知道 Pre-LN vs Post-LN；RMSNorm 为何更常见于 LLaMA 系 |
| **FFN / SwiGLU** | 推理 FLOPs 与显存占比高 | 知道 SwiGLU 结构；gate/up/down 三矩阵；与 Attention 的算力占比 |
| **Causal LM vs Encoder-Decoder** | 影响 KV Cache 结构与 serving 策略 | 知道 GPT 系（decoder-only）与 T5 系差异；推理侧主流是 decoder-only |

#### 进阶（MoE / 多模态岗会问到）
| 主题 | 为什么重要 | 掌握深度 |
|------|-----------|---------|
| **MoE（Mixture of Experts）** | 蚂蚁等 JD 明确提到 MoE co-design | 门控机制、Top-K routing、负载均衡 loss、专家并行通信 |
| **多模态融合架构** | 视觉 token 与文本 token 的 KV 管理差异 | 了解 ViT + LLM 拼接方式；多模态 KV 更长、调度更复杂 |

---

### 二、推理加速专项理论

| 主题 | 为什么重要 | 掌握深度 |
|------|-----------|---------|
| **PagedAttention / Block Manager** ⭐ | vLLM 核心；与 KV Cache 论文直接相关 | 非连续物理内存、block 分配/回收、碎片问题 |
| **Continuous Batching** | 吞吐提升的关键调度策略 | 动态进出 batch；prefill/decode 混合调度 |
| **Speculative Decoding** | 降低 TPOT 的主流方案之一 | draft model + verify；接受率与加速比关系 |
| **量化理论（PTQ / QAT / FP8 / INT8 / 4bit）** | 推理成本与部署必备 | 知道各精度收益边界；KV Cache 量化单独考虑 |
| **Tensor Parallel（推理侧）** | 大模型单卡放不下的标配 | 通信开销估算；何时 TP 收益被通信吃掉 |
| **Pipeline Parallel** | 训练侧更常见，推理侧了解即可 | stage 划分；bubble 问题 |
| **Roofline Model** ⭐ | 性能分析方法论，C++ 背景友好 | 判断算力-bound vs 带宽-bound；指导优化方向 |
| **Prefix Caching / KV Reuse** ⭐ | RAG/多轮对话场景的热点优化 | 相同前缀共享 KV；命中率与收益估算 |

---

### 三、后训练与对齐理论（JD 基础之外的延伸）

#### 已列基础（快速过一遍）
| 主题 | 需要掌握到什么程度 |
|------|-------------------|
| PPO | 策略梯度 + clip；RLHF 中如何用 reward model |
| DPO | 直接偏好优化；无需显式 reward model |
| GRPO | Group Relative Policy Optimization；DeepSeek 等近期热点 |

#### 额外关注（后训练岗高频）
| 主题 | 为什么重要 | 掌握深度 |
|------|-----------|---------|
| **SFT（Supervised Fine-Tuning）** | 后训练第一步，必会 | 数据格式、loss 计算、过拟合与灾难性遗忘 |
| **RLHF 完整流程** | 理解 PPO/DPO/GRPO 在流程中的位置 | SFT → Reward Model → PPO；DPO 如何绕过 RM |
| **Reward Model 训练** | PPO/RLHF 链路核心 | Bradley-Terry 模型；偏好数据构造；RM 过拟合风险 |
| **KL 散度约束 / Reference Model** | 防止对齐后模型偏离太远 | β 参数作用；DPO 中的 implicit KL |
| **偏好数据质量** | 工程落地最大坑之一 | 噪声、一致性、覆盖度；如何过滤与去重 |
| **LoRA / QLoRA / PEFT** | 后训练工程标配 | rank、alpha、target modules；QLoRA 的 NF4 量化 |
| **在线 RL vs 离线 RL** | 理解 GRPO/PPO 的采样差异 | 在线需要 rollout；离线用静态偏好数据 |
| **Constitutional AI / RLAIF** | 无人工标注的对齐思路 | AI 生成偏好；自我批评与修正 |
| **评估体系** | 后训练闭环必备 | perplexity、win-rate、GPT-4 judge、人工评估偏差 |

#### 算法对比速查（面试常问）
```
SFT     → 学格式/任务，不保证偏好
PPO     → 需要 RM + rollout，灵活但工程重
DPO     → 直接用偏好数据，无需 RM，稳定
GRPO    → 组内相对优势，适合多候选采样场景
```

---

### 四、分布式训练理论（推理加速岗也会问）

| 主题 | 为什么重要 | 掌握深度 |
|------|-----------|---------|
| **数据并行（DP）** | 最基础 | all-reduce 梯度；通信量与 batch size 关系 |
| **张量并行（TP）** | Megatron 核心 | 列/行切分；通信次数与 layer 关系 |
| **流水并行（PP）** | 大模型训练标配 | micro-batch；bubble 比例 |
| **专家并行（EP）** | MoE 必备 | all-to-all 通信；负载均衡 |
| **ZeRO / DeepSpeed** | 显存优化 | stage 1/2/3 区别；offload |
| **混合并行策略选择** | 系统设计能力 | 给定模型规模 + 集群，如何选 DP/TP/PP/EP |

---

### 五、学习优先级建议（结合你的背景）

#### 第 1–2 周（推理加速主线，与你论文最相关）
1. GQA/MQA 与 KV Cache 显存建模 ⭐
2. Prefill vs Decode 瓶颈分解 ⭐
3. PagedAttention / Block Manager ⭐
4. Roofline Model ⭐
5. Flash Attention 基本原理

#### 第 3–4 周（推理系统机制）
1. Continuous Batching / Scheduler
2. 量化理论（FP8/INT8/4bit）
3. Speculative Decoding
4. Prefix Caching / KV Reuse ⭐

#### 第 5–6 周（后训练闭环）
1. SFT 基础 + LoRA/QLoRA
2. RLHF 完整流程
3. DPO 或 GRPO（二选一做深）
4. 偏好数据质量与评估体系

#### 第 7–8 周（分布式与 MoE，面试加分）
1. TP/PP/EP 基本原理
2. MoE 结构与负载均衡
3. Megatron 并行策略选择

---

### 六、推荐学习资源（按主题）

| 主题 | 资源 |
|------|------|
| Attention / Transformer | Jay Alammar 博客；《Attention Is All You Need》 |
| KV Cache / PagedAttention | vLLM 论文 + 官方文档；你的 ACL/ECCV 论文 |
| Flash Attention | FlashAttention-1/2 论文 |
| RoPE / 位置编码 | RoFormer 论文；LLaMA 技术报告 |
| GQA | GQA 论文（Ainslie et al.） |
| 量化 | LLM.int8()、GPTQ、AWQ 论文 |
| DPO | DPO 原论文（Rafailov et al.） |
| GRPO | DeepSeekMath 技术报告 |
| MoE | Switch Transformer、Mixtral 技术报告 |
| 分布式 | Megatron-LM 论文；DeepSpeed 文档 |
| 推理系统 | vLLM 文档；TensorRT-LLM 文档 |

---

## 项目实操规划（必须有硬产出）

### 主项目（推理加速）：KV Cache 管理与调度优化
**目标**：基于 baseline（vLLM 或 TRT-LLM），实现并评估至少 2 个优化点，形成可复现报告。

#### 1) 基线与工作负载选择
- 选择固定模型（至少 1 个长度维度覆盖短/长上下文）
- 设计 workload 分布（建议至少包含）：
  - 短 prompt + 较短生成
  - 长 prompt（测试 prefill 压力与 TTFT）
  - 长上下文多轮（测试 decode + KV cache 管理）
  - 混合并发（1/4/16/32 并发档位）

#### 2) 优化点方向（建议选 2–3 个做深）
你可以结合既有 KV Cache 论文经验，优先挑“工程实现成本可控”的点：
- `cache block / page` 管理策略
  - 回收/复用更激进 vs 更保守的显存-吞吐权衡
  - eviction 触发条件与碎片问题
- `scheduler` 调度策略（prefill 与 decode 的资源分配）
  - 如何在 TTFT 与吞吐之间取舍
  - 对不同上下文长度分布的自适应策略
- `memory` 与 `kernel` 层面的优化
  - 通过 profiling 找到热点并解释瓶颈归因（算子 vs 带宽 vs 通信 vs 调度）

#### 3) 验收标准（写进 README 或报告）
- 必须有对照实验：baseline vs your variant
- 必须给出数字：至少包含 TTFT、吞吐、P95/P99、显存峰值中的 3 项
- 必须解释 trade-off：
  - 哪些场景更好，哪些场景更差
  - 失败场景如何回退（feature flag / config 开关）

#### 4) 交付物
- 压测/评估脚本与配置（可一键跑）
- 实验报告（建议 3–8 页 PDF 或仓库 README）
- 图表与结论（用图说明“因变量”与“趋势”）

---

### 支线项目（后训练闭环）：SFT +（DPO 或 GRPO）
**目标**：完成一个可复现的后训练流程，并用推理回归验证训练影响。

#### 最小可跑闭环（务实版）
1. 数据准备
   - 选一个公开指令数据（SFT）
   - 选一个偏好/对比数据（DPO）或轨迹偏好相关（GRPO）
2. 训练
   - `LoRA` 先跑通（降低门槛）
   - 记录训练吞吐、loss 曲线、稳定性信息（是否崩、是否震荡）
3. 推理回归
   - 使用同一套推理服务/工作负载
   - 指标对比：
     - 质量（人工评估 or 自动评估，至少有定性结论）
     - 性能侧：延迟/吞吐（训练后的输出长度/停止行为变化也要考虑）

#### 选择 DPO 还是 GRPO？
- 选择 `DPO`：更适合用偏好数据直接对齐（工程闭环快）
- 选择 `GRPO`：更贴 RL/轨迹训练，但工程复杂度更高

---

## 面试准备规划（建议按“主题 → 答题模板 → 复述素材”）

### 你需要准备的高频问题（推理加速方向）
- KV Cache 为什么会成为瓶颈？瓶颈来源如何分解（显存/带宽/调度/碎片）
- Prefill vs Decode 在性能上的差异（TTFT/TPOT 的来源）
- 如何设计压测来证明你的优化有效（控制变量、分布、并发档位）
- 如何用 profiling 定位瓶颈：从时间线如何推断是算子还是通信还是调度
- vLLM/TRT-LLM 的关键机制如何工作（只要讲清原理与收益即可）

### 后训练方向的高频问题（后训练/对齐基础）
- SFT、PPO、DPO、GRPO 的目标分别在优化什么、适用场景是什么
- 为什么偏好数据噪声会影响训练？怎么做过滤/去噪/鲁棒性
- 离线评估与线上评估的差异（质量、偏差、安全与一致性）

### 你的“证据链”组织方式（强烈建议做成 3 个 STAR）
1. `推理加速 STAR`：你如何从指标异常定位到某个模块，再验证改动收益
2. `KV Cache 工程化 STAR`：你的论文思路如何落到 cache 管理或调度实现
3. `系统架构 STAR`：如何做观测、回归开关、灰度与风险控制（迁移自自动驾驶 SLA 思维）

---

## 12 周逐周计划（每周 10–20h）

### 第 1 周：环境与基线打通（产出：可复现 baseline 压测）
- 搭建推理压测环境（vLLM 或 TRT-LLM）
- 选定模型 + workload 分布 + 指标采集方式
- 产出：baseline 压测脚本 + 一份“初始指标报告”（截图/图即可）

### 第 2 周：KV Cache 机制系统化（产出：你自己的“工程化讲稿”）
- 梳理 prefill/decode 的资源差异
- 梳理 cache block 分配、回收、复用的直觉与可观测指标
- 产出：面试讲稿（1–2 页文字）+ 一张“指标-模块映射图”

### 第 3 周：选 1 个优化点并做对照实验（产出：首个改动有效性）
- 选择 cache 管理或 scheduler 的一个变量
- 完成 baseline vs variant 对照
- 产出：图表 + 结果归因（为什么有效/为什么无效）

### 第 4 周：第 2 个优化点（产出：形成 2 点对照）
- 在另一个方向做改动（scheduler/eviction/memory/parallel 其一）
- 产出：清晰的 trade-off 总结（哪些 workload 更优）

### 第 5 周：长上下文与尾延迟专项（产出：P95/P99 与显存曲线）
- 针对长 prompt/长上下文多轮
- 重点分析尾延迟与显存峰值变化
- 产出：长上下文实验报告（至少 3 张图）

### 第 6 周：自动化与可复现（产出：一键跑实验）
- 把压测流程脚本化（固定随机种子/固定配置）
- 输出统一格式的结果文件（方便后续画图与对比）
- 产出：`run_bench.sh` + 结果目录规范

### 第 7 周：开始后训练最小闭环（产出：SFT 跑通）
- 跑通 SFT（LoRA）
- 产出：训练日志摘要 + 推理回归的初始对比

### 第 8 周：完成后训练（DPO/GRPO 选一并跑通）+ 回归（产出：质量/性能证据）
- 完成 DPO 或 GRPO 训练
- 推理回归：同一套 workload 对比
- 产出：训练后对输出长度/停止行为的影响说明

### 第 9 周：整合主项目报告（产出：主项目终稿）
- 汇总所有对照实验
- 强化“归因解释”与“失败场景/回退策略”
- 产出：主项目报告终稿（可直接用于面试）

### 第 10 周：面试题库集中训练（产出：可复述版本答案）
- 推理加速方向：把你的报告转成问答形式
- 后训练方向：梳理 DPO/GRPO/PPO 的关键对比点
- 产出：每个主题至少准备 2 个“可落地例子”

### 第 11 周：模拟面试（产出：反复打磨的讲述）
- 进行 2–3 轮模拟（可找同事/面试伙伴/录屏自查）
- 产出：你自己的“1 分钟/5 分钟/15 分钟三档版本讲稿”

### 第 12 周：简历与作品集冲刺（产出：投递材料齐全）
- 简历首屏关键词优化（Inference Acceleration / KV Cache / Profiling / Scheduler）
- 作品集链接（仓库 + 报告 PDF + 图表索引）
- 产出：一份可投递的简历 + 一份作品集索引页

---

## 实验记录模板（强烈建议用同一格式写）
每做一次改动，至少记录：
- baseline 配置：模型、精度、并发、上下文分布
- 变量：你改了什么（cache block / scheduler / precision / memory）
- 指标：TTFT、TPOT、吞吐、P95/P99、显存峰值
- 归因：profiling 显示瓶颈在哪个阶段（prefill/decode/通信/调度）
- 结果：收益在哪些 workload，损失在哪些 workload
- 回退方案：如何关闭改动，避免线上风险
