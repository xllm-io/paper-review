---
title: "LongCat-Flash Technical Report"
description: "LongCat-Flash 技术报告，探索大语言模型高效推理与训练加速。"
date: "2025-10-01"
tags:
  - Deep Learning
  - Paper Summary
  - Attention
  - Sparse
  - Inference
  - Distributed
  - MoE
  - KV Cache
  - Compression
  - Decoding
draft: false
---

# LongCat-Flash Technical Report

## 论文图表

| 图表 | 文件 | 说明 |
|------|------|------|
| Figure 1 | [fig1.png](figures/longcat-flash/fig1.png) | LongCat-Flash 基准性能概览 |
| Figure 2 | [fig2.png](figures/longcat-flash/fig2.png) | LongCat-Flash 架构：ScMoE + Zero-computation Experts |
| Figure 3 | [fig3.png](figures/longcat-flash/fig3.png) | Zero-computation Experts 验证损失曲线 |
| Figure 3b | [fig4.png](figures/longcat-flash/fig4.png) | 训练过程中平均激活专家数 |
| Figure 3c | [fig5.png](figures/longcat-flash/fig5.png) | 激活专家数标准差 |
| Figure 4 | [fig6.png](figures/longcat-flash/fig6.png) | ScMoE vs 基线训练损失比较 |
| Figure 5 | [fig9.png](figures/longcat-flash/fig9.png) | MLA Scale-Correction 效果 |
| Figure 5b | [fig10.png](figures/longcat-flash/fig10.png) | Model Growth 初始化效果 |
| Figure 6 | [fig11.png](figures/longcat-flash/fig11.png) | Hidden z-loss 对隐藏状态的影响 |
| Figure 7 | [fig12.png](figures/longcat-flash/fig12.png) | 梯度 RMS 与 Epsilon 对损失的影响 |
| Figure 8 | [fig13.png](figures/longcat-flash/fig13.png) | 计算-通信重叠效率比较 |
| Figure 9 | [fig14.png](figures/longcat-flash/fig14.png) | SBO 重叠策略概览 |
| Figure 10 | [fig15.png](figures/longcat-flash/fig15.png) | 多步重叠调度器 |
| Figure 11 | [fig16.png](figures/longcat-flash/fig16.png) | 平均激活 FFN 专家数 |

---

## 一、论文概述

| 项目 | 内容 |
|------|------|
| **标题** | LongCat-Flash Technical Report |
| **作者** | Meituan LongCat Team |
| **机构** | 美团 |
| **论文** | https://arxiv.org/abs/2509.01322 |
| **代码** | https://github.com/meituan-longcat |
| **在线** | https://longcat.ai |
| **发布** | 2025年9月 |

## 二、核心创新

### 2.1 模型规模与效率

**LongCat-Flash** 是一个 **5600 亿参数**的 Mixture-of-Experts (MoE) 语言模型，专为**计算效率**和**高级代理能力**设计。

| 指标 | 数值 |
|------|------|
| **总参数量** | 560B |
| **每 token 激活参数** | 18.6B - 31.3B（平均 27B） |
| **训练数据** | >20T tokens |
| **训练时间** | 30 天 |
| **推理速度** | >100 TPS（H800） |
| **推理成本** | $0.70 / 百万输出 token |
| **上下文长度** | 128K |

### 2.2 两大架构创新

#### 创新一：Zero-computation Experts（零计算专家）

**核心思想**：不是所有 token 都需要相同计算量。

```
┌─────────────────────────────────────────────────────────┐
│                Zero-computation Experts                  │
├─────────────────────────────────────────────────────────┤
│  • 512 个 FFN 专家 + 256 个零计算专家                      │
│  • 每个 token 激活 12 个专家（来自两种类型）                 │
│  • 零计算专家直接返回输入，不消耗计算资源                     │
│  • 根据上下文重要性动态分配计算预算                          │
│  • 平均激活 ~27B 参数，范围 18.6B-31.3B                    │
└─────────────────────────────────────────────────────────┘
```

**计算预算控制**：
- 使用 **PID 控制器**调整专家偏置
- 确保平均激活专家数收敛到期望值
- 收敛后波动 <1%

**负载均衡控制**：
- 设备级负载均衡损失
- 将零计算专家作为额外组处理
- 确保 FFN 专家与零计算专家比例收敛

#### 创新二：Shortcut-connected MoE (ScMoE)

**核心思想**：扩大计算-通信重叠窗口。

```
传统 MoE 执行流程：
[All-to-All Dispatch] → [MoE 计算] → [All-to-All Combine]
     通信瓶颈 ↑

ScMoE 执行流程：
[Layer N Dense FFN] ─────────────────────────────────┐
     ↓                                                │
[Layer N+1 All-to-All Dispatch] ← 与 Dense FFN 重叠   │
     ↓                                                │
[Layer N+1 MoE 计算]                                  │
     ↓                                                │
[Layer N+1 All-to-All Combine] ← 与下一层 Dense FFN 重叠
```

**ScMoE 优势**：
- **质量中性**：训练损失曲线与基线几乎一致
- **训练效率**：非重叠通信时间从 25.3% 降至 8.4%
- **推理效率**：理论 TPOT 降低近 50%
- **带宽利用**：NVLink（节点内）与 RDMA（节点间）并行

### 2.3 方差对齐设计

#### MLA Scale-Correction

**问题**：MLA 中 Q/K 向量的方差不匹配。

**解决方案**：

```
α_q = sqrt(d_model / d_q)
α_kv = sqrt(d_model / d_kv)
```

**效果**：在 1B 激活 MoE 模型上显著改善收敛。

#### Variance Compensation for Experts Initialization

**问题**：细粒度专家分割导致方差降低。

**解决方案**：缩放因子 γ = m（分割因子）

**效果**：保持 MoE 层输出方差与分割前一致。

---

## 三、训练策略

### 3.1 超参数迁移

**方法**：基于宽度缩放的超参数迁移。

| 参数 | 迁移规则 |
|------|----------|
| **Embedding 初始化方差** | σ²_target = σ²_proxy |
| **Embedding 学习率** | η_target = η_proxy |
| **Hidden/Unembedding 初始化方差** | σ²_target = σ²_proxy / s |
| **Hidden/Unembedding 学习率** | η_target = η_proxy / s |

其中 s = n_target / n_proxy（宽度缩放因子 = 8）

**优势**：在小代理模型上搜索最优超参数，然后迁移到目标模型。

### 3.2 Model Growth 初始化

**方法**：从半规模模型增长。

```
L_small = l1 ∘ l2 ∘ ... ∘ ln
L_target = L_small ∘ L_small  (r=2)
```

**效果**：
- 初始损失略高
- 但收敛更快
- 最终性能优于随机初始化

### 3.3 训练稳定性

#### Router 稳定性

**问题**：LM 损失与 LB 损失的梯度冲突。

**监控指标**：
- Router Weight Similarity：专家权重向量的余弦相似度
- Gradient Norm Ratio (R_g)：LB 梯度与 LM 梯度的范数比

**指导原则**：保持 R_g < 0.1

#### 激活稳定性：Hidden z-loss

**问题**：大规模激活导致损失尖峰。

**解决方案**：

```
L_Z = (λ/T) * Σ_t (log Σ_i exp(abs(z_t^i)))²
```

**效果**：极小的损失系数即可显著抑制大规模激活，且不影响训练损失。

#### Adam Epsilon 配置

**发现**：
- 当 ε 接近梯度 RMS 范数时，性能急剧下降
- ε 低于阈值后，进一步减小影响可忽略

**配置**：ε = 1e-16（远小于梯度 RMS 范数）

---

## 四、预训练

### 4.1 三阶段课程

| 阶段 | 内容 | 数据量 |
|------|------|--------|
| **Stage 1** | 通用预训练（8192 序列长度） | ~20T tokens |
| **Stage 2** | 推理和编码增强 | 数万亿 tokens |
| **Stage 3** | 长上下文扩展（128K） | 100B tokens |

### 4.2 数据策略

**通用预训练**：
- 两阶段数据融合
- Stage 1：实例级数据混合（质量 + 多样性）
- Stage 2：推理密集型领域（STEM + 代码 = 70%）

**推理和编码增强**：
- 知识图谱遍历和节点组合
- 多阶段迭代优化
- 双模态生成和验证（文本 + 计算）

**长上下文扩展**：
- Stage 1：8K → 32K（80B tokens，RoPE base 1M → 5M）
- Stage 2：32K → 128K（20B tokens，RoPE base 5M → 10M）

### 4.3 去污染

- Web/Code：13-gram 重叠检测
- 合成数据：语义相似度 >0.9 或 0.7-0.9 + 词法重叠

### 4.4 Base Model 评估结果

**与 SOTA Base Model 比较**：

| 基准 | DeepSeek-V3.1 | Llama-4-Maverick | Kimi-K2 | LongCat-Flash |
|------|---------------|------------------|---------|---------------|
| **总参数** | 671B | 402B | 1043B | **560B** |
| **激活参数** | 37B | 17B | 32B | **27B** |
| MMLU | 87.46 | 84.41 | 87.47 | **87.05** |
| MMLU-Pro | 59.29 | 63.90 | 68.36 | **70.32** |
| GPQA | 47.16 | 48.08 | 45.89 | **51.09** |
| SuperGPQA | - | 40.58* | 44.70* | **52.03** |
| BBH | 89.46 | 87.56 | 89.19 | **90.54** |
| GSM8K | 92.22 | 84.61 | 92.27 | **93.86** |
| MATH | 65.38 | 63.34 | 66.74 | **69.28** |
| MBPP+ | 59.26 | 70.11 | 80.49 | **77.25** |
| HumanEval+ | 67.07 | 60.37 | 69.84 | **65.85** |
| MultiPL-E | 62.00 | 58.35 | 59.22 | **69.25** |
| CRUXEval-I | 65.87 | 62.00 | 65.87 | **71.63** |
| CRUXEval-O | 71.25 | 64.25 | 68.75 | **75.88** |

**关键发现**：
- 在更少参数下匹配或超越更大模型
- 在 MMLU-Pro、GPQA、SuperGPQA 等困难基准上优势明显
- 数学和编码任务上表现突出

---

## 五、后训练

### 5.1 推理和编码

**数学**：
- Persona + Self-instruct 范式
- 多个 LLM 生成答案，选择最一致的解决方案
- 训练生成式奖励模型验证逻辑正确性

**编码**：
- 多来源编码查询（公开数据集、GitHub、论坛）
- Code Evol-Instruct 方法
- Agent-based 系统自主分析代码、修复 bug

**逻辑推理**：
- 演绎、假设、归纳推理
- Pass@k 指标平衡难度
- 填空题格式减少随机猜测

### 5.2 Agentic Tool Use

**任务难度三维度**：

| 维度 | 描述 |
|------|------|
| **信息处理复杂性** | 模型需要复杂推理整合信息 |
| **工具集复杂性** | 工具依赖图的节点数和边密度 |
| **用户交互复杂性** | 多轮战略提问，最小化查询次数 |

**多 Agent 数据合成框架**：

| Agent | 职责 |
|-------|------|
| **UserProfileAgent** | 生成用户画像、对话风格、信息披露模式 |
| **ToolSetAgent** | 40 领域 → 1,600 应用 → 80,000 mock 工具 |
| **InstructionAgent** | 生成描述完整任务的指令 |
| **EnvironmentAgent** | 增强环境信息（物品、位置、时间、天气） |
| **RubricAgent** | 构建检查清单，滑动窗口评估 |
| **ValidatorAgent** | 质量检查和去重 |

### 5.3 通用能力

**指令遵循**：
- 单轮/多轮指令遵循数据集
- 反向提示生成策略（从答案生成问题）

**长上下文**：
- 阅读理解、表格问答、自定义任务
- 多跳推理、多轮对话、复杂计算

**安全**：
- 40+ 安全类别
- 5 种响应类型：遵守、遵守+指导、软拒绝、软拒绝+指导、硬拒绝

### 5.4 Chat Model 评估结果

**与 SOTA Chat Model 比较**：

| 基准 | DeepSeek-V3.1 | Qwen3-235B | Kimi-K2 | GPT-4.1 | Claude4-Sonnet | Gemini2.5-Flash | LongCat-Flash |
|------|---------------|------------|---------|---------|----------------|-----------------|---------------|
| **总参数** | 671B | 235B | 1043B | - | - | - | **560B** |
| **激活参数** | 37B | 22B | 32B | - | - | - | **27B** |
| **通用领域** |
| MMLU | 90.96 | 90.23 | 89.86 | 89.64 | 91.75 | 86.33 | **89.71** |
| MMLU-Pro | 84.45 | 84.83 | 82.06 | 81.72 | 83.74 | 81.95 | **82.68** |
| ArenaHard-V2 | 84.10 | **88.20** | 85.70 | 61.50 | 62.10 | 77.00 | 86.50 |
| CEval | 89.21 | **92.70** | 91.26 | 79.53 | 86.63 | 78.78 | 90.44 |
| **指令遵循** |
| IFEval | 86.69 | 88.54 | 88.91 | 85.58 | 88.35 | 83.92 | **89.65** |
| COLLIE | 43.80 | 49.71 | 56.34 | 50.00 | 51.22 | 48.60 | **57.10** |
| Meeseeks-zh | 33.83 | 35.32 | 42.79 | 41.54 | 35.07 | 34.84 | **43.03** |
| **数学推理** |
| MATH500 | 96.08 | **98.80** | 97.60 | 90.60 | 93.80 | 98.40 | 96.40 |
| AIME24 | 66.30* | **81.67** | 69.60* | 47.00 | 47.00 | 79.67 | 70.42 |
| AIME25 | 49.27 | **68.33** | 50.66 | 32.00 | 37.00 | 67.33 | 61.25 |
| BeyondAIME | 36.50 | **57.60** | 36.60 | 22.10 | 20.50 | 44.20 | 43.00 |
| **通用推理** |
| GPQA-diamond | 74.90* | 77.43 | 75.76 | 67.68 | 70.71 | **80.30** | 73.23 |
| ZebraLogic | 85.30 | **94.22** | 89.11 | 56.30* | 80.10 | 57.00 | 89.30 |
| **编码** |
| LiveCodeBench | **56.40*** | 46.48 | 46.70 | 39.21 | 45.59 | 39.65 | 48.02 |
| SWE-Bench | **66.00*** | 48.40 | 64.60 | 51.00 | 68.00* | 42.40 | 60.40 |
| TerminalBench | 31.30* | 17.28 | 25.93 | 28.40 | **40.74** | 12.35 | 39.51 |
| **Agentic Tool Use** |
| τ²-Bench (telecom) | 38.50 | 25.60 | 67.50 | 35.20 | 46.20 | 16.50 | **73.68** |
| τ²-Bench (airline) | 46.00 | 48.00 | 54.20 | 56.00 | **60.00** | 41.50 | 58.00 |
| τ²-Bench (retail) | 64.90 | 70.50 | 70.80 | 74.10 | **80.00** | 64.80 | 71.27 |
| AceBench | 71.70 | 76.00 | 71.70 | **80.10*** | 76.10 | 74.50* | 76.10 |
| VitaBench | 20.30 | 8.50 | 18.20 | 19.00 | 23.00 | 8.00 | **24.30** |
| **安全** |
| Harmful | 82.79 | 80.82 | 53.91 | 56.19 | 66.56 | - | **83.98** |
| Criminal | 87.83 | 89.13 | 77.19 | 81.58 | 87.58 | - | **91.24** |
| Misinformation | 83.17 | 77.76 | 42.68 | 45.49 | 54.91 | - | **81.72** |

**关键发现**：
- **指令遵循**：IFEval、COLLIE、Meeseeks-zh 均排名第一
- **Agentic Tool Use**：τ²-Bench telecom 排名第一（73.68），VitaBench 排名第一（24.30）
- **安全**：Harmful、Criminal 排名第一
- **数学推理**：AIME25、BeyondAIME 表现优秀
- **通用推理**：ZebraLogic 89.30，排名前列

---

## 六、训练基础设施

### 6.1 数值精度控制

**ULP 评估**：
- 使用 ULP（Unit in the Last Place）度量浮点误差
- 比较加速器 BF16 结果与 CPU FP32 真值
- 收集所有算子类型和形状的 ULP 误差

**SDC 检测**：
- 静默数据损坏（Silent Data Corruption）
- 芯片内原地算子重计算机制
- FlashAttention Gradients 对 SDC 最敏感
- 可调整重计算间隔，平衡检测覆盖和计算成本

### 6.2 确定性计算

**确定性 FAG**：
- 使用有限额外工作空间确定性累积 tiles
- 双缓冲流水线、调优的 tiling 调度、负载均衡
- 性能达到非确定性版本的 0.95×

**确定性 ScatterAdd**：
- 层次化归约算法
- 跨所有可用处理器并行化梯度聚合
- 性能达到非确定性版本的水平

**优化 Grouped GEMM**：
- 双缓冲流水线（计算、内存 I/O、epilogue 重叠）
- 对角 tiling 减轻 L2 缓存冲突
- HBM 带宽控制实现 Grouped GEMM 与 dispatch/combine 通信重叠
- 速度提升 5%–45%

### 6.3 分布式策略

**并行架构**：
- Expert Parallelism Groups (EP)：每组 32 加速器
- 注意力层：Context Parallelism (CP=8)
- FFN 层：EP 分区
- 多 EP 组通过 Pipeline Parallelism (PP) 和 Data Parallelism (DP) 扩展

**ScMoE 优化**：
- MoE 层沿 token 维度分成两个 chunks
- Chunk 1：与 Dense FFN 计算重叠
- Chunk 2：与 Chunk 1 重叠
- 非重叠 dispatch/combine 通信时间：25.3% → 8.4%

**Pipeline 策略**：
- V-ZB 算法平衡所有阶段内存使用
- 峰值内存 <60GB
- 零理论 bubbles
- 使用备份数据代替逆操作保持数值位对齐

### 6.4 可靠性

**可用性**：98.48%

**故障恢复**：
- 异步检查点：训练停滞 2∼4 秒
- 在线关键日志过滤
- 优化初始化
- 全自动化恢复：<10 分钟
- 20 次故障全部自动处理

---

## 七、推理和部署

### 7.1 模型特定推理优化

#### 计算和通信编排

**SBO (Single Batch Overlap)**：四阶段流水线执行。

```
Stage 1: MLA 输出（独立执行，作为后续阶段输入）
    ↓
Stage 2: All-to-All Dispatch ∥ Dense FFN + Attn 0 (QKV Projection)
    ↓
Stage 3: MoE GEMM（独立执行，受益于宽 EP 部署）
    ↓
Stage 4: Attn 1 (Core Attention + Output Projection) + Dense FFN ∥ All-to-All Combine
```

**优势**：
- 模块级重叠（第四维度）
- 有效缓解通信开销
- NVLink（节点内）与 RDMA（节点间）并行

#### 推测解码

**MTP 作为 Draft Model**：

| MTP 层 | 激活参数比例 | 接受率 α |
|--------|-------------|----------|
| Dense layer | 1.41% | 92.1% |
| ScMoE layer | 4.17% | 92.9% |

**优化策略**：
- 最大化接受率 α（MTP 轻量架构）
- 最小化 draft-to-target 成本比
- 最小化 target verification-to-decoding 成本比（C2T 方法）

#### 减少 KV Cache

- MLA 64 heads：跨 m 维度共享 KV
- MQA-like 结构吸收方法
- 最大化硬件利用率（WGMMA 指令对齐）

### 7.2 系统级推理技术

#### 最小化调度开销

**TVD 融合**：Target forward + Verification + Draft forward 融合为单个 CUDA graph。

**多步重叠调度器**：
- 单次调度迭代启动多个前向步骤的 kernel
- 隐藏 CPU 调度和同步在 GPU 前向过程中
- 确保持续 GPU 占用

**KV Cache 分配**：
- 数学归纳证明安全分配
- R_i ∈ [2n, 3n]，无需知道当前迭代的接受长度

#### 自定义 Kernel

**MoE GEMM**：
- SwapAB 技术：权重作为左矩阵，激活作为右矩阵
- 利用 n 维度的灵活 8 元素粒度
- 最大化 tensor core 利用率

**通信 Kernel**：
- NVLink Sharp 硬件加速广播和归约
- 内联 PTX 汇编
- 支持均匀和非均匀 token 分布
- 仅使用 4 个 thread blocks
- 在 4KB 到 96MB 消息大小上优于 NCCL 和 MSCCL++

#### 量化

**方案**：
- 激活：[1,128] blocks 细粒度量化
- 权重：[128,128] blocks 细粒度量化
- 层级混合精度量化
  - 方法一：FPTQ + Super-Expert（检测极端激活幅度）
  - 方法二：逐层计算 block-wise FP8 量化误差
  - 取两种方法的交集

### 7.3 部署性能

**部署架构**：
- PD-Disaggregated（Prefill-Decode 分离）
- 层级传输减少 TTFT
- 最小部署单元：2 节点 16 H800-80GB GPUs
- 宽 EP 部署 + DeepEP
- 支持零计算专家（无通信）

**性能对比**：

| 模型 | Attention | 平均上下文 | Hopper GPUs | TGS | TPS/u |
|------|-----------|-----------|-------------|-----|-------|
| DeepSeek-V3-profile | bf16 | 4096 | 128 | 2324 | 20 |
| DeepSeek-V3-blog | bf16 | 4989 | 144 | 1850 | 20~22 |
| **LongCat-Flash** | bf16 | 5000 | 128 | **3785** | **35** |
| **LongCat-Flash** | bf16 | 5000 | 128 | 2205 | **68.9** |
| **LongCat-Flash** | bf16 | 5000 | 128 | 804 | **100.5** |
| **LongCat-Flash** | fp8 | 5000 | 128 | **4230** | 26.4 |
| **LongCat-Flash** | fp8 | 8192 | 128 | **3240** | 33.8 |

**关键发现**：
- **TGS**：LongCat-Flash bf16 是 DeepSeek-V3 的 1.63×
- **TPS/u**：LongCat-Flash 最高 100.5 TPS（bf16）
- **成本**：$0.70 / 百万输出 token（H800 $2/小时）
- **Agent 应用**：单轮工具调用延迟 <1 秒

---

## 八、总结

### 核心贡献

1. **架构创新**：
   - **Zero-computation Experts**：动态计算预算分配，根据上下文重要性激活 18.6B-31.3B 参数
   - **ScMoE**：扩大计算-通信重叠窗口，理论 TPOT 降低近 50%
   - **方差对齐**：MLA Scale-Correction + Variance Compensation

2. **训练策略**：
   - **超参数迁移**：从代理模型预测目标模型最优配置
   - **Model Growth**：从半规模模型增长初始化
   - **稳定性套件**：Router 梯度平衡 + Hidden z-loss + Adam Epsilon 配置
   - **确定性计算**：位级可复现，SDC 检测

3. **多阶段训练**：
   - 通用预训练 → 推理编码增强 → 长上下文扩展
   - 多 Agent 数据合成框架
   - 三维度任务难度控制

4. **推理优化**：
   - **SBO**：四阶段流水线，模块级重叠
   - **MTP 推测解码**：92.1% 接受率
   - **自定义 Kernel**：MoGEMM SwapAB、NVLink Sharp 通信
   - **量化**：细粒度 block-wise + 层级混合精度

### 关键数据

| 指标 | 数值 |
|------|------|
| **总参数** | 560B |
| **激活参数** | 27B（平均） |
| **训练数据** | >20T tokens |
| **训练时间** | 30 天 |
| **训练可用性** | 98.48% |
| **推理速度** | >100 TPS（H800） |
| **推理成本** | $0.70 / 百万输出 token |
| **上下文长度** | 128K |
| **MTP 接受率** | 92.1% |
| **ScMoE 通信优化** | 25.3% → 8.4% |

### 技术影响

LongCat-Flash 证明了：
- **Zero-computation Experts** 是 MoE 模型的有效计算优化
- **ScMoE** 可以在不牺牲质量的情况下显著提升效率
- **模型-系统协同设计** 是大规模训练和推理的关键
- **Agentic 能力** 可以通过精心设计的数据合成框架培养

### 与同类模型比较

| 模型 | 总参数 | 激活参数 | 特点 |
|------|--------|----------|------|
| **DeepSeek-V3** | 671B | 37B | MLA + DeepSeekMoE |
| **Kimi-K2** | 1043B | 32B | 大规模 MoE |
| **Qwen3-235B** | 235B | 22B | 混合推理模式 |
| **Llama-4-Maverick** | 402B | 17B | Meta MoE |
| **LongCat-Flash** | 560B | 27B | Zero-computation Experts + ScMoE |

LongCat-Flash 在更少参数下实现了与更大模型可比的性能，并在 Agentic 任务上展现出明显优势。

---

## 九、参考资源

- **论文**: https://arxiv.org/abs/2509.01322
- **代码**: https://github.com/meituan-longcat
- **在线**: https://longcat.ai
- **Hugging Face**: https://huggingface.co/meituan-longcat

### 相关论文

- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- MLA (Multi-head Latent Attention): https://arxiv.org/abs/2405.04434
- ScMoE (Shortcut-connected MoE): https://arxiv.org/abs/2410.06664
- Kimi-K2: https://arxiv.org/abs/2507.04299
- Qwen3: https://arxiv.org/abs/2505.09388
- Llama-4: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- MTP (Multi-Token Prediction): https://arxiv.org/abs/2401.14629
