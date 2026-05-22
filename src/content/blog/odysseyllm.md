---
title: "OdysseyLLM: A Speed Odyssey for Deployable Quantization of LLMs"
description: "OdysseyLLM 提出可部署的大语言模型量化加速方案。"
date: 2026-03-30
tags:
  - "Deep Learning"
  - "Paper Summary"
draft: false
---
# OdysseyLLM: A Speed Odyssey for Deployable Quantization of LLMs

## 一、论文概述

| 项目 | 内容 |
|------|------|
| **标题** | A Speed Odyssey for Deployable Quantization of LLMs |
| **作者** | Qingyuan Li, Ran Meng, Yiduo Li, Bo Zhang, Liang Li, Yifan Lu, Xiangxiang Chu, Yerui Sun, Yuchen Xie |
| **机构** | Meituan Inc. |
| **论文** | https://arxiv.org/abs/2311.09550 |
| **发布** | 2023年11月 |

## 二、核心思想

### 2.1 问题定义

现有 LLM 量化方法存在**软件中心化**问题：

| 问题 | 表现 | 后果 |
|------|------|------|
| **忽略部署可行性** | 追求极低比特（W4A4） | 精度严重下降 |
| **硬件约束忽视** | 使用 fine-grained 量化 | 推理开销抵消加速 |
| **算法复杂度过高** | 引入额外计算/内存访问 | 实际加速有限 |

### 2.2 解决方案：硬件中心化方法

**OdysseyLLM** 提出**硬件中心化**的量化方法：

1. **W4A8 量化**：结合 W4A16 的内存带宽优势和 W8A8 的计算加速优势
2. **FastGEMM 内核**：专为 W4A8 设计的高效矩阵乘法
3. **对称量化策略**：消除零点减法，简化计算
4. **自适应权重裁剪**：补偿 per-channel 量化精度损失

### 2.3 核心贡献

1. **范式转变**：从软件中心化转向硬件中心化
2. **首个可部署 W4A8 方案**：FastGEMM 内核 + 量化策略组合
3. **显著加速**：
   - 4× vs HuggingFace FP16
   - 2.23× vs TensorRT-LLM FP16
   - 1.45× vs TensorRT-LLM INT8
4. **精度保持**：与 SmoothQuant W8A8 相当

---

## 三、技术架构

### 3.1 量化方案对比（Figure 2）

| 方案 | 权重 | 激活 | 粒度 | 计算流程 |
|------|------|------|------|----------|
| **W4A16** | INT4 | FP16 | group-wise | Dq(INT4)→FP16→FP16 GEMM |
| **W4A8 (传统)** | INT4 | INT8 | group-wise | UINT4→SINT8→INT8 GEMM→per-group Dq |
| **W8A8** | INT8 | INT8 | per-channel | INT8 GEMM→per-channel Dq |
| **W4A8 (Ours)** | SINT4 | INT8 | per-channel | SINT4→SINT8 (×16)→INT8 GEMM→Dq÷16 |

### 3.2 FastGEMM 设计（Figure 4）

#### 核心创新：SINT4toS8 转换

```
传统 UINT4toS8：
  -7 (1111 1001) → unpack → subtract 8 (INT32) → pack → -7 (0000 0001)
  需要 INT32 减法，开销大

OdysseyLLM SINT4toS8：
  -7 (1001) → 左移4位 → 1001 0000 = -112 = -7 × 16
  仅需位移操作，无减法
```

**优势**：
- 消除 INT8 减法（硬件不支持 SINT8 减法）
- 无需 INT32 类型转换
- GEMM 后除以 16 恢复正确结果（无溢出风险）

#### 三步优化

| 步骤 | 优化 | 效果 |
|------|------|------|
| **Kernel 融合** | SINT4toS8 + GEMM 融合为单个内核 | 减少内存访问 |
| **消除 INT8 减法** | 对称量化，无零点 | 简化计算 |
| **复用符号位** | SINT4 直接放在 SINT8 高 4 位 | 避免类型转换 |

### 3.3 量化策略

#### 自适应权重裁剪（Adaptive Weight Clipping）

```
W_q = clamp(W/S, -2^{N-1}, 2^{N-1}-1)
S = max(|γ·max(W)|, |β·min(W)|) / (2^{N-1}-1)
```

**效果**：
- 权重分布更紧凑：(-0.4, 0.2) → (-0.2, 0.2)
- per-channel 量化 MSE 显著降低
- 对称量化，硬件友好

#### Hessian 训练无关补偿

```
δ_F = -(W_i - Q(W_i)) * [H_F^{-1}]_{ii} * [H_F^{-1}]_{:,i}
```

- 逐层最小化量化误差
- 使用 GPTQ 加速 Hessian 计算
- 无需微调

---

## 四、实验结果

### 4.1 实验配置

| 项目 | 配置 |
|------|------|
| **模型** | LLaMA-1 (7B/13B/65B), LLaMA-2 (7B/13B/70B) |
| **硬件** | NVIDIA A100-80G GPU |
| **校准数据** | C4 数据集 128 条序列 |
| **评估指标** | LAMBADA, C4, WikiText2 困惑度，Common Sense QA 准确率 |
| **基线** | FP16, AWQ-g128, GPTQ-g128, SmoothQuant |

### 4.2 精度对比（Table 2）

**LAMBADA 数据集**：

| 方法 | 比特 | LLaMA-7B | LLaMA-13B | LLaMA-65B | LLaMA-2-7B | LLaMA-2-13B | LLaMA-2-70B |
|------|------|----------|-----------|-----------|------------|-------------|-------------|
| FP16 | W16A16 | 73.74% | 76.19% | 79.20% | 73.70% | 76.64% | 79.57% |
| AWQ-g128 | W4A16 | 66.80% | 73.72% | 78.73% | 70.23% | 75.80% | 78.40% |
| GPTQ-g128 | W4A16 | 70.21% | 75.68% | 78.77% | 72.31% | 75.99% | 79.86% |
| SmoothQuant | W8A8 | 73.49% | 76.15% | 78.07% | 73.36% | 76.05% | 78.71% |
| **OdysseyLLM** | **W4A8** | **73.49%** | **76.23%** | **78.56%** | 70.81% | **76.07%** | **79.43%** |

**关键发现**：
- OdysseyLLM W4A8 与 SmoothQuant W8A8 精度相当
- 在大模型上甚至略优于 SmoothQuant
- 显著优于 AWQ 和 GPTQ 的 W4A16

### 4.3 困惑度对比

**WikiText2 数据集**：

| 方法 | 比特 | LLaMA-7B | LLaMA-13B | LLaMA-65B |
|------|------|----------|-----------|-----------|
| FP16 | W16A16 | 5.73 | 5.10 | 3.51 |
| SmoothQuant | W8A8 | 5.89 | 5.21 | 3.73 |
| **OdysseyLLM** | **W4A8** | **6.17** | **5.37** | **3.92** |

### 4.4 Common Sense QA（Table 3）

| 模型 | 方法 | WinoGrande | PIQA | HellaSwag | ARC-e | 平均 |
|------|------|------------|------|-----------|-------|------|
| LLaMA-1-13B | FP16 | 0.7277 | 0.8009 | 0.7907 | 0.7471 | 0.7666 |
| LLaMA-1-13B | SmoothQuant | 0.7238 | 0.8020 | 0.7836 | 0.7466 | 0.7640 |
| LLaMA-1-13B | **OdysseyLLM** | **0.7238** | **0.7998** | **0.7792** | **0.7441** | **0.7617** |
| LLaMA-2-70B | FP16 | 0.7798 | 0.8275 | 0.8381 | 0.8098 | 0.8138 |
| LLaMA-2-70B | SmoothQuant | 0.7766 | 0.8303 | 0.8345 | 0.8127 | 0.8135 |
| LLaMA-2-70B | **OdysseyLLM** | **0.7751** | **0.8313** | **0.8272** | **0.8060** | **0.8099** |

### 4.5 延迟对比

#### 与 FP16 对比（Figure 6）

| 模型 | 加速比 |
|------|--------|
| LLaMA-2-7B | 1.9× |
| LLaMA-2-13B | 2.15× |
| LLaMA-2-70B | 1.76× |

#### 与 TensorRT-LLM 对比（Table 4）

| 模型 | TensorRT-LLM FP16 | TensorRT-LLM W8A8 | OdysseyLLM W4A8 | vs FP16 | vs W8A8 |
|------|-------------------|-------------------|-----------------|---------|---------|
| LLaMA-2-7B | 1411 ms | 1030 ms | **751 ms** | 1.87× | 1.37× |
| LLaMA-2-13B | 2547 ms | 1657 ms | **1139 ms** | 2.23× | 1.45× |
| LLaMA-2-70B | 4177 ms | 3087 ms | **2263 ms** | 1.83× | 1.36× |

#### 与 QUIK 对比（Table 5）

| 阶段 | QUIK W4A4 | OdysseyLLM W4A8 | 加速比 |
|------|-----------|-----------------|--------|
| Context decode | 0.139 ms | 0.121 ms | 1.14× |
| **Self-decode** | 0.052 ms | **0.012 ms** | **4.33×** |

**关键发现**：
- Self-decode 阶段加速最显著（内存密集型）
- Fine-grained GEMM 开销被消除

### 4.6 FastGEMM 消融实验（Figure 7）

| GEMM 类型 | Self-decode 延迟 | 说明 |
|-----------|------------------|------|
| Fine-grained GEMM | 2.64 × 10^5 ns | 基准 |
| Asym GEMM | 0.87 × 10^5 ns | 3.0× 加速 |
| **FastGEMM** | **0.87 × 10^5 ns** | **3.0× 加速** |

### 4.7 量化策略消融（Table 6）

| 方法 | LLaMA-7B PPL | LLaMA-2-7B PPL |
|------|--------------|----------------|
| Baseline (Vanilla W4A8) | 6.73 | 7.13 |
| + LWC | 6.25 | 6.73 |
| + LWC + GPTQ | **6.17** | **6.11** |

---

## 五、核心技术详解

### 5.1 硬件约束分析

#### NVIDIA GPU 限制
- 无 SINT8 减法指令
- 需要 INT32 类型转换
- 额外开销显著

#### Fine-grained 量化开销
- 每组需要反量化到 FP32
- 累加操作开销大
- 抵消低比特加速收益

### 5.2 SINT4toS8 转换详解

```
SINT4 存储为二进制补码：
  -8: 1000
  -7: 1001
  -6: 1010
  ...
  -1: 1111
   0: 0000
   1: 0001
  ...
   7: 0111

转换为 SINT8：
  1001 (−7) → 1001 0000 = −112 = −7 × 16

GEMM 后除以 16：
  结果 = Σ(A_i × W_i × 16) / 16 = Σ(A_i × W_i)
```

**优势**：
- 无减法操作
- 无 INT32 类型转换
- 内部累加器 INT32 无溢出风险

### 5.3 对称量化优势

| 方面 | 对称量化 | 非对称量化 |
|------|----------|------------|
| **零点** | 无 | 需要计算和存储 |
| **计算** | 无减法 | 需要减法 |
| **硬件** | 直接支持 | 需要类型转换 |
| **精度** | LLaMA 系列更优 | 略差 |

---

## 六、相关工作

### 6.1 量化方法对比

| 方法 | 比特 | 粒度 | 特点 |
|------|------|------|------|
| **SmoothQuant** | W8A8 | per-channel/token | 激活难度转移到权重 |
| **GPTQ** | W4A16 | group-wise | Hessian 补偿 |
| **AWQ** | W4A16 | group-wise | 激活感知 |
| **QUIK** | W4A4 | mixed | 异常值高精度 |
| **OdysseyLLM** | W4A8 | per-channel | 硬件中心化 |

### 6.2 比特宽度选择

| 方案 | Context Decode | Self Decode | 适用场景 |
|------|----------------|-------------|----------|
| W8A8 | 快 | 中等 | 计算密集型 |
| W4A16 | 中等 | 快 | 内存密集型 |
| **W4A8** | **快** | **快** | **两者兼顾** |

---

## 七、总结

### 核心贡献

1. **硬件中心化范式**：
   - 消除不可部署的算法选择
   - 最大化硬件加速收益
   - 首个可部署 W4A8 方案

2. **FastGEMM 内核**：
   - SINT4toS8 位移转换
   - Kernel 融合减少内存访问
   - 对称量化消除减法

3. **量化策略组合**：
   - 自适应权重裁剪（LWC）
   - Hessian 训练无关补偿（GPTQ）
   - Per-channel 权重量化

4. **显著性能提升**：

| 对比 | 加速比 |
|------|--------|
| vs HuggingFace FP16 | 4× |
| vs TensorRT-LLM FP16 | 2.23× |
| vs TensorRT-LLM INT8 | 1.45× |
| vs QUIK (self-decode) | 4.33× |

5. **精度保持**：
   - 与 SmoothQuant W8A8 相当
   - 显著优于 W4A16 方法

### 关键数据

| 指标 | 数值 |
|------|------|
| **最大加速** | 4× (vs FP16) |
| **TensorRT-LLM FP16** | 2.23× |
| **TensorRT-LLM INT8** | 1.45× |
| **QUIK self-decode** | 4.33× |
| **精度损失** | <1% (vs W8A8) |

### 技术影响

OdysseyLLM 证明了：
- 硬件约束是量化算法设计的首要考虑
- W4A8 是比 W4A16 更优的部署选择
- Per-channel 量化 + 补偿策略可达 fine-grained 精度
- Kernel 优化是实际加速的关键

### 局限性

1. **硬件依赖**：针对 NVIDIA GPU 优化
2. **模型范围**：主要验证 LLaMA 系列
3. **比特宽度**：仅支持 W4A8
4. **校准数据**：需要 128 条序列

### 未来方向

1. **更多硬件平台**：AMD, Intel, 移动端
2. **更多模型**：多模态、MoE
3. **更低比特**：W2A8, W4A4
4. **端到端优化**：与推理框架深度集成

---

## 八、参考资源

- **论文**: https://arxiv.org/abs/2311.09550
- **相关论文**:
  - GPTQ: https://arxiv.org/abs/2210.17323
  - AWQ: https://arxiv.org/abs/2306.00978
  - SmoothQuant: https://arxiv.org/abs/2211.10438
  - QUIK: https://arxiv.org/abs/2310.09259
  - OmniQuant: https://arxiv.org/abs/2308.13137
