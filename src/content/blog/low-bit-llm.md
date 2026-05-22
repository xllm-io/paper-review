---
title: "A Survey of Low-bit Large Language Models"
description: "低比特大语言模型综述：基础、系统和算法。"
date: 2026-03-25
tags:
  - "Deep Learning"
  - "Paper Summary"
draft: false
---
# A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms

## 一、论文概述

| 项目 | 内容 |
|------|------|
| **标题** | A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms |
| **作者** | Ruihao Gong, Yifu Ding, Zining Wang, Chengtao Lv, Xingyu Zheng 等 |
| **机构** | Beihang University, ETH Zurich, SenseTime, CUHK |
| **论文** | https://arxiv.org/abs/2409.16694v2 |
| **期刊** | Neural Networks |
| **发布** | 2024年9月 |

## 论文图表

| 图表 | 文件 | 说明 |
|------|------|------|
| Figure 1 | [fig1_skeleton.png](figures/low-bit-llm/fig1_skeleton.png) | LLM 量化方法骨架 |
| Figure 2 | [fig2_granularity.png](figures/low-bit-llm/fig2_granularity.png) | 量化粒度说明 |
| Figure 3 | [fig3_dynamic_static.png](figures/low-bit-llm/fig3_dynamic_static.png) | 动态/静态量化 |
| Figure 4 | [fig4_data_transmission.png](figures/low-bit-llm/fig4_data_transmission.png) | 推理数据传输过程 |
| Figure 5 | [fig5_quantization_process.png](figures/low-bit-llm/fig5_quantization_process.png) | 量化加速过程 |
| Figure 6 | [fig6_kv_cache.png](figures/low-bit-llm/fig6_kv_cache.png) | KV Cache 量化 |
| Figure 7 | [fig7_lora.png](figures/low-bit-llm/fig7_lora.png) | LoRA 结构 |
| Figure 8 | [fig8_ptq_overview.png](figures/low-bit-llm/fig8_ptq_overview.png) | PTQ 算法概览 |
| Figure 9 | [fig9_shifting.png](figures/low-bit-llm/fig9_shifting.png) | 移位变换 |
| Figure 10 | [fig10_scaling.png](figures/low-bit-llm/fig10_scaling.png) | 缩放变换 |
| Figure 11 | [fig11_rotation.png](figures/low-bit-llm/fig11_rotation.png) | 旋转变换 |

---

## 二、核心思想

这是一篇全面综述低比特量化在大语言模型中应用的论文，涵盖基础概念、系统实现和算法策略三个维度。

### 2.1 量化动机

| 挑战 | 量化解决方案 |
|------|--------------|
| **内存需求大** | 减少参数/激活/梯度比特宽度 |
| **计算密集** | 低比特矩阵乘法加速 |
| **部署困难** | 压缩模型便于边缘部署 |

### 2.2 论文结构（Figure 1）

```
┌─────────────────────────────────────────────────────────┐
│              LLM 量化方法骨架                             │
├─────────────────────────────────────────────────────────┤
│  Section 2: 基础                                        │
│  ├── 低比特数据格式（FP8, INT4, NF, SF, Flint, Abfloat）│
│  ├── 量化粒度（tensor/channel/group/element-wise）      │
│  └── 动态/静态量化                                      │
├─────────────────────────────────────────────────────────┤
│  Section 3: 系统支持                                    │
│  ├── 推理框架（TensorRT-LLM, vLLM, llama.cpp 等）      │
│  ├── Weight-only 量化                                   │
│  ├── Weight & Activation 量化                           │
│  ├── KV Cache 量化                                      │
│  └── 量化/反量化实现                                    │
├─────────────────────────────────────────────────────────┤
│  Section 4: 训练量化                                    │
│  ├── BF16/FP16/FP8/INT8 训练                           │
│  └── 低比特微调（LoRA, QLoRA）                         │
├─────────────────────────────────────────────────────────┤
│  Section 5: 推理量化                                    │
│  ├── QAT（量化感知训练）                                │
│  ├── PTQ（训练后量化）                                  │
│  │   ├── 等效变换（shifting/scaling/rotation）          │
│  │   ├── 权重补偿                                      │
│  │   └── 混合精度                                      │
│  └── 工具包                                            │
└─────────────────────────────────────────────────────────┘
```

---

## 三、低比特数据格式

### 3.1 标准格式（Table 1）

| 格式 | 最大值 | 最小值 | 说明 |
|------|--------|--------|------|
| INT4 | 7 | -8 | 4比特整数 |
| INT8 | 127 | -128 | 8比特整数 |
| FP8 (E4M3) | 448 | -448 | 8比特浮点 |
| FP8 (E5M2) | 57344 | -57344 | 8比特浮点 |
| FP16 (E5M10) | 65504 | -65504 | 半精度 |
| BF16 (E8M7) | 3.39e38 | -3.39e38 | 脑浮点 |
| FP32 (E8M23) | 3.40e38 | -3.40e38 | 单精度 |

### 3.2 特殊格式

#### NormalFloat (NF)
- 用于权重量化（weight-only）
- 基于正态分布的分位数
- 信息论最优：每个量化区间期望值相等
- 代表：QLoRA 使用的 NF4

#### Student Float (SF)
- 改进 NF，假设参数服从 Student's t 分布
- 自由度 ν 控制分布形状
- ν→∞ 时收敛到 NF

#### Micro Scaling FP (MX)
- 行业联盟标准（AMD, ARM, Intel, Meta, Microsoft, NVIDIA, Qualcomm）
- E8M0 缩放因子
- 支持 FP8/FP6/FP4/INT8
- 细粒度子块共享缩放

#### Floating-point Integer (Flint)
- 结合浮点和整数优势
- 通过前导零检测器(LZD)扩展表示范围
- 4比特 Flint 可表示更大范围

#### Adaptive Biased Float (Abfloat)
- 用于处理异常值
- 更大 bias 扩展表示范围
- 仅用于异常值，正常值用 INT4/8 或 Flint4

---

## 四、量化粒度（Figure 2）

| 粒度 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **Tensor-wise** | 整个张量一个缩放因子 | 最快 | 精度损失大 |
| **Token-wise** | 每个 token 一个缩放因子 | 捕获 token 变化 | 计算开销 |
| **Channel-wise** | 每个通道一个缩放因子 | 可合并到权重 | 存储开销 |
| **Group-wise** | 每组共享缩放因子 | 平衡精度和效率 | 需要分组 |
| **Element-wise** | 每个元素一个缩放因子 | 最精细 | 仅训练时使用 |

**实践组合**：
- Token-wise 激活 + Channel-wise 权重
- Group-wise（group size 32/64/128）

---

## 五、动态与静态量化（Figure 3）

### 5.1 动态量化

- **特点**：运行时计算激活的缩放因子
- **优点**：无需校准，灵活适应输入分布
- **缺点**：额外计算开销
- **适用**：快速部署场景

### 5.2 静态量化

- **特点**：预先校准确定缩放因子
- **优点**：推理性能更快
- **缺点**：需要校准数据
- **分类**：
  - Weight & Activation：两者都量化，低比特 MatMul
  - Weight-only：仅权重量化，反量化后 FP16 MatMul

---

## 六、系统支持

### 6.1 推理框架对比（Table 2）

| 框架 | 支持算法 | 硬件平台 | 机构 |
|------|----------|----------|------|
| **TensorRT-LLM** | AWQ, GPTQ, SmoothQuant | NVIDIA GPU | NVIDIA |
| **Transformers** | AQLM, AWQ, AutoGPTQ, HQQ 等 | CPU, MPS, GPU | HuggingFace |
| **vLLM** | AQLM, AWQ, GPTQ, SmoothQuant | AMD/NVIDIA GPU, TPU | UC Berkeley |
| **llama.cpp** | AWQ | AMD/NVIDIA GPU | ggml |
| **MLC-LLM** | - | AMD/NVIDIA/Apple/Intel/Mobile GPU | MLC |
| **QServe** | QoQ | NVIDIA GPU | MIT EECS |
| **SGLang** | AWQ, GPTQ | NVIDIA GPU | LMSYS |

### 6.2 比特宽度支持

#### Weight-only
- 支持任意非均匀量化（2-8 bit）
- 权重反量化后 FP16 MatMul
- 加速原理：减少数据传输量

#### Wbit & Abit
- 低比特指令 MatMul
- 主流支持：INT8, FP8
- 部分支持：INT4
- 需要硬件指令集支持

#### KV Cache
- 压缩 KV Cache 减少内存
- 支持：FP16, FP8, INT8, INT4, INT2
- 技术：量化窗口、跳过反量化、异常值处理

### 6.3 数据传输与加速原理（Figure 4, 5）

```
Host Memory → Device Memory (25 GB/s)
    ↓
Off-chip → On-chip (1555 GB/s)
    ↓
Shared Memory → Registers (19400 GB/s)
    ↓
Computation (MatMul)
    ↓
Offload results
```

**Weight-only 加速**：
- 减少 Host→Device 数据量
- 反量化开销 < 传输节省时间
- 即使用 FP16 MatMul 也能加速

**W&A 量化加速**：
- 减少传输 + 低比特 MatMul
- 额外开销：激活量化 + 结果类型转换
- 更大加速潜力

### 6.4 KV Cache 量化（Figure 6）

**技术**：
1. **低比特量化**：4-bit/2-bit KV Cache
2. **量化窗口**：仅量化超出窗口的 KV
3. **跳过 K_new 反量化**：保留更多信息
4. **异常值优化**：高比特存储异常值

---

## 七、训练量化

### 7.1 低比特训练

| 格式 | 特点 | 硬件要求 |
|------|------|----------|
| **BF16** | 稳定，广泛使用 | Ampere/Hopper |
| **FP16** | 需要 loss scaling | 通用 |
| **FP8** | 内存减半 | H100 |
| **INT8** | 研究阶段 | 特定支持 |

### 7.2 低比特微调

#### LoRA（Figure 7）

```
原始权重 W (frozen)
    ↓
W + ΔW = W + BA
    ↓
B (d×r) × A (r×d) << W (d×d)
```

**优势**：仅训练低秩矩阵，大幅减少参数量

#### QLoRA

- 4-bit NormalFloat (NF4) 权重
- 双重量化：缩放因子也量化
- 分页优化器：CPU/GPU 内存管理
- 48GB GPU 可微调 65B 模型

---

## 八、推理量化算法

### 8.1 量化感知训练 (QAT)

- 训练时模拟量化效果
- 适用于极低比特（1-2 bit）
- 代表：LLM-QAT, BitNet

### 8.2 训练后量化 (PTQ)

#### 8.2.1 等效变换（Figure 8）

##### Shifting 变换（Figure 9）
- 移位消除异常值影响
- Δ 可合并到参数中
- 推理无额外开销

##### Scaling 变换（Figure 10）
- 缩放平衡激活范围
- Φ 可合并到参数中
- SmoothQuant 代表方法

##### Rotation 变换（Figure 11）
- 旋转减少异常值
- 激活分布更均匀
- QuIP, QuIP# 代表方法

#### 8.2.2 权重补偿

- 校正量化误差
- 逐层或逐通道补偿
- GPTQ, AWQ 代表方法

#### 8.2.3 混合精度

- 不同层/组件使用不同精度
- 异常值高精度，正常值低精度
- SpQR, LLM.int8() 代表方法

### 8.3 工具包

| 工具 | 功能 |
|------|------|
| **AutoGPTQ** | GPTQ 自动化 |
| **AutoAWQ** | AWQ 自动化 |
| **llm-awq** | AWQ 实现 |
| **QuIP#** | 旋转量化 |
| **AQLM** | 加性量化 |
| **QLoRA** | 4-bit 微调 |

---

## 九、核心算法详解

### 9.1 GPTQ

- 逐层量化，最小化重建误差
- Hessian 信息指导量化顺序
- 支持 2-8 bit 权重量化

### 9.2 AWQ

- 激活感知权重量化
- 显著权重通道保护
- 无需反向传播

### 9.3 SmoothQuant

- 激活-权重平滑
- 将激活难度转移到权重
- W8A8 量化

### 9.4 QuIP / QuIP#

- 旋转量化
- 减少异常值影响
- 2-bit 量化可行

### 9.5 LLM.int8()

- 混合精度分解
- 异常值 FP16，正常值 INT8
- 8-bit MatMul

---

## 十、未来趋势

### 10.1 极低比特量化

- 1-bit / 2-bit 模型
- Binary 量化的新形式
- BitNet, DB-LLM

### 10.2 硬件协同设计

- 专用量化指令
- MX 格式硬件支持
- 端到端优化

### 10.3 量化 + 其他压缩

- 量化 + 剪枝
- 量化 + 知识蒸馏
- 混合压缩策略

### 10.4 新兴应用

- 多模态 LLM 量化
- MoE 模型量化
- 长上下文 KV Cache 压缩

---

## 十一、总结

### 核心贡献

1. **全面综述**：覆盖基础、系统、算法三个维度
2. **数据格式梳理**：标准格式 + 自定义格式（NF, SF, MX, Flint, Abfloat）
3. **系统对比**：20+ 推理框架详细对比
4. **算法分类**：QAT vs PTQ，等效变换 vs 权重补偿
5. **实践指导**：量化粒度、动态/静态选择

### 关键数据

| 方法 | 权重比特 | 激活比特 | 精度影响 |
|------|----------|----------|----------|
| FP8 | 8 | 8 | 轻微 |
| INT8 | 8 | 8 | 轻微 |
| INT4 (weight-only) | 4 | 16 | 可接受 |
| INT4 (W&A) | 4 | 4 | 较大 |
| 2-bit | 2 | 16 | 显著 |
| Binary | 1 | 1 | 严重 |

### 实践建议

| 场景 | 推荐方法 |
|------|----------|
| **快速部署** | Weight-only INT4 (GPTQ/AWQ) |
| **高吞吐推理** | W8A8 (SmoothQuant) |
| **边缘设备** | INT4/INT3 weight-only |
| **微调** | QLoRA (NF4) |
| **极致压缩** | QuIP# (2-bit) |

---

## 十二、参考资源

- **论文**: https://arxiv.org/abs/2409.16694v2
- **相关论文**:
  - GPTQ: https://arxiv.org/abs/2210.17323
  - AWQ: https://arxiv.org/abs/2306.00978
  - SmoothQuant: https://arxiv.org/abs/2211.10438
  - QLoRA: https://arxiv.org/abs/2305.14314
  - QuIP#: https://arxiv.org/abs/2402.04396
  - LLM.int8(): https://arxiv.org/abs/2208.07339
  - BitNet: https://arxiv.org/abs/2310.11453
