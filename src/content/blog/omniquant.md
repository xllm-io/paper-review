---
title: "OmniQuant: Omnidirectionally Calibrated Quantization for LLMs"
description: "OmniQuant 提出全方位校准的大语言模型量化方法。"
date: 2026-03-18
tags:
  - "Deep Learning"
  - "Paper Summary"
draft: false
---
# OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models

## 论文图表

| 图表 | 文件 | 说明 |
|------|------|------|
| Figure 1 | [fig1.png](figures/omniquant/fig1.png) | OmniQuant 概览：QAT 性能 + PTQ 效率 |
| Figure 2 | [fig2.png](figures/omniquant/fig2.png) | LLaMA 家族上的 OmniQuant 特性 |
| Figure 3 | [fig3.png](figures/omniquant/fig3.png) | Transformer Block 中的 OmniQuant 详解 |
| Figure 4 | [fig4.png](figures/omniquant/fig4.png) | W3A16g128 量化在 Vicuna-Bench 上的比较 |
| Figure A1 | [fig5.png](figures/omniquant/fig5.png) | 学习到的裁剪尺度可视化 |
| Figure A2 | [fig6.png](figures/omniquant/fig6.png) | 激活可视化（原始/SmoothQuant/LET后） |
| Figure A3 | [fig7.png](figures/omniquant/fig7.png) | Block-wise 量化误差比较 |
| Figure A4 | [fig8.png](figures/omniquant/fig8.png) | 比特级别的困惑度缩放规律 |
| Figure A5 | [fig9.png](figures/omniquant/fig9.png) | 权重范围变化可视化 |
| Figure A6 | [fig10.png](figures/omniquant/fig10.png) | 性能概览权衡曲线 |

---

## 一、论文概述

| 项目 | 内容 |
|------|------|
| **标题** | OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models |
| **作者** | Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao 等 |
| **机构** | Shanghai AI Laboratory, The University of Hong Kong, The Chinese University of Hong Kong |
| **论文** | https://arxiv.org/abs/2308.13137v3 |
| **代码** | https://github.com/OpenGVLab/OmniQuant |
| **发布** | 2023年8月 |

## 二、核心思想

### 2.1 问题定义

现有 LLM 量化方法存在**手工参数设计**的局限：

| 问题 | 表现 | 后果 |
|------|------|------|
| **手工迁移强度** | SmoothQuant 使用预定义的迁移强度 | 低比特量化性能差 |
| **网格搜索缩放因子** | AWQ 使用网格搜索的通道缩放 | 参数非最优，耗时 |
| **QAT 成本过高** | LLM-QAT 需要 100K 样本和数百 GPU 小时 | 不实用 |

**核心问题**：能否达到 QAT 的性能，同时保持 PTQ 的时间和数据效率？

### 2.2 解决方案：OmniQuant

**OmniQuant** 提出了一种**全方位校准量化**技术：

```
┌─────────────────────────────────────────────────────────┐
│                    OmniQuant 框架                         │
├─────────────────────────────────────────────────────────┤
│  1. 冻结原始全精度权重（不修改权重本身）                    │
│  2. 仅学习少量量化参数（Θ1, Θ2）                          │
│  3. Block-wise 误差最小化（逐层优化）                      │
├─────────────────────────────────────────────────────────┤
│  两大创新组件：                                          │
│  ├── LWC (Learnable Weight Clipping)                    │
│  │   └── 优化裁剪阈值 γ, β，使权重分布更紧凑              │
│  └── LET (Learnable Equivalent Transformation)          │
│      └── 学习缩放因子 s 和偏移量 δ，将量化难度从           │
│          激活转移到权重                                    │
└─────────────────────────────────────────────────────────┘
```

### 2.3 核心优势

| 方面 | QAT | PTQ (GPTQ/AWQ) | OmniQuant |
|------|-----|----------------|-----------|
| **性能** | 最优 | 低比特下差 | 接近 QAT |
| **训练时间** | 数百 GPU 小时 | 1 小时 | 1-16 小时 |
| **数据需求** | 100K 样本 | 128 样本 | 128 样本 |
| **可微优化** | 全模型 | 无/有限 | 仅量化参数 |
| **推理开销** | 无 | 无 | 无（参数可融合） |

---

## 三、技术架构

### 3.1 Block-wise 量化误差最小化（Figure 3）

**优化目标**：

```
min_{Θ1, Θ2} ||F(W, X) - F(Qw(W; Θ1, Θ2), Qa(X, Θ2))||
```

其中：
- `F`：Transformer Block 的映射函数
- `W, X`：全精度权重和激活
- `Qw, Qa`：权重和激活量化器
- `Θ1 = {γ, β}`：LWC 的裁剪强度参数
- `Θ2 = {δ, s, s_a}`：LET 的缩放和偏移参数

**优化策略**：
- 逐层优化（非全模型联合优化）
- 使用 SGD 算法
- 128 个样本，20 个 epoch（W2A16 用 40 epoch）
- 单个 A100-40G GPU

### 3.2 Learnable Weight Clipping (LWC)

**量化公式**：

```
W_q = clamp(⌊W/h⌉ + z, 0, 2^N - 1)
h = (γ·max(W) - β·min(W)) / (2^N - 1)
z = -⌊β·min(W) / h⌉
```

**关键设计**：
- `γ ∈ [0,1]`：上界裁剪强度（通过 Sigmoid 初始化）
- `β ∈ [0,1]`：下界裁剪强度（通过 Sigmoid 初始化）
- 当 γ=1, β=1 时退化为标准 MinMax 量化
- **仅调整裁剪阈值**，不改变权重分布形状

**与 PACT/LSQ 的区别**：

| 方法 | 裁剪方式 | 效果 |
|------|----------|------|
| PACT | 直接学习阈值 α | 性能不稳定 |
| LSQ | 直接学习步长 s | 需要全模型训练 |
| **LWC** | 学习裁剪强度 γ, β | 继承 MinMax 优势，仅微调 |

### 3.3 Learnable Equivalent Transformation (LET)

#### 线性层等价变换

**数学等价**：

```
Y = XW + B = [(X-δ)⊘s] · [s⊙W] + [B+δW]
    = X̃ · W̃ + B̃
```

其中：
- `s ∈ R^{1×C_in}`：通道缩放因子
- `δ ∈ R^{1×C_in}`：通道偏移量
- `⊘`：逐元素除法
- `⊙`：逐元素乘法

**参数融合**：
- s, δ 可融合到前一层的 LayerNorm 或线性层
- 无额外推理开销

#### 注意力等价变换

**Q/K 缩放**：

```
P = Softmax(QK^T) = Softmax((Q⊘s_a)(s_a⊙K^T))
```

其中 `s_a ∈ R^{1×C_out}` 是亲和矩阵的缩放因子。

**设计选择**：
- V 矩阵不做显式变换（已被输出投影的逆变换改变）
- Softmax 输出保持全精度（长尾分布不适合均匀量化）

### 3.4 LET 应用范围

| 层类型 | 是否应用 LET | 原因 |
|--------|-------------|------|
| QKV 投影 | 是 | 激活存在异常值 |
| 注意力缩放 | 是 | 量化 Q/K 矩阵 |
| FFN 第一层 | 是 | 线性层 |
| FFN 第二层 | **否** | 非线性层后特征稀疏，梯度不稳定 |
| 输出投影 | 是 | V 矩阵已隐式变换 |

---

## 四、实验结果

### 4.1 实验配置

| 项目 | 配置 |
|------|------|
| **模型** | OPT (125M-66B), LLaMA-1 (7B-65B), LLaMA-2 (7B-70B), Falcon-180B, LLaMA-2-chat |
| **硬件** | NVIDIA A100-40G GPU（训练）, A100-80G（部署） |
| **校准数据** | WikiText2 128 条 × 2048 tokens |
| **评估** | WikiText2/C4/PTB 困惑度，PIQA/ARC/BoolQ/HellaSwag 准确率 |
| **基线** | RTN, GPTQ, AWQ, SmoothQuant, OS+, RPTQ, LLM-QAT |

### 4.2 Weight-only 量化结果（Table 1）

**WikiText2 困惑度（↓ 越低越好）**：

| 方法 | 比特 | LLaMA-7B | LLaMA-13B | LLaMA-65B | LLaMA-2-7B | LLaMA-2-70B |
|------|------|----------|-----------|-----------|------------|-------------|
| FP16 | W16A16 | 5.68 | 5.09 | 3.53 | 5.47 | 3.31 |
| RTN | W2A16 | 1.1e5 | 6.8e4 | 2.2e4 | 3.8e4 | 2.0e4 |
| GPTQ | W2A16 | 2.1e3 | 5.5e3 | 55.91 | 7.7e3 | 77.95 |
| **OmniQuant** | **W2A16** | **15.47** | **13.21** | **7.58** | **37.37** | **7.81** |
| RTN | W3A16 | 25.73 | 11.39 | 10.68 | 539.48 | 7.52 |
| GPTQ | W3A16 | 8.06 | 6.76 | 5.06 | 8.37 | 4.82 |
| AWQ | W3A16 | 11.88 | 7.45 | 5.21 | 24.00 | - |
| **OmniQuant** | **W3A16** | **6.49** | **5.68** | **4.04** | **6.58** | **3.92** |
| RTN | W4A16 | 6.43 | 5.55 | 3.87 | 6.11 | 3.67 |
| GPTQ | W4A16 | 6.13 | 5.40 | 3.83 | 5.83 | 3.58 |
| AWQ | W4A16 | 6.08 | 5.34 | 3.76 | 6.15 | - |
| **OmniQuant** | **W4A16** | **5.86** | **5.21** | **3.71** | **5.74** | **3.47** |

**关键发现**：
- **W2A16**：OmniQuant 相比 GPTQ 困惑度降低 2-3 个数量级
- **W3A16**：OmniQuant 一致优于 GPTQ 和 AWQ
- **W4A16**：OmniQuant 略优于现有方法
- 比特越低，OmniQuant 优势越明显

### 4.3 Weight-Activation 量化结果（Table 2）

**零样本任务平均准确率（%）**：

| 模型 | 方法 | W6A6 | W4A4 |
|------|------|------|------|
| LLaMA-7B | FP16 | 64.09 | 64.09 |
| LLaMA-7B | SmoothQuant | 62.81 | 38.41 |
| LLaMA-7B | OS+ | 61.13 | 48.43 |
| LLaMA-7B | LLM-QAT | - | 46.43 |
| LLaMA-7B | **OmniQuant** | **63.17** | **52.65** |
| LLaMA-13B | FP16 | 66.33 | 66.33 |
| LLaMA-13B | SmoothQuant | 64.43 | 49.36 |
| LLaMA-13B | OS+ | 64.92 | 49.86 |
| LLaMA-13B | **OmniQuant** | **64.95** | **54.37** |
| LLaMA-65B | FP16 | 71.04 | 71.04 |
| LLaMA-65B | SmoothQuant | 69.80 | 47.71 |
| LLaMA-65B | OS+ | 68.76 | 52.52 |
| LLaMA-65B | **OmniQuant** | **70.28** | **59.22** |

**关键发现**：
- **W6A6**：OmniQuant 接近全精度性能
- **W4A4**：OmniQuant 显著优于所有 PTQ 方法（+4.99% ~ +11.80%）
- 在 LLaMA-7B 上甚至超越 QAT 方法 LLM-QAT（+6.22%）

### 4.4 指令微调模型量化（Figure 4）

**Vicuna-Bench 胜率（W3A16g128）**：

| 模型 | OmniQuant vs RTN | OmniQuant vs AWQ |
|------|------------------|------------------|
| LLaMA-2-7B-chat | 80.3% | 50% |
| LLaMA-2-13B-chat | 显著优于 | 优于 |

**关键发现**：OmniQuant 对指令微调模型同样有效。

### 4.5 实际部署加速（Table 3）

**MLC-LLM 部署结果（A100-80G）**：

| 模型 | 方法 | 权重内存 | 运行内存 | token/s |
|------|------|----------|----------|---------|
| LLaMA-7B | FP16 | 12.6G | 14.4G | 69.2 |
| LLaMA-7B | W4A16g128 | 3.8G | 5.7G | 134.2 |
| LLaMA-7B | W3A16g128 | 3.2G | 5.1G | 83.4 |
| LLaMA-7B | W2A16g128 | 2.2G | 4.1G | 83.9 |
| LLaMA-13B | FP16 | 24.3G | 27.1G | 52.5 |
| LLaMA-13B | W4A16g128 | 7.0G | 10.0G | 91.3 |
| LLaMA-13B | W2A16g128 | 4.0G | 7.5G | 92.6 |
| LLaMA-65B | FP16 | OOM | - | - |
| LLaMA-65B | W4A16g128 | 33.0G | 41.0G | 24.3 |
| LLaMA-65B | W2A16g128 | 18.0G | 25.6G | 24.8 |

**关键发现**：
- W4A16g128 几乎将推理速度翻倍
- W2A16g128 内存减半，速度接近 W4A16
- 无额外推理操作（参数可融合）

### 4.6 消融实验

#### LWC vs 其他裁剪方法

| 方法 | LLaMA-7B W2A16 PPL |
|------|---------------------|
| MinMax (γ=1, β=1) | 极高 |
| PACT (直接学 α) | 不稳定 |
| LSQ (直接学 s) | 需要全模型训练 |
| **LWC (学 γ, β)** | **15.47** |

#### LET 的作用

| 组件 | W4A16 | W4A4 |
|------|-------|------|
| 仅 LWC | 好 | 差 |
| LWC + LET | 更好 | 显著提升 |

**关键发现**：LET 对 W&A 量化至关重要，将量化难度从激活转移到权重。

---

## 五、核心技术详解

### 5.1 为什么 LWC 优于直接裁剪？

**传统方法（PACT）**：
```
W_q = clamp(W, 0, α) / α * (2^N - 1)
```
- 直接学习 α，可能陷入局部最优
- 不继承 MinMax 量化的优势

**OmniQuant LWC**：
```
W_q = clamp(⌊W/h⌉ + z, 0, 2^N - 1)
h = (γ·max(W) - β·min(W)) / (2^N - 1)
```
- 继承 MinMax 量化（γ=1, β=1 时）
- 仅需微调裁剪强度
- 优化空间小，更容易收敛

### 5.2 为什么 LET 有效？

**激活异常值问题**：

```
原始激活：[0.1, 0.2, 0.15, 100.0, 0.1, 0.2]
                                ↑ 异常值
量化后：  [0, 0, 0, 7, 0, 0]  ← 大量信息丢失
```

**LET 解决方案**：

```
应用缩放 s = [1, 1, 1, 50, 1, 1]
变换后激活：[0.1, 0.2, 0.15, 2.0, 0.1, 0.2]
                            ↑ 范围缩小
量化后：  [0, 0, 0, 3, 0, 0]  ← 信息保留更多
```

**关键**：缩放因子 s 被融合到权重中，无推理开销。

### 5.3 Block-wise 优化的优势

| 方面 | 全模型优化 | Block-wise 优化 |
|------|-----------|-----------------|
| **内存** | 需要存储所有层的梯度 | 仅需当前 Block |
| **收敛** | 容易陷入局部最优 | 逐层保证最优 |
| **数据需求** | 需要大量数据 | 128 样本足够 |
| **时间** | 数百 GPU 小时 | 1-16 小时 |

---

## 六、相关工作对比

### 6.1 Weight-only 量化方法

| 方法 | 优化方式 | W2A16 | W3A16 | W4A16 |
|------|----------|-------|-------|-------|
| RTN | 无 | 极差 | 差 | 一般 |
| GPTQ | Hessian 补偿 | 差 | 一般 | 好 |
| AWQ | 网格搜索 | 差 | 一般 | 好 |
| **OmniQuant** | **可微优化** | **好** | **好** | **最好** |

### 6.2 Weight-Activation 量化方法

| 方法 | 优化方式 | W8A8 | W6A6 | W4A4 |
|------|----------|------|------|------|
| SmoothQuant | 手工迁移强度 | 好 | 一般 | 差 |
| OS+ | 手工缩放+偏移 | 好 | 一般 | 差 |
| RPTQ | 分组激活量化 | 好 | 好 | 需要特殊格式 |
| LLM-QAT | 全模型训练 | - | - | 好但成本高 |
| **OmniQuant** | **可微优化** | **好** | **好** | **最好** |

### 6.3 与其他方法的互补性

OmniQuant 可以与以下方法结合：
- **QLoRA**：先 OmniQuant 量化，再 LoRA 微调
- **INT2.1**：先 OmniQuant 量化，再参数高效微调
- **知识蒸馏**：用 OmniQuant 量化作为起点

---

## 七、总结

### 核心贡献

1. **OmniQuant 框架**：
   - 冻结全精度权重，仅学习量化参数
   - 达到 QAT 性能，保持 PTQ 效率
   - 128 样本，1-16 小时，单 GPU

2. **LWC (Learnable Weight Clipping)**：
   - 学习裁剪强度 γ, β（而非直接学阈值）
   - 继承 MinMax 量化优势
   - 使权重分布更紧凑

3. **LET (Learnable Equivalent Transformation)**：
   - 学习缩放因子 s 和偏移量 δ
   - 将量化难度从激活转移到权重
   - 扩展到注意力机制（Q/K 缩放）

4. **显著性能提升**：

| 设置 | 改善 |
|------|------|
| W2A16 | 困惑度降低 2-3 个数量级 |
| W4A4 | 准确率 +4.99% ~ +11.80% |
| vs LLM-QAT | +6.22%（LLaMA-7B W4A4） |

5. **零推理开销**：
   - LWC 裁剪阈值可融合到量化参数
   - LET 缩放因子可融合到前一层
   - 无额外计算或内存访问

### 关键数据

| 指标 | 数值 |
|------|------|
| **训练时间** | 1-16 小时（7B-70B） |
| **训练数据** | 128 样本 |
| **训练硬件** | 单个 A100-40G GPU |
| **W2A16 困惑度** | LLaMA-65B: 7.58（vs GPTQ 55.91） |
| **W4A4 准确率** | LLaMA-7B: 52.65%（vs LLM-QAT 46.43%） |
| **推理加速** | W4A16g128: ~2×（MLC-LLM） |

### 技术影响

OmniQuant 证明了：
- PTQ 可以达到接近 QAT 的性能
- 可微优化是量化参数学习的有效途径
- 低比特量化（W2A16, W4A4）在合理优化下是可行的
- Block-wise 优化是大规模模型量化的高效策略

### 局限性

1. **W4A4 硬件支持**：当前缺乏现成的硬件支持
2. **INT3/INT2 部署**：MLC-LLM 对 INT3/INT2 支持尚不完善
3. **FFN 第二层**：LET 不适用于非线性层后的线性层
4. **极端低比特**：W1A16 等更极端设置未探索

### 未来方向

1. **硬件协同**：与芯片厂商合作支持 W4A4
2. **更多模型**：多模态 LLM、MoE 模型
3. **更低比特**：W1A16、W2A4
4. **端到端优化**：与推理框架深度集成

---

## 八、参考资源

- **论文**: https://arxiv.org/abs/2308.13137v3
- **代码**: https://github.com/OpenGVLab/OmniQuant
- **相关论文**:
  - SmoothQuant: https://arxiv.org/abs/2211.10438
  - GPTQ: https://arxiv.org/abs/2210.17323
  - AWQ: https://arxiv.org/abs/2306.00978
  - OS+: https://arxiv.org/abs/2302.09006
  - RPTQ: https://arxiv.org/abs/2306.17251
  - LLM-QAT: https://arxiv.org/abs/2305.17888
  - QuIP: https://arxiv.org/abs/2307.13304
