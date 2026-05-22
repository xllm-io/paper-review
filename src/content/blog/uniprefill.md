---
title: "UniPrefill: 通用长上下文 Prefill 加速框架"
description: "UniPrefill 提出通用的长上下文 Prefill 加速框架。"
date: 2026-04-08
tags:
  - "Deep Learning"
  - "Attention"
draft: false
---
# UniPrefill: 通用长上下文 Prefill 加速框架

## 一、论文概述

| 项目 | 内容 |
|------|------|
| **标题** | UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification |
| **作者** | Qihang Fan, Huaibo Huang, Zhiying Wu, Bingning Wang, Ran He |
| **机构** | 中国科学院自动化研究所 (CASIA)、中国科学院大学 (UCAS)、腾讯微信 |
| **论文** | https://arxiv.org/abs/2605.06221 |
| **代码** | https://github.com/qhfan/UniPrefill.git |
| **发布** | 2026年5月7日 |
| **许可** | arXiv.org perpetual non-exclusive license |
| **领域** | 计算语言学 (cs.CL) |

## 二、核心思想

### 问题定义

大语言模型 (LLM) 在处理长上下文时面临两大核心挑战：

1. **计算复杂度瓶颈**：标准 Softmax Self-Attention 的计算复杂度为 O(N²)，随序列长度二次增长，在处理 100K+ token 时计算开销极其昂贵

2. **现有加速方法的局限性**：
   - **架构耦合**：稀疏注意力方法（如 MInference、FlexPrefill）仅在全注意力层有效，对新兴的混合架构效果大幅下降
   - **不兼容连续批处理**：现有方法难以集成到 vLLM 等现代推理引擎的连续批处理调度器中
   - **仅加速注意力层**：稀疏注意力方法仅减少注意力 FLOPs，无法减少 FFN 和 GEMM 计算

### 解决方案概述

UniPrefill 提出了一种**架构无关的 prefill 加速框架**，核心思想是：

> **在全注意力层估计 token 重要性，然后将稀疏性传播到所有后续层（包括线性注意力、滑动窗口注意力、FFN），从而同时减少注意力 FLOPs 和 GEMM FLOPs。**

关键创新点：
1. 使用 **Top-p 选择策略**（而非 Top-k），自适应于注意力分布
2. **稀疏性传播**：单次丢弃决策减少所有后续层的计算
3. **生产级集成**：实现为 vLLM 的连续批处理算子，支持张量并行

## 三、技术架构

### 整体框架

```
输入序列 x = [x₁, ..., xₙ]
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid LLM Blocks                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Block b: Full Attention Layer                       │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │ 1. Token Importance Estimation              │   │   │
│  │  │    - 计算最后 n 个 query 的注意力分数       │   │   │
│  │  │    - 按块 G 聚合分数                        │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │ 2. Top-p Token Selection                    │   │   │
│  │  │    - 按分数降序排列                          │   │   │
│  │  │    - 保留累计分数 ≥ p 的最小集合            │   │   │
│  │  │    - 始终保留 attention sinks 和 query 窗口  │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 3. Sparsity Propagation                           │   │
│  │    - 被丢弃的 token 排除在所有后续子层之外       │   │
│  │    - 线性注意力、滑动窗口注意力、FFN             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 4. Fused Kernel Pipeline (Triton)                 │   │
│  │    - 部分 GEMM → online softmax → block reduce    │   │
│  │    - → top-p → keep mask                          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
    输出 hₙ⁽ᴸ⁾
```

### 核心公式

**1. Token 重要性估计 (Token Importance Estimation)**

单 token 重要性分数：
$$s_i^{(b)} = \frac{1}{n}\sum_{j=N-n+1}^{N} A_{j,i}^{(b)}$$

其中 $A_{j,i}^{(b)} = \text{softmax}_i\left(\mathbf{q}_j^{(b)}{\mathbf{K}^{(b)}}^{\top}/\sqrt{d_k}\right)$ 是注意力权重。

块级重要性分数：
$$\bar{s}_g^{(b)} = \frac{1}{G}\sum_{i \in \mathcal{B}_g} \frac{1}{n}\sum_{j=N-n+1}^{N} A_{j,i}^{(b)}$$

其中 $\mathcal{B}_g = \{(g-1)G+1, \ldots, \min(gG, N)\}$ 是大小为 G 的非重叠块。

**2. Top-p Token 选择**

保留最小块集合：
$$\mathcal{S}^{(b)} = \{\pi(1), \ldots, \pi(k^*)\}, \quad k^* = \min k \text{ s.t. } \frac{\sum_{j=1}^{k}\bar{s}_{\pi(j)}^{(b)}}{\sum_{g}\bar{s}_{g}^{(b)}} \geqslant p$$

其中 $\pi$ 是按分数降序排列的置换。

**3. 误差边界 (Error Bound)**

对任意保留位置 j 的扰动：
$$\|\Delta\mathbf{h}_j^{(b,1)}\| \leqslant \left(\sum_{i \in \bar{\mathcal{S}}^{(b)}} A_{j,i}^{(b)}\right) \cdot V_{\max}^{(b)} \leqslant (1-p) \cdot V_{\max}^{(b)}$$

其中 $V_{\max}^{(b)} = \max_i \|\mathbf{v}_i^{(b)}\|$。设置 p=0.99 保证最多丢弃 1% 的注意力质量。

**4. 稀疏性传播 (Sparsity Propagation)**

丢弃的 token 在 block 内所有后续子层中被排除：
$$\mathbf{H}_{\mathcal{S}}^{(b,m+1)} = f_m\left(\mathbf{H}_{\mathcal{S}}^{(b,m)}\right), \quad m = 1, \ldots, M_b$$

下一个 block 重新组合完整序列：
$$\mathbf{H}_i^{(b+1,0)} = \begin{cases} \mathbf{H}_i^{(b,M_b+1)} & i \in \mathcal{S}^{(b)} \\ \mathbf{H}_i^{(b,0)} & i \in \bar{\mathcal{S}}^{(b)} \end{cases}$$

**5. FLOPs 分析**

单次丢弃的 FLOPs 节省：
$$\Delta\text{FLOPs}^{(\ell_1)} = (1-\rho) \cdot (L-\ell_1) \cdot \mathcal{O}(Nd^2)$$

与稀疏注意力的比较：
$$\frac{\Delta\text{FLOPs}_{\text{UniPrefill}}}{\Delta\text{FLOPs}_{\text{SparseAttn}}} = \frac{(L-\ell_1) \cdot Nd^2}{N^2 d_k} \xrightarrow{N \to \infty} \infty$$

**6. 误差传播 (Error Propagation)**

假设每个子层 $f_m$ 是 $L_m$-Lipschitz 的：
$$\|\Delta\mathbf{h}_j^{(b,M_b+1)}\| \leqslant (1-p) \cdot V_{\max}^{(b)} \cdot \prod_{m=1}^{M_b} L_m$$

Layer normalization 和残差连接约束 $\prod_m L_m$，防止误差无限放大。

**7. 融合核流水线 (Fused Kernel Pipeline)**

$$\mathbf{S} = \mathbf{Q}_{[N-n:N]}\mathbf{K}^{\top} \xrightarrow{\text{online softmax}} \mathbf{o} \in \mathbb{R}^N \xrightarrow{\text{block reduce}} \mathbf{b} \in \mathbb{R}^{\lceil N/G \rceil} \xrightarrow{\text{top-}p} \mathcal{M} \in \{0,1\}^N$$

**8. Top-p 核的 GPU 实现**

使用 IEEE-754 位操作编码 (score, index) 对：
$$\varphi(x) = \begin{cases} \text{bits}(x) \oplus \texttt{0x80000000} & x \geqslant 0 \\ \text{bits}(x) \oplus \texttt{0xFFFFFFFF} & x < 0 \end{cases}$$
$$\texttt{packed} = (\varphi(b_g) \ll 32) \mid g$$

**9. 张量并行同步**

$$\mathbf{b} = \sum_{t=1}^{T} \mathbf{b}^{(t)}$$

**10. vLLM 集成 - KV Cache Slot 映射**

$$\text{slot}_i^{(\ell')} = \text{block\_table}^{(\ell')}\left[r_i, \lfloor p_i/B \rfloor\right] \cdot B + (p_i \bmod B)$$

**11. Decode 阶段的序列长度修正**

$$\text{seqused}_r^{(\ell')} = s_r^{(\ell^-)} + \Delta_r, \quad \Delta_r = \text{kv\_len}_r - \text{orig\_len}_r$$

### 模型组件

| 组件 | 说明 | 关键参数 |
|------|------|----------|
| Token Importance Estimator | 在全注意力层估计每个 token 的重要性 | n=128 (query 窗口), G=64 (块大小) |
| Top-p Selector | 按重要性分数选择保留的 token 块 | p=0.99/0.98 (阈值) |
| Sparsity Propagator | 将稀疏性传播到所有后续子层 | - |
| Fused Kernel Pipeline | 4 个 Triton kernel 实现高效计算 | - |
| vLLM Scheduler Extension | 扩展 vLLM 调度器支持连续批处理 | TP=8 |

### 训练流程

UniPrefill 是一个**推理时加速框架**，不需要训练。它直接应用于预训练好的模型，不修改模型权重。

## 四、核心创新

| 创新点 | 说明 | 理论/实验依据 |
|--------|------|---------------|
| **架构无关性** | 通过在全注意力层做决策并传播稀疏性，适用于任何混合架构 | 在纯全注意力、线性/全注意力混合、滑动窗口/全注意力混合三种架构上均有效 |
| **Top-p 选择策略** | 自适应于注意力分布，提供统一的误差边界 | 当注意力集中时保留较少 token，分散时保留较多；误差边界 $\leqslant (1-p) \cdot V_{\max}$ |
| **稀疏性传播** | 单次丢弃决策减少所有后续层的 FLOPs（包括 GEMM） | FLOPs 节省随 $(L-\ell_1)$ 线性增长；与稀疏注意力的比值在长上下文下趋于无穷 |
| **生产级集成** | 实现为 vLLM 的连续批处理算子，支持张量并行 | 在 vLLM v0.16.0 上实现，TP=8 实验验证 |
| **融合 Triton Kernel** | 4 个 kernel 无缝衔接，无中间数据物化 | 使用 IEEE-754 位操作实现 GPU 上的高效排序 |

## 五、代码实现分析

### 项目结构

```
UniPrefill/
├── README.md
├── vllm/                    # vLLM 集成代码
│   ├── attention/           # 注意力层修改
│   ├── worker/              # Worker 进程修改
│   └── model_executor/      # 模型执行器修改
├── kernels/                 # Triton kernel 实现
│   ├── importance_estimation.py  # Token 重要性估计
│   ├── top_p_selection.py        # Top-p 选择
│   └── sparsity_propagation.py   # 稀疏性传播
└── benchmarks/              # 基准测试
    └── ruler_eval.py        # RULER 评估
```

### 关键实现细节

1. **融合 Triton Kernel**：4 个 kernel 无缝衔接
   - 部分 GEMM kernel：计算 Q-K 注意力分数
   - Softmax kernel：两遍 online softmax 算法
   - Block-reduce kernel：空间和头维度聚合
   - Top-p kernel：GPU 上的排序和阈值化

2. **vLLM 集成**：
   - 修改调度器支持 prefill-decode 协处理
   - 更新 query_start_loc, seq_lens, num_actual_tokens
   - 重计算物理 KV cache slot 映射
   - 维护每请求的丢弃历史

3. **张量并行支持**：
   - 通过 all-reduce 同步跨 rank 的分数
   - 确保所有 rank 做出一致的丢弃决策

## 六、实验结果

### 基准测试

**测试模型**：

| 模型 | 架构类型 | 比例 | Top-p 阈值 |
|------|----------|------|-----------|
| LLaMA-3.1-8B-Instruct | 纯全注意力 | - | 0.99 |
| Qwen3-Next-80B-A3B | 线性/全注意力混合 | 3:1 | 0.99 |
| Gemma-3-12B | 滑动窗口/全注意力混合 | 5:1 | 0.98 |

**Table 1: RULER 基准精度 vs TTFT 加速**

| 模型 | 方法 | 4K | 8K | 16K | 32K | 64K | 128K | Avg | 128K 加速 |
|------|------|----|----|-----|-----|-----|------|-----|----------|
| **LLaMA-3.1-8B** | Baseline | 97.36 | 95.98 | 94.62 | 91.02 | 86.29 | 76.89 | 90.36 | 1.00x |
| | MInference | 96.71 | 95.78 | 95.51 | 90.76 | 87.12 | 78.21 | 90.68 | 1.34x |
| | FlexPrefill | 96.34 | 95.12 | 94.83 | 88.96 | 84.31 | 78.13 | 89.62 | 1.46x |
| | **UniPrefill** | **96.53** | **95.83** | **95.41** | **89.77** | **85.28** | **79.87** | **90.45** | **2.26x** |
| **Qwen3-Next-80B** | Baseline | 96.83 | 95.67 | 95.07 | 94.38 | 94.51 | 92.09 | 94.76 | 1.00x |
| | MInference | 96.62 | 94.38 | 94.49 | 94.27 | 94.28 | 91.81 | 94.31 | 1.05x |
| | FlexPrefill | 96.36 | 95.03 | 94.17 | 93.91 | 92.89 | 91.44 | 93.97 | 1.08x |
| | **UniPrefill** | **96.67** | **94.49** | **94.29** | **93.63** | **93.13** | **91.41** | **93.94** | **1.68x** |
| **Gemma-3-12B** | Baseline | 94.01 | 89.12 | 85.98 | 80.76 | 68.89 | 61.22 | 79.99 | 1.00x |
| | MInference | 93.56 | 89.51 | 86.01 | 80.04 | 67.09 | 59.31 | 79.25 | 1.03x |
| | FlexPrefill | 93.63 | 89.16 | 85.49 | 79.23 | 65.69 | 58.63 | 78.64 | 1.04x |
| | **UniPrefill** | **93.18** | **89.76** | **86.47** | **79.08** | **66.32** | **58.38** | **78.87** | **1.49x** |

**Table 2: vLLM 吞吐量 (TP=8, tokens/s)**

| 模型 | BSZ | 标准 Prefill 128K | UniPrefill 128K | 提升 |
|------|-----|------------------|----------------|------|
| LLaMA-3.1-8B | 1 | 21,013 | 43,672 | **+107%** |
| | 4 | 21,054 | 43,698 | **+108%** |
| | 16 | 21,062 | 44,042 | **+109%** |
| Qwen3-Next-80B | 1 | 33,512 | 49,732 | **+48%** |
| | 4 | 33,364 | 52,442 | **+57%** |
| | 16 | 33,489 | 56,398 | **+68%** |
| Gemma-3-12B | 1 | 18,103 | 25,673 | **+42%** |
| | 4 | 18,403 | 25,932 | **+41%** |
| | 16 | 18,513 | 26,231 | **+42%** |

### 消融实验

**Block Size (G) 对 LLaMA-3.1-8B 的影响**：

| G | 平均精度 | 128K 加速 |
|---|---------|----------|
| 32 | 89.88 | +121% |
| 64 | 90.45 | +109% |
| 128 | 88.57 | +96% |

**Last n 对 LLaMA-3.1-8B 的影响**：

| n | 平均精度 |
|---|---------|
| 32 | 87.77 |
| 128 | 90.45 |
| 512 | 90.49 |

**随机种子鲁棒性**：

| Seed | 4K | 8K | 16K | 32K | 64K | 128K |
|------|----|----|-----|-----|-----|------|
| 0 | 96.53 | 95.83 | 95.41 | 89.77 | 85.28 | 79.87 |
| 321 | 96.53 | 95.83 | 95.41 | 89.77 | 85.28 | 79.87 |
| 3467 | 96.53 | 95.83 | 95.41 | 89.77 | 85.28 | 79.87 |

### 与现有方法对比

| 方法 | 架构兼容性 | GEMM 加速 | 连续批处理 | 128K 加速 (LLaMA) | 精度保持 |
|------|-----------|----------|-----------|------------------|---------|
| MInference | 仅全注意力 | 否 | 否 | 1.34x | 好 |
| FlexPrefill | 仅全注意力 | 否 | 否 | 1.46x | 好 |
| XAttention | 仅全注意力 | 否 | 否 | 1.38x | 好 |
| ProxyAttn | 仅全注意力 | 否 | 否 | 1.79x | 好 |
| LazyLLM | 通用 | 是 | 否 | 2.51x | 差 (-21.86) |
| SlimInfer | 通用 | 是 | 否 | 2.07x | 差 (-21.49) |
| **UniPrefill** | **通用** | **是** | **是** | **2.26x** | **好 (+0.09)** |

## 七、相关工作

### Hybrid LLM Architectures

- **线性/全注意力混合**：Mamba, Mamba2, GLA, ReCT, DeltaNet
  - 用线性递归机制替换部分注意力层，复杂度从 O(N²) 降至 O(N)
- **滑动窗口/全注意力混合**：Gemma-3, Mistral-7B
  - 限制大多数注意力层到固定局部上下文窗口

### Sparse Attention for Prefill Acceleration

- **MInference**：识别静态/动态稀疏模式（vertical, slash, block-sparse）
- **FlexPrefill**：自适应稀疏注意力
- **XAttention**：跨注意力稀疏模式
- **ProxyAttn**：代理注意力加速

**共同局限**：
1. 加速紧密耦合到注意力操作本身
2. 无法减少 FFN 和 GEMM 计算
3. 不兼容连续批处理

### Relationship to SnapKV

| 特性 | SnapKV | UniPrefill |
|------|--------|------------|
| 目标 | 压缩 KV cache（decode 阶段） | 加速 prefill 阶段 |
| 时机 | prefill 完成后选择 | prefill 过程中选择 |
| 影响范围 | 仅减少 decode 内存 | 减少所有后续层的 FLOPs |
| 节省量 | O(N·d_kv) per layer | (1-ρ)·M_b·O(Nd²) per block |

## 八、总结

### 核心贡献

1. **架构无关的加速框架**：通过在全注意力层做 token 丢弃并传播稀疏性，同时减少注意力和 GEMM FLOPs，适用于纯全注意力和各种混合架构

2. **生产级系统集成**：实现为连续批处理算子，深度集成到 vLLM v0.16.0，支持 prefill-decode 协处理和张量并行

3. **最佳精度-效率权衡**：在 RULER 基准上实现最高 2.26x TTFT 加速，精度损失可忽略（平均 < 1.5 分）

### 技术影响

- **实用性**：直接可用于生产环境，无需修改模型权重
- **通用性**：适用于任何包含全注意力层的模型架构
- **可扩展性**：加速效果随上下文长度和批量大小增加而增强
- **示范性**：展示了 token 级稀疏性在推理加速中的巨大潜力

### 局限性

1. **仅针对 prefill 阶段**：未涉及 decoding 阶段加速
2. **未涉及训练效率**：仅关注推理时优化
3. **模型验证有限**：仅在 3 个模型上验证
4. **硬件平台单一**：仅在 CUDA 12.8 上测试

### 未来方向

1. 将框架扩展到 decoding 阶段加速
2. 探索训练时的稀疏性利用
3. 在更多架构和任务上验证
4. 适配不同硬件平台（AMD、Intel 等）

## 九、参考资源

- **论文**: https://arxiv.org/abs/2605.06221
- **代码**: https://github.com/qhfan/UniPrefill.git
- **RULER 基准**: https://github.com/hsiehjupyter/RULER
- **vLLM**: https://github.com/vllm-project/vllm
