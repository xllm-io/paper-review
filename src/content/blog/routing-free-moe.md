---
title: "Routing-Free Mixture-of-Experts"
description: "Routing-Free MoE 提出无需路由的混合专家模型架构。"
date: 2026-04-05
tags:
  - "Deep Learning"
  - "Sparse"
draft: false
---
# Routing-Free Mixture-of-Experts

## 一、论文概述

| 项目 | 内容 |
|------|------|
| **标题** | Routing-Free Mixture-of-Experts |
| **作者** | Yilun Liu*, Jinru Han*, Sikuan Yan, Volker Tresp, Yunpu Ma |
| **机构** | Ludwig Maximilian University of Munich, University of California, Los Angeles, Munich Center for Machine Learning |
| **论文** | https://arxiv.org/abs/2604.00801 |
| **代码** | https://github.com/liuyilun2000/RoutingFreeMoE/tree/release |
| **发布** | 2026年4月1日 |

## 二、核心思想

Routing-Free MoE 提出了一种完全去中心化的 MoE 架构，消除了传统 MoE 中的集中式路由机制（包括外部路由器、Softmax、TopK 和负载均衡），让每个专家独立决定自己的激活状态。

### 传统 MoE 的问题

标准 MoE 依赖集中式路由机制，存在以下结构性限制：

1. **路由器容量瓶颈**：路由器参数量远小于专家本身，却需要压缩所有专家的激活偏好
2. **TopK 刚性约束**：固定的稀疏比率 K/N 无视输入复杂度差异
3. **Softmax 竞争性归一化**：丢弃绝对激活幅度信息
4. **负载均衡互斥**：Token-Choice 和 Expert-Choice 难以兼顾

### 解决方案概述

Routing-Free MoE 的核心创新：

1. **专家自主激活**：每个专家通过内部置信度分数独立决定是否激活
2. **去除集中式组件**：消除外部路由器、Softmax、TopK
3. **统一自适应负载均衡**：同时优化 token-balancing 和 expert-balancing
4. **自适应稀疏控制**：通过可配置阈值和自适应系数实现

## 三、技术架构

### 架构设计

#### 标准 MoE

```
h = Σ(G(x)_i * E_i(x)) + x
G(x) = Softmax(TopK(xG, K))
```

其中 G(·) 是路由函数，E_i(·) 是第 i 个专家。

#### AoE（Autonomy-of-Experts）

```
FFN(x) = [σ(xA_gate B_gate) ⊙ (xW_up)] W_down
G(x) = Softmax(TopK(||xA_gate||_2, K))
```

- 使用低秩投影 A_gate 从专家内部产生激活分数
- 但仍保留 TopK 和 Softmax

#### ReMoE

```
G(x) = ReLU(xG)
```

- 用 ReLU 替代 Softmax 和 TopK
- 但仍保留外部路由器

#### Routing-Free MoE（本文）

```
G_i(x) = ReLU(||xA_gate,i||_2 - b_i)
f_i(x) = 1{G_i(x) - θ ≥ 0}
```

- **A_gate,i**：每个专家内部的低秩投影矩阵
- **b_i**：每个专家的可学习偏置项
- **θ**：全局后激活阈值（可配置超参数）
- **f_i(x)**：二值激活决策

### 关键组件

| 组件 | 作用 | 特点 |
|------|------|------|
| **A_gate** | 低秩投影，产生内部激活分数 | 秩 r=32（默认） |
| **b_i** | 每个专家的可学习偏置 | 控制单个专家激活率 |
| **θ** | 全局后激活阈值 | 推理时可调节稀疏度 |
| **ReLU** | 激活函数 | 自然产生稀疏性 |

### 训练框架

#### 统一负载均衡损失

```
L_LB = μ * L_EB + (1-μ) * L_TB
```

**Expert-Balancing Loss (L_EB)**：
```
L_EB = (1/|E||B|) * Σ_i Σ_x f_i(x) * G_i(x)
```
鼓励 token 在专家间均匀分布。

**Token-Balancing Loss (L_TB)**：
```
L_TB = (1/|E||B|) * Σ_x Σ_i f_i(x) * G_i(x)
```
鼓励每个 token 激活均匀数量的专家。

**可配置插值参数 μ**：
- μ=1：纯 expert-balancing
- μ=0：纯 token-balancing
- μ=0.5：平衡两者（最佳）

#### 自适应稀疏控制

```
L = L_LM + λ_t * L_LB
λ_{t+1} = λ_t * (1+η)^sign(ρ_t - ρ_∞)
```

- **ρ_t**：当前激活密度
- **ρ_∞**：目标激活密度
- **η**：响应步长
- **λ_t**：自适应系数

当密度超过目标时，λ 增加以增强负载均衡压力；低于目标时，λ 减少以允许更多激活。

#### 训练初始化

- 偏置 b_i 初始化为 e^{-6}，允许所有专家在早期被激活
- 随着训练进行，λ 增加，稀疏正则化逐渐增强
- 专家在早期建立初始特化，避免专家坍塌

## 四、核心创新

| 创新点 | 说明 | 理论/实验依据 |
|--------|------|---------------|
| **去除集中式路由** | 消除外部路由器、Softmax、TopK | 消除信息瓶颈 |
| **专家自主激活** | 每个专家独立决定激活 | 底层自组织涌现 |
| **统一负载均衡** | 同时优化 token 和 expert balancing | μ=0.5 最佳 |
| **自适应稀疏控制** | 自动调节激活密度至目标 | 无需手动调参 |
| **全局密度目标** | 放弃逐层强制稀疏 | PPL 从 39.44 降至 28.74 |
| **可配置阈值 θ** | 推理时调节稀疏度 | 灵活部署控制 |

## 五、实验结果

### 实验配置

- **模型规模**：S (92M), M (290M), L (808M)
- **训练数据**：OpenWebText，1 epoch
- **评估基准**：9 个 benchmark（PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA, QQP, QNLI, SST-2）
- **基线**：Standard MoE, AoE, ReMoE

### 核心结果

#### 主要结果对比

| 模型 | 规模 | FLOPs | PPL↓ | 平均准确率 |
|------|------|-------|------|-----------|
| MoE | 92M | 90.9M | 31.22 | 38.96% |
| AoE | 94M | 88.6M | 30.00 | 38.82% |
| ReMoE | 92M | 90.9M | 29.60 | 39.10% |
| **RFMoE** | 95M | 91.1M | **27.42** | **39.77%** |
| MoE | 290M | 248M | 25.00 | 39.64% |
| **RFMoE** | 307M | 249M | **22.08** | **40.40%** |
| MoE | 808M | 608M | 24.58 | 40.00% |
| **RFMoE** | 871M | 613M | **19.97** | **40.76%** |

**关键发现**：
- Routing-Free MoE 在所有规模下 consistently 优于基线
- 增益不随规模减小，表明良好的可扩展性
- FLOPs 匹配在 ~1% 以内

#### 架构消融实验

| 配置 | PPL↓ |
|------|------|
| Standard MoE | 31.22 |
| w/o router (AoE, r=16) | 30.00 |
| w/o router (AoE, r=32) | 30.31 |
| w/o TopK&Softmax (ReMoE) | 29.60 |
| **Routing-Free MoE (r=16)** | **28.73** |
| **Routing-Free MoE (r=32)** | **28.33** |

#### 低秩投影维度影响

| r | PPL↓ |
|---|------|
| 8 | 29.16 |
| 16 | 28.74 |
| 32 | 28.34 |
| 64 | 28.24 |

增加 r 持续改善 PPL，但收益递减。默认 r=32。

#### 负载均衡插值 μ 的影响

| μ | PPL↓ | 吞吐量↑ |
|---|------|---------|
| 0.0 (纯 TB) | 28.41 | 645.7 |
| 0.2 | 28.35 | 648.3 |
| **0.5 (平衡)** | **28.34** | **662.3** |
| 0.8 | 28.38 | 643.9 |
| 1.0 (纯 EB) | 28.43 | 648.8 |

**关键发现**：μ=0.5 在 PPL 和吞吐量上都最佳。

#### 全局 vs 逐层密度目标

| 方式 | PPL↓ |
|------|------|
| 逐层强制 | 39.44 |
| **全局目标** | **28.74** |

放松逐层稀疏偏置带来显著改善。

### 训练动态分析

1. **激活密度 ρ**：从 ~1 开始，随 λ 增长急剧下降至目标 ρ_∞，之后保持稳定
2. **自适应系数 λ**：初始预热阶段急剧上升，ρ 收敛后趋于平稳
3. **正则化项 λL_LB**：在 λ 指数增长时出现瞬时峰值，随后衰减至可忽略水平
4. **训练稳定性**：Routing-Free MoE 在更大 α 下保持稳定，而标准 MoE 在 α=2e-3 时崩溃

### 专家激活模式

Figure 6 展示了全局密度目标下各层专家激活的演化：

- 不同层自然发展出不同的激活密度
- 某些层激活更多专家（计算密集型层）
- 某些层使用稀疏表示即可
- 模型自组织形成更有效的功能对齐结构

### 热力图分析

Figure 7 展示了不同 μ 下的专家激活热力图：

- **μ=0.0（纯 TB）**：专家间激活概率差异大，少数专家被频繁激活
- **μ=1.0（纯 EB）**：专家负载更均匀，但对输入域分布过度敏感
- **μ=0.5（平衡）**：两个轴向都更均匀，跨输入更一致

## 六、相关工作

### MoE 基础

| 工作 | 贡献 |
|------|------|
| Jacobs et al. (1991) | 提出自适应局部专家混合 |
| Shazeer et al. (2017) | 将 MoE 扩展到深度网络 |
| Switch Transformers | 简单高效稀疏，扩展到万亿参数 |
| Mixtral | 前沿 MoE 模型 |
| DeepSeekMoE | 极致专家特化 |

### 路由机制

| 方法 | 特点 | 局限 |
|------|------|------|
| **标准路由** | xG + TopK + Softmax | 信息瓶颈，刚性约束 |
| **AoE** | 专家内部评分，去除路由器 | 保留 TopK + Softmax |
| **ReMoE** | ReLU 替代 Softmax/TopK | 保留外部路由器 |
| **Routing-Free** | 完全去除集中式组件 | 需要自适应负载均衡 |

### 负载均衡

| 策略 | 特点 | 局限 |
|------|------|------|
| **Token Choice** | 保证每 token 计算 | 不保证专家均衡 |
| **Expert Choice** | 保证专家负载均衡 | 可能次优匹配 |
| **Routing-Free** | 统一框架，可配置 μ | 需要训练自适应 |

## 七、总结

### 核心贡献

1. **Routing-Free MoE 架构**：
   - 消除外部路由器、Softmax、TopK 和刚性负载均衡
   - 每个专家独立决定激活，全局模式自底向上涌现

2. **统一自适应负载均衡框架**：
   - 同时优化 token-balancing 和 expert-balancing
   - 可配置插值参数 μ 灵活适应部署需求

3. **实验验证**：
   - 3 个规模，9 个 benchmark，consistently 优于基线
   - 更好的可扩展性和鲁棒性
   - PPL 从 24.58 降至 19.97（Scale L）

### 技术影响

Routing-Free MoE 证明了：
- 集中式路由并非 MoE 的必要组件
- 专家自主激活可以产生更有效的激活模式
- Token-balancing 和 Expert-balancing 可以互补而非互斥
- 全局稀疏目标优于逐层强制

### 局限性

1. **规模限制**：实验最大 0.8B 参数，更大规模待验证
2. **评估范围**：仅 9 个英文 benchmark，应用范围有限
3. **从头训练**：未探索预训练模型转换
4. **资源限制**：大规模预训练超出当前资源

### 未来方向

1. **更大规模验证**：扩展到数十亿参数的预训练
2. **预训练模型转换**：将现有 MoE 模型转换为 Routing-Free
3. **更多任务**：扩展到更广泛的 NLP 任务
4. **推理优化**：结合专家并行优化部署效率

## 八、参考资源

- **论文**: https://arxiv.org/abs/2604.00801
- **代码**: https://github.com/liuyilun2000/RoutingFreeMoE/tree/release
- **相关论文**:
  - Mixtral: https://arxiv.org/abs/2401.04088
  - Switch Transformers: https://arxiv.org/abs/2101.03961
  - DeepSeekMoE: https://arxiv.org/abs/2401.06066
  - AoE: https://arxiv.org/abs/2501.13074
  - ReMoE: https://arxiv.org/abs/2402.xxxxx
