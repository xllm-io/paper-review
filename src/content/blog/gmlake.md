---
title: "GMLake: 高效透明的 GPU 内存碎片整理"
description: "GMLake 提出高效透明的 GPU 内存碎片整理方法，提升大模型训练效率。"
date: 2026-03-10
tags:
  - "Deep Learning"
  - "Paper Summary"
draft: false
---
# GMLake: 高效透明的 GPU 内存碎片整理

## 一、论文概述

| 项目 | 内容 |
|------|------|
| **标题** | GMLake: Efficient and Transparent GPU Memory Defragmentation for Large-scale DNN Training with Virtual Memory Stitching |
| **作者** | Cong Guo*, Rui Zhang*, Jiale Xu, Jingwen Leng†, Zihan Liu, Ziyu Huang, Minyi Guo†, Hao Wu, Shouren Zhao, Junping Zhao†, Ke Zhang |
| **机构** | Shanghai Jiao Tong University, Shanghai Qi Zhi Institute, Ant Group |
| **论文** | https://arxiv.org/abs/2401.08156 |
| **代码** | https://github.com/intelligent-machine-learning/glake/tree/main/GMLake |
| **会议** | ASPLOS '24 (Accepted) |
| **发布** | 2024年1月16日 |

## 二、核心思想

GMLake 是一种基于 GPU 虚拟内存管理的高效内存分配框架，专门解决大规模 DNN 训练中的内存碎片问题。通过虚拟内存拼接（VMS）机制，将非连续的物理内存块映射到连续的虚拟地址空间，显著减少内存碎片。

### 问题定义

大规模 DNN 模型（如 LLM）的训练面临严重的内存碎片问题：

1. **缓存分配器的局限**：PyTorch/TensorFlow 使用的 BFC（Best Fit with Coalescing）算法通过"分割"机制管理内存池，但会导致碎片化
2. **内存优化技术的副作用**：
   - **重计算（Recomputation）**：丢弃部分激活值，引入频繁的小内存分配/释放
   - **卸载（Offloading）**：在 CPU 和 GPU 之间频繁交换张量
   - **分布式训练**：多 GPU 并行导致更多不规则内存分配
   - **LoRA**：低秩适配引入额外的内存分配模式
3. **碎片化程度**：可达 30% 的 GPU 内存碎片，导致 OOM 错误

### 解决方案概述

GMLake 的核心创新：

1. **虚拟内存拼接（VMS）**：利用 CUDA 低级虚拟内存管理 API，将非连续物理内存块映射到连续虚拟地址
2. **双层内存池设计**：
   - **原始内存池（pPool）**：管理基本物理内存块（pBlock）
   - **拼接内存池（sPool）**：管理拼接后的虚拟内存块（sBlock）
3. **智能分配策略**：基于 BestFit 算法的四种状态处理机制
4. **训练模式利用**：利用 DNN 训练的周期性特征，实现内存模式缓存和复用

## 三、技术架构

### 内存管理策略对比

GMLake 对比了三种内存管理策略：

| 策略 | 优点 | 缺点 |
|------|------|------|
| **原生分配器** (cudaMalloc/cudaFree) | 简单直接 | 开销大（10x 慢），需要设备同步 |
| **缓存分配器** (BFC 算法) | 快速分配，无需同步 | 分割机制导致碎片化 |
| **虚拟内存分配器** (GMLake) | 减少碎片，透明集成 | 需要优化 VMM API 开销 |

### 核心数据结构

#### 原始块（pBlock）
- 最小可访问单元
- 通过 VMM API 创建：`cuMemAddressReserve` + `cuMemCreate` + `cuMemMap`
- 使用统一 2MB 块大小优化碎片整理

#### 拼接块（sBlock）
- 由多个 pBlock 拼接而成
- 不创建新的物理块，只映射已有 pBlock 的物理块
- 支持多个 sBlock 指向相同 pBlock（软链接机制）

#### 内存池（pPool 和 sPool）
- **pPool**：按大小降序存储 pBlock，严格一对一映射 GPU 内存
- **sPool**：按大小降序存储 sBlock，记录与 pBlock 的链接关系

### 核心算法

#### BestFit 算法

```
输入: 分配大小 bSize, 非活跃 sBlock 和 pBlock 列表
输出: 状态 state, 候选 pBlock 列表 CB

四种状态:
S1 - 精确匹配: block.size == bSize (可直接分配)
S2 - 单块匹配: 找到最小的 > bSize 的 pBlock (需要分割)
S3 - 多块匹配: 多个小 pBlock 拼接 >= bSize (需要拼接)
S4 - 块不足: 可用块总和 < bSize (需要 Alloc 新块)
```

#### 分配策略

| 状态 | 处理方式 | 操作 |
|------|----------|------|
| **S1** | 直接返回匹配的 pBlock/sBlock | 无 |
| **S2** | 分割大 pBlock，分配部分，拼接剩余 | Split + Stitch |
| **S3** | 拼接多个小 pBlock，可能分割最后一个 | Stitch + Split |
| **S4** | 分配新 pBlock，与已有块拼接 | Alloc + Stitch |
| **S5** | 无足够内存，报告 OOM | 错误 |

#### 释放策略

- **Update**：仅更新 pBlock/sBlock 的活跃状态，不释放物理内存
- **StitchFree**：基于 LRU 策略释放 sPool 中的非活跃 sBlock

### 关键优化

1. **统一块大小**：使用 2MB 作为最小物理块大小，平衡碎片整理效果和 VMM API 开销
2. **小块回退**：< 2MB 的分配使用原始 PyTorch 分割方法
3. **模式缓存**：利用 DNN 训练的周期性，经过几次迭代后达到"收敛"状态
4. **LRU 释放**：当 sPool 容量超限时，释放最近最少使用的 sBlock

## 四、核心创新

| 创新点 | 说明 | 理论/实验依据 |
|--------|------|---------------|
| **虚拟内存拼接** | 将非连续物理块映射到连续虚拟地址 | 减少 15-33% 碎片 |
| **双层内存池** | pPool 管理物理块，sPool 管理拼接块 | 支持高效复用 |
| **训练模式利用** | 利用周期性实现 sBlock 复用 | 4 次迭代后收敛 |
| **透明集成** | 替换 PyTorch 缓存分配器，用户无感知 | 无需修改代码 |
| **DNN 特定优化** | 2MB 统一块大小，小块回退策略 | 平衡效果和开销 |

## 五、实验结果

### 实验配置

- **硬件**：8x NVIDIA A100 GPU (80GB HBM)，Intel Xeon Platinum 8369B CPU，1TB DRAM
- **软件**：CUDA 11.4，cuDNN 8.5，PyTorch 1.13.1/2.0
- **模型**：OPT-1.3B, OPT-13B, Vicuna-13B, GPT-NeoX-20B, GLM-10B, GPT-2 等
- **平台**：DeepSpeed ZeRO-3, FSDP, Colossal-AI

### 核心结果

#### 内存碎片减少

| 指标 | 平均值 | 最佳值 |
|------|--------|--------|
| **碎片率减少** | 15% | 33% |
| **内存节省** | 9.2 GB | 25 GB |
| **最终碎片率** | 5-10% | - |

#### 可扩展性分析

**1. 内存优化策略可扩展性**

| 模型 | 策略 | 碎片减少 | 内存节省 |
|------|------|----------|----------|
| OPT-13B | LRO | ~20% | ~17 GB |
| Vicuna-13B | LRO | ~15% | ~12 GB |
| GPT-NeoX-20B | LRO | ~24% | ~15 GB |

**2. GPU 规模可扩展性**

| GPU 数量 | 碎片率（PyTorch） | 碎片率（GMLake） | 改善 |
|----------|-------------------|------------------|------|
| 1 | ~10% | ~5% | 5% |
| 4 | ~20% | ~8% | 12% |
| 8 | ~22% | ~9% | 13% |
| 16 | ~24% | ~10% | 14% |

**3. 平台可扩展性**

| 平台 | 模型 | 碎片减少 | 内存节省 |
|------|------|----------|----------|
| DeepSpeed | OPT-13B | 33% | 25 GB |
| FSDP | GLM-10B | 9% | 7 GB |
| Colossal-AI | GPT-2 | 15% | 10 GB |

#### 端到端性能

| 模型 | 批量大小 | PyTorch | GMLake | 吞吐量 |
|------|----------|---------|--------|--------|
| OPT-1.3B | 249 | OOM | 成功 | 相当 |
| OPT-13B | 120 | OOM | 成功 | 相当 |
| GPT-NeoX-20B | 72 | OOM | 成功 | 相当 |

### 内存轨迹分析

GMLake 在 GPT-NeoX-20B 训练中的内存行为：

1. **PyTorch**：在 ~200s 因 OOM 终止
2. **GMLake**：
   - 活跃内存与 PyTorch 相当
   - 保留内存显著更低（碎片少）
   - 经过 4 次迭代后达到稳定状态
   - 吞吐量与 PyTorch 相当

## 六、相关工作

### 内存碎片整理

| 方法 | 特点 | 局限 |
|------|------|------|
| **固定大小块** | 消除数据移动开销 | 访问开销大，灵活性差 |
| **压缩策略** | 合并小块为大块 | 需要数据移动 |
| **垃圾回收** | 简化移动逻辑 | 临时内存浪费 |
| **GMLake** | 虚拟内存拼接 | 需要 VMM API 支持 |

### 与相关工作的技术差异

| 方法 | 作用范围 | 特点 |
|------|----------|------|
| **vLLM** | 张量级别 | 算法级解决方案，针对 Self-Attention |
| **GMLake** | 内存池级别 | 系统级解决方案，适用于所有 DNN 训练 |
| **CUDA VMM** | 物理内存级别 | 底层工具，无内存池感知 |

## 七、总结

### 核心贡献

1. **虚拟内存拼接机制**：利用 CUDA VMM API 将非连续物理内存映射到连续虚拟地址
2. **双层内存池设计**：pPool 和 sPool 分别管理物理块和拼接块
3. **智能分配策略**：BestFit 算法的四种状态处理，覆盖所有分配场景
4. **训练模式利用**：利用 DNN 训练的周期性实现内存模式缓存
5. **显著效果**：
   - 平均减少 15% 碎片（最高 33%）
   - 平均节省 9.2 GB 内存（最高 25 GB）
   - 碎片率从 20-30% 降至 5-10%
6. **透明集成**：完全兼容 PyTorch，无需修改用户代码

### 技术影响

GMLake 证明了通过虚拟内存管理技术，可以有效解决大规模 DNN 训练中的内存碎片问题。其透明集成设计使得用户无需修改代码即可获得显著的内存优化效果。

### 局限性

1. **VMM API 开销**：原始 VMM API 调用开销较大，需要通过缓存机制摊销
2. **2MB 块大小限制**：小于 2MB 的分配回退到原始方法
3. **平台依赖**：依赖 CUDA 虚拟内存管理 API
4. **训练场景限定**：主要针对训练场景，推理场景效果待验证

## 八、参考资源

- **论文**: https://arxiv.org/abs/2401.08156
- **代码**: https://github.com/intelligent-machine-learning/glake/tree/main/GMLake
- **会议**: ASPLOS '24
- **相关论文**:
  - PyTorch: https://pytorch.org/
  - DeepSpeed ZeRO: https://arxiv.org/abs/1910.02054
  - FlashAttention: https://arxiv.org/abs/2205.14135
  - vLLM: https://arxiv.org/abs/2309.06180
