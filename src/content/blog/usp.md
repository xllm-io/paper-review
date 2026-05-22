---
title: "USP: 统一序列并行方法用于长上下文生成式AI"
description: "USP 提出统一序列并行方法，用于长上下文生成式 AI 模型。"
date: 2026-04-02
tags:
  - "Deep Learning"
  - "Attention"
draft: false
---
# USP: 统一序列并行方法用于长上下文生成式AI

## 一、论文概述

| 项目 | 内容 |
|------|------|
| **标题** | USP: A Unified Sequence Parallelism Approach for Long Context Generative AI |
| **作者** | Jiarui Fang, Shangchun Zhao |
| **机构** | 腾讯 (Tencent) |
| **论文** | https://arxiv.org/abs/2405.07719 |
| **代码** | https://github.com/feifeibear/long-context-attention |
| **发布** | 2024年5月13日 (v5: 2024年7月2日) |
| **许可** | CC BY 4.0 |
| **领域** | 分布式训练、序列并行、长上下文LLM |

## 二、核心思想

### 问题定义

随着生成式AI模型上下文长度不断增长（Claude 100K、GPT-4 128K、Gemini 1.5 Pro 10M tokens），序列并行（SP）成为解锁长上下文能力的关键技术。然而，现有两种主要SP方法各有局限：

1. **DeepSpeed-Ulysses (SP-Ulysses)**：
   - 并行度受限于注意力头数量 `hc`
   - 不适用于GQA/MQA场景（如LLaMA3-8B KV头数仅8）
   - 与Tensor Parallelism存在维度冲突

2. **Ring-Attention (SP-Ring)**：
   - 将Q/K/V/O张量细分为小块，降低计算效率
   - 因果注意力下存在负载不均衡问题（GPU3负载可达GPU0的7倍）
   - 即使通信完全重叠，总执行时间仍落后于Ulysses

### 解决方案概述

USP提出**统一序列并行方法**，将SP-Ulysses和SP-Ring结合为混合并行策略：

> **将SP进程组划分为两个正交的进程组集合：SP-Ring进程组（跨列）和SP-Ulysses进程组（跨行），形成2D网格拓扑。**

关键创新点：
1. **统一框架**：覆盖SP-Ulysses和SP-Ring的能力，同时提供额外优势
2. **负载均衡**：通过序列重排解决因果注意力的负载不均问题
3. **4D并行最佳实践**：系统分析SP与DP/TP/PP/ZeRO的交互关系

## 三、技术架构

### 整体框架

```
输入序列 Q, K, V ∈ R^(L×d)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   USP-Attention 算法                         │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. AllToAll4D (Ulysses维度)                         │    │
│  │    - 合并L维度，划分hc维度                           │    │
│  │    - Q,K,V: (bs, L/N, hs, hd) → (hc/N, bs, L, hd) │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 2. LoadBalance-RingAttention (Ring维度)              │    │
│  │    - 跨Ring进程组进行P2P通信                         │    │
│  │    - 重叠计算与通信                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 3. AllToAll4D (反向)                                 │    │
│  │    - 恢复序列维度划分                                │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
    输出 O
```

### 核心算法

**Algorithm 1: USP-Attention**

```
输入: ulysses_pg, ring_pg, Q, K, V, scatter_idx, gather_idx

1. Q ← AllToAll4D(Q, scatter_idx, gather_idx, group=ulysses_pg)
2. K ← AllToAll4D(K, scatter_idx, gather_idx, group=ulysses_pg)
3. V ← AllToAll4D(V, scatter_idx, gather_idx, group=ulysses_pg)
4. O ← LoadBalance-RingAttention(Q, K, V, group=ring_pg)
5. O ← AllToAll4D(O, gather_idx, scatter_idx, group=ulysses_pg)
6. return O
```

**Algorithm 2: 负载均衡序列分段**

```
输入: seq, ring_process_group, ulysses_process_group

1. ring_degree ← ring_process_group.get_world_size()
2. ring_rank ← ring_process_group.get_rank()
3. ulysses_rank ← ulysses_process_group.get_rank()
4. seq_chunks ← seq.chunk(2 × ring_degree)
5. reorder_seq ← concat([seq_chunks[r_rank], seq_chunks[2×rd-r_rank-1]])
6. local_seq ← reorder_seq.chunk(ud)[u_rank]
7. return local_seq
```

### 进程组拓扑示例

```
8个进程 → 2×4 网格

Ulysses维度 (行): 2个进程
Ring维度 (列): 4个进程

        Ring 0    Ring 1    Ring 2    Ring 3
       ┌─────────┬─────────┬─────────┬─────────┐
Ulysses 0 │  GPU 0  │  GPU 1  │  GPU 2  │  GPU 3  │
       ├─────────┼─────────┼─────────┼─────────┤
Ulysses 1 │  GPU 4  │  GPU 5  │  GPU 6  │  GPU 7  │
       └─────────┴─────────┴─────────┴─────────┘
```

## 四、核心创新

| 创新点 | 说明 | 理论/实验依据 |
|--------|------|---------------|
| **统一SP框架** | 将Ulysses和Ring结合为2D网格并行 | 涵盖两者能力，当Ulysses度=N时退化为SP-Ulysses，Ring度=N时退化为SP-Ring |
| **负载均衡重排** | 解决因果注意力下的负载不均问题 | GPU负载从7倍差异降至完美均衡 |
| **网络拓扑鲁棒性** | All2All在高带宽互联，P2P在低带宽段 | 适用于PCIe Switch、以太网等异构网络 |
| **4D并行最佳实践** | 系统分析SP与DP/TP/PP/ZeRO的交互 | 提供7条实践建议（Tips） |
| **GQA通信优化** | GQA可进一步降低SP通信成本 | K/V通信成本降至1/G |

## 五、4D并行最佳实践

### 通信与内存成本对比

| 方法 | 参数通信 | 激活通信 | 分割维度 | P/G内存 | OS内存 | 激活内存 |
|------|---------|---------|---------|---------|--------|---------|
| SP-Ulysses | allreduce 12O(d²) | 8×all2all 8/N·O(bs·L·d) | hc/L | P+G | 6P | A/N |
| SP-Ring | allreduce 12O(d²) | P2P 4O(bs·L·d) | L/L | P+G | 6P | A/N |
| DP | allreduce 12O(d²) | 0 | bs/bs | P+G | 6P | A/N |
| ZeRO-1 | allgather+reducescatter | 0 | hc/L | P+G | 6P/N | A/N |
| TP | 0 | 4×allreduce 8O(bs·L·d) | hc/d | (P+G)/N | 6P/N | αA |
| TP-sp | 0 | 6×allgather+4×reducescatter | hc/d | (P+G)/N | 6P/N | A/N |

### 7条最佳实践建议

| Tip | 建议内容 |
|-----|---------|
| **Tip 1** | 使用Unified-SP替代SP-Ring和SP-Ulysses，它涵盖两者能力并提供额外优势 |
| **Tip 2** | 优先使用DP而非SP；仅当batch size不足时考虑SP |
| **Tip 3** | 使用SP时应始终配合ZeRO-1/2；可考虑ZeRO-3和Offload以通信换内存 |
| **Tip 4** | SP在大规模下比TP-sp有通信优势；GQA可进一步降低SP通信成本 |
| **Tip 5** | 从TP-sp切换到SP不能增加训练序列长度；SP+ZeRO-3可达到类似TP-sp的序列长度 |
| **Tip 6** | 更高SP并行度（需大Ring度）可训练更长序列，这是TP-sp无法实现的优势 |
| **Tip 7** | 4D混合并行中进程组维度从低到高顺序：TP → SP-Ulysses → SP-Ring → ZeRO-DP → PP |

## 六、实验结果

### 实验设置

- **硬件**：
  - 8×L20 PCIe GPU集群
  - 8×A100-SXM4 NVLink节点
  - 2×8×A800 NVLink节点（节点间1.6Tbps RDMA）
- **模型**：LLaMA2-7B, LLaMA3-8B
- **框架**：Megatron-LM (commit 2196398, 2024年4月12日)

### Table 3: SP-Unified在8×L20 PCIe上的吞吐量 (iters/sec)

| 序列长度 | Ulysses度 | Ring度 | basic-ring | lb-ring |
|---------|----------|--------|-----------|---------|
| 8K | 8 | 1 | 57.346 | 57.098 |
| 8K | 4 | 2 | 153.134 | 152.189 |
| 8K | 2 | 4 | 415.5 | **454.93** |
| 8K | 1 | 8 | 358.595 | 361.969 |
| 32K | 4 | 2 | 28.584 | 32.818 |
| 32K | 2 | 4 | 44.348 | **62.754** |
| 128K | 4 | 2 | 3.217 | 4.235 |
| 128K | 2 | 4 | 3.399 | **5.476** |

**发现**：在PCIe异构网络下，Ulysses度=2、Ring度=4配置最优，验证了Tip 1。

### Table 4: SP-Unified在8×A100 NVLink上的吞吐量 (iters/sec)

| 序列长度 | Ulysses度 | Ring度 | basic-ring | lb-ring |
|---------|----------|--------|-----------|---------|
| 32K | 8 | 1 | 135.569 | **136.375** |
| 32K | 4 | 2 | 103.525 | 132.979 |
| 128K | 8 | 1 | 2.782 | **2.785** |
| 128K | 4 | 2 | 2.024 | 2.771 |

**发现**：在NVLink高带宽互联下，SP-Ulysses（度=8）最优。

### Table 5: LLAMA2-7B混合并行性能 (8×A800 NVLink单节点)

| 序列长度 | TP度 | Ulysses度 | Ring度 | FLOPS/GPU | MFU |
|---------|------|----------|--------|-----------|-----|
| 64K | 4 | 2 | 1 | 154.49 | 0.50 |
| 30K | 1 | 8 | 1 | **163.42** | **0.52** |

**发现**：SP-Ulysses在NVLink下比TP-sp高26%吞吐量。

### Table 6: LLAMA3-8B双节点性能 (2×8×A800 NVLink)

| 序列长度 | TP度 | Ulysses度 | Ring度 | FLOPS/GPU | MFU |
|---------|------|----------|--------|-----------|-----|
| 64K | 1 | 4 | 4 | 137.48 | 0.44 |
| 80K | 1 | 4 | 4 | **148.90** | **0.48** |
| 120K | 4 | 2 | 2 | **152.51** | **0.49** |

**发现**：
- 64K/80K：SP-only最优（Ulysses=4, Ring=4），Unified SP比SP-Ring高13%/12%
- 120K：TP+SP混合最优（TP=4, Ulysses=2, Ring=2），SP-only因OOM失败

### Table 7: 序列长度上限探索

| 序列长度 | TP度 | Ulysses度 | Ring度 | FLOPS/GPU | MFU |
|---------|------|----------|--------|-----------|-----|
| 160K | 4 | 1 | 4 | 159.37 | 0.51 |
| 190K | 4 | 1 | 4 | 157.08 | 0.50 |
| 208K | 8 | 1 | 2 | 147.26 | 0.47 |

**发现**：最大序列长度208K，使用TP=8 + Ring=2配置，达到47% MFU。

### 收敛验证

USP与DP的loss曲线在10K迭代上完全重叠，验证了负载均衡重排和RoPE修改的正确性。

## 七、相关工作

### 序列并行发展脉络

| 方法 | 年份 | 特点 | 局限 |
|------|------|------|------|
| Megatron-LM SP | 2022 | TP的激活内存优化 | 不能独立使用，通信量不变 |
| DeepSpeed-Ulysses | 2023.9 | All2All通信，通信量恒定 | 并行度≤hc，不适用于GQA/MQA |
| Ring-Attention | 2023.10 | P2P通信重叠计算 | 计算效率下降，负载不均 |
| Striped Attention | 2023.11 | 改进Ring的负载均衡 | 仅解决Ring的局部问题 |
| **USP** | 2024.5 | 统一Ulysses+Ring | - |

### 与其他并行方法的关系

| 并行方法 | 与SP的关系 |
|---------|-----------|
| **DP** | SP通信开销>DP；优先使用DP（Tip 2） |
| **ZeRO-1/2** | 与SP完全兼容，应始终配合使用（Tip 3） |
| **ZeRO-3** | 与SP兼容，可达到类似TP-sp的内存效率 |
| **TP-sp** | SP在大规模下通信更优；TP-sp内存更优 |
| **PP** | 与SP完全互补（PP跨层，SP跨层内） |

## 八、代码实现分析

### 项目结构

```
long-context-attention/
├── README.md
├── usp/                    # USP核心实现
│   ├── usp_attention.py   # USP-Attention算法
│   ├── ring_attention.py  # Ring-Attention实现
│   └── ulysses_attention.py  # Ulysses-Attention实现
├── megatron/              # Megatron-LM集成
│   ├── sp_utils.py        # SP工具函数
│   └── context_parallel.py  # Context Parallel实现
└── benchmarks/            # 基准测试
    └── attention_benchmark.py
```

### 关键实现细节

1. **AllToAll4D**：4D张量的All2All通信
   - 支持任意scatter/gather维度
   - 处理GQA/MQA下K/V头数与Q不同的情况

2. **LoadBalance-RingAttention**：
   - 序列重排算法：`concat([seq_chunks[r_rank], seq_chunks[2×rd-r_rank-1]])`
   - RoPE位置编码同步调整

3. **进程组管理**：
   - 正交进程组划分：`ulysses_pg` × `ring_pg` = `sp_pg`
   - 支持任意Ulysses度×Ring度组合

## 九、总结

### 核心贡献

1. **统一SP框架**：将DeepSpeed-Ulysses和Ring-Attention结合为统一方法，克服两者局限，对模型架构和网络硬件更鲁棒

2. **4D并行最佳实践**：系统分析SP与DP/TP/PP/ZeRO的交互关系，提供7条实践建议

3. **实验验证**：
   - 在PCIe异构网络下，最优配置为Ulysses度=2, Ring度=4
   - 在NVLink高带宽下，SP-Ulysses（度=8）最优
   - LLaMA3-8B在2节点16×A800上训练208K序列，达到47% MFU
   - Unified SP比SP-Ring高13%吞吐量（64K序列）

### 技术影响

- **实用性**：已集成到Megatron-LM，可直接用于生产
- **通用性**：适用于任何Transformer架构，包括GQA/MQA
- **可扩展性**：Ring度可任意扩展，突破TP-sp的hc限制
- **示范性**：为4D混合并行设计提供了系统性指导

### 局限性

1. **验证规模有限**：仅在2节点16 GPU上验证
2. **未涉及推理**：仅关注训练场景
3. **未集成ZeRO-3**：Megatron-LM官方不支持
4. **未涉及MoE**：SP与专家并行的结合待研究

### 未来方向

1. **大规模集群验证**：在10K+ GPU上验证SP优势
2. **SP+ZeRO-3集成**：实现内存效率的进一步提升
3. **SP+MoE结合**：设计SP与专家并行的All2All通信

## 十、参考资源

- **论文**: https://arxiv.org/abs/2405.07719
- **代码**: https://github.com/feifeibear/long-context-attention
- **DeepSpeed-Ulysses**: https://arxiv.org/abs/2309.14509
- **Ring-Attention**: https://arxiv.org/abs/2310.01889
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
