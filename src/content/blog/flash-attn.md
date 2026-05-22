---
title: "FlashAttention 常用接口简单介绍"
description: "FlashAttention 常用接口介绍，包括基本使用方法和性能优化技巧。"
date: 2026-03-15
tags:
  - "Deep Learning"
  - "Attention"
  - "Sparse"
draft: false
---
# FlashAttention 常用接口简单介绍

## 一、概述

FlashAttention 是一种高效的注意力机制实现，通过 IO 感知的算法设计显著提升 Transformer 模型的训练和推理速度。它利用 GPU 的内存层次结构（SRAM vs HBM）来减少内存访问，实现 2-4 倍的加速，同时将内存使用从 O(N²) 降低到 O(N)。

### 核心优势

- **内存高效**：注意力计算的内存复杂度从 O(N²) 降至 O(N)
- **速度提升**：相比标准实现，训练速度提升 2-4 倍
- **精确计算**：不使用近似，结果与标准注意力完全一致
- **支持长序列**：可处理更长的输入序列

## 二、安装

### 从 PyPI 安装

```bash
pip install flash-attn --no-build-isolation
```

### 从源码安装

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e . --no-build-isolation
```

### 系统要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+
- GPU: Ampere (A100, A10, A30) 或更新架构（支持 Tensor Core）

## 三、核心接口

### 3.1 flash_attn_func

这是最常用的标准接口，适用于大多数场景。

```python
from flash_attn import flash_attn_func

def flash_attn_func(
    q,              # Query张量 [batch_size, seqlen_q, num_heads, head_dim]
    k,              # Key张量 [batch_size, seqlen_k, num_heads_k, head_dim]
    v,              # Value张量 [batch_size, seqlen_k, num_heads_k, head_dim]
    dropout_p=0.0,  # Dropout概率
    softmax_scale=None,  # Softmax缩放因子，默认为 1/sqrt(head_dim)
    causal=False,   # 是否使用因果注意力（用于自回归模型）
    window_size=(-1, -1),  # 滑动窗口大小 (left, right)
    softcap=0.0,    # Softcap值，用于限制注意力分数范围
    alibi_slopes=None,  # ALiBi位置编码的斜率
    deterministic=False,  # 是否使用确定性算法
    return_attn_probs=False,  # 是否返回注意力概率
):
    """
    返回:
        out: 注意力输出 [batch_size, seqlen_q, num_heads, head_dim]
        softmax_lse: Softmax的log-sum-exp值，用于数值稳定性
    """
```

#### 基本使用示例

```python
import torch
from flash_attn import flash_attn_func

# 创建输入张量
batch_size, seqlen, num_heads, head_dim = 2, 1024, 8, 64
q = torch.randn(batch_size, seqlen, num_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, seqlen, num_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, seqlen, num_heads, head_dim, device='cuda', dtype=torch.float16)

# 标准注意力
out = flash_attn_func(q, k, v)

# 因果注意力（用于GPT类模型）
out = flash_attn_func(q, k, v, causal=True)

# 带Dropout的注意力
out = flash_attn_func(q, k, v, dropout_p=0.1)

# 滑动窗口注意力（Mistral风格）
out = flash_attn_func(q, k, v, window_size=(256, 256))
```

### 3.2 flash_attn_varlen_func

用于处理变长序列，适用于批量处理不同长度的序列。

```python
from flash_attn import flash_attn_varlen_func

def flash_attn_varlen_func(
    q,              # Query张量 [total_q, num_heads, head_dim]
    k,              # Key张量 [total_k, num_heads_k, head_dim]
    v,              # Value张量 [total_k, num_heads_k, head_dim]
    cu_seqlens_q,   # Query的累积序列长度 [batch_size + 1]
    cu_seqlens_k,   # Key的累积序列长度 [batch_size + 1]
    max_seqlen_q,   # Query的最大序列长度
    max_seqlen_k,   # Key的最大序列长度
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """
    返回:
        out: 注意力输出 [total_q, num_heads, head_dim]
        softmax_lse: Softmax的log-sum-exp值
    """
```

#### 使用示例

```python
import torch
from flash_attn import flash_attn_varlen_func

# 假设有3个不同长度的序列
# 序列长度: [512, 1024, 768]
cu_seqlens_q = torch.tensor([0, 512, 1536, 2304], device='cuda', dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, 512, 1536, 2304], device='cuda', dtype=torch.int32)
max_seqlen_q = 1024
max_seqlen_k = 1024

# 总token数 = 512 + 1024 + 768 = 2304
total_tokens = 2304
num_heads, head_dim = 8, 64

q = torch.randn(total_tokens, num_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(total_tokens, num_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(total_tokens, num_heads, head_dim, device='cuda', dtype=torch.float16)

# 变长注意力
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    causal=True
)
```

### 3.3 flash_attn_with_kvcache

用于推理时的 KV 缓存管理，特别适合自回归生成。

```python
from flash_attn import flash_attn_with_kvcache

def flash_attn_with_kvcache(
    q,              # Query张量 [batch_size, seqlen_q, num_heads, head_dim]
    k_cache,        # Key缓存 [batch_size, seqlen_k_max, num_heads_k, head_dim]
    v_cache,        # Value缓存 [batch_size, seqlen_k_max, num_heads_k, head_dim]
    k=None,         # 新的Key（可选，用于更新缓存）
    v=None,         # 新的Value（可选，用于更新缓存）
    cache_seqlens=None,  # 缓存中已有的序列长度
    cache_leftpad=None,  # 缓存左侧填充
    cache_batch_idx=None,  # 缓存批次索引（用于beam search）
    cache_seqlens_k=None,  # Key的缓存序列长度
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    num_splits=0,   # 用于分布式注意力的分割数
    return_softmax_lse=False,
):
    """
    返回:
        out: 注意力输出 [batch_size, seqlen_q, num_heads, head_dim]
        softmax_lse (可选): Softmax的log-sum-exp值
    """
```

#### 使用示例

```python
import torch
from flash_attn import flash_attn_with_kvcache

batch_size = 1
max_seq_len = 2048
num_heads, head_dim = 8, 64

# 初始化KV缓存
k_cache = torch.zeros(batch_size, max_seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
v_cache = torch.zeros(batch_size, max_seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)

# 第一步：prefill
q = torch.randn(batch_size, 128, num_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, 128, num_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, 128, num_heads, head_dim, device='cuda', dtype=torch.float16)

# 更新缓存并计算注意力
out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=k, v=v,
    cache_seqlens=torch.tensor([128], device='cuda', dtype=torch.int32),
    causal=True
)

# 后续步骤：单token生成
q_new = torch.randn(batch_size, 1, num_heads, head_dim, device='cuda', dtype=torch.float16)
k_new = torch.randn(batch_size, 1, num_heads, head_dim, device='cuda', dtype=torch.float16)
v_new = torch.randn(batch_size, 1, num_heads, head_dim, device='cuda', dtype=torch.float16)

out_new = flash_attn_with_kvcache(
    q_new, k_cache, v_cache,
    k=k_new, v=v_new,
    cache_seqlens=torch.tensor([129], device='cuda', dtype=torch.int32),
    causal=True
)
```

### 3.4 flash_attn_combine

用于组合多个注意力输出，常用于分布式注意力或分块注意力。

```python
from flash_attn import flash_attn_combine

def flash_attn_combine(
    out_partial,    # 部分注意力输出 [batch_size, seqlen, num_heads, head_dim, num_splits]
    lse_partial,    # 部分log-sum-exp值 [batch_size, seqlen, num_heads, num_splits]
):
    """
    返回:
        out: 合并后的注意力输出 [batch_size, seqlen, num_heads, head_dim]
    """
```

## 四、Transformer 集成接口

### 4.1 FlashAttention 应用于 Multi-Head Attention

```python
import torch
import torch.nn as nn
from flash_attn import flash_attn_func

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=True):
        batch_size, seq_len, _ = x.shape

        # 投影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # FlashAttention
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=causal
        )

        # 输出投影
        out = out.reshape(batch_size, seq_len, -1)
        return self.out_proj(out)
```

### 4.2 FlashAttention 应用于 GQA (Grouped Query Attention)

```python
import torch
import torch.nn as nn
from flash_attn import flash_attn_func

class FlashGQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_heads_per_group = num_heads // num_kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=True):
        batch_size, seq_len, _ = x.shape

        # 投影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # FlashAttention 自动处理GQA
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=causal
        )

        # 输出投影
        out = out.reshape(batch_size, seq_len, -1)
        return self.out_proj(out)
```

### 4.3 滑动窗口注意力 (Mistral 风格)

```python
import torch
import torch.nn as nn
from flash_attn import flash_attn_func

class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size=256, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 滑动窗口注意力
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, self.window_size)  # (left, right)
        )

        out = out.reshape(batch_size, seq_len, -1)
        return self.out_proj(out)
```

## 五、高级用法

### 5.1 ALiBi 位置编码

```python
import torch
from flash_attn import flash_attn_func

def get_alibi_slopes(num_heads):
    """计算ALiBi斜率"""
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(torch.math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if torch.math.log2(num_heads).is_integer():
        slopes = get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** torch.floor(torch.math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
        slopes.extend(extra_slopes[0::2][:num_heads - closest_power_of_2])

    return torch.tensor(slopes, device='cuda', dtype=torch.float32)

# 使用ALiBi
num_heads = 8
alibi_slopes = get_alibi_slopes(num_heads)

q = torch.randn(2, 1024, num_heads, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 1024, num_heads, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 128, num_heads, 64, device='cuda', dtype=torch.float16)

out = flash_attn_func(q, k, v, alibi_slopes=alibi_slopes, causal=True)
```

### 5.2 Softcap 限制注意力分数

```python
from flash_attn import flash_attn_func

# softcap 限制注意力分数在 [-softcap, softcap] 范围内
# 用于稳定训练，特别是在长序列上
out = flash_attn_func(q, k, v, softcap=50.0, causal=True)
```

### 5.3 确定性模式

```python
from flash_attn import flash_attn_func

# 确保结果完全可复现（可能会略微降低性能）
out = flash_attn_func(q, k, v, deterministic=True)
```

## 六、性能优化建议

### 6.1 数据类型选择

```python
# 推荐使用 float16 或 bfloat16
q = q.to(torch.float16)  # 或 torch.bfloat16
k = k.to(torch.float16)
v = v.to(torch.float16)
```

### 6.2 内存优化

```python
# 使用 torch.cuda.amp 自动混合精度
from torch.cuda.amp import autocast

with autocast():
    out = flash_attn_func(q, k, v, causal=True)
```

### 6.3 序列长度对齐

```python
# FlashAttention 对序列长度有特定要求
# 推荐将序列长度对齐到 8 或 64 的倍数
def pad_to_multiple(seq_len, multiple=64):
    return ((seq_len + multiple - 1) // multiple) * multiple
```

### 6.4 使用 KV 缓存优化推理

```python
# 对于自回归生成，使用 flash_attn_with_kvcache
# 避免重复计算历史token的KV
out = flash_attn_with_kvcache(q, k_cache, v_cache, k=k_new, v=v_new)
```

## 七、常见问题

### Q1: 如何检查 FlashAttention 是否可用？

```python
try:
    from flash_attn import flash_attn_func
    print("FlashAttention is available")
except ImportError:
    print("FlashAttention is not installed")
```

### Q2: 如何处理序列长度不匹配？

```python
# Q 和 K/V 可以有不同的序列长度（交叉注意力）
q = torch.randn(2, 256, 8, 64, device='cuda', dtype=torch.float16)  # decoder
k = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)  # encoder
v = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)  # encoder

# 不使用 causal，因为这是交叉注意力
out = flash_attn_func(q, k, v, causal=False)
```

### Q3: 如何调试注意力权重？

```python
# 设置 return_attn_probs=True 返回注意力概率
out, softmax_lse, attn_weights = flash_attn_func(
    q, k, v,
    return_attn_probs=True
)
# 注意：返回的 attn_weights 可能是稀疏的或部分的
```

### Q4: FlashAttention v1 vs v2 vs v3？

- **FlashAttention v1**: 初始版本，基本功能
- **FlashAttention v2**: 性能优化，支持更多特性（推荐使用）
- **FlashAttention v3**: 最新版本，针对 Hopper 架构优化

```python
# FlashAttention 2 通常自动选择
from flash_attn import flash_attn_func  # 默认使用最新版本

# 显式指定版本
from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_v2
```

## 八、最佳实践

1. **始终使用 float16/bfloat16**: FlashAttention 针对这些数据类型优化
2. **对齐序列长度**: 将序列长度对齐到 8 或 64 的倍数
3. **使用 causal 参数**: 对于自回归模型，设置 `causal=True`
4. **利用 KV 缓存**: 推理时使用 `flash_attn_with_kvcache`
5. **处理变长序列**: 使用 `flash_attn_varlen_func` 而非 padding
6. **监控内存使用**: 虽然 FlashAttention 节省内存，但长序列仍需大量内存

## 九、参考资料

- **官方仓库**: https://github.com/Dao-AILab/flash-attention
- **论文**: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- **论文**: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- **文档**: https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn

---

*最后更新: 2025年*
