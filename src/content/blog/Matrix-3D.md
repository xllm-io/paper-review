---
title: "Matrix-3D: 全方位可探索 3D 世界生成"
description: "Matrix-3D 提出从单张图像生成全方位可探索 3D 世界的统一框架。"
date: 2026-04-20
tags:
  - "Deep Learning"
  - "Paper Summary"
draft: false
---
# Matrix-3D: 全方位可探索 3D 世界生成

## 一、项目概述

**Matrix-3D** 是由 **Skywork（昆仑万维）** 开发的 **全方位可探索 3D 世界生成框架**，能够从单张图像或文本提示生成可 360° 自由探索的 3D 场景。

- **论文**: [arXiv:2508.08086](https://arxiv.org/abs/2508.08086)（2025年8月11日提交）
- **GitHub**: [SkyworkAI/Matrix-3D](https://github.com/SkyworkAI/Matrix-3D)
- **HuggingFace**: [Skywork/Matrix-3D](https://huggingface.co/Skywork/Matrix-3D)
- **许可**: MIT
- **领域**: 计算机视觉 (cs.CV)、图形学 (cs.GR)

---

## 二、核心能力

| 特性 | 说明 |
|------|------|
| **大场景生成** | 支持生成更广阔、更完整的场景，允许 360° 全方位自由探索 |
| **高可控性** | 支持文本和图像输入，可自定义轨迹，无限可扩展 |
| **强泛化能力** | 基于自研 3D 数据和视频模型先验，生成多样化高质量 3D 场景 |
| **速度-质量平衡** | 提供两种全景 3D 重建方法：快速重建和精细重建 |

---

## 三、技术架构

Matrix-3D 的核心流程分为 **三个阶段**：

### 1. 全景图像生成（Text/Image → Panorama）
- 从文本提示或输入图像生成 **360° 全景图像**
- 基于 **FLUX.1** 模型微调的 Text2PanoImage LoRA

### 2. 全景视频生成（Panorama → Video）
- 使用 **轨迹引导的全景视频扩散模型**
- 以场景网格渲染为条件，实现高质量、几何一致的场景视频生成
- 基于 **Wan2.1** 视频模型
- 支持 480p（960×480）和 720p（1440×720）分辨率
- 提供三种运动模式：直线旅行、S 曲线旅行、右侧前进
- 支持自定义相机轨迹（.json 格式，OpenCV 世界到相机矩阵）

### 3. 3D 场景提取（Video → 3D Scene）
提供 **两种 3D 重建方法**：

| 方法 | 特点 | 输出 |
|------|------|------|
| **优化重建**（Optimization-based） | 高精度、细节丰富 | `.ply` 格式 3D 高斯 |
| **前馈重建**（Feed-forward） | 快速、高效 | `.ply` 格式 3D 高斯 + 透视视频 |

---

## 四、预训练模型

| 模型名称 | 文件 | 用途 |
|----------|------|------|
| Text2PanoImage | `text2panoimage_lora.safetensors` | 文本到全景图像生成 |
| PanoVideoGen-480p | `pano_video_gen_480p.ckpt` | 480p 全景视频生成 |
| PanoVideoGen-720p | `pano_video_gen_720p.bin` | 720p 全景视频生成 |
| PanoLRM-480p | `pano_lrm_480p.pt` | 前馈式全景 3D 重建 |

---

## 五、数据集：Matrix-Pano

论文引入了 **Matrix-Pano 数据集**——首个大规模合成全景视频数据集：
- **116K** 高质量静态全景视频序列
- 包含深度和轨迹标注
- 专为全景视频生成和 3D 重建训练设计

---

## 六、使用方式

```bash
# 克隆仓库
git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
cd Matrix-3D

# 创建环境
conda create -n matrix3d python=3.10
conda activate matrix3d
pip3 install torch==2.7.1 torchvision==0.22.1
chmod +x install.sh && ./install.sh

# 下载模型
python scripts/download_checkpoints.py

# 一键生成 3D 世界
./generate.sh

# 或分步执行：
# Step 1: 文本/图像 → 全景图像
python code/panoramic_image_generation.py --mode=t2p --prompt="..." --output_path="./output/example1"

# Step 2: 全景图像 → 全景视频
torchrun --nproc_per_node 1 code/panoramic_image_to_video.py --inout_dir="./output/example1" --resolution=720

# Step 3: 视频 → 3D 场景
python code/panoramic_video_to_3DScene.py --inout_dir="./output/example1" --resolution=720
```

---

## 七、技术依赖与致谢

Matrix-3D 构建在以下开源项目之上：

- **[FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)** — 文本到图像扩散模型
- **[Wan2.1](https://github.com/Wan-Video/Wan2.1)** — 视频生成模型
- **[WorldGen](https://github.com/ZiYang-xie/WorldGen/)** — 3D 世界生成
- **[MoGe](https://github.com/microsoft/MoGe)** — 微软单目几何估计
- **[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)** — 3D 高斯溅射
- **[nvdiffrast](https://github.com/NVlabs/nvdiffrast)** — 可微分渲染
- **[StableSR](https://github.com/IceClear/StableSR)** — 超分辨率
- **[VEnhancer](https://github.com/Vchitect/VEnhancer)** — 视频增强

---

## 八、相关方向工作综述

**全方位 3D 世界生成** 是当前 AI 3D 生成领域的热门方向，以下是主要相关工作和趋势：

### 1. 全景图像生成
- **PanoDiffusion (2024)**: 使用扩散模型进行 360° 全景图外绘/生成
- **Blockade Labs / Skybox AI (2024)**: 文本到全景天空盒生成，已商业化

### 2. 多视角一致性生成
- **MVDiffusion (2024)**: 从文本生成多视角一致的图像，包括全景设置
- **WonderJourney (2024)**: 从单张图像生成连续可探索的 3D 场景

### 3. 无限/可扩展 3D 场景
- **WorldGen (2024)**: 无界 3D 场景生成，Matrix-3D 的基础之一
- **SceneScape (2024)**: 利用全景先验实现一致性场景扩展
- **Text2Room (2024)**: 从文本生成 3D 房间场景
- **WonderWorld (2024)**: 循环/扩展 3D 环境生成

### 4. 单图像 3D 重建
- **LRM / LRM-large (2024)**: 大型重建模型，单图像前馈 3D 重建
- **InstantMesh (2024)**: 腾讯出品，高效网格重建
- **TripoSR (2024)**: Stability AI 的快速 3D 重建
- **Splatter Image (Omni)**: 从单张全景图像生成 3D 场景

### 5. 3D 高斯溅射 + 全景
- **DrivingGaussian / Street Gaussians (2024)**: 从全景驾驶视频重建街道级 3D 场景
- **3DGS + 360° 室内重建 (2024-2025)**: 从全景视频重建室内场景

### 6. 视频到 3D
- **4D-fy / DreamGaussian4D (2024)**: 从视频生成 4D 场景
- **基于扩散的 3D 生成综述 (2025)**: 覆盖分数蒸馏、多视角扩散、3D 感知生成

### 7. Matrix-3D 的差异化优势
与上述工作相比，Matrix-3D 的独特之处在于：
- **全景表示** + **视频扩散** 的结合，实现真正的 360° 可探索
- **首个大规模全景视频数据集**（Matrix-Pano, 116K 序列）
- **两种重建方案**兼顾速度与质量
- 完整的 **text/image → panorama → video → 3D** 流水线

---

## 九、总结

Matrix-3D 代表了 **全方位 3D 世界生成** 领域的一个重要进展。它通过将全景表示引入视频扩散模型，并结合自研的全景 3D 重建技术，实现了从文本或图像生成可 360° 自由探索的高质量 3D 世界。该项目开源、使用 MIT 许可，为社区提供了完整的训练数据集和预训练模型，具有很高的研究和应用价值。
