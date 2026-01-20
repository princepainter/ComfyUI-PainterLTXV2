# ComfyUI-PainterLTXV2

# 此节点由绘画小子制作

**Advanced LTXV Sampling & Latent Preparation for ComfyUI**

## 2026-1-19进行了更新。现在支持：文生视频+图生视频+尾帧视频+首尾帧视频+潜空间放大。
不连图片就是文生视频，可单独连首帧或者尾帧，或者首尾帧

# PainterLTX2VPlus

<img width="890" height="455" alt="image" src="https://github.com/user-attachments/assets/e5009b8c-f73b-4bfe-9fd6-5eb78f9e2c93" />


**全能型 LTXV 视频潜变量生成节点 | All-in-One LTXV Latent Generator**

---
## 带放大功能工作流
<img width="2083" height="1008" alt="image" src="https://github.com/user-attachments/assets/6cb90d03-9cd8-4782-bfa6-38c369eaf09c" />

## 不带放大功能工作流
<img width="1760" height="1012" alt="image" src="https://github.com/user-attachments/assets/8d9fdbdc-5d45-4204-a9ce-f226278de9bf" />

## 功能特点 | Features

支持文生视频、图生视频、首尾帧控制与潜在空间放大。  
Supports text-to-video, image-to-video, first/last frame control, and latent upscaling.

整合 LTXVLatentUpsampler 与首末帧控制功能于一体。  
Combines LTXVLatentUpsampler and first/last frame control in one node.

可接入音频 VAE 生成音视频同步内容。  
Supports audio VAE for synchronized audio-video generation.

输出口自动分离视频与音频潜变量，适配双采样器工作流。  
Automatically splits video and audio latents for dual-sampler workflows.

返回设定的宽高数值，便于下游节点直接调用。  
Returns configured width and height for downstream nodes.

---

## 输入说明 | Inputs

**video_vae**: 视频 VAE 模型 | Video VAE model  
**width/height**: 视频分辨率 | Video resolution  
**length**: 视频帧数 | Number of frames  
**frame_rate**: 帧率 | Frame rate  
**batch_size**: 批处理大小 | Batch size  
**audio_vae (可选)**: 音频 VAE | Audio VAE (optional)  
**start_image (可选)**: 首帧图像 | First frame image (optional)  
**end_image (可选)**: 尾帧图像 | Last frame image (optional)  
**latent (可选)**: 输入潜变量 | Input latent (optional)  
**upscale_model (可选)**: 放大模型 | Upscale model (optional)

---

## 输出说明 | Outputs

**latent**: 主潜变量 | Main latent  
**video_latent**: 视频潜变量 | Video latent  
**audio_latent**: 音频潜变量 | Audio latent  
**width**: 设定宽度 | Configured width  
**height**: 设定高度 | Configured height

---

## 使用场景 | Use Cases

仅连始帧 = 图生视频基础模式  
Start-frame only = Basic image-to-video mode

首尾帧同连 = 视频插值与过渡  
Both frames = Video interpolation and transition

接入 latent = 潜变量续写与重采样  
With latent = Latent continuation and resampling

叠加放大模型 = 高清化预处理  
Plus upscale model = High-resolution preprocessing

---

## 一句话介绍 | One-Liner

**Generate LTXV latents with frame control and upscaling for advanced video synthesis.**







## Nodes | 节点功能

### PainterSamplerLTXV


<img width="1876" height="899" alt="image" src="https://github.com/user-attachments/assets/6b2c1111-3911-40a2-ae46-54ec8a29bc42" />

**[English]**
- Full KSampler parameters with LTXV native audio-video latent separation
- External sigmas input support (overrides scheduler when connected)
- Flexible step control for partial denoising workflows
- Optimized for dynamic preservation and color fidelity

**[中文]**
- 完整的 KSampler 参数支持，原生 LTXV 音视频潜空间分离输出
- 外部 sigmas 输入支持（连接时覆盖调度器）
- 灵活的步骤控制，支持局部去噪工作流
- 针对动态效果保留和色彩保真优化


## Installation | 安装方法

**[English]**
1. Clone this repository into `ComfyUI/custom_nodes/`
2. Restart ComfyUI
3. Nodes will appear in `sampling` and `latent/video/ltxv` categories

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/princepainter/ComfyUI-PainterLTXV2.git
```

**[中文]**
1. 将本仓库克隆到 `ComfyUI/custom_nodes/` 目录
2. 重启 ComfyUI
3. 节点将出现在 `sampling` 和 `latent/video/ltxv` 分类中

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/princepainter/ComfyUI-PainterLTXV2.git
```

## Usage | 使用说明

**[English]**  
Connect **PainterLTXVtoVideo** to prepare latents, then feed into **PainterSamplerLTXV** for generation. Use external sigmas for custom sampling schedules.

**[中文]**  
将 **PainterLTXVtoVideo** 连接到 **PainterSamplerLTXV** 进行生成。可使用外部 sigmas 实现自定义采样调度。

## Requirements | 系统要求

**[English]**
- ComfyUI with `comfy_api` support
- LTXV model files
- Optional: Audio VAE for sound generation

**[中文]**
- 支持 `comfy_api` 的 ComfyUI 环境
- LTXV 模型文件
- 可选：Audio VAE 用于声音生成

## Note | 注意事项

**[English]**
Designed for professional workflows requiring pixel-perfect color fidelity and dynamic preservation. All parameters exposed for fine-grained control.

**[中文]**
专为需要像素级色彩保真和动态保留的专业工作流设计，所有参数开放用于精细控制。

看到这里了，请给我点颗星星，谢谢！
If you find this helpful, please star the repository. Thank you!
