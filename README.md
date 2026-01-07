# ComfyUI-PainterLTXV2

**Advanced LTXV Sampling & Latent Preparation for ComfyUI**

ComfyUI custom nodes providing LTXV audio-video separation sampling and latent preparation capabilities for professional video generation workflows.

---

**ComfyUI 的 LTXV 高级采样与潜空间准备节点**

为专业视频生成工作流提供 LTXV 音视频分离采样和潜空间准备功能的 ComfyUI 自定义节点。

<img width="2111" height="869" alt="image" src="https://github.com/user-attachments/assets/f04aed96-acd5-4645-b655-187cfc17cf87" />


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

### PainterLTXVtoVideo


<img width="1251" height="656" alt="image" src="https://github.com/user-attachments/assets/6d6bea7e-18df-4c65-aa67-d671a620c831" />


**[English]**
- Precise latent dimension calculation for LTXV models
- Optional image input for first-frame conditioning
- Audio VAE integration for complete audio-video pipeline
- Automatic noise mask generation for temporal control

**[中文]**
- 精确的 LTXV 模型潜空间维度计算
- 可选图像输入用于首帧条件控制
- Audio VAE 集成，支持完整音视频 pipeline
- 自动生成噪声遮罩实现时序控制


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
