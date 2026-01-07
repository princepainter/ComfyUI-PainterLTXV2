import torch
import comfy.model_management
import comfy.samplers
import comfy.sample
import latent_preview
import comfy.utils
import comfy.nested_tensor
import folder_paths
from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class PainterSamplerLTXV(io.ComfyNode):
    """Advanced sampler with LTXV audio-video separation and external sigmas support"""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterSamplerLTXV",
            category="sampling",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("add_noise", options=["enable", "disable"]),
                io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Int.Input("start_at_step", default=0, min=0, max=10000),
                io.Int.Input("end_at_step", default=10000, min=0, max=10000),
                io.Combo.Input("return_with_leftover_noise", options=["disable", "enable"]),
                io.Sigmas.Input("sigmas", optional=True, tooltip="Optional external sigmas input. When connected, scheduler parameter will be ignored"),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ]
        )

    @classmethod
    def execute(cls, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, 
                latent_image, start_at_step, end_at_step, return_with_leftover_noise, sigmas=None) -> io.NodeOutput:
        
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        latent = latent_image
        latent_image_tensor = latent["samples"]
        latent_image_tensor = comfy.sample.fix_empty_latent_channels(model, latent_image_tensor)
        latent["samples"] = latent_image_tensor

        if disable_noise:
            noise_tensor = torch.zeros(latent_image_tensor.size(), dtype=latent_image_tensor.dtype, 
                                      layout=latent_image_tensor.layout, device="cpu")
        else:
            batch_inds = latent.get("batch_index", None)
            noise_tensor = comfy.sample.prepare_noise(latent_image_tensor, noise_seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback_steps = len(sigmas) - 1 if sigmas is not None and len(sigmas) > 0 else steps
        callback = latent_preview.prepare_callback(model, callback_steps, x0_output)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        if sigmas is not None and len(sigmas) > 0:
            sampler = comfy.samplers.sampler_object(sampler_name)
            
            total_sigmas = len(sigmas) - 1
            start_step = min(start_at_step, total_sigmas)
            end_step = min(end_at_step, total_sigmas) if end_at_step < 10000 else total_sigmas
            
            if end_step <= start_step:
                end_step = start_step + 1
                
            sigmas_to_use = sigmas[start_step:end_step + 1] if start_step > 0 or end_step < total_sigmas else sigmas
            
            samples = comfy.sample.sample_custom(
                model, noise_tensor, cfg, sampler, sigmas_to_use, positive, negative, 
                latent_image_tensor, noise_mask=noise_mask, callback=callback, 
                disable_pbar=disable_pbar, seed=noise_seed
            )
        else:
            samples = comfy.sample.sample(
                model, noise_tensor, steps, cfg, sampler_name, scheduler, positive, negative, 
                latent_image_tensor, denoise=1.0, disable_noise=disable_noise, 
                start_step=start_at_step, last_step=end_at_step, 
                force_full_denoise=force_full_denoise, noise_mask=noise_mask, 
                callback=callback, disable_pbar=disable_pbar, seed=noise_seed
            )

        out = latent.copy()
        out["samples"] = samples

        video_latent = out.copy()
        audio_latent = out.copy()
        
        if isinstance(samples, comfy.nested_tensor.NestedTensor):
            latents = samples.unbind()
            if len(latents) >= 2:
                video_latent["samples"] = latents[0]
                audio_latent["samples"] = latents[1]
                
                if "noise_mask" in out and isinstance(out["noise_mask"], comfy.nested_tensor.NestedTensor):
                    masks = out["noise_mask"].unbind()
                    if len(masks) >= 2:
                        video_latent["noise_mask"] = masks[0]
                        audio_latent["noise_mask"] = masks[1]
            else:
                video_latent["samples"] = latents[0] if len(latents) > 0 else samples
                audio_latent["samples"] = torch.empty(0, device=samples.device, dtype=samples.dtype)
        else:
            video_latent["samples"] = samples
            audio_latent["samples"] = torch.empty(0, device=samples.device, dtype=samples.dtype)
            
            if "noise_mask" in out:
                video_latent["noise_mask"] = out["noise_mask"]
                audio_latent["noise_mask"] = torch.empty(0, device=out["noise_mask"].device, dtype=out["noise_mask"].dtype)

        return io.NodeOutput(out, video_latent, audio_latent)


class PainterLTXVtoVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterLTXVtoVideo",
            display_name="Painter LTXV to Video",
            category="latent/video/ltxv",
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input("image", optional=True),
                io.Vae.Input(id="audio_vae", display_name="Audio VAE", optional=True),
                io.Int.Input("width", default=768, min=64, max=4096, step=16),
                io.Int.Input("height", default=512, min=64, max=4096, step=16),
                io.Int.Input("length", default=97, min=1, max=1024, step=1),
                io.Float.Input("frame_rate", default=25.0, min=1.0, max=120.0, step=0.1),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, vae, image=None, audio_vae=None, width=768, height=512, length=97, frame_rate=25.0, batch_size=1):
        latent_frames = ((length - 1) // 8) + 1
        latent_height = height // 32
        latent_width = width // 32
        
        samples = torch.zeros(
            [batch_size, 128, latent_frames, latent_height, latent_width],
            device=comfy.model_management.intermediate_device()
        )

        if image is not None:
            if image.shape[1] != height or image.shape[2] != width:
                pixels = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            else:
                pixels = image
            encode_pixels = pixels[:, :, :, :3]
            t = vae.encode(encode_pixels)
            samples[:, :, :t.shape[2]] = t
            strength = 1.0
            conditioning_latent_frames_mask = torch.ones(
                (batch_size, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=samples.device,
            )
            conditioning_latent_frames_mask[:, :, :t.shape[2]] = 1.0 - strength
        else:
            conditioning_latent_frames_mask = torch.ones(
                (batch_size, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=samples.device,
            )

        video_latent = {
            "samples": samples,
            "noise_mask": conditioning_latent_frames_mask
        }

        if audio_vae is not None:
            z_channels = audio_vae.latent_channels
            audio_freq = audio_vae.latent_frequency_bins
            sampling_rate = int(audio_vae.sample_rate)
            num_audio_latents = audio_vae.num_of_latents_from_frames(length, int(frame_rate))

            audio_latents = torch.zeros(
                (batch_size, z_channels, num_audio_latents, audio_freq),
                device=comfy.model_management.intermediate_device(),
            )

            audio_latent = {
                "samples": audio_latents,
                "sample_rate": sampling_rate,
                "type": "audio",
                "noise_mask": torch.ones_like(audio_latents)
            }

            output = {}
            output.update(video_latent)
            output.update(audio_latent)
            video_noise_mask = video_latent.get("noise_mask", None)
            audio_noise_mask = audio_latent.get("noise_mask", None)

            if video_noise_mask is not None or audio_noise_mask is not None:
                if video_noise_mask is None:
                    video_noise_mask = torch.ones_like(video_latent["samples"])
                if audio_noise_mask is None:
                    audio_noise_mask = torch.ones_like(audio_latent["samples"])
                output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_noise_mask, audio_noise_mask))

            output["samples"] = comfy.nested_tensor.NestedTensor((video_latent["samples"], audio_latent["samples"]))

            return io.NodeOutput(output)
        else:
            return io.NodeOutput(video_latent)


class PainterSamplerLTXVExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PainterSamplerLTXV, PainterLTXVtoVideo]


async def comfy_entrypoint() -> PainterSamplerLTXVExtension:
    return PainterSamplerLTXVExtension()
