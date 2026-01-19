import torch
import comfy.model_management
import comfy.samplers
import comfy.sample
import latent_preview
import comfy.utils
import comfy.nested_tensor
from typing_extensions import override
from comfy_api.latest import io


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
                io.Combo.Input("return_noise", options=["disable", "enable"]),
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
                latent_image, start_at_step, end_at_step, return_noise, sigmas=None) -> io.NodeOutput:
        
        force_full_denoise = True
        if return_noise == "enable":
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
