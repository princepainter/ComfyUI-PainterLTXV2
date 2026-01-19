import torch
import comfy.model_management
import comfy.utils
import comfy.nested_tensor
import math
from typing import Dict, Any, Tuple


class PainterLTX2V:
    """LTXV latent generator with first/last frame control"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE",),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 97, "min": 1, "max": 1024, "step": 1}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("latent", "video_latent", "audio_latent")
    FUNCTION = "execute"
    CATEGORY = "latent/video/ltxv"
    DESCRIPTION = "Generate LTXV latent with optional first/last frame control"

    def execute(self, video_vae, width=768, height=512, length=97, frame_rate=25.0,
                batch_size=1, audio_vae=None, start_image=None, end_image=None):
        
        # Hardcoded strength values
        first_strength = 1.0
        last_strength = 1.0
        
        # Step 1: Create empty latent tensor (always create new, no external latent input)
        samples, noise_mask = self._create_empty_latent(video_vae, width, height, length, batch_size)

        # Step 2: Apply first and last frame conditioning
        if start_image is not None or end_image is not None:
            samples, noise_mask = self._apply_frame_control(
                video_vae, samples, noise_mask, start_image, end_image,
                first_strength, last_strength
            )

        # Step 3: Build main output dictionary (no upscaling)
        output = {"samples": samples}
        if noise_mask is not None:
            output["noise_mask"] = noise_mask

        # Step 4: Add audio latent if audio_vae provided
        if audio_vae is not None:
            output = self._attach_audio_latent(output, audio_vae, length, frame_rate, batch_size)

        # Step 5: Split and return A/V outputs (no width/height output)
        return self._split_av_outputs(output, audio_vae is not None)

    def _create_empty_latent(self, video_vae, width, height, length, batch_size):
        """Create empty latent tensor for text-to-video generation"""
        latent_frames = ((length - 1) // 8) + 1
        latent_height = height // 32
        latent_width = width // 32
        
        samples = torch.zeros(
            [batch_size, 128, latent_frames, latent_height, latent_width],
            device=comfy.model_management.intermediate_device()
        )
        
        noise_mask = torch.ones(
            (batch_size, 1, latent_frames, 1, 1),
            dtype=torch.float32,
            device=samples.device,
        )
        
        return samples, noise_mask

    def _apply_frame_control(self, video_vae, samples, noise_mask, start_image, end_image,
                             first_strength, last_strength):
        """Embed first and last frames into latent with strength control"""
        batch, _, latent_frames, latent_height, latent_width = samples.shape
        
        _, height_scale_factor, width_scale_factor = video_vae.downscale_index_formula
        target_width = latent_width * width_scale_factor
        target_height = latent_height * height_scale_factor
        
        # Process start image
        if start_image is not None and first_strength > 0.0:
            start_latent = self._encode_image(video_vae, start_image, target_height, target_width)
            start_frames = start_latent.shape[2]
            
            embed_frames = min(start_frames, latent_frames)
            samples[:, :, :embed_frames] = start_latent[:, :, :embed_frames]
            noise_mask[:, :, :embed_frames] = 1.0 - first_strength
        
        # Process end image
        if end_image is not None and last_strength > 0.0:
            end_latent = self._encode_image(video_vae, end_image, target_height, target_width)
            end_frames = end_latent.shape[2]
            
            start_idx = latent_frames - end_frames
            if start_idx < 0:
                end_latent = end_latent[:, :, :latent_frames]
                start_idx = 0
                end_frames = latent_frames
            
            samples[:, :, start_idx:] = end_latent
            noise_mask[:, :, start_idx:] = 1.0 - last_strength
        
        return samples, noise_mask

    @staticmethod
    def _encode_image(video_vae, image, target_height, target_width):
        """Encode image to latent space with size adjustment"""
        if image.shape[1] != target_height or image.shape[2] != target_width:
            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1), target_width, target_height, "bilinear", "center"
            ).movedim(1, -1)
        else:
            pixels = image
        
        encode_pixels = pixels[:, :, :, :3]
        return video_vae.encode(encode_pixels)

    @staticmethod
    def _attach_audio_latent(video_dict, audio_vae, length, frame_rate, batch_size):
        """Attach audio latent to video latent using NestedTensor"""
        output = video_dict.copy()
        
        z_channels = audio_vae.latent_channels
        audio_freq = audio_vae.latent_frequency_bins
        sampling_rate = int(audio_vae.sample_rate)
        num_audio_latents = audio_vae.num_of_latents_from_frames(length, int(frame_rate))
        
        audio_latents = torch.zeros(
            (batch_size, z_channels, num_audio_latents, audio_freq),
            device=comfy.model_management.intermediate_device(),
        )
        
        video_samples = output["samples"]
        
        # Combine video and audio latents
        output["samples"] = comfy.nested_tensor.NestedTensor((video_samples, audio_latents))
        
        # Handle noise masks
        video_mask = output.get("noise_mask", None)
        audio_mask = torch.ones_like(audio_latents)
        
        if video_mask is not None:
            output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        
        return output

    def _split_av_outputs(self, output, has_audio):
        """Split combined A/V latent into separate outputs"""
        main_latent = output
        
        if has_audio and isinstance(output["samples"], comfy.nested_tensor.NestedTensor):
            latents = output["samples"].unbind()
            if len(latents) >= 2:
                video_latent = output.copy()
                audio_latent = output.copy()
                video_latent["samples"] = latents[0]
                audio_latent["samples"] = latents[1]
                
                if "noise_mask" in output and isinstance(output["noise_mask"], comfy.nested_tensor.NestedTensor):
                    masks = output["noise_mask"].unbind()
                    if len(masks) >= 2:
                        video_latent["noise_mask"] = masks[0]
                        audio_latent["noise_mask"] = masks[1]
                
                return (main_latent, video_latent, audio_latent)
        
        # No audio case
        video_latent = output
        empty_tensor = torch.empty(0, device=output["samples"].device, dtype=output["samples"].dtype)
        audio_latent = {"samples": empty_tensor}
        
        if "noise_mask" in output:
            video_latent["noise_mask"] = output["noise_mask"]
            audio_latent["noise_mask"] = torch.empty(0, device=output["noise_mask"].device, dtype=output["noise_mask"].dtype)
        
        return (main_latent, video_latent, audio_latent)


# Legacy support for ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "PainterLTX2V": PainterLTX2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterLTX2V": "Painter LTX2V",
}
