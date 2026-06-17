from comfy_extras.nodes_lt import LTXVAddGuide
import torch
import comfy.utils
from comfy_api.latest import io
from .ltx_director import GuideData


class LTXDirectorGuide(LTXVAddGuide):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirectorGuide",
            display_name="LTX Director Guide",
            category="WhatDreamsCost",
            description=(
                "Applies guide images from a Prompt Relay Timeline node at the frame positions "
                "and strengths defined on the timeline. Connect guide_data from the timeline node."
            ),
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive conditioning to add guide keyframe info to."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning to add guide keyframe info to."),
                io.Vae.Input("vae", tooltip="Video VAE used to encode the guide images."),
                io.Latent.Input("latent", tooltip="Video latent — guides are inserted into this latent."),
                GuideData.Input("guide_data", tooltip="Guide data produced by Prompt Relay Encode (Timeline)."),
                io.Float.Input("scale_by", default=1.0, min=0.01, max=8.0, step=0.01, tooltip="Scale the latent by this factor."),
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], default="bicubic", tooltip="Method used to upscale/downscale the latent."),
                io.Float.Input("msr_strength", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Licon MSR only: per-stage reference strength override. 0 = use the Director's value (full pull, right for stage 1). On a refinement/upscale stage set ~0.4 to hold detail without the references repainting the opening (fixes stage-2 mist/ghosting)."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Video latent with guide frames applied."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, guide_data, scale_by=1.0, upscale_method="bicubic", msr_strength=0.0) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula

        # Clone latents to avoid mutating upstream nodes
        latent_image = latent["samples"].clone()

        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].clone()
        else:
            batch, _, latent_frames, latent_height, latent_width = latent_image.shape
            noise_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=latent_image.device,
            )

        # Apply scale factor if not 1.0
        if scale_by != 1.0:
            B, C, F, H, W = latent_image.shape
            width = round(W * scale_by)
            height = round(H * scale_by)
            
            # Reshape to 4D for common_upscale
            latent_4d = latent_image.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            latent_resized_4d = comfy.utils.common_upscale(latent_4d, width, height, upscale_method, "disabled")
            latent_image = latent_resized_4d.reshape(B, F, C, height, width).permute(0, 2, 1, 3, 4)

            # Also resize noise mask if it's not a broadcasted mask
            if noise_mask.shape[-1] > 1 or noise_mask.shape[-2] > 1:
                mask_4d = noise_mask.permute(0, 2, 1, 3, 4).reshape(B * F, 1, H, W)
                mask_resized_4d = comfy.utils.common_upscale(mask_4d, width, height, upscale_method, "disabled")
                noise_mask = mask_resized_4d.reshape(B, F, 1, height, width).permute(0, 2, 1, 3, 4)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        # MSR mode: the Director handed us the raw references; inject them at THIS node's
        # resolution (so a half-res Stage 1 and a full-res Stage 2 each stay self-consistent).
        msr = guide_data.get("msr")
        if msr is not None:
            return cls._inject_msr(positive, negative, vae, latent_image, noise_mask, msr, msr_strength)

        images = guide_data.get("images", [])
        insert_frames = guide_data.get("insert_frames", [])
        strengths = guide_data.get("strengths", [])

        for idx, img_tensor in enumerate(images):
            f_idx = insert_frames[idx] if idx < len(insert_frames) else 0
            strength = strengths[idx] if idx < len(strengths) else 1.0

            image_1, t = cls.encode(vae, latent_width, latent_height, img_tensor, scale_factors)
            frame_idx, latent_idx = cls.get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)

            assert latent_idx + t.shape[2] <= latent_length, (
                f"Guide image {idx + 1}: conditioning frames exceed the length of the latent sequence."
            )

            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive, negative, frame_idx, latent_image, noise_mask, t, strength, scale_factors,
            )

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

    @classmethod
    def _inject_msr(cls, positive, negative, vae, latent_image, noise_mask, msr, msr_strength=0.0):
        """Inject the Licon MSR references at this node's current resolution."""
        from nodes import NODE_CLASS_MAPPINGS
        from .ltx_director import _execute_comfy_node, _unpack

        IC = NODE_CLASS_MAPPINGS.get("LTXAddVideoICLoRAGuide")
        AG = NODE_CLASS_MAPPINGS.get("LTXVAddGuide")
        if IC is None or AG is None:
            raise ValueError("MSR mode needs LTXAddVideoICLoRAGuide (ComfyUI-LTXVideo) and LTXVAddGuide (core LTXV nodes).")

        def clamp01(v):
            return max(0.0, min(1.0, float(v)))

        slideshow = msr["slideshow"]
        keyframes = msr["keyframes"]
        prefix_latents = int(msr["prefix_latents"])
        # Per-stage override: msr_strength > 0 wins (set it low on a refinement stage so the
        # references hold detail without repainting the opening); 0 = use the Director's value.
        strength = clamp01(msr_strength) if float(msr_strength) > 0.0 else clamp01(msr["strength"])
        downscale = float(msr["downscale"])

        # Pad the generated region so (pad + keyframes) == prefix_latents, so the downstream crop
        # (which trims the slideshow's temporal footprint) lands on the true clean length.
        pad_latents = max(0, prefix_latents - len(keyframes))
        if pad_latents > 0:
            B, C, F, H, W = latent_image.shape
            latent_image = torch.cat(
                [latent_image, torch.zeros((B, C, pad_latents, H, W), dtype=latent_image.dtype, device=latent_image.device)],
                dim=2,
            )
            mb, mc, mf, mh, mw = noise_mask.shape
            noise_mask = torch.cat(
                [noise_mask, torch.ones((mb, mc, pad_latents, mh, mw), dtype=noise_mask.dtype, device=noise_mask.device)],
                dim=2,
            )

        latent = {"samples": latent_image, "noise_mask": noise_mask}

        # CONDITIONING PATH: slideshow -> IC-LoRA -> keep conditioning, discard latent.
        cp, cn, _ = _unpack(_execute_comfy_node(
            IC, positive=positive, negative=negative, vae=vae, latent=latent, image=slideshow,
            frame_idx=0, strength=strength, latent_downscale_factor=downscale,
            crop="center", use_tiled_encode=False, tile_size=256, tile_overlap=64,
        ))

        positive, negative = cp, cn

        # LATENT PATH: one frozen keyframe per reference at frame 0 -> keep latent, drop cond.
        for kf in keyframes:
            _p, _n, latent = _unpack(_execute_comfy_node(
                AG, positive=positive, negative=negative, vae=vae, latent=latent,
                image=kf, frame_idx=0, strength=strength,
            ))

        return io.NodeOutput(positive, negative, latent)