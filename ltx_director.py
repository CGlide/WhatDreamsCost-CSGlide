# --- START OF FILE ltx_director.py ---

import logging
import json
import base64
import io as _io
import math
import time

import numpy as np
import torch
import av
from PIL import Image

import os
import folder_paths
import comfy.model_management

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)

from .patches import detect_model_type, apply_patches

log = logging.getLogger(__name__)

# Custom socket type shared with LTXSequencer
GuideData = io.Custom("GUIDE_DATA")


def _preprocess_prompts_with_characters(global_prompt, local_prompts, char1="", char2="", char3=""):
    """Invisibly swaps out @character1/@char1 tags with their high-fidelity VLM descriptions."""
    gp = global_prompt
    char1 = char1 if char1 else ""
    char2 = char2 if char2 else ""
    char3 = char3 if char3 else ""
    
    # Process Global Prompt
    for tag in ["@character1", "@char1"]:
        if tag in gp:
            gp = gp.replace(tag, char1)
    for tag in ["@character2", "@char2"]:
        if tag in gp:
            gp = gp.replace(tag, char2)
    for tag in ["@character3", "@char3"]:
        if tag in gp:
            gp = gp.replace(tag, char3)
            
    # Process Local Timeline Prompts
    locals_list = [p.strip() for p in local_prompts.split("|")] if local_prompts else []
    processed_locals = []
    for lp in locals_list:
        for tag in ["@character1", "@char1"]:
            if tag in lp:
                lp = lp.replace(tag, char1)
        for tag in ["@character2", "@char2"]:
            if tag in lp:
                lp = lp.replace(tag, char2)
        for tag in ["@character3", "@char3"]:
            if tag in lp:
                lp = lp.replace(tag, char3)
        processed_locals.append(lp)
        
    return gp, " | ".join(processed_locals)


def _format_timeline_to_text(global_prompt, duration_frames, frame_rate, epsilon, 
                             custom_width, custom_height, resize_method,
                             timeline_data, local_prompts, segment_lengths, guide_strength):
    lines = []
    lines.append("LTX Director Timeline Export")
    lines.append("============================")
    lines.append("")
    lines.append("=== Global Parameters ===")
    lines.append(f"Global Prompt:\n{global_prompt}\n")
    
    fr = float(frame_rate) if frame_rate else 24.0
    if fr <= 0: fr = 24.0
    
    lines.append(f"Duration: {duration_frames} frames ({(duration_frames / fr):.2f}s @ {fr} FPS)")
    lines.append(f"Epsilon (Penalty Decay): {epsilon}")
    
    if custom_width > 0 or custom_height > 0:
        lines.append(f"Target Dimensions: {custom_width}x{custom_height} (Resize Method: {resize_method})")
    else:
        lines.append("Target Dimensions: Auto (Based on first image)")
        
    lines.append("")
    lines.append("=== Timeline Segments ===")
    
    # --- 1. Text Prompts (Extracted from direct inputs) ---
    locals_list = [p.strip() for p in local_prompts.split("|")] if local_prompts else []
    lengths_list = [l.strip() for l in segment_lengths.split(",")] if segment_lengths else []
    
    if locals_list and any(locals_list):
        lines.append("\n--- Text Prompts ---")
        current_frame = 0.0
        for i, prompt in enumerate(locals_list):
            try:
                len_f = float(lengths_list[i]) if i < len(lengths_list) and lengths_list[i] else 0.0
            except ValueError:
                len_f = 0.0
            
            start_f = current_frame
            end_f = start_f + len_f
            
            start_s = start_f / fr
            end_s = end_f / fr
            len_s = len_f / fr
            
            lines.append(f"\n[Prompt {i+1}]")
            lines.append(f"Time: {start_s:.2f}s - {end_s:.2f}s (Duration: {len_s:.2f}s)")
            lines.append(f"Frames: {start_f:.1f} - {end_f:.1f} (Length: {len_f:.1f})")
            lines.append(f"Prompt:\n{prompt}")
            lines.append("-" * 40)
            
            current_frame += len_f
            
    # --- 2. Images & Audio (Extracted from JSON) ---
    try:
        tdata = json.loads(timeline_data) if timeline_data else {}
        segs = tdata.get("segments", [])
        
        img_segs = [s for s in segs if s.get("type", "image") == "image"]
        img_segs.sort(key=lambda s: float(s.get("start", 0)))
        if img_segs:
            lines.append("\n--- Image Guides ---")
            strengths = [float(x.strip()) for x in guide_strength.split(",")] if guide_strength and guide_strength.strip() else []
            for i, seg in enumerate(img_segs):
                start_f = float(seg.get("start", 0))
                start_s = start_f / fr
                strength = strengths[i] if i < len(strengths) else 1.0
                
                lines.append(f"\n[Image {i+1}]")
                lines.append(f"Time Inserted: {start_s:.2f}s (Frame {start_f:.1f})")
                lines.append(f"Guide Strength: {strength}")
                lines.append("-" * 40)
                
        audio_segs = [s for s in segs if s.get("type", "audio") == "audio"]
        audio_segs.sort(key=lambda s: float(s.get("start", 0)))
        if audio_segs:
            lines.append("\n--- Audio Segments ---")
            for i, seg in enumerate(audio_segs):
                start_f = float(seg.get("start", 0))
                len_f = float(seg.get("length", 0))
                start_s = start_f / fr
                len_s = len_f / fr
                file_name = seg.get("fileName", "Unknown")
                lines.append(f"\n[Audio {i+1}] {file_name}")
                lines.append(f"Time: {start_s:.2f}s (Frame {start_f:.1f}) | Duration: {len_s:.2f}s")
                lines.append("-" * 40)
                
    except Exception as e:
        lines.append(f"\n[Note: Could not parse detailed timeline JSON for images/audio. Error: {e}]")
        
    return "\n".join(lines)


# Register API endpoint for instant export from the JS UI if needed
try:
    import server
    from aiohttp import web

    @server.PromptServer.instance.routes.post("/ltx_director/export_timeline")
    async def export_timeline_endpoint(request):
        try:
            data = await request.json()
            global_prompt = data.get("global_prompt", "")
            timeline_data = data.get("timeline_data", "{}")
            duration_frames = int(data.get("duration_frames", 0))
            frame_rate = float(data.get("frame_rate", 24))
            epsilon = float(data.get("epsilon", 0.001))
            custom_width = int(data.get("custom_width", 0))
            custom_height = int(data.get("custom_height", 0))
            resize_method = data.get("resize_method", "maintain aspect ratio")
            local_prompts = data.get("local_prompts", "")
            segment_lengths = data.get("segment_lengths", "")
            guide_strength = data.get("guide_strength", "")
            
            formatted_text = _format_timeline_to_text(
                global_prompt, duration_frames, frame_rate, epsilon, 
                custom_width, custom_height, resize_method,
                timeline_data, local_prompts, segment_lengths, guide_strength
            )
            
            out_dir = folder_paths.get_output_directory()
            filename = f"ltx_director_prompts_{int(time.time())}.txt"
            filepath = os.path.join(out_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(formatted_text)
                
            return web.json_response({
                "status": "success", 
                "filepath": filepath, 
                "filename": filename, 
                "content": formatted_text
            })
        except Exception as e:
            log.error(f"[PromptRelay] Export timeline endpoint error: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)
except Exception as e:
    log.warning(f"[PromptRelay] Could not register /ltx_director/export_timeline endpoint: {e}")


def _load_image_tensor(seg: dict) -> torch.Tensor:
    """Decode an image from the ComfyUI input folder (if imageFile provided) or fallback to base64
    to a ComfyUI-style image tensor of shape [1, H, W, 3], float32 in [0, 1]."""
    if seg.get("imageFile"):
        file_path = os.path.join(folder_paths.get_input_directory(), seg["imageFile"])
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

    b64_str = seg.get("imageB64", "")
    if not b64_str or b64_str.startswith("/view?"):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int, method: str, divisible_by: int) -> torch.Tensor:
    """Resize a [1, H, W, 3] float32 tensor to target dimensions using the given method,
    then snap the final dimensions to be divisible by `divisible_by`."""
    from PIL import Image as _PilImage
    import torchvision.transforms.functional as TF

    def snap(val, div):
        return max(div, (val // div) * div)

    tw = snap(target_w, divisible_by)
    th = snap(target_h, divisible_by)

    img_np = (tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = _PilImage.fromarray(img_np)
    src_w, src_h = pil.size

    if method == "stretch to fit":
        resized = pil.resize((tw, th), _PilImage.LANCZOS)

    elif method == "maintain aspect ratio":
        ratio = min(tw / src_w, th / src_h)
        new_w = int(src_w * ratio)
        new_h = int(src_h * ratio)
        new_w = snap(new_w, divisible_by)
        new_h = snap(new_h, divisible_by)
        resized = pil.resize((new_w, new_h), _PilImage.LANCZOS)

    elif method == "pad":
        ratio = min(tw / src_w, th / src_h)
        new_w = snap(int(src_w * ratio), divisible_by)
        new_h = snap(int(src_h * ratio), divisible_by)
        inner = pil.resize((new_w, new_h), _PilImage.LANCZOS)
        resized = _PilImage.new("RGB", (tw, th), (0, 0, 0))
        resized.paste(inner, ((tw - new_w) // 2, (th - new_h) // 2))

    elif method == "crop":
        ratio = max(tw / src_w, th / src_h)
        new_w = int(src_w * ratio)
        new_h = int(src_h * ratio)
        inner = pil.resize((new_w, new_h), _PilImage.LANCZOS)
        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        resized = inner.crop((left, top, left + tw, top + th))

    else:
        resized = pil.resize((tw, th), _PilImage.LANCZOS)

    arr = np.array(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _compress_image(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    """Apply H.264 compression artefacts to a [1, H, W, 3] float32 tensor (ComfyUI image format).
    crf=0 means no compression. Uses PyAV to encode/decode a single frame in-memory."""
    if crf == 0:
        return tensor
    img = tensor[0]  # [H, W, 3]
    # Dimensions must be even for H.264
    h = (img.shape[0] // 2) * 2
    w = (img.shape[1] // 2) * 2
    img_np = (img[:h, :w] * 255.0).byte().cpu().numpy()  # uint8 [H, W, 3]

    try:
        buf = _io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=1)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "ultrafast"}
        frame = av.VideoFrame.from_ndarray(img_np, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()

        buf.seek(0)
        container_r = av.open(buf, mode="r")
        decoded = None
        for frame_r in container_r.decode(video=0):
            decoded = frame_r.to_ndarray(format="rgb24")  # [H, W, 3]
            break
        container_r.close()

        if decoded is None:
            return tensor
        arr = torch.from_numpy(decoded.astype(np.float32) / 255.0).to(tensor.device, tensor.dtype)
        # Re-embed into original tensor shape (may have been cropped by even-rounding)
        out = tensor.clone()
        out[0, :h, :w] = arr
        return out
    except Exception as e:
        log.warning("[PromptRelay] img_compression encode/decode failed: %s", e)
        return tensor


def _build_combined_audio(timeline_data_str: str, duration_frames: int, frame_rate: float) -> dict:
    """Parses timeline JSON, loads/trims audio directly from memory using PyAV, 
    and aligns to a global timeline yielding ComfyUI's format.
    Output length explicitly mimics the timeline's duration_frames length."""
    target_sr = 44100
    total_samples = max(1, int(math.ceil(duration_frames / frame_rate * target_sr)))
    empty_audio = {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32), "sample_rate": target_sr}

    if not timeline_data_str:
        return empty_audio

    try:
        data = json.loads(timeline_data_str)
        audio_segs = data.get("audioSegments", [])
    except Exception:
        return empty_audio

    if not audio_segs:
        return empty_audio

    out_waveform = torch.zeros((2, total_samples), dtype=torch.float32)

    for seg in audio_segs:
        buffer = None
        if seg.get("audioFile"):
            file_path = os.path.join(folder_paths.get_input_directory(), seg["audioFile"])
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    buffer = _io.BytesIO(f.read())
        
        if not buffer and seg.get("audioB64"):
            b64 = seg.get("audioB64")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                audio_bytes = base64.b64decode(b64)
                buffer = _io.BytesIO(audio_bytes)
            except:
                pass
                
        if not buffer:
            continue

        try:
            clip_frames = []
            
            # Use PyAV to decode directly from memory buffer
            with av.open(buffer) as container:
                stream = container.streams.audio[0]
                
                # Setup resampler to ensure output is 44.1kHz, Stereo, Float32 Planar
                resampler = av.AudioResampler(
                    format='fltp',
                    layout='stereo',
                    rate=target_sr,
                )
                
                for frame in container.decode(stream):
                    for resampled_frame in resampler.resample(frame):
                        # to_ndarray() on fltp gives shape (channels, samples)
                        arr = resampled_frame.to_ndarray()
                        clip_frames.append(torch.from_numpy(arr))
                
                # Flush the resampler to get any remaining samples
                for resampled_frame in resampler.resample(None):
                    arr = resampled_frame.to_ndarray()
                    clip_frames.append(torch.from_numpy(arr))

            if not clip_frames:
                continue

            # Concatenate all frame blocks along the samples dimension (dim 1)
            waveform = torch.cat(clip_frames, dim=1) # Shape: [2, total_clip_samples]

            # Calculate interactive trim boundaries
            trim_start_frames = float(seg.get("trimStart", 0))
            length_frames = float(seg.get("length", 1))
            start_frames = float(seg.get("start", 0))

            start_sample_src = int(trim_start_frames / frame_rate * target_sr)
            length_samples = int(length_frames / frame_rate * target_sr)
            end_sample_src = start_sample_src + length_samples

            if start_sample_src < 0: start_sample_src = 0
            if end_sample_src > waveform.shape[1]:
                end_sample_src = waveform.shape[1]

            actual_length = end_sample_src - start_sample_src
            if actual_length <= 0: continue

            # Extract the correct segment of the audio
            clip_waveform = waveform[:, start_sample_src:end_sample_src]

            # Position onto the timeline
            start_sample_dst = int(start_frames / frame_rate * target_sr)
            
            if start_sample_dst >= out_waveform.shape[1]:
                continue
                
            end_sample_dst = start_sample_dst + actual_length

            # Clip any trailing overflow so we don't index past the timeline bounds
            if end_sample_dst > out_waveform.shape[1]:
                actual_length = out_waveform.shape[1] - start_sample_dst
                clip_waveform = clip_waveform[:, :actual_length]
                end_sample_dst = start_sample_dst + actual_length
                
            if actual_length <= 0:
                continue

            # Additive composite (allows clips overlapping to sum together naturally)
            out_waveform[:, start_sample_dst:end_sample_dst] += clip_waveform

        except Exception as e:
            log.warning("[PromptRelay] Audio process error for segment %s: %s", seg.get("fileName"), e)
            continue

    return {"waveform": out_waveform.unsqueeze(0), "sample_rate": target_sr}


def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """Convert pixel-space segment lengths to integer latent-space lengths using the
    largest-remainder method. Targets the full `latent_frames` when the pixel sum looks
    like full coverage (within one stride of latent_frames * stride). Otherwise targets
    round(total_pixel / temporal_stride) so partial-coverage timelines stay partial.
    """
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    # Within one frame of full → user clearly intended full coverage; pin to latent_frames.
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    # Ensure every segment has ≥ 1 latent frame (steal from the largest if needed).
    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(
                f"PromptRelay: '{name}' arrived as None. "
                "Likely causes: a stale workflow JSON saved with null, the timeline "
                "editor's web extension failing to load, or an upstream node returning None. "
                "Set the field to an empty string or fix the upstream connection."
            )

    # Split prompts but do NOT filter out empty ones yet, so we can detect them
    locals_list = [p.strip() for p in local_prompts.split("|")]
    
    # Check if any specific segment is empty
    for p in locals_list:
        if not p:
            raise ValueError("There is a segment on the timeline missing a prompt!")

    if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
        raise ValueError("At least one local prompt is required.")

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(float(x.strip())) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[PromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[PromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[PromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning


class LTXDirector(io.ComfyNode):
    """WYSIWYG timeline variant — segments and lengths come from a visual editor in the node UI."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirector",
            display_name="LTX Director",
            category="WhatDreamsCost",
            description=(
                "Same as Prompt Relay Encode, but local prompts and segment lengths are edited "
                "visually as draggable blocks on a timeline. The duration_frames input only sets the "
                "timeline scale (pixel space) — actual frame count is still read from the latent."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("audio_vae", optional=True, tooltip="Optional. Connect an Audio VAE to generate audio latents."),
                io.Latent.Input("optional_latent", optional=True, tooltip="Optional. Connect a latent to override the auto-generated one."),
                io.String.Input("global_prompt", multiline=True, default="", tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context."),
                io.Int.Input("duration_frames", default=120, min=1, max=10000, step=1, tooltip="Total timeline length in pixel-space frames. Used by the editor for visual scale only."),
                io.Float.Input("duration_seconds", default=5.0, min=0.1, max=1000.0, step=0.01, tooltip="Total timeline duration in seconds (computed/synced from frames)."),
                io.String.Input("timeline_data", default="", tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand)."),
                io.Boolean.Input("use_custom_audio", default=False, tooltip="Toggle between using timeline audio (ON) and generating audio from scratch (OFF)."),
                io.String.Input("local_prompts", multiline=True, default="", tooltip="Auto-populated from the timeline editor."),
                io.String.Input("segment_lengths", default="", tooltip="Auto-populated from the timeline editor (pixel-space frame counts)."),
                io.Float.Input("epsilon", default=0.001, min=0.0001, max=0.99, step=0.0001, tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher."),
                io.Float.Input("frame_rate", default=24.0, min=1.0, max=240.0, step=1.0, tooltip="Frames per second — only affects how time is displayed in the timeline editor when time_units is set to 'seconds'."),
                io.Combo.Input("display_mode", options=["frames", "seconds"], default="seconds", tooltip="Display the ruler, segment ranges, length input, and total in frames or seconds. Internal storage is always pixel-space frames."),
                io.String.Input("guide_strength", default="", tooltip="Auto-populated from the timeline editor (comma-separated guide strengths for image segments)."),
                io.Int.Input("custom_width", default=0, min=0, max=8192, step=1, tooltip="Target output width for all image segments. Set to 0 to use the original image width."),
                io.Int.Input("custom_height", default=0, min=0, max=8192, step=1, tooltip="Target output height for all image segments. Set to 0 to use the original image height."),
                io.Combo.Input("resize_method", options=["maintain aspect ratio", "stretch to fit", "pad", "crop"], default="maintain aspect ratio", tooltip="How to resize image segments to fit the target dimensions."),
                io.Int.Input("divisible_by", default=32, min=1, max=256, step=1, tooltip="Snap the final output image dimensions to be divisible by this number (e.g. 32 for LTX)."),
                io.Int.Input("img_compression", default=18, min=0, max=100, step=1, tooltip="H.264 CRF compression to apply to each guide image. 0 = no compression, higher = more artefacts."),
                io.Boolean.Input("save_prompts_to_file", default=False, optional=True, tooltip="Save the timeline prompts and parameters to a text file in your ComfyUI output directory during execution."),
                io.Image.Input("reference_image", optional=True, tooltip="Invisible character sheet/style reference. Can also be a Batch of images. Automatically appended to the sequence out-of-bounds."),
                io.Image.Input("reference_image_2", optional=True, tooltip="Second optional reference image."),
                io.Image.Input("reference_image_3", optional=True, tooltip="Third optional reference image."),
                io.Float.Input("reference_strength", default=1.0, min=0.0, max=5.0, step=0.05, optional=True, tooltip="Guide strength for the reference images."),
                
                # New descriptive inputs for automatic under-the-hood character replacement
                io.String.Input("char1_description", multiline=True, default="", optional=True, tooltip="Plug in a detailed description for @character1 / @char1. You can write it manually or connect a VLM/Caption node."),
                io.String.Input("char2_description", multiline=True, default="", optional=True, tooltip="Plug in a detailed description for @character2 / @char2."),
                io.String.Input("char3_description", multiline=True, default="", optional=True, tooltip="Plug in a detailed description for @character3 / @char3."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="video_latent", tooltip="Auto-generated LTXV empty latent (only populated when no latent is connected)."),
                io.Latent.Output(display_name="audio_latent", tooltip="Auto-generated audio latent (uses custom audio if enabled)."),
                GuideData.Output(display_name="guide_data"),
                io.Float.Output(display_name="frame_rate", tooltip="The frame rate used for the timeline."),
                io.Audio.Output(display_name="combined_audio", tooltip="Combined timeline audio layout."),
                io.Int.Output(display_name="clean_latent_frames", tooltip="Plug this into a CleanLatentSlice node 'length' to effortlessly cut off the invisible reference image."),
                io.Int.Output(display_name="clean_pixel_frames", tooltip="Plug this into your Video Combine 'frame_load_cap' to effortlessly cut off the invisible reference image."),
            ],
        )

    @classmethod
    def execute(cls, model, clip, global_prompt, duration_frames, duration_seconds,
                timeline_data, local_prompts, segment_lengths, guide_strength="", epsilon=1e-3,
                frame_rate=24.0, display_mode="seconds",
                custom_width=768, custom_height=512, resize_method="maintain aspect ratio",
                divisible_by=32, img_compression=0, audio_vae=None, optional_latent=None,
                use_custom_audio=False, save_prompts_to_file=False,
                reference_image=None, reference_image_2=None, reference_image_3=None, reference_strength=1.0,
                char1_description="", char2_description="", char3_description="") -> io.NodeOutput:

        # --- Calculate Clean Output Bounds First ---
        clean_pixel_frames = duration_frames + 1
        clean_latent_frames = ((clean_pixel_frames - 1) // 8) + 1

        # --- Build guide_data from image segments FIRST (to derive output dimensions) ---
        guide_data = {"images": [], "insert_frames": [], "strengths": [], "frame_rate": float(frame_rate)}
        derived_w, derived_h = custom_width, custom_height
        
        # Consolidate all reference inputs into a flat list, unraveling any image batches.
        refs_to_process = []
        for ref_input in (reference_image, reference_image_2, reference_image_3):
            if ref_input is not None:
                for i in range(ref_input.shape[0]):
                    refs_to_process.append(ref_input[i:i+1])
                    
        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
            img_segs = [
                s for s in tdata.get("segments", [])
                if s.get("type", "image") == "image"
                and (s.get("imageFile") or s.get("imageB64"))
                and int(s.get("start", 0)) < duration_frames  # exclude segments fully outside duration
            ]
            img_segs.sort(key=lambda s: s["start"])

            strengths = []
            if guide_strength.strip():
                strengths = [float(x.strip()) for x in guide_strength.split(",") if x.strip()]

            for idx, seg in enumerate(img_segs):
                tensor = _load_image_tensor(seg)

                # Apply resize
                src_h, src_w = tensor.shape[1], tensor.shape[2]

                def snap(val, div):
                    return max(div, (val // div) * div)

                if custom_width > 0 and custom_height > 0:
                    # Both dimensions set — apply selected resize_method (pad, crop, stretch, maintain AR)
                    tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by)
                elif custom_width > 0:
                    # Width only — scale height from AR, snap both, then resize to exact dimensions
                    tgt_w = snap(custom_width, divisible_by)
                    tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
                elif custom_height > 0:
                    # Height only — scale width from AR, snap both, then resize to exact dimensions
                    tgt_h = snap(custom_height, divisible_by)
                    tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
                else:
                    # Both zero — keep original dimensions, just snap to divisible_by
                    tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)


                # Apply compression
                if img_compression > 0:
                    tensor = _compress_image(tensor, img_compression)

                # Record dimensions of the first processed image for latent generation
                if idx == 0:
                    derived_h = tensor.shape[1]
                    derived_w = tensor.shape[2]

                strength = strengths[idx] if idx < len(strengths) else 1.0
                guide_data["images"].append(tensor)
                
                # Keep timeline images at their exact timeline frames (no shifting)
                guide_data["insert_frames"].append(int(seg["start"]))
                
                guide_data["strengths"].append(float(strength))
            
            # If no images were loaded from the timeline, create a dummy image at strength 0
            # to prevent artifacts in text-to-video mode.
            if not guide_data["images"] and not refs_to_process:
                w = derived_w if derived_w > 0 else 768
                h = derived_h if derived_h > 0 else 512
                w = (w // 32) * 32
                h = (h // 32) * 32
                
                dummy_image = torch.zeros((1, h, w, 3), dtype=torch.float32)
                guide_data["images"].append(dummy_image)
                guide_data["insert_frames"].append(0)
                guide_data["strengths"].append(0.0)
                
                derived_w = w
                derived_h = h
        except Exception as e:
            log.warning("[PromptRelay] Could not build guide_data: %s", e)

        # --- Handle Reference Image Injection (End-Hiding) ---
        if refs_to_process:
            if optional_latent is not None:
                log.warning("[PromptRelay] You connected reference images AND an external 'optional_latent'. Make sure your custom latent is long enough to fit the appended reference frames!")

            for i, single_ref in enumerate(refs_to_process):
                src_h, src_w = single_ref.shape[1], single_ref.shape[2]
                def snap(val, div): return max(div, (val // div) * div)
                
                if custom_width > 0 and custom_height > 0:
                    ref_tensor = _resize_image(single_ref, custom_width, custom_height, resize_method, divisible_by)
                elif custom_width > 0:
                    tgt_w = snap(custom_width, divisible_by)
                    tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
                    ref_tensor = _resize_image(single_ref, tgt_w, tgt_h, "stretch to fit", divisible_by)
                elif custom_height > 0:
                    tgt_h = snap(custom_height, divisible_by)
                    tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
                    ref_tensor = _resize_image(single_ref, tgt_w, tgt_h, "stretch to fit", divisible_by)
                else:
                    ref_tensor = _resize_image(single_ref, src_w, src_h, "maintain aspect ratio", divisible_by)
                
                if img_compression > 0:
                    ref_tensor = _compress_image(ref_tensor, img_compression)
                    
                guide_data["images"].append(ref_tensor)
                
                # Safely hide reference sheets at the very end of the video sequence
                insert_point = (clean_latent_frames + i) * 8
                guide_data["insert_frames"].append(insert_point)
                guide_data["strengths"].append(float(reference_strength))
                
                # Record dimensions for empty latent generation from the FIRST reference image if none exist
                if derived_w <= 0 or derived_h <= 0:
                    derived_w = ref_tensor.shape[2]
                    derived_h = ref_tensor.shape[1]

        # --- Auto-generate LTXV latent ---
        total_latents = clean_latent_frames + len(refs_to_process)
        ltxv_length = ((total_latents - 1) * 8) + 1  # Map back to pixel frames for the empty latent logic

        if optional_latent is None:
            latent_w = max(32, (derived_w // 32) * 32)
            latent_h = max(32, (derived_h // 32) * 32)
            
            samples = torch.zeros(
                [1, 128, total_latents, latent_h // 32, latent_w // 32],
                device=comfy.model_management.intermediate_device(),
            )
            latent = {"samples": samples}
            log.info(
                "[PromptRelay] Auto-generated LTXV latent: %dx%d, %d pixel frames (%d latent frames, %d refs hidden)",
                latent_w, latent_h, ltxv_length, total_latents, len(refs_to_process)
            )
        else:
            latent = optional_latent

        # --- Preprocess Prompts with Character Tags ---
        processed_global, processed_local = _preprocess_prompts_with_characters(
            global_prompt, local_prompts, char1_description, char2_description, char3_description
        )

        patched, conditioning = _encode_relay(
            model, clip, latent, processed_global, processed_local, segment_lengths, epsilon,
        )

        # --- Build Audio Output ---
        audio_out = _build_combined_audio(timeline_data, ltxv_length, float(frame_rate))

        # --- Audio Latent Generation ---
        audio_latent = {}
        
        if audio_vae is not None:
            def get_empty_latent():
                # Support both raw AudioVAE objects and ComfyUI VAE wrappers.
                inner = getattr(audio_vae, "first_stage_model", audio_vae)
                z_channels = audio_vae.latent_channels
                audio_freq = inner.latent_frequency_bins
                num_audio_latents = inner.num_of_latents_from_frames(ltxv_length, float(frame_rate))
                audio_latents = torch.zeros(
                    (1, z_channels, num_audio_latents, audio_freq),
                    device=comfy.model_management.intermediate_device(),
                )
                return {"samples": audio_latents, "type": "audio"}

            if use_custom_audio:
                try:
                    if audio_out is not None:
                        waveform = audio_out["waveform"]
                        if waveform.ndim == 2:
                            waveform = waveform.unsqueeze(0)
                        if waveform.ndim != 3:
                            raise ValueError(f"Expected custom audio waveform with 2 or 3 dims, got shape {tuple(waveform.shape)}")

                        if hasattr(audio_vae, "first_stage_model"):
                            latent_samples = audio_vae.encode(waveform.movedim(1, -1))
                        else:
                            latent_samples = audio_vae.encode({
                                "waveform": waveform,
                                "sample_rate": audio_out["sample_rate"],
                            })
                        
                        if latent_samples.numel() == 0:
                            raise ValueError("Encoded audio latent is empty (0 elements).")
                        
                        mask = torch.full(
                            (1, latent_samples.shape[-2], latent_samples.shape[-1]), 
                            0.0, 
                            dtype=torch.float32, 
                            device=comfy.model_management.intermediate_device()
                        )
                        
                        audio_latent = {
                            "samples": latent_samples,
                            "type": "audio",
                            "noise_mask": mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
                        }
                        log.info("[PromptRelay] Generated custom audio latent with noise mask (value=0.0).")
                    else:
                        raise ValueError("No audio waveform to encode.")
                except Exception as e:
                    log.error("[PromptRelay] Failed to generate custom audio latent: %s", e)
                    raise e
            else:
                try:
                    audio_latent = get_empty_latent()
                    log.info("[PromptRelay] Auto-generated empty audio latent.")
                except Exception as e:
                    log.error("[PromptRelay] Could not generate empty audio latent: %s", e)
                    raise e

        # --- Save formatted Prompts logic ---
        if save_prompts_to_file:
            try:
                formatted_text = _format_timeline_to_text(
                    processed_global, duration_frames, float(frame_rate), epsilon,
                    custom_width, custom_height, resize_method,
                    timeline_data, processed_local, segment_lengths, guide_strength
                )
                out_dir = folder_paths.get_output_directory()
                filename = f"ltx_director_prompts_{int(time.time())}.txt"
                filepath = os.path.join(out_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(formatted_text)
                log.info(f"[PromptRelay] Saved timeline prompts to {filepath}")
            except Exception as e:
                log.warning(f"[PromptRelay] Failed to save prompts to txt: {e}")

        return io.NodeOutput(
            patched, 
            conditioning, 
            latent, 
            audio_latent, 
            guide_data, 
            float(frame_rate), 
            audio_out, 
            clean_latent_frames, 
            clean_pixel_frames
        )


NODE_CLASS_MAPPINGS = {
    "LTXDirector": LTXDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
}