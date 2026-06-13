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


# Register API endpoint for instant character analysis via local Ollama
try:
    import server
    from aiohttp import web
    import aiohttp

    @server.PromptServer.instance.routes.post("/ltx_director/analyze_character")
    async def analyze_character_endpoint(request):
        try:
            data = await request.json()
            image_b64 = data.get("image_b64", "")
            char_index = int(data.get("char_index", 0))

            if not image_b64:
                return web.json_response({"status": "error", "message": "No image provided for analysis."})

            b64_list = image_b64 if isinstance(image_b64, list) else [image_b64]
            cleaned_b64_list = []
            for b64 in b64_list:
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                cleaned_b64_list.append(b64)

            if not cleaned_b64_list:
                return web.json_response({"status": "error", "message": "No valid base64 images decoded."})

            # Configured to use huihui_ai/qwen3.5-abliterated:2b on Ollama
            model_name = "huihui_ai/qwen3.5-abliterated:2b"
            
            # Detailed visual prompt for high-fidelity descriptions
            prompt = (
                "Describe the character's physical appearance in two concise sentences. "
                "Specify their hair color/style, face details, and their clothing type/color. "
                "Keep the entire response very brief."
            )

            log.info(f"[PromptRelay] Analyzing Character {char_index+1} with local Ollama model '{model_name}'...")

            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": cleaned_b64_list,
                "stream": False,
                "keep_alive": 0  # Fixed: Immediately unloads the model from VRAM after generating the description
            }

            ollama_url = "http://127.0.0.1:11434/api/generate"
            
            # Send the request asynchronously to Ollama without blocking ComfyUI's main thread
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(ollama_url, json=payload, timeout=60) as response:
                        if response.status != 200:
                            err_txt = await response.text()
                            return web.json_response({
                                "status": "error", 
                                "message": f"Ollama returned HTTP {response.status}: {err_txt}"
                            })
                        
                        resp_json = await response.json()
                        generated_text = resp_json.get("response", "").strip()
                except aiohttp.ClientConnectorError:
                    return web.json_response({
                        "status": "error", 
                        "message": (
                            "Could not connect to Ollama. Please ensure Ollama is installed, "
                            "running in the background, and that you have pulled the model "
                            "via 'ollama run huihui_ai/qwen3.5-abliterated:2b' in your terminal."
                        )
                    })

            if "<think>" in generated_text:
                generated_text = generated_text.split("</think>")[-1].strip()

            log.info(f"[PromptRelay] Analysis complete: {generated_text}")
            return web.json_response({"status": "success", "description": generated_text})

        except Exception as e:
            log.error(f"[PromptRelay] Failed to analyze character: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)
except Exception as e:
    log.warning(f"[PromptRelay] Could not register /ltx_director/analyze_character endpoint: {e}")


def _load_image_tensor(seg: dict) -> torch.Tensor:
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
    if crf == 0:
        return tensor
    img = tensor[0] 
    h = (img.shape[0] // 2) * 2
    w = (img.shape[1] // 2) * 2
    img_np = (img[:h, :w] * 255.0).byte().cpu().numpy() 

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
            decoded = frame_r.to_ndarray(format="rgb24") 
            break
        container_r.close()

        if decoded is None:
            return tensor
        arr = torch.from_numpy(decoded.astype(np.float32) / 255.0).to(tensor.device, tensor.dtype)
        out = tensor.clone()
        out[0, :h, :w] = arr
        return out
    except Exception as e:
        log.warning("[PromptRelay] img_compression encode/decode failed: %s", e)
        return tensor


def _build_combined_audio(timeline_data_str: str, duration_frames: int, frame_rate: float) -> dict:
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
            
            with av.open(buffer) as container:
                stream = container.streams.audio[0]
                
                resampler = av.AudioResampler(
                    format='fltp',
                    layout='stereo',
                    rate=target_sr,
                )
                
                for frame in container.decode(stream):
                    for resampled_frame in resampler.resample(frame):
                        arr = resampled_frame.to_ndarray()
                        clip_frames.append(torch.from_numpy(arr))
                
                for resampled_frame in resampler.resample(None):
                    arr = resampled_frame.to_ndarray()
                    clip_frames.append(torch.from_numpy(arr))

            if not clip_frames:
                continue

            waveform = torch.cat(clip_frames, dim=1) 

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

            clip_waveform = waveform[:, start_sample_src:end_sample_src]

            start_sample_dst = int(start_frames / frame_rate * target_sr)
            
            if start_sample_dst >= out_waveform.shape[1]:
                continue
                
            end_sample_dst = start_sample_dst + actual_length

            if end_sample_dst > out_waveform.shape[1]:
                actual_length = out_waveform.shape[1] - start_sample_dst
                clip_waveform = clip_waveform[:, :actual_length]
                end_sample_dst = start_sample_dst + actual_length
                
            if actual_length <= 0:
                continue

            out_waveform[:, start_sample_dst:end_sample_dst] += clip_waveform

        except Exception as e:
            log.warning("[PromptRelay] Audio process error for segment %s: %s", seg.get("fileName"), e)
            continue

    return {"waveform": out_waveform.unsqueeze(0), "sample_rate": target_sr}


def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

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

    locals_list = [p.strip() for p in local_prompts.split("|")]
    
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
                io.Vae.Input("vae", optional=True, tooltip="Optional. Connect the LTX Autoencoder/VAE here to natively encode the MSR visual reference slideshow into the latent prefix."),
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
                io.Float.Input("reference_strength", default=1.0, min=0.0, max=5.0, step=0.05, optional=True, tooltip="Guide strength for the reference images."),
                io.Combo.Input("reference_mode", options=["Ghost Mask (End)", "Licon MSR (Prefix)"], default="Ghost Mask (End)", tooltip="Choose whether to hide the references at the end with attention masks (Ghost Mask) or stack them sequentially at the beginning (Licon MSR Prefix) for the MSR LoRA."),
                io.Int.Input("msr_prefix_frames", default=17, min=1, max=120, step=1, tooltip="The number of visual slideshow frames to generate at the start of the sequence for MSR LoRA (typically 17, 25, 33, or 41)."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
                GuideData.Output(display_name="guide_data"),
                io.Float.Output(display_name="frame_rate"),
                io.Audio.Output(display_name="combined_audio"),
                io.Int.Output(display_name="clean_latent_frames"),
                io.Int.Output(display_name="clean_pixel_frames"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, global_prompt, duration_frames, duration_seconds,
                timeline_data, local_prompts, segment_lengths, guide_strength="", epsilon=1e-3,
                frame_rate=24.0, display_mode="seconds",
                custom_width=768, custom_height=512, resize_method="maintain aspect ratio",
                divisible_by=32, img_compression=0, vae=None, audio_vae=None, optional_latent=None,
                use_custom_audio=False, save_prompts_to_file=False,
                reference_strength=1.0, reference_mode="Ghost Mask (End)", msr_prefix_frames=17) -> io.NodeOutput:

        # Force Ollama to unload the vision model to maximize VRAM before generation begins
        try:
            import requests
            unload_url = "http://127.0.0.1:11434/api/generate"
            unload_payload = {
                "model": "huihui_ai/qwen3.5-abliterated:2b",
                "keep_alive": 0
            }
            requests.post(unload_url, json=unload_payload, timeout=2.0)
            log.info("[PromptRelay] Dispatched VRAM eviction request to Ollama to maximize memory for LTX generation.")
        except Exception as e:
            log.debug("[PromptRelay] Ollama VRAM eviction skipped: %s", e)

        clean_pixel_frames = duration_frames + 1
        clean_latent_frames = ((clean_pixel_frames - 1) // 8) + 1

        guide_data = {"images": [], "insert_frames": [], "strengths": [], "frame_rate": float(frame_rate)}
        derived_w, derived_h = custom_width, custom_height
        
        char_images = []
                    
        # Extract Timeline Character Descriptions (Fallback) and process dropped style images
        char1_val = ""
        char2_val = ""
        char3_val = ""
        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
            characters = tdata.get("characters", [])
            
            if len(characters) > 0: char1_val = characters[0].get("description", "")
            if len(characters) > 1: char2_val = characters[1].get("description", "")
            if len(characters) > 2: char3_val = characters[2].get("description", "")
            
            for idx, char_info in enumerate(characters):
                images_list = char_info.get("images", [])
                legacy_b64 = char_info.get("imageB64", "")
                if legacy_b64 and not images_list:
                    images_list = [{"b64": legacy_b64, "name": char_info.get("fileName", "")}]
                    
                for img_info in images_list:
                    image_b64 = img_info.get("b64", "")
                    if image_b64:
                        if "," in image_b64:
                            image_b64 = image_b64.split(",", 1)[1]
                        img_bytes = base64.b64decode(image_b64)
                        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
                        
                        arr = np.array(img, dtype=np.float32) / 255.0
                        tensor = torch.from_numpy(arr).unsqueeze(0)
                        char_images.append(tensor)
        except Exception as e:
            log.warning("[PromptRelay] Could not process character slot inputs: %s", e)
                    
        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
            img_segs = [
                s for s in tdata.get("segments", [])
                if s.get("type", "image") == "image"
                and (s.get("imageFile") or s.get("imageB64"))
                and int(s.get("start", 0)) < duration_frames
            ]
            img_segs.sort(key=lambda s: s["start"])

            strengths = []
            if guide_strength.strip():
                strengths = [float(x.strip()) for x in guide_strength.split(",") if x.strip()]

            for idx, seg in enumerate(img_segs):
                tensor = _load_image_tensor(seg)
                src_h, src_w = tensor.shape[1], tensor.shape[2]

                def snap(val, div):
                    return max(div, (val // div) * div)

                if custom_width > 0 and custom_height > 0:
                    tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by)
                elif custom_width > 0:
                    tgt_w = snap(custom_width, divisible_by)
                    tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
                elif custom_height > 0:
                    tgt_h = snap(custom_height, divisible_by)
                    tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
                else:
                    tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)

                if img_compression > 0:
                    tensor = _compress_image(tensor, img_compression)

                if idx == 0:
                    derived_h = tensor.shape[1]
                    derived_w = tensor.shape[2]

                strength = strengths[idx] if idx < len(strengths) else 1.0
                guide_data["images"].append(tensor)
                guide_data["insert_frames"].append(int(seg["start"]))
                guide_data["strengths"].append(float(strength))
            
            if not guide_data["images"] and not char_images:
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

        # Initialize dimension checks
        derived_w = max(32, (derived_w // 32) * 32) if derived_w > 0 else 768
        derived_h = max(32, (derived_h // 32) * 32) if derived_h > 0 else 512

        # Dual Mode execution branches
        if reference_mode == "Licon MSR (Prefix)":
            if vae is None:
                raise ValueError("reference_mode is set to 'Licon MSR (Prefix)' but no VAE is connected to the 'vae' input pin of LTX Director! Please connect your LTX VAE to encode the reference slideshow.")

            # Compile character reference images and first timeline background image
            prepared_images = []
            
            # Prepare characters
            for char_img in char_images:
                prepared = _resize_image(char_img, derived_w, derived_h, resize_method, divisible_by)
                prepared_images.append(prepared)
                
            # Prepare background scene
            bg_tensor = None
            if img_segs:
                bg_tensor = _load_image_tensor(img_segs[0])
            if bg_tensor is None:
                bg_tensor = torch.zeros((1, derived_h, derived_w, 3), dtype=torch.float32)
                
            prepared_bg = _resize_image(bg_tensor, derived_w, derived_h, resize_method, divisible_by)
            prepared_images.append(prepared_bg)
            
            # Expand frames
            base_count = msr_prefix_frames // len(prepared_images)
            remainder = msr_prefix_frames % len(prepared_images)
            
            slideshow_tensors = []
            for index, prepared_img in enumerate(prepared_images):
                repeats = base_count + (1 if index < remainder else 0)
                for _ in range(repeats):
                    slideshow_tensors.append(prepared_img[0])
                    
            slideshow_video = torch.stack(slideshow_tensors)
            
            # VAE encode slideshow_video
            log.info(f"[PromptRelay] Encoding {msr_prefix_frames} Licon-MSR slideshow frames...")
            
            # Fixed: directly capture the output Tensor returned from standard vae.encode
            msr_latent_samples = vae.encode(slideshow_video)
            
            # Ensure msr_latent_samples is 5D [B, C, F, H, W] to match LTX Video standard
            if msr_latent_samples.dim() == 4:
                msr_latent_samples = msr_latent_samples.unsqueeze(2)
                
            prefix_latent_frames = msr_latent_samples.shape[2]
            
            if optional_latent is not None:
                # Concatenate slideshow and optional_latent along temporal dimension (dim=2)
                log.info("[PromptRelay] Prepending reference slideshow to incoming optional_latent...")
                opt_samples = optional_latent["samples"]
                if opt_samples.dim() == 4:
                    opt_samples = opt_samples.unsqueeze(2)
                
                samples = torch.cat([msr_latent_samples.to(device=opt_samples.device, dtype=opt_samples.dtype), opt_samples], dim=2)
                total_latents = samples.shape[2]
            else:
                total_latents = prefix_latent_frames + clean_latent_frames
                # Initialize 5D samples
                samples = torch.zeros(
                    [1, 128, total_latents, derived_h // 32, derived_w // 32],
                    device=comfy.model_management.intermediate_device(),
                )
                # Copy encoded slideshow to the beginning of our latent
                samples[:, :, :prefix_latent_frames, :, :] = msr_latent_samples
            
            # Protect prefix frames using a 5D noise mask [B, 1, F, H, W] matching LTX Video standard
            if optional_latent is not None and "noise_mask" in optional_latent and optional_latent["noise_mask"] is not None:
                opt_mask = optional_latent["noise_mask"]
                if opt_mask.dim() == 4:
                    opt_mask = opt_mask.unsqueeze(1) # Convert 4D to 5D [B, 1, F, H, W]
                
                prefix_mask = torch.ones((1, 1, prefix_latent_frames, samples.shape[3], samples.shape[4]), dtype=torch.float32, device=samples.device)
                mask = torch.cat([prefix_mask, opt_mask.to(device=samples.device, dtype=prefix_mask.dtype)], dim=2)
            else:
                mask = torch.zeros(
                    (1, 1, total_latents, samples.shape[3], samples.shape[4]), 
                    dtype=torch.float32, 
                    device=samples.device
                )
                mask[:, :, :prefix_latent_frames, :, :] = 1.0
            
            latent = {"samples": samples, "noise_mask": mask}
            log.info(
                "[PromptRelay] Created Licon-MSR Latent Prefix: %dx%d, %d total latent frames (%d prefix, %d generated)",
                derived_w, derived_h, total_latents, prefix_latent_frames, total_latents - prefix_latent_frames
            )
        else: # Standard "Ghost Mask (End)"
            total_latents = clean_latent_frames + len(char_images)
            
            if optional_latent is None:
                samples = torch.zeros(
                    [1, 128, total_latents, derived_h // 32, derived_w // 32],
                    device=comfy.model_management.intermediate_device(),
                )
                latent = {"samples": samples}
            else:
                latent = optional_latent
                
            # Process references as guide_data appended to the end of the video
            if char_images:
                for i, single_ref in enumerate(char_images):
                    ref_tensor = _resize_image(single_ref, derived_w, derived_h, resize_method, divisible_by)
                    if img_compression > 0:
                        ref_tensor = _compress_image(ref_tensor, img_compression)
                    guide_data["images"].append(ref_tensor)
                    insert_point = (clean_latent_frames + i) * 8
                    guide_data["insert_frames"].append(insert_point)
                    guide_data["strengths"].append(float(reference_strength))

        processed_global, processed_local = _preprocess_prompts_with_characters(
            global_prompt, local_prompts, char1_val, char2_val, char3_val
        )

        patched, conditioning = _encode_relay(
            model, clip, latent, processed_global, processed_local, segment_lengths, epsilon,
        )

        ltxv_length = ((total_latents - 1) * 8) + 1 
        audio_out = _build_combined_audio(timeline_data, ltxv_length, float(frame_rate))
        audio_latent = {}
        
        if audio_vae is not None:
            def get_empty_latent():
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
                        
                        # Fix variable shadowing: renamed audio mask tensor to prevent overwriting video mask
                        audio_mask_tensor = torch.full(
                            (1, latent_samples.shape[-2], latent_samples.shape[-1]), 
                            0.0, 
                            dtype=torch.float32, 
                            device=comfy.model_management.intermediate_device()
                        )
                        
                        audio_latent = {
                            "samples": latent_samples,
                            "type": "audio",
                            "noise_mask": audio_mask_tensor.reshape((-1, 1, audio_mask_tensor.shape[-2], audio_mask_tensor.shape[-1]))
                        }
                    else:
                        raise ValueError("No audio waveform to encode.")
                except Exception as e:
                    log.error("[PromptRelay] Failed to generate custom audio latent: %s", e)
                    raise e
            else:
                try:
                    audio_latent = get_empty_latent()
                except Exception as e:
                    log.error("[PromptRelay] Could not generate empty audio latent: %s", e)
                    raise e

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