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

# --- Licon MSR engine fixed parameters (were node widgets; hardcoded for a clean UI) ---
# Slideshow length. 41 is the Licon training default (valid: 17 / 25 / 33 / 41).
MSR_PREFIX_FRAMES = 41
# latent_downscale_factor for the IC-LoRA reference frames. Tied to how the MSR LoRA was
# trained; Licon MSR V1 uses full-resolution references (1.0). This is INDEPENDENT of any
# multi-stage scale_by / x2 upscaling, which LTXDirectorGuide handles downstream.
MSR_LATENT_DOWNSCALE = 1.0


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


def _execute_comfy_node(node_class, **kwargs):
    """Executes any ComfyUI node class programmatically by resolving its entrypoint dynamically."""
    if hasattr(node_class, "execute"):
        return node_class.execute(**kwargs)
        
    node_instance = node_class()
    func_name = getattr(node_class, "FUNCTION", None)
    if func_name is None:
        for candidate in ("generate", "encode", "process", "execute"):
            if hasattr(node_instance, candidate):
                func_name = candidate
                break
                
    if func_name is not None:
        func = getattr(node_instance, func_name)
        import inspect
        sig = inspect.signature(func)
        filtered_kwargs = {}
        for param in sig.parameters.values():
            if param.name in kwargs:
                filtered_kwargs[param.name] = kwargs[param.name]
            elif param.default is not inspect.Parameter.empty:
                pass
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                filtered_kwargs.update(kwargs)
                break
        return func(**filtered_kwargs)
    else:
        raise AttributeError(f"Could not resolve entrypoint for node class {node_class.__name__}")


def _unpack(out):
    """Safely extracts positional returns from ComfyUI API outputs."""
    try:
        return out[0], out[1], out[2]
    except Exception:
        t = tuple(out)
        return t[0], t[1], t[2]


def _load_image_source(b64_or_url: str, filename: str = None) -> torch.Tensor:
    if not b64_or_url and not filename:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        
    if b64_or_url and "view?" in b64_or_url:
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(b64_or_url)
            q = parse_qs(parsed.query)
            fname = q.get("filename", [None])[0]
            subfolder = q.get("subfolder", [""])[0]
            if fname:
                file_path = os.path.join(folder_paths.get_input_directory(), subfolder, fname)
                if os.path.exists(file_path):
                    img = Image.open(file_path).convert("RGB")
                    arr = np.array(img, dtype=np.float32) / 255.0
                    return torch.from_numpy(arr).unsqueeze(0)
        except Exception as e:
            log.debug(f"[PromptRelay] URL parsing failed for {b64_or_url}: {e}")

    if filename:
        file_path = os.path.join(folder_paths.get_input_directory(), filename)
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

    if b64_or_url:
        try:
            b64_str = b64_or_url
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
        except Exception as e:
            log.debug(f"[PromptRelay] Base64 decoding failed: {e}")
            
    return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


def _load_image_tensor(seg: dict) -> torch.Tensor:
    if seg.get("imageFile"):
        file_path = os.path.join(folder_paths.get_input_directory(), seg["imageFile"])
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

    b64_str = seg.get("imageB64", "")
    
    if b64_str and "view?" in b64_str:
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(b64_str)
            q = parse_qs(parsed.query)
            fname = q.get("filename", [None])[0]
            subfolder = q.get("subfolder", [""])[0]
            if fname:
                file_path = os.path.join(folder_paths.get_input_directory(), subfolder, fname)
                if os.path.exists(file_path):
                    img = Image.open(file_path).convert("RGB")
                    arr = np.array(img, dtype=np.float32) / 255.0
                    return torch.from_numpy(arr).unsqueeze(0)
        except Exception as e:
            log.debug(f"[PromptRelay] URL parsing failed for {b64_str}: {e}")

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
                resampler = av.AudioResampler(format='fltp', layout='stereo', rate=target_sr)
                
                for frame in container.decode(stream):
                    for resampled_frame in resampler.resample(frame):
                        clip_frames.append(torch.from_numpy(resampled_frame.to_ndarray()))
                
                for resampled_frame in resampler.resample(None):
                    clip_frames.append(torch.from_numpy(resampled_frame.to_ndarray()))

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


# Register API endpoint for instant export
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
            return web.json_response({"status": "error", "message": str(e)}, status=500)
except Exception as e:
    pass

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

            # Restored your preferred model configuration
            model_name = "huihui_ai/qwen3.5-abliterated:2b"
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
                "keep_alive": 0
            }

            ollama_url = "http://127.0.0.1:11434/api/generate"
            
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
                            f"Could not connect to Ollama. Please ensure Ollama is installed and "
                            f"running, and that you have pulled the model via 'ollama run {model_name}'."
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
    pass


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
                io.Conditioning.Input("negative", optional=True, tooltip="Optional. Connect your negative prompt conditioning here. Highly recommended for Licon MSR to prevent noise artifacts."),
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
                io.Combo.Input("reference_mode", options=["Ghost Mask (End)", "Licon MSR (Prefix)"], default="Ghost Mask (End)", tooltip="Choose whether to hide the references at the end with attention masks (Ghost Mask) or stack them sequentially at the beginning (Licon MSR Prefix) for the MSR LoRA."),
                io.Boolean.Input("save_prompts_to_file", default=False, optional=True, tooltip="Save the timeline prompts and parameters to a text file in your ComfyUI output directory during execution."),
                io.Float.Input("reference_strength", default=1.0, min=0.0, max=5.0, step=0.05, optional=True, tooltip="Guide strength for the reference images."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
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
    def execute(cls, model, clip, negative=None, vae=None, audio_vae=None, optional_latent=None,
                global_prompt="", duration_frames=120, duration_seconds=5.0,
                timeline_data="", use_custom_audio=False, local_prompts="", segment_lengths="",
                epsilon=1e-3, frame_rate=24.0, display_mode="seconds", guide_strength="",
                custom_width=0, custom_height=0, resize_method="maintain aspect ratio",
                divisible_by=32, img_compression=18,
                reference_mode="Ghost Mask (End)",
                save_prompts_to_file=False, reference_strength=1.0) -> io.NodeOutput:

        if isinstance(optional_latent, list):
            if len(optional_latent) > 0:
                optional_latent = optional_latent[0]
            else:
                optional_latent = None

        clean_pixel_frames = duration_frames + 1
        clean_latent_frames = ((clean_pixel_frames - 1) // 8) + 1

        guide_data = {"images": [], "insert_frames": [], "strengths": [], "frame_rate": float(frame_rate)}
        derived_w, derived_h = custom_width, custom_height
        char_images = []
        char_slot_images = []  # per-slot tensors, for MSR @char tag filtering
                    
        char1_val, char2_val, char3_val = "", "", ""
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
                    
                slot_tensors = []
                for img_info in images_list:
                    image_b64 = img_info.get("b64", "")
                    file_name = img_info.get("name", "")
                    tensor = _load_image_source(image_b64, file_name)
                    char_images.append(tensor)
                    slot_tensors.append(tensor)
                char_slot_images.append(slot_tensors)  # keep slot boundaries for @char filtering
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
            strengths = [float(x.strip()) for x in guide_strength.split(",") if x.strip()] if guide_strength.strip() else []

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
            
            # Prevent crashes if completely empty
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

        derived_w = max(32, (derived_w // 32) * 32) if derived_w > 0 else 768
        derived_h = max(32, (derived_h // 32) * 32) if derived_h > 0 else 512

        if reference_mode == "Licon MSR (Prefix)":
            processed_global, processed_local = global_prompt, local_prompts
        else:
            processed_global, processed_local = _preprocess_prompts_with_characters(
                global_prompt, local_prompts, char1_val, char2_val, char3_val
            )

        # If no global prompt was entered, fall back to the first timeline/local prompt as
        # the global anchor (restores prior behaviour; avoids an empty tail segment in the
        # relay, which would otherwise trip the 'segment missing a prompt' guard).
        if not (processed_global or "").strip() and processed_local:
            processed_global = processed_local.split("|")[0].strip()

        # ====== LICON MSR DIRECTOR ENGINE (relay-aware) ======
        # Axis separation, so the relay and IC-LoRA never fight over the same bookkeeping:
        #   - Character slots  -> IDENTITY. Built into the IC-LoRA slideshow AND pinned as
        #                          one identity keyframe each (the MSR LoRA's job).
        #   - Timeline images  -> SCENE/background. The first seeds the slideshow's background
        #                          slot; every timeline image is also pinned at its own
        #                          frame_idx as a positional scene reference.
        #   - Timeline text    -> the relay's per-segment attention mask over the clean region.
        # The relay conditioning is encoded FIRST, then threaded THROUGH the IC-LoRA guide
        # calls, so keyframe metadata + guide_attention_entries land on the real conditioning
        # natively. The Director bakes everything it conditions on, so the relay mask length
        # stays self-consistent with the latent and nothing downstream needs to add frames.
        if reference_mode == "Licon MSR (Prefix)":
            if vae is None:
                raise ValueError("Licon MSR (Prefix) mode requires connecting the VAE to LTX Director!")

            from nodes import NODE_CLASS_MAPPINGS
            IC_Lora_Guide_Class = NODE_CLASS_MAPPINGS.get("LTXAddVideoICLoRAGuide")
            if IC_Lora_Guide_Class is None:
                raise ValueError("LTXAddVideoICLoRAGuide node not found! Please install the ComfyUI-LTXVideo repository.")

            # 1. Identity references. In MSR the reference IMAGE drives identity, so we
            #    honour the @charN tags by selecting ONLY the slots the prompt references.
            #    @char1/@character1 -> slot 0, @char2 -> slot 1, @char3 -> slot 2.
            #    No tag referenced (or referenced slots empty) -> use every filled slot.
            _prompt_text = (global_prompt or "") + " " + (local_prompts or "")
            _tag_pairs = [("@character1", "@char1"), ("@character2", "@char2"), ("@character3", "@char3")]
            _referenced_slots = [i for i, tags in enumerate(_tag_pairs) if any(t in _prompt_text for t in tags)]
            _selected = []
            for _slot in _referenced_slots:
                if _slot < len(char_slot_images):
                    _selected.extend(char_slot_images[_slot])
            if not _selected:
                _selected = char_images  # fallback: no tags, or tagged slots had no image
            log.info("[PromptRelay] MSR slideshow subjects: referenced slots=%s, images=%d", _referenced_slots, len(_selected))
            identity_images = [
                _resize_image(c, derived_w, derived_h, resize_method, divisible_by)
                for c in _selected
            ]

            # 2. Scene references: every timeline image (already loaded/resized into guide_data
            #    upstream). Snapshot them, then clear guide_data so the downstream LTXDirectorGuide
            #    is a no-op -- the Director bakes them itself for a self-consistent latent length.
            scene_images = list(guide_data["images"])
            scene_frames = list(guide_data["insert_frames"])
            scene_strengths = list(guide_data["strengths"])
            guide_data = {"images": [], "insert_frames": [], "strengths": [], "frame_rate": float(frame_rate)}

            # 3. Slideshow background slide = first timeline image, else black.
            if scene_images:
                bg_slide = scene_images[0]
                if bg_slide.shape[1] != derived_h or bg_slide.shape[2] != derived_w:
                    bg_slide = _resize_image(bg_slide, derived_w, derived_h, resize_method, divisible_by)
            else:
                bg_slide = torch.zeros((1, derived_h, derived_w, 3), dtype=torch.float32)

            # 4. Build the slideshow exactly like Licon MSR: characters then background,
            #    distributed across MSR_PREFIX_FRAMES.
            slideshow_sources = identity_images + [bg_slide]
            base_count = MSR_PREFIX_FRAMES // len(slideshow_sources)
            remainder = MSR_PREFIX_FRAMES % len(slideshow_sources)
            slideshow_tensors = []
            for index, src_img in enumerate(slideshow_sources):
                repeats = base_count + (1 if index < remainder else 0)
                slideshow_tensors.extend([src_img[0]] * repeats)
            slideshow_video = torch.stack(slideshow_tensors)

            # 5. PER-STAGE reference handoff. In two-stage pipelines the references must be
            #    (re)encoded at EACH stage's resolution; a footprint baked here at full res breaks
            #    when scale_by shrinks the latent downstream. So the Director no longer injects --
            #    it hands the raw full-res slideshow + keyframe images to LTXDirectorGuide, which
            #    encodes and injects them at its own resolution. The length budget is purely
            #    TEMPORAL (scale-invariant): LTXVCropGuides always trims the slideshow's temporal
            #    footprint (prefix_latents), so the guide node pads the generated region so that
            #    (pad + keyframes) == prefix_latents at whatever resolution it runs.
            keyframe_images = slideshow_sources          # chars + bg (same set as the slideshow)
            num_keyframes = len(keyframe_images)
            prefix_latents = ((MSR_PREFIX_FRAMES - 1) // 8) + 1
            tail_latents = prefix_latents                 # pad + keyframes, filled in per stage
            total_runtime_latents = clean_latent_frames + tail_latents
            tail_pixels = tail_latents * 8

            # 6. Relay conditioning over the runtime length. This is temporal, so the mask is
            #    valid at every stage regardless of scale_by: clean -> locals, tail -> global.
            local_part = processed_local.strip() if processed_local.strip() else processed_global
            clean_lengths = segment_lengths.strip() if segment_lengths.strip() else str(duration_frames)
            injected_local = f"{local_part} | {processed_global}"
            injected_lengths = f"{clean_lengths},{tail_pixels}"

            dummy_full = {"samples": torch.zeros(
                [1, 128, total_runtime_latents, derived_h // 32, derived_w // 32],
                device=comfy.model_management.intermediate_device(),
            )}

            # FALLBACK (relay vs LoRA attention conflict): replace the next call with
            #     patched = model.clone()
            #     conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(processed_global))
            patched, conditioning = _encode_relay(
                model, clip, dummy_full, processed_global, injected_local, injected_lengths, epsilon
            )

            if negative is None:
                neg_tokens = clip.tokenize("worst quality, blurry, low resolution, jittery, low geometry, bad details")
                conditioning_neg = clip.encode_from_tokens_scheduled(neg_tokens)
            else:
                conditioning_neg = negative

            # 7. Clean base latent: the TRUE clean region only (C frames), at full target res.
            #    LTXDirectorGuide scales it per stage, pads it, and appends the keyframes.
            latent = {
                "samples": torch.zeros([1, 128, clean_latent_frames, derived_h // 32, derived_w // 32], device=comfy.model_management.intermediate_device()),
                "noise_mask": torch.ones((1, 1, clean_latent_frames, derived_h // 32, derived_w // 32), dtype=torch.float32, device=comfy.model_management.intermediate_device()),
            }

            # 8. Hand the RAW full-res references to the guide node via guide_data (no new socket).
            #    Each stage VAE-encodes them at its own scale, so Stage 2 sees full-res detail.
            guide_data = {
                "images": [], "insert_frames": [], "strengths": [], "frame_rate": float(frame_rate),
                "msr": {
                    "slideshow": slideshow_video,        # [F, H, W, 3] full-res montage
                    "keyframes": keyframe_images,         # list of [1, H, W, 3] full-res refs
                    "prefix_latents": int(prefix_latents),
                    "strength": float(reference_strength),
                    "downscale": float(MSR_LATENT_DOWNSCALE),
                    "clean_latent_frames": int(clean_latent_frames),
                },
            }

            total_latents = total_runtime_latents         # clean + prefix (what the sampler sees)

            log.info(
                "[PromptRelay] Licon MSR Engine (per-stage): clean=%d, tail=%d (prefix), "
                "%d keyframes handed to LTXDirectorGuide for per-resolution injection.",
                clean_latent_frames, tail_latents, num_keyframes,
            )
        else: # Standard "Ghost Mask (End)"
            total_latents = clean_latent_frames + len(char_images)
            
            if char_images:
                for i, single_ref in enumerate(char_images):
                    ref_tensor = _resize_image(single_ref, derived_w, derived_h, resize_method, divisible_by)
                    if img_compression > 0:
                        ref_tensor = _compress_image(ref_tensor, img_compression)
                    guide_data["images"].append(ref_tensor)
                    insert_point = (clean_latent_frames + i) * 8
                    guide_data["insert_frames"].append(insert_point)
                    guide_data["strengths"].append(float(reference_strength))

            dummy_latent = {"samples": torch.zeros([1, 128, total_latents, derived_h // 32, derived_w // 32], device=comfy.model_management.intermediate_device())}

            patched, conditioning = _encode_relay(
                model, clip, dummy_latent, processed_global, processed_local, segment_lengths, epsilon,
            )
            
            if optional_latent is None:
                latent = {
                    "samples": torch.zeros([1, 128, total_latents, derived_h // 32, derived_w // 32], device=comfy.model_management.intermediate_device()),
                    "noise_mask": torch.ones((1, 1, total_latents, derived_h // 32, derived_w // 32), dtype=torch.float32, device=comfy.model_management.intermediate_device())
                }
            else:
                latent = optional_latent
                
            if negative is None:
                neg_tokens = clip.tokenize("worst quality, blurry, low resolution, jittery, low geometry, bad details")
                conditioning_neg = clip.encode_from_tokens_scheduled(neg_tokens)
            else:
                conditioning_neg = negative

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
                    log.warning("[PromptRelay] Failed to generate custom audio latent: %s", e)
                    raise e
            else:
                try:
                    audio_latent = get_empty_latent()
                except Exception as e:
                    log.warning("[PromptRelay] Could not generate empty audio latent: %s", e)
                    raise e

        return io.NodeOutput(
            patched, 
            conditioning, 
            conditioning_neg, 
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