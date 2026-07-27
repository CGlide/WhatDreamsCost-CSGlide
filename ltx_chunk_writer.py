# --- START OF FILE ltx_chunk_writer.py ---

import logging
import os
import shutil

import numpy as np
from PIL import Image

import folder_paths

try:
    import av
    from fractions import Fraction
except Exception:  # PyAV missing - PNG output still works, video is skipped.
    av = None

log = logging.getLogger(__name__)

HANDOFF_ROOT = "ltx_director_handoff"
TEMPORAL_STRIDE = 8


def _to_float(frame):
    """[H, W, C] float32 0..1 (or a torch tensor) -> float32 RGB numpy array."""
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _to_uint8(frame_float):
    """float32 RGB 0..1 -> uint8."""
    return np.clip(frame_float * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _match_coeffs(src, ref):
    """Per-channel linear correction mapping `src` onto `ref`.

    Returns (scale, offset) as [3] arrays such that src * scale + offset has ref's
    per-channel mean and standard deviation. Both frames are the SAME moment — one is
    the previous chunk's last frame, the other this chunk's regeneration of it — so a
    simple linear match is well determined and doesn't need histogram matching.
    """
    scale = np.ones(3, dtype=np.float32)
    offset = np.zeros(3, dtype=np.float32)
    for c in range(3):
        s = src[..., c]
        r = ref[..., c]
        s_std = float(s.std())
        r_std = float(r.std())
        if s_std < 1e-5:
            continue
        k = r_std / s_std
        # Clamp: a wild scale means the frames aren't really the same shot.
        k = float(np.clip(k, 0.5, 2.0))
        scale[c] = k
        offset[c] = float(r.mean()) - k * float(s.mean())
    return scale, offset


def _safe_name(name, fallback):
    """Strip anything that could escape the intended folder."""
    cleaned = "".join(c for c in str(name) if c.isalnum() or c in ("-", "_", " ")).strip()
    cleaned = cleaned.replace(" ", "_")
    return cleaned or fallback


class _Mp4Writer:
    """Streaming h264 encoder. Opened lazily so it can take its size from frame 1."""

    def __init__(self, path, fps, crf):
        self.path = path
        self.fps = float(fps) if float(fps) > 0 else 25.0
        self.crf = int(crf)
        self.container = None
        self.stream = None
        self.w = self.h = 0

    def _open(self, h, w):
        # yuv420p needs even dimensions.
        self.h = (h // 2) * 2
        self.w = (w // 2) * 2
        self.container = av.open(self.path, mode="w")
        self.stream = self.container.add_stream("libx264", rate=Fraction(round(self.fps * 1000), 1000))
        self.stream.width = self.w
        self.stream.height = self.h
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(self.crf), "preset": "medium"}

    def add(self, rgb_uint8):
        if self.container is None:
            self._open(rgb_uint8.shape[0], rgb_uint8.shape[1])
        frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(rgb_uint8[:self.h, :self.w, :]), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        if self.container is None:
            return
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


class LTXChunkWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "run_name": ("STRING", {
                    "default": "run01",
                    "tooltip": "Folder name for this whole long-video run. Keep it the same across every chunk of the same video.",
                }),
                "chunk_index": ("INT", {
                    "default": 1, "min": 1, "max": 999, "step": 1,
                    "tooltip": "Which chunk this is. 1 for the first, 2 for the next, and so on.",
                }),
                "handoff_frames": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 8,
                    "tooltip": "How many frames from the END of this chunk to copy into the input folder, ready to guide the next chunk. Snapped to a multiple of 8 because the LTX VAE packs 8 pixel frames into 1 latent frame — an unaligned handoff lands mid-latent and causes a hitch. 0 disables the handoff.",
                }),
                "runs_folder": ("STRING", {
                    "default": "ltx_director_runs",
                    "tooltip": "Subfolder of ComfyUI's output directory where full chunk frames are written.",
                }),
                "save_all_frames": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off = only write the handoff frames. Useful for testing the seam without filling a disk.",
                }),
                "clear_chunk_first": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wipe this chunk's folder before writing, so re-running a chunk doesn't leave stale frames behind. Only ever touches this one chunk folder.",
                }),
                "total_chunks": ("INT", {
                    "default": 1, "min": 1, "max": 999, "step": 1,
                    "tooltip": "How many chunks this run has in total. Set automatically by Render All. When chunk_index reaches this number the run is assembled.",
                }),
                "auto_assemble": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "After writing the final chunk, join every chunk into <run>/final/ with a cross-dissolve across the overlap. Uses handoff_frames as the overlap width.",
                }),
                "write_video": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "After assembling the final chunk, also encode <run>_final.mp4 next to the final/ folder. The PNGs are written either way - the video is a convenience copy, not the master.",
                }),
                "video_fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Frame rate for the assembled video. Render All sets this from the timeline automatically; only set it by hand if you're queueing chunks yourself.",
                }),
                "video_crf": ("INT", {
                    "default": 18, "min": 0, "max": 51, "step": 1,
                    "tooltip": "h264 quality. Lower is better and bigger; 18 is visually near-lossless.",
                }),
                "match_to_previous": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Level-match this chunk to the previous one. Compares this chunk's first frame against the previous chunk's last handoff frame - the same moment, regenerated - and applies that per-channel correction to every frame. Off for chunk 1, or when you'd rather grade in your editor.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "handoff_folder", "info")
    FUNCTION = "write_chunk"
    CATEGORY = "WhatDreamsCost CS"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Writes a chunk of a long video out as PNG frames instead of encoding video, and copies the "
        "last N frames into ComfyUI's input folder so they can be dropped straight onto the timeline "
        "as guides for the next chunk. Passes images through unchanged, so it can sit in front of "
        "Video Combine rather than replacing it."
    )

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def write_chunk(self, images, run_name, chunk_index, handoff_frames,
                    runs_folder, save_all_frames, clear_chunk_first,
                    total_chunks=1, auto_assemble=True, match_to_previous=False,
                    write_video=True, video_fps=25.0, video_crf=18):
        run = _safe_name(run_name, "run01")
        runs_sub = _safe_name(runs_folder, "ltx_director_runs")
        chunk_tag = "chunk_%03d" % int(chunk_index)

        total = int(images.shape[0]) if hasattr(images, "shape") else len(images)
        if total <= 0:
            return (images, "", "[LTXChunkWriter] No frames received — nothing written.")

        # Snap the handoff to the VAE's temporal stride.
        requested = max(0, int(handoff_frames))
        n_handoff = (requested // TEMPORAL_STRIDE) * TEMPORAL_STRIDE
        if requested > 0 and n_handoff == 0:
            n_handoff = TEMPORAL_STRIDE
        if n_handoff > total:
            n_handoff = (total // TEMPORAL_STRIDE) * TEMPORAL_STRIDE
        if n_handoff != requested:
            log.info("[LTXChunkWriter] handoff_frames %d -> %d (multiple of %d).",
                     requested, n_handoff, TEMPORAL_STRIDE)

        chunk_dir = os.path.join(folder_paths.get_output_directory(), runs_sub, run, chunk_tag)
        handoff_dir = os.path.join(folder_paths.get_input_directory(), HANDOFF_ROOT, run)
        handoff_rel = "%s/%s" % (HANDOFF_ROOT, run)

        if clear_chunk_first and os.path.isdir(chunk_dir):
            try:
                shutil.rmtree(chunk_dir)
            except Exception as e:
                log.warning("[LTXChunkWriter] Could not clear %s: %s", chunk_dir, e)

        os.makedirs(chunk_dir, exist_ok=True)
        if n_handoff > 0:
            os.makedirs(handoff_dir, exist_ok=True)

        pbar = None
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(total)
        except Exception:
            pass

        handoff_start = total - n_handoff if n_handoff > 0 else total
        written = 0
        handoff_written = []

        # Level-match against the previous chunk, if asked and if there is one.
        scale, offset, match_note = None, None, "off"
        if match_to_previous and int(chunk_index) > 1:
            prev_tag = "chunk_%03d" % (int(chunk_index) - 1)
            prev_files = []
            if os.path.isdir(handoff_dir):
                prev_files = sorted(
                    f for f in os.listdir(handoff_dir)
                    if f.startswith(prev_tag + "_h") and f.lower().endswith(".png")
                )
            if prev_files:
                ref_path = os.path.join(handoff_dir, prev_files[-1])
                try:
                    ref = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.float32) / 255.0
                    cur = _to_float(images[0])
                    if ref.shape == cur.shape:
                        scale, offset = _match_coeffs(cur, ref)
                        match_note = "matched to %s (scale %s)" % (
                            prev_files[-1], np.round(scale, 4).tolist())
                    else:
                        match_note = "skipped - size mismatch %s vs %s" % (ref.shape, cur.shape)
                except Exception as e:
                    match_note = "failed: %s" % e
            else:
                match_note = "skipped - no %s handoff frame found" % prev_tag
            log.info("[LTXChunkWriter] Level match: %s", match_note)

        for i in range(total):
            is_handoff = i >= handoff_start
            if not save_all_frames and not is_handoff:
                if pbar is not None:
                    pbar.update(1)
                continue

            frame = _to_float(images[i])
            if scale is not None:
                frame = frame * scale + offset
                np.clip(frame, 0.0, 1.0, out=frame)
            img = Image.fromarray(_to_uint8(frame), mode="RGB")

            if save_all_frames:
                img.save(os.path.join(chunk_dir, "frame_%05d.png" % i), compress_level=4)
                written += 1

            if is_handoff:
                # Flat, sortable names — easy to find in the Add Image browser.
                name = "%s_h%02d.png" % (chunk_tag, i - handoff_start)
                img.save(os.path.join(handoff_dir, name), compress_level=4)
                handoff_written.append("%s/%s" % (handoff_rel, name))

            if pbar is not None:
                pbar.update(1)

        assembled = ""
        if auto_assemble and int(chunk_index) >= int(total_chunks) and int(total_chunks) > 1:
            try:
                final_dir, asm_info = _assemble_run(
                    run, runs_sub, n_handoff, True, False,
                    write_video=bool(write_video), video_fps=video_fps, video_crf=video_crf)
                assembled = " | ASSEMBLED -> %s" % final_dir
                log.info(asm_info)
            except Exception as e:
                assembled = " | assemble failed: %s" % e
                log.exception("[LTXChunkWriter] Auto-assemble failed")

        info = (
            "[LTXChunkWriter] run '%s' %s: %d frames in, %d PNGs written to %s. "
            "Handoff: %d frame(s) -> input/%s. Level match: %s%s"
            % (run, chunk_tag, total, written, chunk_dir, len(handoff_written),
               handoff_rel, match_note, assembled)
        )
        log.info(info)

        return (images, handoff_rel, info)


# ---------------------------------------------------------------------------
# Listing endpoint for the timeline UI.
# The "Continue From" row in the settings menu calls this to find handoff sets
# written by the node above. Kept in this file so the feature is self-contained.
# ---------------------------------------------------------------------------
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/ltx_director/handoff_sets")
    async def _ltx_handoff_sets(request):
        root = os.path.join(folder_paths.get_input_directory(), HANDOFF_ROOT)
        sets = {}
        if os.path.isdir(root):
            for run in os.listdir(root):
                run_dir = os.path.join(root, run)
                if not os.path.isdir(run_dir):
                    continue
                for fname in os.listdir(run_dir):
                    if not fname.lower().endswith(".png") or "_h" not in fname:
                        continue
                    chunk = fname.rsplit("_h", 1)[0]
                    key = (run, chunk)
                    entry = sets.setdefault(key, {
                        "run": run, "chunk": chunk, "files": [], "mtime": 0.0,
                    })
                    entry["files"].append("%s/%s/%s" % (HANDOFF_ROOT, run, fname))
                    try:
                        entry["mtime"] = max(entry["mtime"],
                                             os.path.getmtime(os.path.join(run_dir, fname)))
                    except OSError:
                        pass

        out = []
        for entry in sets.values():
            entry["files"].sort()
            entry["count"] = len(entry["files"])
            out.append(entry)
        # Newest first, so the set you just rendered is the default choice.
        out.sort(key=lambda e: e["mtime"], reverse=True)
        return web.json_response({"sets": out})

except Exception as _e:  # pragma: no cover - server not present (e.g. unit tests)
    log.debug("[LTXChunkWriter] handoff_sets route not registered: %s", _e)


class LTXChunkAssembler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "run_name": ("STRING", {
                    "default": "run01",
                    "tooltip": "The run to assemble. Must match what the Chunk Writer used.",
                }),
                "runs_folder": ("STRING", {
                    "default": "ltx_director_runs",
                    "tooltip": "Subfolder of ComfyUI's output directory holding the run.",
                }),
                "overlap_frames": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "How many frames each chunk shares with the next. Must match the handoff_frames you rendered with, and you must have started each window that many frames before the previous chunk ended.",
                }),
                "crossfade": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Blend across the overlap instead of cutting. Off = hard cut, overlap frames taken from the earlier chunk.",
                }),
                "write_video": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also encode <run>_final.mp4 alongside the final/ PNG sequence.",
                }),
                "video_fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Frame rate for the assembled video. Match your timeline.",
                }),
                "video_crf": ("INT", {
                    "default": 18, "min": 0, "max": 51, "step": 1,
                    "tooltip": "h264 quality. Lower is better and bigger; 18 is visually near-lossless.",
                }),
                "match_levels": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offset each chunk's brightness to match the one before it. This averages the MEAN over every overlap frame and never touches contrast - unlike the writer's per-frame match, which was unreliable. Still off by default: try the crossfade alone first.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_folder", "info")
    FUNCTION = "assemble"
    CATEGORY = "WhatDreamsCost CS"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Joins the chunks of a run into one continuous PNG sequence, cross-dissolving the "
        "overlapping frames. Writes to <run>/final/ rather than returning an IMAGE batch: "
        "a 16s 1920x1088 sequence is about 10GB as float32, which is not something to hold "
        "in memory just to hand to a video encoder."
    )

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def assemble(self, run_name, runs_folder, overlap_frames, crossfade, match_levels,
                 write_video=True, video_fps=25.0, video_crf=18):
        return _assemble_run(_safe_name(run_name, "run01"),
                             _safe_name(runs_folder, "ltx_director_runs"),
                             int(overlap_frames), bool(crossfade), bool(match_levels),
                             write_video=bool(write_video), video_fps=video_fps,
                             video_crf=video_crf)


def _assemble_run(run, runs_sub, overlap_frames, crossfade, match_levels,
                  write_video=False, video_fps=25.0, video_crf=18):
        run_dir = os.path.join(folder_paths.get_output_directory(), runs_sub, run)

        if not os.path.isdir(run_dir):
            return ("", "[LTXChunkAssembler] No such run folder: %s" % run_dir)

        chunk_dirs = sorted(
            os.path.join(run_dir, d) for d in os.listdir(run_dir)
            if d.startswith("chunk_") and os.path.isdir(os.path.join(run_dir, d))
        )
        if not chunk_dirs:
            return ("", "[LTXChunkAssembler] No chunk folders in %s" % run_dir)

        chunks = []
        for cd in chunk_dirs:
            files = sorted(f for f in os.listdir(cd) if f.lower().endswith(".png"))
            if files:
                chunks.append((cd, files))
        if not chunks:
            return ("", "[LTXChunkAssembler] Chunk folders contain no PNGs.")

        n_ov = max(0, int(overlap_frames))
        final_dir = os.path.join(run_dir, "final")
        if os.path.isdir(final_dir):
            try:
                shutil.rmtree(final_dir)
            except Exception as e:
                log.warning("[LTXChunkAssembler] Could not clear %s: %s", final_dir, e)
        os.makedirs(final_dir, exist_ok=True)

        def load(cd, fname):
            return np.asarray(Image.open(os.path.join(cd, fname)).convert("RGB"),
                              dtype=np.float32) / 255.0

        mp4 = None
        mp4_path = os.path.join(run_dir, "%s_final.mp4" % run)
        if write_video:
            if av is None:
                log.warning("[LTXChunkAssembler] PyAV not available - skipping video.")
            else:
                mp4 = _Mp4Writer(mp4_path, video_fps, video_crf)

        def save(idx, arr):
            np.clip(arr, 0.0, 1.0, out=arr)
            rgb = _to_uint8(arr)
            Image.fromarray(rgb, mode="RGB").save(
                os.path.join(final_dir, "frame_%05d.png" % idx), compress_level=4)
            if mp4 is not None:
                mp4.add(rgb)

        total_est = sum(len(f) for _, f in chunks) - n_ov * (len(chunks) - 1)
        pbar = None
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(max(1, total_est))
        except Exception:
            pass

        out_idx = 0
        # Cumulative per-chunk brightness offset, so chunk 3 matches corrected chunk 2.
        offset = np.zeros(3, dtype=np.float32)
        prev_offset = np.zeros(3, dtype=np.float32)
        notes = []

        for ci, (cd, files) in enumerate(chunks):
            n = len(files)
            ov = min(n_ov, n) if ci > 0 else 0
            prev_offset = offset.copy()

            if ci > 0 and match_levels and ov > 0:
                prev_cd, prev_files = chunks[ci - 1]
                prev_tail = prev_files[-ov:]
                cur_head = files[:ov]
                prev_mean = np.mean([load(prev_cd, f).reshape(-1, 3).mean(axis=0) for f in prev_tail], axis=0)
                cur_mean = np.mean([load(cd, f).reshape(-1, 3).mean(axis=0) for f in cur_head], axis=0)
                # prev_offset is what the previous chunk was already shifted by, so the
                # target is its CORRECTED brightness, not its raw brightness. Without this
                # the corrections stop accumulating and chunk 3 drifts back.
                offset = (prev_mean + prev_offset) - cur_mean
                notes.append("chunk %d offset %s" % (ci + 1, np.round(offset, 4).tolist()))

            head_start = 0
            if ci > 0 and ov > 0:
                prev_cd, prev_files = chunks[ci - 1]
                prev_tail = prev_files[-ov:]
                for k in range(ov):
                    a = load(prev_cd, prev_tail[k]) + prev_offset
                    b = load(cd, files[k]) + offset
                    if crossfade:
                        w = (k + 1.0) / (ov + 1.0)
                        blended = a * (1.0 - w) + b * w
                    else:
                        blended = a
                    save(out_idx, blended)
                    out_idx += 1
                    if pbar is not None:
                        pbar.update(1)
                head_start = ov

            # Body of this chunk, minus the tail that the NEXT chunk will blend.
            tail_reserved = min(n_ov, n) if ci < len(chunks) - 1 else 0
            for k in range(head_start, n - tail_reserved):
                save(out_idx, load(cd, files[k]) + offset)
                out_idx += 1
                if pbar is not None:
                    pbar.update(1)

        vid_note = ""
        if mp4 is not None:
            try:
                mp4.close()
                vid_note = " | video: %s (%.3g fps)" % (mp4_path, float(video_fps))
            except Exception as e:
                vid_note = " | video failed: %s" % e
                log.exception("[LTXChunkAssembler] Video encode failed")

        info = ("[LTXChunkAssembler] %d chunk(s), overlap %d, %s -> %d frames in %s%s%s"
                % (len(chunks), n_ov, "crossfade" if crossfade else "hard cut",
                   out_idx, final_dir, vid_note,
                   (" | " + "; ".join(notes)) if notes else ""))
        log.info(info)
        return (final_dir, info)


NODE_CLASS_MAPPINGS = {
    "LTXChunkWriterCS": LTXChunkWriter,
    "LTXChunkAssemblerCS": LTXChunkAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXChunkWriterCS": "LTX Chunk Writer CS",
    "LTXChunkAssemblerCS": "LTX Chunk Assembler CS",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
