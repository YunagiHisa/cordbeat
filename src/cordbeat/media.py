"""Bounded preprocessing for video attachments."""

from __future__ import annotations

import asyncio
import io
import logging
import math
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoPreprocessError(RuntimeError):
    """Raised when an attachment cannot be converted safely."""


class VideoTooLongError(VideoPreprocessError):
    """Raised when an attachment exceeds the configured duration limit."""

    def __init__(self, duration: float, limit: float) -> None:
        self.duration = duration
        self.limit = limit
        super().__init__(
            f"video duration {duration:.1f}s exceeds the {limit:.1f}s limit"
        )


@dataclass(frozen=True)
class PreparedVideo:
    """Small visual clip plus full-band speech audio for STT."""

    video_bytes: bytes
    audio_wav: bytes | None
    source_duration_seconds: float
    sampled_frames: int


@dataclass(frozen=True)
class TimedAudioChunk:
    """One timestamped PCM WAV chunk ready for an STT backend."""

    start_seconds: float
    end_seconds: float
    wav_bytes: bytes


def _find_ffmpeg_tool(name: str, ffmpeg: str | None = None) -> str | None:
    found = shutil.which(name)
    if found is not None:
        return found
    if ffmpeg:
        sibling = Path(ffmpeg).with_name(f"{name}{Path(ffmpeg).suffix}")
        if sibling.is_file():
            return str(sibling)
    return None


def _probe_duration(ffprobe: str, source: Path) -> float:
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        duration = float(result.stdout.strip())
    except Exception as exc:
        raise VideoPreprocessError("could not determine video duration") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise VideoPreprocessError("video duration is invalid")
    return duration


def _detect_scene_times(ffmpeg: str, source: Path, threshold: float) -> list[float]:
    """Return timestamps where the picture changes sharply.

    Even sampling alone walks past a cut that happens between two samples,
    which is exactly where a video's meaning usually changes.  Failure here
    is not fatal: sampling falls back to even spacing alone.
    """
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-i",
                str(source),
                "-vf",
                f"select='gt(scene\\,{threshold})',showinfo",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=180,
        )
    except Exception:
        logger.warning("Scene detection failed; sampling evenly", exc_info=True)
        return []
    times = [
        float(match)
        for match in re.findall(r"pts_time:([0-9]+\.?[0-9]*)", result.stderr or "")
    ]
    return sorted(set(times))


def _choose_frame_times(
    duration: float,
    scene_times: list[float],
    budget: int,
) -> list[float]:
    """Pick which moments to show the model, newest information first.

    The opening and closing frames are always kept: they establish what the
    clip is and how it ends, and losing either makes a summary read as if
    the video were about its middle.  Scene changes fill the budget next,
    then even spacing covers whatever is left.
    """
    if budget <= 1 or duration <= 0:
        return [0.0]
    last = max(0.0, duration - 0.05)
    chosen: list[float] = [0.0, last]

    # A cut is only worth a slot if it is not already covered by a kept
    # frame; tolerance scales with how much of the clip each frame stands for.
    tolerance = max(0.05, duration / (budget * 2))

    def add(candidate: float) -> None:
        if len(chosen) >= budget:
            return
        if any(abs(candidate - kept) < tolerance for kept in chosen):
            return
        chosen.append(candidate)

    for time in scene_times:
        add(time)

    # Spread the leftover slots across the whole clip.  Walking a fine grid
    # from the start instead would spend the entire budget in the opening
    # seconds and never reach the end.
    def fill_evenly(slots: int) -> None:
        if slots <= 0:
            return
        for index in range(1, slots + 1):
            add(duration * index / (slots + 1))

    fill_evenly(budget - len(chosen))
    # Near-duplicates may have been skipped above; top up on a finer grid.
    if len(chosen) < budget:
        fill_evenly((budget - len(chosen)) * 4)
    return sorted(chosen)[:budget]


def _prepare_video_sync(
    raw: bytes,
    *,
    max_input_seconds: float,
    max_output_seconds: float,
    output_fps: float,
    max_edge_px: int,
    scene_threshold: float = 0.3,
) -> PreparedVideo:
    ffmpeg = _find_ffmpeg_tool("ffmpeg")
    ffprobe = _find_ffmpeg_tool("ffprobe", ffmpeg)
    if ffmpeg is None or ffprobe is None:
        raise VideoPreprocessError("ffmpeg and ffprobe must be available on PATH")

    frame_budget = max(1, round(max_output_seconds * output_fps))
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "input"
        video_out = Path(tmp) / "sampled.mp4"
        audio_out = Path(tmp) / "audio.wav"
        source.write_bytes(raw)

        duration = _probe_duration(ffprobe, source)
        if duration > max_input_seconds:
            raise VideoTooLongError(duration, max_input_seconds)

        # Sample across the entire source: the opening and closing frames plus
        # scene changes, topped up with even spacing.  The frames are then
        # replayed as a short clip so the model sees at most ``frame_budget``.
        scale = (
            f"scale=min({max_edge_px}\\,iw):min({max_edge_px}\\,ih):"
            "force_original_aspect_ratio=decrease:force_divisible_by=2"
        )
        scene_times = _detect_scene_times(ffmpeg, source, scene_threshold)
        frame_times = _choose_frame_times(duration, scene_times, frame_budget)
        frames_dir = Path(tmp) / "frames"
        frames_dir.mkdir()
        for index, timestamp in enumerate(frame_times):
            frame_path = frames_dir / f"frame_{index:04d}.png"
            try:
                subprocess.run(
                    [
                        ffmpeg,
                        "-nostdin",
                        "-y",
                        # -ss before -i seeks by keyframe, which is what keeps
                        # this affordable at one process per frame.
                        "-ss",
                        f"{timestamp:.3f}",
                        "-i",
                        str(source),
                        "-frames:v",
                        "1",
                        "-vf",
                        scale,
                        str(frame_path),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
            except Exception:
                # One unreadable moment should not discard the whole clip.
                logger.warning(
                    "Could not extract video frame at %.2fs; skipping", timestamp
                )
        extracted = sorted(frames_dir.glob("frame_*.png"))
        if not extracted:
            raise VideoPreprocessError("could not sample video frames")
        # Renumber so the encoder sees a gap-free sequence after any skips.
        for position, frame_path in enumerate(extracted):
            frame_path.rename(frames_dir / f"seq_{position:04d}.png")
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-y",
                    "-framerate",
                    f"{output_fps:.10f}",
                    "-i",
                    str(frames_dir / "seq_%04d.png"),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(video_out),
                ],
                check=True,
                capture_output=True,
                timeout=180,
            )
        except Exception as exc:
            raise VideoPreprocessError("could not assemble sampled video") from exc
        if not video_out.is_file() or video_out.stat().st_size == 0:
            raise VideoPreprocessError("video sampling produced no output")
        sampled_frames = len(extracted)

        # The full accepted duration is converted to a compact Whisper input.
        # Optional mapping lets genuinely silent videos continue visually.
        audio_result = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0?",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(audio_out),
            ],
            check=False,
            capture_output=True,
            timeout=180,
        )
        audio_wav = (
            audio_out.read_bytes()
            if audio_result.returncode == 0
            and audio_out.is_file()
            and audio_out.stat().st_size > 44
            else None
        )
        video_bytes = video_out.read_bytes()

    logger.info(
        "Prepared video attachment: %d -> %d bytes, %.1fs -> %d frames, audio=%s",
        len(raw),
        len(video_bytes),
        duration,
        sampled_frames,
        "yes" if audio_wav is not None else "no",
    )
    return PreparedVideo(
        video_bytes=video_bytes,
        audio_wav=audio_wav,
        source_duration_seconds=duration,
        sampled_frames=sampled_frames,
    )


async def prepare_video(
    raw: bytes,
    *,
    max_input_seconds: float,
    max_output_seconds: float,
    output_fps: float,
    max_edge_px: int,
    scene_threshold: float = 0.3,
) -> PreparedVideo:
    """Preprocess a video off the event loop."""

    return await asyncio.to_thread(
        _prepare_video_sync,
        raw,
        max_input_seconds=max_input_seconds,
        max_output_seconds=max_output_seconds,
        output_fps=output_fps,
        max_edge_px=max_edge_px,
        scene_threshold=scene_threshold,
    )


def split_wav_audio(
    wav_bytes: bytes,
    chunk_seconds: float,
    overlap_seconds: float = 2.0,
) -> list[TimedAudioChunk]:
    """Split PCM WAV data into bounded chunks with coarse timestamps.

    Consecutive chunks overlap: a hard cut lands mid-word often enough that
    without it, STT drops or mangles whatever straddles the boundary.  The
    repeated seconds are cheap next to losing a sentence.
    """

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as source:
            params = source.getparams()
            frame_rate = source.getframerate()
            total_frames = source.getnframes()
            frames_per_chunk = max(1, round(frame_rate * chunk_seconds))
            # Cap the overlap at half the window.  Left unbounded, an overlap
            # as long as the chunk advances one frame at a time and turns a
            # short clip into tens of thousands of near-identical STT calls.
            overlap_frames = max(0, round(frame_rate * overlap_seconds))
            overlap_frames = min(overlap_frames, frames_per_chunk // 2)
            step = max(1, frames_per_chunk - overlap_frames)
            frame_width = params.sampwidth * params.nchannels
            chunks: list[TimedAudioChunk] = []
            start = 0
            while start < total_frames:
                source.setpos(start)
                frames = source.readframes(frames_per_chunk)
                if not frames:
                    break
                frame_count = len(frames) // frame_width
                output = io.BytesIO()
                with wave.open(output, "wb") as target:
                    target.setparams(params)
                    target.writeframes(frames)
                chunks.append(
                    TimedAudioChunk(
                        start_seconds=start / frame_rate,
                        end_seconds=(start + frame_count) / frame_rate,
                        wav_bytes=output.getvalue(),
                    )
                )
                if start + frame_count >= total_frames:
                    break
                start += step
    except (EOFError, wave.Error) as exc:
        raise VideoPreprocessError("could not split extracted video audio") from exc
    return chunks
