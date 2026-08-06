"""Tests for bounded video preprocessing."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from unittest.mock import patch

from cordbeat.config import AIBackendConfig, _apply_video_quality_preset
from cordbeat.media import _choose_frame_times, _detect_scene_times, split_wav_audio


def _wav(seconds: float, frame_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(frame_rate)
        target.writeframes(b"\x00\x00" * int(frame_rate * seconds))
    return buffer.getvalue()


class TestChooseFrameTimes:
    """Even sampling alone walks past cuts and can miss how a clip ends."""

    def test_always_keeps_the_opening_and_closing_frames(self) -> None:
        times = _choose_frame_times(duration=60.0, scene_times=[], budget=8)

        assert times[0] == 0.0
        assert times[-1] > 59.0

    def test_scene_changes_are_preferred_over_even_spacing(self) -> None:
        # A cut at 7.5s sits between the evenly spaced samples of a 60s clip.
        times = _choose_frame_times(duration=60.0, scene_times=[7.5], budget=6)

        assert any(abs(t - 7.5) < 0.01 for t in times)

    def test_never_exceeds_the_budget(self) -> None:
        scenes = [float(i) for i in range(1, 40)]
        times = _choose_frame_times(duration=60.0, scene_times=scenes, budget=8)

        assert len(times) == 8

    def test_output_is_ordered(self) -> None:
        times = _choose_frame_times(duration=30.0, scene_times=[20.0, 5.0], budget=8)

        assert times == sorted(times)

    def test_near_duplicate_scene_times_do_not_crowd_out_coverage(self) -> None:
        """Twenty cuts in one second must not consume the whole budget."""
        scenes = [10.0 + i * 0.01 for i in range(20)]
        times = _choose_frame_times(duration=60.0, scene_times=scenes, budget=8)

        clustered = [t for t in times if 9.9 < t < 10.3]
        assert len(clustered) == 1

    def test_single_frame_budget_is_handled(self) -> None:
        assert _choose_frame_times(duration=10.0, scene_times=[3.0], budget=1) == [0.0]


class TestDetectSceneTimes:
    def test_parses_showinfo_timestamps(self) -> None:
        stderr = (
            "[Parsed_showinfo_1 @ 0x1] n:0 pts:1 pts_time:2.5 duration:1\n"
            "[Parsed_showinfo_1 @ 0x1] n:1 pts:2 pts_time:9 duration:1\n"
        )
        completed = type("R", (), {"stderr": stderr})()
        with patch("subprocess.run", return_value=completed):
            times = _detect_scene_times("ffmpeg", Path("x"), 0.3)

        assert times == [2.5, 9.0]

    def test_failure_falls_back_to_even_sampling(self) -> None:
        """Losing scene detection should cost detail, not the whole video."""
        with patch("subprocess.run", side_effect=OSError("boom")):
            assert _detect_scene_times("ffmpeg", Path("x"), 0.3) == []


class TestSplitWavAudio:
    def test_chunks_overlap_so_speech_is_not_cut_mid_word(self) -> None:
        chunks = split_wav_audio(_wav(10.0), chunk_seconds=4.0, overlap_seconds=2.0)

        assert len(chunks) > 1
        assert chunks[1].start_seconds < chunks[0].end_seconds

    def test_timestamps_advance_and_cover_the_whole_clip(self) -> None:
        chunks = split_wav_audio(_wav(10.0), chunk_seconds=4.0, overlap_seconds=2.0)

        starts = [chunk.start_seconds for chunk in chunks]
        assert starts == sorted(starts)
        assert chunks[0].start_seconds == 0.0
        assert chunks[-1].end_seconds >= 9.9

    def test_overlap_at_least_chunk_length_still_advances(self) -> None:
        """A misconfigured overlap must not loop forever."""
        chunks = split_wav_audio(_wav(6.0), chunk_seconds=2.0, overlap_seconds=5.0)

        assert 0 < len(chunks) < 100

    def test_zero_overlap_keeps_chunks_adjacent(self) -> None:
        chunks = split_wav_audio(_wav(8.0), chunk_seconds=4.0, overlap_seconds=0.0)

        assert chunks[1].start_seconds == chunks[0].end_seconds


class TestVideoQualityPreset:
    def test_balanced_is_the_default_budget(self) -> None:
        config = AIBackendConfig()
        _apply_video_quality_preset(config, {})

        assert (config.video_max_seconds, config.video_max_edge_px) == (8.0, 448)

    def test_detail_widens_the_budget(self) -> None:
        config = AIBackendConfig(video_quality="detail")
        _apply_video_quality_preset(config, {"video_quality": "detail"})

        assert (config.video_max_seconds, config.video_max_edge_px) == (12.0, 448)

    def test_explicit_values_win_over_the_preset(self) -> None:
        """A preset is a starting point, not an override of a stated value."""
        config = AIBackendConfig(video_quality="detail", video_max_edge_px=224)
        _apply_video_quality_preset(
            config, {"video_quality": "detail", "video_max_edge_px": 224}
        )

        assert config.video_max_edge_px == 224
        assert config.video_max_seconds == 12.0

    def test_unknown_preset_falls_back_to_balanced(self) -> None:
        config = AIBackendConfig(video_quality="ultra")
        _apply_video_quality_preset(config, {"video_quality": "ultra"})

        assert (config.video_max_seconds, config.video_max_edge_px) == (8.0, 448)
