"""Tests for workers/diarization_lib.py — pure merge/formatting logic shared by
the faster-whisper worker and the whisper.cpp diarization sidecar.

These import the module by inserting backend/workers/ onto sys.path directly
(mirroring how whisper_worker.py and diarization_worker.py resolve the sibling
import at runtime) rather than via the `workers` namespace package, so the
test exercises the same import shape production code actually uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

import diarization_lib  # noqa: E402


def test_parse_optional_bool_truthy_values():
    for value in ("1", "true", "TRUE", "yes", "on"):
        assert diarization_lib.parse_optional_bool(value) is True


def test_parse_optional_bool_falsy_values():
    for value in ("0", "false", "FALSE", "no", "off"):
        assert diarization_lib.parse_optional_bool(value) is False


def test_parse_optional_bool_none_and_empty():
    assert diarization_lib.parse_optional_bool(None) is None
    assert diarization_lib.parse_optional_bool("") is None
    assert diarization_lib.parse_optional_bool("  ") is None


def test_parse_optional_bool_invalid_raises():
    import pytest

    with pytest.raises(ValueError):
        diarization_lib.parse_optional_bool("maybe")


def test_speaker_for_interval_picks_max_overlap():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    # [3, 6] overlaps SPEAKER_00 by 2s and SPEAKER_01 by 2s -- tie goes to the
    # first one encountered with strictly-greater overlap (SPEAKER_00, since
    # SPEAKER_01's overlap does not exceed it).
    assert diarization_lib.speaker_for_interval(3.0, 6.0, turns) == "SPEAKER_00"
    assert diarization_lib.speaker_for_interval(4.5, 9.0, turns) == "SPEAKER_01"


def test_speaker_for_interval_no_overlap_returns_none():
    turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    assert diarization_lib.speaker_for_interval(5.0, 6.0, turns) is None


def test_split_segments_by_speaker_splits_on_turn_change():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
        {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_00"},
    ]
    segments = [
        {
            "id": 0,
            "start": 0.0,
            "end": 12.0,
            "text": "hola mundo adios gente vale",
            "words": [
                {"word": " hola", "start": 0.5, "end": 1.0},
                {"word": " mundo", "start": 1.0, "end": 2.0},
                {"word": " adios", "start": 6.0, "end": 7.0},
                {"word": " gente", "start": 7.0, "end": 8.0},
                {"word": " vale", "start": 11.0, "end": 12.0},
            ],
        }
    ]

    out = diarization_lib.split_segments_by_speaker(segments, turns)

    assert [s["speaker"] for s in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [s["id"] for s in out] == [0, 1, 2]
    assert out[0]["text"] == "hola mundo"
    assert out[1]["text"] == "adios gente"
    assert out[1]["start"] == 6.0 and out[1]["end"] == 8.0


def test_split_segments_by_speaker_without_words_uses_overlap():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    segments = [{"id": 0, "start": 5.5, "end": 9.0, "text": "x"}]

    out = diarization_lib.split_segments_by_speaker(segments, turns)

    assert len(out) == 1
    assert out[0]["speaker"] == "SPEAKER_01"


def test_collect_diarization_turns_sorts_by_start():
    class _Segment:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class _Annotation:
        def itertracks(self, yield_label):
            assert yield_label is True
            yield _Segment(5.0, 8.0), None, "SPEAKER_01"
            yield _Segment(0.0, 4.0), None, "SPEAKER_00"

    turns = diarization_lib.collect_diarization_turns(_Annotation())
    assert [t["speaker"] for t in turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert turns[0] == {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}


def test_collect_diarization_turns_unwraps_pyannote4_diarize_output():
    class _Segment:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class _InnerAnnotation:
        def itertracks(self, yield_label):
            yield _Segment(0.0, 2.0), None, "SPEAKER_00"

    class _DiarizeOutput:
        speaker_diarization = _InnerAnnotation()

    turns = diarization_lib.collect_diarization_turns(_DiarizeOutput())
    assert turns == [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]


def test_to_text_groups_consecutive_same_speaker_lines():
    segments = [
        {"text": "hola", "speaker": "SPEAKER_00"},
        {"text": "que tal", "speaker": "SPEAKER_00"},
        {"text": "bien y tu", "speaker": "SPEAKER_01"},
    ]
    text = diarization_lib.to_text(segments)
    assert text == "[SPEAKER_00] hola\nque tal\n[SPEAKER_01] bien y tu"


def test_to_text_without_speaker_has_no_prefix():
    segments = [{"text": "hola"}]
    assert diarization_lib.to_text(segments) == "hola"


def test_to_srt_includes_speaker_prefix():
    segments = [{"start": 0.0, "end": 1.5, "text": "hola", "speaker": "SPEAKER_00"}]
    srt = diarization_lib.to_srt(segments)
    assert "1" in srt.splitlines()
    assert "[SPEAKER_00] hola" in srt


def test_to_vtt_includes_speaker_voice_tag():
    segments = [{"start": 0.0, "end": 1.5, "text": "hola", "speaker": "SPEAKER_00"}]
    vtt = diarization_lib.to_vtt(segments)
    assert vtt.startswith("WEBVTT")
    assert "<v SPEAKER_00>hola" in vtt


def test_format_srt_ts_and_vtt_ts():
    assert diarization_lib.format_srt_ts(3661.25) == "01:01:01,250"
    assert diarization_lib.format_vtt_ts(3661.25) == "01:01:01.250"


class _FakePipeline:
    """Runs out of memory while its segmentation batch is above ``fits_at``."""

    def __init__(self, fits_at: int) -> None:
        self.segmentation_batch_size = 32
        self.embedding_batch_size = 32
        self.fits_at = fits_at
        self.calls = 0

    def __call__(self, audio):
        self.calls += 1
        if self.segmentation_batch_size > self.fits_at:
            raise MemoryError("batch_size ( 32) is probably too large.")
        return "annotation"


def test_out_of_memory_retries_with_smaller_batches():
    pipeline = _FakePipeline(fits_at=8)
    assert diarization_lib.run_pipeline_with_backoff(pipeline, {}) == "annotation"
    assert pipeline.calls == 3  # 32 -> 16 -> 8
    assert pipeline.segmentation_batch_size == 8
    assert pipeline.embedding_batch_size == 8


def test_out_of_memory_at_the_floor_raises_a_recognisable_error():
    import pytest

    pipeline = _FakePipeline(fits_at=0)
    with pytest.raises(diarization_lib.DiarizationOutOfMemory, match="diarization_out_of_memory"):
        diarization_lib.run_pipeline_with_backoff(pipeline, {})
    assert pipeline.calls == 4  # 32, 16, 8, 4
    assert pipeline.segmentation_batch_size == 4


def test_other_errors_are_not_retried():
    import pytest

    class _Broken(_FakePipeline):
        def __call__(self, audio):
            self.calls += 1
            raise ValueError("bad audio")

    pipeline = _Broken(fits_at=32)
    with pytest.raises(ValueError):
        diarization_lib.run_pipeline_with_backoff(pipeline, {})
    assert pipeline.calls == 1
