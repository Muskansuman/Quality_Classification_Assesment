"""Speech-to-text using pretrained faster-whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.audio_utils import load_wav_mono_float32
from src.vtt_parser import Segment


def transcribe_wav(
    wav_path: Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    on_progress: Callable[[str], None] | None = None,
) -> list[Segment]:
    if on_progress:
        on_progress(f"Loading Whisper model ({model_size})…")
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    if on_progress:
        on_progress("Loading audio…")
    audio = load_wav_mono_float32(wav_path)
    if on_progress:
        on_progress("Transcribing audio (this may take several minutes on CPU)…")
    segments_iter, _info = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
    )
    segments: list[Segment] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append(
            Segment(start=float(seg.start), end=float(seg.end), speaker="Speaker", text=text)
        )
    if on_progress:
        on_progress(f"Transcription complete ({len(segments)} segments).")
    return segments
