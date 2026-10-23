"""End-to-end analysis pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from src.assess import assess_with_llm
from src.audio_utils import extract_audio_wav, probe_duration_seconds
from src.features import ClassFeatures, compute_features
from src.transcribe import transcribe_wav
from src.vtt_parser import Segment, parse_vtt


ProgressFn = Callable[[str], None]


def _noop(msg: str) -> None:
    pass


def analyze_recording(
    video_path: Path,
    *,
    vtt_path: Path | None = None,
    whisper_model: str = "small",
    use_whisper: bool = True,
    api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    llm_base_url: str | None = None,
    on_progress: ProgressFn | None = None,
) -> dict:
    progress = on_progress or _noop
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    progress("Reading media duration…")
    duration = probe_duration_seconds(video_path)

    segments: list[Segment]
    transcript_source: str

    if vtt_path and Path(vtt_path).is_file():
        progress("Parsing provided WebVTT transcript…")
        segments = parse_vtt(vtt_path)
        transcript_source = "vtt"
    elif use_whisper:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            progress("Extracting audio with ffmpeg…")
            extract_audio_wav(video_path, wav)
            segments = transcribe_wav(
                wav,
                model_size=whisper_model,
                on_progress=progress,
            )
        transcript_source = f"whisper:{whisper_model}"
    else:
        raise ValueError("No VTT provided and transcription disabled.")

    progress("Computing transcript features…")
    features: ClassFeatures = compute_features(segments, duration_sec=duration)

    progress("Assessing class quality…")
    assessment = assess_with_llm(
    features,
    provider=llm_provider,
    api_key=api_key,
    model=llm_model,
    base_url=llm_base_url,
    transcript_segments=segments,
)

    return {
        "assessment": assessment,
        "features": features.to_dict(),
        "meta": {
            "video": video_path.name,
            "duration_sec": duration,
            "transcript_source": transcript_source,
            "segment_count": len(segments),
        },
    }


def find_sibling_vtt(video_path: Path) -> Path | None:
    """If dataset layout pairs mp4 with similarly named vtt, return it."""
    parent = video_path.parent
    stem = video_path.stem
    stem_l = stem.lower()
    # e.g. "B491 Beginner Part  - 2026-06-29" -> "B491"
    session_id = stem.split()[0] if stem.split() else stem
    candidates = list(parent.glob("*.vtt"))
    if not candidates:
        return None
    for c in candidates:
        if session_id and session_id.lower() in c.name.lower():
            return c
    for token in ("malayalam", "english"):
        if token in stem_l:
            for c in candidates:
                if token in c.name.lower():
                    return c
    for c in candidates:
        if stem.split(" - ")[0].lower() in c.name.lower():
            return c
        if c.stem.split("-transcript")[0].lower() in stem_l:
            return c
    if len(candidates) == 1:
        return candidates[0]
    return None
