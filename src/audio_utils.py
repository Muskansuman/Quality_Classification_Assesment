"""Audio extraction and lightweight metrics via ffmpeg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def extract_audio_wav(video_path: Path, wav_path: Path, sample_rate: int = 16000) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def probe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(out.stdout)
    return float(data["format"]["duration"])


def load_wav_mono_float32(path: Path, expected_sample_rate: int = 16000):
    """Load 16-bit PCM WAV as float32 mono (avoids PyAV in Whisper)."""
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit WAV from ffmpeg, got sample width {sampwidth}")
        raw = wf.readframes(wf.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != expected_sample_rate:
        raise ValueError(
            f"Expected {expected_sample_rate} Hz audio, got {rate} Hz. "
            "Re-extract with ffmpeg -ar 16000."
        )
    return audio
