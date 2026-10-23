"""Parse WebVTT transcripts into structured segments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str


_TIME = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
)
_SPEAKER = re.compile(r"^([^:]{1,80}):\s*(.+)$", re.DOTALL)


def _ts_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def parse_vtt(path: str | Path) -> list[Segment]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    segments: list[Segment] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = _TIME.match(line)
        if not m:
            i += 1
            continue
        start = _ts_to_seconds(m.group("start"))
        end = _ts_to_seconds(m.group("end"))
        i += 1
        body_lines: list[str] = []
        while i < len(lines) and lines[i].strip() and not _TIME.match(lines[i]):
            body_lines.append(lines[i].strip())
            i += 1
        body = " ".join(body_lines).strip()
        if not body:
            continue
        sm = _SPEAKER.match(body)
        if sm:
            speaker, utterance = sm.group(1).strip(), sm.group(2).strip()
        else:
            speaker, utterance = "Unknown", body
        segments.append(Segment(start=start, end=end, speaker=speaker, text=utterance))
    return segments
