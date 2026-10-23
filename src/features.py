"""Transcript-derived features for class quality assessment."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

from src.vtt_parser import Segment, format_timestamp

_QUESTION = re.compile(
    r"\?|\b(can you|could you|tell me|what is|who is|how do|why do|shall we)\b",
    re.I,
)
_HW = re.compile(r"\b(homework|home work|\bhw\b|assignment)\b", re.I)
_INTRO = re.compile(
    r"\b(introduc|welcome|hello everyone|good morning|good evening|new join)\b",
    re.I,
)
_CAMERA = re.compile(r"\b(camera|video on|turn on your camera|background)\b", re.I)
_ENERGY = re.compile(r"\b(great job|well done|excellent|awesome|fantastic)\b", re.I)


@dataclass
class ClassFeatures:
    duration_sec: float
    cue_count: int
    speaker_count: int
    teacher_speaker: str
    teacher_talk_ratio: float
    student_talk_ratio: float
    question_count: int
    homework_mentions: int
    intro_signal_count: int
    camera_background_mentions: int
    encouragement_count: int
    longest_gap_sec: float
    avg_cue_duration_sec: float
    words_per_minute: float
    opening_excerpt: str
    closing_excerpt: str
    sample_quotes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _guess_teacher(segments: list[Segment]) -> str:
    by_speaker = Counter(s.speaker for s in segments)
    if not by_speaker:
        return "Unknown"
    # Tutors often have the most lines; exclude generic device names slightly
    ranked = by_speaker.most_common()
    return ranked[0][0]


def _excerpt(segments: list[Segment], start_sec: float, end_sec: float, max_chars: int = 3500) -> str:
    lines: list[str] = []
    n = 0
    for s in segments:
        if s.end < start_sec or s.start > end_sec:
            continue
        line = f"[{format_timestamp(s.start)}] {s.speaker}: {s.text}"
        if n + len(line) > max_chars:
            break
        lines.append(line)
        n += len(line)
    return "\n".join(lines)


def _pick_sample_quotes(segments: list[Segment], teacher: str, limit: int = 15) -> list[str]:
    scored: list[tuple[int, Segment]] = []
    for s in segments:
        score = 0
        if s.speaker == teacher:
            score += 1
        if _QUESTION.search(s.text):
            score += 3
        if _HW.search(s.text):
            score += 4
        if _INTRO.search(s.text):
            score += 2
        if _CAMERA.search(s.text):
            score += 2
        if len(s.text) > 40:
            score += 1
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda x: (-x[0], x[1].start))
    quotes: list[str] = []
    seen: set[str] = set()
    for _, s in scored:
        q = f"[{format_timestamp(s.start)}] {s.speaker}: {s.text}"
        if q in seen:
            continue
        seen.add(q)
        quotes.append(q)
        if len(quotes) >= limit:
            break
    return quotes


def compute_features(segments: list[Segment], duration_sec: float | None = None) -> ClassFeatures:
    if not segments:
        duration = duration_sec or 0.0
        return ClassFeatures(
            duration_sec=duration,
            cue_count=0,
            speaker_count=0,
            teacher_speaker="Unknown",
            teacher_talk_ratio=0.0,
            student_talk_ratio=0.0,
            question_count=0,
            homework_mentions=0,
            intro_signal_count=0,
            camera_background_mentions=0,
            encouragement_count=0,
            longest_gap_sec=0.0,
            avg_cue_duration_sec=0.0,
            words_per_minute=0.0,
            opening_excerpt="",
            closing_excerpt="",
            sample_quotes=[],
        )

    if duration_sec is None:
        duration_sec = max(s.end for s in segments)

    teacher = _guess_teacher(segments)
    speakers = {s.speaker for s in segments}
    teacher_segs = [s for s in segments if s.speaker == teacher]
    teacher_chars = sum(len(s.text) for s in teacher_segs)
    total_chars = sum(len(s.text) for s in segments) or 1

    gaps = [segments[i + 1].start - segments[i].end for i in range(len(segments) - 1)]
    longest_gap = max(gaps) if gaps else 0.0
    cue_durations = [max(0.0, s.end - s.start) for s in segments]
    avg_cue = sum(cue_durations) / len(cue_durations)

    all_text = " ".join(s.text for s in segments)
    words = len(all_text.split())
    wpm = words / (duration_sec / 60) if duration_sec > 0 else 0.0

    return ClassFeatures(
        duration_sec=duration_sec,
        cue_count=len(segments),
        speaker_count=len(speakers),
        teacher_speaker=teacher,
        teacher_talk_ratio=round(teacher_chars / total_chars, 3),
        student_talk_ratio=round(1 - teacher_chars / total_chars, 3),
        question_count=len(_QUESTION.findall(all_text)),
        homework_mentions=len(_HW.findall(all_text)),
        intro_signal_count=len(_INTRO.findall(all_text)),
        camera_background_mentions=len(_CAMERA.findall(all_text)),
        encouragement_count=len(_ENERGY.findall(all_text)),
        longest_gap_sec=round(longest_gap, 1),
        avg_cue_duration_sec=round(avg_cue, 2),
        words_per_minute=round(wpm, 1),
        opening_excerpt=_excerpt(segments, 0, min(600, duration_sec * 0.15)),
        closing_excerpt=_excerpt(segments, max(0, duration_sec - 300), duration_sec),
        sample_quotes=_pick_sample_quotes(segments, teacher),
    )
