"""Quality assessment via open-source (Ollama) or OpenAI LLM, with heuristic fallback."""

from __future__ import annotations

import json
import re
import time
from typing import Any
import urllib.error
import urllib.request
from src.evidence_validation import (
    calibrate_confidence,
    review_assessment_evidence,
)

from src.features import ClassFeatures
from src.vtt_parser import Segment
from src.llm_config import LlmSettings, resolve_llm_settings
from src.prompts import SYSTEM_PROMPT, USER_TEMPLATE
VALID_QUALITIES = {"Poor", "Average", "Good", "Excellent"}


VALID_CONFIDENCE = {"low", "medium", "high"}

ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "quality": {
            "type": "string",
            "enum": ["Poor", "Average", "Good", "Excellent"],
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "summary": {"type": "string"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
        },
        "improvements": {
            "type": "array",
            "items": {"type": "string"},
        },
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "time": {"type": "string"},
                    "quote": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["time", "quote", "note"],
                "additionalProperties": False,
            },
        },
        "metrics_used": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "quality",
        "confidence",
        "summary",
        "strengths",
        "improvements",
        "observations",
        "metrics_used",
    ],
    "additionalProperties": False,
}


def _validate_assessment(result: dict[str, Any]) -> dict[str, Any]:
    """Reject incomplete or malformed assessment responses."""
    required = set(ASSESSMENT_SCHEMA["required"])

    if set(result) != required:
        missing = required - set(result)
        extra = set(result) - required
        raise ValueError(
            f"Assessment keys mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    if result["quality"] not in VALID_QUALITIES:
        raise ValueError(f"Invalid quality label: {result['quality']!r}")

    if result["confidence"] not in VALID_CONFIDENCE:
        raise ValueError(
            f"Invalid confidence value: {result['confidence']!r}"
        )

    for field in ("summary",):
        if not isinstance(result[field], str):
            raise ValueError(f"{field} must be a string")

    for field in ("strengths", "improvements", "metrics_used"):
        value = result[field]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"{field} must be a list of strings")

    if not isinstance(result["observations"], list):
        raise ValueError("observations must be a list")

    observation_keys = {"time", "quote", "note"}
    for index, observation in enumerate(result["observations"]):
        if not isinstance(observation, dict):
            raise ValueError(f"observations[{index}] must be an object")
        if set(observation) != observation_keys:
            raise ValueError(
                f"observations[{index}] must contain exactly "
                f"{sorted(observation_keys)}"
            )
        if not all(
            isinstance(observation[key], str)
            for key in observation_keys
        ):
            raise ValueError(
                f"observations[{index}] values must all be strings"
            )

    return result


def _normalize_quote(text: str) -> str:
    """Normalize text for transcript quote matching."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _timestamp_to_seconds(value: str) -> float | None:
    """Parse MM:SS.ss or HH:MM:SS.ss timestamps."""
    value = value.strip().strip("[]()")
    if not value or value.lower() == "unknown":
        return None

    parts = value.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return (
                int(hours) * 3600
                + int(minutes) * 60
                + float(seconds)
            )
    except ValueError:
        return None

    return None



def _validate_observations_against_transcript(
    result: dict[str, Any],
    segments: list[Segment],
    *,
    timestamp_tolerance_sec: float = 3.0,
    strict: bool = True,
) -> None:
    """Validate transcript observations, optionally discarding unsupported ones."""
    valid_observations = []
    rejected_count = 0

    for index, observation in enumerate(result.get("observations", [])):
        quote = _normalize_quote(observation.get("quote", ""))

        if len(quote) < 8:
            reason = f"Observation {index} quote is too short to verify"
            if strict:
                raise ValueError(reason)
            print(f"[LLM] Rejecting observation {index}: quote too short")
            rejected_count += 1
            continue

                
        raw_quote = observation.get("quote", "").strip()

        # First try the quote exactly as supplied.
        matching_segments = [
            segment
            for segment in segments
            if quote in _normalize_quote(segment.text)
        ]

        # The LLM may include the speaker name, although the transcript stores
        # the speaker separately from the segment text.
        if not matching_segments:
            for segment in segments:
                speaker = segment.speaker.strip()
                if not speaker:
                    continue

                speaker_prefix = _normalize_quote(speaker)
                normalized_quote = _normalize_quote(raw_quote)

                if (
                    speaker_prefix
                    and normalized_quote.startswith(speaker_prefix)
                ):
                    quote_without_speaker = normalized_quote[len(speaker_prefix):]

                    if quote_without_speaker in _normalize_quote(segment.text):
                        matching_segments.append(segment)


        if not matching_segments:
            reason = f"Observation {index} quote was not found in the transcript"
            if strict:
                raise ValueError(reason)
            print(
                f"[LLM] Rejecting observation {index}: quote not found. "
                f"Original quote={observation.get('quote', '')!r}; "
                f"Normalized quote={quote!r}",
                flush=True,
            )
            rejected_count += 1
            continue

        timestamp_text = observation.get("time", "").strip()
        timestamp = _timestamp_to_seconds(timestamp_text)

        if timestamp is None:
            if timestamp_text.lower() != "unknown":
                reason = (
                    f"Observation {index} has an invalid timestamp: "
                    f"{timestamp_text!r}"
                )
                if strict:
                    raise ValueError(reason)
                print(f"[LLM] Rejecting observation {index}: invalid timestamp")
                rejected_count += 1
                continue
        elif not any(
            abs(segment.start - timestamp) <= timestamp_tolerance_sec
            for segment in matching_segments
        ):
            reason = f"Observation {index} timestamp does not match its quote"
            if strict:
                raise ValueError(reason)
            print(f"[LLM] Rejecting observation {index}: timestamp mismatch")
            rejected_count += 1
            continue

        valid_observations.append(observation)

    if strict:
        return

    result["observations"] = valid_observations

    if rejected_count:
        result["grounding_warning"] = (
            f"Discarded {rejected_count} observation(s) that could not "
            "be verified against the transcript."
        )
        print(
            f"[LLM] Grounding validation removed {rejected_count} "
            f"observation(s); retained {len(valid_observations)}"
        )

def _parse_json_content(raw: str) -> dict[str, Any]:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")

        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain a JSON object")

        result = json.loads(raw[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("LLM response must be a JSON object")

    quality = result.get("quality")

    if not isinstance(quality, str):
        raise ValueError("LLM response is missing the 'quality' field")

    quality_lookup = {label.lower(): label for label in VALID_QUALITIES}
    normalized = quality_lookup.get(quality.strip().lower())

    if normalized is None:
        raise ValueError(
            f"Invalid quality label: {quality!r}. "
            f"Expected one of {sorted(VALID_QUALITIES)}"
        )

    result["quality"] = normalized
    return result

def _build_user_message(features: ClassFeatures) -> str:
    return USER_TEMPLATE.format(
        metrics_json=json.dumps(features.to_dict(), indent=2),
        opening=features.opening_excerpt or "(no opening transcript)",
        closing=features.closing_excerpt or "(no closing transcript)",
        quotes="\n".join(features.sample_quotes) or "(no quotes)",
    )

def _format_metrics_used(features: ClassFeatures) -> list[str]:
    """Return computed metrics with their actual values."""
    return [
        f"Duration: {features.duration_sec / 60:.1f} minutes",
        f"Transcript cue count: {features.cue_count}",
        f"Speaker count: {features.speaker_count}",
        f"Teacher talk ratio: {features.teacher_talk_ratio:.1%}",
        f"Student talk ratio: {features.student_talk_ratio:.1%}",
        f"Question count: {features.question_count}",
        f"Longest transcript gap: {features.longest_gap_sec:.1f} seconds",
        f"Average cue duration: {features.avg_cue_duration_sec:.2f} seconds",
        f"Words per minute: {features.words_per_minute:.1f}",
    ]

def _chat_assessment(settings: LlmSettings, user_msg: str) -> dict[str, Any]:
    # Use Ollama's native API for local inference.
    if settings.provider == "ollama":
        return _chat_ollama_native(settings, user_msg)

    from openai import OpenAI

    kwargs: dict[str, Any] = {
        "api_key": settings.api_key,
        "timeout": 90.0,
        "max_retries": 0,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    client = OpenAI(**kwargs)

    system = SYSTEM_PROMPT
    if settings.provider == "ollama":
        system += "\nYou MUST respond with a single valid JSON object only, no markdown."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    attempts: list[dict[str, Any]] = [
        {
            "temperature": 0.2,
            "max_tokens": 700,
            "response_format": {"type": "json_object"},
        },
        {"temperature": 0.2, "max_tokens": 700},
    ]
    last_error: Exception | None = None

    for attempt_number, opts in enumerate(attempts, start=1):
        started = time.perf_counter()
        print(
            f"[LLM DEBUG] Attempt {attempt_number}/2 started "
            f"({settings.provider}/{settings.model})",
            flush=True,
        )
        try:
            resp = client.chat.completions.create(
                model=settings.model,
                messages=messages,
                **opts,
            )
            elapsed = time.perf_counter() - started
            content = resp.choices[0].message.content or ""

            print(
                f"[LLM DEBUG] Response received in {elapsed:.1f}s; "
                f"content length={len(content)} characters",
                flush=True,
            )

            try:
                result = _parse_json_content(content)
            except Exception as parse_error:
                print(
                    f"[LLM DEBUG] JSON parsing failed: {parse_error}\n"
                    f"[LLM DEBUG] Raw response (first 3000 chars): "
                    f"{content[:3000]!r}",
                    flush=True,
                )
                raise

            result["engine"] = settings.engine
            result["model"] = settings.model
            return result

        except Exception as e:
            last_error = e
            print(
                f"[LLM DEBUG] Attempt {attempt_number} failed after "
                f"{time.perf_counter() - started:.1f}s: {e!r}",
                flush=True,
            )

    raise RuntimeError(
    f"LLM assessment failed ({settings.provider}/{settings.model}). "
    f"Last error: {last_error!r}. "
    f"Check Ollama availability and model '{settings.model}'."
) from last_error


def _chat_ollama_native(
    settings: LlmSettings,
    user_msg: str,
) -> dict[str, Any]:
    """Call Ollama's native API with constrained JSON output."""
    base_url = (settings.base_url or "http://localhost:11434/v1").rstrip("/")

    # The config normally uses the OpenAI-compatible /v1 URL.
    # Native Ollama endpoints live at the server root.
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    url = f"{base_url}/api/chat"
    messages = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + "\nReturn every required field. "
                "Use only supplied evidence. "
                "A long gap is not proof of disengagement. "
                "Missing camera or homework evidence is unknown, "
                "not automatically a fault."
            ),
        },
        {"role": "user", "content": user_msg},
    ]

    last_error: Exception | None = None

    for attempt in range(1, 3):
        payload = {
            "model": settings.model,
            "stream": False,
            "format": ASSESSMENT_SCHEMA,
            "options": {
                "temperature": 0,
                "num_predict": 1000 if attempt == 1 else 1400,
            },
            "messages": messages,
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        print(
            f"[LLM DEBUG] Ollama native attempt {attempt}/2 started "
            f"({settings.model})",
            flush=True,
        )

        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                response_data = json.loads(response.read().decode("utf-8"))

            elapsed = time.perf_counter() - started
            content = response_data.get("message", {}).get("content", "")

            print(
                f"[LLM DEBUG] Ollama response in {elapsed:.1f}s; "
                f"content length={len(content)}; "
                f"prompt tokens={response_data.get('prompt_eval_count', 'n/a')}; "
                f"output tokens={response_data.get('eval_count', 'n/a')}; "
                f"done_reason={response_data.get('done_reason', 'n/a')}; "
                f"done={response_data.get('done', 'n/a')}",
                flush=True,
            )

            if not response_data.get("done", False):
                raise ValueError("Ollama response did not finish")

            result = _validate_assessment(_parse_json_content(content))
            result["engine"] = settings.engine
            result["model"] = settings.model
            return result

        except Exception as exc:
            last_error = exc
            print(
                f"[LLM DEBUG] Ollama attempt {attempt} failed after "
                f"{time.perf_counter() - started:.1f}s: {exc!r}",
                flush=True,
            )

    raise RuntimeError(
        f"Ollama structured assessment failed ({settings.model}). "
        f"Last error: {last_error!r}"
    ) from last_error


def assess_with_llm(
    features: ClassFeatures,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    transcript_segments: list[Segment] | None = None,
) -> dict[str, Any]:
    settings = resolve_llm_settings(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    if not settings:
        return assess_heuristic(features)

    user_msg = _build_user_message(features)

    for attempt in range(2):
        try:
            request_msg = user_msg

            if attempt == 1:
                request_msg += (
                    "\n\nIMPORTANT VALIDATION FEEDBACK:\n"
                    "Your previous response contained transcript evidence "
                    "that failed validation. Re-check every observation "
                    "against the supplied excerpts. Copy quotes exactly, "
                    "use the timestamp belonging to each quote, and omit "
                    "any observation you cannot verify. Return the complete "
                    "assessment JSON again."
                )

            result = _chat_assessment(settings, request_msg)
            result["metrics_used"] = _format_metrics_used(features)

            # Check the assessment structure before grounding observations.
            core_result = {
                key: value
                for key, value in result.items()
                if key not in {"engine", "model"}
            }
            _validate_assessment(core_result)

            
            if transcript_segments is not None:
                _validate_observations_against_transcript(
                    result,
                    transcript_segments,
                    strict=False,
                )

            # Apply conservative confidence calibration.
            result["confidence"] = calibrate_confidence(
                result["confidence"],
                observation_count=len(result.get("observations", [])),
                visual_evidence_available=False,
            )

            # Warn about claims that may exceed the supplied evidence.
            # Review claims before returning the assessment.
            result, claim_warnings = review_assessment_evidence(
                result,
                visual_evidence=None,
            )

            if claim_warnings:
                print(
                    f"[LLM] Evidence review flagged "
                    f"{len(claim_warnings)} claim(s).",
                    flush=True,
                )

            return result


        except ValueError as exc:
            # Retry only validation failures, not ordinary request errors.
            is_validation_error = (
                "Observation " in str(exc)
                or "Assessment keys mismatch" in str(exc)
                or "Invalid quality label" in str(exc)
                or "Invalid confidence value" in str(exc)
                or "must be " in str(exc)
            )

            if attempt == 0 and is_validation_error:
                print(
                    f"[LLM] Assessment validation failed; retrying once: {exc}",
                    flush=True,
                )
                continue

            fallback = assess_heuristic(features)
            fallback["summary"] = (
                "LLM assessment failed validation after retry. "
                "Showing heuristic fallback."
            )
            fallback["llm_error"] = str(exc)
            return fallback

        except Exception as exc:
            fallback = assess_heuristic(features)
            fallback["summary"] = (
                f"LLM assessment failed ({exc}). "
                "Showing heuristic fallback."
            )
            fallback["llm_error"] = str(exc)
            return fallback

    # Defensive fallback; the loop should always return.
    fallback = assess_heuristic(features)
    fallback["summary"] = (
        "LLM assessment could not be validated. Showing heuristic fallback."
    )
    fallback["llm_error"] = "Assessment validation failed after retry."
    return fallback


def assess_heuristic(features: ClassFeatures) -> dict[str, Any]:
    """Fallback when LLM is unavailable — transparent rule-based score."""
    score = 0
    notes: list[str] = []

    if features.homework_mentions >= 1:
        score += 2
        notes.append("Homework or assignment mentioned in transcript.")
    else:
        notes.append("No clear homework mention detected.")

    if features.question_count >= 15:
        score += 2
    elif features.question_count >= 8:
        score += 1
    notes.append(f"Question/interaction signals: {features.question_count}.")

    if features.intro_signal_count >= 3:
        score += 1
    if features.encouragement_count >= 3:
        score += 1
    if features.camera_background_mentions >= 1:
        score += 1

    if features.longest_gap_sec > 120:
        score -= 2
        notes.append(f"Long silence gap (~{features.longest_gap_sec}s) detected between cues.")
    if features.teacher_talk_ratio > 0.92:
        score -= 1
        notes.append("Teacher-dominated talk ratio; limited student voice in transcript.")

    if score >= 6:
        quality = "Excellent"
    elif score >= 4:
        quality = "Good"
    elif score >= 2:
        quality = "Average"
    else:
        quality = "Poor"

    observations = []
    for q in features.sample_quotes[:5]:
        m = re.match(r"\[([^\]]+)\]\s*([^:]+):\s*(.+)", q)
        if m:
            observations.append({"time": m.group(1), "quote": m.group(3)[:200], "note": "Sample transcript evidence"})
        else:
            observations.append({"time": "—", "quote": q[:200], "note": "Sample transcript evidence"})

    return {
        "quality": quality,
        "confidence": "unvalidated",
        "summary": (
            "Heuristic assessment (LLM not configured or unreachable). "
            "Use Ollama locally (LLM_PROVIDER=ollama) or set OPENAI_API_KEY."
        ),
        "strengths": [n for n in notes if "mentioned" in n or "signals" in n][:3],
        "improvements": [
            "Run `./scripts/setup_ollama.sh` or `ollama pull qwen2.5:7b-instruct`.",
            "Optionally add video frame checks for background and face visibility.",
        ],
        "observations": observations,
        "metrics_used": [
            f"question_count={features.question_count}",
            f"homework_mentions={features.homework_mentions}",
            f"longest_gap_sec={features.longest_gap_sec}",
            f"teacher_talk_ratio={features.teacher_talk_ratio}",
        ],
        "engine": "heuristic",
        "model": "rules",
    }
