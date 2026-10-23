"""Deterministic checks for unsupported assessment claims."""

import re
from typing import Any


VISUAL_CLAIM_PATTERNS = [
    r"\bcamera\b",
    r"\blighting\b",
    r"\bbackground\b",
    r"\bfacial expressions?\b",
    r"\bgestures?\b",
    r"\bvisual aids?\b",
    r"\bdiagram(s)?\b",
    r"\bscreen visibility\b",
    r"\baudio quality\b",
]


LEARNING_CLAIM_PATTERNS = [
    r"\bstudents?\s+(?:were|are|remained|seemed|appeared)\s+"
    r"(?:actively\s+)?(?:engaged|participating|involved|attentive)\b",
    r"\bstudents?\s+(?:actively\s+)?participat(?:ed|ing|e)\b",
    r"\ball\s+students?\b.{0,50}\bparticipat(?:ed|ing|e)\b",
    r"\bstudents?\s+(?:clearly\s+)?understood\b",
    r"\bstudents?\s+(?:have\s+)?learned\b",
    r"\bstudents?\s+mastered\b",
    r"\bthe\s+students?'?\s+understanding\b",
    r"\b(?:demonstrates?|proves?)\s+(?:that\s+)?students?\s+"
    r"(?:understood|learned|were engaged)\b",
]


VISUAL_EVIDENCE_FIELDS = {
    "visual_evidence",
    "video_observations",
    "frame_observations",
}


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def validate_evidence_claims(
    assessment: dict[str, Any],
    *,
    visual_evidence: list[str] | None = None,
) -> list[str]:
    """Return warnings for claims that need evidence review.

    This function deliberately warns rather than automatically rejecting
    claims. A matching phrase may be justified by evidence not represented
    in the transcript excerpts.
    """
    warnings: list[str] = []

    text_fields = [
    "summary",
    "strengths",
    "improvements",
    "metrics_used",
]

    for field in text_fields:
        value = assessment.get(field, "")
        items = value if isinstance(value, list) else [value]

        for item in items:
            if not isinstance(item, str):
                continue

            if (
                not visual_evidence
                and _contains_any(item, VISUAL_CLAIM_PATTERNS)
            ):
                warnings.append(
                    f"{field}: possible visual claim without supplied "
                    f"visual evidence: {item}"
                )

            if _contains_any(item, LEARNING_CLAIM_PATTERNS):
                warnings.append(
                    f"{field}: possible learning or engagement claim "
                    f"requiring supporting evidence: {item}"
                )

    return warnings


def calibrate_confidence(
    confidence: str,
    *,
    observation_count: int,
    visual_evidence_available: bool = False,
) -> str:
    """Apply a conservative confidence ceiling to transcript-only results.

    This is a guardrail, not a statistical confidence estimate.
    """
    if confidence not in {"low", "medium", "high"}:
        raise ValueError(f"Invalid confidence value: {confidence!r}")

    # A small number of observations cannot establish high confidence.
    if observation_count < 3:
        return "low" if observation_count == 0 else "medium"

    # Transcript-only evidence leaves important teaching dimensions unknown.
    if not visual_evidence_available and confidence == "high":
        return "medium"

    return confidence


def review_assessment_evidence(
    assessment: dict[str, Any],
    *,
    visual_evidence: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Remove unsupported claims from summaries and list fields.

    Warnings describe claims remaining in the final assessment. Removed
    summary claims are reported without repeating the unsupported claim.
    """
    import copy

    reviewed = copy.deepcopy(assessment)
    removed_summary_claim = False
    removed_list_claim = False

    # Remove individual summary sentences that make unsupported visual or
    # learning/engagement claims. Keep the remaining factual description.
    summary = reviewed.get("summary", "")
    if isinstance(summary, str) and summary.strip():
        sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
        safe_sentences = []

        for sentence in sentences:
            unsupported_visual = (
                not visual_evidence
                and _contains_any(sentence, VISUAL_CLAIM_PATTERNS)
            )
            unsupported_learning = _contains_any(
                sentence, LEARNING_CLAIM_PATTERNS
            )

            if unsupported_visual or unsupported_learning:
                removed_summary_claim = True
                continue

            safe_sentences.append(sentence)

        reviewed["summary"] = " ".join(safe_sentences).strip()

        if not reviewed["summary"]:
            reviewed["summary"] = (
                "The session content is described using the available "
                "transcript. Overall engagement and learning outcomes "
                "cannot be established from transcript evidence alone."
            )

    # Remove unsupported claims from strengths and improvements.
    for field in ("strengths", "improvements"):
        items = reviewed.get(field, [])
        if not isinstance(items, list):
            continue

        safe_items = []
        for item in items:
            if not isinstance(item, str):
                safe_items.append(item)
                continue

            unsupported_visual = (
                not visual_evidence
                and _contains_any(item, VISUAL_CLAIM_PATTERNS)
            )
            unsupported_learning = _contains_any(
                item, LEARNING_CLAIM_PATTERNS
            )
            if unsupported_visual or unsupported_learning:
                removed_list_claim = True
                continue

            safe_items.append(item)

        reviewed[field] = safe_items

    # Recalculate warnings after filtering, so they describe the final
    # assessment rather than claims that have already been removed.
    warnings = validate_evidence_claims(
        reviewed,
        visual_evidence=visual_evidence,
    )

    if removed_summary_claim:
        warnings.append(
            "summary: one or more unsupported visual or engagement "
            "sentences were removed; review the summary for completeness"
        )

    if removed_list_claim:
        warnings.append(
            "strengths/improvements: one or more unsupported claims "
            "were removed; review the final assessment for completeness"
        )

    reviewed.pop("evidence_warnings", None)
    if warnings:
        reviewed["evidence_warnings"] = warnings

    return reviewed, warnings
