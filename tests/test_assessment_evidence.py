import pytest

from src.assess import _validate_assessment


@pytest.fixture
def assessment():
    return {
        "quality": "Good",
        "confidence": "high",
        "summary": (
            "The instructor asked students questions about chess puzzles."
        ),
        "strengths": [
            "The instructor asked students to suggest moves."
        ],
        "improvements": [
            "Consider allowing more time for student responses."
        ],
        "observations": [
            {
                "time": "[06:51.31]",
                "quote": (
                    "Alright, have you remembered what are we "
                    "doing in the last class?"
                ),
                "note": "The instructor checks recall of the previous lesson.",
            }
        ],
        "metrics_used": [
            "Teacher talk ratio: 0.745",
            "Transcript evidence of questions",
        ],
    }


def test_assessment_accepts_well_formed_evidence(assessment):
    result = _validate_assessment(assessment)
    assert result["quality"] == "Good"


def test_missing_visual_evidence_is_not_a_schema_error(assessment):
    # Schema validation alone cannot determine whether visual claims
    # are supported. This test documents that limitation.
    assessment["improvements"].append(
        "Improve camera framing and lighting."
    )

    result = _validate_assessment(assessment)
    assert "camera framing" in result["improvements"][-1]


def test_high_confidence_is_currently_accepted(assessment):
    # This documents current behavior, not the desired final policy:
    # confidence is currently checked only against an allowed enum.
    assessment["confidence"] = "high"

    result = _validate_assessment(assessment)
    assert result["confidence"] == "high"


def test_invalid_confidence_label_is_rejected(assessment):
    assessment["confidence"] = "certain"

    with pytest.raises(ValueError, match="Invalid confidence value"):
        _validate_assessment(assessment)
