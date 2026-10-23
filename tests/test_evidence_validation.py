from src.evidence_validation import (
    calibrate_confidence,
    validate_evidence_claims,
)



def test_flags_camera_claim_without_visual_evidence():
    assessment = {
        "summary": "The camera framing was poor.",
        "strengths": [],
        "improvements": [],
    }

    warnings = validate_evidence_claims(assessment)

    assert any("visual claim" in warning for warning in warnings)


def test_does_not_flag_visual_claim_when_evidence_is_supplied():
    assessment = {
        "summary": "The camera framing was clear.",
        "strengths": [],
        "improvements": [],
    }

    warnings = validate_evidence_claims(
        assessment,
        visual_evidence=["The teacher's face is visible in the supplied frame."],
    )

    assert not any("visual claim" in warning for warning in warnings)


def test_flags_unsupported_learning_claim():
    assessment = {
        "summary": "Students understood every concept.",
        "strengths": [],
        "improvements": [],
    }

    warnings = validate_evidence_claims(assessment)

    assert any("learning or engagement claim" in warning for warning in warnings)


def test_transcript_only_high_confidence_is_capped():
    assert calibrate_confidence(
        "high",
        observation_count=4,
        visual_evidence_available=False,
    ) == "medium"


def test_few_observations_cannot_have_high_confidence():
    assert calibrate_confidence(
        "high",
        observation_count=1,
    ) == "medium"


def test_zero_observations_means_low_confidence():
    assert calibrate_confidence(
        "high",
        observation_count=0,
    ) == "low"


def test_invalid_confidence_is_rejected():
    try:
        calibrate_confidence("certain", observation_count=4)
    except ValueError as exc:
        assert "Invalid confidence" in str(exc)
    else:
        raise AssertionError("Invalid confidence should have been rejected")


def test_flags_active_student_participation_claim():
    assessment = {
        "summary": "Students were actively participating in the discussion.",
        "strengths": [],
        "improvements": [],
        "metrics_used": [],
    }

    warnings = validate_evidence_claims(assessment)

    assert any("actively participating" in warning for warning in warnings)


def test_flags_engagement_claim_in_strengths():
    assessment = {
        "summary": "The instructor guided chess puzzles.",
        "strengths": ["Students remained engaged throughout the class."],
        "improvements": [],
        "metrics_used": [],
    }

    warnings = validate_evidence_claims(assessment)

    assert any("remained engaged" in warning for warning in warnings)


def test_flags_visual_claim_in_improvements():
    assessment = {
        "summary": "The class covered chess puzzles.",
        "strengths": [],
        "improvements": ["Add more diagrams to improve instruction."],
        "metrics_used": [],
    }

    warnings = validate_evidence_claims(assessment)

    assert any("visual claim" in warning for warning in warnings)



from src.evidence_validation import review_assessment_evidence


def test_review_removes_unsupported_visual_improvement():
    assessment = {
        "summary": "The instructor discusses chess puzzles.",
        "strengths": ["The instructor asks a specific question."],
        "improvements": [
            "Incorporating more visual aids could enhance understanding."
        ],
    }

    reviewed, warnings = review_assessment_evidence(assessment)

    assert reviewed["improvements"] == []
    assert warnings
    # The input must not be modified.
    assert assessment["improvements"]


def test_review_preserves_supported_neutral_observation():
    assessment = {
        "summary": "The instructor discusses chess puzzles.",
        "strengths": ["The instructor asks a specific question."],
        "improvements": ["Consider summarizing a complex puzzle's key principle."],
    }

    reviewed, warnings = review_assessment_evidence(assessment)

    assert reviewed["strengths"] == assessment["strengths"]
    assert reviewed["improvements"] == assessment["improvements"]



def test_review_removes_unsupported_engagement_from_summary():
    assessment = {
        "summary": (
            "The session covers chess puzzles. "
            "Students were actively engaged in the discussion."
        ),
        "strengths": ["The instructor asks students questions."],
        "improvements": [],
    }

    reviewed, warnings = review_assessment_evidence(assessment)

    assert reviewed["summary"] == "The session covers chess puzzles."
    assert any("removed" in warning for warning in warnings)
    assert "actively engaged" not in reviewed["summary"]


def test_review_warnings_describe_final_assessment():
    assessment = {
        "summary": "The instructor explains chess puzzles.",
        "strengths": [
            "Students were actively participating in the discussion."
        ],
        "improvements": [
            "Consider summarizing a complex puzzle's key principle."
        ],
    }

    reviewed, warnings = review_assessment_evidence(assessment)

    assert reviewed["strengths"] == []
    assert not any("actively participating" in warning for warning in warnings)
    assert reviewed["improvements"] == [
        "Consider summarizing a complex puzzle's key principle."
    ]
