
import pytest

from src.assess import _validate_observations_against_transcript
from src.vtt_parser import Segment


@pytest.fixture
def transcript():
    return [
        Segment(
            start=411.31,
            end=413.98,
            speaker="TSP Abhishek",
            text=(
                "Alright, have you remembered what are we "
                "doing in the last class?"
            ),
        )
    ]


def make_result(time, quote):
    return {
        "observations": [
            {
                "time": time,
                "quote": quote,
                "note": "Instructor asks a recall question.",
            }
        ]
    }


def test_exact_transcript_quote_passes(transcript):
    result = make_result(
        "[06:51.31]",
        "Alright, have you remembered what are we doing in the last class?",
    )

    _validate_observations_against_transcript(result, transcript)


def test_punctuation_differences_pass(transcript):
    result = make_result(
        "06:51.31",
        "Alright have you remembered what are we doing in the last class",
    )

    _validate_observations_against_transcript(result, transcript)


def test_invented_quote_fails(transcript):
    result = make_result(
        "06:51.31",
        "The instructor assigned homework to everyone today.",
    )

    with pytest.raises(ValueError, match="quote was not found"):
        _validate_observations_against_transcript(result, transcript)


def test_incorrect_timestamp_fails(transcript):
    result = make_result(
        "08:51.31",
        "Alright, have you remembered what are we doing in the last class?",
    )

    with pytest.raises(ValueError):
        _validate_observations_against_transcript(result, transcript)


def test_unknown_timestamp_is_allowed(transcript):
    result = make_result(
        "unknown",
        "Alright, have you remembered what are we doing in the last class?",
    )

    _validate_observations_against_transcript(result, transcript)
