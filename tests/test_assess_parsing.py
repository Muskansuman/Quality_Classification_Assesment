import sys
from pathlib import Path

# Add the repository root to Python's import path.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from src.assess import _parse_json_content


def test_valid_quality():
    result = _parse_json_content(
        '{"quality": "Good", "summary": "Clear explanation"}'
    )
    assert result["quality"] == "Good"


def test_normalizes_quality():
    result = _parse_json_content('{"quality": " excellent "}')
    assert result["quality"] == "Excellent"


@pytest.mark.parametrize(
    "raw",
    [
        '{"summary": "No quality label"}',
        '{"quality": "Outstanding"}',
        '[]',
        'not json',
    ],
)
def test_rejects_invalid_assessment(raw):
    with pytest.raises((ValueError, TypeError)):
        _parse_json_content(raw)