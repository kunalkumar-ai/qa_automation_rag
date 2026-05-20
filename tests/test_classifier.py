from unittest.mock import patch, MagicMock
from classifier import classify_department


def _mock_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


@patch("classifier.client")
def test_classify_returns_known_department(mock_client):
    mock_client.chat.completions.create.return_value = _mock_response("legal")
    result = classify_department("What is your warranty period?")
    assert result == "legal"


@patch("classifier.client")
def test_classify_returns_unknown_for_unrecognised_question(mock_client):
    mock_client.chat.completions.create.return_value = _mock_response("unknown")
    result = classify_department("What is the best recipe for chocolate cake?")
    assert result == "unknown"


@patch("classifier.client")
def test_classify_returns_unknown_for_invalid_gpt_response(mock_client):
    mock_client.chat.completions.create.return_value = _mock_response("made_up_department")
    result = classify_department("Some ambiguous question")
    assert result == "unknown"


@patch("classifier.client")
def test_classify_strips_whitespace_and_lowercases(mock_client):
    mock_client.chat.completions.create.return_value = _mock_response("  Quality_Control  ")
    result = classify_department("Is your product ISO certified?")
    assert result == "quality_control"
