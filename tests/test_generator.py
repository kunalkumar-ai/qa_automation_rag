from unittest.mock import patch, MagicMock
from generator import generate_answer, NO_INFO_RESPONSE


def _mock_chat_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


@patch("generator.client")
def test_generate_returns_answer_when_chunks_provided(mock_client):
    mock_client.chat.completions.create.return_value = _mock_chat_response(
        "The warranty period is 24 months from the date of purchase."
    )
    result = generate_answer("What is the warranty?", ["Warranty covers 24 months."])
    assert "24 months" in result


@patch("generator.client")
def test_generate_returns_no_info_response_when_no_chunks(mock_client):
    result = generate_answer("What is the warranty?", [])
    assert result == NO_INFO_RESPONSE
    mock_client.chat.completions.create.assert_not_called()


@patch("generator.client")
def test_generate_includes_all_chunks_in_context(mock_client):
    mock_client.chat.completions.create.return_value = _mock_chat_response("Some answer.")
    chunks = ["Chunk one.", "Chunk two.", "Chunk three."]
    generate_answer("Some question?", chunks)

    system_message = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Chunk one." in system_message
    assert "Chunk two." in system_message
    assert "Chunk three." in system_message
