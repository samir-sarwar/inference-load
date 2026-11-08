import pytest

from inference_load_lab.cost import estimate_cost


def test_cost_accounts_for_all_utf8_message_bytes_and_output_budget():
    payload = {"messages": [{"content": "abcd"}, {"content": "éé"}], "max_tokens": 32}
    assert estimate_cost(payload) == 34
    assert estimate_cost({**payload, "max_completion_tokens": 64}) == 66
    assert estimate_cost({"messages": [{"content": ""}]}) == 4097


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"messages": []},
        {"messages": [{"content": ["image"]}]},
        {"messages": [{"content": "hi"}], "max_tokens": 0},
        {"messages": [{"content": "hi"}], "max_tokens": True},
    ],
)
def test_invalid_length_aware_cost_input_is_rejected(payload):
    with pytest.raises((ValueError, TypeError)):
        estimate_cost(payload)
