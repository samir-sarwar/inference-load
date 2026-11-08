"""Cheap text-request cost proxy, independent of the model's tokenizer."""

import math

ESTIMATOR = "ceil(utf8_content_bytes / 4) + output_budget"
UNKNOWN_OUTPUT_BUDGET = 4096


def estimate_cost(payload):
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Length-aware admission requires nonempty text messages.")
    content_bytes = 0
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TypeError("Length-aware admission supports string message content.")
        content_bytes += len(message["content"].encode("utf-8"))
    budget = payload.get(
        "max_completion_tokens", payload.get("max_tokens", UNKNOWN_OUTPUT_BUDGET)
    )
    if type(budget) is not int or budget <= 0:
        raise ValueError("The output-token budget must be a positive integer.")
    return max(1, math.ceil(content_bytes / 4)) + budget
