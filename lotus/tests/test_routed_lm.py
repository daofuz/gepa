from typing import Any

import pytest
from lotus.models.cascaded_lm import CascadedLM

from lotus.models.lm import LM
from lotus.models.routed_lm import RoutedLM
from lotus.models.routing import RoutingDecision
from lotus.models.soft_prompt_router import (
    _format_router_text,
    _has_non_text_content,
    _resolve_decision_threshold,
    _resolve_strong_class_index,
    messages_to_router_text,
)
from lotus.models.sentiment_router import parse_binary_sentiment
from lotus.sem_ops.sem_filter import sem_filter
from lotus.types import LMOutput


class RecordingLM(LM):
    def __init__(self, name: str, output: str | None = None) -> None:
        super().__init__(model=name, max_tokens=32)
        self.name = name
        self.output = output
        self.calls: list[tuple[list[list[dict[str, str]]], dict[str, Any]]] = []

    def __call__(
        self,
        messages: list[list[dict[str, str]]],
        show_progress_bar: bool = True,
        progress_bar_desc: str = "Processing uncached messages",
        **kwargs: Any,
    ) -> LMOutput:
        self.calls.append((messages, {"progress_bar_desc": progress_bar_desc, **kwargs}))
        outputs = []
        for message_list in messages:
            text = messages_to_router_text(message_list)
            outputs.append(self.output if self.output is not None else f"{self.name}:{text}")
        return LMOutput(outputs=outputs)


class KeywordRouter:
    def route(self, messages: list[list[dict[str, Any]]]) -> list[RoutingDecision]:
        return [
            RoutingDecision(
                route="strong" if "hard" in messages_to_router_text(message_list) else "cheap",
                strong_probability=0.9 if "hard" in messages_to_router_text(message_list) else 0.1,
            )
            for message_list in messages
        ]


def _messages(*texts: str) -> list[list[dict[str, str]]]:
    return [[{"role": "user", "content": text}] for text in texts]


def test_routed_lm_splits_batches_and_restores_order() -> None:
    cheap = RecordingLM("cheap-model")
    strong = RecordingLM("strong-model")
    model = RoutedLM(cheap, strong, KeywordRouter())

    output = model(_messages("easy one", "hard one", "easy two"), show_progress_bar=False, custom="value")

    assert output.outputs == [
        "cheap-model:easy one",
        "strong-model:hard one",
        "cheap-model:easy two",
    ]
    assert [messages_to_router_text(item) for item in cheap.calls[0][0]] == ["easy one", "easy two"]
    assert [messages_to_router_text(item) for item in strong.calls[0][0]] == ["hard one"]
    assert cheap.calls[0][1]["custom"] == "value"
    assert strong.calls[0][1]["custom"] == "value"
    assert model.routing_stats.total == 3
    assert model.routing_stats.cheap == 2
    assert model.routing_stats.strong == 1
    assert model.routing_stats.cheap_rate == pytest.approx(2 / 3)
    assert model.routing_stats.strong_rate == pytest.approx(1 / 3)
    assert [decision.route for decision in model.last_routing_decisions] == ["cheap", "strong", "cheap"]


def test_routed_lm_falls_back_to_strong_when_router_fails() -> None:
    class BrokenRouter:
        def route(self, messages):
            raise RuntimeError("router unavailable")

    cheap = RecordingLM("cheap-model")
    strong = RecordingLM("strong-model")
    model = RoutedLM(cheap, strong, BrokenRouter())

    output = model(_messages("one", "two"), show_progress_bar=False)

    assert output.outputs == ["strong-model:one", "strong-model:two"]
    assert not cheap.calls
    assert model.routing_stats.fallback_to_strong == 2


def test_routed_lm_can_raise_router_errors() -> None:
    class BrokenRouter:
        def route(self, messages):
            raise RuntimeError("router unavailable")

    model = RoutedLM(
        RecordingLM("cheap-model"),
        RecordingLM("strong-model"),
        BrokenRouter(),
        router_error_policy="raise",
    )

    with pytest.raises(RuntimeError, match="router unavailable"):
        model(_messages("one"), show_progress_bar=False)


def test_routed_lm_treats_invalid_router_output_as_failure() -> None:
    class IncompleteRouter:
        def route(self, messages):
            return []

    strong = RecordingLM("strong-model")
    model = RoutedLM(RecordingLM("cheap-model"), strong, IncompleteRouter())

    output = model(_messages("one"), show_progress_bar=False)

    assert output.outputs == ["strong-model:one"]
    assert model.routing_stats.fallback_to_strong == 1


def test_messages_to_router_text_uses_latest_user_turn_and_text_parts() -> None:
    messages = [
        {"role": "user", "content": "example"},
        {"role": "assistant", "content": "example answer"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Context: current row"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                {"type": "text", "text": "Claim: current task"},
            ],
        },
    ]

    assert messages_to_router_text(messages) == "Context: current row\nClaim: current task"
    assert _has_non_text_content(messages)


def test_checkpoint_metadata_controls_threshold_and_class_order() -> None:
    checkpoint = {"selected_decision_threshold": 0.73, "routes": ("mini", "nano")}
    checkpoint_args = {"decision_threshold": 0.5}

    assert _resolve_decision_threshold(checkpoint, checkpoint_args, None) == pytest.approx(0.73)
    assert _resolve_decision_threshold(checkpoint, checkpoint_args, 0.2) == pytest.approx(0.2)
    assert _resolve_strong_class_index(checkpoint, None) == 0
    assert _resolve_strong_class_index(checkpoint, 1) == 1


def test_router_text_supports_input_text_content() -> None:
    messages = [{"role": "user", "content": [{"type": "input_text", "text": "semantic predicate"}]}]

    assert messages_to_router_text(messages) == "semantic predicate"
    assert not _has_non_text_content(messages)


def test_sembench_movie_router_text_format_matches_training_prompt() -> None:
    rendered = (
        "Context:\nA sharp, funny movie.\n\n"
        "Claim: The movie review has positive sentiment."
    )
    assert _format_router_text(rendered, "sembench_movie_review_v1") == (
        "Semantic operator: classify whether a movie review is positive or negative.\n"
        "Review: A sharp, funny movie."
    )


def test_sembench_movie_router_text_format_rejects_other_operators() -> None:
    with pytest.raises(ValueError, match="expects a Lotus filter prompt"):
        _format_router_text("unstructured prompt", "sembench_movie_review_v1")


def test_sem_filter_routes_before_lm_call() -> None:
    cheap = RecordingLM("cheap-model", output="True")
    strong = RecordingLM("strong-model", output="False")
    model = RoutedLM(cheap, strong, KeywordRouter())

    result = sem_filter(
        [{"text": "easy item"}, {"text": "hard item"}],
        model,
        "{text} should be retained",
        show_progress_bar=False,
    )

    assert result.outputs == [True, False]
    assert len(cheap.calls[0][0]) == 1
    assert len(strong.calls[0][0]) == 1


class RetryHardAfterCheap:
    def __init__(self) -> None:
        self.cheap_outputs: list[str] = []

    def route_after_cheap(self, messages, cheap_outputs):
        self.cheap_outputs = list(cheap_outputs)
        return [
            RoutingDecision(
                route=(
                    "strong"
                    if "hard" in messages_to_router_text(message_list)
                    else "cheap"
                ),
                strong_probability=(
                    0.9
                    if "hard" in messages_to_router_text(message_list)
                    else 0.1
                ),
            )
            for message_list in messages
        ]


def test_cascaded_lm_calls_cheap_then_retries_selected_rows() -> None:
    cheap = RecordingLM("cheap-model")
    strong = RecordingLM("strong-model")
    router = RetryHardAfterCheap()
    model = CascadedLM(cheap, strong, router)

    output = model(
        _messages("easy one", "hard one", "easy two"),
        show_progress_bar=False,
    )

    assert output.outputs == [
        "cheap-model:easy one",
        "strong-model:hard one",
        "cheap-model:easy two",
    ]
    assert router.cheap_outputs == [
        "cheap-model:easy one",
        "cheap-model:hard one",
        "cheap-model:easy two",
    ]
    assert len(cheap.calls[0][0]) == 3
    assert len(strong.calls[0][0]) == 1
    assert model.cascade_stats.accepted_cheap == 2
    assert model.cascade_stats.escalated_strong == 1
    assert model.cascade_stats.cheap_calls == 3
    assert model.cascade_stats.total_lm_calls == 4


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("True", True),
        ("POSITIVE", True),
        ("False", False),
        ("NEGATIVE", False),
        ("unclear", None),
        ("True or False", None),
    ],
)
def test_parse_binary_sentiment(output: str, expected: bool | None) -> None:
    assert parse_binary_sentiment(output) is expected

