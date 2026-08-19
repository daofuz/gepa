import re
from pathlib import Path
from typing import Any

from lotus.models.routing import RoutingDecision
from lotus.models.soft_prompt_router import (
    CPUSoftPromptRouter,
    _format_router_text,
    _has_non_text_content,
    messages_to_router_text,
)


def parse_binary_sentiment(output: str) -> bool | None:
    """Return True for positive, False for negative, or None if ambiguous."""

    labels = set(re.findall(r"\b(true|false|positive|negative)\b", output.lower()))
    positive = bool(labels & {"true", "positive"})
    negative = bool(labels & {"false", "negative"})
    if positive == negative:
        return None
    return positive


class CPUSentimentDisagreementRouter(CPUSoftPromptRouter):
    """Escalate when a local sentiment model disagrees with the cheap LM.

    The loaded binary classifier predicts P(positive). If the cheap answer is
    negative, the disagreement score is P(positive); if the cheap answer is
    positive, it is 1 - P(positive). Scores at or above the checkpoint threshold
    are sent to the strong model.
    """

    def __init__(self, checkpoint_path: str | Path, **kwargs: Any) -> None:
        kwargs.setdefault("strong_class_index", 1)
        super().__init__(checkpoint_path, **kwargs)

    def route_after_cheap(
        self,
        messages: list[list[dict[str, Any]]],
        cheap_outputs: list[str],
    ) -> list[RoutingDecision]:
        if len(messages) != len(cheap_outputs):
            raise ValueError("cheap_outputs must have the same length as messages")

        texts = [messages_to_router_text(message_list) for message_list in messages]
        routed_indices: list[int] = []
        routed_texts: list[str] = []
        cheap_sentiments: list[bool] = []
        decisions: list[RoutingDecision | None] = [None] * len(messages)

        for index, (message_list, text, cheap_output) in enumerate(
            zip(messages, texts, cheap_outputs)
        ):
            has_non_text = _has_non_text_content(message_list)
            if has_non_text and self.non_text_policy == "raise":
                raise ValueError("Sentiment router received non-text content")

            cheap_positive = parse_binary_sentiment(cheap_output)
            if (
                not text
                or cheap_positive is None
                or (has_non_text and self.non_text_policy == "strong")
            ):
                decisions[index] = RoutingDecision(
                    route="strong", strong_probability=1.0
                )
                continue

            routed_indices.append(index)
            routed_texts.append(
                _format_router_text(text, self.router_text_format)
            )
            cheap_sentiments.append(cheap_positive)

        with self._torch.inference_mode():
            for start in range(0, len(routed_texts), self.batch_size):
                batch = routed_texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                logits = self._model(
                    input_ids=encoded["input_ids"].to("cpu"),
                    attention_mask=encoded["attention_mask"].to("cpu"),
                )
                positive_probabilities = self._torch.softmax(
                    logits, dim=-1
                )[:, self.strong_class_index].tolist()

                batch_indices = routed_indices[start : start + self.batch_size]
                batch_sentiments = cheap_sentiments[
                    start : start + self.batch_size
                ]
                for original_index, cheap_positive, p_positive in zip(
                    batch_indices, batch_sentiments, positive_probabilities
                ):
                    disagreement = (
                        1.0 - p_positive if cheap_positive else p_positive
                    )
                    decisions[original_index] = RoutingDecision(
                        route=(
                            "strong"
                            if disagreement >= self.decision_threshold
                            else "cheap"
                        ),
                        strong_probability=float(disagreement),
                    )

        if any(decision is None for decision in decisions):
            raise RuntimeError("Cascade router did not decide every request")
        return [decision for decision in decisions if decision is not None]
