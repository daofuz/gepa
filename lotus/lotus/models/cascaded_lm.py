import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal

from lotus.models.lm import LM
from lotus.models.routing import CascadeRouter, RoutingDecision
from lotus.types import LMOutput, LMStats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CascadeStats:
    total: int = 0
    accepted_cheap: int = 0
    escalated_strong: int = 0
    fallback_to_strong: int = 0

    @property
    def cheap_calls(self) -> int:
        return self.total

    @property
    def strong_calls(self) -> int:
        return self.escalated_strong

    @property
    def escalation_rate(self) -> float:
        return self.escalated_strong / self.total if self.total else 0.0

    @property
    def total_lm_calls(self) -> int:
        return self.cheap_calls + self.strong_calls


class CascadedLM(LM):
    """Call the cheap LM first, then selectively retry with the strong LM."""

    def __init__(
        self,
        cheap_lm: LM,
        strong_lm: LM,
        router: CascadeRouter,
        *,
        router_error_policy: Literal["strong", "raise"] = "strong",
        cache: Any = None,
    ) -> None:
        if cheap_lm is strong_lm:
            raise ValueError("cheap_lm and strong_lm must be different LM instances")
        if router_error_policy not in {"strong", "raise"}:
            raise ValueError("router_error_policy must be 'strong' or 'raise'")

        self.cheap_lm = cheap_lm
        self.strong_lm = strong_lm
        self.router = router
        self.router_error_policy = router_error_policy
        super().__init__(
            model=f"cascade/{cheap_lm.model}-to-{strong_lm.model}",
            max_ctx_len=min(cheap_lm.max_ctx_len, strong_lm.max_ctx_len),
            max_tokens=max(cheap_lm.max_tokens, strong_lm.max_tokens),
            max_batch_size=max(cheap_lm.max_batch_size, strong_lm.max_batch_size),
            tokenizer=strong_lm.tokenizer,
            cache=cache,
        )

        shared_stats = cheap_lm.stats + strong_lm.stats
        self.stats = shared_stats
        self.cheap_lm.stats = shared_stats
        self.strong_lm.stats = shared_stats
        self._cascade_stats = CascadeStats()
        self._stats_lock = threading.Lock()
        self._last_routing_decisions: tuple[RoutingDecision, ...] = ()

    @property
    def cascade_stats(self) -> CascadeStats:
        with self._stats_lock:
            return self._cascade_stats

    @property
    def last_routing_decisions(self) -> tuple[RoutingDecision, ...]:
        with self._stats_lock:
            return self._last_routing_decisions

    def __call__(
        self,
        messages: list[list[dict[str, str]]],
        show_progress_bar: bool = True,
        progress_bar_desc: str = "Processing uncached messages",
        **kwargs: Any,
    ) -> LMOutput:
        if not messages:
            return LMOutput(
                outputs=[],
                logprobs=[] if kwargs.get("logprobs") else None,
            )

        cheap_output = self.cheap_lm(
            messages,
            show_progress_bar=show_progress_bar,
            progress_bar_desc=f"{progress_bar_desc} [cheap]",
            **kwargs,
        )
        if len(cheap_output.outputs) != len(messages):
            raise ValueError(
                "Cheap LM returned a different number of outputs than inputs"
            )

        fallback_count = 0
        try:
            decisions = self.router.route_after_cheap(
                messages, cheap_output.outputs
            )
            self._validate_decisions(messages, decisions)
        except Exception:
            if self.router_error_policy == "raise":
                raise
            logger.exception(
                "Cascade router failed; retrying the batch with the strong LM"
            )
            fallback_count = len(messages)
            decisions = [
                RoutingDecision(route="strong", strong_probability=1.0)
                for _ in messages
            ]

        strong_indices = [
            index
            for index, decision in enumerate(decisions)
            if decision.route == "strong"
        ]
        outputs = list(cheap_output.outputs)
        logprobs = (
            list(cheap_output.logprobs)
            if cheap_output.logprobs is not None
            else None
        )

        if strong_indices:
            strong_output = self.strong_lm(
                [messages[index] for index in strong_indices],
                show_progress_bar=show_progress_bar,
                progress_bar_desc=f"{progress_bar_desc} [strong retry]",
                **kwargs,
            )
            if len(strong_output.outputs) != len(strong_indices):
                raise ValueError(
                    "Strong LM returned a different number of outputs than escalations"
                )
            if (logprobs is None) != (strong_output.logprobs is None):
                raise ValueError("Child LMs returned inconsistent logprobs")
            for retry_index, original_index in enumerate(strong_indices):
                outputs[original_index] = strong_output.outputs[retry_index]
                if logprobs is not None and strong_output.logprobs is not None:
                    logprobs[original_index] = strong_output.logprobs[retry_index]

        with self._stats_lock:
            current = self._cascade_stats
            self._cascade_stats = CascadeStats(
                total=current.total + len(messages),
                accepted_cheap=current.accepted_cheap
                + len(messages)
                - len(strong_indices),
                escalated_strong=current.escalated_strong
                + len(strong_indices),
                fallback_to_strong=current.fallback_to_strong
                + fallback_count,
            )
            self._last_routing_decisions = tuple(decisions)

        return LMOutput(outputs=outputs, logprobs=logprobs)

    @staticmethod
    def _validate_decisions(
        messages: list[list[dict[str, str]]],
        decisions: list[RoutingDecision],
    ) -> None:
        if len(decisions) != len(messages):
            raise ValueError(
                f"Router returned {len(decisions)} decisions for "
                f"{len(messages)} requests"
            )
        if any(
            decision.route not in {"cheap", "strong"}
            for decision in decisions
        ):
            raise ValueError("Cascade router returned an invalid route")

    def count_tokens(self, messages: list[dict[str, str]] | str) -> int:
        return self.strong_lm.count_tokens(messages)

    def encode_text(self, text: str) -> list[int]:
        return self.strong_lm.encode_text(text)

    def decode_tokens(self, tokens: list[int]) -> str:
        return self.strong_lm.decode_tokens(tokens)

    def get_model_name(self) -> str:
        return self.strong_lm.get_model_name()

    def is_deepseek(self) -> bool:
        return self.strong_lm.is_deepseek()

    def is_reasoning_model(self) -> bool:
        return self.strong_lm.is_reasoning_model()

    def reset_stats(self) -> None:
        shared_stats = LMStats()
        self.stats = shared_stats
        self.cheap_lm.stats = shared_stats
        self.strong_lm.stats = shared_stats
        with self._stats_lock:
            self._cascade_stats = CascadeStats()
            self._last_routing_decisions = ()

    def reset_cache(self, max_size: int | None = None) -> None:
        seen: set[int] = set()
        for model in (self, self.cheap_lm, self.strong_lm):
            cache = model.cache
            if cache is not None and id(cache) not in seen:
                cache.reset(max_size)
                seen.add(id(cache))
