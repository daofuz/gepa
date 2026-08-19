import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal

from lotus.models.lm import LM
from lotus.models.routing import Router, RoutingDecision
from lotus.types import LMOutput, LMStats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingStats:
    total: int = 0
    cheap: int = 0
    strong: int = 0
    fallback_to_strong: int = 0

    @property
    def cheap_rate(self) -> float:
        return self.cheap / self.total if self.total else 0.0

    @property
    def strong_rate(self) -> float:
        return self.strong / self.total if self.total else 0.0


class RoutedLM(LM):
    """Route each request in an LM batch to a cheap or strong Lotus LM.

    Semantic operators already send one rendered message list per record to the
    configured LM. This wrapper routes those records, invokes the two child LMs in
    sub-batches, and restores the original order before returning ``LMOutput``.
    """

    def __init__(
        self,
        cheap_lm: LM,
        strong_lm: LM,
        router: Router,
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
            model=f"routed/{cheap_lm.model}-to-{strong_lm.model}",
            max_ctx_len=min(cheap_lm.max_ctx_len, strong_lm.max_ctx_len),
            max_tokens=max(cheap_lm.max_tokens, strong_lm.max_tokens),
            max_batch_size=max(cheap_lm.max_batch_size, strong_lm.max_batch_size),
            tokenizer=strong_lm.tokenizer,
            cache=cache,
        )

        # Both children write into one stats object, preserving Lotus's existing
        # usage and operator-cache accounting through the wrapper.
        shared_stats = cheap_lm.stats + strong_lm.stats
        self.stats = shared_stats
        self.cheap_lm.stats = shared_stats
        self.strong_lm.stats = shared_stats

        self._routing_stats = RoutingStats()
        self._routing_stats_lock = threading.Lock()
        self._last_routing_decisions: tuple[RoutingDecision, ...] = ()

    @property
    def routing_stats(self) -> RoutingStats:
        with self._routing_stats_lock:
            return self._routing_stats

    @property
    def last_routing_decisions(self) -> tuple[RoutingDecision, ...]:
        """Decisions from the most recently completed batch, in input order."""

        with self._routing_stats_lock:
            return self._last_routing_decisions

    def __call__(
        self,
        messages: list[list[dict[str, str]]],
        show_progress_bar: bool = True,
        progress_bar_desc: str = "Processing uncached messages",
        **kwargs: Any,
    ) -> LMOutput:
        if not messages:
            return LMOutput(outputs=[], logprobs=[] if kwargs.get("logprobs") else None)

        fallback_count = 0
        try:
            decisions = self.router.route(messages)  # type: ignore[arg-type]
            self._validate_decisions(messages, decisions)
        except Exception:
            if self.router_error_policy == "raise":
                raise
            logger.exception("Router failed; sending the batch to the strong LM")
            fallback_count = len(messages)
            decisions = [RoutingDecision(route="strong", strong_probability=1.0) for _ in messages]

        cheap_indices = [index for index, decision in enumerate(decisions) if decision.route == "cheap"]
        strong_indices = [index for index, decision in enumerate(decisions) if decision.route == "strong"]

        branch_outputs: list[tuple[list[int], LMOutput]] = []
        if cheap_indices:
            branch_outputs.append(
                (
                    cheap_indices,
                    self.cheap_lm(
                        [messages[index] for index in cheap_indices],
                        show_progress_bar=show_progress_bar,
                        progress_bar_desc=f"{progress_bar_desc} [cheap]",
                        **kwargs,
                    ),
                )
            )
        if strong_indices:
            branch_outputs.append(
                (
                    strong_indices,
                    self.strong_lm(
                        [messages[index] for index in strong_indices],
                        show_progress_bar=show_progress_bar,
                        progress_bar_desc=f"{progress_bar_desc} [strong]",
                        **kwargs,
                    ),
                )
            )

        outputs = [""] * len(messages)
        has_logprobs = any(output.logprobs is not None for _, output in branch_outputs)
        logprobs: Any = [[] for _ in messages] if has_logprobs else None
        for indices, branch_output in branch_outputs:
            if len(indices) != len(branch_output.outputs):
                raise ValueError("Child LM returned a different number of outputs than routed inputs")
            if has_logprobs and branch_output.logprobs is None:
                raise ValueError("Child LMs returned inconsistent logprobs")
            for branch_index, original_index in enumerate(indices):
                outputs[original_index] = branch_output.outputs[branch_index]
                if logprobs is not None and branch_output.logprobs is not None:
                    logprobs[original_index] = branch_output.logprobs[branch_index]

        with self._routing_stats_lock:
            current = self._routing_stats
            self._routing_stats = RoutingStats(
                total=current.total + len(messages),
                cheap=current.cheap + len(cheap_indices),
                strong=current.strong + len(strong_indices),
                fallback_to_strong=current.fallback_to_strong + fallback_count,
            )
            self._last_routing_decisions = tuple(decisions)

        return LMOutput(outputs=outputs, logprobs=logprobs)

    @staticmethod
    def _validate_decisions(
        messages: list[list[dict[str, str]]], decisions: list[RoutingDecision]
    ) -> None:
        if len(decisions) != len(messages):
            raise ValueError(f"Router returned {len(decisions)} decisions for {len(messages)} requests")
        invalid = [decision.route for decision in decisions if decision.route not in {"cheap", "strong"}]
        if invalid:
            raise ValueError(f"Router returned invalid routes: {invalid}")

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
        with self._routing_stats_lock:
            self._routing_stats = RoutingStats()
            self._last_routing_decisions = ()

    def reset_cache(self, max_size: int | None = None) -> None:
        seen: set[int] = set()
        for model in (self, self.cheap_lm, self.strong_lm):
            cache = model.cache
            if cache is not None and id(cache) not in seen:
                cache.reset(max_size)
                seen.add(id(cache))

