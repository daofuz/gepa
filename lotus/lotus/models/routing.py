import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol

Route = Literal["cheap", "strong"]


@dataclass(frozen=True)
class RoutingDecision:
    """A router's model choice for one LM request."""

    route: Route
    strong_probability: float

    def __post_init__(self) -> None:
        if self.route not in {"cheap", "strong"}:
            raise ValueError(f"Unsupported route: {self.route!r}")
        if not math.isfinite(self.strong_probability) or not 0.0 <= self.strong_probability <= 1.0:
            raise ValueError("strong_probability must be a finite value between 0 and 1")


class Router(Protocol):
    """Interface used by :class:`RoutedLM` to route a batch of messages."""

    def route(self, messages: list[list[dict[str, Any]]]) -> list[RoutingDecision]: ...


class CascadeRouter(Protocol):
    """Route after observing the cheap model's output."""

    def route_after_cheap(
        self,
        messages: list[list[dict[str, Any]]],
        cheap_outputs: list[str],
    ) -> list[RoutingDecision]: ...

