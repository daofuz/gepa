from pathlib import Path
from typing import Any, Literal

from lotus.models.routing import RoutingDecision


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict) and content.get("type") in {"text", "input_text"}:
        return str(content.get("text", ""))
    return ""


def _has_non_text_content(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    continue
                if not isinstance(item, dict) or item.get("type") not in {"text", "input_text"}:
                    return True
        elif content is not None and not isinstance(content, str):
            if not isinstance(content, dict) or content.get("type") not in {"text", "input_text"}:
                return True
    return False


def messages_to_router_text(messages: list[dict[str, Any]]) -> str:
    """Use the final user turn, which contains the current Lotus row and task."""

    for message in reversed(messages):
        if message.get("role") == "user":
            text = _content_to_text(message.get("content"))
            if text:
                return text
    return "\n".join(
        text for message in messages if (text := _content_to_text(message.get("content")))
    )


def _format_router_text(text: str, router_text_format: str) -> str:
    """Adapt a rendered Lotus prompt to the text distribution used for training."""

    if router_text_format == "raw_messages_v1":
        return text
    if router_text_format == "sembench_movie_review_v1":
        context_prefix = "Context:\n"
        claim_separator = "\n\nClaim:"
        if not text.startswith(context_prefix) or claim_separator not in text:
            raise ValueError(
                "sembench_movie_review_v1 expects a Lotus filter prompt with "
                "'Context:' and 'Claim:' sections"
            )
        review, _claim = text[len(context_prefix) :].rsplit(claim_separator, 1)
        return (
            "Semantic operator: classify whether a movie review is positive or negative.\n"
            f"Review: {review}"
        )
    raise ValueError(f"Unsupported router_text_format: {router_text_format!r}")


def _resolve_decision_threshold(
    checkpoint: dict[str, Any], checkpoint_args: dict[str, Any], override: float | None
) -> float:
    threshold = (
        override
        if override is not None
        else checkpoint.get(
            "selected_decision_threshold",
            checkpoint_args.get("selected_decision_threshold", checkpoint_args.get("decision_threshold", 0.5)),
        )
    )
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1")
    return threshold


def _resolve_strong_class_index(checkpoint: dict[str, Any], override: int | None) -> int:
    if override is not None:
        if override not in {0, 1}:
            raise ValueError("strong_class_index must be 0 or 1")
        return override

    routes = checkpoint.get("routes")
    if isinstance(routes, (list, tuple)) and len(routes) == 2:
        normalized_routes = [str(route).strip().lower() for route in routes]
        strong_aliases = {"strong", "mini", "large", "quality"}
        cheap_aliases = {"cheap", "nano", "small", "fast"}
        strong_matches = [index for index, route in enumerate(normalized_routes) if route in strong_aliases]
        cheap_matches = [index for index, route in enumerate(normalized_routes) if route in cheap_aliases]
        if len(strong_matches) == 1 and len(cheap_matches) == 1 and strong_matches[0] != cheap_matches[0]:
            return strong_matches[0]
        raise ValueError(
            f"Cannot infer cheap/strong classes from checkpoint routes {routes!r}; "
            "pass strong_class_index explicitly"
        )

    # Backward compatibility with the original training format: class 0 was
    # nano/cheap and class 1 was mini/strong.
    return 1


class CPUSoftPromptRouter:
    """Run a DistilBERT soft-prompt binary router on CPU.

    The checkpoint must use the format produced by ``train_softprompt_router.py``:
    ``model_state_dict`` plus an ``args`` mapping. Class 0 routes to ``cheap`` and
    class 1 routes to ``strong``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        decision_threshold: float | None = None,
        batch_size: int = 32,
        max_length: int | None = None,
        hf_model: str | None = None,
        local_files_only: bool = True,
        cpu_threads: int | None = None,
        strong_class_index: int | None = None,
        non_text_policy: Literal["strong", "route_text", "raise"] = "strong",
        router_text_format: str | None = None,
    ) -> None:
        try:
            import torch
            from torch import nn
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "CPUSoftPromptRouter requires the optional dependencies. "
                "Install them with: pip install 'lotus-ai[softprompt_router]'"
            ) from exc

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Soft-prompt checkpoint not found: {checkpoint_path}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if non_text_policy not in {"strong", "route_text", "raise"}:
            raise ValueError("non_text_policy must be 'strong', 'route_text', or 'raise'")
        if cpu_threads is not None:
            if cpu_threads <= 0:
                raise ValueError("cpu_threads must be positive")
            torch.set_num_threads(cpu_threads)

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError as exc:  # pragma: no cover - only reached on unsupported PyTorch versions
            raise RuntimeError("CPUSoftPromptRouter requires a PyTorch version with safe weights_only loading") from exc
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint must contain model_state_dict")

        raw_args = checkpoint.get("args", {})
        checkpoint_args = raw_args if isinstance(raw_args, dict) else vars(raw_args)
        model_name = hf_model or checkpoint_args.get("hf_model", "distilbert-base-uncased")
        soft_prompt_tokens = int(checkpoint_args.get("soft_prompt_tokens", 16))
        self.decision_threshold = _resolve_decision_threshold(checkpoint, checkpoint_args, decision_threshold)
        self.strong_class_index = _resolve_strong_class_index(checkpoint, strong_class_index)
        self.non_text_policy = non_text_policy
        self.router_text_format = str(
            router_text_format
            or checkpoint.get(
                "router_text_format",
                checkpoint_args.get("router_text_format", "raw_messages_v1"),
            )
        )
        if self.router_text_format not in {
            "raw_messages_v1",
            "sembench_movie_review_v1",
        }:
            raise ValueError(
                f"Unsupported router_text_format: {self.router_text_format!r}"
            )
        self.max_length = int(max_length if max_length is not None else checkpoint_args.get("max_length", 384))
        self.batch_size = batch_size

        class _SoftPromptModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
                hidden_size = int(self.encoder.config.hidden_size)
                self.soft_prompt = nn.Parameter(torch.empty(soft_prompt_tokens, hidden_size))
                self.classifier = nn.Linear(hidden_size, 2)
                self.soft_prompt_tokens = soft_prompt_tokens

            def forward(self, input_ids, attention_mask):
                token_embeddings = self.encoder.get_input_embeddings()(input_ids)
                prompt = self.soft_prompt.unsqueeze(0).expand(token_embeddings.shape[0], -1, -1)
                inputs_embeds = torch.cat([prompt, token_embeddings], dim=1)
                prompt_mask = torch.ones(
                    token_embeddings.shape[0],
                    self.soft_prompt_tokens,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
                outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask)
                cls_state = outputs.last_hidden_state[:, self.soft_prompt_tokens, :]
                return self.classifier(cls_state)

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token or self._tokenizer.unk_token
        self._model = _SoftPromptModel()
        self._model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self._model.to("cpu")
        self._model.eval()

    def route(self, messages: list[list[dict[str, Any]]]) -> list[RoutingDecision]:
        texts = [messages_to_router_text(message_list) for message_list in messages]
        routed_indices: list[int] = []
        routed_texts: list[str] = []
        decisions: list[RoutingDecision | None] = [None] * len(messages)
        for index, (message_list, text) in enumerate(zip(messages, texts)):
            has_non_text = _has_non_text_content(message_list)
            if has_non_text and self.non_text_policy == "raise":
                raise ValueError("Soft-prompt router received non-text content")
            if not text or (has_non_text and self.non_text_policy == "strong"):
                decisions[index] = RoutingDecision(route="strong", strong_probability=1.0)
            else:
                routed_indices.append(index)
                routed_texts.append(_format_router_text(text, self.router_text_format))

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
                strong_probabilities = self._torch.softmax(logits, dim=-1)[:, self.strong_class_index].tolist()
                for original_index, probability in zip(
                    routed_indices[start : start + self.batch_size], strong_probabilities
                ):
                    decisions[original_index] = RoutingDecision(
                        route="strong" if probability >= self.decision_threshold else "cheap",
                        strong_probability=float(probability),
                    )

        if any(decision is None for decision in decisions):  # pragma: no cover - defensive invariant
            raise RuntimeError("Router failed to produce a decision for every request")
        return [decision for decision in decisions if decision is not None]

