from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass
class AdaEvolveRecord:
    id: int
    island: int
    candidate: dict[str, str]
    score: float
    summary: dict[str, Any]
    parent_id: int | None = None
    mode: str = "seed"
    tactic: str = ""
    improvement: float = 0.0
    notes: str = ""


@dataclass
class AdaEvolveIsland:
    id: int
    archive: list[int]
    best_id: int
    signal: float = 1e-6
    reward: float = 0.0
    visits: float = 0.0
    raw_visits: int = 0
    tactic: str = ""


def compact_summary(summary: dict[str, Any], max_traces: int = 8) -> dict[str, Any]:
    compact = {key: value for key, value in summary.items() if key != "traces"}
    traces = list(summary.get("traces") or [])
    if max_traces > 0:
        compact["sample_traces"] = traces[:max_traces]
    compact["trace_count"] = len(traces)
    return compact


def compact_trace_for_mutation(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "question_id": trace.get("question_id"),
        "question": trace.get("question", ""),
        "route": trace.get("route"),
        "score": trace.get("score"),
        "failure_type": trace.get("failure_type"),
        "feedback": trace.get("feedback"),
        "nano_pred": trace.get("nano_pred"),
        "mini_pred": trace.get("mini_pred"),
        "gold_answer": trace.get("gold_answer"),
        "router_output": trace.get("router_output", {}),
    }


def parse_mutation_payload(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Mutation response did not contain a JSON object.")
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Mutation response JSON must be an object.")
    return payload


def candidate_key(candidate: Mapping[str, str]) -> tuple[str, str]:
    return (
        str(candidate.get("router_prompt", "")).strip(),
        str(candidate.get("easy_router_code", "")).strip(),
    )


def normalize_candidate(
    payload: Mapping[str, Any],
    parent: dict[str, str],
    hybrid_easy_router: bool,
    compile_easy_router: Callable[[str], tuple[Any | None, str | None]],
) -> tuple[dict[str, str], str]:
    prompt = str(payload.get("router_prompt") or parent.get("router_prompt", "")).strip()
    if not prompt:
        prompt = parent.get("router_prompt", "")
    child = {"router_prompt": prompt}

    notes = str(payload.get("notes", "")).strip()
    if hybrid_easy_router:
        code = str(payload.get("easy_router_code") or parent.get("easy_router_code", "")).strip()
        if code:
            _, error = compile_easy_router(code)
            if error:
                notes = (notes + "\n" if notes else "") + f"Rejected easy_router_code: {error}"
                code = parent.get("easy_router_code", "")
        child["easy_router_code"] = code
    return child, notes


def load_seed_candidates_from_results(
    paths: Sequence[str],
    hybrid_easy_router: bool,
    max_candidates_per_result: int = 1,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"AdaEvolve seed result not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))

        possible: list[dict[str, str]] = []
        if payload.get("best_prompt"):
            possible.append(
                {
                    "router_prompt": str(payload["best_prompt"]),
                    "easy_router_code": str(payload.get("best_easy_router_code") or ""),
                }
            )
        for key in ("gepa_candidates", "adaevolve_candidates"):
            for candidate in payload.get(key, []) or []:
                if isinstance(candidate, Mapping) and candidate.get("router_prompt"):
                    possible.append(
                        {
                            "router_prompt": str(candidate.get("router_prompt", "")),
                            "easy_router_code": str(candidate.get("easy_router_code", "")),
                        }
                    )

        selected = possible if max_candidates_per_result <= 0 else possible[:max_candidates_per_result]
        for candidate in selected:
            if not hybrid_easy_router:
                candidate.pop("easy_router_code", None)
            key = candidate_key(candidate)
            if key[0] and key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def build_mutation_prompt(
    parent: dict[str, str],
    summary: dict[str, Any],
    mode: str,
    tactic: str,
    router_output_format: str,
    include_nano_answer: bool,
    hybrid_easy_router: bool,
) -> str:
    traces = list(summary.get("traces") or [])
    worst = sorted(traces, key=lambda trace: trace.get("score", 0.0))[:8]
    unnecessary = [
        trace for trace in traces if trace.get("failure_type") == "unnecessary_expensive_route"
    ][:5]
    critical = [trace for trace in traces if trace.get("failure_type") == "critical_misroute"][:5]
    compact = compact_summary(summary, max_traces=0)
    examples = {
        "worst_examples": [compact_trace_for_mutation(trace) for trace in worst],
        "critical_misroutes": [compact_trace_for_mutation(trace) for trace in critical],
        "unnecessary_mini": [compact_trace_for_mutation(trace) for trace in unnecessary],
    }

    if mode == "explore":
        instruction = (
            "Create a meaningfully different router policy that tests a new routing hypothesis. "
            "Prefer orthogonal rules or a new risk taxonomy, but keep the output contract exact."
        )
    else:
        instruction = (
            "Refine the current router policy with targeted edits that fix the observed failures "
            "while preserving successful behavior."
        )
    if tactic:
        instruction += f"\nHigh-level tactic to apply: {tactic}"

    easy_router_instruction = ""
    if hybrid_easy_router:
        easy_router_instruction = (
            "\nYou may also revise easy_router_code. It must define exactly route_easy_case("
            "question, options, category='', source='') and return 'nano', 'mini', or 'defer'. "
            "Use it only for obvious cheap cases; defer uncertain cases. No imports, loops, "
            "classes, files, network, side effects, try/with, or raise."
        )

    return (
        "You are mutating a cost-aware MMLU-Pro computer science model-router prompt.\n"
        "The router must choose exactly one saved answer stream: nano or mini.\n"
        "Primary objective: improve risk_averse_utility and held-out routing accuracy while "
        "avoiding critical_misroute cases where nano is wrong and mini is correct.\n\n"
        f"Mutation mode: {mode}\n"
        f"Router output format: {router_output_format}\n"
        f"Includes nano saved answer in input: {include_nano_answer}\n"
        f"{instruction}\n"
        f"{easy_router_instruction}\n\n"
        "Current candidate:\n"
        f"{json.dumps(parent, ensure_ascii=False, indent=2)}\n\n"
        "Current evaluation summary:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n"
        "Diagnostic examples:\n"
        f"{json.dumps(examples, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only with keys:\n"
        '- "router_prompt": the full revised system prompt string\n'
        + ('- "easy_router_code": the full revised Python function string\n' if hybrid_easy_router else "")
        + '- "notes": short explanation of what changed\n'
        "Do not include markdown or prose outside JSON."
    )


def choose_island(
    islands: list[AdaEvolveIsland],
    rng: random.Random,
    total_iterations: int,
    ucb_c: float,
) -> AdaEvolveIsland:
    unvisited = [island for island in islands if island.raw_visits == 0]
    if unvisited:
        return rng.choice(unvisited)
    log_total = math.log(max(2, total_iterations + 1))

    def ucb(island: AdaEvolveIsland) -> float:
        mean_reward = island.reward / max(island.visits, 1e-9)
        bonus = ucb_c * math.sqrt(log_total / max(island.visits, 1e-9))
        return mean_reward + bonus

    return max(islands, key=ucb)


def select_parent(
    island: AdaEvolveIsland,
    records: list[AdaEvolveRecord],
    mode: str,
    rng: random.Random,
) -> AdaEvolveRecord:
    archive = [records[idx] for idx in island.archive]
    if mode == "explore" or len(archive) == 1:
        return rng.choice(archive)
    min_score = min(record.score for record in archive)
    weights = [max(1e-6, record.score - min_score + 1e-3) for record in archive]
    return rng.choices(archive, weights=weights, k=1)[0]


def run_adaevolve(
    *,
    seed_candidates: list[dict[str, str]],
    trainset: list[Any],
    valset: list[Any],
    adapter: Any,
    reflection_lm: Any,
    summarize: Callable[[Any, list[Any], dict[str, str]], dict[str, Any]],
    compile_easy_router: Callable[[str], tuple[Any | None, str | None]],
    max_metric_calls: int,
    run_dir: str,
    seed: int,
    hybrid_easy_router: bool,
    router_output_format: str,
    include_nano_answer: bool,
    eval_set_name: str = "val",
    num_islands: int = 4,
    max_islands: int = 8,
    ucb_c: float = 0.8,
    signal_decay: float = 0.85,
    bandit_decay: float = 0.9,
    min_explore_probability: float = 0.15,
    max_explore_probability: float = 0.75,
    signal_scale: float = 6.0,
    migration_interval: int = 6,
    spawn_signal_threshold: float = 1e-4,
    meta_signal_threshold: float = 5e-5,
    iterations: int | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    log_path = run_path / "adaevolve_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    if eval_set_name == "train":
        eval_examples = trainset
    elif eval_set_name == "trainval":
        eval_examples = trainset + valset
    else:
        eval_examples = valset
        eval_set_name = "val"
    if not eval_examples:
        raise ValueError("AdaEvolve evaluation set is empty.")

    records: list[AdaEvolveRecord] = []
    seen: set[tuple[str, str]] = set()
    metric_calls = 0

    unique_seed_candidates: list[dict[str, str]] = []
    for candidate in seed_candidates:
        key = candidate_key(candidate)
        if key[0] and key not in seen:
            seen.add(key)
            unique_seed_candidates.append(candidate)
    if not unique_seed_candidates:
        raise ValueError("AdaEvolve needs at least one seed candidate.")

    for candidate in unique_seed_candidates:
        if metric_calls + len(eval_examples) > max_metric_calls and records:
            break
        summary = summarize(adapter, eval_examples, candidate)
        records.append(
            AdaEvolveRecord(
                id=len(records),
                island=0,
                candidate=candidate,
                score=summary["mean_score"],
                summary=summary,
            )
        )
        metric_calls += len(eval_examples)

    if not records:
        raise ValueError("Metric budget is too small to evaluate even one AdaEvolve seed.")

    island_count = max(1, min(num_islands, len(records)))
    islands: list[AdaEvolveIsland] = []
    for island_id in range(island_count):
        assigned = [record.id for record in records if record.id % island_count == island_id]
        if not assigned:
            assigned = [records[island_id % len(records)].id]
        best_id = max(assigned, key=lambda idx: records[idx].score)
        for idx in assigned:
            records[idx].island = island_id
        islands.append(AdaEvolveIsland(id=island_id, archive=assigned, best_id=best_id))

    best_id = max(range(len(records)), key=lambda idx: records[idx].score)
    best_score = records[best_id].score
    progress_log: list[dict[str, Any]] = []
    child_evals = 0
    total_iterations = 0

    while metric_calls + len(eval_examples) <= max_metric_calls:
        if iterations is not None and child_evals >= iterations:
            break
        total_iterations += 1
        for island_state in islands:
            island_state.reward *= bandit_decay
            island_state.visits *= bandit_decay

        island = choose_island(islands, rng, total_iterations, ucb_c)
        explore_probability = min_explore_probability + (
            max_explore_probability - min_explore_probability
        ) / (1.0 + signal_scale * math.sqrt(max(island.signal, 0.0)))
        mode = "explore" if rng.random() < explore_probability else "exploit"
        parent = select_parent(island, records, mode, rng)

        if island.signal < meta_signal_threshold and not island.tactic:
            island.tactic = (
                "Audit the previous rules for over-general hard-case escalation. Separate cases "
                "where mini can recover accuracy from cases where both models fail or nano is "
                "already correct, and make the route more discriminating."
            )

        prompt = build_mutation_prompt(
            parent.candidate,
            parent.summary,
            mode,
            island.tactic,
            router_output_format,
            include_nano_answer,
            hybrid_easy_router,
        )
        raw = reflection_lm(prompt)
        notes = ""
        try:
            payload = parse_mutation_payload(raw)
            child_candidate, notes = normalize_candidate(
                payload,
                parent.candidate,
                hybrid_easy_router,
                compile_easy_router,
            )
        except Exception as exc:
            child_candidate = dict(parent.candidate)
            notes = f"Mutation parse failed: {type(exc).__name__}: {exc}"

        key = candidate_key(child_candidate)
        if key in seen:
            child_candidate = dict(parent.candidate)
            notes = (notes + "\n" if notes else "") + "Duplicate candidate; re-evaluated parent."
        else:
            seen.add(key)

        summary = summarize(adapter, eval_examples, child_candidate)
        child_score = summary["mean_score"]
        metric_calls += len(eval_examples)
        child_evals += 1

        local_best_before = records[island.best_id].score
        improvement = max(0.0, child_score - local_best_before)
        normalized = improvement / max(abs(local_best_before), 1e-6)
        island.signal = signal_decay * island.signal + (1.0 - signal_decay) * (
            normalized * normalized
        )
        global_reward = improvement / max(abs(best_score), 1e-6)
        island.reward += global_reward
        island.visits += 1.0
        island.raw_visits += 1

        record = AdaEvolveRecord(
            id=len(records),
            island=island.id,
            candidate=child_candidate,
            score=child_score,
            summary=summary,
            parent_id=parent.id,
            mode=mode,
            tactic=island.tactic,
            improvement=improvement,
            notes=notes,
        )
        records.append(record)
        island.archive.append(record.id)
        if child_score >= records[island.best_id].score:
            island.best_id = record.id
        if child_score >= best_score:
            best_id = record.id
            best_score = child_score
            island.tactic = ""

        if migration_interval > 0 and child_evals % migration_interval == 0 and len(islands) > 1:
            best_by_island = [source.best_id for source in islands]
            for idx, migrated_id in enumerate(best_by_island):
                target = islands[(idx + 1) % len(islands)]
                if migrated_id not in target.archive:
                    target.archive.append(migrated_id)

        if (
            len(islands) < max_islands
            and all(source.signal < spawn_signal_threshold for source in islands)
        ):
            spawned_seed = rng.choice(records).id

            islands.append(
                AdaEvolveIsland(
                    id=len(islands),
                    archive=[spawned_seed],
                    best_id=spawned_seed,
                    tactic="Explore a qualitatively different routing policy from the current archive.",
                )
            )

        progress_entry = {
            "iteration": child_evals,
            "metric_calls": metric_calls,
            "island": island.id,
            "mode": mode,
            "explore_probability": explore_probability,
            "parent_id": parent.id,
            "child_id": record.id,
            "child_score": child_score,
            "best_score": best_score,
            "improvement": improvement,
            "signal": island.signal,
            "notes": notes,
        }
        progress_log.append(progress_entry)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(progress_entry, ensure_ascii=False) + "\n")

    best_record = records[best_id]
    return {
        "algorithm": "adaevolve",
        "eval_set": eval_set_name,
        "metric_calls": metric_calls,
        "iterations": child_evals,
        "best_id": best_id,
        "best_score": best_record.score,
        "best_candidate": best_record.candidate,
        "records": [
            {
                "id": record.id,
                "island": record.island,
                "parent_id": record.parent_id,
                "mode": record.mode,
                "score": record.score,
                "improvement": record.improvement,
                "tactic": record.tactic,
                "notes": record.notes,
                "candidate": record.candidate,
                "summary": compact_summary(record.summary),
            }
            for record in records
        ],
        "islands": [
            {
                "id": island.id,
                "archive": island.archive,
                "best_id": island.best_id,
                "signal": island.signal,
                "reward": island.reward,
                "visits": island.visits,
                "raw_visits": island.raw_visits,
                "tactic": island.tactic,
            }
            for island in islands
        ],
        "progress_log": progress_log,
    }
