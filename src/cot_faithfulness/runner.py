"""
Main evaluation loop: base runs and hinted runs with per-question checkpointing.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from cot_faithfulness.answer_extraction import extract_answer, extract_answer_for_visual_pattern
from cot_faithfulness.clients import OpenRouterClient, get_client
from cot_faithfulness.config import HintType, MODEL_REGISTRY, ModelConfig, Settings
from cot_faithfulness.hints import HintInjector
from cot_faithfulness.schemas import HintedQuestion, InferenceResult, Question


def get_completed_ids(output_path: Path) -> set[str]:
    """Load question IDs already completed from a JSONL checkpoint file."""
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        result = InferenceResult.model_validate_json(line)
                        completed.add(result.question_id)
                    except Exception as exc:
                        warnings.warn(
                            f"Corrupt line {line_num} in {output_path}: {exc}"
                        )
    return completed


def append_result(result: InferenceResult, output_path: Path) -> None:
    """Append a single result to a JSONL file (atomic per-question checkpoint)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(result.model_dump_json() + "\n")


def load_results(output_path: Path) -> list[InferenceResult]:
    """Load all results from a JSONL file."""
    results = []
    if output_path.exists():
        with open(output_path) as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        results.append(InferenceResult.model_validate_json(line))
                    except Exception as exc:
                        warnings.warn(
                            f"Corrupt line {line_num} in {output_path}: {exc}"
                        )
    return results


def log_error(
    question_id: str,
    error_msg: str,
    error_path: Path,
    context: dict | None = None,
) -> None:
    """Append an error record to a JSONL sidecar file."""
    error_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_id": question_id,
        "error": error_msg,
    }
    if context:
        record.update(context)
    with open(error_path, "a") as f:
        f.write(json.dumps(record) + "\n")


class _ConsecutiveFailureTracker:
    """Abort if too many consecutive failures — likely systemic issue."""

    def __init__(self, threshold: int = 10) -> None:
        self._threshold = threshold
        self._count = 0

    def record_success(self) -> None:
        self._count = 0

    def record_failure(self) -> None:
        self._count += 1
        if self._count >= self._threshold:
            raise RuntimeError(
                f"Aborting: {self._threshold} consecutive failures"
            )


class BudgetExceededError(Exception):
    """Raised when cumulative API cost exceeds the budget."""


def _compute_cost(model_config: ModelConfig, input_tokens: int, output_tokens: int) -> float:
    """Compute cost in USD for a single API call."""
    return (
        model_config.cost_per_million_input * input_tokens / 1_000_000
        + model_config.cost_per_million_output * output_tokens / 1_000_000
    )


def precompute_targets(
    questions: list[Question],
    injector: HintInjector,
) -> dict[str, tuple[str, int]]:
    """
    Pre-select target answers for all questions.
    Same target is used across all 6 hint types for fair comparison.
    """
    return {
        q.id: injector.select_target(q)
        for q in questions
    }


def run_base(
    client: OpenRouterClient,
    model_name: str,
    questions: list[Question],
    output_path: Path,
) -> list[InferenceResult]:
    """Run baseline (no-hint) inference on all questions with checkpointing."""
    completed_ids = get_completed_ids(output_path)
    remaining = [q for q in questions if q.id not in completed_ids]

    if not remaining:
        print(f"  Base run already complete ({len(completed_ids)} questions)")
        return load_results(output_path)

    print(f"  Running base: {len(remaining)} remaining ({len(completed_ids)} cached)")
    error_path = output_path.with_suffix(".errors.jsonl")
    tracker = _ConsecutiveFailureTracker()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f, Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task(f"Base [{model_name}]", total=len(remaining))

        for q in remaining:
            try:
                out = client.generate(q.formatted)
                extracted = extract_answer(
                    out["full_response"], out["thinking_text"], out["answer_text"]
                )
                result = InferenceResult(
                    question_id=q.id,
                    model_name=model_name,
                    hint_type=None,
                    full_response=out["full_response"],
                    thinking_text=out["thinking_text"],
                    answer_text=out["answer_text"],
                    extracted_answer=extracted,
                    target_label=None,
                    correct_label=q.correct_label,
                    latency_seconds=out["latency_seconds"],
                    input_tokens=out["input_tokens"],
                    output_tokens=out["output_tokens"],
                    reasoning_tokens=out["reasoning_tokens"],
                )
                f.write(result.model_dump_json() + "\n")
                f.flush()
                tracker.record_success()
            except Exception as e:
                log_error(q.id, str(e), error_path, {"model": model_name, "run": "base"})
                tracker.record_failure()

            progress.advance(task)

    return load_results(output_path)


def run_hinted(
    client: OpenRouterClient,
    model_name: str,
    questions: list[Question],
    hint_type: HintType,
    injector: HintInjector,
    output_path: Path,
    precomputed_targets: dict[str, tuple[str, int]],
) -> list[InferenceResult]:
    """Run hinted inference for a specific hint type with checkpointing."""
    completed_ids = get_completed_ids(output_path)
    remaining = [q for q in questions if q.id not in completed_ids]

    if not remaining:
        print(f"  {hint_type.value} already complete ({len(completed_ids)} questions)")
        return load_results(output_path)

    print(
        f"  Running {hint_type.value}: {len(remaining)} remaining "
        f"({len(completed_ids)} cached)"
    )
    error_path = output_path.with_suffix(".errors.jsonl")
    tracker = _ConsecutiveFailureTracker()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f, Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task(
            f"{hint_type.value} [{model_name}]", total=len(remaining)
        )

        for q in remaining:
            try:
                target_label, target_index = precomputed_targets[q.id]
                hq = injector.inject(q, hint_type, target_label, target_index)

                out = client.generate(hq.formatted)

                # Special handling for visual pattern answer extraction
                if (
                    hint_type == HintType.VISUAL_PATTERN
                    and hq.reorder_mapping
                ):
                    extracted = extract_answer_for_visual_pattern(
                        out["full_response"],
                        out["thinking_text"],
                        out["answer_text"],
                        hq.reorder_mapping,
                    )
                else:
                    extracted = extract_answer(
                        out["full_response"],
                        out["thinking_text"],
                        out["answer_text"],
                    )

                # For visual pattern, target_label in original space
                original_target = target_label

                result = InferenceResult(
                    question_id=q.id,
                    model_name=model_name,
                    hint_type=hint_type.value,
                    full_response=out["full_response"],
                    thinking_text=out["thinking_text"],
                    answer_text=out["answer_text"],
                    extracted_answer=extracted,
                    target_label=original_target,
                    correct_label=q.correct_label,
                    latency_seconds=out["latency_seconds"],
                    input_tokens=out["input_tokens"],
                    output_tokens=out["output_tokens"],
                    reasoning_tokens=out["reasoning_tokens"],
                )
                f.write(result.model_dump_json() + "\n")
                f.flush()
                tracker.record_success()
            except Exception as e:
                log_error(
                    q.id, str(e), error_path,
                    {"model": model_name, "run": hint_type.value},
                )
                tracker.record_failure()

            progress.advance(task)

    return load_results(output_path)


def run_single_model(
    model_name: str,
    questions: list[Question],
    hint_types: Optional[list[HintType]] = None,
    settings: Optional[Settings] = None,
    budget_usd: float | None = None,
) -> None:
    """
    Full evaluation for one model: base + specified hint types.
    Defaults to all 6 hint types if none specified.
    """
    settings = settings or Settings()
    paths = settings.paths

    if hint_types is None:
        hint_types = list(HintType)

    model_config = MODEL_REGISTRY[model_name]

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Questions: {len(questions)}")
    print(f"Hint types: {[h.value for h in hint_types]}")
    if budget_usd is not None:
        print(f"Budget: ${budget_usd:.2f}")
    print(f"{'='*60}\n")

    client = get_client(model_name, settings)
    injector = HintInjector(seed=settings.experiment_seed)

    # Pre-select targets for consistency across hint types
    targets = precompute_targets(questions, injector)
    cumulative_cost = 0.0

    # Base run
    base_path = paths.results_base / f"{model_name}.jsonl"
    run_base(client, model_name, questions, base_path)
    base_results = load_results(base_path)
    for r in base_results:
        cumulative_cost += _compute_cost(model_config, r.input_tokens, r.output_tokens)
    print(f"  Cost so far: ${cumulative_cost:.4f}")

    # Hinted runs
    for ht in hint_types:
        if budget_usd is not None and cumulative_cost >= budget_usd:
            raise BudgetExceededError(
                f"Budget ${budget_usd:.2f} exceeded (spent ${cumulative_cost:.4f}) — "
                f"stopping before {ht.value}"
            )
        hinted_path = paths.results_hinted / model_name / f"{ht.value}.jsonl"
        run_hinted(client, model_name, questions, ht, injector, hinted_path, targets)
        hinted_results = load_results(hinted_path)
        for r in hinted_results:
            cumulative_cost += _compute_cost(model_config, r.input_tokens, r.output_tokens)
        print(f"  Cost so far: ${cumulative_cost:.4f} (after {ht.value})")

    if budget_usd is not None and cumulative_cost >= budget_usd:
        raise BudgetExceededError(
            f"Budget ${budget_usd:.2f} exceeded (spent ${cumulative_cost:.4f})"
        )

    print(f"\nCompleted {model_name} — total cost: ${cumulative_cost:.4f}")
