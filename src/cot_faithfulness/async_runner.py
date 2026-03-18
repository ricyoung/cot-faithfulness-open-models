"""
Concurrent evaluation loop using asyncio for parallel API calls.
Drop-in replacement for runner.py with ~5-10x throughput improvement.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, InternalServerError

from cot_faithfulness.answer_extraction import extract_answer, extract_answer_for_visual_pattern
from cot_faithfulness.clients import split_thinking
from cot_faithfulness.config import HintType, MODEL_REGISTRY, ModelConfig, Settings
from cot_faithfulness.hints import HintInjector
from cot_faithfulness.runner import (
    BudgetExceededError,
    _compute_cost,
    get_completed_ids,
    load_results,
    log_error,
    precompute_targets,
)
from cot_faithfulness.schemas import InferenceResult, Question

_TRANSIENT_EXCEPTIONS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)

MAX_RETRIES = 5


async def _generate_one(
    aclient: AsyncOpenAI,
    model_config: ModelConfig,
    settings: Settings,
    user_message: str,
) -> dict:
    """Single async API call with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            start = time.monotonic()
            kwargs: dict = {
                "model": model_config.api_model_id,
                "messages": [
                    {"role": "system", "content": settings.generation.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": settings.generation.temperature,
                "seed": settings.generation.seed,
                "max_tokens": model_config.max_completion_tokens,
            }
            if model_config.supports_reasoning:
                kwargs["extra_body"] = {"reasoning": {"effort": "high"}}

            response = await aclient.chat.completions.create(**kwargs)
            latency = time.monotonic() - start

            message = response.choices[0].message
            full_text = message.content or ""

            reasoning_field = (
                getattr(message, "reasoning", None)
                or getattr(message, "reasoning_content", None)
                or ""
            )
            if reasoning_field:
                thinking_text = reasoning_field
                answer_text = full_text
            else:
                thinking_text, answer_text = split_thinking(full_text)

            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            reasoning_tokens = 0
            if usage:
                ctd = getattr(usage, "completion_tokens_details", None)
                if ctd:
                    reasoning_tokens = getattr(ctd, "reasoning_tokens", 0) or 0
                if not reasoning_tokens:
                    reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0

            return {
                "full_response": full_text,
                "thinking_text": thinking_text,
                "answer_text": answer_text,
                "latency_seconds": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            }
        except _TRANSIENT_EXCEPTIONS:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = min(2 ** attempt, 60)
            await asyncio.sleep(wait)
    raise RuntimeError("Unreachable")


async def _process_base_question(
    sem: asyncio.Semaphore,
    aclient: AsyncOpenAI,
    model_config: ModelConfig,
    model_name: str,
    settings: Settings,
    q: Question,
    error_path: Path,
) -> Optional[InferenceResult]:
    """Process one base question with semaphore-controlled concurrency."""
    async with sem:
        try:
            out = await _generate_one(aclient, model_config, settings, q.formatted)
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
            return result
        except Exception as exc:
            log_error(q.id, str(exc), error_path, {"model": model_name, "run": "base"})
            return None


async def _process_hinted_question(
    sem: asyncio.Semaphore,
    aclient: AsyncOpenAI,
    model_config: ModelConfig,
    model_name: str,
    settings: Settings,
    q: Question,
    hint_type: HintType,
    injector: HintInjector,
    precomputed_targets: dict[str, tuple[str, int]],
    error_path: Path,
) -> Optional[InferenceResult]:
    """Process one hinted question with semaphore-controlled concurrency."""
    async with sem:
        try:
            target_label, target_index = precomputed_targets[q.id]
            hq = injector.inject(q, hint_type, target_label, target_index)
            out = await _generate_one(aclient, model_config, settings, hq.formatted)

            if hint_type == HintType.VISUAL_PATTERN and hq.reorder_mapping:
                extracted = extract_answer_for_visual_pattern(
                    out["full_response"], out["thinking_text"],
                    out["answer_text"], hq.reorder_mapping,
                )
            else:
                extracted = extract_answer(
                    out["full_response"], out["thinking_text"], out["answer_text"]
                )

            result = InferenceResult(
                question_id=q.id,
                model_name=model_name,
                hint_type=hint_type.value,
                full_response=out["full_response"],
                thinking_text=out["thinking_text"],
                answer_text=out["answer_text"],
                extracted_answer=extracted,
                target_label=target_label,
                correct_label=q.correct_label,
                latency_seconds=out["latency_seconds"],
                input_tokens=out["input_tokens"],
                output_tokens=out["output_tokens"],
                reasoning_tokens=out["reasoning_tokens"],
            )
            return result
        except Exception as e:
            log_error(
                q.id, str(e), error_path,
                {"model": model_name, "run": hint_type.value},
            )
            return None


def _append_results(results: list[InferenceResult], output_path: Path) -> None:
    """Append batch of results to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")


async def run_base_async(
    aclient: AsyncOpenAI,
    model_config: ModelConfig,
    model_name: str,
    questions: list[Question],
    output_path: Path,
    settings: Settings,
    concurrency: int = 5,
) -> list[InferenceResult]:
    """Run baseline inference with concurrent API calls."""
    completed_ids = get_completed_ids(output_path)
    remaining = [q for q in questions if q.id not in completed_ids]

    if not remaining:
        print(f"  Base run already complete ({len(completed_ids)} questions)")
        return load_results(output_path)

    print(f"  Running base: {len(remaining)} remaining ({len(completed_ids)} cached) [concurrency={concurrency}]")
    error_path = output_path.with_suffix(".errors.jsonl")
    sem = asyncio.Semaphore(concurrency)

    # Process in batches to enable incremental checkpointing
    batch_size = concurrency * 2
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        tasks = [
            _process_base_question(
                sem, aclient, model_config, model_name, settings,
                q, error_path,
            )
            for q in batch
        ]
        results = await asyncio.gather(*tasks)
        good = [r for r in results if r is not None]
        _append_results(good, output_path)
        done = len(completed_ids) + i + len(batch)
        print(f"    Base: {done}/{len(questions)} ({100*done//len(questions)}%)")

    return load_results(output_path)


async def run_hinted_async(
    aclient: AsyncOpenAI,
    model_config: ModelConfig,
    model_name: str,
    questions: list[Question],
    hint_type: HintType,
    injector: HintInjector,
    output_path: Path,
    precomputed_targets: dict[str, tuple[str, int]],
    settings: Settings,
    concurrency: int = 5,
) -> list[InferenceResult]:
    """Run hinted inference with concurrent API calls."""
    completed_ids = get_completed_ids(output_path)
    remaining = [q for q in questions if q.id not in completed_ids]

    if not remaining:
        print(f"  {hint_type.value} already complete ({len(completed_ids)} questions)")
        return load_results(output_path)

    print(f"  Running {hint_type.value}: {len(remaining)} remaining ({len(completed_ids)} cached) [concurrency={concurrency}]")
    error_path = output_path.with_suffix(".errors.jsonl")
    sem = asyncio.Semaphore(concurrency)

    batch_size = concurrency * 2
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        tasks = [
            _process_hinted_question(
                sem, aclient, model_config, model_name, settings,
                q, hint_type, injector, precomputed_targets, error_path,
            )
            for q in batch
        ]
        results = await asyncio.gather(*tasks)
        good = [r for r in results if r is not None]
        _append_results(good, output_path)
        done = len(completed_ids) + i + len(batch)
        print(f"    {hint_type.value}: {done}/{len(questions)} ({100*done//len(questions)}%)")

    return load_results(output_path)


async def run_single_model_async(
    model_name: str,
    questions: list[Question],
    hint_types: Optional[list[HintType]] = None,
    settings: Optional[Settings] = None,
    budget_usd: float | None = None,
    concurrency: int = 5,
) -> None:
    """Full evaluation for one model with concurrent API calls."""
    settings = settings or Settings()
    paths = settings.paths

    if hint_types is None:
        hint_types = list(HintType)

    model_config = MODEL_REGISTRY[model_name]

    print(f"\n{'='*60}")
    print(f"Model: {model_name} (concurrency={concurrency})")
    print(f"Questions: {len(questions)}")
    print(f"Hint types: {[h.value for h in hint_types]}")
    if budget_usd is not None:
        print(f"Budget: ${budget_usd:.2f}")
    print(f"{'='*60}\n")

    aclient = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.api.base_url,
        timeout=settings.api.request_timeout,
    )

    injector = HintInjector(seed=settings.experiment_seed)
    targets = precompute_targets(questions, injector)
    cumulative_cost = 0.0

    # Base run
    base_path = paths.results_base / f"{model_name}.jsonl"
    await run_base_async(
        aclient, model_config, model_name, questions, base_path, settings, concurrency
    )
    base_results = load_results(base_path)
    for r in base_results:
        cumulative_cost += _compute_cost(model_config, r.input_tokens, r.output_tokens)
    print(f"  Cost so far: ${cumulative_cost:.4f}")

    # Hinted runs
    for ht in hint_types:
        if budget_usd is not None and cumulative_cost >= budget_usd:
            raise BudgetExceededError(
                f"Budget ${budget_usd:.2f} exceeded (spent ${cumulative_cost:.4f}) "
                f"— stopping before {ht.value}"
            )
        hinted_path = paths.results_hinted / model_name / f"{ht.value}.jsonl"
        await run_hinted_async(
            aclient, model_config, model_name, questions, ht, injector,
            hinted_path, targets, settings, concurrency,
        )
        hinted_results = load_results(hinted_path)
        for r in hinted_results:
            cumulative_cost += _compute_cost(model_config, r.input_tokens, r.output_tokens)
        print(f"  Cost so far: ${cumulative_cost:.4f} (after {ht.value})")

    print(f"\nCompleted {model_name} — total cost: ${cumulative_cost:.4f}")
