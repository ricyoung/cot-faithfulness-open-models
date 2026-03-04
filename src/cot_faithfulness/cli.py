"""
Click CLI for the CoT faithfulness pipeline.

Commands:
  prep   — Download and sample MMLU + GPQA questions
  run    — Run base or hinted inference for a model
  status — Show progress across all models
"""
from __future__ import annotations

import click

from cot_faithfulness.config import HintType, MODEL_REGISTRY, Settings
from cot_faithfulness.data_prep import load_questions, prepare_all
from cot_faithfulness.runner import BudgetExceededError, get_completed_ids, load_results, run_single_model


@click.group()
def main() -> None:
    """CoT Faithfulness evaluation pipeline."""


@main.command()
def prep() -> None:
    """Download MMLU/GPQA and sample questions."""
    settings = Settings()
    combined_path = settings.paths.combined_dataset

    if combined_path.exists():
        existing = load_questions(combined_path)
        click.echo(f"Dataset already exists: {len(existing)} questions at {combined_path}")
        if not click.confirm("Re-download and resample?"):
            return

    questions = prepare_all(settings)
    click.echo(f"Prepared {len(questions)} questions")


@main.command()
@click.option(
    "--model", "-m",
    required=True,
    type=click.Choice(sorted(MODEL_REGISTRY.keys()), case_sensitive=False),
    help="Model name from the registry.",
)
@click.option(
    "--hint-type", "-h",
    multiple=True,
    type=click.Choice([h.value for h in HintType] + ["base", "all"], case_sensitive=False),
    default=("all",),
    help="Hint types to run. Use 'base' for no-hint baseline, 'all' for everything.",
)
@click.option("--dry-run", is_flag=True, default=False, help="Show estimates without making API calls.")
@click.option("--budget", type=float, default=None, help="Maximum spend in USD (default: no limit).")
def run(model: str, hint_type: tuple[str, ...], dry_run: bool, budget: float | None) -> None:
    """Run inference for a model with specified hint types."""
    settings = Settings()
    combined_path = settings.paths.combined_dataset

    if not combined_path.exists():
        click.echo("No dataset found. Run 'cot-faithfulness prep' first.")
        raise SystemExit(1)

    questions = load_questions(combined_path)
    click.echo(f"Loaded {len(questions)} questions from {combined_path}")

    # Parse hint types
    hint_types_set = set(hint_type)

    if "all" in hint_types_set:
        selected_hints: list[HintType] | None = None  # None = all
    elif "base" in hint_types_set and len(hint_types_set) == 1:
        selected_hints = []  # Empty list = base only
    else:
        selected_hints = [
            HintType(h) for h in hint_types_set if h != "base"
        ]

    # If "base" is explicitly listed with other hints, run_single_model
    # always runs base anyway, so just pass the hint types
    if dry_run:
        settings_for_dry = settings
        paths = settings_for_dry.paths
        model_config = MODEL_REGISTRY[model]

        # Determine effective hint types
        effective_hints = list(HintType) if selected_hints is None else selected_hints

        # Count remaining per run type
        base_path = paths.results_base / f"{model}.jsonl"
        base_done = len(get_completed_ids(base_path))
        base_remaining = len(questions) - base_done

        click.echo(f"\n--- Dry Run Estimate for {model} ---")
        click.echo(f"Total questions: {len(questions)}")
        click.echo(f"Base: {base_remaining} remaining ({base_done} cached)")

        total_remaining = base_remaining
        for ht in effective_hints:
            hp = paths.results_hinted / model / f"{ht.value}.jsonl"
            done = len(get_completed_ids(hp))
            remaining_ht = len(questions) - done
            total_remaining += remaining_ht
            click.echo(f"  {ht.value}: {remaining_ht} remaining ({done} cached)")

        # Estimate cost (rough: assume ~1000 input + ~2000 output tokens per call)
        avg_input = 1000
        avg_output = 2000
        est_cost = total_remaining * (
            model_config.cost_per_million_input * avg_input / 1_000_000
            + model_config.cost_per_million_output * avg_output / 1_000_000
        )
        click.echo(f"\nEstimated API calls: {total_remaining}")
        click.echo(f"Estimated cost: ${est_cost:.2f} (assuming ~{avg_input} in + ~{avg_output} out tokens/call)")
        if budget is not None:
            click.echo(f"Budget: ${budget:.2f}")
        click.echo("---")
        return

    try:
        run_single_model(
            model_name=model,
            questions=questions,
            hint_types=selected_hints if selected_hints is not None else None,
            settings=settings,
            budget_usd=budget,
        )
    except BudgetExceededError as e:
        click.echo(f"\nBudget exceeded: {e}", err=True)
        raise SystemExit(1)


@main.command()
def status() -> None:
    """Show progress for all models."""
    settings = Settings()
    paths = settings.paths

    click.echo(f"\n{'Model':<30} {'Base':>6} {'Syco':>6} {'Cons':>6} {'Vis':>6} {'Meta':>6} {'Grad':>6} {'Uneth':>6}")
    click.echo("-" * 84)

    for model_name in sorted(MODEL_REGISTRY.keys()):
        base_path = paths.results_base / f"{model_name}.jsonl"
        base_count = len(load_results(base_path)) if base_path.exists() else 0

        hint_counts = []
        for ht in HintType:
            hp = paths.results_hinted / model_name / f"{ht.value}.jsonl"
            hint_counts.append(len(load_results(hp)) if hp.exists() else 0)

        click.echo(
            f"{model_name:<30} {base_count:>6}"
            + "".join(f" {c:>6}" for c in hint_counts)
        )

    click.echo()


if __name__ == "__main__":
    main()
