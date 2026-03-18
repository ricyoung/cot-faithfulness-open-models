"""
Click CLI for the CoT faithfulness pipeline.

Commands:
  prep     — Download and sample MMLU + GPQA questions
  run      — Run base or hinted inference for a model
  run-all  — Run all 14 models sequentially
  status   — Show progress across all models (live dashboard)
"""
from __future__ import annotations

import time

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

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


TOTAL_QUESTIONS = 498  # 300 MMLU + 198 GPQA Diamond


def _cell_style(count: int, total: int) -> tuple[str, str]:
    """Return (display_text, style) for a progress cell."""
    if count == 0:
        return "0", "dim"
    if count >= total:
        return f"{count}", "bold green"
    pct = count * 100 // total
    return f"{count} ({pct}%)", "yellow"


def _build_status_table(settings: Settings) -> Table:
    """Build a Rich table showing progress across all models."""
    paths = settings.paths

    table = Table(
        title="CoT Faithfulness — Experiment Progress",
        caption=f"Target: {TOTAL_QUESTIONS} questions per cell  |  "
                f"Total: 48,804 inference runs  |  14 models x 7 conditions",
        show_lines=True,
    )

    table.add_column("Model", style="bold cyan", min_width=22)
    table.add_column("Tier", justify="center", width=4)
    table.add_column("Base", justify="right", width=12)
    table.add_column("Syco", justify="right", width=12)
    table.add_column("Cons", justify="right", width=12)
    table.add_column("Visual", justify="right", width=12)
    table.add_column("Meta", justify="right", width=12)
    table.add_column("Grader", justify="right", width=12)
    table.add_column("Uneth", justify="right", width=12)
    table.add_column("Total", justify="right", width=10, style="bold")

    grand_total = 0
    grand_target = len(MODEL_REGISTRY) * 7 * TOTAL_QUESTIONS

    for model_name in sorted(MODEL_REGISTRY.keys()):
        model_cfg = MODEL_REGISTRY[model_name]
        tier = str(model_cfg.tier)

        base_path = paths.results_base / f"{model_name}.jsonl"
        base_count = len(load_results(base_path)) if base_path.exists() else 0

        row_total = base_count
        cells = []
        base_text, base_style = _cell_style(base_count, TOTAL_QUESTIONS)
        cells.append(f"[{base_style}]{base_text}[/{base_style}]")

        for ht in HintType:
            hp = paths.results_hinted / model_name / f"{ht.value}.jsonl"
            count = len(load_results(hp)) if hp.exists() else 0
            row_total += count
            text, style = _cell_style(count, TOTAL_QUESTIONS)
            cells.append(f"[{style}]{text}[/{style}]")

        grand_total += row_total
        model_target = 7 * TOTAL_QUESTIONS
        pct = row_total * 100 // model_target if model_target > 0 else 0
        total_style = "bold green" if row_total >= model_target else "bold yellow" if row_total > 0 else "dim"
        total_text = f"[{total_style}]{row_total}/{model_target}[/{total_style}]"

        table.add_row(model_name, tier, cells[0], *cells[1:], total_text)

    # Summary row
    pct_str = f"{grand_total * 100 / grand_target:.1f}" if grand_target > 0 else "0.0"
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]", "",
        "", "", "", "", "", "", "",
        f"[bold]{grand_total}/{grand_target} ({pct_str}%)[/bold]",
    )

    return table


@main.command()
@click.option("--watch", "-w", is_flag=True, default=False, help="Auto-refresh every 30s.")
def status(watch: bool) -> None:
    """Show progress dashboard for all models."""
    settings = Settings()
    console = Console()

    if not watch:
        console.print(_build_status_table(settings))
        return

    with Live(
        _build_status_table(settings),
        console=console,
        refresh_per_second=1,
        screen=True,
    ) as live:
        try:
            while True:
                time.sleep(30)
                live.update(_build_status_table(settings))
        except KeyboardInterrupt:
            pass


@main.command("run-all")
@click.option("--budget-per-model", type=float, default=10.0, help="Budget per model in USD.")
@click.option("--dry-run", is_flag=True, default=False, help="Show estimates only.")
@click.option(
    "--skip", multiple=True,
    type=click.Choice(sorted(MODEL_REGISTRY.keys()), case_sensitive=False),
    help="Models to skip.",
)
def run_all(budget_per_model: float, dry_run: bool, skip: tuple[str, ...]) -> None:
    """Run all 14 models sequentially with per-model budget caps."""
    settings = Settings()
    combined_path = settings.paths.combined_dataset

    if not combined_path.exists():
        click.echo("No dataset found. Run 'cot-faithfulness prep' first.")
        raise SystemExit(1)

    questions = load_questions(combined_path)
    skip_set = {s.lower() for s in skip}
    models_to_run = [m for m in sorted(MODEL_REGISTRY.keys()) if m not in skip_set]

    console = Console()
    console.print(f"\n[bold]Running {len(models_to_run)} models[/bold] "
                  f"(budget ${budget_per_model:.2f}/model, {len(questions)} questions)")
    if skip_set:
        console.print(f"[dim]Skipping: {', '.join(sorted(skip_set))}[/dim]")
    console.print()

    if dry_run:
        for model_name in models_to_run:
            model_config = MODEL_REGISTRY[model_name]
            avg_input, avg_output = 1000, 2000
            est_cost = TOTAL_QUESTIONS * 7 * (
                model_config.cost_per_million_input * avg_input / 1_000_000
                + model_config.cost_per_million_output * avg_output / 1_000_000
            )
            console.print(f"  {model_name:<28} ~${est_cost:.2f}")
        return

    completed_models = []
    failed_models = []

    for i, model_name in enumerate(models_to_run, 1):
        console.print(f"\n[bold cyan]{'='*60}")
        console.print(f"[bold cyan]  [{i}/{len(models_to_run)}] {model_name}")
        console.print(f"[bold cyan]{'='*60}")

        try:
            run_single_model(
                model_name=model_name,
                questions=questions,
                hint_types=None,
                settings=settings,
                budget_usd=budget_per_model,
            )
            completed_models.append(model_name)
        except BudgetExceededError as e:
            console.print(f"[yellow]Budget exceeded for {model_name}: {e}[/yellow]")
            failed_models.append((model_name, "budget"))
        except Exception as e:
            console.print(f"[red]Error on {model_name}: {e}[/red]")
            failed_models.append((model_name, str(e)))

    console.print(f"\n{'='*60}")
    console.print(f"[bold green]Completed: {len(completed_models)}/{len(models_to_run)} models[/bold green]")
    if failed_models:
        console.print(f"[yellow]Failed: {', '.join(m for m, _ in failed_models)}[/yellow]")
    console.print()
    console.print(_build_status_table(settings))


@main.command("run-fast")
@click.option(
    "--model", "-m",
    required=True,
    type=click.Choice(sorted(MODEL_REGISTRY.keys()), case_sensitive=False),
    help="Model name from the registry.",
)
@click.option("--budget", type=float, default=25.0, help="Budget in USD.")
@click.option("--concurrency", "-c", type=int, default=5, help="Parallel API calls (default: 5).")
def run_fast(model: str, budget: float, concurrency: int) -> None:
    """Run inference with concurrent API calls (5-10x faster)."""
    import asyncio
    from cot_faithfulness.async_runner import run_single_model_async

    settings = Settings()
    combined_path = settings.paths.combined_dataset

    if not combined_path.exists():
        click.echo("No dataset found. Run 'cot-faithfulness prep' first.")
        raise SystemExit(1)

    questions = load_questions(combined_path)
    click.echo(f"Loaded {len(questions)} questions (concurrency={concurrency})")

    try:
        asyncio.run(run_single_model_async(
            model_name=model,
            questions=questions,
            settings=settings,
            budget_usd=budget,
            concurrency=concurrency,
        ))
    except BudgetExceededError as e:
        click.echo(f"\nBudget exceeded: {e}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
