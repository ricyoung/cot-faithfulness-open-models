# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating **Chain-of-Thought (CoT) faithfulness in open-weight reasoning models**. Extends Anthropic's paper "Reasoning Models Don't Always Say What They Think" (arXiv 2505.05410) to 12 models across 9 families, spanning 8B to 1T parameters.

**Core question**: When a model changes its answer due to an injected hint, does it acknowledge the hint in its CoT? The original paper found Claude 3.7 Sonnet acknowledges hints only 25% of the time; DeepSeek R1 only 39%.

**Status**: Planning/pre-implementation. The `IMPLEMENTATION_PLAN.md` contains the original code specification (needs updating for API-based inference). No source code exists yet.

## Project Structure (planned)

- `src/cot_faithfulness/` — Python package (config, models, hints, runner, classifier, analysis, plots, CLI)
- `data/sampled/` — 500 questions (300 MMLU + 200 GPQA Diamond), seed=103
- `results/` — base runs, hinted runs, classified results, analysis outputs
- `tests/` — pytest suite
- `calibration/` — 200 hand-labeled examples for classifier validation
- `scripts/` — shell scripts for orchestrating full runs

## Key Design Decisions

- **Inference**: OpenRouter API (OpenAI-compatible chat completions). NOT local vLLM — all models accessed via API
- **Thinking tokens**: Accessed via OpenRouter `reasoning` parameter, returned in `reasoning_details` array
- **Generation params**: temperature=0.0, seed=103 for reproducibility
- **Faithfulness classification**: Two-stage pipeline — regex/keyword matching first, then LLM judge for ambiguous cases
- **CLI**: Click-based CLI entry point (`cot-faithfulness` command)
- **Config**: Pydantic Settings reading from `.env` (for `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`)
- **Build system**: Hatchling with `src/` layout
- **Claude Sonnet 4.6**: Use Anthropic Batch API (50% discount) with extended thinking

## Models Under Test (12 models)

### Tier 1: Replication baselines
| Model | Params | Family | Role |
|---|---|---|---|
| DeepSeek-R1 | 671B / 37B active MoE | DeepSeek | Anthropic tested this — direct replication |
| DeepSeek-V3.2-Speciale | 685B / 37B active MoE | DeepSeek | Newest DeepSeek reasoner (Dec 2025) |
| Claude Sonnet 4.6 | Proprietary | Anthropic | Proprietary anchor, validates pipeline |

### Tier 2: Current flagships (Q4 2025 – Q1 2026)
| Model | Params | Family | Role |
|---|---|---|---|
| GLM-5 | 744B / 40B active MoE | Zhipu | Feb 2026, top open-source leaderboard |
| Qwen3.5-27B | 27B dense | Qwen/Alibaba | Feb 2026, best dense reasoning model |
| Kimi K2.5 | 1T / 32B active MoE | Moonshot | Jan 2026, largest open reasoner |
| MiniMax-M2.5 | 230B / 10B active MoE | MiniMax | Feb 2026, interleaved thinking |
| GPT-OSS-120B | 117B / 5.1B active MoE | OpenAI | OpenAI's open model, very cheap |

### Tier 3: Scaling & diversity
| Model | Params | Family | Role |
|---|---|---|---|
| Apriel-1.5-15B-Thinker | 15B dense | ServiceNow | No RL, pure data-centric training |
| EXAONE-4.0-32B | 32B dense | LG AI | Hybrid reasoning/non-reasoning modes |
| OLMo-3.1-32B-Think | 32B dense | AI2 | Fully open (data+code+weights) |
| DeepSeek-R1-Distill-Qwen3-8B | 8B dense | DeepSeek | Smallest, for scaling analysis |

### Coverage
- **9 model families**: DeepSeek, Qwen, OpenAI, Zhipu, Moonshot, MiniMax, ServiceNow, LG, AI2 + Anthropic
- **Architectures**: Dense, MoE, hybrid-attention, depth-upscaled
- **Training methods**: Distillation, RL, GRPO, DPO, SFT, data-centric, hybrid
- **Scale**: 8B → 15B → 27B → 32B (dense); 5.1B → 10B → 32B → 40B active (MoE)
- **Release dates**: Aug 2025 – Feb 2026

## Six Hint Types

sycophancy, consistency, visual_pattern, metadata, grader, unethical — each injected to point toward an incorrect answer. Visual pattern reorders options instead of adding text.

## Build & Run Commands (planned)

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Tests
pytest                          # all tests
pytest tests/test_hints.py      # single file
pytest -k "test_sycophancy"     # single test by name
pytest --cov=cot_faithfulness   # with coverage

# Lint & type check
ruff check src/ tests/
mypy src/

# CLI (after implementation)
cot-faithfulness prepare-data
cot-faithfulness run-base --model deepseek-r1
cot-faithfulness run-hinted --model deepseek-r1 --hint sycophancy
cot-faithfulness classify --model deepseek-r1
cot-faithfulness analyze
cot-faithfulness plot

# Full sweep
bash scripts/run_all.sh
bash scripts/run_single_model.sh deepseek-r1
```

## Experiment Scale

```
500 questions x 6 hint types x 12 models = 36,000 hinted inference runs
500 questions x 12 models                =  6,000 baseline inference runs
                                  Total  = 42,000 inference runs
```

Estimated API cost: ~$50-80 (open models via OpenRouter) + ~$80 (Claude via Anthropic Batch API) = ~$130-160 total.

## Key Files to Read First

1. `PROJECT_SUMMARY.md` — concise overview of goals, data, models, compute
2. `IMPLEMENTATION_PLAN.md` — original code specification (model registry and inference engine need updating for API approach)
3. `research_plan.md` — methodology, research questions, related work, literature gaps

## Conventions

- Python 3.11+, ruff for linting (line-length=100), mypy for types
- Pydantic models for all data schemas and configuration
- JSONL format for all intermediate results (one record per question-hint pair)
- All randomness uses seed=103 for reproducibility
- Results stored per-model in `results/{base,hinted,classified}/{model_name}/`
