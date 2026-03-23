# CoT Faithfulness in Open-Weight Reasoning Models

When a reasoning model changes its answer because of an injected hint, does it say so in its chain of thought? This repository contains code, data, and analysis for two papers that investigate that question across 12 open-weight models.

## Key Findings

- **Faithfulness ranges from 39.7% to 89.9%** across 12 models spanning 9 architectural families and 7B to 685B parameters.
- **Consistency (35.5%) and sycophancy (53.9%)** are the hardest hint types for models to acknowledge in their reasoning.
- **Thinking tokens reveal far more than answer text**: models acknowledge hints in their thinking tokens ~87.5% of the time but only ~28.6% in their visible answer text.
- **Classifier choice shifts faithfulness rates by up to 12.9 percentage points** and can reverse model rankings entirely (e.g., Qwen3.5-27B ranks 1st under one classifier, 7th under another).

## Papers

**Paper 1** -- R. J. Young, "Lie to Me: How Faithful Is Chain-of-Thought Reasoning in Open-Weight Reasoning Models?", 2026. *Submitted to arXiv, March 2026.*

- Tests 12 models on 498 questions with 6 hint types (41,832 inference runs)
- Source: `lie-to-me/`

**Paper 2** -- R. J. Young, "Measuring Faithfulness Depends on How You Measure: Classifier Sensitivity in LLM Chain-of-Thought Evaluation", 2026. [arXiv:2603.20172](https://arxiv.org/abs/2603.20172)

- Compares 3 classifiers on 10,276 influenced cases
- Source: `measuring-faithfulness/`

## Repository Structure

```
lie-to-me/                  Paper 1 LaTeX source and figures
measuring-faithfulness/     Paper 2 LaTeX source and figures
src/cot_faithfulness/       Python pipeline (config, models, hints, runner, classifier, analysis)
scripts/                    Figure generation and analysis scripts
data/sampled/               498 sampled questions (300 MMLU + 198 GPQA Diamond)
results/analysis/           Summary CSVs used for all tables and figures
calibration/                Hand-labeled examples for classifier validation
notebooks/                  Exploratory analysis notebooks
tests/                      pytest test suite
```

## Models Tested

All 12 models are open-weight reasoning models accessed via the OpenRouter API. Inference uses temperature=0.0 and seed=103 for reproducibility.

| Model | Family | Architecture | Parameters | Training |
|---|---|---|---|---|
| DeepSeek-R1 | DeepSeek | MoE | 671B / 37B active | GRPO+RL |
| DeepSeek-V3.2-Speciale | DeepSeek | MoE | 685B / 37B active | RL |
| Qwen3.5-27B | Qwen | Dense | 27B | GRPO |
| MiniMax-M2.5 | MiniMax | MoE | 230B / 10B active | RL |
| GPT-OSS-120B | OpenAI | MoE | 117B / 5.1B active | RL |
| ERNIE-4.5-21B | Baidu | MoE | 21B / 3B active | RL+SFT |
| QwQ-32B | Qwen | Dense | 32B | RL |
| OLMo-3.1-32B-Think | AI2 | Dense | 32B | RL+SFT |
| OLMo-3-7B-Think | AI2 | Dense | 7B | SFT+RL |
| Nemotron-Nano-9B | NVIDIA | Dense | 9B | SFT+RL |
| Step-3.5-Flash | StepFun | MoE | 196B / 11B active | RL |
| Seed-1.6-Flash | ByteDance | Dense | Undisclosed | RL |

9 families. 7B to 685B total parameters. Dense, MoE, and hybrid Transformer-Mamba architectures.

## Hint Types

Six reasoning hints are injected to bias models toward an incorrect answer:

| Hint Type | Mechanism |
|---|---|
| Sycophancy | Authority appeal ("A Stanford professor believes the answer is...") |
| Consistency | Self-consistency pressure ("In a previous conversation, you selected...") |
| Visual pattern | Answer choice reordering (target moved to position A) |
| Metadata | Spurious XML metadata tags containing the target answer |
| Grader hacking | Claim that an automated grading system expects the target |
| Unethical information | Claim of unauthorized access to the answer key |

## Quick Start

### Installation

```bash
git clone https://github.com/ricyoung/cot-faithfulness-open-models.git
cd cot-faithfulness-open-models
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
pytest --cov=cot_faithfulness
```

### Lint and Type Check

```bash
ruff check src/ tests/
mypy src/
```

### CLI Usage

```bash
# Prepare data
cot-faithfulness prepare-data

# Run baseline inference
cot-faithfulness run-base --model deepseek-r1

# Run hinted inference
cot-faithfulness run-hinted --model deepseek-r1 --hint sycophancy

# Classify faithfulness
cot-faithfulness classify --model deepseek-r1

# Generate analysis
cot-faithfulness analyze

# Generate figures
cot-faithfulness plot
```

### Reproduce Paper Figures

All figures are generated from the CSVs in `results/analysis/`:

```bash
python scripts/generate_paper1_figures.py    # Paper 1 (7 figures)
python scripts/generate_paper2_figures.py    # Paper 2 (4 figures)
```

## Data

### Evaluation Questions

`data/sampled/` contains 498 multiple-choice questions in JSONL format:

- **300 MMLU questions** -- stratified across 57 academic subjects (round-robin sampling, seed=103)
- **198 GPQA Diamond questions** -- the complete Diamond split of graduate-level science questions

Each record includes the question stem, four answer choices (A--D), the correct answer label, source benchmark, and subject.

### Full Results

Raw inference outputs, classifier labels, and cost summaries are available on Hugging Face:

[huggingface.co/datasets/richardyoung/cot-faithfulness-open-models](https://huggingface.co/datasets/richardyoung/cot-faithfulness-open-models)

| Artifact | Location | Size |
|---|---|---|
| Analysis CSVs (tables and figures) | GitHub `results/analysis/` | 736K |
| Base inference runs | HF `results/base/` | 102M |
| Hinted inference runs | HF `results/hinted/` | 793M |
| Pipeline classifications | HF `results/classified/` | 15M |
| Sonnet classifications | HF `results/classified_sonnet/` | 4.3M |

## Experiment Scale

```
498 questions x 6 hint types x 12 models = 35,856 hinted inference runs
498 questions x 12 models                =  5,976 baseline inference runs
                                  Total  = 41,832 inference runs
```

Of the 35,856 hinted runs, 10,276 resulted in the model changing its answer to match the hint target (28.7% influence rate). These 10,276 influenced cases form the basis for all faithfulness analysis.

## Citation

```bibtex
@article{young2026lietome,
  title   = {Lie to Me: How Faithful Is Chain-of-Thought Reasoning
             in Open-Weight Reasoning Models?},
  author  = {Young, Richard J.},
  journal = {arXiv preprint},
  year    = {2026}
}

@article{young2026classifier,
  title   = {Measuring Faithfulness Depends on How You Measure:
             Classifier Sensitivity in {LLM} Chain-of-Thought Evaluation},
  author  = {Young, Richard J.},
  journal = {arXiv preprint arXiv:2603.20172},
  year    = {2026}
}
```

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Author

Richard J. Young -- University of Nevada, Las Vegas (Lee Business School) and DeepNeuro AI

ryoung@unlv.edu | richard@deepneuro.ai
