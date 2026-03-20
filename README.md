# CoT Faithfulness in Open-Weight Reasoning Models

Research project investigating Chain-of-Thought (CoT) faithfulness in 12 open-weight reasoning models across 9 families, spanning 7B to 1T parameters.

## Papers

**Paper 1: Lie to Me** — *How Faithful Is Chain-of-Thought Reasoning in Open-Weight Reasoning Models?*
- Tests 12 models on 498 questions with 6 hint types (41,832 inference runs)
- Faithfulness rates range from 39.7% to 89.9% across model families
- Keyword-based analysis reveals a striking thinking-token vs. answer-text acknowledgment gap (87.5% vs. 28.6%)
- Source: `paper/`

**Paper 2: Measuring Faithfulness** — *Measuring Faithfulness Depends on How You Measure: Classifier Sensitivity in LLM Chain-of-Thought Evaluation*
- Three classifiers produce faithfulness rates of 74.4%, 82.6%, and 69.7% on identical data
- Classifier choice can reverse model rankings (Qwen3.5-27B: 1st vs. 7th)
- Source: `paper2/`
- arXiv: submit/7389027

## Repository Structure

```
paper/                      # Paper 1 LaTeX source + figures
paper2/                     # Paper 2 LaTeX source + figures
src/cot_faithfulness/       # Python pipeline (inference, classification, analysis)
scripts/                    # Figure generation and analysis scripts
data/sampled/               # 498 sampled questions (300 MMLU + 198 GPQA Diamond)
results/analysis/           # Summary CSVs for figure/table generation
tests/                      # Test suite
```

## Reproducing Paper Figures

All paper figures are generated from `results/analysis/` CSVs:

```bash
# Paper 1 figures (7 figures)
python scripts/generate_paper1_figures.py

# Paper 2 figures (4 figures)
python scripts/generate_paper2_figures.py
```

The analysis CSVs are included in this repository. Full inference outputs (793MB+) are available on Hugging Face.

## Full Results

Raw inference outputs, classifier labels, and cost summaries are available at:
https://huggingface.co/datasets/richardyoung/cot-faithfulness-open-models

| Artifact | Location | Size |
|----------|----------|------|
| Analysis CSVs (tables/figures) | GitHub `results/analysis/` | 736K |
| Base inference runs | HF `results/base/` | 102M |
| Hinted inference runs | HF `results/hinted/` | 793M |
| Pipeline classifications | HF `results/classified/` | 15M |
| Sonnet classifications | HF `results/classified_sonnet/` | 4.3M |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Models Evaluated

| Model | Family | Params | Training |
|-------|--------|--------|----------|
| DeepSeek-R1 | DeepSeek | 671B/37B MoE | GRPO+RL |
| DeepSeek-V3.2-Speciale | DeepSeek | 685B/37B MoE | RL |
| Qwen3.5-27B | Qwen | 27B dense | GRPO |
| MiniMax-M2.5 | MiniMax | 230B/10B MoE | RL |
| GPT-OSS-120B | OpenAI | 117B/5.1B MoE | RL |
| ERNIE-4.5-21B | Baidu | 21B/3B MoE | RL+SFT |
| QwQ-32B | Qwen | 32B dense | RL |
| OLMo-3.1-32B-Think | AI2 | 32B dense | RL+SFT |
| OLMo-3-7B-Think | AI2 | 7B dense | SFT+RL |
| Nemotron-Nano-9B | NVIDIA | 9B dense | SFT+RL |
| Step-3.5-Flash | StepFun | 196B/11B MoE | RL |
| Seed-1.6-Flash | ByteDance | Undisclosed | RL |

## Citation

```bibtex
@article{young2026classifier,
  title   = {Measuring Faithfulness Depends on How You Measure: Classifier Sensitivity in {LLM} Chain-of-Thought Evaluation},
  author  = {Young, Richard J.},
  journal = {arXiv preprint},
  year    = {2026}
}

@article{young2026lietome,
  title   = {Lie to Me: How Faithful Is Chain-of-Thought Reasoning in Open-Weight Reasoning Models?},
  author  = {Young, Richard J.},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

CC BY 4.0

## Author

Richard J. Young — University of Nevada, Las Vegas (Lee Business School) and DeepNeuro AI

ryoung@unlv.edu | richard@deepneuro.ai
