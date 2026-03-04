"""
Dataset download and sampling for MMLU and GPQA Diamond.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from cot_faithfulness.config import Settings
from cot_faithfulness.schemas import Question, make_question_id


def download_mmlu(cache_dir: Path) -> list[dict]:
    """Download MMLU test split via HuggingFace datasets."""
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("cais/mmlu", "all", split="test", cache_dir=str(cache_dir))
    return list(ds)


def download_gpqa(cache_dir: Path) -> list[dict]:
    """Download GPQA Diamond split via HuggingFace datasets."""
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "Idavidrein/gpqa", "gpqa_diamond", split="train", cache_dir=str(cache_dir)
    )
    return list(ds)


def parse_mmlu_question(row: dict, index: int) -> Question:
    """Convert an MMLU row to a Question."""
    choices = [row["choices"][i] for i in range(4)]
    correct_idx = int(row["answer"])
    labels = ["A", "B", "C", "D"]
    return Question(
        id=make_question_id("mmlu", index, row["question"]),
        dataset="mmlu",
        subject=row.get("subject", "unknown"),
        question_text=row["question"],
        choices=choices,
        choice_labels=labels,
        correct_label=labels[correct_idx],
        correct_index=correct_idx,
    )


def parse_gpqa_question(row: dict, index: int) -> Question:
    """Convert a GPQA row to a Question."""
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    # Shuffle choices deterministically so correct isn't always first
    rng = random.Random(42 + index)
    labeled_choices = list(zip(choices, [0, 1, 2, 3]))
    rng.shuffle(labeled_choices)
    shuffled_choices = [c for c, _ in labeled_choices]
    original_indices = [i for _, i in labeled_choices]
    correct_new_idx = original_indices.index(0)

    labels = ["A", "B", "C", "D"]
    return Question(
        id=make_question_id("gpqa", index, row["Question"]),
        dataset="gpqa",
        subject=row.get("Subdomain", "graduate-level science"),
        question_text=row["Question"],
        choices=shuffled_choices,
        choice_labels=labels,
        correct_label=labels[correct_new_idx],
        correct_index=correct_new_idx,
    )


def stratified_sample_mmlu(
    questions: list[Question],
    n: int,
    seed: int = 103,
) -> list[Question]:
    """Sample n questions from MMLU with subject stratification."""
    by_subject: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_subject[q.subject].append(q)

    rng = random.Random(seed)
    for qs in by_subject.values():
        rng.shuffle(qs)

    # Round-robin across subjects
    sampled: list[Question] = []
    subject_iters = {s: iter(qs) for s, qs in sorted(by_subject.items())}

    while len(sampled) < n:
        added_this_round = False
        for subject in sorted(subject_iters.keys()):
            if len(sampled) >= n:
                break
            try:
                sampled.append(next(subject_iters[subject]))
                added_this_round = True
            except StopIteration:
                continue
        if not added_this_round:
            break

    return sampled


def sample_gpqa(
    questions: list[Question],
    n: int,
    seed: int = 103,
) -> list[Question]:
    """Sample n questions from GPQA."""
    rng = random.Random(seed)
    shuffled = list(questions)
    rng.shuffle(shuffled)
    return shuffled[:n]


def save_questions(questions: list[Question], path: Path) -> None:
    """Save questions as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for q in questions:
            f.write(q.model_dump_json() + "\n")


def load_questions(path: Path) -> list[Question]:
    """Load questions from JSONL."""
    questions = []
    with open(path) as f:
        for line in f:
            if line.strip():
                questions.append(Question.model_validate_json(line))
    return questions


def prepare_all(settings: Optional[Settings] = None) -> list[Question]:
    """
    Full data preparation pipeline:
    1. Download MMLU and GPQA
    2. Parse into Question objects
    3. Sample (stratified for MMLU)
    4. Merge and save
    """
    settings = settings or Settings()
    paths = settings.paths

    # Download
    print(f"Downloading MMLU to {paths.data_raw}...")
    mmlu_raw = download_mmlu(paths.data_raw)
    print(f"  Downloaded {len(mmlu_raw)} MMLU questions")

    gpqa_raw: list[dict] = []
    try:
        print(f"Downloading GPQA to {paths.data_raw}...")
        gpqa_raw = download_gpqa(paths.data_raw)
        print(f"  Downloaded {len(gpqa_raw)} GPQA questions")
    except Exception as e:
        print(f"  WARNING: Could not download GPQA: {e}")
        print("  Set HF_TOKEN env var or visit https://huggingface.co/datasets/Idavidrein/gpqa")
        print("  Continuing with MMLU only.")

    # Parse
    mmlu_questions = [parse_mmlu_question(row, i) for i, row in enumerate(mmlu_raw)]
    gpqa_questions = [parse_gpqa_question(row, i) for i, row in enumerate(gpqa_raw)]

    # Sample
    mmlu_sample = stratified_sample_mmlu(
        mmlu_questions, settings.mmlu_sample_size, seed=settings.experiment_seed
    )
    gpqa_sample = sample_gpqa(
        gpqa_questions, settings.gpqa_sample_size, seed=settings.experiment_seed
    ) if gpqa_questions else []

    print(f"Sampled {len(mmlu_sample)} MMLU + {len(gpqa_sample)} GPQA questions")

    # Save individual files
    save_questions(mmlu_sample, paths.data_sampled / f"mmlu_{len(mmlu_sample)}.jsonl")
    if gpqa_sample:
        save_questions(gpqa_sample, paths.data_sampled / f"gpqa_{len(gpqa_sample)}.jsonl")

    # Merge and save combined
    combined = mmlu_sample + gpqa_sample
    save_questions(combined, paths.combined_dataset)
    print(f"Saved {len(combined)} questions to {paths.combined_dataset}")

    return combined
