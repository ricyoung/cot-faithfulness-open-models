"""Shared fixtures for the CoT faithfulness test suite."""
from __future__ import annotations

import pytest

from cot_faithfulness.config import PathConfig, Settings
from cot_faithfulness.schemas import Question


@pytest.fixture
def sample_question() -> Question:
    """A simple 4-choice MMLU-style question."""
    return Question(
        id="mmlu_0001_abcd1234",
        dataset="mmlu",
        subject="astronomy",
        question_text="What is the closest star to Earth?",
        choices=["Proxima Centauri", "Sirius", "Betelgeuse", "Alpha Centauri A"],
        choice_labels=["A", "B", "C", "D"],
        correct_label="A",
        correct_index=0,
    )


@pytest.fixture
def sample_question_b_correct() -> Question:
    """Question where the correct answer is B (not A)."""
    return Question(
        id="mmlu_0002_efgh5678",
        dataset="mmlu",
        subject="biology",
        question_text="What organelle produces ATP?",
        choices=["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"],
        choice_labels=["A", "B", "C", "D"],
        correct_label="B",
        correct_index=1,
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings with a temporary data directory."""
    return Settings(
        openrouter_api_key="test-key-not-real",
        paths=PathConfig(project_root=tmp_path),
    )
