"""
Pydantic data models for questions, inference results, and classifications.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from pydantic import BaseModel, computed_field

from cot_faithfulness.config import HintType


class Question(BaseModel):
    """A single multiple-choice question."""
    id: str
    dataset: str  # "mmlu" or "gpqa"
    subject: str
    question_text: str
    choices: list[str]
    choice_labels: list[str]  # ["A", "B", "C", "D"]
    correct_label: str
    correct_index: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def formatted(self) -> str:
        """Format question with labeled choices for model input."""
        lines = [self.question_text, ""]
        for label, choice in zip(self.choice_labels, self.choices):
            lines.append(f"({label}) {choice}")
        return "\n".join(lines)


class HintedQuestion(BaseModel):
    """A question with an injected hint pointing to a wrong answer."""
    original: Question
    hint_type: HintType
    hint_text: str
    target_label: str
    target_index: int
    reordered_choices: Optional[list[str]] = None
    reordered_labels: Optional[list[str]] = None
    reorder_mapping: Optional[dict[str, str]] = None  # original_label -> new_label

    @computed_field  # type: ignore[prop-decorator]
    @property
    def formatted(self) -> str:
        """Format hinted question for model input."""
        if self.hint_type == HintType.VISUAL_PATTERN and self.reordered_choices:
            lines = [self.original.question_text, ""]
            for label, choice in zip(
                self.reordered_labels or self.original.choice_labels,
                self.reordered_choices,
            ):
                lines.append(f"({label}) {choice}")
            return "\n".join(lines)

        return f"{self.hint_text}\n\n{self.original.formatted}"


class InferenceResult(BaseModel):
    """Result from a single model inference call."""
    question_id: str
    model_name: str
    hint_type: Optional[str] = None  # None for base runs
    full_response: str
    thinking_text: str
    answer_text: str
    extracted_answer: Optional[str] = None  # Parsed A/B/C/D
    target_label: Optional[str] = None  # For hinted runs
    correct_label: str
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


def make_question_id(dataset: str, index: int, question_text: str) -> str:
    """Generate a deterministic question ID."""
    text_hash = hashlib.sha256(question_text.encode()).hexdigest()[:8]
    return f"{dataset}_{index:04d}_{text_hash}"
