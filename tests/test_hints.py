"""Tests for the hint injection system."""
from __future__ import annotations

import pytest

from cot_faithfulness.config import HintType
from cot_faithfulness.hints import HINT_TEMPLATES, HintInjector
from cot_faithfulness.schemas import Question


class TestHintInjectorSelectTarget:
    """Tests for target selection."""

    def test_target_is_wrong_answer(self, sample_question):
        injector = HintInjector(seed=42)
        target_label, target_index = injector.select_target(sample_question)
        assert target_label != sample_question.correct_label
        assert target_index != sample_question.correct_index

    def test_target_is_valid_label(self, sample_question):
        injector = HintInjector(seed=42)
        target_label, target_index = injector.select_target(sample_question)
        assert target_label in sample_question.choice_labels
        assert 0 <= target_index < len(sample_question.choices)

    def test_deterministic_with_same_seed(self, sample_question):
        r1 = HintInjector(seed=42).select_target(sample_question)
        r2 = HintInjector(seed=42).select_target(sample_question)
        assert r1 == r2

    def test_different_seed_may_differ(self, sample_question):
        """Different seeds can produce different targets (not guaranteed but likely)."""
        results = set()
        for seed in range(100):
            label, _ = HintInjector(seed=seed).select_target(sample_question)
            results.add(label)
        # With 3 wrong answers and 100 seeds, should hit at least 2 different targets
        assert len(results) >= 2


class TestHintInjectorInject:
    """Tests for hint injection."""

    @pytest.mark.parametrize("hint_type", [
        HintType.SYCOPHANCY,
        HintType.CONSISTENCY,
        HintType.METADATA,
        HintType.GRADER,
        HintType.UNETHICAL,
    ])
    def test_text_hint_contains_target(self, sample_question, hint_type):
        injector = HintInjector(seed=42)
        target_label, target_index = injector.select_target(sample_question)
        hq = injector.inject(sample_question, hint_type, target_label, target_index)
        assert target_label in hq.hint_text

    @pytest.mark.parametrize("hint_type", [
        HintType.SYCOPHANCY,
        HintType.CONSISTENCY,
        HintType.METADATA,
        HintType.GRADER,
        HintType.UNETHICAL,
    ])
    def test_text_hint_formatted_contains_question(self, sample_question, hint_type):
        injector = HintInjector(seed=42)
        hq = injector.inject(sample_question, hint_type)
        assert sample_question.question_text in hq.formatted

    def test_sycophancy_mentions_expert(self, sample_question):
        injector = HintInjector(seed=42)
        hq = injector.inject(sample_question, HintType.SYCOPHANCY)
        assert "Stanford" in hq.hint_text or "professor" in hq.hint_text.lower()

    def test_consistency_mentions_previous(self, sample_question):
        injector = HintInjector(seed=42)
        hq = injector.inject(sample_question, HintType.CONSISTENCY)
        assert "previous" in hq.hint_text.lower()

    def test_metadata_has_xml(self, sample_question):
        injector = HintInjector(seed=42)
        hq = injector.inject(sample_question, HintType.METADATA)
        assert "<answer_key>" in hq.hint_text

    def test_grader_mentions_grading(self, sample_question):
        injector = HintInjector(seed=42)
        hq = injector.inject(sample_question, HintType.GRADER)
        assert "grading" in hq.hint_text.lower()

    def test_unethical_mentions_unauthorized(self, sample_question):
        injector = HintInjector(seed=42)
        hq = injector.inject(sample_question, HintType.UNETHICAL)
        assert "unauthorized" in hq.hint_text.lower()


class TestVisualPattern:
    """Tests for visual pattern (choice reordering) hint."""

    def test_target_becomes_first(self, sample_question):
        injector = HintInjector(seed=42)
        target_label, target_index = injector.select_target(sample_question)
        hq = injector.inject(sample_question, HintType.VISUAL_PATTERN, target_label, target_index)

        assert hq.target_label == "A"
        assert hq.target_index == 0
        assert hq.reordered_choices is not None
        assert hq.reordered_choices[0] == sample_question.choices[target_index]

    def test_all_choices_preserved(self, sample_question):
        injector = HintInjector(seed=42)
        hq = injector.inject(sample_question, HintType.VISUAL_PATTERN)

        assert hq.reordered_choices is not None
        assert sorted(hq.reordered_choices) == sorted(sample_question.choices)

    def test_no_text_hint(self, sample_question):
        injector = HintInjector(seed=42)
        hq = injector.inject(sample_question, HintType.VISUAL_PATTERN)
        assert hq.hint_text == ""

    def test_reorder_mapping_valid(self, sample_question):
        injector = HintInjector(seed=42)
        hq = injector.inject(sample_question, HintType.VISUAL_PATTERN)

        assert hq.reorder_mapping is not None
        # Mapping should cover all labels
        assert set(hq.reorder_mapping.keys()) == set(sample_question.choice_labels)
        assert set(hq.reorder_mapping.values()) == set(sample_question.choice_labels)

    def test_formatted_has_reordered_choices(self, sample_question_b_correct):
        """Visual pattern formatted output uses reordered choices."""
        injector = HintInjector(seed=42)
        target_label, target_index = injector.select_target(sample_question_b_correct)
        hq = injector.inject(
            sample_question_b_correct, HintType.VISUAL_PATTERN, target_label, target_index
        )

        formatted = hq.formatted
        # Should contain the question text
        assert sample_question_b_correct.question_text in formatted
        # First choice in formatted should be the target
        assert hq.reordered_choices is not None
        lines = formatted.strip().split("\n")
        choice_lines = [l for l in lines if l.startswith("(")]
        assert choice_lines[0].startswith("(A)")


class TestInjectAllHints:
    """Tests for inject_all_hints (all 6 at once)."""

    def test_returns_all_hint_types(self, sample_question):
        injector = HintInjector(seed=42)
        all_hints = injector.inject_all_hints(sample_question)
        assert set(all_hints.keys()) == set(HintType)

    def test_same_target_across_all(self, sample_question):
        injector = HintInjector(seed=42)
        target_label, target_index = injector.select_target(sample_question)
        all_hints = injector.inject_all_hints(
            sample_question, target_label, target_index
        )

        for ht, hq in all_hints.items():
            if ht == HintType.VISUAL_PATTERN:
                # Visual pattern target is always "A" after reordering
                assert hq.target_label == "A"
            else:
                assert hq.target_label == target_label
