"""Tests for answer extraction from model output."""
from __future__ import annotations

import pytest

from cot_faithfulness.answer_extraction import (
    extract_answer,
    extract_answer_for_visual_pattern,
)


class TestExtractAnswer:
    """Tests for the main extract_answer function."""

    def test_explicit_pattern_the_answer_is(self):
        assert extract_answer("", "", "The answer is (B)") == "B"

    def test_explicit_pattern_final_answer(self):
        assert extract_answer("", "", "Final answer: C") == "C"

    def test_explicit_pattern_i_choose(self):
        assert extract_answer("", "", "I'll choose (A)") == "A"

    def test_explicit_pattern_my_answer(self):
        assert extract_answer("", "", "My answer: D") == "D"

    def test_explicit_pattern_i_select(self):
        assert extract_answer("", "", "I select (B)") == "B"

    def test_explicit_pattern_go_with(self):
        assert extract_answer("", "", "I'll go with (C)") == "C"

    def test_boxed_answer(self):
        assert extract_answer("", "", r"\boxed{B}") == "B"

    def test_boxed_answer_with_parens(self):
        assert extract_answer("", "", r"\boxed{(A)}") == "A"

    def test_bare_letter_in_parens(self):
        assert extract_answer("", "", "Looking at the options, (D)") == "D"

    def test_last_bare_letter_wins(self):
        """When multiple bare letters, take the last one."""
        assert extract_answer("", "", "Considering (A) and (B), but (C)") == "C"

    def test_fallback_to_full_response(self):
        """When answer_text has nothing, check full_response."""
        assert extract_answer("The answer is (A)", "", "I'm not sure") == "A"

    def test_returns_none_when_no_answer(self):
        assert extract_answer("no answer here", "", "no answer here either") is None

    def test_case_insensitive(self):
        assert extract_answer("", "", "the answer is (b)") == "B"

    def test_answer_text_preferred_over_full(self):
        """answer_text explicit pattern beats full_response."""
        result = extract_answer(
            "The answer is (A)", "", "The answer is (B)"
        )
        assert result == "B"

    def test_empty_strings(self):
        assert extract_answer("", "", "") is None

    def test_correct_answer_pattern(self):
        assert extract_answer("", "", "The correct answer is A") == "A"

    def test_markdown_bold_answer(self):
        assert extract_answer("", "", "**Answer:** B") == "B"

    def test_markdown_bold_final_answer(self):
        assert extract_answer("", "", "**Final answer:** D") == "D"

    def test_thinking_fallback_when_answer_empty(self):
        """When answer_text is empty, check end of thinking_text."""
        assert extract_answer("", "lots of reasoning... the answer is (C)", "") == "C"

    def test_thinking_fallback_uses_tail(self):
        """Only checks last 500 chars of thinking."""
        thinking = "x" * 600 + "the answer is (A)"
        assert extract_answer("", thinking, "") == "A"


class TestExtractAnswerForVisualPattern:
    """Tests for visual pattern answer extraction with reorder mapping."""

    def test_maps_back_to_original(self):
        """Answer in reordered space should map back to original."""
        # If original B is now A after reordering
        mapping = {"A": "B", "B": "A", "C": "C", "D": "D"}
        result = extract_answer_for_visual_pattern(
            "", "", "The answer is (A)", mapping
        )
        # Model said A in reordered space -> which was originally B
        assert result == "B"

    def test_identity_mapping(self):
        """No reordering should return same answer."""
        mapping = {"A": "A", "B": "B", "C": "C", "D": "D"}
        result = extract_answer_for_visual_pattern(
            "", "", "The answer is (C)", mapping
        )
        assert result == "C"

    def test_returns_none_when_no_answer(self):
        mapping = {"A": "B", "B": "A", "C": "C", "D": "D"}
        result = extract_answer_for_visual_pattern(
            "", "", "I don't know", mapping
        )
        assert result is None
