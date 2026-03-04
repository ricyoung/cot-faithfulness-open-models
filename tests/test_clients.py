"""Tests for client utilities (no network calls)."""
from __future__ import annotations

import pytest

from cot_faithfulness.clients import split_thinking


class TestSplitThinking:
    """Tests for the split_thinking function."""

    def test_think_tags(self):
        text = "<think>Let me reason about this...</think>The answer is B."
        thinking, answer = split_thinking(text)
        assert thinking == "Let me reason about this..."
        assert answer == "The answer is B."

    def test_think_tags_multiline(self):
        text = "<think>\nStep 1: Consider A\nStep 2: Consider B\n</think>\nFinal: (B)"
        thinking, answer = split_thinking(text)
        assert "Step 1" in thinking
        assert "Step 2" in thinking
        assert answer == "Final: (B)"

    def test_pipe_thinking_tags(self):
        text = "<|thinking|>reasoning here<|/thinking|>answer here"
        thinking, answer = split_thinking(text)
        assert thinking == "reasoning here"
        assert answer == "answer here"

    def test_no_tags(self):
        text = "Just a plain answer with no thinking tags."
        thinking, answer = split_thinking(text)
        assert thinking == ""
        assert answer == text

    def test_empty_string(self):
        thinking, answer = split_thinking("")
        assert thinking == ""
        assert answer == ""

    def test_empty_think_tags(self):
        text = "<think></think>The answer is A."
        thinking, answer = split_thinking(text)
        assert thinking == ""
        assert answer == "The answer is A."

    def test_think_tag_priority_over_pipe(self):
        """If both tag styles present, <think> wins (checked first)."""
        text = "<think>first</think><|thinking|>second<|/thinking|>answer"
        thinking, answer = split_thinking(text)
        assert thinking == "first"
