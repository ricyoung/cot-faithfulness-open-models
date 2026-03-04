"""Tests for pipeline hardening: retry, checkpoints, extraction, cost, failure tracking."""
from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openai import BadRequestError, RateLimitError

from cot_faithfulness.answer_extraction import extract_answer
from cot_faithfulness.clients import OpenRouterClient, split_thinking
from cot_faithfulness.config import ModelConfig, Settings
from cot_faithfulness.runner import (
    BudgetExceededError,
    _compute_cost,
    _ConsecutiveFailureTracker,
    get_completed_ids,
    load_results,
)
from cot_faithfulness.schemas import InferenceResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_result_json() -> str:
    """A valid InferenceResult as JSON."""
    return InferenceResult(
        question_id="test_001",
        model_name="deepseek-r1",
        hint_type=None,
        full_response="The answer is A",
        thinking_text="reasoning",
        answer_text="The answer is A",
        extracted_answer="A",
        correct_label="A",
        input_tokens=100,
        output_tokens=200,
    ).model_dump_json()


# ---------------------------------------------------------------------------
# TestCorruptCheckpoint
# ---------------------------------------------------------------------------

class TestCorruptCheckpoint:
    """Verify corrupt JSONL lines are skipped with a warning."""

    def test_get_completed_ids_skips_corrupt(self, tmp_path: Path, valid_result_json: str):
        path = tmp_path / "results.jsonl"
        path.write_text(valid_result_json + "\n" + "NOT VALID JSON\n")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ids = get_completed_ids(path)
        assert ids == {"test_001"}
        assert len(w) == 1
        assert "Corrupt line 2" in str(w[0].message)

    def test_load_results_skips_corrupt(self, tmp_path: Path, valid_result_json: str):
        path = tmp_path / "results.jsonl"
        path.write_text(valid_result_json + "\n" + "{bad json}\n" + valid_result_json.replace("test_001", "test_002") + "\n")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = load_results(path)
        assert len(results) == 2
        assert len(w) == 1

    def test_get_completed_ids_empty_file(self, tmp_path: Path):
        path = tmp_path / "results.jsonl"
        path.write_text("")
        ids = get_completed_ids(path)
        assert ids == set()


# ---------------------------------------------------------------------------
# TestSmartRetry
# ---------------------------------------------------------------------------

class TestSmartRetry:
    """Verify smart retry only retries transient errors."""

    def test_bad_request_no_retry(self):
        """BadRequestError (400) should NOT be retried."""
        settings = Settings(
            openrouter_api_key="test-key",
            api={"max_retries": 3, "retry_min_wait": 0.01, "retry_max_wait": 0.02},
        )
        client = OpenRouterClient("deepseek-r1", settings)

        mock_response = MagicMock()
        err = BadRequestError(
            message="bad request",
            response=mock_response,
            body=None,
        )
        mock_response.status_code = 400
        mock_response.headers = {}

        with patch.object(client.client.chat.completions, "create", side_effect=err) as mock_create:
            with pytest.raises(BadRequestError):
                client.generate("test prompt")
            # Should only have been called once (no retries)
            assert mock_create.call_count == 1

    def test_rate_limit_retries(self):
        """RateLimitError (429) should be retried up to max_retries."""
        settings = Settings(
            openrouter_api_key="test-key",
            api={"max_retries": 3, "retry_min_wait": 0.01, "retry_max_wait": 0.02},
        )
        client = OpenRouterClient("deepseek-r1", settings)

        mock_response = MagicMock()
        err = RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )
        mock_response.status_code = 429
        mock_response.headers = {}

        with patch.object(client.client.chat.completions, "create", side_effect=err) as mock_create:
            with pytest.raises(RateLimitError):
                client.generate("test prompt")
            # Should have retried max_retries times
            assert mock_create.call_count == 3


# ---------------------------------------------------------------------------
# TestNewAnswerPatterns
# ---------------------------------------------------------------------------

class TestNewAnswerPatterns:
    """Test new answer extraction patterns added during hardening."""

    @pytest.mark.parametrize("text,expected", [
        ("Therefore B", "B"),
        ("therefore (A)", "A"),
        ("So the answer is C", "C"),
        ("The best answer is A", "A"),
        ("I believe the answer is D", "D"),
        ("Option D is correct", "D"),
        ("option c is correct", "C"),
    ])
    def test_new_patterns(self, text: str, expected: str):
        assert extract_answer("", "", text) == expected

    def test_sentence_final_letter(self):
        """A letter at the end of a sentence/text should be caught."""
        assert extract_answer("", "", "After analysis:\nB.") == "B"


# ---------------------------------------------------------------------------
# TestMultiBlockThinking
# ---------------------------------------------------------------------------

class TestMultiBlockThinking:
    """Test that multiple <think> blocks are all captured."""

    def test_two_think_blocks(self):
        text = "<think>First reasoning block</think>middle text<think>Second reasoning block</think>final answer"
        thinking, answer = split_thinking(text)
        assert "First reasoning block" in thinking
        assert "Second reasoning block" in thinking
        assert "middle text" in answer
        assert "final answer" in answer

    def test_two_pipe_thinking_blocks(self):
        text = "<|thinking|>Block A<|/thinking|>gap<|thinking|>Block B<|/thinking|>answer"
        thinking, _ = split_thinking(text)
        assert "Block A" in thinking
        assert "Block B" in thinking


# ---------------------------------------------------------------------------
# TestCostTracking
# ---------------------------------------------------------------------------

class TestCostTracking:
    """Test cost computation."""

    def test_compute_cost_basic(self):
        config = ModelConfig(
            name="test",
            api_model_id="test/test",
            alignment_method="grpo_rl",
            param_billions=1.0,
            cost_per_million_input=1.0,
            cost_per_million_output=2.0,
        )
        cost = _compute_cost(config, 1_000_000, 1_000_000)
        assert cost == pytest.approx(3.0)

    def test_compute_cost_zero_pricing(self):
        config = ModelConfig(
            name="test",
            api_model_id="test/test",
            alignment_method="grpo_rl",
            param_billions=1.0,
            cost_per_million_input=0.0,
            cost_per_million_output=0.0,
        )
        cost = _compute_cost(config, 500, 1000)
        assert cost == 0.0

    def test_compute_cost_fractional(self):
        config = ModelConfig(
            name="test",
            api_model_id="test/test",
            alignment_method="grpo_rl",
            param_billions=1.0,
            cost_per_million_input=0.70,
            cost_per_million_output=2.50,
        )
        cost = _compute_cost(config, 1000, 2000)
        expected = 0.70 * 1000 / 1_000_000 + 2.50 * 2000 / 1_000_000
        assert cost == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestConsecutiveFailureAbort
# ---------------------------------------------------------------------------

class TestConsecutiveFailureAbort:
    """Test consecutive failure tracking with abort."""

    def test_raises_at_threshold(self):
        tracker = _ConsecutiveFailureTracker(threshold=3)
        tracker.record_failure()
        tracker.record_failure()
        with pytest.raises(RuntimeError, match="3 consecutive failures"):
            tracker.record_failure()

    def test_reset_on_success(self):
        tracker = _ConsecutiveFailureTracker(threshold=3)
        tracker.record_failure()
        tracker.record_failure()
        tracker.record_success()
        # Should not raise — counter was reset
        tracker.record_failure()
        tracker.record_failure()
        with pytest.raises(RuntimeError):
            tracker.record_failure()

    def test_no_raise_below_threshold(self):
        tracker = _ConsecutiveFailureTracker(threshold=5)
        for _ in range(4):
            tracker.record_failure()
        # Should not raise at 4 with threshold 5


# ---------------------------------------------------------------------------
# TestBudgetExceededError
# ---------------------------------------------------------------------------

class TestBudgetExceededError:
    """Test that BudgetExceededError is a proper exception."""

    def test_is_exception(self):
        err = BudgetExceededError("over budget")
        assert isinstance(err, Exception)
        assert str(err) == "over budget"
