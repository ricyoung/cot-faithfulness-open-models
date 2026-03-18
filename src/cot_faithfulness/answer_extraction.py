"""
Extract the final answer (A/B/C/D) from model output.

Strategy (priority order, answer_text first, then full_response fallback):
1. Explicit ANSWER_PATTERN in answer_text
2. THEREFORE_PATTERN in answer_text
3. OPTION_CORRECT_PATTERN in answer_text
4. Boxed answer (\\boxed{C}) in answer_text
5. SENTENCE_FINAL_LETTER_PATTERN in answer_text
6. Last bare letter in answer_text
7-10. Same patterns in full_response as fallback
11. Standalone letter pattern
12. Thinking text tail (last resort)
13. Return None
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Explicit answer patterns — [*\s]* handles markdown bold (**Answer:** B)
ANSWER_PATTERN = re.compile(
    r"(?:the\s+answer\s+is|answer\s*:[*\s]*|final\s+answer\s*:?[*\s]*|"
    r"my\s+answer\s*:?[*\s]*|correct\s+answer\s+is|"
    r"i(?:'ll)?\s+(?:choose|select|go\s+with|pick))\s*\(?([A-Da-d])\)?",
    re.IGNORECASE,
)

# LaTeX boxed answer
BOXED_PATTERN = re.compile(r"\\boxed\{?\s*\(?([A-Da-d])\)?\s*\}?")

# Bare letter in parentheses
BARE_LETTER_PATTERN = re.compile(r"\(([A-Da-d])\)")

# "Therefore B", "So the answer is C", "The best answer is A", "I believe the answer is D"
THEREFORE_PATTERN = re.compile(
    r"(?:therefore|so\s+the\s+answer\s+is|the\s+best\s+answer\s+is|"
    r"i\s+believe\s+the\s+answer\s+is)\s*\(?([A-Da-d])\)?",
    re.IGNORECASE,
)

# "Option B is correct"
OPTION_CORRECT_PATTERN = re.compile(
    r"[Oo]ption\s+([A-Da-d])\s+is\s+correct",
)

# Sentence-final standalone letter: "B." at end of text or on its own line
SENTENCE_FINAL_LETTER_PATTERN = re.compile(
    r"(?:^|[\s\n])\(?([A-Da-d])\)?\.?\s*$",
    re.MULTILINE,
)

# Standalone letter on its own or as "**B**" at start/end
STANDALONE_LETTER_PATTERN = re.compile(r"(?:^|\n)\s*\*{0,2}\(?([A-Da-d])\)?\*{0,2}\s*(?:\n|$)")


def extract_answer(
    full_response: str,
    thinking_text: str,
    answer_text: str,
) -> Optional[str]:
    """
    Extract final answer (A/B/C/D) from model output.
    Searches answer_text first (post-thinking), then full_response as fallback.
    """
    # 1. Explicit ANSWER_PATTERN in answer_text
    match = ANSWER_PATTERN.search(answer_text)
    if match:
        return match.group(1).upper()

    # 2. THEREFORE_PATTERN in answer_text
    match = THEREFORE_PATTERN.search(answer_text)
    if match:
        return match.group(1).upper()

    # 3. OPTION_CORRECT_PATTERN in answer_text
    match = OPTION_CORRECT_PATTERN.search(answer_text)
    if match:
        return match.group(1).upper()

    # 4. Boxed in answer_text
    match = BOXED_PATTERN.search(answer_text)
    if match:
        return match.group(1).upper()

    # 5. SENTENCE_FINAL_LETTER_PATTERN in answer_text
    match = SENTENCE_FINAL_LETTER_PATTERN.search(answer_text)
    if match:
        return match.group(1).upper()

    # 6. Last bare letter in answer_text
    letters = BARE_LETTER_PATTERN.findall(answer_text)
    if letters:
        return letters[-1].upper()

    # 7. Fallback: ANSWER_PATTERN in full_response
    match = ANSWER_PATTERN.search(full_response)
    if match:
        return match.group(1).upper()

    # 8. Fallback: THEREFORE_PATTERN in full_response
    match = THEREFORE_PATTERN.search(full_response)
    if match:
        return match.group(1).upper()

    # 9. Fallback: OPTION_CORRECT_PATTERN in full_response
    match = OPTION_CORRECT_PATTERN.search(full_response)
    if match:
        return match.group(1).upper()

    # 10. Fallback: last bare letter in full response
    letters = BARE_LETTER_PATTERN.findall(full_response)
    if letters:
        return letters[-1].upper()

    # 11. Fallback: standalone letter pattern in answer_text or full_response
    for text in [answer_text, full_response]:
        match = STANDALONE_LETTER_PATTERN.search(text)
        if match:
            return match.group(1).upper()

    # 12. Last resort: check thinking_text for answer patterns
    # (model may state answer in reasoning when answer_text is empty due to truncation)
    if thinking_text:
        tail = thinking_text[-2000:]

        # 12a. Explicit answer pattern in thinking tail
        match = ANSWER_PATTERN.search(tail)
        if match:
            return match.group(1).upper()

        # 12b. Therefore pattern in thinking tail
        match = THEREFORE_PATTERN.search(tail)
        if match:
            return match.group(1).upper()

        # 12c. Option correct pattern in thinking tail
        match = OPTION_CORRECT_PATTERN.search(tail)
        if match:
            return match.group(1).upper()

        # 12d. Boxed answer in thinking tail
        match = BOXED_PATTERN.search(tail)
        if match:
            return match.group(1).upper()

        # 12e. Last bare letter in thinking tail
        letters = BARE_LETTER_PATTERN.findall(tail)
        if letters:
            return letters[-1].upper()

    # 13. Log debug before returning None
    logger.debug("No answer extracted from question output")
    return None


def extract_answer_for_visual_pattern(
    full_response: str,
    thinking_text: str,
    answer_text: str,
    reorder_mapping: dict[str, str],
) -> Optional[str]:
    """
    For visual_pattern hints: extract answer in reordered space,
    then translate back to original label space.

    reorder_mapping: original_label -> new_label (e.g., {"B": "A", "A": "B", ...})
    Returns the answer in the ORIGINAL label space.
    """
    reordered_answer = extract_answer(full_response, thinking_text, answer_text)
    if reordered_answer is None:
        return None

    # Reverse mapping: new_label -> original_label
    reverse_mapping = {v: k for k, v in reorder_mapping.items()}
    return reverse_mapping.get(reordered_answer)
