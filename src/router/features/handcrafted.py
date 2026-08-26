# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Length / structural / instruction hand features (§5.1): pure functions of a
single raw (pre-truncation) string -- these are extracted from the full text even
though the embedding branch truncates at 512 tokens (§5.1 필수 준수사항 #3: length/
structure features are meant to cover exactly the information the embedding
branch's truncation would otherwise lose).

NOTE on a discrepancy in the source doc: §4's pipeline diagram says "길이/구조/
지시어 피처 ~20" but also separately claims "~34" features overall elsewhere in the
same document, while §5.1's own breakdown (6 length + 8 structural + 6
instruction-keyword = 20) sums to 20, not 34. This module implements the 20-feature
breakdown that matches §5.1's explicit table; the "~34" figure could not be
reconciled against any table in the doc and should be re-confirmed against the
competition spec once available, not assumed away.
"""

from __future__ import annotations

import re
from typing import Dict, List

_SENTENCE_SPLIT_RE = re.compile(r"[.!?。！？]+")
_URL_RE = re.compile(r"https?://\S+")
_CODE_BLOCK_RE = re.compile(r"```")
_BULLET_RE = re.compile(r"^\s*[-*•]", re.MULTILINE)
_BRACKET_RE = re.compile(r"[\[\](){}]")

_INSTRUCTION_KEYWORDS = {
    "has_step_by_step_kw": ("단계별", "step by step", "step-by-step"),
    "has_detailed_kw": ("자세히", "상세히", "in detail", "detailed"),
    "has_simple_kw": ("간단히", "간단하게", "simply", "briefly"),
    "has_summarize_kw": ("요약", "summarize", "summary"),
    "has_short_form_kw": ("문장으로", "words or less", "in one sentence"),
    "has_translate_kw": ("번역", "translate"),
}

FEATURE_NAMES: List[str] = [
    # length (6)
    "char_len",
    "word_len",
    "avg_word_len",
    "num_lines",
    "num_sentences",
    "digit_ratio",
    # structural (8)
    "num_code_blocks",
    "num_bullet_points",
    "num_urls",
    "uppercase_ratio",
    "punctuation_ratio",
    "bracket_count",
    "special_char_ratio",
    "question_mark_count",
    # instruction keywords (6)
    *_INSTRUCTION_KEYWORDS.keys(),
]


def extract_handcrafted(text: str) -> Dict[str, float]:
    char_len = len(text)
    words = text.split()
    word_len = len(words)
    avg_word_len = (sum(len(w) for w in words) / word_len) if word_len else 0.0
    lines = text.split("\n")
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    digits = sum(c.isdigit() for c in text)

    upper = sum(c.isupper() for c in text)
    punctuation = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in ".,!?")

    features: Dict[str, float] = {
        "char_len": float(char_len),
        "word_len": float(word_len),
        "avg_word_len": float(avg_word_len),
        "num_lines": float(len(lines)),
        "num_sentences": float(len(sentences)),
        "digit_ratio": float(digits / char_len) if char_len else 0.0,
        "num_code_blocks": float(len(_CODE_BLOCK_RE.findall(text)) // 2),
        "num_bullet_points": float(len(_BULLET_RE.findall(text))),
        "num_urls": float(len(_URL_RE.findall(text))),
        "uppercase_ratio": float(upper / char_len) if char_len else 0.0,
        "punctuation_ratio": float(punctuation / char_len) if char_len else 0.0,
        "bracket_count": float(len(_BRACKET_RE.findall(text))),
        "special_char_ratio": float(special / char_len) if char_len else 0.0,
        "question_mark_count": float(text.count("?") + text.count("?")),
    }

    lowered = text.lower()
    for name, keywords in _INSTRUCTION_KEYWORDS.items():
        features[name] = float(any(kw in text or kw.lower() in lowered for kw in keywords))

    return features
