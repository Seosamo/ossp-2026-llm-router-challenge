# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Light preprocessing (§4): NFC normalize, collapse whitespace.

Pure single-string function -- never touches other rows in a batch, by
construction. This is part of the structural guarantee behind validate.py's B4
check (no batch-relative computation anywhere in the feature pipeline).
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RE.sub(" ", normalized).strip()
