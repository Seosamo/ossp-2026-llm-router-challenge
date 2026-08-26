# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""LLM router: picks one of MODELS per query to trade off accuracy against token cost.

See 기획안 수정 및 실행 계획 (the planning doc this package implements) and
router/README.md for the section-by-section mapping.
"""
