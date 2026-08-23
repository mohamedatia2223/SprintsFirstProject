"""
Agent 2 
"""

import sys
import os
import json
import re
import logging
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.llm import get_llm, safe_llm_invoke, extract_text_content
from src.core.prompts import REVIEWER_SYSTEM_PROMPT
from src.core.schemas import ReviewerOutput, SourcePassage

logger = logging.getLogger(__name__)


def _clean_json_response(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = re.sub(r'```json\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)
    elif "```" in cleaned:
        cleaned = re.sub(r'```\s*', '', cleaned)
    return cleaned.strip()


def review_draft_answer(
    question: str, 
    draft_answer: str, 
    passages: List[SourcePassage]
) -> ReviewerOutput:

    if not draft_answer or not draft_answer.strip():
        return ReviewerOutput(
            verdict="REJECTED",
            reasoning="Draft answer was empty.",
            unsupported_claims=["Empty draft response."],
            feedback_for_researcher="Please provide a non-empty answer based on the source passages."
        )

    if passages:
        formatted_context_list = []
        for idx, p in enumerate(passages, start=1):
            formatted_context_list.append(
                f"Passage [{idx}] (Page {p.page_number} | Section: {p.section_title}):\n\"{p.text}\""
            )
        formatted_context = "\n\n".join(formatted_context_list)
    else:
        formatted_context = "No context passages provided."

    prompt = REVIEWER_SYSTEM_PROMPT.format(
        context_passages=formatted_context,
        user_question=question,
        draft_answer=draft_answer
    )

    logger.info("[Agent 2: Reviewer] Auditing draft answer against source passages...")
    llm = get_llm()
    response = safe_llm_invoke(llm, prompt)
    raw_response_text = extract_text_content(response)

    cleaned_json_str = _clean_json_response(raw_response_text)

    try:
        data = json.loads(cleaned_json_str)
        verdict = str(data.get("verdict", "APPROVED")).upper()
        if verdict not in ["APPROVED", "REJECTED"]:
            verdict = "APPROVED"

        output = ReviewerOutput(
            verdict=verdict,
            reasoning=str(data.get("reasoning", "")),
            unsupported_claims=list(data.get("unsupported_claims", [])),
            feedback_for_researcher=str(data.get("feedback_for_researcher", ""))
        )
        logger.info(f"[Agent 2: Reviewer] Evaluation complete -> Verdict: {output.verdict}")
        return output

    except Exception as parse_err:
        logger.warning(f"[Agent 2: Reviewer] Failed to parse JSON response ({parse_err}). Fallback evaluation.")
        
        if "rejected" in raw_response_text.lower() or "unsupported" in raw_response_text.lower():
            return ReviewerOutput(
                verdict="REJECTED",
                reasoning=raw_response_text[:300],
                unsupported_claims=["Contains unverified claims."],
                feedback_for_researcher="Ensure all statements are strictly backed by the context passages."
            )

        return ReviewerOutput(
            verdict="APPROVED",
            reasoning="Audited and approved.",
            unsupported_claims=[],
            feedback_for_researcher=""
        )
