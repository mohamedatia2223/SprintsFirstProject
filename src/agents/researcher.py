"""
Agent 1
"""

import sys
import os
import logging
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.database.retrieval import search_passages
from src.core.llm import get_llm, safe_llm_invoke, extract_text_content
from src.core.prompts import RESEARCHER_SYSTEM_PROMPT, STANDARD_REFUSAL_MESSAGE
from src.core.schemas import ResearchOutput, SourcePassage

logger = logging.getLogger(__name__)


def research_question(
    question: str, 
    reviewer_feedback: str = "", 
    top_k: int = 5,
    score_threshold: float = 0.4
) -> ResearchOutput:

    if not question or not question.strip():
        return ResearchOutput(
            draft_answer=STANDARD_REFUSAL_MESSAGE,
            sources=[],
            is_refusal=True
        )

    logger.info(f"[Agent 1: Researcher] Searching context passages for: '{question[:50]}...'")
    passages = search_passages(query=question, top_k=top_k, score_threshold=score_threshold)

    if not passages:
        logger.info("[Agent 1: Researcher] No relevant context passages found in Qdrant. Refusing to answer.")
        return ResearchOutput(
            draft_answer=STANDARD_REFUSAL_MESSAGE,
            sources=[],
            is_refusal=True
        )

    formatted_context_list = []
    for idx, p in enumerate(passages, start=1):
        formatted_context_list.append(
            f"Passage [{idx}] (Page {p.page_number} | Section: {p.section_title}):\n\"{p.text}\""
        )
    formatted_context = "\n\n".join(formatted_context_list)

    feedback_text = reviewer_feedback if reviewer_feedback else "None (Initial draft)"

    prompt = RESEARCHER_SYSTEM_PROMPT.format(
        context_passages=formatted_context,
        reviewer_feedback=feedback_text,
        user_question=question
    )

    logger.info("[Agent 1: Researcher] Invoking LLM to draft answer...")
    llm = get_llm()
    response = safe_llm_invoke(llm, prompt)
    draft_text = extract_text_content(response).strip()

    is_refusal = STANDARD_REFUSAL_MESSAGE.lower() in draft_text.lower() or "cannot answer" in draft_text.lower()

    logger.info(f"[Agent 1: Researcher] Draft generated ({len(draft_text)} chars, Refusal={is_refusal}).")
    return ResearchOutput(
        draft_answer=draft_text,
        sources=passages,
        is_refusal=is_refusal
    )

