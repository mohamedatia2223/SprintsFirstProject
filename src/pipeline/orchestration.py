
import sys
import os
import logging
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langgraph.graph import StateGraph, END
from src.agents.researcher import research_question
from src.agents.reviewer import review_draft_answer
from src.core.schemas import AgentState, SourcePassage
from src.core.logger import log_test_case

from src.core.prompts import STANDARD_REFUSAL_MESSAGE

logger = logging.getLogger(__name__)


def researcher_node(state: Dict[str, Any]) -> Dict[str, Any]:

    question = state["question"]
    reviewer_feedback = state.get("reviewer_feedback", "")
    iteration_count = state.get("iteration_count", 0) + 1

    logger.info(f"[Orchestrator] Running Researcher Node (Iteration {iteration_count})...")
    research_res = research_question(question=question, reviewer_feedback=reviewer_feedback)

    return {
        **state,
        "passages": [p.model_dump() if hasattr(p, "model_dump") else p for p in research_res.sources],
        "draft_answer": research_res.draft_answer,
        "is_refusal": research_res.is_refusal,
        "iteration_count": iteration_count
    }


def reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    draft_answer = state["draft_answer"]
    raw_passages = state.get("passages", [])
    passages = [SourcePassage(**p) if isinstance(p, dict) else p for p in raw_passages]

    if state.get("is_refusal", False):
        logger.info("[Orchestrator] Draft is a refusal. Auto-approving refusal response.")
        return {
            **state,
            "reviewer_verdict": "APPROVED",
            "reviewer_feedback": "",
            "final_answer": draft_answer
        }

    logger.info("[Orchestrator] Running Reviewer Node...")
    review_res = review_draft_answer(question=question, draft_answer=draft_answer, passages=passages)

    final_ans = draft_answer if review_res.verdict == "APPROVED" else ""

    return {
        **state,
        "reviewer_verdict": review_res.verdict,
        "reviewer_feedback": review_res.feedback_for_researcher,
        "final_answer": final_ans
    }


def should_continue(state: Dict[str, Any]) -> str:
    verdict = state.get("reviewer_verdict", "PENDING")
    iteration_count = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 2)

    if verdict == "APPROVED" or state.get("is_refusal", False):
        logger.info("[Orchestrator] Decision: APPROVED -> Ending workflow.")
        return END

    if iteration_count >= max_iterations:
        logger.info(f"[Orchestrator] Decision: Max iterations ({max_iterations}) reached -> Ending workflow.")
        return END

    logger.info(f"[Orchestrator] Decision: REJECTED -> Handoff feedback to Researcher for revision.")
    return "researcher"


def build_orchestration_graph():
    builder = StateGraph(Dict[str, Any])

    builder.add_node("researcher", researcher_node)
    builder.add_node("reviewer", reviewer_node)

    builder.set_entry_point("researcher")

    builder.add_edge("researcher", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "researcher": "researcher",
            END: END
        }
    )

    return builder.compile()


orchestrator_graph = build_orchestration_graph()


def run_assistant(question: str, max_iterations: int = 2) -> AgentState:
    validated_max_iterations = max(1, max_iterations)

    initial_state = {
        "question": question,
        "passages": [],
        "draft_answer": "",
        "reviewer_verdict": "PENDING",
        "reviewer_feedback": "",
        "iteration_count": 0,
        "max_iterations": validated_max_iterations,
        "final_answer": "",
        "is_refusal": False
    }

    logger.info(f"\n==========================================")
    logger.info(f"[Orchestrator] Starting workflow for question: '{question}'")
    logger.info(f"==========================================")

    final_state_dict = orchestrator_graph.invoke(initial_state)

    verdict = final_state_dict.get("reviewer_verdict", "PENDING")
    is_refusal = final_state_dict.get("is_refusal", False)

    if is_refusal:
        final_ans = final_state_dict.get("final_answer") or final_state_dict.get("draft_answer") or STANDARD_REFUSAL_MESSAGE
    elif verdict == "APPROVED":
        final_ans = final_state_dict.get("final_answer") or final_state_dict.get("draft_answer") or ""
    else:
        logger.warning("[Orchestrator] Reviewer rejected final draft after max iterations. Suppressing draft answer.")
        final_ans = STANDARD_REFUSAL_MESSAGE
        is_refusal = True

    passages_list = [SourcePassage(**p) if isinstance(p, dict) else p for p in final_state_dict.get("passages", [])]

    agent_state = AgentState(
        question=question,
        passages=passages_list,
        draft_answer=final_state_dict.get("draft_answer", ""),
        reviewer_verdict=verdict,
        reviewer_feedback=final_state_dict.get("reviewer_feedback", ""),
        iteration_count=final_state_dict.get("iteration_count", 1),
        max_iterations=validated_max_iterations,
        final_answer=final_ans,
        is_refusal=is_refusal
    )

    chunks_got = [
        {"page_number": p.page_number, "section_title": p.section_title, "text": p.text[:150]}
        for p in agent_state.passages
    ]
    log_test_case(
        test_id=1,
        question=question,
        retrieved_chunks=chunks_got,
        draft_answer=agent_state.draft_answer,
        reviewer_verdict=agent_state.reviewer_verdict,
        reviewer_feedback=agent_state.reviewer_feedback,
        final_output=agent_state.final_answer,
        iterations=agent_state.iteration_count,
        is_refusal=agent_state.is_refusal
    )

    return agent_state

