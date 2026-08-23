

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional


LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)

APP_LOG_PATH = os.path.join(LOGS_DIR, "app.log")
TEST_RUNS_JSON_PATH = os.path.join(LOGS_DIR, "test_runs.json")


logger = logging.getLogger("sprints_assistant")
logger.setLevel(logging.INFO)

if not logger.handlers:
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


def log_test_case(
    test_id: int,
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    draft_answer: str,
    reviewer_verdict: str,
    reviewer_feedback: str,
    final_output: str,
    iterations: int = 1,
    is_refusal: bool = False
) -> Dict[str, Any]:

    entry = {
        "test_id": test_id,
        "timestamp": datetime.now().isoformat(),
        "retrieval_question": question,
        "chunks_got": retrieved_chunks,
        "draft_answer": draft_answer,
        "reviewer_verdict": reviewer_verdict,
        "reviewer_feedback": reviewer_feedback,
        "final_output": final_output,
        "iterations": iterations,
        "is_refusal": is_refusal
    }

    existing_entries = []
    if os.path.exists(TEST_RUNS_JSON_PATH):
        try:
            with open(TEST_RUNS_JSON_PATH, "r", encoding="utf-8") as f:
                existing_entries = json.load(f)
        except Exception:
            existing_entries = []

    existing_entries.append(entry)

    with open(TEST_RUNS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_entries, f, indent=2, ensure_ascii=False)

    logger.info(f"Logged Test Case #{test_id}: '{question[:40]}...' -> Verdict: {reviewer_verdict}")
    return entry
