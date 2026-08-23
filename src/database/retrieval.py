import sys
import os
import json
import logging
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.database.connection import get_qdrant_client
from src.core.embedding import embed_query
from src.core.schemas import SourcePassage
from src.core.llm import get_llm, safe_llm_invoke, extract_text_content

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "rich_dad_poor_dad"


def rerank_passages(query: str, candidate_passages: List[SourcePassage], top_k: int = 5) -> List[SourcePassage]:

    if not candidate_passages:
        return []

    if len(candidate_passages) <= top_k:
        return sorted(candidate_passages, key=lambda p: p.score, reverse=True)

    passages_text_formatted = []
    for idx, p in enumerate(candidate_passages):
        passages_text_formatted.append(f"[{idx + 1}] (Page {p.page_number} | Section: {p.section_title}):\n{p.text}")
    
    formatted_candidates = "\n\n".join(passages_text_formatted)

    prompt = f"""You are an expert search reranker. Score the semantic relevance of each candidate passage relative to the user query on a scale from 0.0 (completely irrelevant) to 1.0 (highly relevant and directly answers the query).

User Query: "{query}"

Candidate Passages:
{formatted_candidates}

Output ONLY a raw JSON object mapping passage index integer (1 to {len(candidate_passages)}) to a float relevance score between 0.0 and 1.0. Example:
{{"scores": {{"1": 0.95, "2": 0.15, "3": 0.80}}}}
Do NOT include any markdown formatting, explanation, or code blocks outside the JSON."""

    try:
        logger.info(f"Reranking {len(candidate_passages)} candidate passages using LLM Cross-Encoder...")
        llm = get_llm()
        response = safe_llm_invoke(llm, prompt)
        raw_text = extract_text_content(response).strip()
        
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        
        data = json.loads(raw_text)
        scores_dict = data.get("scores", {})

        for idx, passage in enumerate(candidate_passages, start=1):
            str_idx = str(idx)
            if str_idx in scores_dict:
                reranked_score = float(scores_dict[str_idx])
                passage.score = round(reranked_score, 4)
            elif idx in scores_dict:
                reranked_score = float(scores_dict[idx])
                passage.score = round(reranked_score, 4)

        reranked_passages = sorted(candidate_passages, key=lambda p: p.score, reverse=True)
        return reranked_passages[:top_k]

    except Exception as e:
        logger.warning(f"LLM Reranking failed due to: {e}. Falling back to Qdrant vector similarity ranking.")
        sorted_passages = sorted(candidate_passages, key=lambda p: p.score, reverse=True)
        return sorted_passages[:top_k]


def search_passages(
    query: str, 
    top_k: int = 5, 
    candidate_limit: int = 15,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    score_threshold: float = 0.3
) -> List[SourcePassage]:

    if not query or not query.strip():
        return []

    client = get_qdrant_client()
    
    logger.info(f"Generating query embedding for: '{query[:50]}...'")
    query_vector = embed_query(query)

    logger.info(f"Stage 1: Over-retrieving candidate points from Qdrant '{collection_name}' (candidate_limit={candidate_limit})...")
    res = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=candidate_limit
    )

    candidate_passages = []
    for point in res.points:
        score = float(point.score)
        if score < score_threshold:
            continue
            
        payload = point.payload or {}
        passage = SourcePassage(
            text=payload.get("text", ""),
            page_number=payload.get("page_number", 0),
            section_title=payload.get("section_title", "General"),
            score=score
        )
        candidate_passages.append(passage)

    logger.info(f"Retrieved {len(candidate_passages)} candidate passages above score threshold {score_threshold}.")

    if not candidate_passages:
        return []

    logger.info(f"Stage 2: Reranking {len(candidate_passages)} candidate passages to select top {top_k}...")
    reranked_passages = rerank_passages(query=query, candidate_passages=candidate_passages, top_k=top_k)
    
    logger.info(f"Returning top {len(reranked_passages)} reranked passages.")
    return reranked_passages

