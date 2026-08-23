import sys
import os
import logging
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.database.connection import get_qdrant_client
from src.core.embedding import embed_query
from src.core.schemas import SourcePassage

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "rich_dad_poor_dad"


def search_passages(
    query: str, 
    top_k: int = 5, 
    collection_name: str = DEFAULT_COLLECTION_NAME,
    score_threshold: float = 0.4
) -> List[SourcePassage]:
    if not query or not query.strip():
        return []

    client = get_qdrant_client()
    
    logger.info(f"Generating query embedding for: '{query[:50]}...'")
    query_vector = embed_query(query)

    logger.info(f"Searching Qdrant collection '{collection_name}' (top_k={top_k})...")
    res = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )

    passages = []
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
        passages.append(passage)

    logger.info(f"Retrieved {len(passages)} passages above score threshold {score_threshold}.")
    return passages
