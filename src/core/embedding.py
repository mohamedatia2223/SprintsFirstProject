import sys
import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.config import LLM_API_KEY
from src.core.exceptions import LlmApiException, handle_api_exception

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logger = logging.getLogger(__name__)

PRIMARY_EMBEDDING_MODEL = "models/gemini-embedding-001"

def get_embedding_model(model_name: str = PRIMARY_EMBEDDING_MODEL) -> GoogleGenerativeAIEmbeddings:

    if not LLM_API_KEY:
        raise LlmApiException("LLM_API_KEY is not set in configuration environment variables.")

    return GoogleGenerativeAIEmbeddings(
        model=PRIMARY_EMBEDDING_MODEL,
        google_api_key=LLM_API_KEY
    )


def embed_query(text: str, model_name: str = PRIMARY_EMBEDDING_MODEL) -> list[float]:

    if not text or not text.strip():
        raise ValueError("Text for embedding cannot be empty.")

    try:
        embeddings_model = get_embedding_model(model_name=PRIMARY_EMBEDDING_MODEL)
        return embeddings_model.embed_query(text)
    except Exception as e:
        raise handle_api_exception(e, service_name="embedding service") from e


def embed_documents(texts: list[str], model_name: str = PRIMARY_EMBEDDING_MODEL) -> list[list[float]]:

    if not texts:
        return []

    for attempt in range(5):
        try:
            embeddings_model = get_embedding_model(model_name=PRIMARY_EMBEDDING_MODEL)
            return embeddings_model.embed_documents(texts)
        except Exception as e:
            error_str = str(e).lower()
            high_demand_keywords = [
                "429", "503", "quota", "exhausted", "rate limit", 
                "overloaded", "high demand", "capacity", "service unavailable", 
                "temporarily", "resource_exhausted"
            ]
            if any(keyword in error_str for keyword in high_demand_keywords):
                wait_seconds = (attempt + 1) * 15
                if attempt < 4:
                    logger.warning(f"Rate limit / quota hit (429). Pausing for {wait_seconds}s before retry attempt {attempt + 2}/5...")
                    import time
                    time.sleep(wait_seconds)
                    continue
                else:
                    raise handle_api_exception(e, service_name="embedding service", is_overloaded=True) from e

            raise handle_api_exception(e, service_name="embedding service") from e


