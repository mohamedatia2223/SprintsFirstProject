import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.config import LLM_API_KEY
from src.core.llm import LlmApiException

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
        error_str = str(e).lower()
        if "token" in error_str or "maximum context length" in error_str:
            raise LlmApiException("The text exceeded the maximum token limit for embeddings.") from e
        elif any(auth_kw in error_str for auth_kw in ["401", "403", "invalid api key", "unauthorized", "forbidden"]):
            raise LlmApiException("Authentication issue with the embedding service. Please check API credentials.") from e
        else:
            raise LlmApiException(f"An unexpected embedding service error occurred: {str(e)}") from e


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
                    raise LlmApiException(
                        "Embedding service is currently overloaded or rate limited. Please try again later."
                    ) from e
            elif "token" in error_str or "maximum context length" in error_str:
                raise LlmApiException("The documents exceeded the maximum token limit for embeddings.") from e
            elif any(auth_kw in error_str for auth_kw in ["401", "403", "invalid api key", "unauthorized", "forbidden"]):
                raise LlmApiException("Authentication issue with the embedding service. Please check API credentials.") from e
            else:
                raise LlmApiException(f"An unexpected embedding service error occurred: {str(e)}") from e


