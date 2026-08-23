import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.config import LLM_API_KEY
from src.core.llm import LlmApiException

logger = logging.getLogger(__name__)

PRIMARY_EMBEDDING_MODEL = "models/gemini-embedding-001"
FALLBACK_EMBEDDING_MODEL = "models/gemini-embedding-2"

FALLBACK_EMBEDDING_CHAIN = [PRIMARY_EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL]


def get_embedding_model(model_name: str = PRIMARY_EMBEDDING_MODEL) -> GoogleGenerativeAIEmbeddings:

    if not LLM_API_KEY:
        raise LlmApiException("LLM_API_KEY is not set in configuration environment variables.")

    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=LLM_API_KEY
    )


def embed_query(text: str, model_name: str = PRIMARY_EMBEDDING_MODEL) -> list[float]:

    if not text or not text.strip():
        raise ValueError("Text for embedding cannot be empty.")

    current_model = model_name
    try:
        embeddings_model = get_embedding_model(model_name=current_model)
        return embeddings_model.embed_query(text)
    except Exception as e:
        error_str = str(e).lower()
        high_demand_keywords = [
            "429", "503", "quota", "exhausted", "rate limit", 
            "overloaded", "high demand", "capacity", "service unavailable", 
            "temporarily"
        ]
        if any(keyword in error_str for keyword in high_demand_keywords):
            try:
                start_idx = FALLBACK_EMBEDDING_CHAIN.index(current_model)
            except ValueError:
                start_idx = 0

            for next_model in FALLBACK_EMBEDDING_CHAIN[start_idx + 1:]:
                logger.warning(
                    f"Embedding model '{current_model}' failed due to rate limits/overload. "
                    f"Attempting failover to fallback model '{next_model}'..."
                )
                try:
                    fallback_model = get_embedding_model(model_name=next_model)
                    return fallback_model.embed_query(text)
                except Exception as fallback_err:
                    logger.warning(f"Fallback embedding model '{next_model}' also failed: {fallback_err}")
                    current_model = next_model
                    e = fallback_err
            else:
                raise LlmApiException(
                    "All primary and fallback embedding services are currently overloaded. Please try again later."
                ) from e
        elif "token" in error_str or "maximum context length" in error_str:
            raise LlmApiException("The text exceeded the maximum token limit for embeddings.") from e
        elif any(auth_kw in error_str for auth_kw in ["401", "403", "invalid api key", "unauthorized", "forbidden"]):
            raise LlmApiException("Authentication issue with the embedding service. Please check API credentials.") from e
        else:
            raise LlmApiException(f"An unexpected embedding service error occurred: {str(e)}") from e


def embed_documents(texts: list[str], model_name: str = PRIMARY_EMBEDDING_MODEL) -> list[list[float]]:

    if not texts:
        return []

    current_model = model_name
    try:
        embeddings_model = get_embedding_model(model_name=current_model)
        return embeddings_model.embed_documents(texts)
    except Exception as e:
        error_str = str(e).lower()
        high_demand_keywords = [
            "429", "503", "quota", "exhausted", "rate limit", 
            "overloaded", "high demand", "capacity", "service unavailable", 
            "temporarily"
        ]
        if any(keyword in error_str for keyword in high_demand_keywords):
            try:
                start_idx = FALLBACK_EMBEDDING_CHAIN.index(current_model)
            except ValueError:
                start_idx = 0

            for next_model in FALLBACK_EMBEDDING_CHAIN[start_idx + 1:]:
                logger.warning(
                    f"Embedding model '{current_model}' failed due to rate limits/overload. "
                    f"Attempting failover to fallback model '{next_model}'..."
                )
                try:
                    fallback_model = get_embedding_model(model_name=next_model)
                    return fallback_model.embed_documents(texts)
                except Exception as fallback_err:
                    logger.warning(f"Fallback embedding model '{next_model}' also failed: {fallback_err}")
                    current_model = next_model
                    e = fallback_err
            else:
                raise LlmApiException(
                    "All primary and fallback embedding services are currently overloaded. Please try again later."
                ) from e
        elif "token" in error_str or "maximum context length" in error_str:
            raise LlmApiException("The documents exceeded the maximum token limit for embeddings.") from e
        elif any(auth_kw in error_str for auth_kw in ["401", "403", "invalid api key", "unauthorized", "forbidden"]):
            raise LlmApiException("Authentication issue with the embedding service. Please check API credentials.") from e
        else:
            raise LlmApiException(f"An unexpected embedding service error occurred: {str(e)}") from e


