import logging
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from src.core.config import (
    LLM_API_KEY, 
    PRIMARY_MODEL as DEFAULT_PRIMARY,
    SECONDARY_MODEL as DEFAULT_SECONDARY,
    TERTIARY_MODEL as DEFAULT_TERTIARY
)

logger = logging.getLogger(__name__)

PRIMARY_MODEL = DEFAULT_PRIMARY
SECONDARY_MODEL = DEFAULT_SECONDARY
TERTIARY_MODEL = DEFAULT_TERTIARY

FALLBACK_CHAIN = [PRIMARY_MODEL, SECONDARY_MODEL, TERTIARY_MODEL]


def set_llm_models(primary: str = None, secondary: str = None, tertiary: str = None):

    global PRIMARY_MODEL, SECONDARY_MODEL, TERTIARY_MODEL, FALLBACK_CHAIN
    if primary:
        PRIMARY_MODEL = primary
    if secondary:
        SECONDARY_MODEL = secondary
    if tertiary:
        TERTIARY_MODEL = tertiary
        
    FALLBACK_CHAIN = [PRIMARY_MODEL, SECONDARY_MODEL, TERTIARY_MODEL]
    logger.info(f"Updated LLM Fallback Chain: {FALLBACK_CHAIN}")


def list_available_gemini_models() -> list[str]:

    if not LLM_API_KEY:
        return [PRIMARY_MODEL, SECONDARY_MODEL, TERTIARY_MODEL]
        
    try:
        client = genai.Client(api_key=LLM_API_KEY)
        models = [
            m.name.replace("models/", "") 
            for m in client.models.list() 
            if "gemini" in m.name.lower() and "embed" not in m.name.lower()
        ]
        return models if models else [PRIMARY_MODEL, SECONDARY_MODEL, TERTIARY_MODEL]
    except Exception as e:
        logger.warning(f"Could not fetch available models from Google API: {e}")
        return [PRIMARY_MODEL, SECONDARY_MODEL, TERTIARY_MODEL]


class LlmApiException(Exception):
    pass


def get_llm(model_name=None):
    target_model = model_name or PRIMARY_MODEL
    llm = ChatGoogleGenerativeAI(
        model=target_model,
        google_api_key=LLM_API_KEY,
        temperature=0.2,
        max_output_tokens=10000,
        max_retries=0
    )
    return llm

@retry(
    stop=stop_after_attempt(2),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def _raw_llm_invoke(llm, messages):
    return llm.invoke(messages)

def safe_llm_invoke(llm, messages):
    try:
        return _raw_llm_invoke(llm, messages)
    except Exception as e:
        error_str = str(e).lower()
        high_demand_keywords = [
            "429", "503", "quota", "exhausted", "rate limit", 
            "overloaded", "high demand", "capacity", "service unavailable", 
            "demand", "temporarily"
        ]
        if any(keyword in error_str for keyword in high_demand_keywords):
            current_model = getattr(llm, "model", PRIMARY_MODEL)
            try:
                start_idx = FALLBACK_CHAIN.index(current_model)
            except ValueError:
                start_idx = 0
            
            for next_model in FALLBACK_CHAIN[start_idx + 1:]:
                logger.warning(
                    f"Model '{current_model}' failed due to rate limits/overload. "
                    f"Attempting failover to fallback model '{next_model}'..."
                )
                try:
                    fallback_llm = get_llm(model_name=next_model)
                    return _raw_llm_invoke(fallback_llm, messages)
                except Exception as fallback_err:
                    logger.warning(f"Fallback model '{next_model}' also failed: {fallback_err}")
                    current_model = next_model
                    e = fallback_err
            
            raise LlmApiException(
                "All primary and fallback AI services are currently overloaded. Please try again later."
            ) from e
        elif "token" in error_str or "maximum context length" in error_str:
            raise LlmApiException("The response exceeded the maximum token limit. Please ask a more specific question.") from e
        elif any(auth_kw in error_str for auth_kw in ["401", "403", "invalid api key", "unauthorized", "forbidden"]):
            raise LlmApiException("Authentication issue with the AI service. Please check API credentials.") from e
        else:
            raise LlmApiException(f"An unexpected AI service error occurred (check the API): {str(e)}") from e

def extract_text_content(message_obj):

    content = message_obj.content
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            else:
                text_parts.append(str(part))
        return "".join(text_parts)
    
    return str(content)
