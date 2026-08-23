from google import genai
from src.core.config import LLM_API_KEY
from src.core.llm import FALLBACK_CHAIN, PRIMARY_MODEL, LlmApiException
import time
import logging

logger = logging.getLogger(__name__)
client = genai.Client(api_key=LLM_API_KEY)

def _invoke_voice_model(model_name, contents):
    return client.models.generate_content(
        model=model_name,
        contents=contents
    )

def convert_audio_text(file_path: str):
    audio_file = client.files.upload(file=file_path)
    
    while audio_file.state.name == "PROCESSING":
        time.sleep(1)
        audio_file = client.files.get(name=audio_file.name)

    if audio_file.state.name == "FAILED":
        raise Exception("Audio file processing failed on Google AI")

    contents = [
        "Please transcribe the following audio exactly as spoken. "
        "If it is in Arabic, provide the transcript in Arabic. "
        "Return ONLY the transcript text.",
        audio_file
    ]

    current_model = PRIMARY_MODEL
    try:
        response = _invoke_voice_model(current_model, contents)
    except Exception as e:
        error_str = str(e).lower()
        high_demand_keywords = [
            "429", "503", "quota", "exhausted", "rate limit", 
            "overloaded", "high demand", "capacity", "service unavailable", 
            "demand", "temporarily"
        ]
        if any(keyword in error_str for keyword in high_demand_keywords):
            try:
                start_idx = FALLBACK_CHAIN.index(current_model)
            except ValueError:
                start_idx = 0
            
            for next_model in FALLBACK_CHAIN[start_idx + 1:]:
                logger.warning(
                    f"Voice model '{current_model}' failed due to rate limits/overload. "
                    f"Attempting failover to fallback model '{next_model}'..."
                )
                try:
                    response = _invoke_voice_model(next_model, contents)
                    break
                except Exception as fallback_err:
                    logger.warning(f"Fallback voice model '{next_model}' also failed: {fallback_err}")
                    current_model = next_model
                    e = fallback_err
            else:
                client.files.delete(name=audio_file.name)
                raise LlmApiException(
                    "All primary and fallback AI services are currently overloaded. Please try again later."
                ) from e
        elif "token" in error_str or "maximum context length" in error_str:
            client.files.delete(name=audio_file.name)
            raise LlmApiException("The audio exceeded the maximum processing limit.") from e
        elif any(auth_kw in error_str for auth_kw in ["401", "403", "invalid api key", "unauthorized", "forbidden"]):
            client.files.delete(name=audio_file.name)
            raise LlmApiException("Authentication issue with the AI service. Please check API credentials.") from e
        else:
            client.files.delete(name=audio_file.name)
            raise LlmApiException(f"An unexpected AI service error occurred (check the API): {str(e)}") from e

    client.files.delete(name=audio_file.name)

    return response.text.strip()
