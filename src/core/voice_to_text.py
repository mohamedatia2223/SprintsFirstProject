import sys
import os
import time
import logging
from google import genai
from src.core.config import LLM_API_KEY
from src.core.llm import FALLBACK_CHAIN, PRIMARY_MODEL
from src.core.exceptions import LlmApiException, handle_api_exception

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logger = logging.getLogger(__name__)

client = genai.Client(api_key=LLM_API_KEY)

def _invoke_voice_model(model_name, contents):
    return client.models.generate_content(
        model=model_name,
        contents=contents
    )

def convert_audio_text(file_path: str):
    audio_file = client.files.upload(file=file_path)

    try:
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
                    raise handle_api_exception(e, service_name="voice-to-text service", is_overloaded=True) from e
            
            raise handle_api_exception(e, service_name="voice-to-text service") from e

        return response.text.strip()
    finally:
        client.files.delete(name=audio_file.name)
