import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from google import genai
from google.genai import types
from src.core.config import LLM_API_KEY
from src.core.llm import FALLBACK_CHAIN, PRIMARY_MODEL, LlmApiException

logger = logging.getLogger(__name__)
client = genai.Client(api_key=LLM_API_KEY)


def _invoke_tts_model(model_name: str, text: str, voice_name: str = "Puck"):

    speech_config = None
    if voice_name:
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        )

    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=speech_config
    )

    return client.models.generate_content(
        model=model_name,
        contents=text,
        config=config
    )


def convert_text_to_audio(text: str, output_path: str = None, voice_name: str = "Puck") -> bytes:
    if not text or not text.strip():
        raise ValueError("Text prompt for text-to-voice conversion cannot be empty.")

    current_model = PRIMARY_MODEL
    response = None

    try:
        response = _invoke_tts_model(current_model, text, voice_name=voice_name)
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
                    f"Text-to-voice model '{current_model}' failed due to rate limits/overload. "
                    f"Attempting failover to fallback model '{next_model}'..."
                )
                try:
                    response = _invoke_tts_model(next_model, text, voice_name=voice_name)
                    break
                except Exception as fallback_err:
                    logger.warning(f"Fallback text-to-voice model '{next_model}' also failed: {fallback_err}")
                    current_model = next_model
                    e = fallback_err
            else:
                raise LlmApiException(
                    "All primary and fallback AI services are currently overloaded. Please try again later."
                ) from e
        elif "token" in error_str or "maximum context length" in error_str:
            raise LlmApiException("The text exceeded the maximum processing limit.") from e
        elif any(auth_kw in error_str for auth_kw in ["401", "403", "invalid api key", "unauthorized", "forbidden"]):
            raise LlmApiException("Authentication issue with the AI service. Please check API credentials.") from e
        else:
            raise LlmApiException(f"An unexpected AI service error occurred (check the API): {str(e)}") from e

    audio_bytes = None
    if response and response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.inline_data and part.inline_data.data:
                        audio_bytes = part.inline_data.data
                        break

    if not audio_bytes:
        raise LlmApiException("Failed to retrieve audio data from the AI service response.")

    if output_path:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

    return audio_bytes
