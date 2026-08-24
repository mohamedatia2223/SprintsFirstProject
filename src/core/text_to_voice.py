import sys
import os
import logging
import io
import wave
from google import genai
from google.genai import types
from src.core.config import LLM_API_KEY
from src.core.exceptions import LlmApiException, handle_api_exception

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

PRIMARY_TTS_MODEL = "gemini-2.5-flash-preview-tts"
SECONDARY_TTS_MODEL = "gemini-2.5-flash-native-audio-latest"
TERTIARY_TTS_MODEL = "gemini-2.5-pro-preview-tts"

TTS_FALLBACK_CHAIN = [PRIMARY_TTS_MODEL, SECONDARY_TTS_MODEL, TERTIARY_TTS_MODEL]

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

    tts_prompt = f"Read the following transcript out loud:\n{text}"
    return client.models.generate_content(
        model=model_name,
        contents=tts_prompt,
        config=config
    )

def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:

    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)       
        wf.setsampwidth(2)      
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_io.getvalue()


def convert_text_to_audio(text: str, output_path: str = None, voice_name: str = "Puck") -> bytes:
    if not text or not text.strip():
        raise ValueError("Text prompt for text-to-voice conversion cannot be empty.")

    current_model = PRIMARY_TTS_MODEL
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
                start_idx = TTS_FALLBACK_CHAIN.index(current_model)
            except ValueError:
                start_idx = 0

            for next_model in TTS_FALLBACK_CHAIN[start_idx + 1:]:
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
                raise handle_api_exception(e, service_name="text-to-voice service", is_overloaded=True) from e
        
        raise handle_api_exception(e, service_name="text-to-voice service") from e

    audio_bytes = None
    if response and response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.inline_data and part.inline_data.data:
                        raw_pcm = part.inline_data.data
                        if not raw_pcm.startswith(b'RIFF'):
                            audio_bytes = _pcm_to_wav(raw_pcm, sample_rate=24000)
                        else:
                            audio_bytes = raw_pcm
                        break

    if not audio_bytes:
        raise LlmApiException("Failed to retrieve audio data from the AI service response.")

    if output_path:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

    return audio_bytes
