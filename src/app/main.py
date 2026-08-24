
import sys
import os
import tempfile
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from src.pipeline.orchestration import run_assistant
from src.core.voice_to_text import convert_audio_text
from src.core.text_to_voice import convert_text_to_audio
from src.core.exceptions import LlmApiException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    max_iterations = data.get("max_iterations", 2)

    if not question:
        return jsonify({"error": "Field 'question' cannot be empty."}), 400

    try:
        logger.info(f"[API /api/chat] Received question: '{question}'")
        state = run_assistant(question=question, max_iterations=max_iterations)

        sources_data = [
            {
                "page_number": p.page_number,
                "section_title": p.section_title,
                "score": round(p.score, 4),
                "text": p.text
            }
            for p in state.passages
        ]

        return jsonify({
            "question": state.question,
            "final_answer": state.final_answer,
            "reviewer_verdict": state.reviewer_verdict,
            "reviewer_feedback": state.reviewer_feedback,
            "is_refusal": state.is_refusal,
            "iterations": state.iteration_count,
            "sources": sources_data
        }), 200

    except LlmApiException as e:
        logger.error(f"[API /api/chat] LLM/API Error: {e}")
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.error(f"[API /api/chat] Unexpected Error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/voice-to-text", methods=["POST"])
def voice_to_text_endpoint():

    if "file" not in request.files:
        return jsonify({"error": "No audio file provided in form field 'file'."}), 400

    audio_file = request.files["file"]
    if audio_file.filename == "":
        return jsonify({"error": "Selected file is empty."}), 400

    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"input_{audio_file.filename}")
    audio_file.save(temp_file_path)

    try:
        logger.info(f"[API /api/voice-to-text] Transcribing uploaded file '{audio_file.filename}'...")
        transcript = convert_audio_text(temp_file_path)
        return jsonify({"transcript": transcript}), 200
    except Exception as e:
        logger.error(f"[API /api/voice-to-text] Error: {e}")
        return jsonify({"error": f"Voice transcription failed: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



@app.route("/api/text-to-voice", methods=["POST"])
def text_to_voice_endpoint():

    data = request.get_json() or {}
    text = data.get("text", "").strip()
    voice_name = data.get("voice_name", "Puck")

    if not text:
        return jsonify({"error": "Field 'text' cannot be empty."}), 400

    try:
        logger.info(f"[API /api/text-to-voice] Converting text ({len(text)} chars) to audio...")
        audio_bytes = convert_text_to_audio(text, voice_name=voice_name)

        temp_dir = tempfile.gettempdir()
        temp_audio_path = os.path.join(temp_dir, "speech_response.wav")
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)

        return send_file(
            temp_audio_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="response.wav"
        )
    except Exception as e:
        logger.error(f"[API /api/text-to-voice] Error: {e}")
        return jsonify({"error": f"Text-to-voice conversion failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting Flask API Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
