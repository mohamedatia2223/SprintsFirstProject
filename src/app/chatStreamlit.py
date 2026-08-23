
import sys
import os
import logging
import streamlit as st

# Page Configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Sprints Assistant - Rich Dad Poor Dad",
    page_icon="📚",
    layout="wide"
)

# Ensure project root is in sys.path when running file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.orchestration import run_assistant
from src.core.text_to_voice import convert_text_to_audio
from src.core.voice_to_text import convert_audio_text
from src.core.llm import list_available_gemini_models, set_llm_models, PRIMARY_MODEL, SECONDARY_MODEL, TERTIARY_MODEL

# Initialize Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .verdict-approved {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .verdict-refusal {
        background-color: #FFF3E0;
        color: #EF6C00;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .source-box {
        background-color: #F8F9FA;
        border-left: 4px solid #1E88E5;
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Fetch available models from Gemini API
@st.cache_data(ttl=600)
def load_available_models():
    return list_available_gemini_models()

available_models = load_available_models()

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ System Control")
    st.markdown("---")
    
    st.subheader("Gemini Model Selection")
    
    selected_primary = st.selectbox(
        "Primary Model",
        options=available_models,
        index=available_models.index(PRIMARY_MODEL) if PRIMARY_MODEL in available_models else 0
    )
    
    selected_secondary = st.selectbox(
        "Secondary Failover Model",
        options=available_models,
        index=available_models.index(SECONDARY_MODEL) if SECONDARY_MODEL in available_models else (1 if len(available_models) > 1 else 0)
    )

    selected_tertiary = st.selectbox(
        "Tertiary Failover Model",
        options=available_models,
        index=available_models.index(TERTIARY_MODEL) if TERTIARY_MODEL in available_models else (2 if len(available_models) > 2 else 0)
    )

    # Update active LLM fallback chain
    set_llm_models(selected_primary, selected_secondary, selected_tertiary)

    st.markdown("---")
    st.subheader("Multi-Agent Setup")
    st.markdown("""
    * **Agent 1 (Researcher)**: Searches Qdrant collection & drafts cited answer.
    * **Agent 2 (Reviewer)**: Fact-checks draft against context passages.
    """)
    
    max_iterations = st.slider("Max Revision Iterations", min_value=1, max_value=3, value=2)
    enable_tts = st.checkbox("🔊 Enable Text-to-Speech Output", value=False)

    st.markdown("---")
    st.subheader("Ingested Corpus")
    st.info("**Rich Dad Poor Dad** by Robert T. Kiyosaki\n\nVector Database: **Qdrant Cloud** (Collection: `rich_dad_poor_dad`)")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# Main UI Header
st.markdown("<div class='main-title'>Sprints Practice Task 0: AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Two-Agent RAG Assistant with Qdrant Vector Retrieval & Claim Verification</div>", unsafe_allow_html=True)


# Render Chat Messages History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Sources and Verdict for Assistant Messages
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            
            # Reviewer Verdict Badge
            verdict = meta.get("reviewer_verdict", "APPROVED")
            is_refusal = meta.get("is_refusal", False)
            
            if is_refusal:
                st.markdown("<span class='verdict-refusal'>⚠️ Refusal (Information Not in Document Set)</span>", unsafe_allow_html=True)
            elif verdict == "APPROVED":
                st.markdown(f"<span class='verdict-approved'>✅ Verdict: APPROVED (Verified by Agent 2 Reviewer)</span>", unsafe_allow_html=True)
            
            # Sources Accordion
            sources = meta.get("sources", [])
            if sources:
                with st.expander(f"View Retrieved Sources ({len(sources)} passages)"):
                    for idx, s in enumerate(sources, 1):
                        st.markdown(
                            f"**[Passage {idx}] Page {s.page_number}** | *Section: {s.section_title}* (Score: {s.score:.3f})\n"
                            f"> {s.text}"
                        )

            # Audio Player if TTS generated
            if "audio_bytes" in meta and meta["audio_bytes"]:
                st.audio(meta["audio_bytes"], format="audio/wav")


# Main Input Area: Microphone Recording + Text Chat Bar
input_container = st.container()

with input_container:
    # Microphone recorder right above the chat bar
    recorded_audio = st.audio_input("🎙️ Record Voice Question", key="mic_recorder")
    user_prompt = st.chat_input("Ask a question about Rich Dad Poor Dad...")

# Handle Microphone Recording Transcription
if recorded_audio is not None and not user_prompt:
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, "st_recorded_mic.wav")
    with open(temp_audio_path, "wb") as f:
        f.write(recorded_audio.read())
    try:
        with st.spinner("Transcribing your voice recording..."):
            prompt_text = convert_audio_text(temp_audio_path)
            if prompt_text:
                st.toast(f"🎙️ Transcribed: '{prompt_text}'")
                user_prompt = prompt_text
    except Exception as e:
        st.error(f"Voice transcription error: {e}")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


# Process New Question
if user_prompt:
    # Display User Message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Agent 1 (Researcher) & Agent 2 (Reviewer) are analyzing your question..."):
            try:
                state = run_assistant(question=user_prompt, max_iterations=max_iterations)

                # Render Answer
                st.markdown(state.final_answer)

                # Render Verdict Badge
                if state.is_refusal:
                    st.markdown("<span class='verdict-refusal'>⚠️ Refusal (Information Not in Document Set)</span>", unsafe_allow_html=True)
                elif state.reviewer_verdict == "APPROVED":
                    st.markdown("<span class='verdict-approved'>✅ Verdict: APPROVED (Verified by Agent 2 Reviewer)</span>", unsafe_allow_html=True)

                # Render Sources Expander
                if state.passages:
                    with st.expander(f"View Retrieved Sources ({len(state.passages)} passages)"):
                        for idx, s in enumerate(state.passages, 1):
                            st.markdown(
                                f"**[Passage {idx}] Page {s.page_number}** | *Section: {s.section_title}* (Score: {s.score:.3f})\n"
                                f"> {s.text}"
                            )

                # Optional Text-to-Speech Generation
                audio_data = None
                if enable_tts and state.final_answer:
                    try:
                        audio_data = convert_text_to_audio(state.final_answer)
                        st.audio(audio_data, format="audio/wav")
                    except Exception as tts_err:
                        st.warning(f"TTS generation error: {tts_err}")

                # Save Assistant Message to History
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": state.final_answer,
                    "metadata": {
                        "reviewer_verdict": state.reviewer_verdict,
                        "is_refusal": state.is_refusal,
                        "sources": state.passages,
                        "audio_bytes": audio_data
                    }
                })

            except Exception as err:
                st.error(f"Error processing question: {err}")
