# 📚 Multi-Agent RAG Assistant & Fact-Checking Assistant
> **Sprints Practice Task 0** | Two-Agent Document Q&A Assistant with Qdrant Vector Retrieval, Fact-Checking Reviewer, and Voice Capabilities.

---

## 🌟 Key Features

- **🤖 Multi-Agent Orchestration (`LangGraph`)**:
  - **Agent 1 (Researcher)**: Searches Qdrant vector database, extracts context passages, and drafts cited responses.
  - **Agent 2 (Reviewer)**: Strict fact-checking auditor that verifies draft statements against source passages to prevent hallucinations.
- **🚨 Leak-Proof Refusal Guardrail**:
  - If Agent 2 rejects a draft answer after maximum revision attempts, unverified drafts are suppressed and a standard refusal message (*"I am sorry, but the provided document set does not contain information to answer this question."*) is returned.
- **⚡ Flexible Model Selection & Dynamic Failover Chain**:
  - Full model selection flexibility: choose any Gemini model (e.g., `gemini-3.1-flash-lite`, `gemini-3.5-flash`, `gemini-2.5-flash`) dynamically via the Streamlit sidebar or `.env` configuration.
  - Automatic rate limit / quota overload failovers: `gemini-3.1-flash-lite` → `gemini-3.5-flash-lite` → `gemini-2.5-flash`.
- **🔊 Multi-Modal Voice Features**:
  - **Voice-to-Text**: Speech transcription using Google Gemini multimodal capabilities.
  - **Text-to-Voice**: Generates natural speech audio (`WAV`) output using Gemini TTS models (`Puck` voice).
- **🖥️ Dual Interfaces**:
  - **Streamlit App (`src/app/chatStreamlit.py`)**: Interactive web UI with model selection, chat history, source expanders, and mic recording.
  - **Flask REST API (`src/app/main.py`)**: REST endpoints for `/api/chat`, `/api/voice-to-text`, and `/api/text-to-voice`.
- **📝 PDF-to-Markdown Token Optimization**:
  - Automatically converts raw PDF files (`.pdf`) into structured, clean Markdown (`.md`) with explicit page markers (`<!-- Page N -->`) and header hierarchies before chunking.
  - Eliminates unnecessary binary noise, headers/footers, and duplicate whitespace, **drastically saving token usage** during LLM retrieval, reranking, and prompting.
- **📊 100-Test Benchmark Harness (`testScripts/generate_100_tests.py`)**:
  - Rigorous evaluation suite (70 in-domain + 30 out-of-domain questions) measuring execution success, refusal precision, and supported answer accuracy.

---

## 📊 Benchmark Evaluation Results

| Metric | Score | Details |
| :--- | :---: | :--- |
| **Total Test Cases** | **100** | 70 In-Domain + 30 Out-of-Domain Refusal Questions |
| **Execution Success** | **100 / 100 (100%)** | 0 unhandled exceptions or crashes |
| **Out-of-Domain Refusal Precision** | **30 / 30 (100.0%)** | 100% precision on out-of-domain questions |
| **Supported Questions Accuracy** | **63 / 70 (90.0%)** | Verified and cited in-domain answers |
| **Overall Benchmark Accuracy** | **92.0%** | Comprehensive benchmark score |

---

## 🛠️ Technology Stack

- **Frameworks & Agents**: Python 3.11+, LangGraph, LangChain
- **LLM Provider**: Google GenAI (`gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-2.5-flash`)
- **Vector Database**: Qdrant Cloud (`rich_dad_poor_dad` collection)
- **Embeddings & Reranking**: `gemini-embedding-001` + LLM Cross-Encoder Reranker
- **Document Processing**: PyMuPDF (`fitz`) PDF-to-Markdown parser
- **User Interface**: Streamlit & Flask (CORS enabled)

---

## 📁 Repository Structure

```
Sprints/
├── data/
│   ├── rich_dad_poor_dad_by_robert_t-_kiyosaki.pdf   # Source document (131 pages)
│   └── rich_dad_poor_dad_by_robert_t-_kiyosaki.md    # Parsed markdown with page markers
├── logs/
│   ├── app.log                                       # System execution log
│   └── test_runs.json                                # Benchmark evaluation results
├── src/
│   ├── agents/
│   │   ├── researcher.py                             # Agent 1: Researcher node
│   │   └── reviewer.py                               # Agent 2: Reviewer auditor node
│   ├── app/
│   │   ├── chatStreamlit.py                          # Streamlit web application
│   │   └── main.py                                   # Flask REST API server
│   ├── core/
│   │   ├── config.py                                 # Environment configuration & models
│   │   ├── embedding.py                              # Query & document embedding
│   │   ├── exceptions.py                             # Custom API exception handlers
│   │   ├── llm.py                                    # Model instantiation & failover logic
│   │   ├── logger.py                                 # App & benchmark JSON logger
│   │   ├── pdf_to_text.py                            # PDF text extraction utility
│   │   ├── prompts.py                                # System prompts & standard refusals
│   │   ├── schemas.py                                # Pydantic data schemas
│   │   ├── text_to_voice.py                          # Gemini Text-to-Speech service
│   │   └── voice_to_text.py                          # Gemini Voice-to-Text service
│   ├── database/
│   │   ├── connection.py                             # Qdrant client connection & collection setup
│   │   └── retrieval.py                              # Two-stage vector search & reranking
│   └── pipeline/
│       ├── ingestion.py                              # Document parsing & Qdrant upsert pipeline
│       └── orchestration.py                          # LangGraph state graph workflow
├── testScripts/
│   └── generate_100_tests.py                         # 100-test benchmark evaluation runner
├── .env.example                                      # Environment variables template
├── requirements.txt                                  # Dependencies manifest
└── README.md                                         # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/mohamedatia2223/SprintsFirstProject.git
cd Sprints
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory based on `.env.example`:

```env
VectoreDB_API_Key = your_qdrant_api_key
VectoreDB_endpoint = your_qdrant_cluster_url
LLM_API_Key = your_google_gemini_api_key

PRIMARY_MODEL = gemini-3.1-flash-lite
SECONDARY_MODEL = gemini-3.5-flash-lite
TERTIARY_MODEL = gemini-2.5-flash
```

---

## 💻 Running the Application

### Option A: Launch Streamlit Web UI

```bash
streamlit run src/app/chatStreamlit.py
```
Open your browser at `http://localhost:8501`.

### Option B: Launch Flask REST API Server

```bash
python src/app/main.py
```
The API server runs at `http://localhost:5000`.

#### API Endpoints:
- `POST /api/chat`: Submit text questions `{"question": "What is an asset?"}`
- `POST /api/voice-to-text`: Upload audio file in form data key `file`
- `POST /api/text-to-voice`: Convert response text to WAV audio `{"text": "Hello"}`

---

## 📥 Document Ingestion Pipeline

The document ingestion pipeline uses a two-step process optimized for accuracy and token efficiency:

1. **PDF to Markdown Conversion (`src/core/pdf_to_text.py`)**: Converts raw `.pdf` documents into formatted `.md` markdown files with explicit page markers (`<!-- Page N -->`). Cleaning boilerplate formatting **significantly saves token usage** during LLM prompting and cross-encoder reranking.
2. **Chunking & Vector Embedding (`src/pipeline/ingestion.py`)**: Splits the Markdown text into structured 800-character chunks (with 150-char overlap) preserving page numbers and section headers, embeds them with `gemini-embedding-001`, and upserts into Qdrant.

To re-ingest the full 131-page PDF document into Qdrant:

```bash
python src/pipeline/ingestion.py
```

---

## 🧪 Benchmark Evaluation Suite

To run the complete 100-test benchmark suite:

```bash
python testScripts/generate_100_tests.py
```

Results will be evaluated and saved to `logs/test_runs.json`.

---

## 📜 License

This project is released under the MIT License.