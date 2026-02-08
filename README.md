LangGraph PDF Voice RAG (Offline)
=================================

Offline Streamlit app that answers questions over an uploaded PDF with a LangGraph-powered RAG pipeline. It supports push-to-record speech input (Vosk + PyAudio) and local TTS playback (pyttsx3). All LLM and embedding calls run through a local Ollama daemon—no cloud calls required.

[Live Demo](#)  <!-- replace with a real link when hosted -->

Features
--------
- PDF ingestion with chunking and FAISS retrieval (local, no upload to cloud).
- RAG graph using LangGraph with relevance grading and query rewrite for recall.
- Local LLM + embeddings via Ollama (`llama3.2:1b`, `mxbai-embed-large`).
- One-button voice capture (Vosk + PyAudio) and offline TTS playback (pyttsx3).
- Bullet-point answers (4–8 points) grounded strictly in retrieved context.

Architecture
------------
- Streamlit UI: [src/main.py](src/main.py)
- LangGraph build: [src/graph/graph_builder.py](src/graph/graph_builder.py)
- RAG nodes & prompts: [src/Nodes/chat_with_pdf.py](src/Nodes/chat_with_pdf.py)
- PDF loader, splitter, FAISS retriever: [src/tools/PDF_tool.py](src/tools/PDF_tool.py)
- Voice STT/TTS utilities: [src/voice/voice_input.py](src/voice/voice_input.py)
- Graph state definition: [src/state/graph_State.py](src/state/graph_State.py)

Prerequisites
-------------
- Windows with Python 3.10+ and PowerShell 5.1+
- [Ollama](https://ollama.com/download) running locally
	- Pull models: `ollama pull llama3.2:1b` and `ollama pull mxbai-embed-large`
- Vosk acoustic model downloaded (small EN example): `vosk-model-small-en-us-0.15`
	- Update `MODEL_PATH` in [src/voice/voice_input.py](src/voice/voice_input.py#L7-L11) if your path differs
- Microphone access for PyAudio; a local TTS voice available for pyttsx3

Setup
-----
1) Clone and create virtual environment
```
python -m venv venv
./venv/Scripts/Activate.ps1
```

2) Install dependencies
```
pip install -r requirements.txt
```
(If you do not have `requirements.txt`, install the key libs directly: `streamlit langgraph langchain-core langchain-ollama langchain-community langchain-text-splitters faiss-cpu vosk pyaudio pyttsx3`.)

3) Ensure Ollama is running and models are pulled
```
ollama pull llama3.2:1b
ollama pull mxbai-embed-large
```

4) Download Vosk model and set path
- Download `vosk-model-small-en-us-0.15` (or another language) and extract.
- Set `MODEL_PATH` in [src/voice/voice_input.py](src/voice/voice_input.py#L7-L11) to that directory.

Run
---
```
streamlit run app.py
```
Then in the browser:
- Upload a PDF (stored locally in `uploaded_pdfs/`).
- Click “Start Recording” to capture speech; the transcript fills the input box.
- Edit if needed and click “Submit” to run the RAG chain; answer plus TTS playback will appear.

Configuration Notes
-------------------
- Retrieval returns up to 6 chunks for richer context (see [src/tools/PDF_tool.py](src/tools/PDF_tool.py#L35-L45)).
- Answers are bullet-pointed and context-grounded (prompt in [src/Nodes/chat_with_pdf.py](src/Nodes/chat_with_pdf.py#L16-L32)).
- All processing remains local: PDF parsing, embeddings, LLM generation, STT, and TTS.

Troubleshooting
---------------
- PyAudio install issues: ensure you have appropriate build tools or install a prebuilt wheel for your Python version.
- No audio input: verify mic permissions in the browser and Windows privacy settings.
- Ollama not reachable: start the Ollama service and confirm the models are pulled.
- Vosk path errors: double-check `MODEL_PATH` points to the extracted model folder.


