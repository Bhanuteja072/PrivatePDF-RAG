# import os
# from fastapi import FastAPI, UploadFile, File, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Local LLM
# llm = Ollama(model="llama3.2:1b")

# # Embedding model
# embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# vector_store = None


# class ChatRequest(BaseModel):
#     message: str


# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
#     global vector_store

#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)

#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())

#         loader = PyPDFLoader(file_path)
#         documents = loader.load()

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )

#         chunks = splitter.split_documents(documents)
#         vector_store = FAISS.from_documents(chunks, embeddings)

#         return {"message": "PDF uploaded and indexed successfully!"}
#     except Exception as exc:
#         return {"message": f"Upload/index error: {exc}"}


# @app.post("/chat")
# async def chat(message: ChatRequest):
#     global vector_store

#     if vector_store is None:
#         return {"response": "Please upload a PDF first."}

#     try:
#         query = message.message

#         docs = vector_store.similarity_search(query, k=3)
#         context = "\n\n".join([doc.page_content for doc in docs])

#         prompt = f"""
#         Answer the question based only on the context below.

#         Context:
#         {context}

#         Question:
#         {query}
#         """

#         response = llm.invoke(prompt)
#         return {"response": response}
#     except Exception as exc:
#         return {"response": f"Chat error: {exc}"}



import os
import streamlit as st
from langchain_core.documents import Document
from src.graph.graph_builder import Graph_builder
from src.voice.voice_input import transcribe_once, tts_to_bytes


def app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    Handles UI loading, model setup, graph initialization, and safe exception handling.
    """
    st.set_page_config(page_title="LangGraph AgenticAI", layout="wide")
    st.title("LangGraph AgenticAI Application")
    
    if "pending_text" not in st.session_state:
        st.session_state.pending_text = ""
    if "chat_box" not in st.session_state:
        st.session_state.chat_box = ""

    pdf_path = None
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file:
        pdf_dir="uploaded_pdfs"
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_path = os.path.join(pdf_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {uploaded_file.name} successfully!")
    else:
        st.info("Please upload a PDF file to proceed.")

    st.subheader("Voice or Text Input")

    if st.button("🎙️ Start Recording"):
        with st.spinner("Listening..."):
            try:
                transcript = transcribe_once()
                if transcript:
                    st.session_state.chat_box = transcript
                else:
                    st.warning("No speech detected.")
            except Exception as exc:
                st.error(f"STT error: {exc}")

    st.text_input("Enter your message:", key="chat_box")
    submitted = st.button("Submit")

    if not pdf_path:
        st.stop()

    if not submitted:
        st.stop()

    st.session_state.pending_text = st.session_state.chat_box.strip()
    if not st.session_state.pending_text:
        st.warning("Please enter or record a question before submitting.")
        st.stop()

    graph_builder = Graph_builder(pdf_path=pdf_path)
    try:
        graph = graph_builder.build()
        st.success("Graph built successfully!")
    except Exception as e:
        st.error(f"Error building graph: {e}")
        return

    with st.spinner("Analyzing PDF and generating answer... ⏳"):
        try:
            result = graph.invoke({"question": st.session_state.pending_text})
            def _extract_answer(res):
                if isinstance(res, dict):
                    for k in ("generation", "answer", "output", "response", "result"):
                        v = res.get(k)
                        if isinstance(v, str) and v.strip():
                            return v
                        return None
            answer = _extract_answer(result) 
            docs = result.get("documents", []) if isinstance(result, dict) else []
            if docs is None:
                docs = []
            elif isinstance(docs, Document):
                docs = [docs]
            elif isinstance(docs, dict) and "page_content" in docs:
                docs = [Document(page_content=docs["page_content"])]
            elif not isinstance(docs, list):
                # Fallback: wrap unknown type as string
                docs = [Document(page_content=str(docs))]  
            with st.chat_message("user"):
                st.write(st.session_state.pending_text)  

            if answer:
                with st.chat_message("assistant"):
                    st.subheader(" 📖 Answer:")
                    st.write(answer)
                    try:
                        audio_out = tts_to_bytes(answer)
                        st.audio(audio_out, format="audio/wav")
                    except Exception as exc:
                        st.warning(f"TTS error: {exc}")

            if docs:
                with st.expander("🔎 Supporting PDF Chunks"):
                    for i, d in enumerate(docs[:5]):  # show first 5 chunks
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(d.page_content)

            if not answer and not docs:
                        st.info("No answer or supporting documents returned. Expand Debug to inspect state.")
                        with st.expander("Debug state"):
                            try:
                                st.json(result if isinstance(result, dict) else {"result": str(result)})
                            except Exception:
                                st.write(result)
        except Exception as e:
            st.error(f"Error running graph: {e}")







