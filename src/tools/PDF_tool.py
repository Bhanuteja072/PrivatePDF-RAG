import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFTool:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.embedding =  OllamaEmbeddings(model="mxbai-embed-large")
        self.vectorstore = None
        self.retriever = None
        self._prepare_pdf()

    def _prepare_pdf(self):
        loader = PyPDFLoader(self.pdf_path)
        try:
            docs_list = loader.load()
        except Exception as exc:
            raise ValueError(
                f"Failed to read PDF. Ensure it is not password-protected and is text-based. ({exc})"
            ) from exc
        if not docs_list:
            raise ValueError(
                "PDF produced no text. Please upload a text-based or OCR-processed PDF."
            )
        def clean_page_text(txt: str) -> str:
            # tweak to remove repeated page headers/footers, common line patterns, or "Page X of Y"
            import re
            txt = re.sub(r"Page\s*\d+\s*(of\s*\d+)?", "", txt, flags=re.IGNORECASE)
            txt = re.sub(r"^\s*-+\s*$", "", txt, flags=re.MULTILINE)
            return txt.strip()
        for d in docs_list:
            d.page_content = clean_page_text(d.page_content)
        full_text = "\n\n".join(d.page_content for d in docs_list)


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        docs_splits = text_splitter.split_documents(docs_list)
        self.vectorstore = FAISS.from_documents(docs_splits, self.embedding)
        # Fetch a few more chunks to give the generator richer context.
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
    def get_retriever(self):
        if self.retriever is None:
            self._prepare_pdf()
        return self.retriever

@st.cache_resource
def build_pdf_retriver(pdf_path):
    return PDFTool(pdf_path).get_retriever()