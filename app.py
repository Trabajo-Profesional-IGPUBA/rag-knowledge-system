from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="IGPUBA — Consulta técnica",
    page_icon="🛢️",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
LOGS_DIR = BASE_DIR / "logs"

st.markdown(
    """
<style>
    .stChatMessage { border-radius: 10px; }
    .stSpinner { color: #555; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🛢️ IGPUBA — Sistema de consulta")
st.caption("Consultá documentos técnicos de pozos en lenguaje natural.")
st.divider()


@st.cache_resource(show_spinner="Cargando sistema RAG...")
def load_pipeline():
    import logging

    from src.embeddings.embedder import Embedder
    from src.llm.client import LLMClient, LLMConfig
    from src.llm.prompt_builder import PromptBuilder
    from src.llm.rag_pipeline import RAGConfig, RAGPipeline
    from src.observability import setup_logging
    from src.retrieval.retriever import Retriever
    from src.retrieval.vectorstore import VectorStore

    setup_logging(LOGS_DIR, level=logging.WARNING)

    embedder = Embedder()
    vectorstore = VectorStore(VECTORSTORE_DIR)
    retriever = Retriever(embedder, vectorstore)

    llm_client = LLMClient(LLMConfig(model="llama3:8b", timeout=600))
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        prompt_builder=PromptBuilder(),
        config=RAGConfig(top_k=5, min_score=0.3),
    )
    return pipeline, llm_client


try:
    pipeline, llm_client = load_pipeline()
    ollama_ok = llm_client.is_available()
except Exception as e:  # noqa: BLE001 — falla de inicialización de la app: se muestra el error en la UI y se detiene la ejecución, no hay forma razonable de continuar
    st.error(f"Error al cargar el sistema: {e}")
    st.stop()

if not ollama_ok:
    st.error("⚠️ Ollama no está corriendo. Ejecutá en la terminal: `ollama serve`")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribí tu consulta técnica..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        with st.spinner("Buscando en documentos..."):
            for token in pipeline.query_stream(prompt):
                full_response += token
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
