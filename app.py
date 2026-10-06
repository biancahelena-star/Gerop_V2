import os
from pathlib import Path

import streamlit as st

import config as cfg
from file_server import start_file_server
from rag_engine import HybridRAG

st.set_page_config(
    page_title="Especialista GEROP - Busca de Arquivos",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Especialista GEROP - Busca de Arquivos")
st.caption(
    "Todos os arquivos que correspondem à busca, do que tem mais correspondência "
    "(mesma sequência numérica/alfabética ou muito próxima) para o que tem menos "
    "(apenas alguns números ou palavras). Os links abrem o arquivo original (PDF/planilha)."
)


def _index_stamp() -> float:
    """Muda sempre que o indexador gera um novo bm25.pkl."""
    path = Path(cfg.CHROMA_DB_DIR) / "bm25.pkl"
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource
def load_rag(index_stamp: float):
    # index_stamp faz parte da chave do cache: ao reindexar, o app recarrega
    # o índice sozinho (antes ele ficava preso ao índice antigo).
    return HybridRAG()


@st.cache_resource
def load_file_server():
    return start_file_server(
        cfg.INPUT_DIR,
        host=getattr(cfg, "FILE_SERVER_BIND", "127.0.0.1"),
        port=getattr(cfg, "FILE_SERVER_PORT", 8765),
        extensions=cfg.EXTENSIONS,
    )


try:
    rag = load_rag(_index_stamp())
except Exception as e:
    st.error(f"Erro ao carregar o motor de busca: {e}. Execute 'python indexer.py' no terminal.")
    st.stop()

server = load_file_server()


@st.cache_data
def count_files_on_disk(index_stamp: float) -> int:
    return sum(
        1
        for _, _, names in os.walk(cfg.INPUT_DIR)
        for name in names
        if Path(name).suffix.lower() in cfg.EXTENSIONS
    )


indexed_files = len(rag.files)
files_on_disk = count_files_on_disk(_index_stamp())

with st.sidebar:
    st.header("Opções")
    max_results = st.slider(
        "Máximo de arquivos por resposta", min_value=20, max_value=500, value=150, step=10
    )
    thorough = st.checkbox(
        "Busca minuciosa (mais lenta, amplia a lista quando há poucos resultados)",
        value=True,
    )
    include_semantic = st.checkbox(
        "Incluir resultados só por similaridade de significado (no fim da lista)",
        value=True,
    )
    st.metric("Arquivos no índice", f"{indexed_files} de {files_on_disk}")
    if files_on_disk and indexed_files < 0.95 * files_on_disk:
        st.warning(
            "O índice está INCOMPLETO: só parte dos arquivos da pasta pode ser "
            "encontrada. Rode `python converter.py` e depois `python indexer.py` "
            "(o indexador lista no final quais arquivos ficaram de fora)."
        )
    st.caption(
        "Os links usam um servidor local na porta "
        f"{getattr(cfg, 'FILE_SERVER_PORT', 8765)}. "
        + ("Ativo." if server else "Porta já em uso: se outra instância do app estiver aberta, os links continuam funcionando.")
    )
    st.caption("O caminho completo de cada arquivo também é exibido para abrir pelo Explorer/OneDrive.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite o número do processo (ex: 000034368688), código ou termo..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando..."):
            answer, _results = rag.generate_response(
                prompt,
                max_results=max_results,
                semantic_candidates=25 if include_semantic else 0,
                min_results=(max_results if thorough else 0),
            )
        st.markdown(answer)

    # Guarda a resposta COMPLETA (com os links), não só o resumo.
    st.session_state.messages.append({"role": "assistant", "content": answer})
