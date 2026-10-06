import os
import sys
from pathlib import Path
import streamlit as st

# Forçar codificação UTF-8 para o ambiente Linux do Streamlit Cloud
sys.stdout.reconfigure(encoding='utf-8')

# Importação dos módulos locais com importação da classe correta (HybridRAG)
try:
    import config as cfg
    from rag_engine import HybridRAG
    RAG_AVAILABLE = True
    IMPORT_ERROR = None
except Exception as e:
    RAG_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Configuração da Página
st.set_page_config(
    page_title="Especialista GEROP - Busca de Arquivos",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização do Tema Escuro
CUSTOM_DARK_THEME = """
<style>
    .stApp { background-color: #0D1117; color: #FFFFFF !important; }
    .stApp p, .stApp span, .stApp label, .stApp div, .stMarkdown { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    h1, h2, h3, h4 { color: #FFFFFF !important; }
    .stTextInput input { background-color: #0D1117 !important; color: #FFFFFF !important; border: 1px solid #30363D !important; }
    .stButton > button { background-color: #238636; color: #FFFFFF !important; border-radius: 6px; }
    .stButton > button:hover { background-color: #2EA043; }
</style>
"""
st.markdown(CUSTOM_DARK_THEME, unsafe_allow_html=True)

# Função de verificação do selo do índice
def _index_stamp() -> float:
    try:
        path = Path(cfg.CHROMA_DB_DIR) / "bm25.pkl"
        return path.stat().st_mtime if path.exists() else 0.0
    except Exception:
        return 0.0

# Carregamento do Motor RAG com Cache
@st.cache_resource
def load_rag(index_stamp: float):
    if not RAG_AVAILABLE:
        return None
    return HybridRAG(use_semantic=False) # Desativa busca semântica pesada se não houver GPU na nuvem

# Inicialização
st.title("🔍 Especialista GEROP - Busca de Arquivos")
st.caption("Sistema de recuperação lexical e semântica com ranking de precisão.")

rag = None
if RAG_AVAILABLE:
    try:
        rag = load_rag(_index_stamp())
    except Exception as e:
        st.error(f"Erro ao carregar o banco de dados/índice bm25.pkl: {e}")

# Barra Lateral
with st.sidebar:
    st.header("Opções da Busca")
    max_results = st.slider("Máximo de arquivos por resposta", min_value=10, max_value=250, value=50, step=10)
    thorough = st.checkbox("Busca minuciosa (amplia lista em buscas curtas)", value=True)
    include_semantic = st.checkbox("Incluir resultados por significado", value=False)
    
    st.divider()
    status_txt = "Ativo" if rag else "Inoperante (Verifique o índice)"
    st.caption(f"Status do Servidor: {status_txt}")

# Interface de Chat / Busca
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
        if rag is None:
            st.error(f"O motor de busca não pôde ser iniciado. Detalhes: {IMPORT_ERROR}")
        else:
            with st.spinner("Buscando no acervo..."):
                try:
                    answer, _results = rag.generate_response(
                        prompt,
                        max_results=max_results,
                        semantic_candidates=25 if include_semantic else 0,
                        min_results=(max_results if thorough else 0),
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as err:
                    st.error(f"Falha ao processar a consulta: {err}")
