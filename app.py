import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys

# Forçar a codificação UTF-8 para evitar caracteres corrompidos no Streamlit Cloud
sys.stdout.reconfigure(encoding='utf-8')

# -----------------------------------------------------------------------------
# IMPORTAÇÃO DOS MÓDULOS RAG COM TRATAMENTO DE ERRO
# -----------------------------------------------------------------------------
try:
    from rag_engine import RAGEngine
    from indexer import DocumentIndexer
    RAG_AVAILABLE = True
    IMPORT_ERROR = None
except Exception as e:
    RAG_AVAILABLE = False
    IMPORT_ERROR = str(e)

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agente-Gerop | Inteligência Operacional",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# ESTILIZAÇÃO CUSTOMIZADA (TEMA ESCURO COM TEXTO EM BRANCO)
# -----------------------------------------------------------------------------
CUSTOM_DARK_THEME = """
<style>
    /* Estilo Geral da Aplicação */
    .stApp {
        background-color: #0D1117;
        color: #FFFFFF !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Garantir cor branca legível em todos os elementos de texto */
    .stApp p, .stApp span, .stApp label, .stApp div, .stMarkdown, .stCaption, .stText {
        color: #FFFFFF !important;
    }

    /* Barra Lateral */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* Títulos e Cabeçalhos */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px;
    }

    /* Cards e Contentores */
    div[data-testid="stExpander"] {
        background-color: #161B22;
        border: 1px solid #30363D !important;
        border-radius: 6px;
    }
    
    .status-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
    }

    .status-card-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #FFFFFF !important;
        margin-bottom: 6px;
    }

    .status-card-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #58A6FF !important;
    }

    /* Entradas de Texto */
    .stTextInput input, .stSelectbox > div > div {
        background-color: #0D1117 !important;
        color: #FFFFFF !important;
        border: 1px solid #30363D !important;
        border-radius: 6px !important;
    }
    
    .stTextInput input:focus {
        border-color: #58A6FF !important;
        box-shadow: none !important;
    }

    /* Botões */
    .stButton > button {
        background-color: #238636;
        color: #FFFFFF !important;
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 6px;
        font-weight: 500;
        padding: 0.4rem 1rem;
        transition: background-color 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #2EA043;
        border-color: rgba(240, 246, 252, 0.2);
    }

    /* Linhas e Tabelas */
    hr {
        border-color: #30363D !important;
    }

    .stDataFrame {
        border: 1px solid #30363D;
        border-radius: 6px;
    }
</style>
"""

st.markdown(CUSTOM_DARK_THEME, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CARREGAMENTO DO MOTOR RAG COM CACHE EM MEMÓRIA
# -----------------------------------------------------------------------------
@st.cache_resource
def carregar_motor_rag():
    if not RAG_AVAILABLE:
        return None
    try:
        return RAGEngine()
    except Exception as e:
        st.error(f"Erro ao inicializar o RAGEngine: {e}")
        return None

rag_engine = carregar_motor_rag()

# -----------------------------------------------------------------------------
# BARRA LATERAL (PARÂMETROS DA BUSCA)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("AGENTE-GEROP")
    st.caption("Painel de Controlo e Parâmetros")
    st.divider()

    st.subheader("Parâmetros da Busca")
    
    max_arquivos = st.slider(
        label="Máximo de ficheiros por resposta",
        min_value=1,
        max_value=241,
        value=21,
        step=5,
        help="Define a quantidade limite de documentos de contexto retornados pelo mecanismo RAG."
    )

    # Limiar de similaridade fixado no valor máximo (1.0)
    limiar_similaridade = 1.0

    st.divider()
    st.subheader("Configurações do Modelo")
    
    modelo_llm = st.selectbox(
        "Modelo de Linguagem",
        options=["LLM-Corporativo-v2", "LLM-Especializado-RAG", "LLM-Fast-Inference"],
        index=0
    )

    incluir_metadados = st.checkbox("Exibir metadados detalhados nas fontes", value=True)
    
    st.divider()
    
    status_servidor = "Operacional" if rag_engine else "Atenção (Falha RAG)"
    st.caption(f"Status do Servidor: {status_servidor}")
    st.caption(f"Última sincronização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")


# -----------------------------------------------------------------------------
# PAINEL PRINCIPAL
# -----------------------------------------------------------------------------
st.title("Consulta de Documentos e Recuperação RAG")
st.markdown("Interface corporativa para pesquisa vetorial e análise automatizada de documentos operacionais.")

st.divider()

# Métricas do Sistema
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown("""
        <div class="status-card">
            <div class="status-card-header">Base de Dados</div>
            <div class="status-card-value">241 Ficheiros</div>
        </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
        <div class="status-card">
            <div class="status-card-header">Limite por Consulta</div>
            <div class="status-card-value">{max_arquivos} Docs</div>
        </div>
    """, unsafe_allow_html=True)

with col_m3:
    status_cor = "#3FB950" if rag_engine else "#D29922"
    status_texto = "Ativo" if rag_engine else "Sem Índice"
    st.markdown(f"""
        <div class="status-card">
            <div class="status-card-header">Status do Pipeline</div>
            <div class="status-card-value" style="color: {status_cor} !important;">{status_texto}</div>
        </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown("""
        <div class="status-card">
            <div class="status-card-header">Índice Vetorial</div>
            <div class="status-card-value" style="color: #D29922 !important;">Sincronizado</div>
        </div>
    """, unsafe_allow_html=True)

# Entrada da Pesquisa
st.subheader("Solicitação de Pesquisa")

query_input = st.text_input(
    label="Digite o termo ou comando para pesquisa nos documentos",
    placeholder="Ex: Informe os procedimentos operacionais para análise de risco da unidade...",
    label_visibility="collapsed"
)

col_btn_1, col_btn_2, col_spacer = st.columns([1, 1, 4])

with col_btn_1:
    executar_busca = st.button("Executar Pesquisa", use_container_width=True)

with col_btn_2:
    limpar_campos = st.button("Limpar Consulta", use_container_width=True)

st.divider()

# Execution e Exibição de Resultados Reais
if executar_busca and query_input:
    st.subheader("Resultado da Consulta")
    
    tab_resposta, tab_fontes, tab_metricas = st.tabs(["Resposta Gerada", "Fontes Consultadas", "Métricas de Busca"])

    inicio_tempo = datetime.now()
    resultado_rag = None

    if rag_engine is not None:
        try:
            # Executa a busca através dos métodos disponíveis no RAGEngine original
            if hasattr(rag_engine, 'query'):
                resultado_rag = rag_engine.query(query_input, top_k=max_arquivos)
            elif hasattr(rag_engine, 'search'):
                resultado_rag = rag_engine.search(query_input, top_k=max_arquivos)
            elif callable(rag_engine):
                resultado_rag = rag_engine(query_input)
        except Exception as err:
            st.error(f"Erro ao executar consulta no motor RAG: {err}")
    
    tempo_execucao = (datetime.now() - inicio_tempo).total_seconds()

    with tab_resposta:
        st.markdown("**Síntese dos Documentos:**")
        if resultado_rag:
            if isinstance(resultado_rag, dict):
                resposta_texto = resultado_rag.get('resposta', resultado_rag.get('answer', resultado_rag.get('output', str(resultado_rag))))
            else:
                resposta_texto = str(resultado_rag)
            st.markdown(resposta_texto)
        else:
            if not RAG_AVAILABLE:
                st.error(f"Não foi possível importar os módulos RAG: {IMPORT_ERROR}")
            else:
                st.warning("A consulta foi enviada, mas o motor RAG não devolveu resultados. Certifique-se de que a pasta de dados e o índice (.pkl / .index / vector DB) foram incluídos no repositório GitHub.")

    with tab_fontes:
        st.markdown(f"**Documentos Selecionados (Máximo configurado: {max_arquivos}):**")
        
        fontes = []
        if isinstance(resultado_rag, dict):
            fontes = resultado_rag.get('fontes', resultado_rag.get('sources', resultado_rag.get('docs', [])))
        
        if fontes:
            for idx, fonte in enumerate(fontes[:max_arquivos], start=1):
                if isinstance(fonte, dict):
                    nome_doc = fonte.get('ficheiro', fonte.get('documento', fonte.get('file', f'Documento_{idx}.pdf')))
                    relevancia = fonte.get('score', limiar_similaridade)
                    trecho = fonte.get('conteudo', fonte.get('text', fonte.get('chunk', 'Trecho não disponível.')))
                    caminho = fonte.get('caminho', f'/base_dados/gerop/{nome_doc}')
                else:
                    nome_doc = f"Documento_{idx}.pdf"
                    relevancia = limiar_similaridade
                    trecho = str(fonte)
                    caminho = "/base_dados/gerop/"

                with st.expander(f"{nome_doc} (Relevância: {relevancia:.2f})"):
                    st.write(f"**Caminho:** `{caminho}`")
                    st.write("**Trecho Relevante:**")
                    st.caption(f'"{trecho}"')
                    if incluir_metadados and isinstance(fonte, dict) and 'metadados' in fonte:
                        st.json(fonte['metadados'])
        else:
            st.info("Nenhuma fonte individual foi retornada pela estrutura do RAG para esta consulta.")

    with tab_metricas:
        data_metricas = {
            "Métrica": ["Tempo total da Consulta", "Ficheiros Solicitados", "Limiar de Similaridade"],
            "Valor": [f"{tempo_execucao:.2f}s", f"{max_arquivos}", f"{limiar_similaridade}"]
        }
        st.table(pd.DataFrame(data_metricas))

elif executar_busca and not query_input:
    st.warning("Por favor, digite um termo de pesquisa antes de executar.")
