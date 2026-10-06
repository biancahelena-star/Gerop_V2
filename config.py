import os
from pathlib import Path

# Diretório base do repositório
BASE_DIR = Path(__file__).resolve().parent

# --- CAMINHOS DINÂMICOS (Suporta Windows e Linux/Streamlit Cloud) ---
LOCAL_USER_DIR = Path(r"C:\Users\bianca.mendes\OneDrive\GEROP - Arquivos")

if LOCAL_USER_DIR.exists():
    BASE_USER_DIR = LOCAL_USER_DIR
    CHROMA_DB_DIR = Path(r"C:\Users\bianca.mendes\chroma_db")
else:
    # Fallback para execução na nuvem (Streamlit Cloud)
    BASE_USER_DIR = BASE_DIR / "data"
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"

INPUT_DIR = BASE_USER_DIR / "GEROP - Fundos"
OUTPUT_MD_LEGIBLE = BASE_USER_DIR / "GEROP - Fundos .md"
OUTPUT_MD_ILLEGIBLE = BASE_USER_DIR / "GEROP - Fundos .md (Ilegiveis)"

# --- CONFIGURAÇÕES DE PROCESSAMENTO ---
EXTENSIONS = {".pdf", ".html", ".htm", ".xlsx", ".xls"}
MIN_CHARACTERS = 100
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- MODELOS ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# --- SERVIDOR DE ARQUIVOS LOCAL ---
FILE_SERVER_BIND = "127.0.0.1"
FILE_SERVER_PORT = 8765
FILE_SERVER_PUBLIC_HOST = "localhost"
