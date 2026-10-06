import os
from pathlib import Path

# --- CAMINHOS BASE ---
BASE_USER_DIR = Path(r"C:\Users\bianca.mendes\OneDrive\GEROP - Arquivos")

# Diretório de entrada (sincronizado via OneDrive)
INPUT_DIR = BASE_USER_DIR / "GEROP - Fundos"

# Diretórios de saída para os arquivos intermediários
OUTPUT_MD_LEGIBLE = BASE_USER_DIR / "GEROP - Fundos .md"
OUTPUT_MD_ILLEGIBLE = BASE_USER_DIR / "GEROP - Fundos .md (Ilegiveis)"

# A pasta de ilegíveis pode ficar ao lado da de legíveis (padrão) ou DENTRO dela
# (como no pacote basegerop.zip). Aceita as duas.
if not OUTPUT_MD_ILLEGIBLE.exists() and (OUTPUT_MD_LEGIBLE / OUTPUT_MD_ILLEGIBLE.name).exists():
    OUTPUT_MD_ILLEGIBLE = OUTPUT_MD_LEGIBLE / OUTPUT_MD_ILLEGIBLE.name

# Diretório do banco de dados vetorial local
CHROMA_DB_DIR = Path(r"C:\Users\bianca.mendes\chroma_db")

# --- CONFIGURAÇÕES DE PROCESSAMENTO ---
EXTENSIONS = {".pdf", ".html", ".htm", ".xlsx", ".xls"}
MIN_CHARACTERS = 100
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- MODELOS ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# --- LINKS PARA OS ARQUIVOS ORIGINAIS (servidor local somente-leitura) ---
FILE_SERVER_BIND = "127.0.0.1"          # use "0.0.0.0" para outros PCs da rede acessarem
FILE_SERVER_PORT = 8765
FILE_SERVER_PUBLIC_HOST = "localhost"   # nome/IP que aparece nos links (ex.: nome do seu PC)
