import os
import re
import pickle
import unicodedata
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from rank_bm25 import BM25Okapi

from config import (
    INPUT_DIR,
    EXTENSIONS,
    OUTPUT_MD_LEGIBLE,
    OUTPUT_MD_ILLEGIBLE,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(
        char for char in text
        if not unicodedata.combining(char)
    )


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(text))


def get_original_files_info(
    md_file_path: Path,
    is_illegible: bool,
) -> list[dict]:
    """Retorna TODOS os arquivos originais que geraram este .md.

    Correções em relação à versão anterior:
    - antes usava glob(f"{stem}.*"), que também casava arquivos cujo nome
      apenas começa igual (ex.: "ata.*" pegava "ata.v2.pdf");
    - antes registrava só o primeiro arquivo quando PDF e XLSX tinham o mesmo
      nome-base (o outro nunca aparecia na busca).
    """
    base_dir = Path(
        OUTPUT_MD_ILLEGIBLE if is_illegible else OUTPUT_MD_LEGIBLE
    )

    try:
        relative_path = md_file_path.relative_to(base_dir)
    except ValueError:
        return []

    original_folder = Path(INPUT_DIR) / relative_path.parent
    if not original_folder.is_dir():
        return []

    stem_name = relative_path.stem
    matches = sorted(
        p for p in original_folder.iterdir()
        if p.is_file()
        and p.stem == stem_name
        and p.suffix.lower() in EXTENSIONS
    )

    return [
        {
            "nome_arquivo": original_file.name,
            "caminho_absoluto": str(original_file),
            "status_leitura": "Ilegivel" if is_illegible else "Legivel",
        }
        for original_file in matches
    ]


def build_index():
    print("Iniciando reindexação lexical e semântica...", flush=True)

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR)
    )

    try:
        chroma_client.delete_collection("gerop_acervo")
    except Exception:
        pass

    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
    )

    collection = chroma_client.create_collection(
        name="gerop_acervo",
        embedding_function=embedding_function,
    )

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    md_total = 0
    indexed_paths = set()
    missing_originals = []

    bm25_documents = []
    bm25_records = []
    chroma_batch = []
    chunk_id = 0
    batch_size = 128

    sources = [
        (Path(OUTPUT_MD_LEGIBLE), False),
        (Path(OUTPUT_MD_ILLEGIBLE), True),
    ]

    for source_dir, is_illegible in sources:
        if not source_dir.exists():
            continue

        illegible_dir = Path(OUTPUT_MD_ILLEGIBLE).resolve()

        for root, dirs, files in os.walk(source_dir):
            if not is_illegible:
                # a pasta de ilegíveis pode estar dentro da de legíveis:
                # não percorrer duas vezes nem tratar como legível
                dirs[:] = [
                    d for d in dirs
                    if (Path(root) / d).resolve() != illegible_dir
                ]

            for file_name in files:
                if not file_name.lower().endswith(".md"):
                    continue

                md_total += 1

                md_path = Path(root) / file_name
                infos = get_original_files_info(md_path, is_illegible)

                if not infos:
                    missing_originals.append(md_path)
                    print(
                        f"Arquivo original não encontrado para: {md_path}",
                        flush=True,
                    )
                    continue

                indexed_paths.update(
                    Path(info["caminho_absoluto"]) for info in infos
                )

                content = md_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                markdown_splits = markdown_splitter.split_text(content)

                if markdown_splits:
                    final_chunks = text_splitter.split_documents(
                        markdown_splits
                    )
                else:
                    final_chunks = text_splitter.create_documents([content])

                for chunk in final_chunks:
                    chunk_text = chunk.page_content.strip()
                    if not chunk_text:
                        continue

                    headers = " ".join(
                        str(value)
                        for value in chunk.metadata.values()
                        if value
                    )

                    # Inclui o contexto dos títulos na busca, mas não duplica
                    # o nome do arquivo em todos os chunks do BM25.
                    search_text = f"{headers}\n{chunk_text}".strip()

                    for info in infos:
                        meta = {
                            **info,
                            "header_context": headers,
                        }

                        bm25_documents.append(tokenize_text(search_text))
                        bm25_records.append(
                            {
                                "text": chunk_text,
                                "search_text": search_text,
                                "meta": meta,
                            }
                        )

                        chroma_batch.append(
                            {
                                "id": f"id_{chunk_id}",
                                "document": search_text,
                                "metadata": meta,
                            }
                        )

                        chunk_id += 1

                    if len(chroma_batch) >= batch_size:
                        collection.add(
                            ids=[item["id"] for item in chroma_batch],
                            documents=[
                                item["document"] for item in chroma_batch
                            ],
                            metadatas=[
                                item["metadata"] for item in chroma_batch
                            ],
                        )
                        chroma_batch.clear()

    if chroma_batch:
        collection.add(
            ids=[item["id"] for item in chroma_batch],
            documents=[item["document"] for item in chroma_batch],
            metadatas=[item["metadata"] for item in chroma_batch],
        )

    bm25_path = CHROMA_DB_DIR / "bm25.pkl"

    if bm25_documents:
        bm25 = BM25Okapi(bm25_documents)
        temporary_path = CHROMA_DB_DIR / "bm25.pkl.tmp"

        with temporary_path.open("wb") as file:
            pickle.dump(
                {
                    "bm25": bm25,
                    "data": bm25_records,
                },
                file,
            )

        temporary_path.replace(bm25_path)
    else:
        # Evita servir um BM25 antigo se esta indexação não encontrou dados.
        bm25_path.unlink(missing_ok=True)

    print(
        f"Indexação concluída: {chunk_id} trechos indexados.",
        flush=True,
    )

    # --- Relatório de cobertura: um índice incompleto não pode passar batido
    on_disk = {
        Path(root) / name
        for root, _, names in os.walk(INPUT_DIR)
        for name in names
        if Path(name).suffix.lower() in EXTENSIONS
    }
    not_indexed = sorted(on_disk - indexed_paths)
    print(
        f"Cobertura: {len(on_disk) - len(not_indexed)} de {len(on_disk)} "
        f"arquivos originais indexados | {md_total} .md lidos | "
        f"{len(missing_originals)} .md sem arquivo original.",
        flush=True,
    )
    if not_indexed:
        print(
            f"ATENÇÃO: {len(not_indexed)} arquivo(s) NÃO estão no índice "
            "(sem .md correspondente - rode converter.py):",
            flush=True,
        )
        for path in not_indexed[:15]:
            print(f"   - {path}", flush=True)
        if len(not_indexed) > 15:
            print(f"   ... e mais {len(not_indexed) - 15}", flush=True)


if __name__ == "__main__":
    build_index()