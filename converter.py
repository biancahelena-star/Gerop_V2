import os
import sys
import subprocess
from pathlib import Path

# --- MODO TRABALHADOR (Docling Rápido) ---
if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    file_path = Path(sys.argv[2])
    target_path = Path(sys.argv[3])
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(str(file_path))
    md_content = result.document.export_to_markdown()

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    sys.exit(0)

# --- FUNÇÃO DE OCR FLEXÍVEL ---
def flex_ocr_scanned_pdf(file_path):
    """OCR permissivo para capturar textos em baixa resolução ou ruidosos."""
    try:
        import pypdfium2 as pdfium
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR()
        pdf = pdfium.PdfDocument(str(file_path))
        full_text = []

        for page_num, page in enumerate(pdf, start=1):
            image = page.render(scale=1.5).to_pil()
            result, _ = engine(image)
            
            if not result:
                continue

            page_lines = []
            for item in result:
                text_content = item[1].strip()
                confidence = float(item[2])
                
                if confidence > 0.2 and len(text_content) > 0:
                    page_lines.append(text_content)

            if page_lines:
                page_text = "\n".join(page_lines)
                full_text.append(f"<!-- Pagina {page_num} -->\n{page_text}")

        return "\n\n".join(full_text)

    except Exception:
        return ""

# --- FUNÇÃO DE GERAR METADADOS DE BUSCA NA PASTA SEPARADA ---
def create_search_metadata_stub(file_path, target_path_illegible, input_dir):
    """Gera um Markdown de metadados dentro da pasta dedicada de ilegíveis."""
    relative_path = file_path.relative_to(input_dir)
    folder_context = " > ".join(relative_path.parts[:-1]) if len(relative_path.parts) > 1 else "Raiz"
    clean_tags = file_path.stem.replace('_', ' ').replace('-', ' ').replace('.', ' ')

    stub_content = f"""# METADADOS DE ARQUIVO ILEGÍVEL: {file_path.name}

> **Documento Ilegível:** Este arquivo foi movido para a estrutura de ilegíveis por apresentar baixa resolução ou ausência de camada de texto legível para OCR.

## Informações do Documento Original
* **Nome do Arquivo:** `{file_path.name}`
* **Caminho de Origem:** `{folder_context}`
* **Formato Original:** `{file_path.suffix.upper()}`
* **Tamanho do Arquivo:** `{file_path.stat().st_size / 1024:.2f} KB`

## Metadados para Indexação
* **Termos do Título:** {clean_tags}
* **Estrutura de Pastas:** {folder_context.replace(' > ', ' ')}
"""
    target_path_illegible.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path_illegible, "w", encoding="utf-8") as f:
        f.write(stub_content)

# --- ORQUESTRADOR PRINCIPAL ---
from config import (
    INPUT_DIR as input_dir,
    OUTPUT_MD_LEGIBLE as output_dir_legible,
    OUTPUT_MD_ILLEGIBLE as output_dir_illegible,
    EXTENSIONS as extensions,
    MIN_CHARACTERS,
)

print("Iniciando conversao com SEPARAÇÃO DE ILEGÍVEIS...", flush=True)

CREATE_NEW_PROCESS_GROUP = 0x00000200

for root, _, files in os.walk(input_dir):
    for file in files:
        file_path = Path(root) / file
        if file_path.suffix.lower() in extensions:
            relative_path = file_path.relative_to(input_dir)
            
            target_path_legible = output_dir_legible / relative_path.with_suffix(".md")
            target_path_illegible = output_dir_illegible / relative_path.with_suffix(".md")
            
            # Pula se o arquivo já existir validado em qualquer uma das pastas
            if target_path_legible.exists():
                try:
                    with open(target_path_legible, "r", encoding="utf-8") as check_f:
                        if len(check_f.read().strip()) >= MIN_CHARACTERS:
                            print(f"[PULADO - Legivel] Ja convertido: {file_path.name}", flush=True)
                            continue
                except Exception:
                    pass

            if target_path_illegible.exists():
                print(f"[PULADO - Ilegivel] Ja mapeado na pasta de ilegíveis: {file_path.name}", flush=True)
                continue

            target_path_legible.parent.mkdir(parents=True, exist_ok=True)
            print(f"[PROCESSANDO] {file_path.name} ...", flush=True)

            cmd = [sys.executable, __file__, "--worker", str(file_path), str(target_path_legible)]
            docling_success = False
            
            # 1. Tenta extração rápida digital
            try:
                result = subprocess.run(
                    cmd, 
                    timeout=15, 
                    capture_output=True, 
                    creationflags=CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                )
                if result.returncode == 0 and target_path_legible.exists():
                    size = len(target_path_legible.read_text(encoding="utf-8", errors="ignore").strip())
                    if size >= MIN_CHARACTERS:
                        print(f"[CONCLUIDO - Legivel] -> {target_path_legible.name} ({size} chars)", flush=True)
                        docling_success = True
            except Exception:
                pass

            # 2. Se falhar, tenta OCR Permissivo
            if not docling_success and file_path.suffix.lower() == ".pdf":
                ocr_text = flex_ocr_scanned_pdf(file_path)
                if len(ocr_text.strip()) >= MIN_CHARACTERS:
                    with open(target_path_legible, "w", encoding="utf-8") as f:
                        f.write(ocr_text)
                    print(f"[CONCLUIDO - OCR Legivel] -> {target_path_legible.name} ({len(ocr_text)} chars)", flush=True)
                    continue

            # 3. Se o OCR falhar, remove resquícios da pasta legível e grava na pasta de ilegíveis
            if not docling_success:
                if target_path_legible.exists():
                    try:
                        target_path_legible.unlink()
                    except Exception:
                        pass
                
                create_search_metadata_stub(file_path, target_path_illegible, input_dir)
                print(f"[ILEGIVEL -> SEPARADO] Enviado para pasta de ilegíveis: {file_path.name}", flush=True)

print("\nProcessamento e separacao finalizados com sucesso!", flush=True)