import time
import os
import subprocess
import sys
from pathlib import Path
from config import INPUT_DIR, OUTPUT_MD_LEGIBLE, OUTPUT_MD_ILLEGIBLE, EXTENSIONS

CHECK_INTERVAL_SECONDS = 30

def is_file_ready(file_path):
    try:
        with open(file_path, "rb") as f:
            f.read(1024)
        return True
    except (IOError, PermissionError):
        return False

def check_and_run():
    pending_files = []

    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in EXTENSIONS:
                relative_path = file_path.relative_to(INPUT_DIR)
                target_legible = OUTPUT_MD_LEGIBLE / relative_path.with_suffix(".md")
                target_illegible = OUTPUT_MD_ILLEGIBLE / relative_path.with_suffix(".md")

                if not target_legible.exists() and not target_illegible.exists():
                    if is_file_ready(file_path):
                        pending_files.append(file_path)

    if pending_files:
        print(f"\n[NOVO DOCUMENTO] Detectado(s) {len(pending_files)} novo(s) arquivo(s) no OneDrive!", flush=True)
        
        print("1. Executando extrator OCR...", flush=True)
        subprocess.run([sys.executable, "converter.py"])

        print("2. Atualizando banco de busca híbrida...", flush=True)
        subprocess.run([sys.executable, "indexer.py"])

        print("3. Atualização concluída! Novos documentos disponíveis no Chatbot.\n", flush=True)

if __name__ == "__main__":
    print("==========================================================================", flush=True)
    print(" MONITOR AUTOMÁTICO DE PETIÇÕES E PROCESSOS DO SEI (GEROP)", flush=True)
    print("==========================================================================\n", flush=True)

    while True:
        try:
            check_and_run()
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nMonitor encerrado.")
            break
        except Exception as e:
            print(f"[ERRO] {e}", flush=True)
            time.sleep(CHECK_INTERVAL_SECONDS)