"""
Servidor local somente-leitura dos arquivos originais (PDF / XLSX ...).

Por que existe: navegadores bloqueiam links file:/// vindos de uma página
http://localhost (o Streamlit). Servindo a pasta de entrada por HTTP, o link
devolvido no chat abre o PDF direto no navegador e baixa as planilhas, no
formato original.

Segurança: só serve arquivos DENTRO da pasta de entrada e só das extensões
permitidas; não lista diretórios. Por padrão escuta apenas em 127.0.0.1.
"""
from __future__ import annotations

import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

DEFAULT_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".html", ".htm"}


def start_file_server(root, host: str = "127.0.0.1", port: int = 8765,
                      extensions=None):
    """Inicia o servidor em segundo plano.

    Retorna o servidor, ou None se a porta já estiver em uso (por exemplo,
    outra instância do app já está servindo os arquivos)."""
    root = Path(root).resolve()
    allowed = {e.lower() for e in (extensions or DEFAULT_EXTENSIONS)}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *args):  # silencioso
            pass

        def list_directory(self, path):
            self.send_error(403, "Listagem de pastas desabilitada")
            return None

        def send_head(self):
            target = Path(self.translate_path(self.path))
            try:
                target.resolve().relative_to(root)
            except (ValueError, OSError):
                self.send_error(403, "Fora da pasta permitida")
                return None
            if target.is_dir() or target.suffix.lower() not in allowed:
                self.send_error(403, "Tipo de arquivo não permitido")
                return None
            self._served_name = target.name
            return super().send_head()

        def end_headers(self):
            name = getattr(self, "_served_name", None)
            if name:
                # PDF abre no navegador; planilhas/HTML viram download.
                disposition = "inline" if name.lower().endswith(".pdf") else "attachment"
                self.send_header(
                    "Content-Disposition",
                    f"{disposition}; filename*=UTF-8''{quote(name)}",
                )
                self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # Navegador fechou o PDF/aba no meio do download: não é erro.
            if isinstance(sys.exc_info()[1], (ConnectionError, BrokenPipeError)):
                return
            super().handle_error(request, client_address)

    try:
        server = Server((host, port), Handler)
    except OSError:
        return None

    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
