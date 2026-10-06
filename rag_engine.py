"""
Motor de busca do Especialista GEROP.

Objetivo: devolver TODOS os arquivos que correspondem à busca, do que tem
mais correspondência lexical (mesma sequência numérica/alfabética ou muito
próxima) para o que tem menos (apenas alguns números ou palavras).

Como funciona
-------------
1. A consulta vira "itens": números (>= 4 dígitos, sem zeros à esquerda) e
   palavras (sem acento, sem stopwords, singular simples).
2. Cada item é comparado com o vocabulário do acervo em 4 níveis:
      exato (1.0) > contém/contido (0.6-0.9) > muito próximo (0.55-0.85)
      > trecho em comum (0.30-0.45)
   Palavras seguem a mesma lógica (exata, prefixo, erro de digitação).
3. Cada item tem um peso (IDF): termos que existem em quase todos os arquivos
   ("fundo", "gerop") pesam pouco; números e termos raros pesam muito. É isso
   que evita devolver sempre os mesmos arquivos.
4. A nota do arquivo = soma(peso * força do melhor casamento) / soma(pesos),
   com pequena preferência por onde casou: nome (1.00) > pasta (0.95) >
   conteúdo (0.90). Empates: nº de termos no nome, depois frequência do termo
   no texto.
   Números parecidos só valem no CONTEÚDO se forem MUITO parecidos (planilhas
   grandes têm milhares de números e casariam por acaso com qualquer busca).
5. Se a busca rigorosa devolver poucos arquivos, roda uma 2ª passada mais
   permissiva (mais lenta) para ampliar a lista.
6. A busca semântica (Chroma) só COMPLEMENTA: o que ela traz e não tem nenhum
   termo em comum aparece no fim, em faixa própria.

A resposta já sai pronta em Markdown (links para o arquivo original, com
informações básicas), então o histórico do chat guarda os links também.
"""
from __future__ import annotations

import math
import os
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

import config as cfg

try:  # opcional: acelera muito a comparação aproximada (pip install rapidfuzz)
    from rapidfuzz import fuzz as _rf_fuzz
except ImportError:  # pragma: no cover
    _rf_fuzz = None


# --------------------------------------------------------------------------
# Texto
# --------------------------------------------------------------------------
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos",
    "das", "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "e",
    "ou", "que", "qual", "quais", "como", "sobre", "me", "meu", "minha",
    "arquivo", "arquivos", "documento", "documentos", "processo", "processos",
    "localize", "localizar", "buscar", "busque", "busca", "procure", "achar",
    "encontre", "encontrar", "mostre", "mostrar", "traga", "todos", "todas",
    "numero", "nr", "sei", "pdf", "xlsx", "xls", "planilha", "planilhas",
    "relacionado", "relacionados", "referente", "referentes",
}


def _build_fold_table() -> dict[int, str]:
    """Tabela (mesmo tamanho de caractere) para remover acentos rapidamente."""
    table: dict[int, str] = {}
    for codepoint in range(0xC0, 0x250):
        char = chr(codepoint)
        base = "".join(
            c for c in unicodedata.normalize("NFKD", char)
            if not unicodedata.combining(c)
        )
        if base and base[0] != char:
            table[codepoint] = base[0].lower()
    return table


_FOLD_TABLE = _build_fold_table()


def fold(text: Any) -> str:
    """minúsculas + sem acento, preservando o tamanho (útil p/ trechos)."""
    return str(text or "").lower().translate(_FOLD_TABLE)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RUN_RE = re.compile(r"\d+")
_JOINED_NUM_RE = re.compile(r"\d+(?:[./\-]\d+)+")  # 12345.678901/2024-12


def stem(token: str) -> str:
    return token[:-1] if len(token) > 4 and token.endswith("s") else token


def words(text: Any, keep_short_digits: bool = False) -> list[str]:
    out = []
    for token in _TOKEN_RE.findall(fold(text)):
        if token.isdigit():
            if keep_short_digits and len(token) < 4:
                out.append(token)  # "098" e "98" são equivalentes só na 2ª opção
            continue
        if len(token) < 2 or token in STOPWORDS:
            continue
        out.append(stem(token))
    return out


def _norm_num(digits: str) -> str:
    normalized = digits.lstrip("0")
    return normalized if len(normalized) >= 4 else ""


def numbers(text: Any) -> set[str]:
    """Sequências numéricas (>= 4 dígitos), sem zeros à esquerda.
    Também junta números separados por . / - (ex.: 12345.678901/2024-12)."""
    text = str(text or "")
    found = set()
    for joined in _JOINED_NUM_RE.findall(text):
        n = _norm_num(re.sub(r"\D", "", joined))
        if n:
            found.add(n)
    for run in _RUN_RE.findall(text):
        n = _norm_num(run)
        if n:
            found.add(n)
    return found


def parse_query(query: str) -> list[tuple[str, str]]:
    """Retorna itens [("num", "34368688"), ("word", "regulamento"), ...]."""
    query = str(query or "")
    items: list[tuple[str, str]] = []

    def add(kind: str, value: str) -> None:
        if value and (kind, value) not in items:
            items.append((kind, value))

    for joined in _JOINED_NUM_RE.findall(query):
        add("num", _norm_num(re.sub(r"\D", "", joined)))

    rest = _JOINED_NUM_RE.sub(" ", query)
    for run in _RUN_RE.findall(rest):
        add("num", _norm_num(run))

    # "0000 3436 8688" digitado com espaços -> também vale como um número só
    if re.fullmatch(r"[\d\s./\-]+", query.strip() or "x"):
        compact = re.sub(r"\D", "", query)
        if len(compact) >= 8:
            add("num", _norm_num(compact))

    for word in words(rest, keep_short_digits=True):
        add("word", word)
    return items


# --------------------------------------------------------------------------
# Constantes de ranking
# --------------------------------------------------------------------------
_LOCS = (("name", 1.00), ("folder", 0.95), ("text", 0.90))
_LOC_LABEL = {
    "name": "no nome do arquivo",
    "folder": "na pasta/caminho",
    "text": "no conteúdo",
}
_HOW_LABEL = {
    "contem": "parcial (contém / está contido)",
    "proximo": "muito próximo",
    "parcial": "com trecho em comum",
}
MIN_SCORE = 0.08
LOOSE_MIN_SCORE = 0.05     # piso da 2ª passada (ampliada)
LOOSE_FACTOR = 0.80        # resultados da 2ª passada valem 80% da nota
DEFAULT_MAX_RESULTS = 100
DEFAULT_MIN_RESULTS = 40   # abaixo disso, roda a passada ampliada

TIERS = [
    (0.90, "🟢 Correspondência exata"),
    (0.70, "🟡 Alta correspondência"),
    (0.45, "🟠 Correspondência média"),
    (0.00, "🔴 Correspondência parcial (apenas alguns números/palavras)"),
]
SEMANTIC_TIER = "🔵 Somente busca semântica (sem número/termo em comum)"

_FILE_KIND = {
    ".pdf": "PDF",
    ".xlsx": "Planilha Excel (XLSX)",
    ".xls": "Planilha Excel (XLS)",
    ".html": "HTML",
    ".htm": "HTML",
}


def _tier_for(score: float) -> str:
    for minimum, label in TIERS:
        if score >= minimum:
            return label
    return TIERS[-1][1]


def _human_size(size: int | None) -> str:
    if size is None:
        return "tamanho desconhecido"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|~$])")


def _md(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text)


def _put(out: dict, term: str, strength: float, how: str) -> None:
    current = out.get(term)
    if current is None or strength > current[0]:
        out[term] = (strength, how)


# --------------------------------------------------------------------------
# Motor
# --------------------------------------------------------------------------
class HybridRAG:
    def __init__(self, records=None, input_dir=None, db_dir=None,
                 use_semantic: bool = True):
        self.input_dir = Path(input_dir) if input_dir else Path(cfg.INPUT_DIR)
        self.db_dir = Path(db_dir) if db_dir else Path(cfg.CHROMA_DB_DIR)
        self.collection = None

        if records is None:
            records = self._load_records()
            if use_semantic:
                self._connect_chroma()

        self._build_index(records)

    # ---------------------------------------------------------- carga
    def _load_records(self) -> list:
        bm25_path = self.db_dir / "bm25.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"Índice lexical não encontrado: {bm25_path}. "
                "Execute novamente o indexador."
            )
        with bm25_path.open("rb") as file:
            saved = pickle.load(file)
        records = saved.get("data", [])
        if not isinstance(records, list):
            raise ValueError("O campo 'data' de bm25.pkl precisa ser uma lista.")
        return records

    def _connect_chroma(self) -> None:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.db_dir))
            self.collection = client.get_collection(name="gerop_acervo")
        except Exception:
            # A busca lexical continua funcionando sem a coleção semântica.
            self.collection = None

    def _folder_parts(self, path: Path) -> tuple[str, ...]:
        try:
            return path.relative_to(self.input_dir).parent.parts
        except ValueError:
            return path.parent.parts[-3:]

    def _prepare_file(self, path: str, name: str, status: str) -> dict:
        p = Path(path)
        parts = self._folder_parts(p)
        return {
            "path": path,
            "name": name or p.name,
            "status": status,
            "ext": p.suffix.lower(),
            "folder": " / ".join(parts) if parts else "(raiz)",
            "folder_src": " ".join(parts),
            "chunks": [],
        }

    def _build_index(self, records: list) -> None:
        by_path: dict[str, dict] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            meta = record.get("meta") or {}
            path = str(meta.get("caminho_absoluto") or "")
            if not path:
                continue
            file = by_path.get(path)
            if file is None:
                file = self._prepare_file(
                    path,
                    str(meta.get("nome_arquivo") or ""),
                    str(meta.get("status_leitura") or "Legivel"),
                )
                by_path[path] = file
            text = str(record.get("search_text") or record.get("text") or "")
            if text:
                file["chunks"].append(text)

        self.files = list(by_path.values())
        self.file_index = {f["path"]: i for i, f in enumerate(self.files)}
        self.inv = {loc: defaultdict(set) for loc, _ in _LOCS}
        self.num = {loc: defaultdict(set) for loc, _ in _LOCS}

        for fi, file in enumerate(self.files):
            stem_name = Path(file["name"]).stem
            full_text = "\n".join(file["chunks"])
            sources = {
                "name": stem_name,
                "folder": file["folder_src"],
                "text": full_text,
            }
            for loc, source in sources.items():
                tokens = words(source, keep_short_digits=(loc != "text"))
                if loc == "text":
                    file["tf"] = Counter(tokens)  # frequência (desempate)
                for token in set(tokens):
                    self.inv[loc][token].add(fi)
                for number in numbers(source):
                    self.num[loc][number].add(fi)

        self.word_vocab: set[str] = set()
        self.num_vocab: set[str] = set()
        for loc, _ in _LOCS:
            self.word_vocab.update(self.inv[loc])
            self.num_vocab.update(self.num[loc])

        self.words_by_first: dict[str, list[str]] = defaultdict(list)
        self.short_by_value: dict[int, list[str]] = defaultdict(list)
        for word in self.word_vocab:
            self.words_by_first[word[0]].append(word)
            if word.isdigit():  # números curtos (<4 dígitos) do nome/pasta
                self.short_by_value[int(word)].append(word)

    # ---------------------------------------------------------- expansão
    def _expand_num(self, q: str, loose: bool = False):
        """{local: {número_do_acervo: (força, tipo)}} para um número da busca.

        Regras por local (nome/pasta são identificadores; conteúdo é ruidoso):
          exato           -> qualquer local
          contém          -> busca dentro de número maior (>=4 dígitos no
                             nome/pasta, >=6 no conteúdo) ou número do acervo
                             (>=6 dígitos, >=70% do tamanho) dentro da busca
          muito próximo   -> 1 dígito trocado/faltando (nome/pasta: >=6 dígitos;
                             conteúdo: >=8 dígitos e 85% de semelhança)
          trecho em comum -> só nome/pasta (>=4 dígitos seguidos, >=50%);
                             no conteúdo só trechos longos (>=8 e >=75%)
        """
        length = len(q)
        grams = {q[i:i + 4] for i in range(length - 3)}
        need = max(1, math.ceil(length * 0.5) - 3)
        by_loc: dict[str, dict[str, tuple[float, str]]] = {}

        for loc, _ in _LOCS:
            is_text = loc == "text"
            vocab = self.num[loc]
            min_contain = 6 if is_text else 4
            near_len = 8 if is_text else 6
            near_min = 0.85 if is_text else 0.80
            out: dict[str, tuple[float, str]] = {}
            if q in vocab:
                out[q] = (1.0, "exato")

            for d in vocab:
                if d == q:
                    continue
                ld = len(d)
                if ld > length and length >= min_contain and q in d:
                    _put(out, d, 0.60 + 0.30 * length / ld, "contem")
                    continue
                if ld < length and ld >= 6 and d in q and ld / length >= 0.7:
                    _put(out, d, 0.60 + 0.30 * ld / length, "contem")
                    continue
                if sum(1 for g in grams if g in d) < need:
                    continue
                if length >= near_len and ld >= near_len and abs(length - ld) <= 1:
                    r = (
                        _rf_fuzz.ratio(q, d) / 100.0
                        if _rf_fuzz else SequenceMatcher(None, q, d).ratio()
                    )
                    if r >= near_min:
                        _put(out, d,
                             0.55 + 0.30 * (r - near_min) / (1 - near_min),
                             "proximo")
                        continue
                lcs = SequenceMatcher(None, q, d, autojunk=False) \
                    .find_longest_match(0, length, 0, ld).size
                if is_text:
                    ok = (lcs >= 8 and lcs / length >= 0.75) or (
                        loose and lcs >= 5 and lcs / length >= 0.6)
                else:
                    ok = lcs >= 4 and lcs / length >= (0.4 if loose else 0.5)
                if ok:
                    _put(out, d, 0.15 + 0.30 * (lcs / length), "parcial")
            by_loc[loc] = out
        return by_loc

    def _expand_word(self, t: str, loose: bool = False):
        out: dict[str, tuple[float, str]] = {}
        if t in self.word_vocab:
            out[t] = (1.0, "exato")
        length = len(t)
        if t.isdigit():
            # "098" encontra "098" (exato) e, com força menor, "98"
            for w in self.short_by_value.get(int(t), ()):
                if w != t:
                    _put(out, w, 0.80, "proximo")
            return out
        if length < 4:
            return out

        prefix_min = 0.5 if loose else 0.6
        fuzzy_min = 0.75 if loose else 0.82
        fuzzy_len = 5 if loose else 6
        candidates = self.word_vocab if loose else self.words_by_first.get(t[0], ())
        matcher = None
        if _rf_fuzz is None:
            matcher = SequenceMatcher(None)
            matcher.set_seq2(t)

        for w in candidates:
            if w == t:
                continue
            lw = len(w)
            if lw >= 4 and (w.startswith(t) or t.startswith(w)):
                ratio = min(length, lw) / max(length, lw)
                if ratio >= prefix_min:
                    _put(out, w, 0.55 + 0.30 * ratio, "contem")
                    continue
            if loose and lw >= 4 and (t in w or w in t):
                _put(out, w, 0.35, "contem")
                continue
            if w[0] != t[0] or length < fuzzy_len or lw < fuzzy_len \
                    or abs(length - lw) > 2:
                continue
            if _rf_fuzz is not None:
                r = _rf_fuzz.ratio(t, w) / 100.0
            else:
                matcher.set_seq1(w)
                if (matcher.real_quick_ratio() < fuzzy_min
                        or matcher.quick_ratio() < fuzzy_min):
                    continue
                r = matcher.ratio()
            if r >= fuzzy_min:
                _put(out, w,
                     0.40 + 0.40 * (r - fuzzy_min) / (1 - fuzzy_min),
                     "proximo")
        return out

    def _weight(self, kind: str, value: str, n_files: int) -> float:
        index = self.num if kind == "num" else self.inv
        files: set[int] = set()
        for loc, _ in _LOCS:
            files.update(index[loc].get(value, ()))
        df = len(files)
        idf = math.log(1 + (n_files - df + 0.5) / (df + 0.5))
        weight = max(0.05, idf)
        return weight * 1.5 if kind == "num" else weight

    # ---------------------------------------------------------- busca
    def _score_pass(self, items, weights, loose: bool):
        total_weight = sum(weights)
        per_file: dict[int, dict[int, tuple]] = defaultdict(dict)
        for i, (kind, value) in enumerate(items):
            if kind == "num":
                by_loc = self._expand_num(value, loose)
                index = self.num
            else:
                expansion = self._expand_word(value, loose)
                by_loc = {loc: expansion for loc, _ in _LOCS}
                index = self.inv
            for loc, mult in _LOCS:
                for term, (strength, how) in by_loc[loc].items():
                    for fi in index[loc].get(term, ()):
                        val = strength * mult
                        current = per_file[fi].get(i)
                        if current is None or val > current[0]:
                            per_file[fi][i] = (val, loc, term, how, strength)
        return {
            fi: (sum(weights[i] * h[0] for i, h in hits.items()) / total_weight, hits)
            for fi, hits in per_file.items()
        }

    def search_lexical_ranked(self, query: str,
                              max_results: int | None = DEFAULT_MAX_RESULTS,
                              semantic_candidates: int = 25,
                              min_results: int = DEFAULT_MIN_RESULTS) -> list[dict]:
        items = parse_query(query)
        if not items:
            return []

        n_files = max(1, len(self.files))
        weights = [self._weight(k, v, n_files) for k, v in items]

        # 1ª passada: rigorosa
        candidates = {
            fi: v for fi, v in self._score_pass(items, weights, False).items()
            if v[0] >= MIN_SCORE
        }
        # 2ª passada (mais lenta): só se a lista ficou curta
        if len(candidates) < min_results:
            for fi, (score, hits) in self._score_pass(items, weights, True).items():
                if fi in candidates:
                    continue
                score *= LOOSE_FACTOR
                if score >= LOOSE_MIN_SCORE:
                    candidates[fi] = (score, hits)

        scored = []
        for fi, (score, hits) in candidates.items():
            file = self.files[fi]
            name_hits = sum(1 for h in hits.values() if h[1] == "name")
            tf = sum(math.log1p(file["tf"].get(h[2], 0))
                     for h in hits.values() if h[1] == "text")
            scored.append((fi, score, name_hits, tf, hits))

        scored.sort(key=lambda s: (
            -round(s[1], 2), -s[2], -len(s[4]), -s[3],
            self.files[s[0]]["name"].lower(),
        ))
        if max_results:
            scored = scored[:max_results]

        results = [
            self._make_result(self.files[fi], score, hits, items)
            for fi, score, _, _, hits in scored
        ]

        room = (max_results - len(results)) if max_results else semantic_candidates
        limit = max(0, min(semantic_candidates, room))
        results.extend(
            self._semantic_results(query, {r["path"] for r in results}, limit)
        )
        return results

    def _semantic_results(self, query: str, exclude: set[str], limit: int):
        if self.collection is None or limit <= 0:
            return []
        out = []
        try:
            total = self.collection.count()
            if total <= 0:
                return []
            data = self.collection.query(
                query_texts=[query],
                n_results=min(total, max(60, limit * 4)),
                include=["documents", "metadatas", "distances"],
            )
            documents = data.get("documents", [[]])[0]
            metadatas = data.get("metadatas", [[]])[0]
            distances = data.get("distances", [[]])[0]
            for text, meta, distance in zip(documents, metadatas, distances):
                meta = meta or {}
                path = str(meta.get("caminho_absoluto") or "")
                if not path or path in exclude:
                    continue
                exclude.add(path)
                file = self.files[self.file_index[path]] \
                    if path in self.file_index else self._prepare_file(
                        path, str(meta.get("nome_arquivo") or ""),
                        str(meta.get("status_leitura") or "Legivel"))
                if not file["chunks"]:
                    file = {**file, "chunks": [str(text or "")]}
                result = self._make_result(file, 0.0, {}, [])
                result["tier"] = SEMANTIC_TIER
                result["semantic_only"] = True
                result["semantic_distance"] = float(distance)
                result["reasons"] = [
                    "Recuperado pela similaridade de significado "
                    "(nenhum número ou termo da busca aparece no arquivo)"
                ]
                out.append(result)
                if len(out) >= limit:
                    break
        except Exception:
            return []
        return out

    # ---------------------------------------------------------- resultado
    def link_for(self, path: str) -> str:
        p = Path(path)
        try:
            rel = p.relative_to(self.input_dir).as_posix()
        except ValueError:
            return p.as_uri() if p.is_absolute() else ""
        host = getattr(cfg, "FILE_SERVER_PUBLIC_HOST", "localhost")
        port = getattr(cfg, "FILE_SERVER_PORT", 8765)
        return f"http://{host}:{port}/{quote(rel)}"

    @staticmethod
    def _numbers_in_name(name: str) -> list[str]:
        found: list[str] = []
        stem_name = Path(name).stem
        for joined in _JOINED_NUM_RE.findall(stem_name):
            if len(re.sub(r"\D", "", joined)) >= 6 and joined not in found:
                found.append(joined)
        for run in re.findall(r"\d{6,}", _JOINED_NUM_RE.sub(" ", stem_name)):
            if run not in found:
                found.append(run)
        return found

    @staticmethod
    def _parse_sei(name: str) -> dict | None:
        """"[011]-000034368688_Anexo.pdf" -> ordem 011, nº SEI, assunto."""
        match = re.match(r"^\[(\d+)\]-(\d+)_?(.*)$", Path(name).stem)
        if not match:
            return None
        ordem, sei_id, resto = match.groups()
        return {
            "ordem": ordem,
            "sei": sei_id,
            "assunto": resto.replace("_", " ").strip() or "-",
        }

    @staticmethod
    def _clean_snippet(text: str) -> str:
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
        text = re.sub(r"[#>*|`]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _snippet(self, file: dict, term: str | None) -> str:
        if term:
            for chunk in file["chunks"][:300]:
                position = fold(chunk).find(term)
                if position >= 0:
                    start = max(0, position - 80)
                    end = min(len(chunk), position + len(term) + 120)
                    snippet = self._clean_snippet(chunk[start:end])
                    if snippet:
                        return ("…" if start else "") + snippet + "…"
        if file["status"] != "Ilegivel" and file["chunks"]:
            snippet = self._clean_snippet(file["chunks"][0])[:200]
            if snippet:
                return snippet + "…"
        return ""

    def _make_result(self, file: dict, score: float, hits: dict,
                     items: list[tuple[str, str]]) -> dict[str, Any]:
        path = file["path"]
        try:
            stat = os.stat(path)
            size, modified, exists = stat.st_size, stat.st_mtime, True
        except OSError:
            size, modified, exists = None, None, False

        reasons, missing, snippet_term = [], [], None
        best_text_val = -1.0
        for i, (kind, value) in enumerate(items):
            hit = hits.get(i)
            if hit is None:
                missing.append(value)
                continue
            val, loc, term, how, _ = hit
            what = "Número" if kind == "num" else "Termo"
            where = _LOC_LABEL[loc]
            if how == "exato":
                reasons.append(f"{what} exato {where}: {value}")
            else:
                reasons.append(
                    f"{what} {_HOW_LABEL[how]} {where}: {term} (busca: {value})"
                )
            if loc == "text" and val > best_text_val:
                best_text_val, snippet_term = val, term

        return {
            "path": path,
            "name": file["name"],
            "ext": file["ext"],
            "tipo": _FILE_KIND.get(file["ext"], file["ext"].upper().lstrip(".") or "Arquivo"),
            "legivel": file["status"] != "Ilegivel",
            "folder": file["folder"],
            "size": size,
            "modified": (
                datetime.fromtimestamp(modified).strftime("%d/%m/%Y")
                if modified else None
            ),
            "exists": exists,
            "link": self.link_for(path),
            "score": round(score, 4),
            "tier": _tier_for(score),
            "reasons": reasons,
            "missing": missing,
            "numbers_in_name": self._numbers_in_name(file["name"]),
            "sei": self._parse_sei(file["name"]),
            "snippet": self._snippet(file, snippet_term),
            "semantic_only": False,
        }

    # ---------------------------------------------------------- resposta
    def format_markdown(self, query: str, results: list[dict],
                        items: list[tuple[str, str]] | None = None) -> str:
        items = items if items is not None else parse_query(query)
        if not items:
            return ("Digite um número de processo (4 dígitos ou mais), um "
                    "código ou um termo para buscar.")

        nums = [v for k, v in items if k == "num"]
        terms = [v for k, v in items if k == "word"]
        criteria = []
        if nums:
            criteria.append("números: " + ", ".join(f"`{n}`" for n in nums))
        if terms:
            criteria.append("termos: " + ", ".join(f"`{t}`" for t in terms))
        criteria_text = " · ".join(criteria)

        if not results:
            return (f"Nenhum arquivo correspondente a {criteria_text}.\n\n"
                    "Tente só o número principal do processo ou menos palavras.")

        lexical = [r for r in results if not r["semantic_only"]]
        header = (
            f"Encontrei **{len(results)}** arquivo(s) para {criteria_text}, "
            "em ordem da maior para a menor correspondência lexical."
        )
        if lexical and len(lexical) != len(results):
            header += (f" ({len(lexical)} com termo/número em comum; "
                       f"{len(results) - len(lexical)} só por similaridade de significado.)")

        counts: dict[str, int] = {}
        for r in results:
            counts[r["tier"]] = counts.get(r["tier"], 0) + 1

        lines = [header, ""]
        current_tier = None
        for position, r in enumerate(results, start=1):
            if r["tier"] != current_tier:
                current_tier = r["tier"]
                lines += [f"### {current_tier} ({counts[current_tier]})", ""]

            title = f"**{position}.** "
            title += f"[{_md(r['name'])}]({r['link']})" if r["link"] else f"**{_md(r['name'])}**"
            meta = [r["tipo"], _human_size(r["size"])]
            if r["modified"]:
                meta.append(f"modificado em {r['modified']}")
            if not r["semantic_only"]:
                meta.append(f"correspondência {round(r['score'] * 100)}%")
            lines.append(title + " — " + " · ".join(meta))

            lines.append(f"- 📁 Pasta: {_md(r['folder'])}")
            if not r["legivel"]:
                lines.append("- ⚠️ Documento sem texto legível (digitalizado/imagem): "
                             "a busca considerou só o nome e a pasta")
            if not r["exists"]:
                lines.append("- ⚠️ Arquivo não encontrado na origem (o link pode não abrir)")
            if r["reasons"]:
                lines.append("- 🔎 Por que apareceu: " + "; ".join(_md(x) for x in r["reasons"]))
            if r["missing"] and not r["semantic_only"]:
                lines.append("- ➖ Sem correspondência para: " + ", ".join(f"`{m}`" for m in r["missing"]))
            if r["sei"]:
                lines.append(
                    f"- 🏷️ Documento nº {r['sei']['ordem']} do processo · "
                    f"Nº SEI `{r['sei']['sei']}` · Assunto: {_md(r['sei']['assunto'])}"
                )
            elif r["numbers_in_name"]:
                lines.append("- 🔢 Número(s) no nome: " + ", ".join(f"`{n}`" for n in r["numbers_in_name"]))
            if r["snippet"]:
                lines.append(f"- 💬 Trecho: {_md(r['snippet'])}")
            lines.append(f"- 🗂️ Caminho: `{r['path']}`")
            lines.append("")

        return "\n".join(lines)

    def generate_response(self, query: str,
                          max_results: int | None = DEFAULT_MAX_RESULTS,
                          llm_callback=None, semantic_candidates: int = 25,
                          min_results: int = DEFAULT_MIN_RESULTS):
        items = parse_query(query)
        if not items:
            return self.format_markdown(query, [], items), []
        results = self.search_lexical_ranked(
            query, max_results=max_results,
            semantic_candidates=semantic_candidates,
            min_results=min_results,
        )
        return self.format_markdown(query, results, items), results
