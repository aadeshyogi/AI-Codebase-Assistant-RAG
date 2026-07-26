"""
core.py
-------
Sab kuch ek jagah: repo cloning, AST-aware chunking (tree-sitter),
ChromaDB indexing, aur hybrid (semantic + BM25) retrieval.

Pehle ye teen alag files thi (chunker.py, indexer.py, retriever.py) —
ab sirf 2 files rakhne ke liye sab yaha merge kar diya hai.
UI-wala code (app.py) isse import karke use karega.
"""

import os
import shutil
import hashlib
import tempfile
from dataclasses import dataclass

import git
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser
from rank_bm25 import BM25Okapi

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# =========================================================
# 1) CHUNKER — AST-aware code chunking using tree-sitter
# =========================================================
#
# Char-count ke hisaab se split karne ki jagah, hum actual syntax
# tree parse karte hain aur poore functions/classes ko chunk ki
# tarah nikalte hain — isse retrieval me complete, meaningful
# code units milte hain, adhure fragments nahi.

PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjs.language())

CHUNK_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
}

EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", ".venv", "env",
    "dist", "build", ".next", "site-packages", ".idea", ".vscode",
}

MAX_FILE_SIZE_BYTES = 500_000  # bahut badi generated files skip karo


@dataclass
class CodeChunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    node_type: str
    name: str
    language: str


def _get_parser(language: str) -> Parser:
    if language == "python":
        return Parser(PY_LANGUAGE)
    elif language == "javascript":
        return Parser(JS_LANGUAGE)
    raise ValueError(f"Unsupported language: {language}")


def _extract_name(node, source_bytes: bytes) -> str:
    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
        if child.type == "property_identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
    return "<anonymous>"


def _walk_and_extract(node, source_bytes: bytes, language: str, chunks: list, file_path: str):
    node_types = CHUNK_NODE_TYPES[language]

    if node.type in node_types:
        name = _extract_name(node, source_bytes)
        content = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        chunks.append(CodeChunk(
            content=content,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            node_type=node.type,
            name=name,
            language=language,
        ))
        return

    for child in node.children:
        _walk_and_extract(child, source_bytes, language, chunks, file_path)


def chunk_file(file_path: str, repo_root: str) -> list[CodeChunk]:
    ext = os.path.splitext(file_path)[1]
    language = EXT_TO_LANG.get(ext)
    rel_path = os.path.relpath(file_path, repo_root)

    try:
        with open(file_path, "rb") as f:
            source_bytes = f.read()
    except (OSError, UnicodeDecodeError):
        return []

    if len(source_bytes) == 0 or len(source_bytes) > MAX_FILE_SIZE_BYTES:
        return []

    if language is None:
        text = source_bytes.decode("utf-8", errors="ignore")
        if not text.strip():
            return []
        return [CodeChunk(
            content=text,
            file_path=rel_path,
            start_line=1,
            end_line=text.count("\n") + 1,
            node_type="file",
            name=os.path.basename(file_path),
            language="text",
        )]

    parser = _get_parser(language)
    tree = parser.parse(source_bytes)

    chunks: list[CodeChunk] = []
    _walk_and_extract(tree.root_node, source_bytes, language, chunks, rel_path)

    if not chunks:
        text = source_bytes.decode("utf-8", errors="ignore")
        if text.strip():
            chunks.append(CodeChunk(
                content=text,
                file_path=rel_path,
                start_line=1,
                end_line=text.count("\n") + 1,
                node_type="file",
                name=os.path.basename(file_path),
                language=language,
            ))

    return chunks


def chunk_repo(repo_root: str) -> list[CodeChunk]:
    all_chunks: list[CodeChunk] = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]

        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            if ext not in EXT_TO_LANG:
                continue
            full_path = os.path.join(dirpath, fname)
            all_chunks.extend(chunk_file(full_path, repo_root))

    return all_chunks


# =========================================================
# 2) INDEXER — clone repo, embed chunks, store in ChromaDB
# =========================================================
#
# Ek repo = ek Chroma collection (repo ke hash se naam), taaki
# multiple repos ek hi app me index karne par ek dusre ke
# search results me mix na ho.

PERSIST_ROOT = "indexed_repos"


def _repo_id(repo_url_or_path: str) -> str:
    return hashlib.sha1(repo_url_or_path.encode()).hexdigest()[:12]


def _clone_repo(repo_url: str, dest_dir: str) -> str:
    git.Repo.clone_from(repo_url, dest_dir, depth=1)
    return dest_dir


def _chunks_to_documents(chunks: list[CodeChunk]) -> list[Document]:
    docs = []
    for c in chunks:
        header = f"# File: {c.file_path}\n# {c.node_type}: {c.name}\n\n"
        docs.append(Document(
            page_content=header + c.content,
            metadata={
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "node_type": c.node_type,
                "name": c.name,
                "language": c.language,
            },
        ))
    return docs


def index_repo(source: str, openai_api_key: str, is_local: bool = False, progress_cb=None) -> tuple[Chroma, dict]:
    """Repo ko clone (ya local path use) karke chunk + embed + store karta hai.
    Agar pehle se indexed hai to cache se load ho jata hai (no re-embed)."""
    repo_id = _repo_id(source)
    persist_dir = os.path.join(PERSIST_ROOT, repo_id)
    embedding = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")

    vector_store = Chroma(
        collection_name=repo_id,
        embedding_function=embedding,
        persist_directory=persist_dir,
    )

    if vector_store._collection.count() > 0:
        if progress_cb:
            progress_cb("Already indexed — loading from cache.")
        return vector_store, {"cached": True, "num_chunks": vector_store._collection.count()}

    tmp_clone_dir = None
    try:
        if is_local:
            repo_root = source
        else:
            if progress_cb:
                progress_cb(f"Cloning {source} ...")
            tmp_clone_dir = tempfile.mkdtemp(prefix="repomind_")
            repo_root = _clone_repo(source, tmp_clone_dir)

        if progress_cb:
            progress_cb("Parsing files with tree-sitter ...")
        chunks = chunk_repo(repo_root)

        if not chunks:
            raise ValueError("No supported source files (.py/.js/.ts) found in this repo.")

        if progress_cb:
            progress_cb(f"Embedding {len(chunks)} code chunks ...")
        documents = _chunks_to_documents(chunks)

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            vector_store.add_documents(documents[i:i + batch_size])
            if progress_cb:
                progress_cb(f"Embedded {min(i + batch_size, len(documents))}/{len(documents)} chunks ...")

        return vector_store, {"cached": False, "num_chunks": len(chunks)}

    finally:
        if tmp_clone_dir and os.path.exists(tmp_clone_dir):
            shutil.rmtree(tmp_clone_dir, ignore_errors=True)


def clear_all_indexes():
    if os.path.exists(PERSIST_ROOT):
        shutil.rmtree(PERSIST_ROOT)
    os.makedirs(PERSIST_ROOT, exist_ok=True)


# =========================================================
# 3) RETRIEVER — hybrid semantic (vector) + BM25 keyword search
# =========================================================
#
# Pure semantic search kabhi-kabhi exact identifier match miss
# kar deta hai (e.g. "handleLogin" search karne par embedding
# usse top pe rank nahi karta). BM25 exact/near-exact token
# matches me achha hota hai. Dono ko combine karna better result
# deta hai.

def _tokenize(text: str) -> list[str]:
    return text.lower().replace(".", " ").replace("_", " ").split()


class HybridRetriever:
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store

        raw = vector_store._collection.get(include=["documents", "metadatas"])
        self.doc_texts = raw["documents"]
        self.doc_metadatas = raw["metadatas"]
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.doc_texts]) if self.doc_texts else None

    def retrieve(self, query: str, k: int = 8, semantic_weight: float = 0.6) -> list[dict]:
        """Returns top-k chunks, each as {"content", "metadata", "score"}."""
        semantic_hits = self.vector_store.similarity_search_with_relevance_scores(query, k=k * 2)

        scored: dict[str, dict] = {}
        for doc, score in semantic_hits:
            key = doc.page_content
            scored[key] = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "semantic_score": max(score, 0.0),
                "bm25_score": 0.0,
            }

        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(_tokenize(query))
            max_bm25 = max(bm25_scores) if len(bm25_scores) else 1.0
            for text, meta, raw_score in zip(self.doc_texts, self.doc_metadatas, bm25_scores):
                norm_score = raw_score / max_bm25 if max_bm25 > 0 else 0.0
                if text in scored:
                    scored[text]["bm25_score"] = norm_score
                elif norm_score > 0:
                    scored[text] = {
                        "content": text,
                        "metadata": meta,
                        "semantic_score": 0.0,
                        "bm25_score": norm_score,
                    }

        for entry in scored.values():
            entry["combined_score"] = (
                semantic_weight * entry["semantic_score"]
                + (1 - semantic_weight) * entry["bm25_score"]
            )

        ranked = sorted(scored.values(), key=lambda x: x["combined_score"], reverse=True)
        return ranked[:k]
