# core.py
# All the "backend" logic for RepoMind lives here.
# 3 things happen in this file:
#   1. clone the repo + break it into chunks (functions/classes) using tree-sitter
#   2. turn those chunks into embeddings and store them in Chroma
#   3. search: mix of normal vector search + BM25 keyword search
#
# I kept everything in one file instead of splitting chunker/indexer/retriever
# into separate files, just easier to manage for a small project like this.

import os
import shutil
import hashlib
import tempfile

import git
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser
from rank_bm25 import BM25Okapi

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ---------- setup ----------

PY_LANG = Language(tspython.language())
JS_LANG = Language(tsjs.language())

# which tree-sitter node types count as "one chunk" for each language
CHUNK_TYPES = {
    "python": ["function_definition", "class_definition"],
    "javascript": ["function_declaration", "class_declaration", "method_definition"],
}

# map file extension -> language name
EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
}

# folders we never want to look inside
IGNORE_FOLDERS = [".git", "__pycache__", "node_modules", "venv", ".venv",
                   "env", "dist", "build", ".next", "site-packages", ".idea", ".vscode"]

MAX_FILE_SIZE = 500_000  # skip huge generated files, not useful anyway

PERSIST_ROOT = "indexed_repos"  # where chroma dbs get saved on disk


# ---------- part 1: chunking the code ----------

def get_parser(lang):
    if lang == "python":
        return Parser(PY_LANG)
    if lang == "javascript":
        return Parser(JS_LANG)
    raise ValueError("language not supported: " + lang)


def get_node_name(node, source):
    # grabs the function/class name from the AST node
    for child in node.children:
        if child.type in ("identifier", "property_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8", "ignore")
    return "unknown"


def walk_tree(node, source, lang, out_list, rel_path):
    # recursively walks the AST, and whenever it finds a function/class
    # node, saves it as one chunk (and doesn't go deeper into it)
    wanted_types = CHUNK_TYPES[lang]

    if node.type in wanted_types:
        name = get_node_name(node, source)
        code_text = source[node.start_byte:node.end_byte].decode("utf-8", "ignore")
        out_list.append({
            "content": code_text,
            "file_path": rel_path,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "node_type": node.type,
            "name": name,
            "language": lang,
        })
        return  # don't recurse inside a function we already grabbed

    for child in node.children:
        walk_tree(child, source, lang, out_list, rel_path)


def chunk_one_file(full_path, repo_root):
    ext = os.path.splitext(full_path)[1]
    lang = EXT_LANG.get(ext)
    rel_path = os.path.relpath(full_path, repo_root)

    try:
        with open(full_path, "rb") as f:
            raw = f.read()
    except (OSError, UnicodeDecodeError):
        return []

    if len(raw) == 0 or len(raw) > MAX_FILE_SIZE:
        return []

    # not a language we parse -> just treat whole file as one chunk
    if lang is None:
        text = raw.decode("utf-8", "ignore")
        if text.strip() == "":
            return []
        return [{
            "content": text,
            "file_path": rel_path,
            "start_line": 1,
            "end_line": text.count("\n") + 1,
            "node_type": "file",
            "name": os.path.basename(full_path),
            "language": "text",
        }]

    parser = get_parser(lang)
    tree = parser.parse(raw)

    chunks = []
    walk_tree(tree.root_node, raw, lang, chunks, rel_path)

    # if tree-sitter didn't find any function/class (e.g. just a script
    # with top level code), fall back to using the whole file
    if len(chunks) == 0:
        text = raw.decode("utf-8", "ignore")
        if text.strip() != "":
            chunks.append({
                "content": text,
                "file_path": rel_path,
                "start_line": 1,
                "end_line": text.count("\n") + 1,
                "node_type": "file",
                "name": os.path.basename(full_path),
                "language": lang,
            })

    return chunks


def chunk_repo(repo_root):
    all_chunks = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # skip ignored folders (edit dirnames in place so os.walk skips them)
        dirnames[:] = [d for d in dirnames if d not in IGNORE_FOLDERS and not d.startswith(".")]

        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            if ext not in EXT_LANG:
                continue
            full_path = os.path.join(dirpath, fname)
            all_chunks += chunk_one_file(full_path, repo_root)

    return all_chunks


# ---------- part 2: indexing (embed + store in chroma) ----------

def repo_id_from_url(url_or_path):
    # short hash so we get a consistent folder/collection name per repo
    return hashlib.sha1(url_or_path.encode()).hexdigest()[:12]


def clone_repo(url, dest_folder):
    git.Repo.clone_from(url, dest_folder, depth=1)
    return dest_folder


def chunks_to_docs(chunks):
    # wraps each chunk dict into a langchain Document, with a small
    # header at the top so the model always sees file path + name
    docs = []
    for c in chunks:
        header = "# File: " + c["file_path"] + "\n# " + c["node_type"] + ": " + c["name"] + "\n\n"
        docs.append(Document(
            page_content=header + c["content"],
            metadata={
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "node_type": c["node_type"],
                "name": c["name"],
                "language": c["language"],
            },
        ))
    return docs


def index_repo(source, api_key, is_local=False, progress_cb=None):
    """
    Clones the repo (or uses a local folder), chunks it, embeds it and
    stores it in Chroma. If we already indexed this repo before, just
    load it from disk instead of doing all the work again.
    """
    rid = repo_id_from_url(source)
    persist_dir = os.path.join(PERSIST_ROOT, rid)

    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    db = Chroma(collection_name=rid, embedding_function=embeddings, persist_directory=persist_dir)

    if db._collection.count() > 0:
        if progress_cb:
            progress_cb("Already indexed before, loading from cache...")
        return db, {"cached": True, "num_chunks": db._collection.count()}

    tmp_dir = None
    try:
        if is_local:
            repo_root = source
        else:
            if progress_cb:
                progress_cb("Cloning " + source + " ...")
            tmp_dir = tempfile.mkdtemp(prefix="repomind_")
            repo_root = clone_repo(source, tmp_dir)

        if progress_cb:
            progress_cb("Parsing files (tree-sitter)...")
        chunks = chunk_repo(repo_root)

        if len(chunks) == 0:
            raise ValueError("Couldn't find any .py/.js/.ts files in this repo.")

        if progress_cb:
            progress_cb("Embedding " + str(len(chunks)) + " chunks...")
        docs = chunks_to_docs(chunks)

        # add in batches so we don't send one giant request
        batch = 100
        for i in range(0, len(docs), batch):
            db.add_documents(docs[i:i + batch])
            if progress_cb:
                done = min(i + batch, len(docs))
                progress_cb("Embedded " + str(done) + "/" + str(len(docs)) + " chunks...")

        return db, {"cached": False, "num_chunks": len(chunks)}

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def clear_all_indexes():
    if os.path.exists(PERSIST_ROOT):
        shutil.rmtree(PERSIST_ROOT)
    os.makedirs(PERSIST_ROOT, exist_ok=True)


# ---------- part 3: retrieval (semantic + BM25 mixed together) ----------
#
# Plain vector search alone sometimes misses exact keyword matches
# (like a function name "handleLogin"), so BM25 is added on top and
# both scores get combined into one final score.

def tokenize(text):
    return text.lower().replace(".", " ").replace("_", " ").split()


class HybridRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

        raw = vector_store._collection.get(include=["documents", "metadatas"])
        self.texts = raw["documents"]
        self.metas = raw["metadatas"]

        if len(self.texts) > 0:
            self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        else:
            self.bm25 = None

    def retrieve(self, query, k=8, semantic_weight=0.6):
        # step 1: semantic (vector) search
        semantic_hits = self.vector_store.similarity_search_with_relevance_scores(query, k=k * 2)

        scored = {}
        for doc, score in semantic_hits:
            key = doc.page_content
            scored[key] = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "semantic_score": max(score, 0.0),
                "bm25_score": 0.0,
            }

        # step 2: BM25 keyword search, merged into the same dict
        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(tokenize(query))
            top_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0

            for text, meta, score in zip(self.texts, self.metas, bm25_scores):
                norm = score / top_bm25 if top_bm25 > 0 else 0.0
                if text in scored:
                    scored[text]["bm25_score"] = norm
                elif norm > 0:
                    scored[text] = {
                        "content": text,
                        "metadata": meta,
                        "semantic_score": 0.0,
                        "bm25_score": norm,
                    }

        # step 3: combine both scores into one final ranking
        for entry in scored.values():
            entry["combined_score"] = (semantic_weight * entry["semantic_score"]
                                        + (1 - semantic_weight) * entry["bm25_score"])

        ranked = sorted(scored.values(), key=lambda x: x["combined_score"], reverse=True)
        return ranked[:k]
