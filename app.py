# AI Codebase Assistant - AI codebase assistant (single-file version)
# Flow: index a GitHub repo (tree-sitter chunks -> embeddings in Chroma) ->
# ask questions in chat -> hybrid (vector + BM25) retrieval -> gpt-4o-mini
# answers with file/line citations. Explain button gives full-code drill-down.

import os, shutil, hashlib, tempfile
import streamlit as st
from dotenv import load_dotenv
import git
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Config ----------
PY_LANG = Language(tspython.language())
JS_LANG = Language(tsjs.language())
CHUNK_TYPES = {
    "python": ["function_definition", "class_definition"],
    "javascript": ["function_declaration", "class_declaration", "method_definition"],
}
EXT_LANG = {".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "javascript", ".tsx": "javascript"}
IGNORE_FOLDERS = {".git", "__pycache__", "node_modules", "venv", ".venv", "env",
                   "dist", "build", ".next", "site-packages", ".idea", ".vscode"}
MAX_FILE_SIZE = 500_000  # skip huge generated files
PERSIST_ROOT = "indexed_repos"
MAX_HISTORY_TURNS = 6  # only send last N turns to the LLM so tokens don't explode

SYSTEM_PROMPT = """You are a senior software engineer explaining a codebase to a teammate.
Rules:
- Only use the retrieved code excerpts given to you, don't make things up.
- Never invent function names, file paths, or behavior that isn't in the excerpts.
- Always mention the exact file path (and line numbers if given) inside your answer text,
  not just as a footnote, because the user might later ask "which file did you use".
- If asked to explain a function, walk through it step by step in plain language.
- If the excerpts don't have enough info, just say so instead of guessing.
- Keep it clear and to the point, like you're onboarding a new dev.
"""

# ---------- Chunking (tree-sitter) ----------

def get_parser(lang):
    return Parser(PY_LANG if lang == "python" else JS_LANG)

def get_node_name(node, source):
    for child in node.children:
        if child.type in ("identifier", "property_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8", "ignore")
    return "unknown"

def walk_tree(node, source, lang, out_list, rel_path):
    if node.type in CHUNK_TYPES[lang]:
        out_list.append({
            "content": source[node.start_byte:node.end_byte].decode("utf-8", "ignore"),
            "file_path": rel_path,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "node_type": node.type,
            "name": get_node_name(node, source),
        })
        return  # don't recurse inside a chunk we already grabbed
    for child in node.children:
        walk_tree(child, source, lang, out_list, rel_path)

def chunk_one_file(full_path, repo_root):
    lang = EXT_LANG.get(os.path.splitext(full_path)[1])
    rel_path = os.path.relpath(full_path, repo_root)
    try:
        with open(full_path, "rb") as f:
            raw = f.read()
    except (OSError, UnicodeDecodeError):
        return []
    if not raw or len(raw) > MAX_FILE_SIZE:
        return []

    tree = get_parser(lang).parse(raw)
    chunks = []
    walk_tree(tree.root_node, raw, lang, chunks, rel_path)

    if not chunks:  # fallback: whole file as one chunk (e.g. top-level script)
        text = raw.decode("utf-8", "ignore")
        if text.strip():
            chunks.append({"content": text, "file_path": rel_path, "start_line": 1,
                            "end_line": text.count("\n") + 1, "node_type": "file",
                            "name": os.path.basename(full_path)})
    return chunks

def chunk_repo(repo_root):
    all_chunks = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_FOLDERS and not d.startswith(".")]
        for fname in filenames:
            if os.path.splitext(fname)[1] in EXT_LANG:
                all_chunks += chunk_one_file(os.path.join(dirpath, fname), repo_root)
    return all_chunks

# ---------- Indexing (embed + store in Chroma) ----------

def repo_id_from_url(url):
    return hashlib.sha1(url.encode()).hexdigest()[:12]

def chunks_to_docs(chunks):
    docs = []
    for c in chunks:
        header = f'# File: {c["file_path"]}\n# {c["node_type"]}: {c["name"]}\n\n'
        docs.append(Document(
            page_content=header + c["content"],
            metadata={k: c[k] for k in ("file_path", "start_line", "end_line", "node_type", "name")},
        ))
    return docs

def index_repo(repo_url, api_key, progress_cb=None):
    """Clone + chunk + embed a repo into Chroma. Reuses cache if already indexed."""
    rid = repo_id_from_url(repo_url)
    persist_dir = os.path.join(PERSIST_ROOT, rid)
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    db = Chroma(collection_name=rid, embedding_function=embeddings, persist_directory=persist_dir)

    if db._collection.count() > 0:
        if progress_cb:
            progress_cb("Already indexed before, loading from cache...")
        return db, {"cached": True, "num_chunks": db._collection.count()}

    tmp_dir = tempfile.mkdtemp(prefix="ai_codebase_assistant_")
    try:
        if progress_cb:
            progress_cb("Cloning " + repo_url + " ...")
        git.Repo.clone_from(repo_url, tmp_dir, depth=1)

        if progress_cb:
            progress_cb("Parsing files (tree-sitter)...")
        chunks = chunk_repo(tmp_dir)
        if not chunks:
            raise ValueError("Couldn't find any .py/.js/.ts files in this repo.")

        if progress_cb:
            progress_cb("Embedding " + str(len(chunks)) + " chunks...")
        docs = chunks_to_docs(chunks)

        batch = 100
        for i in range(0, len(docs), batch):
            db.add_documents(docs[i:i + batch])
            if progress_cb:
                done = min(i + batch, len(docs))
                progress_cb("Embedded " + str(done) + "/" + str(len(docs)) + " chunks...")

        return db, {"cached": False, "num_chunks": len(chunks)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def clear_all_indexes():
    if os.path.exists(PERSIST_ROOT):
        shutil.rmtree(PERSIST_ROOT)
    os.makedirs(PERSIST_ROOT, exist_ok=True)

# ---------- Retrieval: hybrid vector search + BM25 keyword search ----------

def tokenize(text):
    return text.lower().replace(".", " ").replace("_", " ").split()

class HybridRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        raw = vector_store._collection.get(include=["documents", "metadatas"])
        self.texts = raw["documents"]
        self.metas = raw["metadatas"]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts]) if self.texts else None

    def retrieve(self, query, k=8, semantic_weight=0.6):
        semantic_hits = self.vector_store.similarity_search_with_relevance_scores(query, k=k * 2)

        scored = {}
        for doc, score in semantic_hits:
            scored[doc.page_content] = {
                "content": doc.page_content, "metadata": doc.metadata,
                "semantic_score": max(score, 0.0), "bm25_score": 0.0,
            }

        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(tokenize(query))
            top_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
            for text, meta, score in zip(self.texts, self.metas, bm25_scores):
                norm = score / top_bm25 if top_bm25 > 0 else 0.0
                if text in scored:
                    scored[text]["bm25_score"] = norm
                elif norm > 0:
                    scored[text] = {"content": text, "metadata": meta,
                                     "semantic_score": 0.0, "bm25_score": norm}

        for entry in scored.values():
            entry["combined_score"] = (semantic_weight * entry["semantic_score"]
                                        + (1 - semantic_weight) * entry["bm25_score"])

        ranked = sorted(scored.values(), key=lambda x: x["combined_score"], reverse=True)
        return ranked[:k]

# ---------- LLM call + context building ----------

def call_llm(messages_for_llm):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    result = llm.invoke(messages_for_llm)
    usage = getattr(result, "usage_metadata", None)
    if usage:
        st.session_state.total_input_tokens += usage.get("input_tokens", 0)
        st.session_state.total_output_tokens += usage.get("output_tokens", 0)
    return result.content

def build_context_text(hits, max_chars=None):
    # max_chars=None -> full code (Explain button). max_chars=700 -> truncated (normal chat).
    blocks = []
    for h in hits:
        m = h["metadata"]
        loc = f'{m["file_path"]}:{m["start_line"]}-{m["end_line"]}'
        content = h["content"]
        if max_chars and len(content) > max_chars:
            content = content[:max_chars] + "\n... (truncated, click Explain for full code)"
        blocks.append(f'# {loc} ({m.get("name", "")})\n{content}')
    return "\n\n---\n\n".join(blocks)

def build_chat_history():
    history = [SystemMessage(content=SYSTEM_PROMPT)]
    recent = st.session_state.messages[-MAX_HISTORY_TURNS:]
    for m in recent:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        history.append(cls(content=m["content"]))
    return history

def render_sources(hits, key_prefix):
    st.markdown("**Sources:**")
    for h in hits:
        m = h["metadata"]
        st.markdown(f'<span class="source-chip">{m["file_path"]}:{m["start_line"]}-{m["end_line"]}</span>',
                    unsafe_allow_html=True)

    with st.expander("View retrieved code (full chunks)"):
        st.text(build_context_text(hits))

    cols = st.columns(min(len(hits), 4))
    for i, h in enumerate(hits):
        fname = h["metadata"]["file_path"].split("/")[-1]
        if cols[i % len(cols)].button("🔍 Explain " + fname, key=f"{key_prefix}_explain_{i}"):
            st.session_state["_pending_explain"] = h
            st.rerun()

# ---------- Streamlit UI ----------

st.set_page_config(page_title="AI Codebase Assistant - AI Codebase Assistant", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .title-box { background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.3rem 2rem; border-radius: 12px; margin-bottom: 1.3rem; }
    .title-box h1 { color: white; margin: 0; font-size: 1.9rem; }
    .title-box p { color: #d1d5db; margin: 0.3rem 0 0 0; }
    .source-chip { display: inline-block; background: #e0e7ff; color: #3730a3;
        padding: 0.2rem 0.7rem; border-radius: 8px; font-family: monospace;
        font-size: 0.85rem; margin: 0.2rem 0.3rem 0.2rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title-box"><h1>🧠 AI Codebase Assistant</h1>'
    '<p>Ask questions about any GitHub repo and get answers with file + line citations.</p></div>',
    unsafe_allow_html=True,
)

for key, default in [("vector_store", None), ("retriever", None), ("repo_label", None),
                      ("messages", []), ("total_input_tokens", 0), ("total_output_tokens", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- sidebar ----------

with st.sidebar:
    st.header("📂 Index a repository")
    repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/user/repo.git")

    if st.button("Index repo", use_container_width=True) and repo_url.strip():
        progress_area = st.empty()
        try:
            with st.spinner("Indexing repository..."):
                vs, stats = index_repo(repo_url.strip(), OPENAI_API_KEY, progress_cb=progress_area.info)
            st.session_state.vector_store = vs
            st.session_state.retriever = HybridRetriever(vs)
            st.session_state.repo_label = repo_url.strip()
            st.session_state.messages = []
            note = "(loaded from cache)" if stats["cached"] else "(freshly embedded)"
            progress_area.success(f"Indexed! {stats['num_chunks']} chunks {note}.")
        except Exception as e:
            progress_area.error("Failed to index repo: " + str(e))

    if st.session_state.repo_label:
        st.divider()
        st.caption("Currently indexed:")
        st.code(st.session_state.repo_label, language=None)

    st.divider()
    st.header("⚙️ Retrieval settings")
    st.caption("Fewer / smaller chunks = fewer tokens, but less detail.")
    top_k = st.slider("Chunks to retrieve per question", min_value=2, max_value=10, value=4)
    full_code_mode = st.checkbox("Send full code every time (uses more tokens)", value=False,
                                  help="When off, chunks are truncated to ~700 chars. Full code is "
                                       "always available via the Explain button either way.")
    st.session_state["_top_k"] = top_k
    st.session_state["_full_code_mode"] = full_code_mode

    st.divider()
    st.header("📊 Token usage")
    total_in, total_out = st.session_state.total_input_tokens, st.session_state.total_output_tokens
    st.metric("Input tokens", f"{total_in:,}")
    st.metric("Output tokens", f"{total_out:,}")
    st.metric("Total tokens", f"{total_in + total_out:,}")

    if st.session_state.messages:
        if st.button("🗑️ Clear chat & reset tokens", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_input_tokens = 0
            st.session_state.total_output_tokens = 0
            st.rerun()

# ---------- main chat area ----------

if not st.session_state.vector_store:
    st.info("👈 Paste a GitHub repo URL in the sidebar and click **Index repo** to get started.")
    st.stop()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            render_sources(msg["hits"], key_prefix="hist_" + str(i))

# handle "Explain this code" button click
if "_pending_explain" in st.session_state:
    h = st.session_state.pop("_pending_explain")
    m = h["metadata"]
    loc = f'{m["file_path"]}:{m["start_line"]}-{m["end_line"]}'
    user_msg = f"Explain the code at `{loc}` in detail."
    context_text = build_context_text([h])  # full code, no truncation

    st.session_state.messages.append({"role": "user", "content": user_msg, "hits": None})
    llm_input = build_chat_history()
    llm_input.append(HumanMessage(content=(
        "Retrieved Code Context\n======================\n" + context_text +
        "\n======================\n\nQuestion:\n" + user_msg + "\n\nGrounded Answer:"
    )))

    with st.spinner("Explaining code..."):
        answer = call_llm(llm_input)
    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": [h]})
    st.rerun()

# handle new question
query = st.chat_input("Ask something about this codebase (e.g. 'how is auth handled?')")

if query:
    st.session_state.messages.append({"role": "user", "content": query, "hits": None})
    with st.chat_message("user"):
        st.markdown(query)

    top_k = st.session_state.get("_top_k", 4)
    full_code_mode = st.session_state.get("_full_code_mode", False)

    with st.spinner("Retrieving relevant code..."):
        hits = st.session_state.retriever.retrieve(query, k=top_k)

    llm_input = build_chat_history()
    if hits:
        max_chars = None if full_code_mode else 700
        context_text = build_context_text(hits, max_chars=max_chars)
        user_turn = ("Retrieved Code Context\n======================\n" + context_text +
                     "\n======================\n\nQuestion:\n" + query + "\n\nGrounded Answer:")
    else:
        # no hits usually means a meta question like "which file did you use"
        user_turn = query
    llm_input.append(HumanMessage(content=user_turn))

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            answer = call_llm(llm_input)
        st.markdown(answer)
        if hits:
            render_sources(hits, key_prefix="new_" + str(len(st.session_state.messages)))

    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits or None})
    st.rerun()
