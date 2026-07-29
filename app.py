# app.py
# Streamlit UI for RepoMind.
# Basically: user pastes a github link -> we index it (core.py does the work)
# -> user asks questions in a chat box -> we retrieve relevant code chunks
# -> send it to gpt-4o-mini -> show the answer + which files it used.

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from core import index_repo, HybridRetriever

st.set_page_config(page_title="RepoMind - AI Codebase Assistant", page_icon="🧠", layout="wide")

# little bit of custom css just so it doesn't look 100% default streamlit
st.markdown("""
<style>
    .title-box {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.3rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.3rem;
    }
    .title-box h1 { color: white; margin: 0; font-size: 1.9rem; }
    .title-box p { color: #d1d5db; margin: 0.3rem 0 0 0; }
    .source-chip {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        padding: 0.2rem 0.7rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title-box"><h1>🧠 RepoMind</h1>'
    '<p>Ask questions about any GitHub repo and get answers with file + line citations.</p></div>',
    unsafe_allow_html=True,
)

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

MAX_HISTORY_TURNS = 6  # only send last N turns to the LLM so tokens don't explode

# ---------- session state ----------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "repo_label" not in st.session_state:
    st.session_state.repo_label = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {role, content, hits}
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0


# ---------- helper functions ----------

def call_llm(messages_for_llm):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    result = llm.invoke(messages_for_llm)

    # use the real usage numbers from openai instead of guessing
    usage = getattr(result, "usage_metadata", None)
    if usage:
        st.session_state.total_input_tokens += usage.get("input_tokens", 0)
        st.session_state.total_output_tokens += usage.get("output_tokens", 0)

    return result.content


def build_context_text(hits, max_chars=None):
    # max_chars=None -> send full code (used for the Explain button)
    # max_chars=700  -> truncate each chunk (used for normal chat, saves tokens)
    blocks = []
    for h in hits:
        meta = h["metadata"]
        location = meta["file_path"] + ":" + str(meta["start_line"]) + "-" + str(meta["end_line"])
        content = h["content"]
        if max_chars and len(content) > max_chars:
            content = content[:max_chars] + "\n... (truncated, click Explain for full code)"
        blocks.append("# " + location + " (" + meta.get("name", "") + ")\n" + content)
    return "\n\n---\n\n".join(blocks)


def build_chat_history():
    # turns our session_state messages into langchain message objects,
    # only keeping the last few turns so the request doesn't grow forever
    history = [SystemMessage(content=SYSTEM_PROMPT)]
    recent = st.session_state.messages[-MAX_HISTORY_TURNS:]
    for m in recent:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history


def render_sources(hits, key_prefix):
    st.markdown("**Sources:**")
    for h in hits:
        meta = h["metadata"]
        loc = meta["file_path"] + ":" + str(meta["start_line"]) + "-" + str(meta["end_line"])
        st.markdown(f'<span class="source-chip">{loc}</span>', unsafe_allow_html=True)

    with st.expander("View retrieved code (full chunks)"):
        st.text(build_context_text(hits))

    cols = st.columns(min(len(hits), 4))
    for i, h in enumerate(hits):
        meta = h["metadata"]
        fname = meta["file_path"].split("/")[-1]
        col = cols[i % len(cols)]
        if col.button("🔍 Explain " + fname, key=key_prefix + "_explain_" + str(i)):
            st.session_state["_pending_explain"] = h
            st.rerun()


# ---------- sidebar ----------

with st.sidebar:
    st.header("📂 Index a repository")
    repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/user/repo.git")
    index_clicked = st.button("Index repo", use_container_width=True)

    if index_clicked and repo_url.strip() != "":
        progress_area = st.empty()

        def progress_cb(msg):
            progress_area.info(msg)

        try:
            with st.spinner("Indexing repository..."):
                vs, stats = index_repo(repo_url.strip(), OPENAI_API_KEY, progress_cb=progress_cb)
            st.session_state.vector_store = vs
            st.session_state.retriever = HybridRetriever(vs)
            st.session_state.repo_label = repo_url.strip()
            st.session_state.messages = []  # new repo = fresh chat
            cache_note = "(loaded from cache)" if stats.get("cached") else "(freshly embedded)"
            progress_area.success(f"Indexed! {stats['num_chunks']} chunks {cache_note}.")
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
    full_code_mode = st.checkbox(
        "Send full code every time (uses more tokens)",
        value=False,
        help="When off, chunks are truncated to ~700 chars. Full code is always "
             "available via the Explain button either way.",
    )
    st.session_state["_top_k"] = top_k
    st.session_state["_full_code_mode"] = full_code_mode

    st.divider()
    st.header("📊 Token usage")
    total_in = st.session_state.total_input_tokens
    total_out = st.session_state.total_output_tokens
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

# show old messages
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            render_sources(msg["hits"], key_prefix="hist_" + str(i))

# handle "Explain this code" button click (set in render_sources above)
if "_pending_explain" in st.session_state:
    h = st.session_state.pop("_pending_explain")
    meta = h["metadata"]
    loc = meta["file_path"] + ":" + str(meta["start_line"]) + "-" + str(meta["end_line"])
    user_msg = f"Explain the code at `{loc}` in detail."
    context_text = build_context_text([h])  # full code, no truncation here

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

# handle new question typed by user
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
        user_turn = (
            "Retrieved Code Context\n======================\n" + context_text +
            "\n======================\n\nQuestion:\n" + query + "\n\nGrounded Answer:"
        )
    else:
        # no hits usually means a meta question like "which file did you use" -
        # let the model just answer from chat history instead
        user_turn = query

    llm_input.append(HumanMessage(content=user_turn))

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            answer = call_llm(llm_input)
        st.markdown(answer)
        if hits:
            render_sources(hits, key_prefix="new_" + str(len(st.session_state.messages)))

    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits if hits else None})
    st.rerun()
