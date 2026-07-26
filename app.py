import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from core import index_repo, HybridRetriever

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="RepoMind — AI Codebase Assistant", page_icon="🧠", layout="wide")

# ----------------------------
# Custom CSS
# ----------------------------
CUSTOM_CSS = """
<style>
    .title-box {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }
    .title-box h1 { color: white; margin: 0; font-size: 2rem; }
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
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="title-box"><h1>🧠 RepoMind — AI Codebase Assistant</h1>'
    '<p>Ask questions about any codebase — get grounded answers with exact file &amp; line citations.</p></div>',
    unsafe_allow_html=True,
)

SYSTEM_PROMPT = """You are a senior software engineer explaining a codebase to a teammate,
in an ongoing chat conversation.

Rules:
- Base every statement strictly on the retrieved code excerpts you are given.
- Never invent function names, file paths, or behavior not shown in the excerpts.
- ALWAYS mention the exact file path (and line numbers, if given) for any code you
  reference, directly inside your answer text — not just as a footnote — because the
  user may later ask "which file/code did you use" and you must be able to answer
  that from this conversation history alone.
- If asked to "explain the code" or "explain this function", walk through the given
  code excerpt logic step by step, in plain language.
- If the retrieved excerpts don't contain enough information, say so explicitly
  rather than guessing.
- Keep explanations clear and concise, as if onboarding a new engineer.
"""

# ----------------------------
# Session state init
# ----------------------------
defaults = {
    "vector_store": None,
    "retriever": None,
    "repo_label": None,
    "messages": [],          # [{"role": "user"/"assistant", "content": str, "hits": [...] or None}]
    "total_input_tokens": 0,
    "total_output_tokens": 0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def call_llm(messages_for_llm):
    """Calls the LLM and updates the cumulative token counters using the
    actual usage reported by the OpenAI response (not an estimate)."""
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    result = llm.invoke(messages_for_llm)

    usage = getattr(result, "usage_metadata", None)
    if usage:
        st.session_state.total_input_tokens += usage.get("input_tokens", 0)
        st.session_state.total_output_tokens += usage.get("output_tokens", 0)
    return result.content


MAX_HISTORY_TURNS = 6  # only the last N turns are sent to the LLM, to keep token usage from growing unbounded


def build_context_text(hits, max_chars_per_chunk=None):
    """max_chars_per_chunk=None => full, untruncated chunk content (used for
    the "Explain" button, where full detail matters). Passing a number
    truncates each chunk to that many characters (used for normal chat
    turns, to keep input tokens down when full code isn't needed)."""
    context_blocks = []
    for h in hits:
        meta = h["metadata"]
        location = f"{meta['file_path']}:{meta['start_line']}-{meta['end_line']}"
        content = h["content"]
        if max_chars_per_chunk and len(content) > max_chars_per_chunk:
            content = content[:max_chars_per_chunk] + "\n... (truncated — click 'Explain' for full code)"
        context_blocks.append(f"# {location} ({meta.get('name', '')})\n{content}")
    return "\n\n---\n\n".join(context_blocks)


def build_chat_history_for_llm():
    """Converts the stored session messages into LangChain message objects,
    limited to the last MAX_HISTORY_TURNS turns. This lets the model answer
    meta-questions like "which code did you use earlier?" purely from the
    recent conversation history, without resending the entire chat every
    single time (which would make token usage grow without bound)."""
    lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    recent = st.session_state.messages[-MAX_HISTORY_TURNS:]
    for m in recent:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    return lc_messages


def render_sources(hits, key_prefix):
    """Renders source chips, a full-context expander, and a per-chunk
    'Explain this code' button."""
    st.markdown("**Sources:**")
    for h in hits:
        meta = h["metadata"]
        loc = f"{meta['file_path']}:{meta['start_line']}-{meta['end_line']}"
        st.markdown(f'<span class="source-chip">{loc}</span>', unsafe_allow_html=True)

    with st.expander("View retrieved context (full chunks)"):
        st.text(build_context_text(hits))

    cols = st.columns(min(len(hits), 4))
    for idx, h in enumerate(hits):
        meta = h["metadata"]
        fname = meta["file_path"].split("/")[-1]
        col = cols[idx % len(cols)]
        if col.button(f"🔍 Explain {fname}", key=f"{key_prefix}_explain_{idx}"):
            st.session_state["_pending_explain"] = h
            st.rerun()


# ----------------------------
# Sidebar: repo indexing + retrieval settings + token usage
# ----------------------------
with st.sidebar:
    st.header("📂 Index a repository")
    repo_url = st.text_input(
        "GitHub repo URL",
        placeholder="https://github.com/user/repo.git",
    )
    index_btn = st.button("Index repo", use_container_width=True)

    if index_btn and repo_url.strip():
        progress_area = st.empty()

        def progress_cb(msg):
            progress_area.info(msg)

        try:
            with st.spinner("Indexing repository..."):
                vs, stats = index_repo(repo_url.strip(), OPENAI_API_KEY, progress_cb=progress_cb)
            st.session_state.vector_store = vs
            st.session_state.retriever = HybridRetriever(vs)
            st.session_state.repo_label = repo_url.strip()
            st.session_state.messages = []  # clear old chat when a new repo is indexed
            progress_area.success(
                f"Indexed! {stats['num_chunks']} chunks "
                f"{'(loaded from cache)' if stats.get('cached') else '(freshly embedded)'}."
            )
        except Exception as e:
            progress_area.error(f"Failed to index repo: {e}")

    if st.session_state.get("repo_label"):
        st.divider()
        st.caption("Currently indexed:")
        st.code(st.session_state.repo_label, language=None)

    st.divider()
    st.header("⚙️ Retrieval settings")
    st.caption("Fewer chunks / smaller context = fewer input tokens, at the cost of some detail.")
    top_k = st.slider("Chunks to retrieve per question (k)", min_value=2, max_value=10, value=4, step=1)
    full_code_mode = st.checkbox(
        "Send full code every time (uses more tokens)",
        value=False,
        help="When off, each chunk is truncated to ~700 characters. Full code is always "
             "available via the 'Explain' button regardless of this setting.",
    )
    st.session_state["_top_k"] = top_k
    st.session_state["_full_code_mode"] = full_code_mode

    st.divider()
    st.header("📊 Token usage")
    total_in = st.session_state.total_input_tokens
    total_out = st.session_state.total_output_tokens
    st.metric("Input tokens (your questions)", f"{total_in:,}")
    st.metric("Output tokens (AI answers)", f"{total_out:,}")
    st.metric("Total tokens", f"{total_in + total_out:,}")

    if st.session_state.messages:
        if st.button("🗑️ Clear chat & reset token count", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_input_tokens = 0
            st.session_state.total_output_tokens = 0
            st.rerun()

# ----------------------------
# Main: chat interface
# ----------------------------
if not st.session_state.get("vector_store"):
    st.info("👈 Paste a GitHub repo URL in the sidebar and click **Index repo** to get started.")
    st.stop()

# Render past messages
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            render_sources(msg["hits"], key_prefix=f"hist_{i}")

# Handle a pending "explain this code" request triggered by a button click above
if "_pending_explain" in st.session_state:
    h = st.session_state.pop("_pending_explain")
    meta = h["metadata"]
    loc = f"{meta['file_path']}:{meta['start_line']}-{meta['end_line']}"
    user_msg = f"Explain the code at `{loc}` in detail."
    context_text = build_context_text([h])  # full, untruncated content for this single chunk

    st.session_state.messages.append({"role": "user", "content": user_msg, "hits": None})

    llm_input = build_chat_history_for_llm()
    llm_input.append(HumanMessage(content=(
        f"Retrieved Code Context\n======================\n{context_text}\n======================\n\n"
        f"Question:\n{user_msg}\n\nGrounded Answer:"
    )))
    with st.spinner("Explaining code..."):
        answer = call_llm(llm_input)

    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": [h]})
    st.rerun()

# Chat input for new questions
query = st.chat_input("Ask a question about this codebase (e.g. 'how is auth handled?', 'which code did you use?')")

if query:
    st.session_state.messages.append({"role": "user", "content": query, "hits": None})

    with st.chat_message("user"):
        st.markdown(query)

    top_k = st.session_state.get("_top_k", 4)
    full_code_mode = st.session_state.get("_full_code_mode", False)

    with st.spinner("Retrieving relevant code..."):
        hits = st.session_state.retriever.retrieve(query, k=top_k)

    llm_input = build_chat_history_for_llm()
    if hits:
        max_chars = None if full_code_mode else 700
        context_text = build_context_text(hits, max_chars_per_chunk=max_chars)
        user_turn = (
            f"Retrieved Code Context\n======================\n{context_text}\n======================\n\n"
            f"Question:\n{query}\n\nGrounded Answer:"
        )
    else:
        # No fresh retrieval hits (e.g. a meta-question like "which file did
        # you use earlier?") — let the model answer purely from chat history.
        user_turn = query
    llm_input.append(HumanMessage(content=user_turn))

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            answer = call_llm(llm_input)
        st.markdown(answer)
        if hits:
            render_sources(hits, key_prefix=f"new_{len(st.session_state.messages)}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits if hits else None})
    st.rerun()
