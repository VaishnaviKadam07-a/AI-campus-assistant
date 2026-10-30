"""CampusAI - Streamlit UI. All answering logic lives in assistant.py."""
import os

import streamlit as st

try:  # optional: load OPENAI_API_KEY from a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from assistant import KnowledgeBaseError, answer_question, get_api_key, load_knowledge_base

st.set_page_config(page_title="CampusAI", page_icon="🎓", layout="wide")

SUGGESTED = [
    "What is the minimum attendance requirement?",
    "What are the library opening hours?",
    "How do I get my hall ticket for the exams?",
    "What scholarships are available and how do I apply?",
    "How can I prepare for campus placements?",
    "Where can I find the academic calendar?",
]
DISCLAIMER = ("CampusAI currently uses fictional demonstration data. Always verify important "
              "information with official college sources.")

st.markdown(
    """
    <style>
      .block-container {max-width: 900px; padding-top: 4.5rem;}
      .campus-title {font-size: 2.1rem; font-weight: 700; margin: 0; letter-spacing: -0.5px;}
      .campus-sub {font-size: 1.05rem; opacity: 0.75; margin: 0 0 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- state & helpers
@st.cache_resource(show_spinner=False)
def get_knowledge_base():
    """Returns (kb, error_message). Cached so the JSON is parsed once per server run."""
    try:
        return load_knowledge_base(), None
    except KnowledgeBaseError as exc:
        return None, str(exc)


def configured_api_key():
    try:
        secret = st.secrets.get("OPENAI_API_KEY")  # Streamlit Cloud secrets
    except Exception:  # no secrets.toml locally
        secret = None
    return get_api_key(secret)


def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)


def queue_question(question: str):
    st.session_state.pending = question


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending = None


def reload_kb():
    get_knowledge_base.clear()


def render_assistant_meta(meta: dict):
    if meta.get("notice"):
        st.warning(meta["notice"], icon="⚠️")
    if meta.get("found"):
        label = {"ai": "AI-generated from FAQ", "faq": "FAQ answer"}.get(meta["mode"], "Answer")
        st.caption(f"{label} · {meta['category']} · demo data, please verify")


# ----------------------------------------------------------------------------- pages
def chat_page(kb, api_key, use_ai):
    st.markdown('<p class="campus-title">🎓 CampusAI</p>', unsafe_allow_html=True)
    st.markdown('<p class="campus-sub">Your Smart Campus Companion</p>', unsafe_allow_html=True)

    messages = st.session_state.messages
    if not messages:
        st.info("👋 Ask me anything about admissions, exams, attendance, the library, "
                "scholarships or placements. Try one of these:")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED):
            cols[i % 2].button(q, key=f"sugg-{i}", on_click=queue_question, args=(q,),
                               use_container_width=True)

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_assistant_meta(msg["meta"])

    typed = st.chat_input("Ask a campus question, e.g. 'What are the library hours?'", max_chars=500)
    question = typed or st.session_state.pending
    st.session_state.pending = None

    if question:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Looking through the campus knowledge base..."):
                answer = answer_question(question, kb, api_key=api_key if use_ai else None)
            st.markdown(answer.text)
            meta = {"mode": answer.mode, "found": answer.found, "category": answer.category,
                    "notice": answer.notice, "related": answer.related}
            render_assistant_meta(meta)
        messages.append({"role": "assistant", "content": answer.text, "meta": meta})

    if messages and messages[-1]["role"] == "assistant" and messages[-1]["meta"]["related"]:
        st.markdown("**Related questions**")
        for i, q in enumerate(messages[-1]["meta"]["related"]):
            st.button(q, key=f"rel-{len(messages)}-{i}", on_click=queue_question, args=(q,))

    st.caption(DISCLAIMER)


def browse_page(kb):
    st.markdown('<p class="campus-title">📚 Browse FAQs</p>', unsafe_allow_html=True)
    st.markdown('<p class="campus-sub">Everything in the knowledge base, by category</p>',
                unsafe_allow_html=True)
    choice = st.selectbox("Category", ["All"] + kb.categories)
    shown = [e for e in kb.entries if choice in ("All", e["category"])]
    for e in shown:
        with st.expander(f"{e['question']}  ·  {e['category']}"):
            st.write(e["answer"])
    st.caption(DISCLAIMER)


def about_page(kb, ai_ready):
    st.markdown('<p class="campus-title">ℹ️ About CampusAI</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
**How it works**
1. Your question is cleaned and validated.
2. It is matched against FAQ entries (keywords, synonyms, typo tolerance).
3. If nothing relevant is found, CampusAI says so instead of guessing.
4. With an OpenAI API key, the matched FAQs are given to the model as the *only* allowed context.
   Without a key (or if the API fails), you get the FAQ answer directly.

**Current mode:** {"AI-enhanced (OpenAI)" if ai_ready else "FAQ-only (no API key configured)"}

**Limitations**
- Knowledge is limited to the FAQ file (`knowledge_base.json`); it is fictional demo data.
- Matching is keyword-based, so unusual phrasings may be missed.
- AI answers can still contain mistakes, so verify with official sources.
- Chat history lives only in your browser session.

**Customise it:** edit `knowledge_base.json`, then press *Reload knowledge base* in the sidebar.
"""
    )
    st.caption(DISCLAIMER)


# ----------------------------------------------------------------------------- main
def main():
    init_state()
    kb, kb_error = get_knowledge_base()
    api_key = configured_api_key()

    with st.sidebar:
        st.markdown("## 🎓 CampusAI")
        st.caption("Your Smart Campus Companion")
        page = st.radio("Navigate", ["Chat", "Browse FAQs", "About"], label_visibility="collapsed")
        st.divider()

        st.markdown("**Knowledge base**")
        if kb:
            st.success(f"Loaded · {len(kb.entries)} FAQs · {len(kb.categories)} categories")
            st.caption(kb.meta.get("label", "Demonstration data."))
        else:
            st.error("Not loaded")
        st.button("Reload knowledge base", on_click=reload_kb, use_container_width=True)

        st.markdown("**Assistant mode**")
        if api_key:
            use_ai = st.toggle("Use AI answers (OpenAI)", value=True)
            st.caption("AI mode" if use_ai else "FAQ-only mode")
        else:
            use_ai = False
            st.caption("FAQ-only mode. Set OPENAI_API_KEY to enable AI answers.")

        if page == "Chat":
            st.divider()
            st.markdown("**Quick questions**")
            for i, q in enumerate(SUGGESTED[:4]):
                st.button(q, key=f"quick-{i}", on_click=queue_question, args=(q,), use_container_width=True)
            st.button("🗑️ Clear chat", on_click=clear_chat, use_container_width=True)

        st.divider()
        st.caption(DISCLAIMER)

    if kb is None:
        st.error(f"CampusAI can't start: {kb_error}")
        st.info("Check that `knowledge_base.json` exists next to `app.py` and is valid JSON, "
                "then press *Reload knowledge base*.")
        return

    if page == "Chat":
        chat_page(kb, api_key, use_ai)
    elif page == "Browse FAQs":
        browse_page(kb)
    else:
        about_page(kb, bool(api_key and use_ai))


main()
