# app.py – Streamlit-only PlatypAI chatbot (no Flask)
import os, time, json, hashlib, pathlib, textwrap
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import openai
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
import lancedb, docx
from openai import OpenAI


# ────────────────────────── CONFIG ──────────────────────────
api_key = os.getenv("OPENAI_API_KEY") or "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"
client=OpenAI(api_key=api_key)
# openai.api_key = "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"
ASSISTANT_ID = "asst_xzJKnd6qxS7lrV2PKDXmAWz9"  # replace if needed

DOCX_PATH = "Structured Content for ChatBot.docx"
DB_FOLDER = "./platypai_chatbot"
TABLE_TAG = "platypai_chatbot"

CHUNK_SIZE, CHUNK_OVER = 300, 40
TOP_K, FETCH_K = 3, 10
TIMEOUT_S = 45

PRONOUN_RULE = (
    "PRONOUN RULE: When a user says “you”, “your”, or “yourself” in the "
    "context of services, projects, capabilities, or other business matters, "
    "interpret those pronouns as *PlatypAI the company*, **not** the AI assistant.\n"
)
FORMAT_RULE = (
    "FORMAT RULE: Put each list item on ONE line, e.g.\n"
    "`1. Consulting Services – Expert advice …`"
)

# ────────────────────────── BUILD VECTOR DB ──────────────────────────
retrieval_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
os.makedirs(DB_FOLDER, exist_ok=True)
db = lancedb.connect(DB_FOLDER)

def _docx_hash(path):
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()[:8]

TABLE_NAME = f"{TABLE_TAG}_{_docx_hash(DOCX_PATH)}_{CHUNK_SIZE}_{CHUNK_OVER}"
if TABLE_NAME not in db.table_names():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVER)
    chunks = []
    for p in docx.Document(DOCX_PATH).paragraphs:
        t = p.text.strip()
        if not t: continue
        for c in splitter.split_text(t):
            chunks.append(Document(page_content=c, metadata={"source": "docx"}))
    LanceDB.from_documents(chunks, embedding=retrieval_emb, connection=db, table_name=TABLE_NAME)

vs = LanceDB(connection=db, table_name=TABLE_NAME, embedding=retrieval_emb)
retriever = vs.as_retriever(search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "maximal_marginal_relevance": True})

executor = ThreadPoolExecutor(max_workers=2)

# ────────────────────────── HELPER FUNCTIONS ──────────────────────────
def _ensure_thread(tid=None):
    return tid or client.beta.threads.create().id

def _post_user(thread_id, msg):
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=msg)

def _run(thread_id, instr):
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID, instructions=instr)
    start = time.time()
    while run.status not in ("completed", "failed", "expired", "cancelled"):
        if time.time() - start > TIMEOUT_S:
            raise TimeoutError("Assistant run timed out.")
        time.sleep(0.35)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
    if run.status != "completed":
        raise RuntimeError(f"Run ended with status {run.status}")

def _chat_history(thread_id):
    msgs = sorted(client.beta.threads.messages.list(thread_id=thread_id).data, key=lambda m: m.created_at)
    history, last = [], ""
    for m in msgs:
        if m.role not in ("user", "assistant"): continue
        text = "\n\n".join(blk.text.value.strip() for blk in m.content or [] if getattr(blk, "text", None))
        history.append({"role": m.role, "message": text})
        if m.role == "assistant": last = text
    return history, last or "I’m here! Ask me anything about PlatypAI."

def ask_question(question, thread_id=None):
    thread_id = _ensure_thread(thread_id)
    _post_user(thread_id, question)

    chunks = retriever.invoke(question)
    ctx = "\n\n".join(f"<doc id={i}>\n{c.page_content}\n</doc>" for i, c in enumerate(chunks, 1))

    is_company_q = any(kw in question.lower() for kw in ("platypai", "you ", "your "))
    must_refuse = not ctx.strip() and not is_company_q

    if must_refuse:
        system = PRONOUN_RULE + FORMAT_RULE
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": question}]
        )
        final = resp.choices[0].message.content
        client.beta.threads.messages.create(thread_id=thread_id, role="assistant", content=final)
        history, _ = _chat_history(thread_id)
        return {"response": final, "thread_id": thread_id, "chat_history": history}

    instr = PRONOUN_RULE + FORMAT_RULE
    if ctx:
        instr += "\n\nUse these <doc …> blocks when relevant:\n\n" + ctx
    else:
        instr += "\n\nNo relevant internal documents were found."

    _run(thread_id, instr)
    history, final = _chat_history(thread_id)
    return {"response": final, "thread_id": thread_id, "chat_history": history}

# ────────────────────────── STREAMLIT UI ──────────────────────────
st.set_page_config(page_title="Chat with PlatypAI Bot", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    html,body,[data-testid="stAppViewContainer"]{background:#FFF8F2;color:#1F1F1F}
    #MainMenu,header,footer{visibility:hidden}
    .chatbox{background:#fff;border:2px solid #A6E3E9;
             box-shadow:0 4px 15px rgba(91,55,124,.1);
             border-radius:16px;padding:30px 30px 10px;
             max-width:800px;margin:auto}
    .msg{max-width:75%;padding:12px 16px;margin-bottom:12px;
         border-radius:14px;font-size:15px;line-height:1.4;
         box-shadow:0 1px 4px rgba(0,0,0,.08)}
    .user      {background:#A6E3E9;color:#1F1F1F;margin-left:auto;text-align:right}
    .assistant {background:#fff;border:1px solid #8F69A0;color:#5B377C;margin-right:auto}
    .typing    {font-style:italic;color:#8F69A0;animation:blink 1s step-start infinite}
    @keyframes blink{50%{opacity:.4}}
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

pending_q = st.session_state.get("pending_question")

def md_to_html(md):
    md = md.replace("**", "<strong>").replace("*", "<em>")
    lines, out, in_ul = textwrap.dedent(md).split("\n"), [], False
    for ln in lines:
        if ln.lstrip().startswith(("-", "•")):
            if not in_ul: out.append("<ul>"); in_ul=True
            out.append(f"<li>{ln.lstrip()[1:].strip()}</li>")
        else:
            if in_ul: out.append("</ul>"); in_ul=False
            out.append(ln)
    if in_ul: out.append("</ul>")
    return "<br>".join(out)

def draw_messages():
    for m in st.session_state.messages:
        st.markdown(
            f'<div class="msg {m["role"]}"><strong>{"You" if m["role"]=="user" else "PlatypAI Bot"}</strong>{md_to_html(m["message"])}</div>',
            unsafe_allow_html=True,
        )

with st.container():
    st.markdown('<div class="chatbox">', unsafe_allow_html=True)
    st.markdown("## Ask me anything… PlatypAI related!")
    draw_messages()

    user_input = st.chat_input("Type your question…")
    if user_input:
        st.session_state.messages.append({"role": "user", "message": user_input})
        st.session_state.pending_question = user_input
        st.rerun()

    if pending_q:
        placeholder = st.empty()
        placeholder.markdown('<div class="msg assistant typing">PlatypAI Bot is typing…</div>', unsafe_allow_html=True)
        try:
            data = ask_question(pending_q, st.session_state.thread_id)
            st.session_state.thread_id = data.get("thread_id")
            st.session_state.messages = data.get("chat_history", [])
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "message": f"Error: {e}"})
        del st.session_state.pending_question
        placeholder.empty()
        st.rerun()

    st.markdown('<div style="margin-top:40px;font-size:13px;text-align:center;color:#8F69A0">Chat UI styled for PlatypAI</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
