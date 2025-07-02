from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain.embeddings import HuggingFaceEmbeddings
import lancedb, docx, os, time, hashlib, pathlib
from concurrent.futures import ThreadPoolExecutor

# ── OPENAI & ASSISTANT IDs ───────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")

client        = OpenAI(api_key=api_key)
ASSISTANT_ID  = "asst_xzJKnd6qxS7lrV2PKDXmAWz9"

# ── RAG / VECTOR SETTINGS ────────────────────────────────────────────
DOCX_PATH   = "Structured Content for ChatBot.docx"
DB_FOLDER   = "./platypai_chatbot"
TABLE_TAG   = "platypai_chatbot"

CHUNK_SIZE, CHUNK_OVER = 300, 40
TOP_K, FETCH_K        = 3, 10
TIMEOUT_S             = 45

# ── RULES ─────────────────────────────────────────────────────────────
PRONOUN_RULE = (
    "PRONOUN RULE: When a user says “you”, “your”, or “yourself” in the "
    "context of services, projects, capabilities, or other business matters, "
    "interpret those pronouns as *PlatypAI the company*, **not** the AI "
    "assistant, unless the user explicitly states they are asking about the "
    "assistant or chatbot.\n"
)
FORMAT_RULE = (
    "FORMAT RULE: Put each list item on ONE line, e.g.\n"
    "`1. Consulting Services – Expert advice …`"
)

# ── Build / reuse LanceDB vector table ────────────────────────────────
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
            chunks.append(Document(page_content=c, metadata={"source":"docx"}))
    LanceDB.from_documents(chunks, embedding=retrieval_emb, connection=db, table_name=TABLE_NAME)
else:
    print("✅ Using existing table:", TABLE_NAME)

vs        = LanceDB(connection=db, table_name=TABLE_NAME, embedding=retrieval_emb)
retriever = vs.as_retriever(search_kwargs={
    "k": TOP_K, "fetch_k": FETCH_K, "maximal_marginal_relevance": True
})

# ── Assistant helpers ─────────────────────────────────────────────────
def _ensure_thread(tid=None):
    return tid or client.beta.threads.create().id

def _post_user(thread_id, msg):
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=msg)

def _run(thread_id, instr):
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID, instructions=instr)
    start = time.time()
    while run.status not in ("completed","failed","expired","cancelled"):
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
        if m.role not in ("user","assistant"): continue
        text = "\n\n".join(blk.text.value.strip() for blk in m.content or [] if getattr(blk,"text",None))
        history.append({"role":m.role,"message":text})
        if m.role=="assistant": last = text
    return history, last or "I’m here! Ask me anything about PlatypAI."

# ── Flask app + parallel retrieval ────────────────────────────────────
app = Flask(__name__)
CORS(app)
executor = ThreadPoolExecutor(max_workers=2)

@app.route("/", methods=["GET"])
def root():
    return "✅ PlatypAI RAG backend running."

@app.route("/ask", methods=["POST"])
def ask():
    data      = request.json or {}
    question  = (data.get("question") or "").strip()
    thread_id = data.get("thread_id")

    if not question:
        return jsonify({"response":"No question received."}), 400

    # 1) ensure thread & record user
    thread_id = _ensure_thread(thread_id)
    _post_user(thread_id, question)

    # 2) parallel retrieval
    future = executor.submit(retriever.invoke, question)
    chunks = future.result()
    ctx    = "\n\n".join(f"<doc id={i}>\n{c.page_content}\n</doc>" for i,c in enumerate(chunks,1))

    # 3) relevance check
    is_company_q = any(kw in question.lower() for kw in ("platypai","you ","your "))
    no_ctx      = not bool(ctx.strip())
    must_refuse = no_ctx and not is_company_q

    # 4) fallback via gpt-3.5-turbo if refusing
    if must_refuse:
        system = PRONOUN_RULE + FORMAT_RULE
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":system},
                {"role":"user",  "content":question}
            ]
        )
        final = resp.choices[0].message.content
        # record assistant reply into thread
        client.beta.threads.messages.create(thread_id=thread_id, role="assistant", content=final)
        history, _ = _chat_history(thread_id)
        return jsonify({"response":final,"thread_id":thread_id,"chat_history":history}), 200

    # 5) compose instructions + run Assistant thread
    instr = PRONOUN_RULE + FORMAT_RULE
    if ctx:
        instr += "\n\nUse these <doc …> blocks when relevant:\n\n" + ctx
    else:
        instr += "\n\nNo relevant internal documents were found. Politely explain you lack information and guide the user to ask about PlatypAI’s offerings."

    try:
        _run(thread_id, instr)
    except Exception as e:
        return jsonify({"response":f"Error: {e}","thread_id":thread_id}), 500

    # 6) gather and return
    history, final = _chat_history(thread_id)
    return jsonify({"response":final,"thread_id":thread_id,"chat_history":history}), 200

@app.errorhandler(Exception)
def handle_any(e):
    return jsonify({"response":f"Unhandled error: {e}","thread_id":None}),500

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",10000)),debug=False)
