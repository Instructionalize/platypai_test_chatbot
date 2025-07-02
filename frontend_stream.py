# app.py – PlatypAI chat (bug-free + rerun fix)
import json, requests, textwrap, streamlit as st

# ────────────────────────── Page Setup & CSS ──────────────────────────
st.set_page_config(page_title="Chat with PlatypAI Bot", page_icon="🤖", layout="centered")

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

# ────────────────────────── Session State ──────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history [{role, message}]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

pending_q = st.session_state.get("pending_question")  # Queue

# ────────────────────────── Markdown Formatter ──────────────────────────
def md_to_html(md: str) -> str:
    md = md.replace("**", "<strong>").replace("__", "<strong>")
    md = md.replace("*", "<em>").replace("_", "<em>")
    out, in_ul = [], False
    for ln in textwrap.dedent(md).split("\n"):
        if ln.lstrip().startswith(("-", "•")):
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append(f"<li>{ln.lstrip()[1:].strip()}</li>")
        else:
            if in_ul: out.append("</ul>"); in_ul = False
            out.append(ln)
    if in_ul: out.append("</ul>")
    return "<br>".join(out)

# ────────────────────────── Draw Chat History ──────────────────────────
def draw_messages():
    for m in st.session_state.messages:
        st.markdown(
            f'''
            <div class="msg {m["role"]}">
              <strong>{"You" if m["role"]=="user" else "PlatypAI Bot"}</strong>
              {md_to_html(m["message"])}
            </div>
            ''',
            unsafe_allow_html=True,
        )

# ────────────────────────── Chat UI ──────────────────────────
with st.container():
    st.markdown('<div class="chatbox">', unsafe_allow_html=True)
    st.markdown("## Ask me anything… PlatypAI related!")
    draw_messages()

    # 1️⃣ Handle user input
    user_text = st.chat_input("Type your question…")

    if user_text:
        st.session_state.messages.append({"role": "user", "message": user_text})
        st.session_state.pending_question = user_text
        st.rerun()  # show message immediately

    # 2️⃣ Make backend request if needed
    if pending_q:
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="msg assistant typing">PlatypAI Bot is typing…</div>',
            unsafe_allow_html=True,
        )
        try:
            r = requests.post(
                "https://platypai-test-chatbot.onrender.com",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "question": pending_q,
                    "thread_id": st.session_state.thread_id
                }),
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("thread_id"):
                st.session_state.thread_id = data["thread_id"]

            # handle either full history or just new reply
            if "chat_history" in data:
                st.session_state.messages = data["chat_history"]
            elif "answer" in data:
                st.session_state.messages.append(
                    {"role": "assistant", "message": data["answer"]}
                )
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "message": "[No reply received]"}
                )

        except Exception as exc:
            st.session_state.messages.append(
                {"role": "assistant", "message": f"Error: {exc}"}
            )
        finally:
            del st.session_state.pending_question
            placeholder.empty()
            st.rerun()  # show bot reply

    st.markdown(
        '<div style="margin-top:40px;font-size:13px;text-align:center;color:#8F69A0">'
        "Chat UI styled for PlatypAI</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
