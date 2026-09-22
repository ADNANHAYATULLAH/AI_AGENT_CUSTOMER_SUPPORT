import os
import html
import streamlit as st
from agent import run_support_agent
from tools import get_pending_escalations

st.set_page_config(page_title="Daraz AI Support", page_icon="🛍️", layout="wide")

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 85% 5%,rgba(255,82,82,.10),transparent 25%),radial-gradient(circle at 10% 15%,rgba(98,0,238,.08),transparent 28%),#f7f8fc}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#171725,#292943)}
[data-testid="stSidebar"] *{color:#f7f7fb!important}
.hero{background:linear-gradient(135deg,#fff,#f2f0ff);border:1px solid #e7e4f7;border-radius:24px;padding:28px 30px;margin-bottom:18px;box-shadow:0 10px 35px rgba(31,31,55,.07)}
.hero-title{font-size:34px;font-weight:850;color:#171725}
.hero-text{color:#656579;font-size:15px;line-height:1.6}
.badge{display:inline-block;padding:6px 11px;border-radius:999px;background:#ece9ff;color:#5b45d6;font-size:12px;font-weight:700;margin-bottom:10px}
.card{background:white;border:1px solid #ececf2;border-radius:18px;padding:18px;min-height:110px;box-shadow:0 7px 22px rgba(30,30,50,.045)}
.chat-user,.chat-bot{padding:15px 17px;border-radius:18px;margin:8px 0 12px;line-height:1.6;font-size:14px}
.chat-user{background:#ebe8ff;border:1px solid #ded9ff;margin-left:8%}
.chat-bot{background:white;border:1px solid #e9e9ef;margin-right:8%;box-shadow:0 5px 18px rgba(30,30,50,.035)}
.label{font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.6px;color:#77778a;margin-bottom:5px}
</style>
""", unsafe_allow_html=True)

api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
st.session_state.setdefault("messages", [])
st.session_state.setdefault("quick_question", "")

with st.sidebar:
    st.markdown("## 🛍️ Daraz AI Support")
    st.caption("Smart customer assistance")
    pending = get_pending_escalations()
    st.metric("👤 Pending tickets", len(pending))
    st.divider()
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("### 🧠 AI capabilities")
    st.markdown("📚 **Policy knowledge**\n\n📦 **Order lookup**\n\n👤 **Human escalation**\n\n💬 **Session context**")
    st.divider()
    st.caption("Fictional demo • Gemini + CrewAI + FAISS")

st.markdown("""
<div class="hero">
<div class="badge">AI CUSTOMER SUPPORT AGENT</div>
<div class="hero-title">How can we help you today? 👋</div>
<div class="hero-text">Ask about orders, payments, delivery, returns, refunds, or store policies. The assistant checks available knowledge and order data before answering.</div>
</div>
""", unsafe_allow_html=True)

st.subheader("✨ Quick help")
a,b,c,d=st.columns(4)
questions=[("📦 Order status","Where is my order?"),("↩️ Returns","Can I return my order?"),("💳 Payments","What payment methods are supported?"),("🚚 Delivery","How long does delivery take?")]
for col,(label,q) in zip((a,b,c,d),questions):
    with col:
        st.markdown(f'<div class="card"><b>{label}</b><br><span style="color:#777;font-size:13px">{q}</span></div>',unsafe_allow_html=True)
        if st.button("Ask AI",key=q,use_container_width=True):
            st.session_state.quick_question=q

st.subheader("💬 Support chat")
for m in st.session_state.messages:
    content=html.escape(str(m["content"])).replace("\n","<br>")
    if m["role"]=="user":
        st.markdown(f'<div class="chat-user"><div class="label">You</div>{content}</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot"><div class="label">🤖 AI Support</div>{content}</div>',unsafe_allow_html=True)

prompt=st.chat_input("Ask about your order, return, payment, delivery or refund...")
if st.session_state.quick_question:
    prompt=st.session_state.quick_question
    st.session_state.quick_question=""

if prompt:
    if not api_key:
        st.error("GEMINI_API_KEY is not configured in Streamlit Secrets.")
        st.stop()
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.spinner("🔎 Checking policies and order information..."):
        try:
            answer=run_support_agent(prompt,st.session_state.messages[:-1])
        except Exception as exc:
            answer=f"Sorry, I could not complete that request.\n\nTechnical error: {exc}"
    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()

with st.expander("👤 Human support queue"):
    pending=get_pending_escalations()
    if not pending:
        st.info("No pending human escalation tickets.")
    else:
        for t in reversed(pending):
                      st.markdown(f"**{t.get('ticket_id','Unknown')}** · `{t.get('status','Pending')}`\n\nReason: {t.get('reason','Not specified')}  \nOrder: {t.get('order_id','Not provided')}")
            st.divider()

st.markdown('<div style="text-align:center;color:#999;padding:25px">Daraz-style fictional customer support demo • Not an official Daraz system</div>',unsafe_allow_html=True)
