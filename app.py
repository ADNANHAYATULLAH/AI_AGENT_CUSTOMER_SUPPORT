import os
import json
from pathlib import Path

import streamlit as st

from agent import run_support_agent
from tools import get_pending_escalations


st.set_page_config(
    page_title="Daraz Customer Support AI",
    page_icon="🛍️",
    layout="wide",
)


def load_api_key():
    # Streamlit Cloud: st.secrets
    # Local testing: environment variable
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GEMINI_API_KEY", "")


def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_seen" not in st.session_state:
        st.session_state.pending_seen = 0


def render_sidebar():
    with st.sidebar:
        st.title("🛍️ Daraz Support")
        st.caption("Single-agent RAG customer support")

        st.divider()

        st.subheader("Agent tools")
        st.write("📚 Company policy RAG")
        st.write("📦 Order.xlsx lookup")
        st.write("👤 Human escalation")

        st.divider()

        pending = get_pending_escalations()
        st.metric("Pending human tickets", len(pending))

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.caption(
            "Demo application. Policy and order data are fictional and are used only "
            "for RAG/agent development."
        )


def render_pending():
    st.subheader("📋 Pending Human Support")

    pending = get_pending_escalations()

    if not pending:
        st.info("No pending escalations.")
        return

    for ticket in reversed(pending):
        with st.expander(
            f"{ticket.get('ticket_id', 'Unknown')} — {ticket.get('status', 'Pending')}"
        ):
            st.write(f"**Created:** {ticket.get('created_at', '')}")
            st.write(f"**Order:** {ticket.get('order_id') or 'Not provided'}")
            st.write(f"**Reason:** {ticket.get('reason', '')}")
            st.write(f"**Customer message:** {ticket.get('customer_message', '')}")
            st.write(f"**Summary:** {ticket.get('conversation_summary', '')}")


def main():
    initialize_session()
    render_sidebar()

    st.title("🛍️ Daraz Customer Support AI Agent")
    st.caption(
        "Ask about company policies or your order. The agent can escalate unresolved "
        "issues to human support."
    )

    api_key = load_api_key()

    if not api_key:
        st.error(
            "GEMINI_API_KEY is missing. For Streamlit Cloud, add it under "
            "Settings → Secrets. For local testing, set GEMINI_API_KEY in your environment."
        )
        st.stop()

    tab_chat, tab_pending = st.tabs(["💬 Customer Chat", "👤 Pending"])

    with tab_chat:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("How can I help you?")

        if prompt:
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Checking company knowledge and order information..."):
                    try:
                        answer = run_support_agent(
                            prompt,
                            st.session_state.messages,
                        )
                    except Exception as exc:
                        answer = (
                            "I’m sorry, I could not process your request right now. "
                            "Please try again or ask for human support."
                        )
                        st.error(f"Agent error: {type(exc).__name__}: {exc}")

                st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

    with tab_pending:
        render_pending()


if __name__ == "__main__":
    main()
