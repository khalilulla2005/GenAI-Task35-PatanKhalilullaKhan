from __future__ import annotations

from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from src.agent import create_chat_model, create_math_agent, run_agent_turn

load_dotenv()
st.set_page_config(page_title="Text-to-Math Agent", page_icon="🧮", layout="centered")


@st.cache_resource
def _get_agent(provider: str):
    return create_math_agent(create_chat_model(provider))


if "math_agent_thread_id" not in st.session_state:
    st.session_state.math_agent_thread_id = uuid4().hex
if "math_agent_history" not in st.session_state:
    st.session_state.math_agent_history = []
if "math_agent_provider" not in st.session_state:
    st.session_state.math_agent_provider = None

st.title("Text-to-Math Problem Solver")
st.caption("Ask a word problem. The agent explains its setup and uses math tools to verify calculations.")

provider = st.selectbox(
    "Language model",
    options=("ollama", "openai", "huggingface"),
    format_func=lambda value: {
        "ollama": "Ollama (local)",
        "openai": "OpenAI",
        "huggingface": "Hugging Face",
    }[value],
)
if st.session_state.math_agent_provider != provider:
    if st.session_state.math_agent_provider is not None:
        st.session_state.math_agent_thread_id = uuid4().hex
        st.session_state.math_agent_history = []
        st.info("Provider changed; started a fresh conversation to avoid mixing model sessions.")
    st.session_state.math_agent_provider = provider

with st.sidebar:
    st.subheader("Try a problem")
    st.markdown(
        "- 3 notebooks cost $4 each. What is the total?\n"
        "- What is 15% of 80?\n"
        "- Solve `2*x + 3 = 11`."
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.math_agent_thread_id = uuid4().hex
        st.session_state.math_agent_history = []
        st.rerun()

for turn in st.session_state.math_agent_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("trace"):
            with st.expander("Calculation tool calls"):
                for index, step in enumerate(turn["trace"], start=1):
                    st.markdown(f"**Step {index} — `{step['tool']}`**")
                    st.code(f"{step['input']}\n= {step['observation']}", language="text")

question = st.chat_input("Enter an arithmetic, percentage, or algebra problem")
if question:
    st.session_state.math_agent_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Working through the problem..."):
            agent = _get_agent(provider)
            answer, trace = run_agent_turn(
                agent,
                question,
                session_id=st.session_state.math_agent_thread_id,
            )
        st.markdown(answer)
        if trace:
            with st.expander("Calculation tool calls", expanded=True):
                for index, step in enumerate(trace, start=1):
                    st.markdown(f"**Step {index} — `{step['tool']}`**")
                    st.code(f"{step['input']}\n= {step['observation']}", language="text")
    st.session_state.math_agent_history.append(
        {"role": "assistant", "content": answer, "trace": trace}
    )
