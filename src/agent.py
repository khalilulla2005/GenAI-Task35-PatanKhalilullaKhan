from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.math_tools import get_math_tools

SYSTEM_PROMPT = """You are a careful text-to-math problem solver.
Translate word problems into clear mathematical steps. Use the calculator tool for
arithmetic and percentages, and use solve_linear_equation for one-variable linear
equations. Never invent a tool result. After the tool returns, explain the setup,
show the important calculation steps, and state the final answer with units when
they are known. If the problem is ambiguous, ask a brief clarifying question."""


def create_chat_model(provider: str | None = None) -> BaseChatModel:
    load_dotenv()
    provider_name = (provider or os.getenv("MATH_AGENT_PROVIDER", "ollama")).casefold()

    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=api_key,
        )

    if provider_name == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )

    if provider_name == "huggingface":
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN or HF_TOKEN is required for Hugging Face."
            )
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            huggingfacehub_api_token=token,
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError("Unsupported provider. Choose openai, ollama, or huggingface.")


def create_math_agent(
    model: BaseChatModel,
    checkpointer: InMemorySaver | None = None,
):
    return create_agent(
        model=model,
        tools=get_math_tools(),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
        name="text_to_math_agent",
    )


def _content_as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _turn_messages(messages: list[Any], question: str) -> list[Any]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage) and _content_as_text(message.content) == question:
            return messages[index:]
    raise RuntimeError("The agent response did not contain the submitted question.")


def run_agent_turn(
    agent,
    question: str,
    session_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Enter a math question before submitting.")

    thread_id = session_id or uuid4().hex
    result = agent.invoke(
        {"messages": [HumanMessage(content=normalized_question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = _turn_messages(result.get("messages", []), normalized_question)

    trace_by_id: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                step = {
                    "tool": tool_call["name"],
                    "input": tool_call.get("args", {}),
                    "observation": "",
                }
                trace.append(step)
                if tool_call.get("id"):
                    trace_by_id[str(tool_call["id"])] = step
        elif isinstance(message, ToolMessage):
            step = trace_by_id.get(str(message.tool_call_id))
            if step is not None:
                step["observation"] = _content_as_text(message.content)

    final_message = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, AIMessage) and not message.tool_calls
        ),
        None,
    )
    if final_message is None:
        raise RuntimeError("The math agent did not return a final answer.")
    answer = _content_as_text(final_message.content).strip()
    if not answer:
        raise RuntimeError("The math agent returned an empty final answer.")
    return answer, trace
