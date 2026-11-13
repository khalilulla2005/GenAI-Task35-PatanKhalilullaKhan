from __future__ import annotations

import argparse
import json
from uuid import uuid4

from dotenv import load_dotenv

from src.agent import create_chat_model, create_math_agent, run_agent_turn


def _print_trace(trace: list[dict]) -> None:
    for index, step in enumerate(trace, start=1):
        arguments = json.dumps(step["input"], ensure_ascii=False)
        print(f"\nStep {index}: {step['tool']}({arguments})")
        print(f"Calculation: {step['observation']}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Solve arithmetic, percentage, and simple algebra word problems."
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "ollama", "huggingface"),
        help="LLM provider; defaults to MATH_AGENT_PROVIDER or ollama.",
    )
    parser.add_argument("--question", help="Solve one question instead of starting chat.")
    args = parser.parse_args()

    agent = create_math_agent(create_chat_model(args.provider))
    session_id = uuid4().hex
    if args.question:
        answer, trace = run_agent_turn(agent, args.question, session_id=session_id)
        _print_trace(trace)
        print(f"\nAnswer: {answer}")
        return

    print("Text-to-Math Agent. Type 'exit' to end this session.")
    while True:
        question = input("\nMath problem: ").strip()
        if question.casefold() in {"exit", "quit", "q"}:
            break
        if not question:
            print("Enter a math problem, or type 'exit'.")
            continue
        answer, trace = run_agent_turn(agent, question, session_id=session_id)
        _print_trace(trace)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
