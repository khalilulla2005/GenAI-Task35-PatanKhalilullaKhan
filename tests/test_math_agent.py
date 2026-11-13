from __future__ import annotations

import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.agent import create_math_agent, run_agent_turn
from src.math_tools import calculator, solve_linear_equation


class FakeMathModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "math-agent-test-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        latest_question = next(
            message.content
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        earlier_questions = [
            message.content
            for message in messages[:-1]
            if isinstance(message, HumanMessage)
        ]
        latest_question_index = next(
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
        )
        current_turn_messages = messages[latest_question_index + 1 :]
        if latest_question.casefold().startswith("what was my target"):
            answer = (
                "Your target was 80, as mentioned earlier."
                if any("target is 80" in question.casefold() for question in earlier_questions)
                else "I do not have a target in this conversation."
            )
            response = AIMessage(content=answer)
        elif any(isinstance(message, ToolMessage) for message in current_turn_messages):
            observation = next(
                message.content
                for message in reversed(current_turn_messages)
                if isinstance(message, ToolMessage)
            )
            response = AIMessage(content=f"The verified result is {observation}.")
        else:
            question = latest_question.casefold()
            if "solve" in question or "equation" in question:
                name = "solve_linear_equation"
                args = {"equation": "2*x + 3 = 11"}
            elif "percent" in question:
                name = "calculator"
                args = {"expression": "15 / 100 * 80"}
            else:
                name = "calculator"
                args = {"expression": "3 * 4"}
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": args,
                        "id": f"call-{len(messages)}",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=response)])


class MathToolTests(unittest.TestCase):
    def test_calculator_supports_arithmetic_and_percentages(self):
        self.assertEqual(calculator.invoke({"expression": "(3 * 4) + 2"}), "14")
        self.assertEqual(calculator.invoke({"expression": "15 / 100 * 80"}), "12")

    def test_linear_solver_handles_simple_equations_and_edge_cases(self):
        self.assertEqual(solve_linear_equation.invoke({"equation": "2*x + 3 = 11"}), "x = 4")
        self.assertEqual(
            solve_linear_equation.invoke({"equation": "x + 2 = x + 2"}),
            "Every value of x is a solution.",
        )
        self.assertEqual(
            solve_linear_equation.invoke({"equation": "x + 1 = x + 2"}),
            "There is no solution.",
        )

    def test_tools_reject_code_and_nonlinear_equations(self):
        with self.assertRaises(ValueError):
            calculator.invoke({"expression": "__import__('os').system('whoami')"})
        with self.assertRaises(ValueError):
            solve_linear_equation.invoke({"equation": "x*x = 4"})


class AgentTests(unittest.TestCase):
    def test_agent_uses_tools_for_word_problem_and_preserves_session_context(self):
        agent = create_math_agent(FakeMathModel())

        answer, trace = run_agent_turn(
            agent,
            "I have 3 boxes with 4 apples each. How many apples?",
            session_id="test-session",
        )
        self.assertIn("12", answer)
        self.assertEqual(trace[0]["tool"], "calculator")
        self.assertEqual(trace[0]["observation"], "12")

        percentage_answer, percentage_trace = run_agent_turn(
            agent,
            "My target is 80. What is 15 percent of it?",
            session_id="test-session",
        )
        self.assertIn("12", percentage_answer)
        self.assertEqual(percentage_trace[0]["tool"], "calculator")

        context_answer, context_trace = run_agent_turn(
            agent,
            "What was my target?",
            session_id="test-session",
        )
        self.assertIn("80", context_answer)
        self.assertEqual(context_trace, [])

        isolated_answer, _ = run_agent_turn(
            agent,
            "What was my target?",
            session_id="separate-session",
        )
        self.assertIn("do not have a target", isolated_answer)

    def test_agent_uses_equation_tool_for_algebra(self):
        agent = create_math_agent(FakeMathModel())
        answer, trace = run_agent_turn(
            agent,
            "Solve the equation 2*x + 3 = 11.",
            session_id="algebra-session",
        )
        self.assertIn("x = 4", answer)
        self.assertEqual(trace[0]["tool"], "solve_linear_equation")


if __name__ == "__main__":
    unittest.main()
