from __future__ import annotations

import ast
import operator
from decimal import Decimal, InvalidOperation
from typing import Callable

from langchain_core.tools import BaseTool, tool

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Decimal, Decimal], Decimal]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Decimal], Decimal]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPRESSION_LENGTH = 256
_MAX_AST_NODES = 64
_MAX_RESULT_MAGNITUDE = Decimal("1e50")


def _format_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("The result must be a finite number.")
    if abs(value) > _MAX_RESULT_MAGNITUDE:
        raise ValueError("The result is outside the supported numeric range.")
    if value == value.to_integral_value():
        return format(value.to_integral_value(), "f")
    return format(value.normalize(), "f")


def _parse_expression(expression: str) -> ast.Expression:
    if not expression.strip() or len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Enter a non-empty expression of at most {_MAX_EXPRESSION_LENGTH} characters."
        )
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError("The expression is not valid arithmetic.") from exc
    if sum(1 for _ in ast.walk(parsed)) > _MAX_AST_NODES:
        raise ValueError("The expression is too complex.")
    return parsed


def _constant_value(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Only numeric constants are supported.")
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid numeric constant.") from exc


def _apply_binary_operator(
    operator_node: ast.operator,
    left: Decimal,
    right: Decimal,
) -> Decimal:
    operation = _BINARY_OPERATORS.get(type(operator_node))
    if operation is None:
        raise ValueError("Only basic arithmetic operators are allowed.")
    if isinstance(operator_node, ast.Pow):
        if right != right.to_integral_value() or abs(right) > 12:
            raise ValueError("Exponents must be whole numbers between -12 and 12.")
    try:
        result = operation(left, right)
    except (ArithmeticError, InvalidOperation) as exc:
        raise ValueError(f"Could not evaluate arithmetic: {exc}") from exc
    if not result.is_finite() or abs(result) > _MAX_RESULT_MAGNITUDE:
        raise ValueError("The result is outside the supported numeric range.")
    return result


def _evaluate(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        return _constant_value(node.value)
    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Only unary plus and minus are allowed.")
        return operation(_evaluate(node.operand))
    if isinstance(node, ast.BinOp):
        return _apply_binary_operator(node.op, _evaluate(node.left), _evaluate(node.right))
    raise ValueError("Only numeric literals and basic arithmetic are allowed.")


def _linear_form(node: ast.AST) -> tuple[Decimal, Decimal]:
    if isinstance(node, ast.Expression):
        return _linear_form(node.body)
    if isinstance(node, ast.Name) and node.id == "x":
        return Decimal("1"), Decimal("0")
    if isinstance(node, ast.Constant):
        return Decimal("0"), _constant_value(node.value)
    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Only unary plus and minus are allowed.")
        coefficient, constant = _linear_form(node.operand)
        return operation(coefficient), operation(constant)
    if not isinstance(node, ast.BinOp):
        raise ValueError("Use only x, numbers, and basic arithmetic in the equation.")

    left_a, left_b = _linear_form(node.left)
    right_a, right_b = _linear_form(node.right)
    if isinstance(node.op, ast.Add):
        return left_a + right_a, left_b + right_b
    if isinstance(node.op, ast.Sub):
        return left_a - right_a, left_b - right_b
    if isinstance(node.op, ast.Mult):
        if left_a and right_a:
            raise ValueError("The equation must be linear; x cannot be multiplied by x.")
        if left_a:
            return left_a * right_b, left_b * right_b
        if right_a:
            return right_a * left_b, right_b * left_b
        return Decimal("0"), left_b * right_b
    if isinstance(node.op, ast.Div):
        if right_a:
            raise ValueError("Division by an expression containing x is not supported.")
        if right_b == 0:
            raise ValueError("Division by zero is not allowed.")
        return left_a / right_b, left_b / right_b
    if isinstance(node.op, ast.Pow):
        if right_a:
            raise ValueError("An exponent containing x is not supported.")
        if right_b == 1:
            return left_a, left_b
        if left_a:
            if right_b == 0:
                return Decimal("0"), Decimal("1")
            raise ValueError("The equation must be linear; powers of x are not supported.")
        return Decimal("0"), _apply_binary_operator(node.op, left_b, right_b)
    raise ValueError("Only basic arithmetic operators are allowed in the equation.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression such as '(18 * 7) + 4'."""
    parsed = _parse_expression(expression)
    return _format_decimal(_evaluate(parsed))


@tool
def solve_linear_equation(equation: str) -> str:
    """Solve a one-variable linear equation, using x (example: '2*x + 3 = 11')."""
    if not equation.strip() or len(equation) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Enter a non-empty equation of at most {_MAX_EXPRESSION_LENGTH} characters."
        )
    if equation.count("=") != 1:
        raise ValueError("Enter one equation with exactly one equals sign.")

    left, right = (part.strip() for part in equation.split("=", maxsplit=1))
    left_form = _linear_form(_parse_expression(left))
    right_form = _linear_form(_parse_expression(right))
    coefficient = left_form[0] - right_form[0]
    constant = right_form[1] - left_form[1]
    if coefficient == 0:
        if constant == 0:
            return "Every value of x is a solution."
        return "There is no solution."
    try:
        solution = constant / coefficient
    except (ArithmeticError, InvalidOperation) as exc:
        raise ValueError(f"Could not solve the equation: {exc}") from exc
    return f"x = {_format_decimal(solution)}"


def get_math_tools() -> list[BaseTool]:
    return [calculator, solve_linear_equation]
