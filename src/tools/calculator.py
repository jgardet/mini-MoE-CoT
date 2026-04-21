"""
tools/calculator.py — Safe math evaluation tool.

The model generates <tool>calc(expression)</tool> calls.
This module parses and safely evaluates the math expression.

Safety: We use a whitelist of allowed operations rather than
raw eval(), which would be a security risk.
"""

import re
import math
import operator
from typing import Any


# Whitelist of safe operations the calculator can perform
SAFE_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def safe_eval(expression: str) -> Any:
    """Safely evaluate a mathematical expression.

    Uses Python's ast module to parse the expression and only
    allows whitelisted node types. No arbitrary code execution.

    Args:
        expression: Math expression string, e.g. "sqrt(16) + 3*4"

    Returns:
        Numeric result.

    Raises:
        ValueError: If expression contains disallowed operations.
    """
    import ast

    # Clean the expression
    expression = expression.strip()

    # Replace common math words
    for func_name, func_val in SAFE_FUNCTIONS.items():
        if isinstance(func_val, float):
            expression = expression.replace(func_name, str(func_val))

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_map = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }
            op_fn = op_map.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_fn(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return +operand
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            func_name = node.func.id
            if func_name not in SAFE_FUNCTIONS:
                raise ValueError(f"Function not allowed: {func_name}")
            func = SAFE_FUNCTIONS[func_name]
            args = [eval_node(a) for a in node.args]
            return func(*args)
        elif isinstance(node, ast.Name):
            if node.id in SAFE_FUNCTIONS and isinstance(SAFE_FUNCTIONS[node.id], float):
                return SAFE_FUNCTIONS[node.id]
            raise ValueError(f"Unknown name: {node.id}")
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    result = eval_node(tree)

    # Format result nicely
    if isinstance(result, float):
        if result == int(result):
            return int(result)
        return round(result, 6)
    return result


def run(expression: str) -> str:
    """Tool entry point called by the ToolDispatcher.

    Args:
        expression: Math expression from model's tool call.

    Returns:
        String result or error message.
    """
    try:
        result = safe_eval(expression)
        return f"Calculator result: {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"
