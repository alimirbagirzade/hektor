"""Güvenli aritmetik değerlendirici — ``eval``/``exec`` OLMADAN (CLAUDE.md Kural 5).

Yalnızca skaler aritmetik bir ifadeyi, verilen sembol→sayı eşlemesi üzerinde hesaplar.
``ast`` ağacını whitelist'li bir ``NodeVisitor`` ile gezer; izin verilmeyen her düğüm
(``__import__``, attribute erişimi, subscript, lambda, comprehension, isim çağrıları vb.)
``UnsafeExpressionError`` fırlatır. Bu motor, registry-dışı (makaleden çıkarılmış)
formüllerin SAYISAL referansını üretmek için kullanılır; modelin ürettiği KOD asla
çalıştırılmaz — yalnız bilinen güvenli ifadeler değerlendirilir.

İzin verilen: sabit sayılar, tanımlı semboller, +,-,*,/,**,%,// , tekli +/- ,
ve {abs, min, max, sqrt, exp, log, log2, log10} fonksiyon çağrıları.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["UnsafeExpressionError", "safe_eval"]


class UnsafeExpressionError(ValueError):
    """İfade whitelist dışı bir yapı içeriyor (güvenlik reddi)."""


# Skaler, yan-etkisiz fonksiyonlar. min/max/abs builtin; geri kalanı math.
_ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,  # math.log(x) doğal log; math.log(x, base) de desteklenir
    "log2": math.log2,
    "log10": math.log10,
}

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def safe_eval(expr: str, variables: Mapping[str, float]) -> float:
    """``expr`` ifadesini ``variables`` üzerinde güvenle değerlendirir.

    eval/exec kullanmaz. Whitelist dışı yapı → ``UnsafeExpressionError``.
    Bölme/üs gibi işlemlerdeki ``ZeroDivisionError``/``OverflowError`` doğal olarak
    yükselir (matematiksel geçersizlik — çağıran ele alır).
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Ayrıştırılamadı: {exc}") from exc
    return _eval_node(tree.body, variables)


def _eval_node(node: ast.AST, variables: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant):
        # bool, int'in alt tipidir — açıkça reddet; yalnız gerçek sayı kabul
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError(f"İzin verilmeyen sabit: {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id in variables:
            return float(variables[node.id])
        raise UnsafeExpressionError(f"Tanımsız sembol: {node.id!r}")

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise UnsafeExpressionError(f"İzin verilmeyen işlem: {type(node.op).__name__}")
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        return _apply_binop(node.op, left, right)

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise UnsafeExpressionError(f"İzin verilmeyen tekli işlem: {type(node.op).__name__}")
        operand = _eval_node(node.operand, variables)
        return operand if isinstance(node.op, ast.UAdd) else -operand

    if isinstance(node, ast.Call):
        if node.keywords or not isinstance(node.func, ast.Name):
            raise UnsafeExpressionError("Yalnız konumsal argümanlı basit fonksiyon çağrısı")
        fname = node.func.id
        if fname not in _ALLOWED_FUNCS:
            raise UnsafeExpressionError(f"İzin verilmeyen fonksiyon: {fname!r}")
        args = [_eval_node(a, variables) for a in node.args]
        return float(_ALLOWED_FUNCS[fname](*args))

    raise UnsafeExpressionError(f"İzin verilmeyen ifade düğümü: {type(node).__name__}")


def _apply_binop(op: ast.operator, left: float, right: float) -> float:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div):
        return left / right
    if isinstance(op, ast.Pow):
        return left**right
    if isinstance(op, ast.Mod):
        return left % right
    if isinstance(op, ast.FloorDiv):
        return left // right
    raise UnsafeExpressionError(f"İzin verilmeyen işlem: {type(op).__name__}")
