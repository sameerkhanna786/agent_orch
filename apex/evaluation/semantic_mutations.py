"""Semantic mutation operators for Decisive-Edge Phase D.3.

The legacy mutation_engine generates *syntactic* mutants by swapping AST
operator tokens (e.g. ``<`` → ``<=``, ``+`` → ``-``) at every eligible
node and splicing the mutated source back in by character offsets.
Those mutants exercise an agent's tests against arithmetic and boundary
edges but they leave a large class of plausible-but-wrong behaviors
untested: off-by-one loop bounds (``range(n)`` → ``range(n+1)``),
function-call confusion (``min`` ↔ ``max``), Pythonic-but-wrong indexing
(``arr[0]`` → ``arr[1]``), branch removal, etc. Those are the mutations
human reviewers actually catch in production code review.

This module provides 10 *semantic* operators that operate at higher
levels of the AST. Each operator:

* takes a parsed ``ast.AST`` source tree and a target node;
* returns ``(mutated_source, description)`` if it can apply, else
  ``None``;
* uses ``ast.unparse`` for source emission (no string slicing) so the
  result is always syntactically valid Python; and
* is **deterministic** — the same input always yields the same output
  (no RNG, no per-process state).

Operators register into :class:`MutationEngineRegistry`, which exposes
``apply_to_function`` for the integration with
:mod:`apex.evaluation.mutation_engine`.

The operators are designed to *replace one specific behavior* per mutant
rather than transforming the whole module — this keeps the kill signal
attributable. ``RemoveBranch`` / ``EmptyLoopBody`` are the only operators
that can produce visibly large diffs, and they do so on a single block.
"""

from __future__ import annotations

import ast
import copy
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Constant tag for the report so reviewers can break down kill rate by
# mutation source (syntactic vs. semantic). The mutation_engine layer
# stamps this onto every Mutant emitted from this module.
MUTATION_KIND = "semantic"


# ---------------------------------------------------------------------------
# Operator infrastructure
# ---------------------------------------------------------------------------


@dataclass
class SemanticMutation:
    """Result of a successful operator application."""

    name: str
    description: str
    mutated_source: str
    line: int
    col: int
    mutation_kind: str = MUTATION_KIND


class SemanticOperator:
    """Base class for a semantic operator.

    Subclasses implement :meth:`apply` and supply a class-level ``name``.
    The :meth:`apply_to_module` helper walks the module AST and yields
    one :class:`SemanticMutation` per applicable site, deterministically
    ordered by ``(lineno, col_offset, operator_name)``.
    """

    name: str = "base"
    mutation_kind: str = MUTATION_KIND

    def apply(self, source: str, target_node: ast.AST) -> Optional[tuple[str, str]]:
        """Apply this operator at ``target_node``.

        Args:
            source: Original module source. Used to render the mutated
                module via ``ast.unparse`` after deep-copying the tree.
            target_node: The node to mutate. Implementations type-check
                the node and return ``None`` if it is not eligible.

        Returns:
            ``(mutated_source, description)`` if the operator applied,
            else ``None``. The mutated source is the full module text
            ready to be written back to disk.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers shared by subclasses.
    # ------------------------------------------------------------------

    @staticmethod
    def _render(tree: ast.AST) -> str:
        """Re-emit a (possibly-mutated) tree to source.

        ``ast.unparse`` is the canonical way to do this in 3.9+. We add
        a trailing newline so file writes are POSIX-clean.
        """
        text = ast.unparse(tree)
        if not text.endswith("\n"):
            text += "\n"
        return text

    @staticmethod
    def _replace_node(tree: ast.Module, target_id: int, replacement: ast.AST) -> bool:
        """Swap ``target_id`` (an id() of a node in ``tree``) with
        ``replacement`` in-place. Returns True on success.

        Walks the tree once and replaces the first child node whose
        ``id()`` matches. Used by operators that can express their edit
        as "swap node X with node Y" without restructuring the parent.
        """
        for parent in ast.walk(tree):
            for field_name, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    for index, child in enumerate(value):
                        if id(child) == target_id:
                            value[index] = replacement
                            return True
                elif id(value) == target_id:
                    setattr(parent, field_name, replacement)
                    return True
        return False


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class OffByOneInLoopBound(SemanticOperator):
    """Shift a ``range()`` upper bound by ±1.

    Targets ``for _ in range(n)`` and ``for _ in range(start, end[, step])``.
    Mutates the *last positional* argument (the stop value) by adding 1.
    The "by 1" choice is deliberate — most off-by-one bugs in the wild
    are over-counts, so over-counting in the mutant tends to expose
    agent tests that don't pin the loop length precisely.
    """

    name = "off_by_one_loop_bound"

    def apply(self, source: str, target_node: ast.AST) -> Optional[tuple[str, str]]:
        if not isinstance(target_node, ast.Call):
            return None
        func = target_node.func
        if not isinstance(func, ast.Name) or func.id != "range":
            return None
        if not target_node.args:
            return None
        # Mutate the *stop* argument: positional index 0 for range(n),
        # index 1 for range(start, stop[, step]).
        stop_index = 0 if len(target_node.args) == 1 else 1
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        twin = _equivalent_in(other=tree, original=target_node)
        if twin is None or not isinstance(twin, ast.Call) or stop_index >= len(twin.args):
            return None
        original_stop = twin.args[stop_index]
        new_stop = ast.BinOp(
            left=copy.deepcopy(original_stop),
            op=ast.Add(),
            right=ast.Constant(value=1),
        )
        twin.args[stop_index] = new_stop
        ast.fix_missing_locations(twin)
        return self._render(tree), "Increment range() stop by 1"


class SwapSimilarFunctionCalls(SemanticOperator):
    """Swap calls to confusable builtin / method names.

    Pairs (bidirectional):
      min ↔ max, sum ↔ len, any ↔ all, lower ↔ upper,
      strip ↔ lstrip, append ↔ extend, keys ↔ values
    """

    name = "swap_similar_function_calls"

    _PAIRS = {
        "min": "max",
        "max": "min",
        "sum": "len",
        "len": "sum",
        "any": "all",
        "all": "any",
        "lower": "upper",
        "upper": "lower",
        "strip": "lstrip",
        "lstrip": "strip",
        "append": "extend",
        "extend": "append",
        "keys": "values",
        "values": "keys",
    }

    def apply(self, source: str, target_node: ast.AST) -> Optional[tuple[str, str]]:
        if not isinstance(target_node, ast.Call):
            return None
        func = target_node.func
        original_name: Optional[str] = None
        is_attr = False
        if isinstance(func, ast.Name) and func.id in self._PAIRS:
            original_name = func.id
        elif isinstance(func, ast.Attribute) and func.attr in self._PAIRS:
            original_name = func.attr
            is_attr = True
        if original_name is None:
            return None
        new_name = self._PAIRS[original_name]
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        twin = _equivalent_in(other=tree, original=target_node)
        if twin is None or not isinstance(twin, ast.Call):
            return None
        if is_attr and isinstance(twin.func, ast.Attribute):
            twin.func.attr = new_name
        elif isinstance(twin.func, ast.Name):
            twin.func.id = new_name
        else:
            return None
        return (
            self._render(tree),
            f"Swap call {original_name}() -> {new_name}()",
        )


class SwapComparisonOperators(SemanticOperator):
    """Swap a ``Compare`` operator with its near-neighbor.

    Pairings: ``>`` → ``>=``, ``>=`` → ``>``, ``<`` → ``<=``,
    ``<=`` → ``<``, ``==`` → ``!=``, ``!=`` → ``==``.

    Only the first comparator op is mutated to keep the kill signal
    pinned to a single source location.
    """

    name = "swap_comparison_operators"

    _SWAP = {
        ast.Gt: (ast.GtE, ">", ">="),
        ast.GtE: (ast.Gt, ">=", ">"),
        ast.Lt: (ast.LtE, "<", "<="),
        ast.LtE: (ast.Lt, "<=", "<"),
        ast.Eq: (ast.NotEq, "==", "!="),
        ast.NotEq: (ast.Eq, "!=", "=="),
    }

    def apply(self, source: str, target_node: ast.AST) -> Optional[tuple[str, str]]:
        if not isinstance(target_node, ast.Compare) or not target_node.ops:
            return None
        first_op = target_node.ops[0]
        op_type = type(first_op)
        if op_type not in self._SWAP:
            return None
        replacement_cls, original_str, replacement_str = self._SWAP[op_type]
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        twin = _equivalent_in(other=tree, original=target_node)
        if twin is None or not isinstance(twin, ast.Compare) or not twin.ops:
            return None
        twin.ops[0] = replacement_cls()
        ast.fix_missing_locations(twin)
        return (
            self._render(tree),
            f"Swap comparison {original_str} -> {replacement_str}",
        )


class SignFlip(SemanticOperator):
    """Flip the sign of a numeric expression.

    Operates on (a) a top-level numeric literal — ``42`` → ``-42`` — and
    (b) a ``BinOp`` with ``Add`` or ``Sub`` — ``a + b`` → ``a - b`` and
    vice versa. Conceptually one operator with two dispatches; for a
    given node only one branch fires.
    """

    name = "sign_flip"

    def apply(self, source: str, target_node: ast.AST) -> Optional[tuple[str, str]]:
        if (
            isinstance(target_node, ast.Constant)
            and isinstance(target_node.value, (int, float))
            and not isinstance(target_node.value, bool)
        ):
            value = target_node.value
            if value == 0:
                return None  # negating 0 yields no behavioral diff
            new_value = -value
            try:
                tree = ast.parse(source)
            except SyntaxError:
                return None
            twin = _equivalent_in(other=tree, original=target_node)
            if twin is None or not isinstance(twin, ast.Constant):
                return None
            twin.value = new_value
            ast.fix_missing_locations(twin)
            return (
                self._render(tree),
                f"Flip sign of constant {value} -> {new_value}",
            )
        if isinstance(target_node, ast.BinOp) and isinstance(target_node.op, (ast.Add, ast.Sub)):
            try:
                tree = ast.parse(source)
            except SyntaxError:
                return None
            twin = _equivalent_in(other=tree, original=target_node)
            if twin is None or not isinstance(twin, ast.BinOp):
                return None
            if isinstance(twin.op, ast.Add):
                twin.op = ast.Sub()
                desc = "Flip + -> -"
            else:
                twin.op = ast.Add()
                desc = "Flip - -> +"
            ast.fix_missing_locations(twin)
            return self._render(tree), desc
        return None


class BooleanInvert(SemanticOperator):
    """Invert booleans, negation, and short-circuits.

    * ``True`` → ``False`` (and vice versa)
    * ``not x`` → ``x`` (drop the negation)
    * ``x and y`` → ``x or y`` (and vice versa)
    """

    name = "boolean_invert"

    def apply(self, source: str, target_node: ast.AST) -> Optional[tuple[str, str]]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        twin = _equivalent_in(other=tree, original=target_node)
        if twin is None:
            return None
        if isinstance(twin, ast.Constant) and isinstance(twin.value, bool):
            old = twin.value
            twin.value = not old
            ast.fix_missing_locations(twin)
            return (
                self._render(tree),
                f"Invert boolean {old} -> {twin.value}",
            )
        if isinstance(twin, ast.UnaryOp) and isinstance(twin.op, ast.Not):
            # Replace ``not X`` with ``X`` by swapping the parent's slot.
            replacement = twin.operand
            ok = self._replace_node(tree=tree, target_id=id(twin), replacement=replacement)
            if not ok:
                return None
            ast.fix_missing_locations(tree)
            return self._render(tree), "Drop unary not"
        if isinstance(twin, ast.BoolOp):
            if isinstance(twin.op, ast.And):
                twin.op = ast.Or()
                desc = "Swap and -> or"
            elif isinstance(twin.op, ast.Or):
                twin.op = ast.And()
                desc = "Swap or -> and"
            else:
                return None
            ast.fix_missing_locations(twin)
            return self._render(tree), desc
        return None


class OffByOneIndexing(SemanticOperator):
    """Shift a ``Subscript`` literal index by 1.

    Cases:
      * ``arr[0]`` → ``arr[1]``
      * ``arr[-1]`` → ``arr[-2]``
      * ``arr[i]`` → ``arr[i + 1]`` (for non-literal indices)

    Slicing (``arr[a:b]``) and tuple-indexing (``arr[i, j]``) are out
    of scope — they belong to a separate operator.
    """

    name = "off_by_one_indexing"

    def apply(self, source: str, target_node: ast.AST) -> Optional[tuple[str, str]]:
        if not isinstance(target_node, ast.Subscript):
            return None
        slice_node = target_node.slice
        # Skip slices and tuples — they have their own off-by-one mode.
        if isinstance(slice_node, (ast.Slice, ast.Tuple)):
            return None
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        twin = _equivalent_in(other=tree, original=target_node)
        if twin is None or not isinstance(twin, ast.Subscript):
            return None
        existing = twin.slice
        # Detect literal positive int (Constant) or literal negative int
        # (UnaryOp(USub, Constant)) and shift away from zero.
        if (
            isinstance(existing, ast.Constant)
            and isinstance(existing.value, int)
            and not isinstance(existing.value, bool)
        ):
            old = existing.value
            new = old + 1 if old >= 0 else old - 1
            twin.slice = ast.Constant(value=new)
            desc = f"Shift index {old} -> {new}"
        elif (
            isinstance(existing, ast.UnaryOp)
            and isinstance(existing.op, ast.USub)
            and isinstance(existing.operand, ast.Constant)
            and isinstance(existing.operand.value, int)
            and not isinstance(existing.operand.value, bool)
        ):
            magnitude = existing.operand.value
            new_magnitude = magnitude + 1
            twin.slice = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=new_magnitude))
            desc = f"Shift index -{magnitude} -> -{new_magnitude}"
        else:
            twin.slice = ast.BinOp(
                left=copy.deepcopy(existing), op=ast.Add(), right=ast.Constant(value=1)
            )
            desc = "Shift dynamic index by +1"
        ast.fix_missing_locations(twin)
        return self._render(tree), desc


class ConstantPerturb(SemanticOperator):
    """Perturb numeric literals.

    * Small ints (``|n| <= 100``): ``n`` → ``n + 1``.
    * Floats: ``v`` → ``v + 0.1`` (shifts ``0.5`` → ``0.6``).

    Booleans, large ints, and complex numbers are out of scope.
    """

    name = "constant_perturb"

    def apply(self, source: str, target_node: ast.AST) -> Optional[tuple[str, str]]:
        if not isinstance(target_node, ast.Constant):
            return None
        value = target_node.value
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and abs(value) <= 100:
            new_value: float | int = value + 1
            desc = f"Perturb int {value} -> {new_value}"
        elif isinstance(value, float):
            new_value = round(value + 0.1, 6)
            desc = f"Perturb float {value} -> {new_value}"
        else:
            return None
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        twin = _equivalent_in(other=tree, original=target_node)
        if twin is None or not isinstance(twin, ast.Constant):
            return None
        twin.value = new_value
        ast.fix_missing_locations(twin)
        return self._render(tree), desc


class RemoveBranch(SemanticOperator):
    """Remove the ``else`` arm of an ``if`` / ``orelse`` of a loop.

    For ``if cond: <body> else: <else>`` the mutant becomes
    ``if cond: <body>``. If there is no ``else`` arm, removes the
    ``if`` body instead (replacing it with ``pass``) — that flips the
    branch the agent's tests are exercising.
    """

    name = "remove_branch"

    def apply(self, source: str, target_node: ast.AST) -> Optional[tuple[str, str]]:
        if not isinstance(target_node, ast.If):
            return None
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        twin = _equivalent_in(other=tree, original=target_node)
        if twin is None or not isinstance(twin, ast.If):
            return None
        if twin.orelse:
            twin.orelse = []
            desc = "Drop else branch of if"
        else:
            twin.body = [ast.Pass()]
            desc = "Replace if body with pass"
        ast.fix_missing_locations(twin)
        return self._render(tree), desc


class SwapArgs(SemanticOperator):
    """Swap the first two positional args of a 2+-arg call.

    Many bugs read "subtract these in the wrong order" or "compare
    expected vs actual the wrong way". This operator detects functions
    with at least 2 positional args and swaps args[0] and args[1].
    """

    name = "swap_args"

    def apply(self, source: str, target_node: ast.AST) -> Optional[tuple[str, str]]:
        if not isinstance(target_node, ast.Call) or len(target_node.args) < 2:
            return None
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        twin = _equivalent_in(other=tree, original=target_node)
        if twin is None or not isinstance(twin, ast.Call) or len(twin.args) < 2:
            return None
        twin.args[0], twin.args[1] = twin.args[1], twin.args[0]
        ast.fix_missing_locations(twin)
        return self._render(tree), "Swap first two positional args"


class EmptyLoopBody(SemanticOperator):
    """Replace a ``for`` / ``while`` loop body with ``pass``.

    Forces the agent's tests to depend on observable side-effects of
    the loop. Tests that only check "no exception was raised" survive;
    tests that check accumulated state, returned counts, or written
    output kill it.
    """

    name = "empty_loop_body"

    def apply(self, source: str, target_node: ast.AST) -> Optional[tuple[str, str]]:
        if not isinstance(target_node, (ast.For, ast.While)):
            return None
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        twin = _equivalent_in(other=tree, original=target_node)
        if twin is None or not isinstance(twin, (ast.For, ast.While)):
            return None
        twin.body = [ast.Pass()]
        ast.fix_missing_locations(twin)
        return self._render(tree), f"Empty {type(twin).__name__.lower()} body"


# ---------------------------------------------------------------------------
# Tree-clone helpers
# ---------------------------------------------------------------------------


def _node_signature(node: ast.AST) -> tuple:
    """Stable signature of a node for matching across clones.

    Keys on ``(type, lineno, col_offset, end_lineno, end_col_offset)``.
    Two structurally different ASTs can never share both their types and
    their full position tuple, so this is collision-free for the
    "find me the same node in a freshly parsed tree" use case.
    """
    return (
        type(node).__name__,
        getattr(node, "lineno", -1),
        getattr(node, "col_offset", -1),
        getattr(node, "end_lineno", -1),
        getattr(node, "end_col_offset", -1),
    )


def _equivalent_in(*, other: ast.AST, original: ast.AST) -> Optional[ast.AST]:
    """Find the node in ``other`` that corresponds to ``original``.

    Used by operators that take the user's ``target_node`` (which lives
    in tree A) and need to mutate the freshly parsed tree B that we
    will render. The match is exact on signature; if there are multiple
    matches (rare for distinct AST nodes), returns the first.
    """
    target_sig = _node_signature(original)
    for candidate in ast.walk(other):
        if _node_signature(candidate) == target_sig:
            return candidate
    return None


def _find_target_in_clone(*, tree: ast.AST, target_id: int) -> Optional[ast.AST]:
    """Locate the node with id() == target_id in the given tree.

    A no-op helper used by operators that received the target via
    ``id()`` but want a structural reference.
    """
    for candidate in ast.walk(tree):
        if id(candidate) == target_id:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# Order matters for determinism when multiple operators apply to the
# same node — we emit them in this exact sequence.
_DEFAULT_OPERATORS: tuple[SemanticOperator, ...] = (
    OffByOneInLoopBound(),
    SwapSimilarFunctionCalls(),
    SwapComparisonOperators(),
    SignFlip(),
    BooleanInvert(),
    OffByOneIndexing(),
    ConstantPerturb(),
    RemoveBranch(),
    SwapArgs(),
    EmptyLoopBody(),
)


class MutationEngineRegistry:
    """Registry that applies all semantic operators to a source module.

    The registry is the public entry point this module exposes to
    :mod:`apex.evaluation.mutation_engine`. It walks the AST once, asks
    every operator to ``apply`` at every node, and yields one
    :class:`SemanticMutation` per non-None result.

    Determinism: operators are applied in registration order; AST nodes
    are visited in ``ast.walk`` order; results are post-sorted by
    ``(line, col, name, description)`` so the output is identical
    across runs. Mutations are deduplicated by mutated_source so two
    operators producing the same mutant emit only once.
    """

    def __init__(self, operators: Optional[tuple[SemanticOperator, ...]] = None):
        self.operators: tuple[SemanticOperator, ...] = (
            tuple(operators) if operators is not None else _DEFAULT_OPERATORS
        )

    def operator_names(self) -> list[str]:
        return [op.name for op in self.operators]

    def apply_to_source(self, source: str) -> list[SemanticMutation]:
        """Apply every operator to every eligible node in ``source``.

        Returns deterministically-ordered, deduplicated mutations.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        mutations: list[SemanticMutation] = []
        seen_sources: set[str] = set()
        for node in ast.walk(tree):
            for op in self.operators:
                try:
                    result = op.apply(source, node)
                except Exception as exc:  # pragma: no cover — defensive
                    logger.debug(
                        "Semantic operator %s failed at node %s: %s",
                        op.name,
                        type(node).__name__,
                        exc,
                    )
                    continue
                if result is None:
                    continue
                mutated_source, description = result
                if mutated_source == source:
                    continue  # operator was a no-op — skip
                if mutated_source in seen_sources:
                    continue
                seen_sources.add(mutated_source)
                mutations.append(
                    SemanticMutation(
                        name=op.name,
                        description=description,
                        mutated_source=mutated_source,
                        line=getattr(node, "lineno", 0) or 0,
                        col=(getattr(node, "col_offset", 0) or 0) + 1,
                    )
                )
        mutations.sort(key=lambda m: (m.line, m.col, m.name, m.description))
        return mutations

    def apply_to_function(
        self,
        source: str,
        function_name: str,
    ) -> list[SemanticMutation]:
        """Restrict mutation generation to nodes inside ``function_name``.

        Used when the caller wants to score the agent's tests against
        mutants of a specific function (e.g. the function the gold patch
        modified) rather than the whole module.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        target_func: Optional[ast.AST] = None
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ):
                target_func = node
                break
        if target_func is None:
            return []
        mutations: list[SemanticMutation] = []
        seen_sources: set[str] = set()
        # Walk only nodes inside the target function.
        for node in ast.walk(target_func):
            for op in self.operators:
                try:
                    result = op.apply(source, node)
                except Exception as exc:  # pragma: no cover — defensive
                    logger.debug(
                        "Semantic operator %s failed at node %s: %s",
                        op.name,
                        type(node).__name__,
                        exc,
                    )
                    continue
                if result is None:
                    continue
                mutated_source, description = result
                if mutated_source == source:
                    continue
                if mutated_source in seen_sources:
                    continue
                seen_sources.add(mutated_source)
                mutations.append(
                    SemanticMutation(
                        name=op.name,
                        description=description,
                        mutated_source=mutated_source,
                        line=getattr(node, "lineno", 0) or 0,
                        col=(getattr(node, "col_offset", 0) or 0) + 1,
                    )
                )
        mutations.sort(key=lambda m: (m.line, m.col, m.name, m.description))
        return mutations


# Module-level default registry — callers usually want this.
DEFAULT_REGISTRY = MutationEngineRegistry()


__all__ = [
    "MUTATION_KIND",
    "SemanticMutation",
    "SemanticOperator",
    "OffByOneInLoopBound",
    "SwapSimilarFunctionCalls",
    "SwapComparisonOperators",
    "SignFlip",
    "BooleanInvert",
    "OffByOneIndexing",
    "ConstantPerturb",
    "RemoveBranch",
    "SwapArgs",
    "EmptyLoopBody",
    "MutationEngineRegistry",
    "DEFAULT_REGISTRY",
]
