from __future__ import annotations

import hashlib
import random
import sys
from dataclasses import dataclass

from .puzzle import Puzzle  # package usage

# ---------------------------------------------------------------------------
# Proposition AST
# ---------------------------------------------------------------------------

class Prop:
    def precedence(self) -> int:
        raise NotImplementedError

    def render(self, parent_prec: int = 10) -> str:
        raise NotImplementedError

    def atoms(self) -> frozenset[str]:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Atom(Prop):
    name: str

    def precedence(self) -> int: return 0
    def render(self, parent_prec: int = 10) -> str: return self.name
    def atoms(self) -> frozenset[str]: return frozenset({self.name})
    def size(self) -> int: return 1


@dataclass(frozen=True, slots=True)
class Bot(Prop):
    def precedence(self) -> int: return 0
    def render(self, parent_prec: int = 10) -> str: return "⊥"
    def atoms(self) -> frozenset[str]: return frozenset()
    def size(self) -> int: return 1


@dataclass(frozen=True, slots=True)
class And(Prop):
    left: Prop
    right: Prop

    def precedence(self) -> int: return 1
    def render(self, parent_prec: int = 10) -> str:
        s = f"{self.left.render(self.precedence())} ∧ {self.right.render(self.precedence())}"
        return f"({s})" if self.precedence() >= parent_prec else s
    def atoms(self) -> frozenset[str]: return self.left.atoms() | self.right.atoms()
    def size(self) -> int: return 1 + self.left.size() + self.right.size()


@dataclass(frozen=True, slots=True)
class Or(Prop):
    left: Prop
    right: Prop

    def precedence(self) -> int: return 2
    def render(self, parent_prec: int = 10) -> str:
        s = f"{self.left.render(self.precedence())} ∨ {self.right.render(self.precedence())}"
        return f"({s})" if self.precedence() >= parent_prec else s
    def atoms(self) -> frozenset[str]: return self.left.atoms() | self.right.atoms()
    def size(self) -> int: return 1 + self.left.size() + self.right.size()


@dataclass(frozen=True, slots=True)
class Imp(Prop):
    left: Prop
    right: Prop

    def precedence(self) -> int: return 3
    def render(self, parent_prec: int = 10) -> str:
        s = f"{self.left.render(self.precedence())} → {self.right.render(self.precedence())}"
        return f"({s})" if self.precedence() >= parent_prec else s
    def atoms(self) -> frozenset[str]: return self.left.atoms() | self.right.atoms()
    def size(self) -> int: return 1 + self.left.size() + self.right.size()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerateSettings:
    """
    - `num_vars`: count of atomic names.
    - `depth`: recursion cap for goals and assumptions.
    - `max_attempts`: resample budget when proof synthesis fails.
    - `allow_higher_order`: allow (A→B)→C style assumptions and goals.
    - `allow_or_elim`: allow Or hypotheses in assumptions (requires cases mid-proof).
    """
    num_vars: int = 4
    depth: int = 6
    seed: int | None = None
    max_attempts: int = 96
    allow_higher_order: bool = True
    allow_or_elim: bool = True


# ---------------------------------------------------------------------------
# Variable pool
# ---------------------------------------------------------------------------

def _vars(num_vars: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    for i in range(max(1, num_vars)):
        out.append(chr(ord("A") + i) if i < 26 else f"Z{i - 26}")
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Proposition generation
# ---------------------------------------------------------------------------

def _gen_prop(
    rng: random.Random,
    atoms: list[str],
    depth: int,
    *,
    allow_imp: bool = True,
    allow_or: bool = True,
    allow_and: bool = True,
    leaf_p: float = 0.35,
    imp_w: float = 1.0,
    and_w: float = 1.0,
    or_w: float = 1.0,
) -> Prop:
    """Generic recursive proposition generator."""
    if depth <= 0 or rng.random() < leaf_p:
        return Atom(rng.choice(atoms))

    weights: list[tuple[str, float]] = []
    if allow_and:
        weights.append(("and", and_w))
    if allow_or:
        weights.append(("or", or_w))
    if allow_imp:
        weights.append(("imp", imp_w))

    if not weights:
        return Atom(rng.choice(atoms))

    chosen = rng.choices([n for n, _ in weights], [w for _, w in weights])[0]

    dl = rng.randint(0, depth - 1)
    dr = rng.randint(0, depth - 1)

    kw = dict(allow_imp=allow_imp, allow_or=allow_or, allow_and=allow_and,
              leaf_p=leaf_p, imp_w=imp_w, and_w=and_w, or_w=or_w)

    if chosen == "and":
        return And(_gen_prop(rng, atoms, dl, **kw), _gen_prop(rng, atoms, dr, **kw))
    if chosen == "or":
        return Or(_gen_prop(rng, atoms, dl, **kw), _gen_prop(rng, atoms, dr, **kw))
    # imp: antecedent can be any prop (enables higher-order)
    antecedent = _gen_prop(rng, atoms, dl, **kw)
    consequent = _gen_prop(rng, atoms, dr, **kw)
    return Imp(antecedent, consequent)


def _gen_assumption(rng: random.Random, atoms: list[str], depth: int, *, allow_higher_order: bool, allow_or_elim: bool) -> Prop:
    """
    Generate a single assumption. The family is chosen with explicit weights so
    that harder patterns (nested implications, or-antecedent) are reachable but
    not overwhelming.
    """
    r = rng.random()
    a = rng.choice(atoms)
    b = rng.choice(atoms)
    # avoid trivial A→A when we have choices
    while b == a and len(atoms) > 1:
        b = rng.choice(atoms)
    c = rng.choice(atoms)

    # Simple families (always available)
    if r < 0.20:
        return Atom(a)
    if r < 0.38:
        return Imp(Atom(a), Atom(b))
    if r < 0.52:
        return And(Atom(a), Atom(b))

    # Or hypothesis: requires or-elim to use (harder)
    if allow_or_elim and r < 0.63:
        return Or(Atom(a), Atom(b))

    # Higher-order: (A→B)→C  — requires applying the hypothesis to a proof of A
    if allow_higher_order and r < 0.75:
        return Imp(Imp(Atom(a), Atom(b)), Atom(c))

    # Implication chain: A→B∧C
    if r < 0.84:
        return Imp(Atom(a), And(Atom(b), Atom(c)))

    # (A∨B)→C : or-elimination as an implication
    if allow_or_elim and r < 0.92:
        return Imp(Or(Atom(a), Atom(b)), Atom(c))

    # Deeper: generate a random mid-depth prop
    return _gen_prop(rng, atoms, min(depth, 3), leaf_p=0.4)


def _gen_goal(rng: random.Random, atoms: list[str], depth: int, settings: GenerateSettings) -> Prop:
    """Generate the goal proposition — can be arbitrarily nested."""
    leaf_p = rng.uniform(0.15, 0.45)
    imp_w = rng.betavariate(1.5, 2.0) if settings.allow_higher_order else rng.betavariate(1.2, 2.5)
    and_w = rng.betavariate(2.0, 2.0)
    or_w  = rng.betavariate(2.0, 2.0)
    return _gen_prop(
        rng, atoms, depth,
        allow_imp=True,
        allow_or=settings.allow_or_elim,
        allow_and=True,
        leaf_p=leaf_p,
        imp_w=imp_w,
        and_w=and_w,
        or_w=or_w,
    )


# ---------------------------------------------------------------------------
# Proof search — full backtracking DFS
# ---------------------------------------------------------------------------
#
# State: (goal, hyp) where hyp is a dict[int, Prop].
# We emit tactics as we go; on backtrack we discard emitted tactics from
# that branch.  A simple list-of-lists approach avoids copying.
#
# The kernel's tactic semantics (from the Lean source):
#   intro       : Imp(A,B) goal → add hyp[varCount]=A, new goal B; varCount++
#   cases i     : hyp[i]=Or(A,B) → two subgoals, each with a new hyp
#               : hyp[i]=And(A,B) → one subgoal with two new hyps; varCount+=2
#               : hyp[i]=Bot → close goal
#   constructor : And(A,B) goal → two subgoals A, B (left before right)
#   left        : Or(A,B) goal → new goal A
#   right       : Or(A,B) goal → new goal B
#   apply i     : hyp[i]=Imp(A,B), goal==B → new goal A
#   exact i     : hyp[i]==goal → close
#
# We do NOT emit `cases` for And hypotheses eagerly here (unlike the old
# code) — instead we do it lazily inside `intro` handling, which matches
# what the old _eliminate_all_and did and keeps the kernel happy.

_PROOF_FUEL = 1024

# We represent the proof trace as a flat list[str] with checkpoints (int
# indices into the list). Backtracking = truncate list to checkpoint.

class _ProofState:
    __slots__ = ("tactics", "var_count")

    def __init__(self, var_count: int) -> None:
        self.tactics: list[str] = []
        self.var_count = var_count

    def checkpoint(self) -> tuple[int, int]:
        return (len(self.tactics), self.var_count)

    def restore(self, cp: tuple[int, int]) -> None:
        tlen, vc = cp
        del self.tactics[tlen:]
        self.var_count = vc

    def emit(self, t: str) -> None:
        self.tactics.append(t)

    def alloc(self) -> int:
        i = self.var_count
        self.var_count += 1
        return i


def _expand_and_hyps(hyp: dict[int, Prop], state: _ProofState) -> dict[int, Prop]:
    """
    Eagerly split all And hypotheses, returning a new hyp dict.
    Emits 'cases' tactics into state.  Mirrors _eliminate_all_and from
    the old code but returns a new dict rather than mutating.
    """
    hyp = dict(hyp)
    already: set[int] = set()
    for _ in range(512):
        found = next((i for i, p in hyp.items() if i not in already and isinstance(p, And)), None)
        if found is None:
            return hyp
        already.add(found)
        p = hyp[found]
        assert isinstance(p, And)
        state.emit(f"cases {found}")
        ia = state.alloc()
        ib = state.alloc()
        hyp[ia] = p.left
        hyp[ib] = p.right
    raise RuntimeError("_expand_and_hyps: exceeded iteration cap")


def _prove(
    goal: Prop,
    hyp: dict[int, Prop],
    state: _ProofState,
    fuel: int,
    rng: random.Random,
    split_used: frozenset[int] = frozenset(),
) -> bool:
    """
    Recursive backtracking proof search.
    `split_used` tracks which Or-hypothesis indices have already been case-split
    on the current path, preventing infinite Or-elim loops.
    Returns True iff a proof was found and tactics have been appended to state.
    On failure, leaves state unchanged (restores checkpoint).
    """
    if fuel <= 0:
        return False

    # --- exact ---
    for i, p in hyp.items():
        if p == goal:
            state.emit(f"exact {i}")
            return True

    # --- Bot hypothesis (ex falso) ---
    for i, p in hyp.items():
        if isinstance(p, Bot):
            state.emit(f"cases {i}")
            return True

    # --- goal-directed dispatch ---

    if isinstance(goal, And):
        cp = state.checkpoint()
        state.emit("constructor")
        if (_prove(goal.left,  hyp, state, fuel - 1, rng, split_used) and
            _prove(goal.right, hyp, state, fuel - 1, rng, split_used)):
            return True
        state.restore(cp)
        return False

    if isinstance(goal, Or):
        # Try left, then right
        for side, sub in (("left", goal.left), ("right", goal.right)):
            cp = state.checkpoint()
            state.emit(side)
            if _prove(sub, hyp, state, fuel - 1, rng, split_used):
                return True
            state.restore(cp)
        # Or-elim: case-split a hypothesis we haven't split yet on this path
        for i, p in list(hyp.items()):
            if not isinstance(p, Or) or i in split_used:
                continue
            cp = state.checkpoint()
            state.emit(f"cases {i}")
            ia = state.alloc()
            ib = state.alloc()
            hyp1 = {**hyp, ia: p.left}
            hyp2 = {**hyp, ib: p.right}
            new_used = split_used | {i}
            if (_prove(goal, hyp1, state, fuel - 1, rng, new_used) and
                _prove(goal, hyp2, state, fuel - 1, rng, new_used)):
                return True
            state.restore(cp)
        return False

    if isinstance(goal, Imp):
        cp = state.checkpoint()
        state.emit("intro")
        idx = state.alloc()
        new_hyp = dict(hyp)
        new_hyp[idx] = goal.left
        # If the introduced hypothesis is an And, split it immediately (kernel compat)
        if isinstance(goal.left, And):
            state.emit(f"cases {idx}")
            ia = state.alloc()
            ib = state.alloc()
            new_hyp[ia] = goal.left.left
            new_hyp[ib] = goal.left.right
        new_hyp = _expand_and_hyps(new_hyp, state)
        if _prove(goal.right, new_hyp, state, fuel - 1, rng, split_used):
            return True
        state.restore(cp)
        return False

    if isinstance(goal, Bot):
        for i, p in list(hyp.items()):
            if isinstance(p, Imp) and isinstance(p.right, Bot):
                cp = state.checkpoint()
                state.emit(f"apply {i}")
                if _prove(p.left, hyp, state, fuel - 1, rng, split_used):
                    return True
                state.restore(cp)
        return False

    if isinstance(goal, Atom):
        return _prove_atom(goal, hyp, state, fuel, rng, split_used, frozenset())

    return False


def _prove_atom(
    goal: Atom,
    hyp: dict[int, Prop],
    state: _ProofState,
    fuel: int,
    rng: random.Random,
    split_used: frozenset[int],
    chain_visited: frozenset[str],
) -> bool:
    """Prove an atom goal with cycle detection on apply chains."""
    if fuel <= 0:
        return False

    # exact
    for i, p in hyp.items():
        if p == goal:
            state.emit(f"exact {i}")
            return True

    gname = goal.name
    if gname in chain_visited:
        return False  # cycle on this apply chain

    new_chain = chain_visited | {gname}

    # apply: h: X→goal
    for i, p in list(hyp.items()):
        if not (isinstance(p, Imp) and p.right == goal):
            continue
        ante = p.left
        cp = state.checkpoint()
        state.emit(f"apply {i}")
        if isinstance(ante, Atom):
            ok = _prove_atom(ante, hyp, state, fuel - 1, rng, split_used, new_chain)
        else:
            ok = _prove(ante, hyp, state, fuel - 1, rng, split_used)
        if ok:
            return True
        state.restore(cp)

    # Or-elim
    for i, p in list(hyp.items()):
        if not isinstance(p, Or) or i in split_used:
            continue
        cp = state.checkpoint()
        state.emit(f"cases {i}")
        ia = state.alloc()
        ib = state.alloc()
        hyp1 = {**hyp, ia: p.left}
        hyp2 = {**hyp, ib: p.right}
        new_used = split_used | {i}
        if (_prove(goal, hyp1, state, fuel - 1, rng, new_used) and
            _prove(goal, hyp2, state, fuel - 1, rng, new_used)):
            return True
        state.restore(cp)

    return False


_SAVED_LIMIT = sys.getrecursionlimit()

def _synthesize_proof(assumptions: list[Prop], goal: Prop, rng: random.Random) -> list[str] | None:
    state = _ProofState(var_count=0)
    hyp: dict[int, Prop] = {}
    for a in assumptions:
        state.emit("intro")
        idx = state.alloc()
        hyp[idx] = a
        if isinstance(a, And):
            state.emit(f"cases {idx}")
            ia = state.alloc()
            ib = state.alloc()
            hyp[ia] = a.left
            hyp[ib] = a.right
    hyp = _expand_and_hyps(hyp, state)
    sys.setrecursionlimit(4000)
    try:
        ok = _prove(goal, hyp, state, _PROOF_FUEL, rng, frozenset())
    except RecursionError:
        ok = False
    finally:
        sys.setrecursionlimit(_SAVED_LIMIT)
    return state.tactics if ok else None


# ---------------------------------------------------------------------------
# Statement assembly & fingerprinting
# ---------------------------------------------------------------------------

def _build_statement(assumptions: list[Prop], goal: Prop) -> Prop:
    p: Prop = goal
    for a in reversed(assumptions):
        p = Imp(a, p)
    return p


def _statement_fingerprint(stmt: str) -> str:
    return hashlib.sha256(stmt.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Single-attempt generation
# ---------------------------------------------------------------------------

def _try_generate_once(rng: random.Random, settings: GenerateSettings):
    atoms = _vars(settings.num_vars, rng)
    cap = min(len(atoms), max(2, settings.depth + 1))
    num_assumptions = rng.randint(1, cap)

    assumptions = [
        _gen_assumption(
            rng, atoms, settings.depth,
            allow_higher_order=settings.allow_higher_order,
            allow_or_elim=settings.allow_or_elim,
        )
        for _ in range(num_assumptions)
    ]
    goal = _gen_goal(rng, atoms, settings.depth, settings)

    # Sanity: all atoms referenced must be in our pool
    need = goal.atoms()
    for a in assumptions:
        need |= a.atoms()
    if not need.issubset(frozenset(atoms)):
        return None, "atom mismatch"

    # Reject trivially easy puzzles: goal must not already be in assumptions
    if any(a == goal for a in assumptions):
        return None, "goal is trivial (in assumptions)"

    stmt_prop = _build_statement(assumptions, goal)
    statement = stmt_prop.render()
    proof = _synthesize_proof(assumptions, goal, rng)
    if proof is None:
        return None, "proof synthesis failed"

    puzzle = Puzzle(
        statement=statement,
        proof=proof,
        settings={
            "num_vars": settings.num_vars,
            "depth": settings.depth,
            "seed": settings.seed,
            "statement_sha256_16": _statement_fingerprint(statement),
            "goal_kind": type(goal).__name__,
            "num_assumptions": num_assumptions,
            "proof_length": len(proof),
            "assumption_kinds": [type(a).__name__ for a in assumptions],
        },
    )
    return puzzle, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_puzzle(settings: GenerateSettings = GenerateSettings()) -> "Puzzle":
    base_seed = settings.seed if settings.seed is not None else random.randrange(1 << 30)
    last_err: str | None = None
    attempts_cap = min(max(1, settings.max_attempts), 4096)
    for attempt in range(attempts_cap):
        rng = random.Random(base_seed + attempt * 100_003)
        puzzle, err = _try_generate_once(rng, settings)
        if puzzle is None:
            last_err = err
            continue
        return puzzle
    msg = f"generate_puzzle gave up after {attempts_cap} attempts"
    if last_err:
        msg += f" ({last_err})"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    configs = [
        ("easy",   GenerateSettings(num_vars=3, depth=3, seed=1, allow_higher_order=False, allow_or_elim=False)),
        ("medium", GenerateSettings(num_vars=4, depth=5, seed=2)),
        ("hard",   GenerateSettings(num_vars=5, depth=7, seed=3)),
        ("expert", GenerateSettings(num_vars=5, depth=8, seed=4, allow_higher_order=True, allow_or_elim=True)),
    ]

    for label, cfg in configs:
        p = generate_puzzle(cfg)
        print(f"\n{'='*60}")
        print(f"[{label}]")
        print(f"  Statement : {p.statement}")
        print(f"  Goal kind : {p.settings['goal_kind']}")
        print(f"  Assumps   : {p.settings['assumption_kinds']}")
        print(f"  Proof len : {p.settings['proof_length']}")
        print(f"  Proof     : {p.proof}")