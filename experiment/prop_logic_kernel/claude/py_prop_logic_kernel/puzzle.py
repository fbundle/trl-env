from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .repl import Repl, ReplError

_ALL_DONE_MARKER = "all goals accomplished!"

_STATUS_RE = re.compile(
    r"^\s*(?:--\s*)?new_count\s+(\d+)\s+sorry_count\s+(\d+)\s+var_count\s+(\d+)\s+goals_remaining\s+(\d+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ALL_DONE_RE = re.compile(r"^\s*(?:--\s*)?all\s+goals\s+accomplished!\s*$", re.IGNORECASE | re.MULTILINE)
_INDEX_TACTIC_RE = re.compile(r"^\s*\b(exact|apply|cases|bridge|refine)\b\s+(\d+)\s*$", re.IGNORECASE)



@dataclass(frozen=True, slots=True)
class Puzzle:
    """
    A self-contained logic puzzle: a proposition statement plus a proof script.

    - `statement` uses the same surface syntax as the kernel (`⊥`, atoms, `∧`, `∨`, `→`, parens).
    - `proof` is a sequence of REPL commands (tactics), *excluding* the initial `new <statement>`.
    """

    statement: str
    proof: Sequence[str]
    settings: Mapping[str, object] | None = None

    def check(self, kernel_path: str | Path = ".lake/build/bin/Main-lean") -> bool | None:
        """
        Run this puzzle against the kernel binary using stdin (no REPL wrapper).

        Returns:
        - True: kernel exits with code 0 and stderr contains `all goals accomplished!`
        - False: kernel exits with nonzero code or does not report the success marker
        - None: failed to run (missing binary, timeout, or OS error)
        """
        exe = Path(kernel_path)
        if not exe.exists():
            return None

        lines = [f"new {self.statement}", *[line.rstrip("\n") for line in self.proof]]
        stdin_payload = "\n".join(lines) + "\n"

        try:
            proc = subprocess.run(
                [str(exe)],
                input=stdin_payload.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=5.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        err_text = proc.stderr.decode("utf-8", errors="replace")
        return (proc.returncode == 0) and (_ALL_DONE_MARKER in err_text)


def load_puzzle_from_file(path: str | Path) -> Puzzle:
    """
    Load a single puzzle from a `stdin.txt`-style file:

    - first non-empty line must be `new <statement>`
    - subsequent non-empty lines are tactics
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    raw_lines = [ln.rstrip("\n") for ln in text.splitlines()]

    lines = [ln.strip() for ln in raw_lines if ln.strip() != ""]
    if not lines:
        raise ValueError(f"empty puzzle file: {p}")

    first = lines[0]
    if not first.lower().startswith("new "):
        raise ValueError(f"expected first line `new <statement>` in {p}, got: {first!r}")

    statement = first[4:].strip()
    proof = lines[1:]
    return Puzzle(statement=statement, proof=proof, settings={"source_path": str(p)})


@dataclass(frozen=True)
class Step:
    out: str
    err: str
    new_count: int | None
    sorry_count: int | None
    var_count: int | None
    goals_remaining: int | None
    all_goals_accomplished: bool
    errors: tuple[str, ...]


class Client:
    """
    Persistent client for the proposition-logic kernel.

    Unlike `Puzzle.check()` (one subprocess per puzzle), this keeps a single kernel
    process alive and can `check()` multiple puzzles, shifting local tactic indices
    using the Lean-reported `var_count`.
    """

    def __init__(
        self,
        *,
        cwd: str | Path | None = None,
        exe: str | Path | None = None,
        prompt: str = "> ",
        read_timeout_s: float = 5.0,
    ) -> None:
        self._cwd = Path(cwd) if cwd is not None else Path(__file__).resolve().parents[1]
        self._exe = Path(exe) if exe is not None else (self._cwd / ".lake" / "build" / "bin" / "Main-lean")
        self._prompt = prompt
        self._read_timeout_s = float(read_timeout_s)

        self._repl: Repl | None = None
        self._last: Step | None = None
        self._graceful_exit_code: int | None = None

    def start(self) -> Step:
        if self._repl is not None:
            raise RuntimeError("client already started")
        if not self._exe.exists():
            raise FileNotFoundError(f"executable not found at {self._exe!s}; run `lake build` first")

        self._graceful_exit_code = None
        self._repl = Repl([str(self._exe)], cwd=str(self._cwd), prompt=self._prompt, read_timeout_s=self._read_timeout_s)
        repl_step = self._repl.start()
        self._last = self._to_step(repl_step.out, repl_step.err)
        return self._last

    def finish(self, *, timeout_s: float = 2.0) -> int | None:
        if self._repl is None:
            raise RuntimeError("client not started; call start() first")
        code = self._repl.finish(timeout_s=float(timeout_s))
        self._graceful_exit_code = code
        self._repl = None
        return code

    def graceful_exit_code(self) -> int | None:
        return self._graceful_exit_code

    def last(self) -> Step:
        if self._last is None:
            raise RuntimeError("client has no steps yet; call start() first")
        return self._last

    def send(self, line: str) -> Step:
        """
        Send a raw line to the kernel (can be `new ...` or a tactic).
        """
        if self._repl is None:
            raise RuntimeError("client not started; call start() first")
        try:
            repl_step = self._repl.step(line.rstrip("\n"))
        except ReplError as e:
            raise RuntimeError(str(e)) from e
        self._last = self._to_step(repl_step.out, repl_step.err)
        return self._last

    def check(self, puzzle: Puzzle) -> bool:
        """
        Check a puzzle in the *current* kernel session.

        Raises RuntimeError on kernel errors. Returns True when the goal was solved.
        """
        if self._repl is None:
            self.start()

        step_after_new = self.send(f"new {puzzle.statement}")
        if step_after_new.errors:
            raise RuntimeError(f"kernel error on `new`: {step_after_new.errors[-1]}")

        base_var = step_after_new.var_count
        goals_after_new = step_after_new.goals_remaining
        if base_var is None or goals_after_new is None:
            raise RuntimeError(f"kernel did not report status after `new`: {step_after_new.err}")

        for raw in puzzle.proof:
            s = str(raw).rstrip("\n")
            m = _INDEX_TACTIC_RE.match(s)
            if m:
                op, n_str = m.group(1).lower(), m.group(2)
                s = f"{op} {base_var + int(n_str)}"
            step = self.send(s)
            if step.errors:
                raise RuntimeError(f"kernel error on `{s}`: {step.errors[-1]}")

        # Heuristic: after proving, goal stack should go back down by 1.
        step_final = self.last()
        if step_final.goals_remaining is None:
            raise RuntimeError(f"kernel did not report status after proof: {step_final.err}")
        if step_final.goals_remaining != goals_after_new - 1:
            raise RuntimeError(
                f"puzzle did not discharge exactly one goal (after_new={goals_after_new}, after_proof={step_final.goals_remaining})"
            )
        return True

    def __enter__(self) -> "Client":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.finish()
        except Exception:
            pass

    def _to_step(self, out: str, err: str) -> Step:
        all_goals_accomplished = _ALL_DONE_RE.search(err) is not None
        m = _STATUS_RE.search(err)
        if m:
            new_count: int | None = int(m.group(1))
            sorry_count: int | None = int(m.group(2))
            var_count: int | None = int(m.group(3))
            goals: int | None = int(m.group(4))
        else:
            new_count = None
            sorry_count = None
            var_count = None
            goals = None

        # Treat any stderr line that isn't status or all-done as an error.
        errors: list[str] = []
        for line in [ln.strip() for ln in err.splitlines() if ln.strip()]:
            if _STATUS_RE.fullmatch(line) is not None:
                continue
            if _ALL_DONE_RE.fullmatch(line) is not None:
                continue
            errors.append(line)

        return Step(
            out=out,
            err=err,
            new_count=new_count,
            sorry_count=sorry_count,
            var_count=var_count,
            goals_remaining=goals,
            all_goals_accomplished=all_goals_accomplished,
            errors=tuple(errors),
        )
