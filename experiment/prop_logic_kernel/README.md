# PropLogicKernel

A small kernel for propositional logic, written in **Lean 4**. It supports goal-directed natural deduction in both intuitionistic and classical logic, with an interactive REPL and tooling for automated puzzle generation and verification.

## Overview

The kernel models propositional proofs as a **stack of goals**. Each goal is a pair ⟨Γ, φ⟩ where Γ is a context of indexed hypotheses and φ is the target proposition. Tactics define transitions on this stack; a proof is complete when the stack is empty.

### Propositions

```
P ::= ⊥ | <var> | P ∧ P | P ∨ P | P → P
```

### Tactics

| Tactic | Effect |
|---|---|
| `intro` | `⊢ A → B` → introduce `A` as a hypothesis, new goal `⊢ B` |
| `exact n` | Close goal if hypothesis `n` matches it |
| `apply n` | `hyp[n] = A → B`, `⊢ B` → new goal `⊢ A` |
| `constructor` | `⊢ A ∧ B` → two subgoals `⊢ A` and `⊢ B` |
| `left` / `right` | `⊢ A ∨ B` → choose a side |
| `cases n` | Split `∨`/`∧` hypothesis, or close goal from `⊥` |
| `lem P` | Introduce `¬P ∨ P` (classical logic) |

## Lean Source Layout

```
PropLogicKernel/
  Kernel.lean     — core types and tactic semantics
  Auto.lean       — iterative-deepening proof search
  Parser.lean     — proposition and tactic parser
  Printer.lean    — precedence-aware pretty printer
  Serialize.lean  — serialization helpers
REPL/
  REPL.lean       — REPL protocol
  Command.lean    — command dispatch
Main.lean         — entry point
```

## Build

Requires [Lean 4 / elan](https://github.com/leanprover/elan). Toolchain version is pinned in `lean-toolchain`.

```bash
lake build
```

### Run the kernel REPL directly

```bash
./.lake/build/bin/Main-lean
```

Then type propositions and tactics interactively:

```
> new A ∧ B → B ∧ A
> intro
> cases 0
> constructor
> exact 2
> exact 1
```

## Web UI

An interactive browser-based proof REPL lives in `claude/web_repl`. It lets you type propositions, apply tactics step by step, undo moves, and run an auto-solver — all without the Lean binary.

**Prerequisites:** Node.js

```bash
# 1. Build the TypeScript kernel library
cd claude/ts_prop_logic_kernel
npm install
npm run build

# 2. Start the dev server
cd ../web_repl
npm install
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) in your browser.

## Puzzle Generation and Checking

Python tooling for generating datasets of propositional logic puzzles and verifying them against the built kernel.

**Prerequisites:** [uv](https://docs.astral.sh/uv/), and `lake build` completed first.

```bash
cd claude
uv sync
```

### Generate puzzles

```bash
uv run python examples/generate_puzzles.py \
  --min-vars 4 --max-vars 6 \
  --min-depth 4 --max-depth 8 \
  --examples-per-config 50
```

Output goes to `output/prop_logic_puzzle/`, organized by `num_vars` and `depth`. Run with `--help` for all options.

### Verify puzzles

```bash
uv run python examples/check_puzzles.py
```

Runs every puzzle in `output/prop_logic_puzzle/` through the kernel binary and writes a `result.csv` with pass/fail per file.

### Upload to HuggingFace

```bash
HF_USER=your-username uv run python examples/upload_huggingface.py
```

---

*The Lean 4 source (`PropLogicKernel/`, `REPL/`, `Main.lean`) is written by the author. The Python tooling (`claude/py_prop_logic_kernel/`, `claude/examples/`, `claude/main.py`), TypeScript port (`claude/ts_prop_logic_kernel/`), web UI (`claude/web_repl/`), and this README were generated with AI assistance (Claude).*
