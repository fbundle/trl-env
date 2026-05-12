from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
import sys
from pathlib import Path

from tqdm import tqdm


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Allow running as: `python examples/generate_puzzles.py`
sys.path.insert(0, str(_repo_root()))

from py_prop_logic_kernel.generate import GenerateSettings, generate_puzzle  # noqa: E402
from py_prop_logic_kernel.puzzle import Puzzle  # noqa: E402


def _write_puzzle_file(path: Path, puzzle: Puzzle) -> None:
    lines = [f"new {puzzle.statement}", *[str(x).rstrip("\n") for x in puzzle.proof]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _puzzle_filename(puzzle: Puzzle) -> str:
    sha = puzzle.settings.get("statement_sha256_16", "") if puzzle.settings else ""
    return f"puzzle_{sha}.txt"


def _count_existing(out_config_dir: Path) -> int:
    return len(list(out_config_dir.glob("puzzle_*.txt")))


def _worker_generate_one_puzzle(args: tuple[str, int, int, int, int, str | None]) -> bool:
    """Generate a single puzzle. Returns True if a new file was written."""
    out_dir_s, num_vars, depth, seed, already, kernel_exe_s = args
    out_config_dir = Path(out_dir_s) / f"example_num_vars{num_vars}_depth{depth}"
    out_config_dir.mkdir(parents=True, exist_ok=True)
    kernel_exe = Path(kernel_exe_s) if kernel_exe_s is not None else None

    puzzle = generate_puzzle(GenerateSettings(num_vars=num_vars, depth=depth, seed=seed))
    out_path = out_config_dir / _puzzle_filename(puzzle)
    if out_path.exists():
        return False
    _write_puzzle_file(out_path, puzzle)

    return True


def main() -> None:
    out_dir = _repo_root() / "output" / "prop_logic_puzzle"
    cwd = _repo_root()

    parser = argparse.ArgumentParser(description="Generate proposition-logic puzzles dataset.")
    parser.add_argument("--min-vars", type=int, default=4)
    parser.add_argument("--max-vars", type=int, default=10)
    parser.add_argument("--min-depth", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--examples-per-config", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Parallel workers (default: all cores).")
    parser.add_argument(
        "--check-kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify each generated file by running the built kernel once on it.",
    )
    parser.add_argument(
        "--kernel-exe",
        type=str,
        default=None,
        help="Path to kernel executable (default: .lake/build/bin/Main-lean).",
    )
    parser.add_argument("--kernel-timeout-s", type=float, default=60.0)
    parser.add_argument("--num-vars", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    args = parser.parse_args()

    min_vars, max_vars = args.min_vars, args.max_vars
    min_depth, max_depth = args.min_depth, args.max_depth
    n = args.examples_per_config
    exe = Path(args.kernel_exe) if args.kernel_exe is not None else (cwd / ".lake" / "build" / "bin" / "Main-lean")
    kernel_exe = exe if args.check_kernel else None
    kernel_exe_s = str(kernel_exe) if kernel_exe is not None else None
    jobs = max(1, int(args.jobs))

    # Build the flat list of (num_vars, depth) configs to cover.
    if args.num_vars is not None or args.depth is not None:
        configs = [(
            args.num_vars if args.num_vars is not None else min_vars,
            args.depth if args.depth is not None else min_depth,
        )]
    else:
        configs = sorted(
            ((nv, d) for d in range(min_depth, max_depth + 1) for nv in range(min_vars, max_vars + 1)),
            key=lambda x: (x[1], x[0]),
        )

    # Expand configs into individual puzzle tasks, skipping already-complete configs.
    work: list[tuple[str, int, int, int, int, str | None]] = []
    for nv, d in configs:
        out_config_dir = out_dir / f"example_num_vars{nv}_depth{d}"
        already = _count_existing(out_config_dir)
        remaining = n - already
        if remaining <= 0:
            continue
        for i in range(remaining):
            seed = nv * 1_000_000 + d * 1_000 + already + i
            work.append((str(out_dir), nv, d, seed, already, kernel_exe_s))

    if not work:
        print("Nothing to do — all configs already have enough puzzles.")
        return

    print(f"Generating {len(work)} puzzles across {len(configs)} configs ({jobs} workers).")

    if jobs == 1:
        for w in tqdm(work, desc="puzzles"):
            _worker_generate_one_puzzle(w)
    else:
        with Pool(processes=jobs) as pool:
            for _ in tqdm(pool.imap_unordered(_worker_generate_one_puzzle, work),
                          total=len(work), desc="puzzles"):
                pass


if __name__ == "__main__":
    main()