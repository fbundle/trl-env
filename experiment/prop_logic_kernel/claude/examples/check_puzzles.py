#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path
import argparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

# Allow running as: `python examples/check_puzzles.py`
sys.path.insert(0, str(_repo_root()))

from py_prop_logic_kernel.puzzle import Client, load_puzzle_from_file  # noqa: E402

def main() -> None:
    root = _repo_root()
    puzzle_dir = root / "output" / "prop_logic_puzzle"
    
    parser = argparse.ArgumentParser(description="Check proposition-logic puzzles.")
    parser.add_argument("--dir", type=str, default=str(puzzle_dir), help="Directory containing puzzles.")
    parser.add_argument("--output", type=str, default=str(puzzle_dir / "result.csv"), help="Output CSV file.")
    parser.add_argument("--exe", type=str, default=None, help="Path to kernel executable.")
    args = parser.parse_args()

    puzzle_dir = Path(args.dir)
    result_csv = Path(args.output)
    
    # Ensure output directory exists
    result_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results to skip them
    results = {}
    if result_csv.exists():
        with open(result_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    if len(row) >= 2:
                        results[row[0]] = row[1]

    # Find all .txt files recursively
    all_files = sorted([
        p for p in puzzle_dir.glob("**/*.txt") 
        if p.name != "stdin.txt" and p.suffix == ".txt"
    ])
    
    # Filter out already checked files
    puzzle_files = []
    for p_path in all_files:
        try:
            rel_path = str(p_path.relative_to(root))
        except ValueError:
            rel_path = str(p_path)
        
        if rel_path not in results:
            puzzle_files.append((p_path, rel_path))

    if not puzzle_files:
        if all_files:
            print(f"All {len(all_files)} puzzles already checked in {result_csv}")
        else:
            print(f"No puzzle files found in {puzzle_dir}")
        return

    print(f"Found {len(all_files)} total puzzles. Checking {len(puzzle_files)} new ones...")

    client = Client(exe=args.exe)
    client.start()
    
    try:
        # Use tqdm for progress tracking
        for p_path, rel_path in tqdm(puzzle_files, desc="Checking"):
            try:
                puzzle = load_puzzle_from_file(p_path)
                correct = client.check(puzzle)
            except Exception:
                correct = False
                try:
                    client.finish()
                except Exception:
                    pass
                try:
                    client = Client(exe=args.exe)
                    client.start()
                except Exception:
                    pass
            
            results[rel_path] = str(correct)
    finally:
        try:
            client.finish()
        except Exception:
            pass
            
    # Sort results by path before writing
    sorted_items = sorted(results.items())
    
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file path", "puzzle and proof correct"])
        for path, status in sorted_items:
            writer.writerow([path, status])
    
    print(f"Results written to {result_csv}")

if __name__ == "__main__":
    main()
