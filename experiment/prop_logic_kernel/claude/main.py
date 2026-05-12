from __future__ import annotations

import sys

from py_prop_logic_kernel import Client


def main() -> int:
    """
    Python equivalent of `Main.lean`:
    - starts the built Lean binary
    - prints each step's stderr then stdout
    - prints the prompt
    - reads one line from stdin and sends it
    """
    c = Client()
    try:
        step = c.start()
        while True:
            if step.err:
                sys.stderr.write(step.err.rstrip("\n") + "\n")
                sys.stderr.flush()
            if step.out:
                sys.stdout.write(step.out.rstrip("\n") + "\n")
                sys.stdout.flush()

            sys.stderr.write("> ")
            sys.stderr.flush()
            line = sys.stdin.readline()
            if line == "":
                return 0
            step = c.send(line)
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        c.finish()


if __name__ == "__main__":
    raise SystemExit(main())

