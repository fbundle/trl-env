from __future__ import annotations

import os
import selectors
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class ReplError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplStep:
    out: str
    err: str
    prompt: str


class Repl:
    """
    Generic Python client for the `REPL.run` protocol used by this repo.
    """

    def __init__(
        self,
        argv: Sequence[str | os.PathLike[str]],
        *,
        cwd: str | os.PathLike[str] | None = None,
        prompt: str = "> ",
        encoding: str = "utf-8",
        read_timeout_s: float = 5.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self._argv = [str(a) for a in argv]
        self._cwd = str(Path(cwd)) if cwd is not None else None
        self._prompt_str = prompt
        self._prompt_bytes = prompt.encode(encoding)
        self._encoding = encoding
        self._read_timeout_s = float(read_timeout_s)
        self._env = env

        self._p: subprocess.Popen[bytes] | None = None
        self._last: ReplStep | None = None
        self._graceful_exit_code: int | None = None

    def graceful_exit_code(self) -> int | None:
        return self._graceful_exit_code

    def start(self) -> ReplStep:
        """
        Start the subprocess (if needed) and return the initial REPL step.
        """
        if self._p is not None:
            raise ReplError("process already started")

        self._graceful_exit_code = None
        self._p = subprocess.Popen(
            self._argv,
            cwd=self._cwd,
            env=self._env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        out, err = self._read_until_prompt()
        self._last = ReplStep(out=out, err=err, prompt=self._prompt_str)
        return self._last

    def step(self, line: str) -> ReplStep:
        """
        Send one input line and return the next REPL step (reads until prompt).
        """
        if self._p is None:
            raise ReplError("process not started; call start() first")
        assert self._p.stdin is not None

        self._p.stdin.write((line.rstrip("\n") + "\n").encode(self._encoding))
        self._p.stdin.flush()

        out, err = self._read_until_prompt()
        self._last = ReplStep(out=out, err=err, prompt=self._prompt_str)
        return self._last

    def finish(self, *, timeout_s: float = 2.0) -> int | None:
        """
        Gracefully finish the REPL by closing stdin (EOF) and waiting.

        Returns graceful exit code when graceful EOF shutdown succeeds.
        Returns None if graceful shutdown timed out and forceful fallback was used.
        """
        if self._p is None:
            raise ReplError("process not started; call start() first")

        if self._p.stdin:
            try:
                self._p.stdin.close()
            except Exception:
                pass

        p = self._p
        try:
            code = p.wait(timeout=float(timeout_s))
            self._graceful_exit_code = int(code)
            return self._graceful_exit_code
        except subprocess.TimeoutExpired:
            # Fallback: forceful shutdown if graceful EOF doesn't exit in time.
            try:
                p.terminate()
                try:
                    p.wait(timeout=1.0)
                    return None
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=1.0)
                    return None
            finally:
                self._p = None
        finally:
            self._p = None

    def __enter__(self) -> "Repl":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()

    def _read_until_prompt(self) -> tuple[str, str]:
        assert self._p is not None and self._p.stdout is not None and self._p.stderr is not None

        out_buf = bytearray()
        err_buf = bytearray()

        sel = selectors.DefaultSelector()
        sel.register(self._p.stdout, selectors.EVENT_READ)
        sel.register(self._p.stderr, selectors.EVENT_READ)

        try:
            while True:
                events = sel.select(timeout=self._read_timeout_s)
                if not events:
                    code = self._p.poll()
                    if code is not None:
                        raise ReplError(f"process terminated (exit_code={code})")
                    continue

                for key, _mask in events:
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if chunk == b"":
                        code = self._p.poll()
                        raise ReplError(f"process terminated (exit_code={code})")

                    if key.fileobj is self._p.stdout:
                        out_buf.extend(chunk)
                    else:
                        err_buf.extend(chunk)

                    if self._prompt_bytes in err_buf:
                        idx = err_buf.index(self._prompt_bytes)
                        prompt_end = idx + len(self._prompt_bytes)
                        captured_err = bytes(err_buf[:idx])
                        del err_buf[:prompt_end]

                        while True:
                            extra = sel.select(timeout=0)
                            if not extra:
                                break
                            for k2, _m2 in extra:
                                if k2.fileobj is not self._p.stdout:
                                    continue
                                more = os.read(k2.fileobj.fileno(), 4096)
                                if more:
                                    out_buf.extend(more)

                        return (
                            out_buf.decode(self._encoding, errors="replace").rstrip("\n"),
                            captured_err.decode(self._encoding, errors="replace").rstrip("\n"),
                        )
        finally:
            for stream in (self._p.stdout, self._p.stderr):
                try:
                    sel.unregister(stream)
                except Exception:
                    pass
            sel.close()

