#!/usr/bin/env python3

"""Run a Python script once per CSV row, feeding it one column's value.

Usage::

    for_each_in_csv.py <csv_path> <column_name> <python_script> [other args...]

Reads ``<csv_path>`` as a CSV whose first line holds the column names, locates
``<column_name>``, and for every subsequent row invokes::

    <python_script> <value_in_that_column> [other args...]

The script is run with the same Python interpreter. ``[other args...]`` are
passed through verbatim after the per-row value, so they may include flags.

Generic and self-contained: standard library only.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import threading


class SpaceStopWatcher:
    """Background watcher that trips a flag when SPACE is pressed.

    Lets the operator ask for a *clean stop*: the currently running child is
    never interrupted (it finishes normally), and the caller checks
    :attr:`requested` between items to stop before starting the next one.

    Cross-platform and best-effort: it uses ``msvcrt`` on Windows and
    ``termios``/``select`` on POSIX. When stdin is not an interactive terminal
    (e.g. output is redirected), it stays disabled and the batch simply runs to
    completion. Because the child processes never read stdin, the shared console
    lets this parent-side thread capture keystrokes even while a child runs.
    """

    def __init__(self) -> None:
        self._requested = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._restore = None
        self.enabled = False

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    def start(self) -> None:
        if not sys.stdin or not sys.stdin.isatty():
            return
        try:
            import msvcrt  # noqa: F401  (Windows only)

            target = self._run_windows
        except ImportError:
            if not self._setup_posix():
                return
            target = self._run_posix
        self.enabled = True
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def _trigger(self) -> None:
        if not self._requested.is_set():
            self._requested.set()
            print(
                "\n[stop] SPACE pressed — will stop cleanly after the current "
                "item finishes.",
                flush=True,
            )

    def _run_windows(self) -> None:
        import msvcrt
        import time

        while not self._stop.is_set():
            while msvcrt.kbhit():
                if msvcrt.getwch() == " ":
                    self._trigger()
            time.sleep(0.05)

    def _setup_posix(self) -> bool:
        try:
            import termios
            import tty
        except ImportError:
            return False
        self._fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(self._fd)
        except termios.error:
            return False
        tty.setcbreak(self._fd)
        self._restore = lambda: termios.tcsetattr(
            self._fd, termios.TCSADRAIN, old
        )
        return True

    def _run_posix(self) -> None:
        import select

        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready and sys.stdin.read(1) == " ":
                self._trigger()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)
        if self._restore is not None:
            try:
                self._restore()
            except Exception:
                pass
            self._restore = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each CSV row, run: <python_script> <column value> "
            "[other args...]"
        ),
        usage=(
            "%(prog)s <csv_path> <column_name> <python_script> "
            "[other args...]"
        ),
    )
    parser.add_argument("csv_path", help="Path to the CSV file to read.")
    parser.add_argument(
        "column_name",
        help="Name of the column (from the header row) to feed to the script.",
    )
    parser.add_argument(
        "python_script",
        help="Path to the Python script to run once per row.",
    )
    parser.add_argument(
        "other_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to the script after the column value.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # utf-8-sig transparently strips a leading BOM so the first header name
    # matches cleanly.
    try:
        handle = open(args.csv_path, newline="", encoding="utf-8-sig")
    except OSError as error:
        print(f"Cannot open CSV: {error}", file=sys.stderr)
        return 2

    with handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            print("CSV is empty (no header row).", file=sys.stderr)
            return 2

        try:
            column_index = header.index(args.column_name)
        except ValueError:
            print(
                f"Column {args.column_name!r} not found. "
                f"Available columns: {', '.join(header)}",
                file=sys.stderr,
            )
            return 2

        rows = 0
        skipped = 0
        failures = 0
        stopped_early = False

        watcher = SpaceStopWatcher()
        watcher.start()
        if watcher.enabled:
            print(
                "Press SPACE at any time to stop cleanly after the current "
                "item finishes."
            )
        try:
            for line_number, row in enumerate(reader, start=2):
                # Clean stop: honor a SPACE press before starting the next item
                # (the item that was already running has finished by now).
                if watcher.requested:
                    stopped_early = True
                    print(
                        f"[stop] Clean stop requested — not starting line "
                        f"{line_number} or any later rows."
                    )
                    break

                if column_index >= len(row):
                    skipped += 1
                    print(
                        f"[line {line_number}] skipped: row has no value in "
                        f"column {args.column_name!r}",
                        file=sys.stderr,
                    )
                    continue

                value = row[column_index].strip()
                if not value:
                    skipped += 1
                    print(f"[line {line_number}] skipped: empty value")
                    continue

                command = [
                    sys.executable,
                    args.python_script,
                    value,
                    *args.other_args,
                ]
                rows += 1
                print(f"[line {line_number}] running: {' '.join(command)}")

                result = subprocess.run(command)
                if result.returncode != 0:
                    failures += 1
                    print(
                        f"[line {line_number}] script exited with "
                        f"code {result.returncode}",
                        file=sys.stderr,
                    )
        finally:
            watcher.close()

    print()
    print(f"Rows run:  {rows}")
    print(f"Skipped:   {skipped}")
    print(f"Failures:  {failures}")
    if stopped_early:
        print("Stopped early on user request (SPACE).")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
