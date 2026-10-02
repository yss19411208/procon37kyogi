"""進捗表示。外部ライブラリを使わずに1行更新の進捗バーを出します。"""

from __future__ import annotations

import shutil
import sys
import time


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}時間{minutes:02d}分{secs:02d}秒"
    if minutes:
        return f"{minutes}分{secs:02d}秒"
    return f"{secs}秒"


class Progress:
    """使い方:

        with Progress("問題を生成中", total=100) as bar:
            for i in range(100):
                ...
                bar.step(note="...")
    """

    def __init__(self, label: str, total: int, min_interval: float = 0.1) -> None:
        self.label = label
        self.total = max(1, total)
        self.done = 0
        self.started_at = time.monotonic()
        self.last_draw = 0.0
        self.min_interval = min_interval
        self.note = ""
        self._fill = "█"
        self._empty = "░"

    def __enter__(self) -> "Progress":
        self._draw(force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.done = self.total
            self._draw(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def step(self, count: int = 1, note: str = "") -> None:
        self.done = min(self.total, self.done + count)
        if note:
            self.note = note
        self._draw()

    def set(self, done: int, note: str = "") -> None:
        self.done = min(self.total, max(0, done))
        if note:
            self.note = note
        self._draw()

    def _draw(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_draw < self.min_interval:
            return
        self.last_draw = now

        ratio = self.done / self.total
        elapsed = now - self.started_at
        if self.done > 0:
            eta = elapsed * (self.total - self.done) / self.done
            eta_text = f"残り{format_duration(eta)}"
        else:
            eta_text = "残り--"

        width = shutil.get_terminal_size((100, 20)).columns
        bar_width = 24
        filled = int(bar_width * ratio)
        bar = self._fill * filled + self._empty * (bar_width - filled)
        line = (
            f"{self.label} [{bar}] {ratio * 100:5.1f}% "
            f"({self.done}/{self.total}) 経過{format_duration(elapsed)} {eta_text}"
        )
        if self.note:
            line += f" | {self.note}"
        if len(line) > width - 1:
            line = line[: width - 2] + "…"
        try:
            sys.stdout.write("\r" + line.ljust(width - 1))
            sys.stdout.flush()
        except UnicodeEncodeError:
            # Windows の cp932 コンソールなどで記号が出せない場合の逃げ道。
            # まず記号だけ ASCII に置き換える（日本語は cp932 でも出せる）。
            self._fill, self._empty = "#", "-"
            plain = line.replace("█", "#").replace("░", "-").replace("…", "...")
            try:
                sys.stdout.write("\r" + plain.ljust(width - 1))
                sys.stdout.flush()
            except UnicodeEncodeError:
                ascii_only = plain.encode("ascii", "replace").decode("ascii")
                sys.stdout.write("\r" + ascii_only.ljust(width - 1))
                sys.stdout.flush()
