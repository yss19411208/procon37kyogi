"""六角形マップの隣接計算。

質問回答集その1 Q1: 偶数行（0行目を含む）が右にずれる形で固定。
質問回答集その1 Q2: マップはトーラスではない。
質問回答集その2 Q47: 方向は 0=左上 1=右上 2=右 3=右下 4=左下 5=左。
"""

from __future__ import annotations

from functools import lru_cache

# 偶数行は右に半セルずれているので、上下の隣は (row-1, col) と (row-1, col+1)。
_EVEN_ROW_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),   # 0 左上
    (-1, 1),   # 1 右上
    (0, 1),    # 2 右
    (1, 1),    # 3 右下
    (1, 0),    # 4 左下
    (0, -1),   # 5 左
)

# 奇数行はずれていないので、上下の隣は (row-1, col-1) と (row-1, col)。
_ODD_ROW_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, -1),  # 0 左上
    (-1, 0),   # 1 右上
    (0, 1),    # 2 右
    (1, 0),    # 3 右下
    (1, -1),   # 4 左下
    (0, -1),   # 5 左
)


def neighbor(position: int, direction: int, width: int, height: int) -> int | None:
    """指定方向の隣接セル番号。マップ外なら None。"""

    if not 0 <= direction <= 5:
        raise ValueError(f"方向は 0～5 で指定してください: {direction}")
    row, column = divmod(position, width)
    deltas = _EVEN_ROW_DELTAS if row % 2 == 0 else _ODD_ROW_DELTAS
    row_delta, column_delta = deltas[direction]
    next_row = row + row_delta
    next_column = column + column_delta
    if not (0 <= next_row < height and 0 <= next_column < width):
        return None
    return next_row * width + next_column


@lru_cache(maxsize=64)
def neighbor_table(width: int, height: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """全セルについて ((方向, 隣接セル), ...) を返す表。マップ外は含めない。"""

    table: list[tuple[tuple[int, int], ...]] = []
    for position in range(width * height):
        entries: list[tuple[int, int]] = []
        for direction in range(6):
            next_position = neighbor(position, direction, width, height)
            if next_position is not None:
                entries.append((direction, next_position))
        table.append(tuple(entries))
    return tuple(table)
