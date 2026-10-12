"""公式フォーマットの問題（マップ構成）を表すモデルと入出力。

フォーマット資料 p.1 の「試合開始前のマップ構成フォーマット」に対応します。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from hexa import rules
from hexa.hexgrid import neighbor_table
from hexa.rules import Terrain


@dataclass(frozen=True)
class Spot:
    brand: int
    position: int
    stocks: int


@dataclass(frozen=True)
class Problem:
    starts_at: int
    day_seconds: tuple[int, ...]
    day_steps: tuple[int, ...]
    height: int
    width: int
    cells: tuple[int, ...]          # 行優先で平坦化した地形タイプ
    spots: tuple[Spot, ...]
    agent_positions: tuple[int, ...]
    fuel_limit: int
    players: int
    busy_threshold: int
    jammed_threshold: int
    # 地形の並び。最初に呼ばれたときに1度だけ作ります
    # （`frozen=True` なので `object.__setattr__` で入れます）。
    _terrains: tuple | None = field(default=None, compare=False, repr=False)

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    @property
    def day_count(self) -> int:
        return len(self.day_steps)

    @property
    def agent_count(self) -> int:
        return len(self.agent_positions)

    def terrain_at(self, position: int) -> Terrain:
        """そのマスの地形。

        **`Terrain(...)` を毎回作らないようにしています。**

        1日ぶんの計画を1回作るだけで600回ほど呼ばれ、磨きでは
        その再現を何万回も回します。実測（32x32の1日を300回）では
        enum を作る時間だけで全体の10%でした。問題は作ったあと
        変わらないので、最初に1度だけ並べて使い回します。
        """

        table = self._terrains
        if table is None:
            table = tuple(Terrain(value) for value in self.cells)
            object.__setattr__(self, "_terrains", table)
        return table[position]

    def to_json_dict(self) -> dict[str, Any]:
        rows = [
            list(self.cells[row * self.width:(row + 1) * self.width])
            for row in range(self.height)
        ]
        return {
            "startsAt": self.starts_at,
            "daySeconds": list(self.day_seconds),
            "daySteps": list(self.day_steps),
            "map": {"height": self.height, "width": self.width, "cells": rows},
            "spots": [
                {"brand": spot.brand, "pos": spot.position, "stocks": spot.stocks}
                for spot in self.spots
            ],
            "agents": list(self.agent_positions),
            "fuelLimits": self.fuel_limit,
            "players": self.players,
            "busyThreshold": self.busy_threshold,
            "jammedThreshold": self.jammed_threshold,
        }


def problem_from_json_dict(raw: dict[str, Any]) -> Problem:
    map_info = raw["map"]
    height = int(map_info["height"])
    width = int(map_info["width"])
    rows = map_info["cells"]
    if len(rows) != height:
        raise ValueError("cells の行数が height と一致しません")
    cells: list[int] = []
    for row in rows:
        if len(row) != width:
            raise ValueError("cells の列数が width と一致しません")
        cells.extend(int(value) for value in row)
    return Problem(
        starts_at=int(raw["startsAt"]),
        day_seconds=tuple(int(value) for value in raw["daySeconds"]),
        day_steps=tuple(int(value) for value in raw["daySteps"]),
        height=height,
        width=width,
        cells=tuple(cells),
        spots=tuple(
            Spot(brand=int(s["brand"]), position=int(s["pos"]), stocks=int(s["stocks"]))
            for s in raw["spots"]
        ),
        agent_positions=tuple(int(value) for value in raw["agents"]),
        fuel_limit=int(raw["fuelLimits"]),
        players=int(raw["players"]),
        busy_threshold=int(raw["busyThreshold"]),
        jammed_threshold=int(raw["jammedThreshold"]),
    )


def load_problem(path: str | Path) -> Problem:
    with open(path, encoding="utf-8") as handle:
        return problem_from_json_dict(json.load(handle))


def save_problem(problem: Problem, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(problem.to_json_dict(), handle, ensure_ascii=False, indent=2)


def validate_problem(problem: Problem) -> list[str]:
    """公式資料で確認できる範囲の制約を検査し、違反の一覧を返す。"""

    issues: list[str] = []

    if not rules.MIN_MAP_SIDE <= problem.height <= rules.MAX_MAP_SIDE:
        issues.append(f"縦幅は{rules.MIN_MAP_SIDE}～{rules.MAX_MAP_SIDE}です: {problem.height}")
    if not rules.MIN_MAP_SIDE <= problem.width <= rules.MAX_MAP_SIDE:
        issues.append(f"横幅は{rules.MIN_MAP_SIDE}～{rules.MAX_MAP_SIDE}です: {problem.width}")
    if not rules.MIN_DAYS <= problem.day_count <= rules.MAX_DAYS:
        issues.append(f"日数は{rules.MIN_DAYS}～{rules.MAX_DAYS}日です: {problem.day_count}")
    if len(problem.day_seconds) != problem.day_count:
        issues.append("daySeconds と daySteps の要素数が一致しません")
    if not rules.MIN_AGENTS <= problem.agent_count <= rules.MAX_AGENTS:
        issues.append(
            f"エージェント数は{rules.MIN_AGENTS}～{rules.MAX_AGENTS}体です: {problem.agent_count}"
        )

    # 質問回答集その1 Q20: 各日のステップ数は W+H 以上 4*(W+H) 以下
    span = problem.width + problem.height
    for day_index, steps in enumerate(problem.day_steps):
        if not span * rules.DAY_STEPS_MIN_RATIO <= steps <= span * rules.DAY_STEPS_MAX_RATIO:
            issues.append(
                f"{day_index + 1}日目のステップ数が範囲外です"
                f"（{span}～{span * rules.DAY_STEPS_MAX_RATIO}）: {steps}"
            )

    # 質問回答集その1 Q14 / その3 Q62: 燃料上限は1日目のステップ数の1～3倍
    first_day_steps = problem.day_steps[0] if problem.day_steps else 0
    if first_day_steps:
        low = first_day_steps * rules.FUEL_LIMIT_MIN_RATIO
        high = first_day_steps * rules.FUEL_LIMIT_MAX_RATIO
        if not low <= problem.fuel_limit <= high:
            issues.append(
                f"燃料上限は1日目のステップ数の1～3倍（{int(low)}～{int(high)}）です: "
                f"{problem.fuel_limit}"
            )

    if not rules.MIN_BUSY_THRESHOLD <= problem.busy_threshold <= rules.MAX_BUSY_THRESHOLD:
        issues.append(f"混雑基準値は1～5です: {problem.busy_threshold}")
    if not rules.MIN_JAMMED_THRESHOLD <= problem.jammed_threshold <= rules.MAX_JAMMED_THRESHOLD:
        issues.append(f"渋滞基準値は2～10です: {problem.jammed_threshold}")
    if problem.busy_threshold >= problem.jammed_threshold:
        issues.append("混雑基準値は渋滞基準値より小さい必要があります")
    if problem.players < 1:
        issues.append("players は1以上です")

    # 質問回答集その1 Q15: スポット数はエージェント数以上、max(縦,横)以下
    spot_count = len(problem.spots)
    if spot_count < problem.agent_count:
        issues.append(f"スポット数はエージェント数以上です: {spot_count}")
    if spot_count > max(problem.height, problem.width):
        issues.append(f"スポット数はmax(縦,横)以下です: {spot_count}")

    # 質問回答集その2 Q34 / その1 Q18: 1セルに2つ以上のスポットは置かれない
    if len({spot.position for spot in problem.spots}) != spot_count:
        issues.append("同一セルに複数のスポットがあります")

    for spot in problem.spots:
        if problem.terrain_at(spot.position) != Terrain.PLAIN:
            issues.append(f"スポットは平地に置かれます: セル{spot.position}")
        # 募集要項 p.9: 最大在庫数は 1 ～ 1チームのエージェント数
        if not 1 <= spot.stocks <= problem.agent_count:
            issues.append(f"最大在庫数は1～{problem.agent_count}です: セル{spot.position}")

    brands = {spot.brand for spot in problem.spots}
    if brands and (min(brands) != 0 or max(brands) != len(brands) - 1):
        issues.append("系列は0から連番で割り当ててください")

    # 募集要項 p.10 / 質問回答集その2 Q37: 初期位置は全て異なるスポットなしの平地
    if len(set(problem.agent_positions)) != problem.agent_count:
        issues.append("エージェントの初期位置が重複しています")
    spot_positions = {spot.position for spot in problem.spots}
    for position in problem.agent_positions:
        if problem.terrain_at(position) != Terrain.PLAIN:
            issues.append(f"エージェント初期位置は平地です: セル{position}")
        if position in spot_positions:
            issues.append(f"エージェント初期位置にスポットは置かれません: セル{position}")

    # 質問回答集その2 Q35: 各地形は必ず1つ以上存在する
    present = set(problem.cells)
    for terrain in (Terrain.PLAIN, Terrain.ROAD, Terrain.MOUNTAIN, Terrain.POND):
        if int(terrain) not in present:
            issues.append(f"{terrain.name} のセルが1つもありません")

    # 募集要項 p.9 / 質問回答集その2 Q46: 移動可能セルは全て連結
    if not _is_fully_connected(problem):
        issues.append("移動可能なセルが連結していません")

    return issues


@lru_cache(maxsize=32)
def _walkable_neighbors(problem: Problem) -> tuple[tuple[tuple[int, int], ...], ...]:
    """池を除いた隣接表。(方向, 隣接セル) のタプル列を全セル分返す。"""

    table = neighbor_table(problem.width, problem.height)
    result: list[tuple[tuple[int, int], ...]] = []
    for position in range(problem.cell_count):
        entries = tuple(
            (direction, next_position)
            for direction, next_position in table[position]
            if problem.terrain_at(next_position) != Terrain.POND
        )
        result.append(entries)
    return tuple(result)


def walkable_neighbors(problem: Problem) -> tuple[tuple[tuple[int, int], ...], ...]:
    return _walkable_neighbors(problem)


def _is_fully_connected(problem: Problem) -> bool:
    walkable = [
        position
        for position in range(problem.cell_count)
        if problem.terrain_at(position) != Terrain.POND
    ]
    if not walkable:
        return False
    neighbors = walkable_neighbors(problem)
    seen = {walkable[0]}
    stack = [walkable[0]]
    while stack:
        current = stack.pop()
        for _direction, next_position in neighbors[current]:
            if next_position not in seen:
                seen.add(next_position)
                stack.append(next_position)
    return len(seen) == len(walkable)
