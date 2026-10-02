"""経路探索。AIの「アルゴリズム側」の土台。

同じ日の中では地形も道路状態も変わらないので、ある始点からの最短経路表は
一度だけ計算してキャッシュします。これをやらないと、1手を選ぶたびに
マップ全体のダイクストラを何万回も回すことになり、学習が実用速度に届きません。

コストは (ステップ数, 燃料) の辞書順です。ステップ数が同じなら燃料の少ない
経路を選びます。どちらも移動元のセルで決まるため（質問回答集その1 Q10 /
その2 Q25）、通常のダイクストラで正しく解けます。
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

from hexa.problem import Problem, walkable_neighbors
from hexa.rules import RoadStatus, Terrain, fuel_cost, move_steps


# 「燃料をどれだけ重く見るか」の段階数。細かくするとキャッシュが効かなくなる。
FUEL_WEIGHT_STEPS = 4


def fuel_weight(remaining_steps: int, remaining_fuel: int) -> int:
    """残りステップと残り燃料から、燃料の重み（0〜4）を決める。

    1マス進むのに必要なものは地形で違います。

        道路(順調): 1ステップ / 燃料2
        平地      : 2ステップ / 燃料1

    道路は2倍速いぶん、**燃料は4倍**かかります。
    ステップ最短の道だけを選ぶと道路ばかり通り、燃料が先に尽きて動けません。
    実戦のマップで測ると、最短経路の燃料消費は1ステップあたり1.75
    （道路の2.0にほぼ張り付き）でした。燃料110・総ステップ268の試合では、
    燃料が全体の4割ぶんしか持たない計算になります。

    残りステップ S と残り燃料 F のとき、1マスのコストを

        ステップ数 / S + 燃料 / F

    で測れば、足りないほうの資源が自動的に重くなります。
    """

    if remaining_fuel <= 0:
        return FUEL_WEIGHT_STEPS
    ratio = max(0.0, min(1.0, remaining_steps / max(1.0, float(remaining_fuel))))
    return int(round(ratio * FUEL_WEIGHT_STEPS))


def _blend(steps: int, fuel: int, weight: int) -> int:
    """1マス分のコスト。weight が大きいほど燃料を重く見る。"""

    return steps * (FUEL_WEIGHT_STEPS - weight) + fuel * weight


@dataclass(frozen=True)
class PathTable:
    """ある始点からの最短経路表。"""

    origin: int
    steps: dict[int, int]                  # セル -> 到達に要するステップ数
    fuel: dict[int, int]                   # セル -> 到達に要する燃料
    previous: dict[int, tuple[int, int]]   # セル -> (直前セル, その時の方向)

    def reachable(self, position: int) -> bool:
        return position in self.steps

    def directions_to(self, position: int) -> list[int]:
        """始点から position までの移動方向の列を返す。"""

        if position == self.origin:
            return []
        directions: list[int] = []
        current = position
        while current != self.origin:
            entry = self.previous.get(current)
            if entry is None:
                raise KeyError(f"セル{position}へは到達できません")
            previous_position, direction = entry
            directions.append(direction)
            current = previous_position
        directions.reverse()
        return directions


def road_key(road_status: dict[int, RoadStatus]) -> tuple[tuple[int, int], ...]:
    """道路状態を辞書からハッシュ可能なキーに変換する。順調は省略して短くする。"""

    return tuple(
        sorted(
            (position, int(status))
            for position, status in road_status.items()
            if status != RoadStatus.SMOOTH
        )
    )


# 経路表のキャッシュに使ってよい「セル数」の合計。
# 1つの表はマップのセル数に比例したメモリを使い、実測で 1セルあたり約150バイト。
# 20万セルなら 1プロセスあたり約30MBに収まります。
#
# ここを表の「個数」で制限すると事故ります。32x32 のマップでは1つの表が
# 150KB あるので、4096個まで貯めると1プロセスで610MB、64並列なら38GBになり、
# メモリ不足でプロセスごと殺されます（実際に起きました）。
CACHE_CELL_BUDGET = 200_000


class PathFinder:
    """1つの問題に対する経路探索器。日ごとに道路状態が変わるので状態別にキャッシュする。"""

    def __init__(self, problem: Problem, cache_limit: int | None = None) -> None:
        self.problem = problem
        self.neighbors = walkable_neighbors(problem)
        self.terrains = tuple(problem.terrain_at(p) for p in range(problem.cell_count))
        self._cache: dict[tuple[tuple[tuple[int, int], ...], int], PathTable] = {}
        if cache_limit is None:
            # マップが大きいほど1つの表が重いので、貯める数を減らす。
            cache_limit = max(64, CACHE_CELL_BUDGET // max(1, problem.cell_count))
        self._cache_limit = cache_limit
        self.compute_count = 0
        self.query_count = 0
        # 直前に使った道路状態とそのキー（作り直しを避けるため）
        self._last_road_object: object = None
        self._last_road_key: tuple = ()
        # 1マスのコスト表（道路状態と重みごと）
        self._cost_cache: dict[tuple[int, int], list] = {}

    def table(
        self,
        origin: int,
        road_status: dict[int, RoadStatus],
        weight: int = 0,
    ) -> PathTable:
        """始点からの経路表。weight=0 ならステップ最短、大きいほど燃料節約。"""

        if weight < 0:
            weight = 0
        elif weight > FUEL_WEIGHT_STEPS:
            weight = FUEL_WEIGHT_STEPS

        # road_key は毎回ソートし直すので、同じ道路状態なら使い回す。
        # 道路状態は1日のあいだ変わらないので、ほぼ必ず当たります。
        # 1案を作るのに8万回も引くため、ここが効きます。
        if road_status is self._last_road_object:
            road = self._last_road_key
        else:
            road = road_key(road_status)
            self._last_road_object = road_status
            self._last_road_key = road

        key = (road, origin, weight)
        self.query_count += 1
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        table = self._compute(origin, road_status, weight)
        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[key] = table
        return table

    def _cost_table(
        self,
        road_status: dict[int, RoadStatus],
        weight: int,
    ) -> list[tuple[int, int, int] | None]:
        """各マスを出るときの (歩数, 燃料, 混ぜたコスト)。池は None。

        同じ道路状態と重みなら使い回します。探索1回ごとに作り直すと、
        それだけで全体の2割を食います。
        """

        key = (id(road_status), weight)
        cached = self._cost_cache.get(key)
        if cached is not None:
            return cached

        pond = int(Terrain.POND)
        road = int(Terrain.ROAD)
        table: list[tuple[int, int, int] | None] = []
        # 道路以外は道路状態に左右されないので、地形ごとに1回だけ作る。
        plain: dict[int, tuple[int, int, int]] = {}
        for position, terrain in enumerate(self.terrains):
            if terrain == pond:
                table.append(None)
                continue
            if terrain == road:
                status = road_status.get(position, RoadStatus.SMOOTH)
                steps = move_steps(terrain, status)
                burn = fuel_cost(terrain)
                table.append((steps, burn, _blend(steps, burn, weight)))
                continue
            entry = plain.get(terrain)
            if entry is None:
                steps = move_steps(terrain, RoadStatus.SMOOTH)
                burn = fuel_cost(terrain)
                entry = (steps, burn, _blend(steps, burn, weight))
                plain[terrain] = entry
            table.append(entry)

        if len(self._cost_cache) > 24:
            self._cost_cache.clear()
        self._cost_cache[key] = table
        return table

    def _compute(
        self,
        origin: int,
        road_status: dict[int, RoadStatus],
        weight: int = 0,
    ) -> PathTable:
        """始点から全セルへの最短経路。

        ここが処理時間の9割を占めるので、辞書ではなくリストで持ちます。
        セル番号は 0 から cell_count-1 の連番なので、リストで引けます。
        辞書の get を1回の試走で900万回呼んでいたのを無くします。
        """

        self.compute_count += 1
        neighbors = self.neighbors
        terrains = self.terrains
        cell_count = self.problem.cell_count
        unreached = 1 << 60

        best_cost = [unreached] * cell_count
        best_steps = [0] * cell_count
        best_fuel = [0] * cell_count
        previous_cell = [-1] * cell_count
        previous_direction = [-1] * cell_count
        settled = [False] * cell_count

        best_cost[origin] = 0
        frontier: list[tuple[int, int]] = [(0, origin)]
        pond = int(Terrain.POND)

        # 1マスを出るときの「歩数・燃料・コスト」は地形と道路状態だけで決まります。
        # 探索の中で60万回も計算していたので、先に表にしておきます。
        # 道路以外は道路状態に左右されないので、地形ごとに1回で済みます。
        cost_table = self._cost_table(road_status, weight)

        while frontier:
            cost, position = heappop(frontier)
            if settled[position] or cost > best_cost[position]:
                continue
            settled[position] = True
            entry = cost_table[position]
            if entry is None:
                continue
            step_cost, burn, blended = entry
            next_cost = cost + blended
            next_steps = best_steps[position] + step_cost
            next_fuel = best_fuel[position] + burn
            for direction, next_position in neighbors[position]:
                if settled[next_position]:
                    continue
                if next_cost >= best_cost[next_position]:
                    # 同じコストなら、ステップ数が少ないほうを採る。
                    if next_cost > best_cost[next_position]:
                        continue
                    if next_steps >= best_steps[next_position]:
                        continue
                best_cost[next_position] = next_cost
                best_steps[next_position] = next_steps
                best_fuel[next_position] = next_fuel
                previous_cell[next_position] = position
                previous_direction[next_position] = direction
                heappush(frontier, (next_cost, next_position))

        settled_steps = {
            index: best_steps[index] for index in range(cell_count) if settled[index]
        }
        settled_fuel = {
            index: best_fuel[index] for index in range(cell_count) if settled[index]
        }
        previous = {
            index: (previous_cell[index], previous_direction[index])
            for index in range(cell_count)
            if settled[index] and previous_cell[index] >= 0
        }
        return PathTable(
            origin=origin,
            steps=settled_steps,
            fuel=settled_fuel,
            previous=previous,
        )
