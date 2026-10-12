"""**その日に取れる玉数の上限を先に測る。**

準決勝の結果ではっきりしました。

  16x16: 日別満点55 / 1位 53 / こちら53（**並んだ**）。玉 71 対 65
  32x32: 日別満点132 / **上位18チーム全員が132**。玉 304 対 216

**日別はもう差がつかず、勝負は玉数**です。だとすれば「その日に何玉
まで行けるのか」を先に知っておきたい。知っていれば

  - 届いているのか、まだ余地があるのかが分かる
  - 何が縛っているのか（歩数・燃料・在庫）で作戦を変えられる
  - 上限に届いたら、そこで探索を止められる（回答時間は4番目の基準）

の3つができます。

**解き方**

1台が k 軒まわる最短を、**(踏んだ店の集合, いまの店) ごとに最短
歩数と最小燃料を持つ探索**で出します。予算を超えた枝は捨てるので、
32店でも1台1〜2秒で解けます。店を2度は数えません（1台が同じ店から
取れるのは1日1玉）。

盤面ぜんたいは、さらに**店ごとの1日の在庫**で抑えます。

**実測（準決勝32x32・22系列32店・巡回6台・燃料80・1日96歩）**

| 条件 | 玉の上限 |
|---|---|
| 手持ちの燃料だけ（補給なし） | **243**（こちらの実績は240） |
| 1台1日0.5回の補給 | 356 |
| 毎朝満タン・補給なし | **373** |
| 毎朝満タン・夜に1店へ戻る条件つき | 354 |
| 1位の実績 | 304 |
"""
from __future__ import annotations

from hexa.problem import Problem
from hexa.rules import AgentKind

# 前線の大きさの上限。超えたら歩数の安い順に残すので、そのときは
# 「上限」ではなく「近い値」になります（`exact` で報せます）。
FRONTIER_CAP = 5_000
# 1日に見込む補給の回数（1台あたり）。0 なら手持ちの燃料だけ。
DEFAULT_REFUELS = 0
BIG = 10 ** 9


def _tables(problem: Problem, finder, road_status):
    spots = list(problem.spots)
    count = len(spots)
    steps = [[0] * count for _ in range(count)]
    fuel = [[0] * count for _ in range(count)]
    for i, a in enumerate(spots):
        table = finder.table(a.position, road_status)
        for j, b in enumerate(spots):
            steps[i][j] = table.steps.get(b.position, BIG)
            fuel[i][j] = table.fuel.get(b.position, BIG)
    return spots, steps, fuel


def car_ceiling(between_steps, between_fuel, first_steps, first_fuel,
                step_budget: int, fuel_budget: int) -> tuple[int, bool]:
    """**1台が1日に踏める店の最大数**と、それが厳密かどうか。"""

    count = len(first_steps)
    if count == 0 or step_budget <= 0:
        return 0, True
    frontier: dict[tuple[int, int], tuple[int, int]] = {}
    for j in range(count):
        if first_steps[j] <= step_budget and first_fuel[j] <= fuel_budget:
            frontier[(1 << j, j)] = (first_steps[j], first_fuel[j])
    best = 1 if frontier else 0
    exact = True
    while frontier:
        nxt: dict[tuple[int, int], tuple[int, int]] = {}
        for (mask, here), (used_steps, used_fuel) in frontier.items():
            row_steps = between_steps[here]
            row_fuel = between_fuel[here]
            for j in range(count):
                if (mask >> j) & 1:
                    continue
                go = used_steps + row_steps[j]
                if go > step_budget:
                    continue
                burn = used_fuel + row_fuel[j]
                if burn > fuel_budget:
                    continue
                key = (mask | (1 << j), j)
                got = nxt.get(key)
                if got is None or go < got[0]:
                    nxt[key] = (go, burn)
        if not nxt:
            break
        best += 1
        if len(nxt) > FRONTIER_CAP:
            exact = False
            keep = sorted(nxt.items(), key=lambda item: item[1][0])
            nxt = dict(keep[:FRONTIER_CAP])
        frontier = nxt
    return best, exact


def day_ceiling(problem: Problem, finder, kinds, positions, fuels,
                road_status, day_steps: int,
                refuels: float = DEFAULT_REFUELS) -> dict:
    """その日に取れる玉数の上限と、**何が縛っているか**を返す。

    返す辞書: balls / by_steps / by_fuel / by_stock / binds / per_car / exact
    """

    touring = [index for index, kind in enumerate(kinds)
               if kind == AgentKind.TOURING]
    if not touring or not problem.spots:
        return {"balls": 0, "by_steps": 0, "by_fuel": 0, "by_stock": 0,
                "binds": "stock", "per_car": [], "exact": True}
    spots, between_steps, between_fuel = _tables(problem, finder, road_status)
    per_steps = []
    per_fuel = []
    exact = True
    # **燃料を無限にした「歩数だけ」の上限は出しません。**
    # 予算が緩いぶん前線がふくれ、32x32で8秒かかります（1日の
    # 持ち時間の13%）。縛っているのはほぼ燃料のほうなので、
    # 満タンで見た数字を「歩数だけ」の代わりにします。
    for index in touring:
        table = finder.table(positions[index], road_status)
        first_steps = [table.steps.get(s.position, BIG) for s in spots]
        first_fuel = [table.fuel.get(s.position, BIG) for s in spots]
        got, ok = car_ceiling(between_steps, between_fuel, first_steps,
                              first_fuel, day_steps, problem.fuel_limit)
        per_steps.append(got)
        exact = exact and ok
        room = fuels[index] + int(problem.fuel_limit * refuels)
        if room >= problem.fuel_limit:
            per_fuel.append(got)
        else:
            got2, ok2 = car_ceiling(between_steps, between_fuel, first_steps,
                                    first_fuel, day_steps, room)
            per_fuel.append(got2)
            exact = exact and ok2
    by_steps = sum(per_steps)
    by_fuel = sum(per_fuel)
    # 1つの店から1日に出るのは stocks 個まで。1台が同じ店から取れる
    # のは1日1玉なので、台数でも抑えます。
    by_stock = sum(min(spot.stocks, len(touring)) for spot in problem.spots)
    balls = min(by_steps, by_fuel, by_stock)
    if by_stock <= by_fuel and by_stock <= by_steps:
        binds = "在庫"
    elif by_fuel < by_steps:
        binds = "燃料"
    else:
        binds = "歩数"
    return {"balls": balls, "by_steps": by_steps, "by_fuel": by_fuel,
            "by_stock": by_stock, "binds": binds, "per_car": per_fuel,
            "exact": exact}


def describe(got: dict) -> str:
    """1行で読める形にする。"""

    mark = "" if got.get("exact", True) else "（近い値）"
    return (f"この日の上限 玉{got['balls']}{mark}"
            f"（満タンなら{got['by_steps']} / いまの燃料で{got['by_fuel']}"
            f" / 在庫{got['by_stock']} → {got['binds']}が縛っています）")
