"""巡回車の経路を、全車まとめて設計する。

**なぜ必要か**

これまでは「1台ずつ、その場で次に良い店を選ぶ」繰り返しでした。
実戦1535の5日目（99歩）を調べると、

    上限64玉（在庫の合計）
    いまの実装 50玉 / 1位は推定60玉

    全24店を訪問済み、16店は取り切り
    残り14玉は在庫の多い店の2台目・3台目

取りこぼしではなく、**全部回ったうえで同じ店に戻る余力がない**
状態でした。1玉あたり7.9歩かかっており、1位は6.6歩で回っています。

理想的な貪欲（在庫を全車で共有し、常に一番近い店へ向かう）を
書いて測ると48玉。いまの実装の50玉はすでにそれを超えています。
**「次に近い店」を選ぶ考え方では、ここが限界**です。

**何をするか**

1日の初めに、全車の行き先をまとめて決めます。

  1. 「どの車が、どの店を、何番目に訪れるか」を一度に割り当てる
  2. 割り当てたあと、車ごとに訪問順を短くなるよう並べ替える
  3. 2台を入れ替えると短くなる組があれば交換する（改善を繰り返す）

その場しのぎではなく、1日を通した配分を先に決めるのが違いです。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hexa.pathfind import PathFinder
from hexa.problem import Problem
from hexa.rules import AgentKind, RoadStatus


# 並べ替えて余った歩数で、何回まで取りに戻るか。
REFILL_ROUNDS = 3

# 残った在庫を、まだ行っていない車へ振り直す回数。
SWAP_ROUNDS = 3

# 訪問の多い車を、どれだけ後回しにするか。
#
# **実測で効かなかったので 0 にしています。**
# 実戦1535の5日目で車1が9店、他は15〜17店という偏りがありました。
# 均そうとしても 441→439〜442玉 とほぼ変わりません。
# 偏りは「その車の近くに店がない」結果であって、原因ではないためです。
BALANCE_WEIGHT = 0.0


@dataclass
class CarPlan:
    """1台ぶんの訪問計画。"""

    agent: int
    spots: list[int] = field(default_factory=list)
    steps: int = 0
    fuel_used: int = 0


# **設計も実行と同じ見方で経路を測るか。**
#
# 設計はここで `weight=0`（歩数最短）の経路表を引いていました。
# いっぽう実行（`PlannedRoutePolicy`）は `fuel_weight` をかけた
# 「燃料を考えた経路」を歩きます。準決勝32x32の1日目で同じ6店を
# 回ると 歩212/燃料205（設計の見方）対 歩**223**/燃料190（実行の
# 見方）で、**5%ずれます**。設計が「96歩で入る」と言った順路が
# 実際には101歩かかり、最後の店が落ちます。
#
# 実行を設計に合わせる向きは実測で悪くなりました（29試合で
# 順位318→328。燃料のほうが縛っているため）。
#
# 逆向き（設計を実行に合わせる）も試しましたが、**準決勝32x32で
# 玉231のまま1つも動きません**。磨きが設計の差を洗い流すためです。
# 未測定の既定を残さないため False で止め、仕掛けだけ残します。
# 設計の段階で終点まで歩き切る必要がある形（夜合流）を作るときに
# 必要になります。
DESIGN_FUEL_WEIGHT = False


def _table(finder, origin, road_status, cache, weight=0):
    if not DESIGN_FUEL_WEIGHT:
        weight = 0
    key = origin if not weight else (origin, weight)
    got = cache.get(key)
    if got is None:
        got = finder.table(origin, road_status, weight)
        cache[key] = got
    return got


def design_day(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    available: dict[int, int] | None = None,
    stock_weight: float = 0.0,
    order: tuple[int, ...] | None = None,
    brand_weight: float = 0.0,
    acquired_brands: set[int] | None = None,
) -> dict[int, list[int]]:
    """全車の訪問順をまとめて決める。戻り値は 車番号 -> 店の並び。

    available は「その店にあと何玉あるか」。省略すると在庫から作ります。
    """

    touring = [i for i, kind in enumerate(kinds) if kind == AgentKind.TOURING]
    if not touring:
        return {}
    if order:
        # 車を見る順番を変えると、割り当ても変わります。
        touring = [touring[i % len(touring)] for i in order][:len(touring)]
        seen = []
        for index in touring:
            if index not in seen:
                seen.append(index)
        for index in [i for i, k in enumerate(kinds) if k == AgentKind.TOURING]:
            if index not in seen:
                seen.append(index)
        touring = seen

    if available is None:
        available = {
            spot.position: min(spot.stocks, len(touring)) for spot in problem.spots
        }
    left = {pos: count for pos, count in available.items() if count > 0}
    if not left:
        return {index: [] for index in touring}

    # **その日まだ取っていない系列を優先できるようにします。**
    #
    # 実戦1627の1日目で、設計は11系列中9系列しか押さえられず、
    # 候補として採用されませんでした。玉数だけを見て系列を
    # 揃えないと、日別の得点を落とします。
    brand_of = {spot.position: spot.brand for spot in problem.spots}
    fresh = set()
    if brand_weight > 0:
        taken = acquired_brands or set()
        fresh = {b for b in brand_of.values() if b not in taken}

    cache: dict[int, object] = {}
    plans = {index: CarPlan(agent=index) for index in touring}
    # 各車が「いまどこにいて、あと何歩・何燃料あるか」
    state = {
        index: (positions[index], day_steps, fuels[index]) for index in touring
    }
    # 同じ車が同じ店から2玉取れないので、車ごとに訪問済みを覚える。
    visited = {index: set() for index in touring}

    while True:
        best = None
        for index in touring:
            here, steps_left, fuel_left = state[index]
            if steps_left <= 0:
                continue
            table = _table(finder, here, road_status, cache)
            for spot, count in left.items():
                if count <= 0 or spot in visited[index]:
                    continue
                if not table.reachable(spot):
                    continue
                need_steps = table.steps[spot]
                need_fuel = table.fuel[spot]
                if need_steps > steps_left or need_fuel > fuel_left:
                    continue
                # 近いほど良い。同じ距離なら在庫の多い店を先に。
                #
                # stock_weight を上げると、少し遠くても在庫の多い店を
                # 選びます。1回の設計が0.06秒と軽いので、重みを変えて
                # 何通りも試し、いちばん良いものを採ります。
                #
                # **訪問の少ない車を優先します。**
                # 実戦1535の5日目で、車1が9店に対し他は15〜17店という
                # 偏りが出ていました。手の空いている車に回すほうが、
                # 全体では多く取れます。
                behind = len(plans[index].spots) * BALANCE_WEIGHT
                bonus = 0.0
                if brand_weight > 0 and brand_of.get(spot) in fresh:
                    bonus = brand_weight
                score = (need_steps - count * stock_weight + behind - bonus,
                         -count)
                if best is None or score < best[0]:
                    best = (score, index, spot, need_steps, need_fuel)
        if best is None:
            break
        _score, index, spot, used_steps, used_fuel = best
        here, steps_left, fuel_left = state[index]
        state[index] = (spot, steps_left - used_steps, fuel_left - used_fuel)
        plans[index].spots.append(spot)
        plans[index].steps += used_steps
        plans[index].fuel_used += used_fuel
        visited[index].add(spot)
        fresh.discard(brand_of.get(spot))
        left[spot] -= 1
        if left[spot] <= 0:
            del left[spot]

    # **残った在庫を、まだ行っていない車に回します。**
    #
    # 実戦1535の5日目で、店518に2玉残っているのに誰も取りに
    # 行きませんでした。すでに2台が訪問済みで、残る2台は
    # 余り3歩しかなかったためです。
    #
    # 割り当ての最後に、余力のある車へ残りを振り直します。
    for _round in range(SWAP_ROUNDS):
        if not left:
            break
        moved = False
        for spot, count in list(left.items()):
            if count <= 0:
                continue
            for index in touring:
                if spot in visited[index]:
                    continue
                spots = plans[index].spots
                start = positions[index]
                # この店を足したときに、歩数と燃料が足りるか
                trial = spots + [spot]
                cost = route_cost(finder, road_status, start, trial, cache)
                fuel = _route_fuel(finder, road_status, start, trial, cache)
                if cost <= day_steps and fuel <= fuels[index]:
                    plans[index].spots = trial
                    visited[index].add(spot)
                    left[spot] -= 1
                    if left[spot] <= 0:
                        del left[spot]
                    moved = True
                    break
        if not moved:
            break

    # **並べ替えて余った歩数で、もう一度取りに行きます。**
    #
    # 訪問順を短くすると歩数が余ります（13店で74歩→73歩など）。
    # その余りで、まだ在庫の残る店をもう一巡します。
    for _round in range(REFILL_ROUNDS):
        added = False
        for index in touring:
            spots = plans[index].spots
            if not spots:
                continue
            better = improve_order(finder, road_status, positions[index], spots)
            cost = route_cost(finder, road_status, positions[index], better)
            if cost > day_steps:
                continue
            plans[index].spots = better
            # 余った歩数と燃料で、追加の店へ行けるか
            here = better[-1]
            steps_left = day_steps - cost
            fuel_left = fuels[index] - _route_fuel(
                finder, road_status, positions[index], better, cache
            )
            while steps_left > 0 and fuel_left > 0:
                table = _table(finder, here, road_status, cache)
                pick = None
                for spot, count in left.items():
                    if count <= 0 or spot in visited[index]:
                        continue
                    if not table.reachable(spot):
                        continue
                    if (table.steps[spot] > steps_left
                            or table.fuel[spot] > fuel_left):
                        continue
                    score = (table.steps[spot], -count)
                    if pick is None or score < pick[0]:
                        pick = (score, spot)
                if pick is None:
                    break
                spot = pick[1]
                steps_left -= table.steps[spot]
                fuel_left -= table.fuel[spot]
                plans[index].spots.append(spot)
                visited[index].add(spot)
                left[spot] -= 1
                if left[spot] <= 0:
                    del left[spot]
                here = spot
                added = True
        if not added:
            break

    return {index: plan.spots for index, plan in plans.items()}


def _route_fuel(finder, road_status, start, spots, cache) -> int:
    """その並びで回るのに使う燃料。"""

    total = 0
    here = start
    for spot in spots:
        table = _table(finder, here, road_status, cache)
        if not table.reachable(spot):
            return 10 ** 9
        total += table.fuel[spot]
        here = spot
    return total


def route_cost(finder, road_status, start, spots, cache=None) -> int:
    """その並びで回るのにかかる歩数。届かない店があれば大きな値。"""

    if cache is None:
        cache = {}
    total = 0
    here = start
    for spot in spots:
        table = _table(finder, here, road_status, cache)
        if not table.reachable(spot):
            return 10 ** 9
        total += table.steps[spot]
        here = spot
    return total


def improve_order(finder, road_status, start, spots, rounds: int = 2) -> list[int]:
    """訪問順を入れ替えて、歩数が短くなるなら採用する。

    2つの店を入れ替えて短くなるかを順に試します。店が10軒なら
    45通りで、数回まわしても軽い処理です。
    """

    if len(spots) < 3:
        return list(spots)
    cache: dict[int, object] = {}
    best = list(spots)
    best_cost = route_cost(finder, road_status, start, best, cache)
    for _ in range(rounds):
        improved = False
        for a in range(len(best) - 1):
            for b in range(a + 1, len(best)):
                # 入れ替え
                trial = list(best)
                trial[a], trial[b] = trial[b], trial[a]
                cost = route_cost(finder, road_status, start, trial, cache)
                if cost < best_cost:
                    best = trial
                    best_cost = cost
                    improved = True
                    continue
                # **区間の反転**も試します。
                # 道すじが交差しているとき、間をひっくり返すと
                # 短くなることがあります（巡回の定石）。
                trial = best[:a] + best[a:b + 1][::-1] + best[b + 1:]
                cost = route_cost(finder, road_status, start, trial, cache)
                if cost < best_cost:
                    best = trial
                    best_cost = cost
                    improved = True
        if not improved:
            break
    return best


def refine_plans(problem, finder, kinds, positions, fuels, road_status,
                 day_steps, routes, available=None, rounds: int = 30,
                 rng=None):
    """できた順路を、少しずつ入れ替えて改善する。

    **なぜ必要か**

    実戦1535の5日目（99歩・上限64玉）で、設計は53玉でした。
    歩数も燃料も使い切っており、取り残しではありません。
    1位は同じ99歩で60玉。**1玉あたり7.4歩に対し6.6歩**で、
    より短い道すじを見つけています。

    最初に「近い店から順に」決めた割り当ては、全体で見ると
    遠回りになっていることがあります。そこで

      1. ある店を、別の車に移してみる
      2. 2台の間で店を交換してみる

    を繰り返し、歩数が減って玉が増えるなら採用します。

    **実測では改善しませんでした**（5日目53玉のまま）。設計を
    200通り試しても53で頭打ちです。歩数も燃料も使い切っており、
    入れ替えで浮く余地がありません。

    呼び出し側は使っていませんが、別の割り当て方を試すときの
    土台として残します。
    """

    import random as _random

    if rng is None:
        rng = _random.Random(12345)
    touring = [i for i, kind in enumerate(kinds) if kind == AgentKind.TOURING]
    if len(touring) < 2:
        return routes

    if available is None:
        available = {
            spot.position: min(spot.stocks, len(touring)) for spot in problem.spots
        }

    cache: dict[int, object] = {}
    best = {i: list(routes.get(i, [])) for i in touring}

    def ok(index, spots):
        """その並びが歩数と燃料に収まるか。"""

        if len(spots) != len(set(spots)):
            return False
        cost = route_cost(finder, road_status, positions[index], spots, cache)
        if cost > day_steps:
            return False
        used = _route_fuel(finder, road_status, positions[index], spots, cache)
        return used <= fuels[index]

    def total(plans):
        return sum(len(v) for v in plans.values())

    def spare(plans):
        """全車の余った歩数の合計。多いほど、あとで足せる。"""

        out = 0
        for index in touring:
            cost = route_cost(finder, road_status, positions[index],
                              plans[index], cache)
            out += max(0, day_steps - cost)
        return out

    def score_of(plans):
        # 玉数が同じなら、余りの多いほうが良い。
        return (total(plans), spare(plans))

    best_score = score_of(best)
    for _ in range(rounds):
        moved = False
        # 1) 店を別の車へ移す
        for giver in touring:
            if not best[giver]:
                continue
            spot = best[giver][rng.randrange(len(best[giver]))]
            for taker in touring:
                if taker == giver or spot in best[taker]:
                    continue
                trial_give = [x for x in best[giver] if x != spot]
                trial_take = improve_order(
                    finder, road_status, positions[taker],
                    best[taker] + [spot],
                )
                if not ok(taker, trial_take):
                    continue
                trial = dict(best)
                trial[giver] = improve_order(
                    finder, road_status, positions[giver], trial_give
                )
                trial[taker] = trial_take
                if not ok(giver, trial[giver]):
                    continue
                score = score_of(trial)
                if score > best_score:
                    best = trial
                    best_score = score
                    moved = True
                    break
        # 2) 2台で店を交換する
        for a in touring:
            for b in touring:
                if a >= b or not best[a] or not best[b]:
                    continue
                x = best[a][rng.randrange(len(best[a]))]
                y = best[b][rng.randrange(len(best[b]))]
                if x == y or x in best[b] or y in best[a]:
                    continue
                left_a = improve_order(
                    finder, road_status, positions[a],
                    [v for v in best[a] if v != x] + [y],
                )
                left_b = improve_order(
                    finder, road_status, positions[b],
                    [v for v in best[b] if v != y] + [x],
                )
                if not ok(a, left_a) or not ok(b, left_b):
                    continue
                trial = dict(best)
                trial[a] = left_a
                trial[b] = left_b
                score = score_of(trial)
                if score > best_score:
                    best = trial
                    best_score = score
                    moved = True
        if not moved:
            break

    # **縮んだぶんで、残っている店を足します。**
    import collections

    taken = collections.Counter(sp for v in best.values() for sp in v)
    left = {sp: n - taken.get(sp, 0) for sp, n in available.items()}
    left = {sp: n for sp, n in left.items() if n > 0}
    for index in touring:
        while left:
            spots = best[index]
            here = spots[-1] if spots else positions[index]
            cost = route_cost(finder, road_status, positions[index], spots, cache)
            used = _route_fuel(finder, road_status, positions[index], spots, cache)
            table = _table(finder, here, road_status, cache)
            pick = None
            for spot, count in left.items():
                if count <= 0 or spot in spots:
                    continue
                if not table.reachable(spot):
                    continue
                if (cost + table.steps[spot] > day_steps
                        or used + table.fuel[spot] > fuels[index]):
                    continue
                if pick is None or table.steps[spot] < pick[0]:
                    pick = (table.steps[spot], spot)
            if pick is None:
                break
            spot = pick[1]
            best[index] = spots + [spot]
            left[spot] -= 1
            if left[spot] <= 0:
                del left[spot]

    return best


# 1日に何回ぶんの補給を見込んで設計するか。
#
# **なぜ必要か**
#
# これまでの設計は「いま持っている燃料」だけで回れる順路を作って
# いました。補給車は燃料を1滴も使わず、合流すれば満タンになるのに、
# その効果を設計に入れていませんでした。
#
# 実戦1650（24x24・燃料69）のログでは、4日目の開始燃料が
# [15, 62, 43, 7] です。1号車と4号車は3店ぶんしか設計できません。
# 燃料を10倍にして同じ試合を再現すると玉120→152（+32）でした。
# **不足の3割は燃料そのものではなく、補給を計画していないこと**が
# 原因だと分かります。
#
# そこで「1回は補給してもらえる」前提で順路を伸ばし、燃料が切れる
# 地点を補給車の迎え先として並べます。**燃料が切れる場所は動きません**
# （その車はそこで止まって待つ）ので、待ち合わせの時刻を合わせる
# 必要がありません。迎えに行く順番だけを決めれば足ります。
SUPPLY_REFUELS = (1, 2)


def design_with_supply(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    stock_weight: float = 0.0,
    brand_weight: float = 0.0,
    acquired_brands: set[int] | None = None,
    refuels: int = 1,
) -> tuple[dict[int, list[int]], list[tuple[int, int]]]:
    """補給を見込んだ順路と、補給車の迎え先の並びを返す。

    戻り値は (車番号 -> 店の並び, [(迎え先のセル, 迎える車番号), ...])。
    """

    touring = [i for i, kind in enumerate(kinds) if kind == AgentKind.TOURING]
    supply = [i for i, kind in enumerate(kinds) if kind == AgentKind.SUPPLY]
    if not touring:
        return {}, []
    if not supply:
        return design_day(
            problem, finder, kinds, positions, fuels, road_status, day_steps,
            stock_weight=stock_weight, brand_weight=brand_weight,
            acquired_brands=acquired_brands,
        ), []

    # 補給ぶんを足した燃料で順路を設計する。
    budget = list(fuels)
    for index in touring:
        budget[index] = fuels[index] + problem.fuel_limit * refuels
    routes = design_day(
        problem, finder, kinds, positions, budget, road_status, day_steps,
        stock_weight=stock_weight, brand_weight=brand_weight,
        acquired_brands=acquired_brands,
    )

    # それぞれの車が、どこで燃料を切らすか。
    cache: dict[int, object] = {}
    meetings: list[tuple[int, int, int]] = []   # (到着歩数, セル, 車)
    for index in touring:
        here = positions[index]
        steps = 0
        spent = 0
        tank = fuels[index]
        rounds = 0
        for spot in routes.get(index, []):
            table = _table(finder, here, road_status, cache)
            if not table.reachable(spot):
                break
            need_fuel = table.fuel[spot]
            need_steps = table.steps[spot]
            if spent + need_fuel > tank:
                # ここで止まる。**この場所が迎え先**。
                meetings.append((steps, here, index))
                rounds += 1
                if rounds >= refuels:
                    break
                spent = 0
                tank = problem.fuel_limit
                continue
            spent += need_fuel
            steps += need_steps
            here = spot

    # 止まるのが早い車から順に迎えに行く。
    meetings.sort()
    if SUPPLY_SCHEDULE and 1 < len(meetings) <= SUPPLY_SCHEDULE_LIMIT:
        route = _best_meeting_order(finder, road_status, cache,
                                    positions[supply[0]], day_steps, meetings)
    else:
        route = []
        here = positions[supply[0]]
        clock = 0
        for _deadline, target, index in meetings:
            table = _table(finder, here, road_status, cache)
            if not table.reachable(target):
                continue
            clock += table.steps[target]
            if clock > day_steps:
                break
            route.append((target, index))
            here = target
    return routes, route


# **迎える順番は、締め切りの順ではありません。**
#
# `design_with_supply` は「燃料が切れる歩数」が早い車から順に迎えに
# 行きます。ところが補給車の移動時間を見ていないので、間に合わない
# 順番になることがあります。間に合わなければ巡回車はそこで待ちます。
#
# 第1703（32x32）で待ちの歩を数えると、7割が燃料でした
# （「燃料が足りず店に行けない」340歩＋「燃料切れ」271歩）。
#
# 迎え先はせいぜい巡回車の台数ぶんなので、**並べ方を全部試せます**
# （6台で720通り、1通り6回の表引き）。遅れの合計がいちばん小さい
# 並びを選びます。
# **実測で入れませんでした（False で止めています）。**
# 燃料の仕掛け一式でまとめて測ると日別が落ちました（733→729）。
# 考え方は正しい（鹿児島51・沖縄12と同じ）ので、迎え先のセルを
# 探索に入れる仕掛け（`improve.SUPPLY_MEET_SPOTS`）とあわせて
# 作り直す余地があります。
SUPPLY_SCHEDULE = False
# これ以下の件数なら全部試す（7件で5040通り）。
SUPPLY_SCHEDULE_LIMIT = 7


def _best_meeting_order(finder, road_status, cache, start, day_steps,
                        meetings):
    """迎え先の並びのうち、**遅れの合計がいちばん小さいもの**を返す。

    `meetings` は (燃料が切れる歩数, 迎え先のセル, 車番号) の並び。
    戻り値は [(セル, 車番号), ...]。

    遅れ = max(0, 補給車の到着歩数 - その車が止まる歩数)。これが
    そのまま巡回車の待ち歩数になります。迎えられた台数を先に見て、
    同じ台数なら遅れの小さいほうを選びます。
    """

    from itertools import permutations

    best = None
    for order in permutations(range(len(meetings))):
        here = start
        clock = 0
        late = 0
        picked: list[tuple[int, int]] = []
        for at in order:
            deadline, target, index = meetings[at]
            table = _table(finder, here, road_status, cache)
            gap = table.steps.get(target)
            if gap is None:
                break
            clock += gap
            if clock > day_steps:
                break
            late += max(0, clock - deadline)
            picked.append((target, index))
            here = target
        key = (-len(picked), late)
        if best is None or key < best[0]:
            best = (key, picked)
    return best[1] if best else []


def design_insert(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    stock_weight: float = 0.0,
    brand_weight: float = 0.0,
    acquired_brands: set[int] | None = None,
    refuels: int = 0,
    start: dict[int, list[int]] | None = None,
) -> dict[int, list[int]]:
    """**店を順路の「途中」に差し込みながら組み立てる。**

    `start` を渡すと、その順路を土台にして足りないぶんだけ差し込みます
    （崩したあとの埋め直しに使います）。

    **なぜ必要か**

    `design_day` は「いまいる場所から次に良い店へ」と**末尾に足す**
    だけでした。あとで `improve_order` が並べ替えますが、
    足す順番で決まった受け持ちそのものは変わりません。

    実戦1703（32x32）で測ると、1玉あたり9.5歩。3位は8.0歩、
    1位は7.1歩です。立ち止まりは全体の10%しかないので、差は
    **回り方そのもの**にあります。

    そこで、まだ行っていない店を**どの車のどの位置に入れるのが
    いちばん安いか**を毎回全部見て、安い順に差し込みます。
    2店のあいだに割り込めるので、末尾に足すだけでは作れない形が
    できます。

    差し込みの増える歩数は、3回の経路表引きで出せます。

        増える歩数 = (前の店→新しい店) + (新しい店→次の店)
                     - (前の店→次の店)

    `route_cost` が隣どうしの最短経路の合計なので、この式が
    そのまま正しい値になります。全部を組み直す必要はありません。
    """

    touring = [i for i, kind in enumerate(kinds) if kind == AgentKind.TOURING]
    if not touring:
        return {}
    cache: dict[object, object] = {}
    # **車ごとの経路の重み。** 実行（`PlannedRoutePolicy`）が歩く経路と
    # 同じ見方で測るためです（理由は `_table` の上に実測つきで）。
    from hexa.pathfind import fuel_weight as _fuel_weight
    weights = {index: _fuel_weight(day_steps, fuels[index])
               for index in touring}
    left = {spot.position: min(spot.stocks, len(touring))
            for spot in problem.spots}
    known = {spot.position for spot in problem.spots}
    routes: dict[int, list[int]] = {
        index: [spot for spot in (start or {}).get(index, [])
                if spot in known]
        for index in touring
    }
    # 土台のぶんを在庫から引く（同じ車の重複は落とす）
    for index in touring:
        seen: list[int] = []
        for spot in routes[index]:
            if spot in seen or left.get(spot, 0) <= 0:
                continue
            seen.append(spot)
            left[spot] -= 1
        routes[index] = seen
    for spot in [p for p, c in left.items() if c <= 0]:
        del left[spot]
    steps_used = {
        index: route_cost(finder, road_status, positions[index],
                          routes[index], cache)
        for index in touring
    }
    fuel_used = {
        index: _route_fuel(finder, road_status, positions[index],
                           routes[index], cache)
        for index in touring
    }
    budget = {index: fuels[index] + problem.fuel_limit * refuels
              for index in touring}
    brand_of = {spot.position: spot.brand for spot in problem.spots}
    fresh = set()
    if brand_weight > 0:
        taken = acquired_brands or set()
        fresh = {brand for brand in brand_of.values() if brand not in taken}
        for spots in routes.values():
            for spot in spots:
                fresh.discard(brand_of.get(spot))

    def hop(origin: int, goal: int):
        """2点の間の (歩数, 燃料)。届かなければ None。"""

        table = _table(finder, origin, road_status, cache)
        if goal not in table.steps:
            return None
        return (table.steps[goal], table.fuel[goal])

    for index in touring:
        if (steps_used[index] > day_steps
                or fuel_used[index] > budget[index]):
            # 土台がすでに歩数か燃料を超えているので、この車は白紙から
            for spot in routes[index]:
                left[spot] = left.get(spot, 0) + 1
            routes[index] = []
            steps_used[index] = 0
            fuel_used[index] = 0

    while True:
        best = None
        for index in touring:
            route = routes[index]
            start = positions[index]
            for spot, count in left.items():
                if count <= 0 or spot in route:
                    continue
                for at in range(len(route) + 1):
                    before = route[at - 1] if at > 0 else start
                    first = hop(before, spot)
                    if first is None:
                        continue
                    add_steps, add_fuel = first
                    if at < len(route):
                        after = route[at]
                        second = hop(spot, after)
                        old = hop(before, after)
                        if second is None or old is None:
                            continue
                        add_steps += second[0] - old[0]
                        add_fuel += second[1] - old[1]
                    if steps_used[index] + add_steps > day_steps:
                        continue
                    if fuel_used[index] + add_fuel > budget[index]:
                        continue
                    score = add_steps - count * stock_weight
                    if brand_weight > 0 and brand_of.get(spot) in fresh:
                        score -= brand_weight
                    if best is None or score < best[0]:
                        best = (score, index, at, spot, add_steps, add_fuel)
        if best is None:
            break
        _score, index, at, spot, add_steps, add_fuel = best
        routes[index] = routes[index][:at] + [spot] + routes[index][at:]
        steps_used[index] += add_steps
        fuel_used[index] += add_fuel
        left[spot] -= 1
        if left[spot] <= 0:
            del left[spot]
        fresh.discard(brand_of.get(spot))
    return routes


# ---------------------------------------------------------------------------
# 通りがかりを数える設計
#
# **なぜ必要か**
#
# 車は歩いた先の店を無条件で拾います（泊まる必要はありません）。実測で、
# 玉のうち通りがかりで拾ったものが8〜42%ありました:
#
#   第1958 (32x32) 1日目 玉85 のうち通りがかり10（12%）
#   第1958 (32x32) 3日目 玉59 のうち通りがかり**25（42%）**
#   第1891 (24x24) 1日目 玉49 のうち通りがかり12（24%）
#   第1981 (16x16) 3日目 玉20 のうち通りがかり 4（20%）
#
# 店から店へ行く途中で踏む店は、第1958の3日目で**平均0.62店**、
# 46%の区間で1店以上です。
#
# ところが `design_day` と `design_insert` は**歩数だけ**で組み立てて
# いて、この拾いを数えていません。だから次の見立てがどれも外れました:
#
#   「訪問数を増やす」  → 計画の訪問を60→107にしても玉54→50
#   「歩数を縮める」    → 歩556→535に縮めたら玉59→**55**
#
# ここでは差し込みの損得を正しく計算します。
#
#   得 = |店(a→x) ∪ 店(x→b) ∪ {x}| − |店(a→b)|   （新たに踏む店の数）
#   損 = 増える歩数
#
# 経路の途中で踏む店は `PathTable.path_spots` がビット列で返すので、
# 和も差も1命令で出ます。


def _leg_spots(finder, road_status, origin, target, spot_bits, cache):
    """origin から target へ行く途中で踏む店（ビット列）。target も含む。"""

    table = _table(finder, origin, road_status, cache)
    if target not in table.steps:
        return None
    return table.path_spots(target, spot_bits)


def design_passing(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    brand_weight: float = 0.0,
    acquired_brands: set[int] | None = None,
    refuels: int = 0,
    per_step: bool = True,
) -> dict[int, list[int]]:
    """**通りがかりで踏む店まで数えて**順路を組み立てる。

    `design_insert` と同じ差し込み方ですが、得の数え方が違います。
    「その店に着く」ではなく「その差し込みで**新たに踏む店が何軒
    増えるか**」で選びます。

    車ごとに「区間ごとに踏む店」をビット列で持ち、前からの和と
    後ろからの和を別に持ちます。そうすると「ある区間を差し替えた
    ときの全体の和」が1命令で出ます（候補ごとに数え直しません）。

    `per_step` が真なら「1歩あたり何軒増えるか」で選びます。偽なら
    「増える軒数が多い順、同じなら歩数が短い順」です。
    """

    touring = [i for i, kind in enumerate(kinds) if kind == AgentKind.TOURING]
    if not touring:
        return {}
    cache: dict[int, object] = {}
    # 探索器が持っている1つの表を使います（覚え書きを温かく保つため)。
    spot_bits = finder.spot_bits
    brand_of = {spot.position: spot.brand for spot in problem.spots}
    left = {spot.position: min(spot.stocks, len(touring))
            for spot in problem.spots}
    routes: dict[int, list[int]] = {index: [] for index in touring}
    steps_used = {index: 0 for index in touring}
    fuel_used = {index: 0 for index in touring}
    # 区間ごとに踏む店（区間 j は routes[j-1] から routes[j] へ行く分。
    # j=0 は出発点から routes[0] へ行く分）
    legs: dict[int, list[int]] = {index: [] for index in touring}
    # 前からの和 pre[j] = legs[0:j] の和 / 後ろからの和 suf[j] = legs[j:] の和
    pre: dict[int, list[int]] = {index: [0] for index in touring}
    suf: dict[int, list[int]] = {index: [0] for index in touring}
    budget = {index: fuels[index] + problem.fuel_limit * refuels
              for index in touring}
    fresh: set[int] = set()
    if brand_weight > 0:
        got = acquired_brands or set()
        fresh = {brand for brand in brand_of.values() if brand not in got}

    def rebuild(index: int) -> None:
        row = legs[index]
        count = len(row)
        before = [0] * (count + 1)
        for at in range(count):
            before[at + 1] = before[at] | row[at]
        after = [0] * (count + 1)
        for at in range(count - 1, -1, -1):
            after[at] = after[at + 1] | row[at]
        pre[index] = before
        suf[index] = after

    def hop(origin: int, goal: int):
        table = _table(finder, origin, road_status, cache)
        if goal not in table.steps:
            return None
        return (table.steps[goal], table.fuel[goal],
                table.path_spots(goal, spot_bits))

    while True:
        best = None
        for index in touring:
            route = routes[index]
            start_at = positions[index]
            row = legs[index]
            before_union = pre[index]
            after_union = suf[index]
            mine = after_union[0]
            room_steps = day_steps - steps_used[index]
            room_fuel = budget[index] - fuel_used[index]
            for spot, stock_left in left.items():
                if stock_left <= 0 or spot in route:
                    continue
                for at in range(len(route) + 1):
                    source = route[at - 1] if at > 0 else start_at
                    first = hop(source, spot)
                    if first is None:
                        continue
                    add_steps, add_fuel, bits = first
                    if at < len(route):
                        # 区間 at を「source→spot」と「spot→route[at]」に割る
                        second = hop(spot, route[at])
                        if second is None:
                            continue
                        old = hop(source, route[at])
                        if old is None:
                            continue
                        add_steps = first[0] + second[0] - old[0]
                        add_fuel = first[1] + second[1] - old[1]
                        bits = first[2] | second[2]
                        rest = before_union[at] | after_union[at + 1]
                    else:
                        rest = mine
                    if add_steps > room_steps or add_fuel > room_fuel:
                        continue
                    gained = bin((rest | bits) & ~rest).count('1')
                    if gained <= 0:
                        continue
                    bonus = brand_weight if (
                        brand_weight > 0 and brand_of.get(spot) in fresh
                    ) else 0.0
                    if per_step:
                        score = max(1, add_steps) / (gained + bonus)
                    else:
                        score = -(gained + bonus) * 1000 + add_steps
                    if best is None or score < best[0]:
                        best = (score, index, at, spot, add_steps, add_fuel,
                                bits, first[2],
                                second[2] if at < len(route) else 0)
        if best is None:
            break
        (_score, index, at, spot, add_steps, add_fuel, bits,
         head_bits, tail_bits) = best
        routes[index] = routes[index][:at] + [spot] + routes[index][at:]
        if at < len(legs[index]):
            legs[index][at] = tail_bits
            legs[index].insert(at, head_bits)
        else:
            legs[index].append(head_bits)
        rebuild(index)
        steps_used[index] += add_steps
        fuel_used[index] += add_fuel
        # **踏む店は全部「使った」と数えます。** 通りがかりでも在庫は減ります。
        for where, bit in list(left.items()):
            if bits & spot_bits.get(where, 0):
                left[where] -= 1
                if left[where] <= 0:
                    del left[where]
        fresh.discard(brand_of.get(spot))
    return routes


# **系列をまず全部押さえる設計。**
#
# 第1627（16x16・系列10・巡回3台）の盤面を数えると、**系列10個の
# うち7個は店が1軒しかありません**（系列0はマス0だけ、1は10、
# 4は79、6は144、7は192、8は224、9は245）。つまり日別を満点に
# するには、毎日その7マスを必ず踏むしかありません。
#
# いまの設計はどれも「玉数を増やしながら系列も見る」形なので、
# 高い系列が後ろに回って切り捨てられます。実戦では2・3・4日目に
# 系列0と1（どちらも盤面の左上の角）を落として、日別36でした
# （1位は44）。
#
# ここでは順番を逆にします。
#
#   1. まだ押さえていない系列のうち、**いちばん高くつくもの**から
#      先に場所を取る（取りにくいものを後回しにすると落ちる）
#   2. 全系列ぶん取り終わってから、余った歩数で店を足す
#
# 日別累計は玉数より上位の得点なので、全系列そろう案が候補に1つ
# でもあれば採点が選びます。
def design_cover(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    acquired_brands: set[int] | None = None,
    refuels: int = 1,
    fill: bool = True,
) -> dict[int, list[int]]:
    """**全系列を押さえてから**、余りで店を足す順路を返す。"""

    touring = [i for i, kind in enumerate(kinds) if kind == AgentKind.TOURING]
    if not touring:
        return {}
    cache: dict[object, object] = {}
    # **車ごとの経路の重み。** 実行が歩く経路と同じ見方で測るため
    #（理由は `_table` の上に実測つきで）。`DESIGN_FUEL_WEIGHT` が
    # False のときは `_table` が 0 に戻すので、値は使われません。
    from hexa.pathfind import fuel_weight as _fuel_weight
    weights = {index: _fuel_weight(day_steps, fuels[index])
               for index in touring}
    spots_by_brand: dict[int, list[int]] = {}
    for spot in problem.spots:
        spots_by_brand.setdefault(spot.brand, []).append(spot.position)
    routes: dict[int, list[int]] = {index: [] for index in touring}
    used_steps = {index: 0 for index in touring}
    used_fuel = {index: 0 for index in touring}
    budget = {index: fuels[index] + problem.fuel_limit * refuels
              for index in touring}
    left = {spot.position: spot.stocks for spot in problem.spots}
    taken: set[int] = set()

    def best_insert(spot: int):
        """その店を入れるのにいちばん安い (増える歩数, 車, 位置, 燃料)。"""

        found = None
        for index in touring:
            line = routes[index]
            if spot in line:
                continue
            room_steps = day_steps - used_steps[index]
            room_fuel = budget[index] - used_fuel[index]
            weight = weights[index]
            for at in range(len(line) + 1):
                source = line[at - 1] if at > 0 else positions[index]
                head = _table(finder, source, road_status, cache, weight)
                gap = head.steps.get(spot)
                if gap is None:
                    continue
                add_steps = gap
                add_fuel = head.fuel[spot]
                if at < len(line):
                    mid = _table(finder, spot, road_status, cache, weight)
                    tail = mid.steps.get(line[at])
                    was = head.steps.get(line[at])
                    if tail is None or was is None:
                        continue
                    add_steps = gap + tail - was
                    add_fuel = (head.fuel[spot] + mid.fuel[line[at]]
                                - head.fuel[line[at]])
                if add_steps > room_steps or add_fuel > room_fuel:
                    continue
                if found is None or add_steps < found[0]:
                    found = (add_steps, index, at, add_fuel)
        return found

    # 1. 取りにくい系列から場所を取る。
    while True:
        wants = [brand for brand in spots_by_brand if brand not in taken]
        if not wants:
            break
        hardest = None
        for brand in wants:
            cheapest = None
            for spot in spots_by_brand[brand]:
                if left.get(spot, 0) <= 0:
                    continue
                got = best_insert(spot)
                if got is None:
                    continue
                if cheapest is None or got[0] < cheapest[0]:
                    cheapest = (got[0], got[1], got[2], got[3], spot)
            if cheapest is None:
                continue
            # **高くつく系列を先に取ります。**
            if hardest is None or cheapest[0] > hardest[0]:
                hardest = (cheapest[0], cheapest[1], cheapest[2],
                           cheapest[3], cheapest[4], brand)
        if hardest is None:
            break
        add_steps, index, at, add_fuel, spot, brand = hardest
        routes[index] = routes[index][:at] + [spot] + routes[index][at:]
        used_steps[index] += add_steps
        used_fuel[index] += add_fuel
        left[spot] = left.get(spot, 0) - 1
        taken.add(brand)

    if not fill:
        return routes
    # 2. 余った歩数で、玉を増やす店を足す。
    try:
        return design_insert(
            problem, finder, kinds, positions, fuels, road_status, day_steps,
            stock_weight=0.0, brand_weight=0.0,
            acquired_brands=acquired_brands, refuels=refuels,
            start={index: list(line) for index, line in routes.items()},
        )
    except Exception:
        return routes


# **補給車を動かさない設計（ハブ）。**
#
# 第1703（32x32・巡回6台・9日・平均80.9歩）の数字:
#
#   1位は1台1日11軒。81歩で11軒なら、1台は盤面の3分の1を掃くような
#   大きな周回をしている。
#   81歩の周回には燃料90が要る（1歩1.11燃料）。タンクは113で1.26日分
#   しかないので、**6台が毎日1回ずつ満タンにしてもらう**必要がある。
#   補給車の持ち歩は81歩なので、6回の移動に**1回13.5歩**しか使えない。
#
# つまり往復して配るのでは間に合いません。**6台の周回が共通して通る
# 一点に補給車を置けば、補給車は移動ゼロで6台を満タンにできます。**
#
# 第37回の解法紹介でも4チームがこの形を書いています:
#
#   佐世保11「補給車はマップ中に点在させ、基本は動かさないものとする」
#   仙台名取41「補給車をマップの中央付近に固定することで不要な移動を
#     減らし、数を必要最低限にする」
#   秋田23「補給車を中心とした円があると考え、その円からはみ出さない
#     範囲で巡回車がうどんを回収する」
#   釧路27「他の巡回車と補給車は別々のスポットに駐在する」
#
# **置く場所は店にします。** そうすれば巡回車は普通の目的地として
# 寄るので、順路に特別な印を入れる必要がありません（順路は店の並び
# なので、店でない待ち合わせ点は表せない）。おまけに玉も入ります。
def pick_hub(problem: Problem, finder: PathFinder, road_status) -> int | None:
    """**店のなかで、ほかの店からの距離の合計がいちばん小さい店。**

    そこに補給車を置けば、どの車の周回からも寄りやすくなります。
    """

    cache: dict[int, object] = {}
    best = None
    for spot in problem.spots:
        table = _table(finder, spot.position, road_status, cache)
        total = 0
        reach = 0
        for other in problem.spots:
            gap = table.steps.get(other.position)
            if gap is None:
                continue
            total += gap
            reach += 1
        if reach < len(problem.spots):
            # 届かない店があるなら中心ではありません。
            total += (len(problem.spots) - reach) * problem.cell_count
        if best is None or total < best[0]:
            best = (total, spot.position)
    return best[1] if best else None


def design_hub(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    acquired_brands: set[int] | None = None,
    hub: int | None = None,
    refuels: int = 1,
) -> tuple[dict[int, list[int]], int | None]:
    """ハブを経由する順路と、補給車を置く店を返す。

    戻り値は (車番号 -> 店の並び, ハブの店)。

    各巡回車の順路は「ハブから始めて、そこから差し込みで組む」形に
    します。ハブで満タンになるので、そのあとの歩と燃料をまるごと
    周回に使えます。
    """

    touring = [i for i, kind in enumerate(kinds) if kind == AgentKind.TOURING]
    supply = [i for i, kind in enumerate(kinds) if kind == AgentKind.SUPPLY]
    if not touring or not supply:
        return {}, None
    if hub is None:
        hub = pick_hub(problem, finder, road_status)
    if hub is None:
        return {}, None
    cache: dict[int, object] = {}
    # ハブに届かない車は、ふつうに組みます。
    start: dict[int, list[int]] = {}
    for index in touring:
        table = _table(finder, positions[index], road_status, cache)
        gap = table.steps.get(hub)
        need = table.fuel.get(hub)
        if gap is None or gap > day_steps or need is None or need > fuels[index]:
            start[index] = []
            continue
        start[index] = [hub]
    if not any(start.values()):
        return {}, None
    # ハブを先頭に置いた形から、差し込みで埋めます。ハブで満タンに
    # なるぶんを `refuels` で見込みます。
    routes = design_insert(
        problem, finder, kinds, positions, fuels, road_status, day_steps,
        stock_weight=0.0, brand_weight=0.0,
        acquired_brands=acquired_brands, refuels=refuels, start=start,
    )
    return routes, hub


# **系列を車に割り振って、毎日同じ周回を繰り返す設計（縄張り）。**
#
# 本選1回戦（16x16・燃料40・巡回3台・5日）の数字から出てきた形です。
#
#   日別の満点は10系列×5日=50。ところが**1位でも44**で、誰も満点を
#   取れていません（過去問の24x24・32x32では1位が29試合すべて満点）。
#   つまりこの盤面は**日別が勝負**です。
#
#   第6試合の店を座標に直すと、縄張りが成立します:
#     144(行9列0)・192(行12列0)・224(行14列0)・245(行15列5)
#       → 列0に縦に密集。1台で4系列
#     0(行0列0)・10(行0列10) → 最上行。1台で2系列
#     79(行4列15) と複数店の系列2・3・5 → 右側。1台で4系列
#   各台の周回は20〜30歩で、32〜48歩に収まります。**毎日同じ周回を
#   繰り返せば日別は満点**です。1位の44は、燃料の配達が足りない分の
#   損だと説明がつきます。
#
#   こちらは37しか取れていません。原因は日ごとに組み直すことで、
#   1日目に燃料を使い切り、2〜3日目に2台が動けなくなるからです
#   （実測の燃料残 [40,40,2,3] → [33,40,2,3] → [27,40,2,40]）。
#
# ここでは**系列を車に割り振り**、各車の周回が短くなるようにします。
# 割り振りは試合中ずっと同じなので、毎日同じ形に戻れます。
def _tour_cost(finder, road_status, start, spots, cache, closed=True):
    """近い順につないだ周回の歩数。`closed` なら出発点に戻る分も足す。"""

    if not spots:
        return 0
    here = start
    left = list(spots)
    total = 0
    while left:
        table = _table(finder, here, road_status, cache)
        best = None
        for spot in left:
            gap = table.steps.get(spot)
            if gap is None:
                continue
            if best is None or gap < best[0]:
                best = (gap, spot)
        if best is None:
            # 届かない店が残った。大きな値で表す。
            return total + len(left) * 9999
        total += best[0]
        here = best[1]
        left.remove(best[1])
    if closed:
        table = _table(finder, here, road_status, cache)
        back = table.steps.get(start)
        total += back if back is not None else 9999
    return total


def _tour_order(finder, road_status, start, spots, cache):
    """近い順につないだ並び。"""

    here = start
    left = list(spots)
    out = []
    while left:
        table = _table(finder, here, road_status, cache)
        best = None
        for spot in left:
            gap = table.steps.get(spot)
            if gap is None:
                continue
            if best is None or gap < best[0]:
                best = (gap, spot)
        if best is None:
            break
        out.append(best[1])
        here = best[1]
        left.remove(best[1])
    return out


def pick_territory(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    road_status: dict[int, RoadStatus],
    rounds: int = 60,
) -> dict[int, list[int]]:
    """**系列を巡回車に割り振る。** 戻り値は 車番号 -> 店の並び。

    系列ごとに使う店も決めます。各車の周回がいちばん長いものを
    短くする向きに、系列の移動と交換を繰り返します（min-max）。
    """

    touring = [i for i, kind in enumerate(kinds) if kind == AgentKind.TOURING]
    if not touring:
        return {}
    cache: dict[int, object] = {}
    spots_by_brand: dict[int, list[int]] = {}
    for spot in problem.spots:
        spots_by_brand.setdefault(spot.brand, []).append(spot.position)
    brands = sorted(spots_by_brand)

    # 最初の割り振り: 系列ごとに「いちばん近い車とその店」を選ぶ。
    owner: dict[int, int] = {}
    place: dict[int, int] = {}
    for brand in brands:
        best = None
        for index in touring:
            table = _table(finder, positions[index], road_status, cache)
            for spot in spots_by_brand[brand]:
                gap = table.steps.get(spot)
                if gap is None:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, index, spot)
        if best is None:
            continue
        owner[brand] = best[1]
        place[brand] = best[2]

    def mine(index):
        return [place[b] for b in brands if owner.get(b) == index]

    def cost_of(index):
        return _tour_cost(finder, road_status, positions[index], mine(index),
                          cache)

    # **いちばん長い車から系列を移す／交換する。**
    for _round in range(rounds):
        costs = {index: cost_of(index) for index in touring}
        worst = max(costs, key=lambda i: costs[i])
        before = max(costs.values())
        moved = False
        for brand in [b for b in brands if owner.get(b) == worst]:
            for index in touring:
                if index == worst:
                    continue
                for spot in spots_by_brand[brand]:
                    was_owner, was_place = owner[brand], place[brand]
                    owner[brand], place[brand] = index, spot
                    after = max(cost_of(i) for i in touring)
                    if after < before:
                        before = after
                        moved = True
                    else:
                        owner[brand], place[brand] = was_owner, was_place
            if moved:
                break
        if not moved:
            break
    return {index: _tour_order(finder, road_status, positions[index],
                               mine(index), cache)
            for index in touring}


def design_territory(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    acquired_brands: set[int] | None = None,
    refuels: int = 1,
    fill: bool = True,
) -> dict[int, list[int]]:
    """縄張りの周回を、今日の歩数と燃料に合わせて切った順路を返す。"""

    touring = [i for i, kind in enumerate(kinds) if kind == AgentKind.TOURING]
    if not touring:
        return {}
    cache: dict[int, object] = {}
    plan = pick_territory(problem, finder, kinds, positions, road_status)
    routes: dict[int, list[int]] = {}
    for index in touring:
        here = positions[index]
        used = burn = 0
        room = fuels[index] + problem.fuel_limit * refuels
        line = []
        for spot in plan.get(index, []):
            table = _table(finder, here, road_status, cache)
            gap = table.steps.get(spot)
            if gap is None:
                continue
            need = table.fuel[spot]
            if used + gap > day_steps or burn + need > room:
                break
            used += gap
            burn += need
            line.append(spot)
            here = spot
        routes[index] = line
    if not any(routes.values()):
        return {}
    if not fill:
        return routes
    try:
        return design_insert(
            problem, finder, kinds, positions, fuels, road_status, day_steps,
            stock_weight=0.0, brand_weight=0.0,
            acquired_brands=acquired_brands, refuels=refuels,
            start={index: list(line) for index, line in routes.items()},
        )
    except Exception:
        return routes


# **明日の陣形で終わる設計**（`design_formation`）。
#
# 部分集合DP（`an/cap.py`）で「その日に取れる系列数の厳密な上限」を
# 出すと、本選1回戦第1試合（10系列・5日・満点50）は
#
#   | 条件 | 上限 |
#   |---|---|
#   | 出発点が自由・巡回3台・燃料無視 | **50（毎日満点）** |
#   | 出発点が自由・巡回2台 | 42 |
#   | 実際の終わり方・燃料無視 | 39 |
#   | 実際の終わり方・朝の燃料で絞る | 31 |
#
# **届かない系列は1つもありません。** 差は到達性ではなく
# **終わり方（位置と燃料）**だけです。1台は1日あたり約2系列ぶんの
# 価値があります（3台10系列 / 2台8系列 / 1台5系列）。
#
# そこで**先に「明日どこで朝を迎えるか」を決めてから**、その日の
# 順路をそこで終わるように組みます。終点は**店**にします。順路は
# 店の並びなので店以外は表せませんし、店で夜を越せば翌朝の1歩目で
# 1玉入ります（在庫は毎日戻る）。
#
# 選び方は貪欲です。系列の集合をふやす問題は劣モジュラなので、
# 「いちばん多くふえる店」から順に取るだけで上限の 1-1/e 以上が
# 取れます。全組み合わせ（32店の6台で10億通り）は試せません。
def pick_formation(
    problem: Problem,
    finder: PathFinder,
    road_status: dict[int, RoadStatus],
    cars: int,
    next_steps: int,
    margin: float = 1.0,
) -> list[int]:
    """**明日いちばん多くの系列に届く、店の組**を cars 個返す。

    各店について「明日の歩数で届く系列の集合」を出し、和がいちばん
    大きくなるように貪欲に選びます。同点なら「届く店の数」が多い
    ほうを採ります。
    """

    if cars <= 0 or next_steps <= 0:
        return []
    limit = max(1, int(next_steps * margin))
    brands = sorted({spot.brand for spot in problem.spots})
    bit = {brand: 1 << index for index, brand in enumerate(brands)}
    cache: dict[int, object] = {}
    # 店ごとの「明日届く系列」と「明日届く店の数」。
    reach: dict[int, tuple[int, int]] = {}
    for spot in problem.spots:
        table = _table(finder, spot.position, road_status, cache)
        mask = bit[spot.brand]
        count = 1
        for other in problem.spots:
            gap = table.steps.get(other.position)
            if gap is not None and gap <= limit:
                mask |= bit[other.brand]
                count += 1
        reach[spot.position] = (mask, count)
    picked: list[int] = []
    have = 0
    while len(picked) < cars:
        best = None
        for position, (mask, count) in reach.items():
            if position in picked:
                continue
            gain = (have | mask).bit_count() - have.bit_count()
            score = (gain, count)
            if best is None or score > best[0]:
                best = (score, position, mask)
        if best is None:
            break
        picked.append(best[1])
        have |= best[2]
    return picked


def design_formation(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    next_steps: int = 0,
    acquired_brands: set[int] | None = None,
    refuels: int = 1,
    margin: float = 1.0,
    with_supply: bool = False,
):
    """**明日の陣形で終わる順路**を返す。理由は `pick_formation` の上に。

    まず `design_cover` で今日の順路を組み、そのうえで各巡回車の
    終点を陣形の店にします。入らなければ末尾の店を落として入れ直し、
    それでも入らなければその車は陣形をあきらめます（今日の玉を
    捨てないため）。
    """

    touring = [index for index, kind in enumerate(kinds)
               if kind == AgentKind.TOURING]
    empty = ({}, []) if with_supply else {}
    if not touring or next_steps <= 0:
        return empty
    wanted = pick_formation(problem, finder, road_status, len(touring),
                            next_steps, margin)
    if not wanted:
        return empty
    try:
        routes = design_cover(
            problem, finder, kinds, positions, fuels, road_status, day_steps,
            acquired_brands=acquired_brands, refuels=refuels)
    except Exception:
        return empty
    if not routes:
        return empty
    cache: dict[int, object] = {}
    # **近い車に近い終点を渡します。** 遠い組み合わせにすると、
    # 陣形のために今日の店を落とす数が増えます。
    free = list(wanted)
    pairs = []
    for index in touring:
        here = routes.get(index) or []
        tail = here[-1] if here else positions[index]
        table = _table(finder, tail, road_status, cache)
        for target in free:
            gap = table.steps.get(target)
            pairs.append((gap if gap is not None else 10 ** 9, index, target))
    pairs.sort()
    taken_car: set[int] = set()
    taken_spot: set[int] = set()
    assigned: dict[int, int] = {}
    for _, index, target in pairs:
        if index in taken_car or target in taken_spot:
            continue
        assigned[index] = target
        taken_car.add(index)
        taken_spot.add(target)
    for index, target in assigned.items():
        line = list(routes.get(index) or [])
        if line and line[-1] == target:
            continue
        budget_fuel = fuels[index] + problem.fuel_limit * refuels
        head = [spot for spot in line if spot != target]
        found = None
        # **末尾から1店ずつ落として、入るところを探します。**
        # 全部落としてもよい（終点の店だけでも1玉は入る）。
        for drop in range(len(head) + 1):
            trial = head[:len(head) - drop] + [target]
            if (route_cost(finder, road_status, positions[index], trial,
                           cache) <= day_steps
                    and _route_fuel(finder, road_status, positions[index],
                                    trial, cache) <= budget_fuel):
                found = trial
                break
        if found is not None:
            routes[index] = found
    if not with_supply:
        return routes
    # **補給車にも陣形を教えます。**
    #
    # 補給車は燃料を1滴も使わないので、制約は歩数だけです。そして
    # `simulator` の補給は**歩0の行動段階でも走る**ので、補給車と
    # 同じマスで夜を越した巡回車は、翌日1歩も動かないうちに満タんに
    # なります。つまり「夜をどこで越すか」を合わせるだけで、
    # 1台ぶんの満タンが毎朝ただで手に入ります。
    #
    # 迎える順は**いま燃料が少ない車から**で、待ち受け先はその車の
    # 陣形の店にします。順番だけ合っていれば必ず拾えます
    #（燃料が切れた車はその場で止まって待つ）。
    order = sorted(touring, key=lambda index: fuels[index])
    supply_route = [(assigned.get(index), index) for index in order]
    return routes, supply_route


# **夜に補給車のマスへ集まる設計**（`design_night`）。
#
# 準決勝32x32（第1791641357試合・巡回6台・燃料80・1日96歩）を
# 厳密な上限で切り分けると、こうなりました:
#
#   | 条件 | 玉の上限 |
#   |---|---|
#   | 実際の朝の燃料・補給なし | **243** ← こちらは240で張り付き |
#   | 実際の朝の燃料・1台1日0.5回の補給 | 356 |
#   | **毎朝満タン・補給なし** | **373** |
#   | 毎朝満タン・1台1日0.5回の補給 | 373（増えない） |
#   | 1位の実績 | 304 |
#
# **毎朝満タンで始まれば、1日の途中の補給は1回も要りません。**
# 燃料80は96歩をまかなえます（平地なら1歩に燃料0.5）。日中に
# 落ち合うのは、巡回車の歩と補給車の歩を両方使う高い買い物でした。
#
# 補給は `simulator` の**歩0の行動段階でも走る**ので、補給車と
# 同じマスで夜を越した巡回車は、翌日1歩も動かないうちに満タンに
# なります。しかも同じマスに何台いてもかまいません。つまり
# **全車が夜に補給車のマスへ集まれば、6台ぶんの満タンがただで
# 手に入ります**。
#
# 集まる先は**店**にします。順路は店の並びなので店以外は表せず、
# 店なら在庫のぶんだけ玉も入ります。
def pick_night_cell(
    problem: Problem,
    finder: PathFinder,
    road_status: dict[int, RoadStatus],
    positions: list[int],
    touring: list[int],
    day_steps: int,
) -> int | None:
    """**夜に集まる店**を選ぶ。

    「どの店からも近く、どの車からも届く」店です。周回の帰り道が
    短くなるよう、店どうしの距離の合計がいちばん小さいものから
    選びます。
    """

    cache: dict[int, object] = {}
    best = None
    for spot in problem.spots:
        table = _table(finder, spot.position, road_status, cache)
        far = 0
        reach = 0
        ok = True
        for other in problem.spots:
            gap = table.steps.get(other.position)
            if gap is None:
                continue
            far += gap
            reach += 1
        for index in touring:
            car = _table(finder, positions[index], road_status, cache)
            gap = car.steps.get(spot.position)
            if gap is None or gap > day_steps:
                ok = False
                break
        if not ok:
            continue
        score = (-reach, far)
        if best is None or score < best[0]:
            best = (score, spot.position)
    return best[1] if best is not None else None


def design_night(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    acquired_brands: set[int] | None = None,
    refuels: int = 0,
    cell: int | None = None,
):
    """**全車が夜に同じ店へ集まる順路**と、補給車を置く店を返す。

    戻り値は (車番号 -> 店の並び, 集まる店)。理由は
    `pick_night_cell` の上に。
    """

    touring = [index for index, kind in enumerate(kinds)
               if kind == AgentKind.TOURING]
    supply = [index for index, kind in enumerate(kinds)
              if kind == AgentKind.SUPPLY]
    if not touring or not supply:
        return {}, None
    if cell is None:
        cell = pick_night_cell(problem, finder, road_status, positions,
                               touring, day_steps)
    if cell is None:
        return {}, None
    # **系列網羅の上に組みます。** `design_insert` は系列を見ないので、
    # 1軒しかない系列を落として日別で即負けします（日別は玉数より
    # 上位の得点）。準決勝32x32は22系列32店で、1軒の系列が
    # いくつもありました。
    try:
        routes = design_cover(
            problem, finder, kinds, positions, fuels, road_status, day_steps,
            acquired_brands=acquired_brands, refuels=refuels)
    except Exception:
        return {}, None
    if not routes:
        return {}, None
    cache: dict[int, object] = {}
    for index in touring:
        line = [spot for spot in (routes.get(index) or []) if spot != cell]
        budget_fuel = fuels[index] + problem.fuel_limit * refuels
        found = None
        # **末尾から1店ずつ落として、集まる店が入るところを探します。**
        for drop in range(len(line) + 1):
            trial = line[:len(line) - drop] + [cell]
            if (route_cost(finder, road_status, positions[index], trial,
                           cache) <= day_steps
                    and _route_fuel(finder, road_status, positions[index],
                                    trial, cache) <= budget_fuel):
                found = trial
                break
        if found is not None:
            routes[index] = found
    return routes, cell


# **1台ぶんの順路を厳密に解く設計**（`design_exact`）。
#
# 準決勝32x32（第1791641357試合）の数字で、縛っているものが
# 順に変わっていきました。
#
#   1. 燃料の配達 … 補給なしの上限243に対して240で張り付き
#   2. 待ち        … 夜合流を強いると待ちが16〜40%→8〜17%に下がる
#   3. **順路の質** … それでも玉は231。1店あたりの移動が
#                     こちら6.25回に対し、上限の形は4.4回
#
# 1台が1日に踏める店の最大数は、**(踏んだ店の集合, いまの店) ごとに
# 最短歩数と最小燃料を持つ探索**で厳密に解けます。予算を超えた枝を
# 捨てるので、32店でも1台1〜2秒で解けます（`an/maxballs.py` で実測）。
#
# 車のあいだは順番に解きます（1台目が取った店は在庫を減らし、
# 尽きた店は2台目の候補から外す）。車のなかは厳密、車のあいだは
# 貪欲、という形です。
#
# `end` を渡すと「**最後にそのマスへ戻る**」ことを条件にします
# （夜に補給車のマスへ集まる形）。その代償は小さく、32x32の上限は
# 373→354で、1位の304より上に残ります。
EXACT_FRONTIER_CAP = 120_000


def exact_route(
    finder: PathFinder,
    road_status: dict[int, RoadStatus],
    start: int,
    spots: list[int],
    day_steps: int,
    fuel: int,
    end: int | None = None,
    cache: dict | None = None,
) -> list[int]:
    """**1台ぶんの最善の順路**（踏める店がいちばん多い並び）を返す。

    同じ数なら歩数の短いほうを選びます。店を2度は数えません。
    """

    if cache is None:
        cache = {}
    count = len(spots)
    if count == 0 or day_steps <= 0:
        return []
    big = 10 ** 9
    table_from = _table(finder, start, road_status, cache)
    first_steps = [table_from.steps.get(spot, big) for spot in spots]
    first_fuel = [table_from.fuel.get(spot, big) for spot in spots]
    between_steps = []
    between_fuel = []
    for spot in spots:
        table = _table(finder, spot, road_status, cache)
        between_steps.append([table.steps.get(other, big) for other in spots])
        between_fuel.append([table.fuel.get(other, big) for other in spots])
    if end is None:
        back_steps = [0] * count
        back_fuel = [0] * count
    else:
        back_steps = []
        back_fuel = []
        for index, spot in enumerate(spots):
            table = _table(finder, spot, road_status, cache)
            back_steps.append(table.steps.get(end, big))
            back_fuel.append(table.fuel.get(end, big))

    def fits(index, steps, used):
        return (steps + back_steps[index] <= day_steps
                and used + back_fuel[index] <= fuel)

    # 状態: (踏んだ集合, いまの店) -> (歩数, 燃料, 1つ前の状態)
    frontier: dict[tuple[int, int], tuple[int, int, object]] = {}
    for index in range(count):
        if (first_steps[index] <= day_steps and first_fuel[index] <= fuel
                and fits(index, first_steps[index], first_fuel[index])):
            frontier[(1 << index, index)] = (first_steps[index],
                                             first_fuel[index], None)
    best_state = None
    best_depth = 0
    depth = 1
    while frontier:
        # その深さでいちばん歩数の短い状態を覚えておきます。
        for key, value in frontier.items():
            if depth > best_depth or (depth == best_depth
                                      and best_state is not None
                                      and value[0] < best_state[1][0]):
                if depth > best_depth:
                    best_depth = depth
                    best_state = (key, value)
                elif value[0] < best_state[1][0]:
                    best_state = (key, value)
        nxt: dict[tuple[int, int], tuple[int, int, object]] = {}
        for (mask, here), (steps, used, _) in frontier.items():
            row_steps = between_steps[here]
            row_fuel = between_fuel[here]
            for index in range(count):
                if (mask >> index) & 1:
                    continue
                go = steps + row_steps[index]
                if go > day_steps:
                    continue
                burn = used + row_fuel[index]
                if burn > fuel or not fits(index, go, burn):
                    continue
                key = (mask | (1 << index), index)
                got = nxt.get(key)
                if got is None or go < got[0]:
                    nxt[key] = (go, burn, (mask, here))
        if not nxt:
            break
        if len(nxt) > EXACT_FRONTIER_CAP:
            keep = sorted(nxt.items(), key=lambda item: item[1][0])
            nxt = dict(keep[:EXACT_FRONTIER_CAP])
        # さかのぼるために、1つ前の深さを残しておきます。
        previous = frontier
        frontier = nxt
        depth += 1
        _EXACT_LAYERS.append(previous)
    if best_state is None:
        return []
    # さかのぼって並びを作ります。
    line: list[int] = []
    key, value = best_state
    layers = list(_EXACT_LAYERS)
    _EXACT_LAYERS.clear()
    current_key = key
    current = value
    while current is not None:
        mask, here = current_key
        line.append(spots[here])
        parent = current[2]
        if parent is None:
            break
        found = None
        for layer in reversed(layers):
            got = layer.get(parent)
            if got is not None:
                found = got
                break
        if found is None:
            break
        current_key = parent
        current = found
    line.reverse()
    if end is not None and (not line or line[-1] != end):
        line.append(end)
    return line


_EXACT_LAYERS: list[dict] = []


def design_exact(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    acquired_brands: set[int] | None = None,
    refuels: int = 0,
    end: int | None = None,
) -> dict[int, list[int]]:
    """**車ごとに厳密な順路**を組む。理由は上に。"""

    touring = [index for index, kind in enumerate(kinds)
               if kind == AgentKind.TOURING]
    if not touring:
        return {}
    left = {spot.position: spot.stocks for spot in problem.spots}
    cache: dict[int, object] = {}
    routes: dict[int, list[int]] = {}
    # **燃料の少ない車から先に選びます。** 選べる店が少ないので、
    # 後回しにすると行き先が残っていません。
    order = sorted(touring, key=lambda index: fuels[index])
    for index in order:
        spots = [position for position, stock in left.items() if stock > 0]
        if not spots:
            routes[index] = []
            continue
        budget = fuels[index] + problem.fuel_limit * refuels
        line = exact_route(finder, road_status, positions[index], spots,
                           day_steps, budget, end=end, cache=cache)
        routes[index] = line
        for position in line:
            if left.get(position, 0) > 0:
                left[position] -= 1
    return routes


# **決まった店をそのまま、並びだけ厳密にして、余った歩で足す**
#（`polish_exact`）。
#
# `design_exact` は「軒数」しか見ないので、1軒しかない系列を落とし、
# 日別で即負けします（日別は玉数より上位の得点）。準決勝32x32は
# 22系列32店で、1軒しかない系列がいくつもあります。
#
# そこで**店の集合は変えず、並びだけ最善にします**。集合が同じなら
# 系列も同じなので、日別は1つも減りません。並べ替えで余った歩で
# 店を足せば、玉だけが増えます。
#
# 1店あたりの移動は、こちらの設計が6.25回、厳密な形が4.4回でした。
# つまり並びだけで3割ぶんの歩が浮きます。
#
# 12店までなら部分集合DP（2^12×12＝5万）で厳密に解けます。
# それを超える車は、近い順につないだあと2-optで詰めます。
EXACT_ORDER_LIMIT = 13


def _best_order(finder, road_status, start, spots, cache):
    """**その店全部を回る最短の並び**（12店までは厳密）。"""

    count = len(spots)
    if count <= 1:
        return list(spots)
    if count > EXACT_ORDER_LIMIT:
        return improve_order(finder, road_status, start, list(spots), rounds=3)
    big = 10 ** 9
    table = _table(finder, start, road_status, cache)
    first = [table.steps.get(spot, big) for spot in spots]
    between = []
    for spot in spots:
        row = _table(finder, spot, road_status, cache)
        between.append([row.steps.get(other, big) for other in spots])
    size = 1 << count
    cost = [[big] * count for _ in range(size)]
    back = [[-1] * count for _ in range(size)]
    for index in range(count):
        cost[1 << index][index] = first[index]
    for mask in range(size):
        row = cost[mask]
        for here in range(count):
            now = row[here]
            if now >= big or not (mask >> here) & 1:
                continue
            line = between[here]
            for index in range(count):
                if (mask >> index) & 1:
                    continue
                target = mask | (1 << index)
                go = now + line[index]
                if go < cost[target][index]:
                    cost[target][index] = go
                    back[target][index] = here
    full = size - 1
    last = min(range(count), key=lambda index: cost[full][index])
    if cost[full][last] >= big:
        return improve_order(finder, road_status, start, list(spots), rounds=3)
    order = []
    mask = full
    here = last
    while here >= 0:
        order.append(spots[here])
        previous = back[mask][here]
        mask ^= (1 << here)
        here = previous
    order.reverse()
    return order


def polish_exact(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    routes: dict[int, list[int]],
    refuels: int = 0,
    end: int | None = None,
) -> dict[int, list[int]]:
    """**並びを最善にして、余った歩で店を足す。** 系列は減りません。"""

    touring = [index for index, kind in enumerate(kinds)
               if kind == AgentKind.TOURING]
    if not touring:
        return {}
    cache: dict[int, object] = {}
    out: dict[int, list[int]] = {}
    taken: dict[int, int] = {}
    for index in touring:
        for spot in routes.get(index) or []:
            taken[spot] = taken.get(spot, 0) + 1
    for index in touring:
        line = list(routes.get(index) or [])
        if not line:
            out[index] = []
            continue
        budget_fuel = fuels[index] + problem.fuel_limit * refuels
        order = _best_order(finder, road_status, positions[index], line, cache)
        if (route_cost(finder, road_status, positions[index], order, cache)
                > day_steps
                or _route_fuel(finder, road_status, positions[index], order,
                               cache) > budget_fuel):
            # 並べ替えで入らなくなることはないはずですが、念のため。
            order = line
        out[index] = order
    # **余った歩で店を足します。** 在庫の残っている店から、
    # いちばん安く入るものを繰り返し足します。
    left = {spot.position: spot.stocks for spot in problem.spots}
    for spot, count in taken.items():
        if spot in left:
            left[spot] = max(0, left[spot] - count)
    for index in touring:
        line = out[index]
        budget_fuel = fuels[index] + problem.fuel_limit * refuels
        while True:
            now_steps = route_cost(finder, road_status, positions[index],
                                   line, cache)
            now_fuel = _route_fuel(finder, road_status, positions[index],
                                   line, cache)
            best = None
            for spot, stock in left.items():
                if stock <= 0 or spot in line:
                    continue
                for at in range(len(line) + 1):
                    trial = line[:at] + [spot] + line[at:]
                    if end is not None and trial[-1] != end:
                        continue
                    add = route_cost(finder, road_status, positions[index],
                                     trial, cache) - now_steps
                    if add < 0:
                        continue
                    if now_steps + add > day_steps:
                        continue
                    burn = _route_fuel(finder, road_status, positions[index],
                                       trial, cache)
                    if burn > budget_fuel:
                        continue
                    if best is None or add < best[0]:
                        best = (add, trial, spot)
            if best is None:
                break
            line = best[1]
            left[best[2]] -= 1
        out[index] = line
    return out


# **逆回りループ**（宇部高専・解説集 p15下・玉51）。
#
# > フィールド全体のうどん屋を巡る大きなループを1本作り、巡回車は
# > 右回り、補給車は左回りに回す。こうすることで巡回車が1周する間に
# > 補給車の数の2倍の回数の燃料補給を行うことができる。
#
# **なぜこれが要るのか。** 準決勝32x32で「補給車が何歩使って何回
# 配れたか」を数えると、こうでした（`an/refuelmix.py`）。
#
#   | 日 | 補給の回数 | 配った燃料 | 補給車が使った歩 |
#   |---|---|---|---|
#   | 1 | **1回** | 50 | 59/64 |
#   | 2 | **1回** | 80 | 79/80 |
#   | 3 | 5回 | 248 | 95/96 |
#
# **補給車は1日の歩をほぼ全部使って動いているのに、初日と2日目は1回
# しか配れていません。** 必要なのは1日120燃料（6台×20。厳密な上限から、
# 1位に並ぶのに要るのは1台1日20燃料）。そして3日目の朝の燃料は
# [80, 22, 0, 1, 69, 74] で、2台が1歩も動けませんでした。
#
# 原因は**追いかけているから**です。迎え先を「その車のいる場所」に
# すると相対速度がほぼ0で、1日かけて1回しか会えません。先回りして
# 待つ形（`SUPPLY_INTERCEPT`）も作りましたが、補給車をそのマスに縛る
# ので救援に行けなくなり、29試合では負けました（318→370）。
#
# **逆回りなら相対速度が倍になります。** 追いかけるのでも待つのでも
# なく、向かい合うので、縛らずに出会う回数が増えます。
LOOP_IMPROVE_ROUNDS = 3


def design_loop(
    problem: Problem,
    finder: PathFinder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    acquired_brands: set[int] | None = None,
    refuels: int = 1,
) -> tuple[dict[int, list[int]], list[int]]:
    """**1本の大きなループを作り、巡回車は順方向・補給車は逆方向に回す。**

    戻り値は (巡回車の順路, 補給車が通るマスの並び) です。理由と実測は
    `LOOP_IMPROVE_ROUNDS` の上に。

    ループは全部の店を1周する並びです。巡回車はいまいる場所にいちばん
    近いところから順方向に、入るだけ取ります。補給車は同じループを
    **逆方向**にたどります（補給車は燃料を使わないので歩数だけが縛り）。
    """

    touring = [index for index, kind in enumerate(kinds)
               if kind == AgentKind.TOURING]
    supply = [index for index, kind in enumerate(kinds)
              if kind == AgentKind.SUPPLY]
    if not touring:
        return {}, []
    cache: dict[object, object] = {}
    spots = [spot.position for spot in problem.spots]
    if not spots:
        return {}, []

    # 1. ループを作ります。いちばん近い店へ繋ぐ形から始めて、並べ替えで
    #    縮めます（`improve_order` は始点からの一本道を縮めるので、
    #    ここでは1台目の出発点を始点にしてから輪として使います）。
    start = positions[touring[0]]
    loop = improve_order(finder, road_status, start, list(spots),
                         rounds=LOOP_IMPROVE_ROUNDS)
    if not loop:
        return {}, []
    size = len(loop)
    where = {cell: number for number, cell in enumerate(loop)}

    # 2. 巡回車は、いまいる場所にいちばん近いところから順方向へ。
    #    受け持ちが重ならないよう、出発位置を台数で散らします。
    stock = {spot.position: spot.stocks for spot in problem.spots}
    routes: dict[int, list[int]] = {}
    for order, index in enumerate(touring):
        table = _table(finder, positions[index], road_status, cache)
        best = None
        for number, cell in enumerate(loop):
            steps = table.steps.get(cell)
            if steps is None:
                continue
            if best is None or steps < best[0]:
                best = (steps, number)
        if best is None:
            routes[index] = []
            continue
        head = best[1]
        budget_fuel = fuels[index] + problem.fuel_limit * refuels
        line: list[int] = []
        for step in range(size):
            cell = loop[(head + step) % size]
            if stock.get(cell, 0) <= 0:
                continue
            trial = line + [cell]
            if route_cost(finder, road_status, positions[index], trial,
                          cache) > day_steps:
                break
            if _route_fuel(finder, road_status, positions[index], trial,
                           cache) > budget_fuel:
                break
            line = trial
            stock[cell] = stock.get(cell, 0) - 1
        routes[index] = line

    # 3. 補給車は同じループを**逆方向**に。燃料は使わないので歩数だけ。
    cells: list[int] = []
    if supply:
        index = supply[0]
        table = _table(finder, positions[index], road_status, cache)
        best = None
        for number, cell in enumerate(loop):
            steps = table.steps.get(cell)
            if steps is None:
                continue
            if best is None or steps < best[0]:
                best = (steps, number)
        if best is not None:
            head = best[1]
            for step in range(size):
                cell = loop[(head - step) % size]
                trial = cells + [cell]
                if route_cost(finder, road_status, positions[index], trial,
                              cache) > day_steps:
                    break
                cells = trial
    return routes, cells
