"""「次にどの店へ行くか」ではなく「どういう作戦で回るか」を選ばせる。

**なぜ必要か**

練習場の実戦データでは、全チームが種類数・日別累計で並び、**勝負は玉数だけ**でした。
そして私たちは毎回15〜25%負けていました。調べると、差は経路の細かさではなく
**「どの店をどの順で回るか」という計画の質**です。

これまでのAIは「いま一番よさそうな1店」を都度選ぶだけで、数店を連ねた周回を
考えていませんでした。しかも選べるのが「店」なので、学習しても
「この店は良い/悪い」しか覚えられず、**作戦の良し悪しを覚える余地がありません。**

**どう変えるか**

作戦を「何を重く見るか」の数値の組み合わせで表し、そこから数店を連ねた
道すじを作ります。AIはその中から選び、実際に取れた玉数が報酬になります。

名前の付いた作戦は出発点にすぎません。学習の中でこの数値が変異し、
良かったものが残るので、**最初になかった作戦が生まれます。**
選択肢を固定すると、そこが天井になってしまいます。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from hexa.pathfind import PathFinder, fuel_weight
from hexa.problem import Problem
from hexa.rules import RoadStatus, Terrain

# 1つの作戦で連ねる店の数。到着ごとに組み直すので、3店先まで見えれば実質つながる。
# 長くすると1回の判断が重くなり、学習量を稼げなくなる。
MAX_ROUTE_LENGTH = 4

# 先読みを長くするマス数の範囲。24x24（576マス）だけが対象です。
#
# 16x16（256マス）では逆効果でした（日別138→134）。店が近いので、
# 長く見ると目先の1店を取り逃します。
#
# 32x32（1024マス）はさらに悪く、**1案が2倍重くなるうえ質も落ちます**。
#
#   4店: 54案/8秒（1案150ms）最良玉55
#   6店: 40案/8秒（1案203ms）最良玉52
#   8店: 28案/8秒（1案286ms）最良玉47
#
# 店の数は24で同じでも、マップが広いと経路の計算が重くなるためです。
BIG_MAP_CELLS = 500
BIG_MAP_MAX_CELLS = 700
BIG_MAP_ROUTE_LENGTH = 8
# 燃料上限がこれ以下のときだけ、先読みを長くします。
#
# 実測（実戦の渋滞をそのまま使って再現）:
#   200以下: 1228=82% / 1233=94% / 1533=87%
#   **130以下**: 1228=88% / 1233=92% / 1535=94%  ← 採用
#   使わない: 1233=90%
#
# 燃料124・213の試合でも8店は不利でした（1533で玉265→261、
# 1535で427→424）。候補が379〜1,470件しか作れないためです。
BIG_MAP_FUEL_LIMIT = 130

# 道が混んでいる日は、もう1店先まで見ます。
#
# **なぜ日によって変えるか**
#
# 渋滞すると店どうしが遠くなり（実戦927の2日目は中央値118歩）、
# 3店先まででは「次の次」が見えません。4店先まで見ると、
# その試合で日別が満点になり玉数も増えました（120/457 → 125/467）。
#
# ところが渋滞のない日に4店先まで見ると、かえって悪くなります
# （9試合で日別の満点が7/8から6/8に減少）。先が見えるぶん
# 「いまは良いが後で外れる」予測に引きずられるためだと見ています。
#
# そこで、**その日に渋滞している道路が多いときだけ**長く見ます。
# 渋滞した日は、以前は1店多く見ていました。通常が4店になったので、
# さらに増やすと1案あたりが重くなりすぎます（候補が作れず逆効果）。
# いまは同じ4店で、切り替えは実質ありません。
# 歩数が厳しい日に先を長く見る仕組み。**実測で入れませんでした。**
#
# 実戦のログで達成率は「上限1玉あたり使える歩数」でほぼ決まります
# （3〜5歩で52%、9〜11歩で80%）。そこで厳しい日だけ5店先まで
# 見るようにしたところ、厳しい試合では玉1008→1024（+16）でしたが、
# **通常の試合で2489→2429（-60）と大きく損**をしました。
# 閾値を4.0歩まで下げても2472で届きません。
#
# 0 にすると常に MAX_ROUTE_LENGTH（4店）を使います。
TIGHT_STEPS_PER_NOODLE = 0.0
TIGHT_ROUTE_LENGTH = 5
JAMMED_ROUTE_LENGTH = 4
# 渋滞した道路がこの割合を超えた日だけ、先を長く見ます。
#
# 実戦927（32x32・渋滞の多い試合）での実測:
#
#   3店固定        日別121 / 玉457
#   渋滞25%で切替  日別123 / 玉443
#   渋滞12%で切替  日別123 / 玉461  ← 採用
#
# 25%だと切り替わる日が少なく、玉数だけ落ちました。
# 12%なら渋滞が始まった日から長く見られます。
JAMMED_SHARE_FOR_LONGER = 0.12

# 作戦を表す観点の数。
CRITERIA_COUNT = 8

# 作戦の出発点。学習の中で変異して、名前のない作戦に育っていく。
#
#   0 近さ          大きいほど近い店を選ぶ
#   1 燃料の安さ    大きいほど燃料を使わない道を選ぶ
#   2 在庫の多さ    大きいほど太い店を選ぶ
#   3 未獲得の系列  大きいほど未獲得を優先する
#   4 今日初の系列  大きいほど今日まだ取っていない系列を優先する
#   5 道路を避ける  正なら道路を避け、負なら道路を好む
#   6 敵の幹線      負なら敵がよく使う道路をあえて踏む（妨害）
#   7 明日の混雑    正なら明日混みそうな道を避ける
SEED_RECIPES: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("最寄り",     (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("在庫重視",   (0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("未獲得系列", (0.2, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0)),
    ("燃料節約",   (0.2, 1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0)),
    ("穴場狙い",   (-1.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("密集地周回", (1.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("妨害寄り",   (0.5, 0.0, 0.3, 0.0, 0.0, -0.6, -1.0, 0.0)),
    ("渋滞回避",   (0.6, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0)),
)

RECIPE_POOL_LIMIT = 48


@dataclass
class Route:
    """1つの作戦で作った道すじ。"""

    recipe_index: int = 0
    spots: list[int] = field(default_factory=list)
    steps: int = 0
    fuel: int = 0
    road_cells: int = 0
    new_brands: int = 0
    new_today: int = 0
    stock_total: int = 0
    congestion: float = 0.0
    enemy_road: float = 0.0
    # その道すじが渋滞したときに、どれだけ遠回りになるか。
    # 道路の幅が1マスしかなく、横が池、というような場所を通っていると
    # ここが跳ね上がります。1マス渋滞しただけで全スポットへの合計が
    # 19ステップ伸びる地形を実際に確認しました（1日43ステップの試合で）。
    detour: int = 0

    @property
    def target(self) -> int | None:
        return self.spots[0] if self.spots else None


def default_recipes() -> list[list[float]]:
    return [list(weights) for _name, weights in SEED_RECIPES]


def mutate_recipe(recipe: list[float], rng: random.Random, strength: float = 0.35) -> list[float]:
    """作戦を少し変えた新しい作戦を作る。ここから名前のない作戦が生まれる。"""

    return [max(-2.0, min(2.0, value + rng.gauss(0.0, strength))) for value in recipe]


def blend_recipes(first: list[float], second: list[float], rng: random.Random) -> list[float]:
    """2つの作戦を混ぜて、間の作戦を作る。"""

    ratio = rng.random()
    return [
        max(-2.0, min(2.0, a * ratio + b * (1.0 - ratio)))
        for a, b in zip(first, second)
    ]


def _score(recipe, spot, steps, need, acquired, seen, road_share, enemy, congestion,
           left=None):
    """作戦に従って店を採点する。小さいほど良い。

    1案を作るのに18万回呼ばれるので、ここの速さがそのまま
    1日に試せる候補の数になります。tuple と zip と sum をやめ、
    8項目を直接書き下しています（実測で2倍以上速くなります）。
    """

    # 0.0333333 のような近似を使うと元の式と微妙にずれます。
    # 速さより一致を優先し、割り算のまま書きます。
    total = recipe[0] * (steps / 30.0) + recipe[1] * (need / 30.0)
    w = recipe[2]
    if w:
        # 元の在庫と**残っている在庫**の両方を見ます。
        #
        # 残りだけで見ると、取り切りかけた店を避けすぎて逆効果でした。
        # 元の在庫だけだと、取り切った店と手つかずの店が同じ評価になります。
        # 実戦のログでは在庫4以上の店の43%に1台以下しか来ておらず、
        # 取り逃しの93%が「届いたのに行かなかった」でした。
        value = spot.stocks if left is None else (spot.stocks * 0.3 + left * 0.7)
        total -= w * (value / 7.0)
    w = recipe[3]
    if w and spot.brand in acquired:
        total += w
    w = recipe[4]
    if w and spot.brand in seen:
        total += w
    w = recipe[5]
    if w:
        total += w * road_share
    w = recipe[6]
    if w:
        total -= w * enemy
    w = recipe[7]
    if w:
        total += w * congestion
    return total


# 1回の枝分かれで残す道すじの本数。
#
# 数本を並行して伸ばせば良い順番が見つかるはず、と考えて実装しましたが、
# **実測では逆効果でした**（未見6問での玉数）。
#
#   入れ替えあり・枝1本 : 228.2  ← 採用
#   入れ替えあり・枝3本 : 219.3
#   入れ替えなし・枝1本 : 220.5
#   入れ替えなし・枝3本 : 211.0
#
# 枝分かれが効かないのは、枝を選ぶ基準が「作戦の評価値の合計」であって
# 玉数そのものではないためだと見ています。評価値の良い順番が
# 玉数の多い順番とは限りません。
#
# 仕組みは残してあるので、選ぶ基準を変えて試すときは 2 以上にしてください。
BEAM_WIDTH = 1


def _build_one(
    problem: Problem,
    finder: PathFinder,
    road_status: dict[int, RoadStatus],
    start: int,
    fuel: int,
    remaining: int,
    recipe: list[float],
    recipe_index: int,
    available: dict[int, int],
    acquired_brands: set[int],
    brands_today: set[int],
    road_cells: set[int],
    congestion_risk: dict[int, float] | None,
    road_heat: dict[int, float] | None,
    max_length: int = MAX_ROUTE_LENGTH,
) -> Route:
    """1つの作戦で、数本の道すじを並行して伸ばし、一番良かったものを返す。

    道路の踏み方（share / enemy / jam）を出すには経路を歩き直す必要があり、
    候補の店ぜんぶで毎回やると1試走が貪欲法の5倍近くになります。
    まず歩き直さずに絞り込み、上位だけ正確に測り直します。
    """

    needs_walk = any(abs(recipe[index]) > 1e-9 for index in (5, 6, 7))
    shortlist = 5
    # 在庫が残っている店だけを毎回調べる。全店を舐め直す必要はない。
    candidates_cache = [
        spot for spot in problem.spots if available.get(spot.position, 0) > 0
    ]
    if not candidates_cache:
        return Route(recipe_index=recipe_index)

    # 1本の枝: (作戦の評価値の合計, 道すじ, 現在地, 残りステップ, 残り燃料, 訪問済み, 見た系列)
    beams: list[tuple[float, Route, int, int, int, frozenset, frozenset]] = [
        (0.0, Route(recipe_index=recipe_index), start, remaining, fuel,
         frozenset(), frozenset(brands_today))
    ]

    for _ in range(max_length):
        grown: list[tuple[float, Route, int, int, int, frozenset, frozenset]] = []
        for score_sum, route, position, left_steps, left_fuel, used, seen in beams:
            table = finder.table(position, road_status, fuel_weight(left_steps, left_fuel))
            # 内側の最も熱い輪なので、属性参照と辞書引きを変数に出しておく。
            step_of = table.steps
            fuel_of = table.fuel
            rough: list[tuple[float, object, int, int]] = []
            for spot in candidates_cache:
                position_of_spot = spot.position
                if position_of_spot in used:
                    continue
                steps = step_of.get(position_of_spot)
                if steps is None or steps > left_steps:
                    continue
                need = fuel_of[position_of_spot]
                if need > left_fuel:
                    continue
                key = _score(recipe, spot, steps, need, acquired_brands, seen,
                             0.0, 0.0, 0.0, available.get(position_of_spot))
                rough.append((key, spot, steps, need))
            if not rough:
                continue

            rough.sort(key=lambda item: item[0])
            picks: list[tuple[float, object, int, int, float, float, float]] = []
            if needs_walk:
                for key, spot, steps, need in rough[:shortlist]:
                    share, enemy, jam = _walk_stats(
                        table, spot.position, finder, problem, road_cells,
                        congestion_risk, road_heat,
                    )
                    exact = _score(recipe, spot, steps, need, acquired_brands, seen,
                                   share, enemy, jam, available.get(spot.position))
                    picks.append((exact, spot, steps, need, share, enemy, jam))
                picks.sort(key=lambda item: item[0])
            else:
                for key, spot, steps, need in rough[:BEAM_WIDTH]:
                    picks.append((key, spot, steps, need, 0.0, 0.0, 0.0))

            for exact, spot, steps, need, share, enemy, jam in picks[:BEAM_WIDTH]:
                grown.append((
                    score_sum + exact,
                    _extended(route, spot, steps, need, share, enemy, jam,
                              available, acquired_brands, seen),
                    spot.position,
                    left_steps - steps,
                    left_fuel - need,
                    used | {spot.position},
                    seen | {spot.brand},
                ))

        if not grown:
            break
        # 評価値の良い順に上位だけ残す。
        grown.sort(key=lambda item: item[0])
        beams = grown[:BEAM_WIDTH]

    # 同じ歩数ならより多くの店を回れた道すじを採る。
    best = min(
        beams,
        key=lambda item: (-len(item[1].spots), item[0]),
    )
    return best[1]


def _extended(route, spot, steps, need, share, enemy, jam,
              available, acquired_brands, seen) -> Route:
    """既存の道すじに1店を足した新しい道すじを作る（元は壊さない）。"""

    grown = Route(
        recipe_index=route.recipe_index,
        spots=list(route.spots) + [spot.position],
        steps=route.steps + steps,
        fuel=route.fuel + need,
        road_cells=route.road_cells + int(share * max(1, steps)),
        new_brands=route.new_brands + (1 if spot.brand not in acquired_brands else 0),
        new_today=route.new_today + (1 if spot.brand not in seen else 0),
        stock_total=route.stock_total + available.get(spot.position, 0),
        congestion=route.congestion + jam,
        enemy_road=route.enemy_road + enemy,
        detour=route.detour,
    )
    return grown


def _walk_stats(table, target, finder, problem, road_cells, congestion_risk, road_heat):
    """その道すじが道路をどれだけ踏むか、敵の幹線をどれだけ踏むかを数える。"""

    try:
        directions = table.directions_to(target)
    except KeyError:
        return (0.0, 0.0, 0.0)
    if not directions:
        return (0.0, 0.0, 0.0)
    position = table.origin
    roads = 0
    enemy = 0.0
    jam = 0.0
    for direction in directions:
        if position in road_cells:
            roads += 1
            if road_heat:
                enemy += road_heat.get(position, 0.0)
            if congestion_risk:
                jam += congestion_risk.get(position, 0.0)
        for candidate_direction, candidate in finder.neighbors[position]:
            if candidate_direction == direction:
                position = candidate
                break
    count = len(directions)
    return (roads / count, enemy, jam / count)


def reorder_route(
    finder: PathFinder,
    road_status: dict[int, RoadStatus],
    start: int,
    spots: list[int],
    left_steps: int,
    left_fuel: int,
) -> tuple[list[int], int, int]:
    """訪問する店の順番を並べ替えて、移動の合計を短くする。

    **なぜ必要か**

    作戦は「次に一番良い店」を継ぎ足して道すじを作ります。
    1店ずつ見ると最善でも、**並び全体としては遠回りになる**ことがあります。

    実戦の9日ぶんを調べると、実際に回った順番の合計2574歩のうち
    **249歩（9%）が並べ替えだけで削れました。** 1日目の車1は
    110歩が64歩まで縮みます。浮いた歩数でもう1〜2店回れます。

    ここでは隣り合う2店を入れ替える操作を、改善が止まるまで繰り返します。
    店は最大3つなので全部試しても6通りですが、将来長くしても効くように
    一般の形で書いてあります。

    歩数と燃料が足りない並びは採用しません。
    **順番を変えるだけなので、取れる系列も玉数も変わりません。**
    日別累計を落とす心配がないのが、この方法の良いところです。
    """

    if len(spots) < 2:
        steps, fuel = _walk_cost(finder, road_status, start, spots, left_steps, left_fuel)
        return (list(spots), steps, fuel)

    best_order = list(spots)
    best_steps, best_fuel = _walk_cost(
        finder, road_status, start, best_order, left_steps, left_fuel
    )
    if best_steps < 0:
        return (best_order, best_steps, best_fuel)

    for _round in range(len(spots)):
        improved = False
        for first in range(len(best_order) - 1):
            for second in range(first + 1, len(best_order)):
                trial = list(best_order)
                trial[first], trial[second] = trial[second], trial[first]
                steps, fuel = _walk_cost(
                    finder, road_status, start, trial, left_steps, left_fuel
                )
                if steps < 0:
                    continue
                if steps < best_steps:
                    best_order, best_steps, best_fuel = trial, steps, fuel
                    improved = True
        if not improved:
            break
    return (best_order, best_steps, best_fuel)


def _walk_cost(
    finder: PathFinder,
    road_status: dict[int, RoadStatus],
    start: int,
    spots: list[int],
    left_steps: int,
    left_fuel: int,
) -> tuple[int, int]:
    """その順番で回ったときの合計歩数と燃料。届かなければ (-1, -1)。"""

    position = start
    steps = 0
    fuel = 0
    for target in spots:
        table = finder.table(
            position, road_status, fuel_weight(left_steps - steps, left_fuel - fuel)
        )
        leg = table.steps.get(target)
        if leg is None:
            return (-1, -1)
        need = table.fuel[target]
        if steps + leg > left_steps or fuel + need > left_fuel:
            return (-1, -1)
        steps += leg
        fuel += need
        position = target
    return (steps, fuel)


def road_cells_of(finder: PathFinder, problem: Problem) -> set[int]:
    """道路セルの一覧。問題ごとに1回だけ作って使い回す。

    毎回作り直すと terrain_at を何百万回も呼ぶことになり、
    1試走の時間の4割がここで消えていました。
    """

    cached = getattr(finder, "_road_cells", None)
    if cached is None:
        cached = {
            cell for cell in range(problem.cell_count)
            if problem.terrain_at(cell) == Terrain.ROAD
        }
        finder._road_cells = cached
    return cached


def _jammed_status(finder: PathFinder, problem: Problem) -> dict[int, RoadStatus]:
    """全ての道路が渋滞した場合の状態。道すじの脆さを測るために使う。"""

    cached = getattr(finder, "_jam_status", None)
    if cached is None:
        cached = {cell: RoadStatus.JAMMED for cell in road_cells_of(finder, problem)}
        finder._jam_status = cached
    return cached


def route_length_for(problem, road_status, remaining: int = 0,
                     available: dict | None = None, touring: int = 1) -> int:
    """その日に何店先まで見るか。

    **「上限1玉あたりに使える歩数」で決めます。**

    実戦のログ133日分を集計すると、達成率はこの値でほぼ決まります。

        3〜5歩/玉:  83日 達成52%
        6〜8歩/玉:  26日 達成74%
        9〜11歩/玉: 17日 達成80%

    歩数が足りない日ほど、先を長く見て「近くを取り切る順番」を
    見つける価値があります。実測（歩数の厳しい5試合）:

        3店: 日別486 / 玉1021
        4店: 日別485 / 玉1008
        5店: 日別487 / 玉1024  ← 厳しい日はこれ

    余裕のある日は4店が勝つので（12試合で玉2444→2489）、
    日によって変えます。
    """

    # **大きいマップでは長く見ます。**
    #
    # 24x24 の試合で、先読みを4店から8店に伸ばすと玉数が伸びました。
    #
    #   1228: 83% → 90%
    #   1233: 87% → 91%
    #   1320: 95% → 99%
    #
    # 店が散っているぶん、順番を工夫する余地が大きいためです。
    # 一方 16x16 では逆効果でした（1219の日別が138→134）。
    # 店が近いので、長く見ると目先の1店を取り逃します。
    # **24x24 で、燃料が厳しい試合だけ先を長く見ます。**
    #
    # 燃料が潤沢な試合（上限361）では4店が勝ちました。
    #
    #   4店: 77案/8秒（1案105ms）最良玉64  ← 燃料361の試合
    #   8店: 30案/8秒（1案271ms）最良玉61
    #
    # 燃料が厳しい試合（上限110〜120）では8店が勝ちます。
    # 動ける歩数が限られるぶん、順番を練る価値が大きいためです。
    base = MAX_ROUTE_LENGTH
    if (BIG_MAP_CELLS <= problem.cell_count <= BIG_MAP_MAX_CELLS
            and problem.fuel_limit <= BIG_MAP_FUEL_LIMIT):
        base = BIG_MAP_ROUTE_LENGTH

    if remaining <= 0 or not available:
        return base
    cap = 0
    for spot in problem.spots:
        left = available.get(spot.position, 0)
        if left > 0:
            cap += min(left, touring)
    if cap <= 0:
        return base
    per_noodle = remaining * touring / cap
    if per_noodle <= TIGHT_STEPS_PER_NOODLE:
        return TIGHT_ROUTE_LENGTH
    return base


def build_routes(
    problem: Problem,
    finder: PathFinder,
    road_status: dict[int, RoadStatus],
    position: int,
    fuel: int,
    remaining: int,
    available: dict[int, int],
    acquired_brands: set[int],
    brands_today: set[int],
    recipes: list[list[float]],
    congestion_risk: dict[int, float] | None = None,
    road_heat: dict[int, float] | None = None,
) -> list[Route]:
    """作戦ごとに道すじを1本ずつ作る。同じ結果になったものは1つにまとめる。"""

    road_cells = road_cells_of(finder, problem)
    jam_status = _jammed_status(finder, problem)
    # 必要になったときだけ引く。道路を踏まない計画ばかりなら一度も引きません。
    jammed_table = None
    plain_table = None
    routes: list[Route] = []
    seen: set[tuple[int, ...]] = set()
    # 渋滞が多い日は、もう1店先まで見ます。
    max_length = route_length_for(
        problem, road_status, remaining, available,
        max(1, sum(1 for _ in problem.agent_positions) - 1),
    )
    for index, recipe in enumerate(recipes):
        route = _build_one(
            problem, finder, road_status, position, fuel, remaining,
            recipe, index, available, acquired_brands, brands_today,
            road_cells, congestion_risk, road_heat, max_length,
        )
        if not route.spots:
            continue
        signature = tuple(route.spots)
        # 並べ替えて移動を短くする。取れる店は変わらないので、
        # 系列も玉数も落ちません。浮いた歩数がそのまま余裕になります。
        if len(route.spots) >= 2:
            order, steps, fuel_used = reorder_route(
                finder, road_status, position, route.spots, remaining, fuel
            )
            if steps >= 0 and steps < route.steps:
                route.spots = order
                route.steps = steps
                route.fuel = fuel_used

        if signature in seen:
            continue
        seen.add(signature)

        # 同じ道すじを「全道路が渋滞している」と仮定して引き直し、
        # どれだけ遠回りになるかを測る。1マス幅の道路に頼った計画は
        # ここが大きくなるので、学習が避けるか備えるかを判断できる。
        #
        # ただし道路を1マスも踏まない道すじは、渋滞しても遠回りにならないので
        # measure する必要がありません。ここを省くと経路探索が1回減ります。
        if route.road_cells > 0:
            if jammed_table is None:
                jammed_table = finder.table(position, jam_status, 0)
            if plain_table is None:
                plain_table = finder.table(position, road_status, 0)
            first = route.spots[0]
            normal = plain_table.steps.get(first)
            risky = jammed_table.steps.get(first)
            if normal is not None and risky is not None:
                route.detour = max(0, risky - normal)
        routes.append(route)
    return routes
