"""**歩数だけを見て、訪問数の多い順路を探す。**

`hexa/improve.py` の磨きは、1手ごとに1日ぶんを丸ごと再現します。
正確ですが重く、32x32では**1手3ms**かかります。1日57秒では2万手も
回せず、6台で32店を回る形はまだ荒いままで終わります。

実測（第1958の32x32・7体・3日目・6000手）の内訳:

    plan_day            79%
      action_phase        60%
    dict.get             20%
    経路表を引く         （作り直しは1手0.3回だけ。キャッシュは効いている）

一方、歩数だけを見る検査（`allocate.route_cost`）は**10マイクロ秒**で、
300倍速い。さらに「車の出発点と店」のあいだの距離を**1日1回だけ**
表にしておけば、1手は配列引きだけになります。

そこでこの module は、

    1. 距離表と燃料表を作る（節点 = 巡回車の出発点 + 店）
    2. 安いところへ差し込みながら組み立てる
    3. 崩して埋め直す（ruin & recreate）＋入れ替え・区間反転で磨く
    4. 良かった形を**何通りか**返す

をやります。返した形は `improve.py` の磨きの**出発点**として渡すので、
採用の判断は今までどおり公式シミュレータに通した正確な採点が行います。
**この module の見立てが外れても、点数が下がることはありません。**

見ないもの（だから出発点にしか使いません）:

  * 補給車がいつ誰に追いつくか（燃料は「満タン◯回ぶん」で概算）
  * 同じ日に複数台が同じ店へ来たときの取り合いの時刻
  * 店に着いた順で在庫が尽きること

乱数は使いません。同じ入力なら必ず同じ答えになります。
"""

from __future__ import annotations

from hexa.problem import Problem
from hexa.rules import AgentKind, RoadStatus

# 届かないことを表す大きな値。
FAR = 10 ** 9

# 崩すときに抜く店の数（この並びを順に使います）。乱数は使いません。
RUIN_SIZES = (1, 2, 3, 2, 4, 1, 3, 5, 2, 6)


class DayMatrix:
    """1日ぶんの距離表。節点は「巡回車の出発点」と「店」です。

    節点番号は 0..cars-1 が巡回車の出発点、cars.. が店です。
    `steps[i][j]` が i から j までの最短歩数、`fuel[i][j]` がそのときの
    燃料です。届かないところは `FAR` を入れます。
    """

    __slots__ = ("cars", "spots", "steps", "fuel", "stock", "brand",
                 "index_of", "spot_at")

    def __init__(self, problem: Problem, finder,
                 kinds: tuple[AgentKind, ...], positions: list[int],
                 road_status: dict[int, RoadStatus]):
        touring = [index for index, kind in enumerate(kinds)
                   if kind == AgentKind.TOURING]
        self.cars = touring
        places = [positions[index] for index in touring]
        spots = [spot.position for spot in problem.spots]
        self.spots = spots
        nodes = places + spots
        count = len(nodes)
        self.index_of = {}
        for number, spot in enumerate(spots):
            self.index_of[spot] = len(places) + number
        self.spot_at = {len(places) + number: spot
                        for number, spot in enumerate(spots)}
        self.steps = [[FAR] * count for _ in range(count)]
        self.fuel = [[FAR] * count for _ in range(count)]
        for origin in range(count):
            table = finder.table(nodes[origin], road_status)
            row = self.steps[origin]
            burn = self.fuel[origin]
            for goal in range(count):
                where = nodes[goal]
                got = table.steps.get(where)
                if got is None:
                    continue
                row[goal] = got
                burn[goal] = table.fuel[where]
        limit = len(touring)
        self.stock = {self.index_of[spot.position]:
                      min(spot.stocks, limit) for spot in problem.spots}
        self.brand = {self.index_of[spot.position]: spot.brand
                      for spot in problem.spots}


class Plan:
    """巡回車ごとの順路（節点番号の並び）と、その歩数・燃料。"""

    __slots__ = ("matrix", "routes", "steps", "burn", "left", "budget",
                 "fuel_budget")

    def __init__(self, matrix: DayMatrix, budget: int,
                 fuel_budget: list[int]):
        self.matrix = matrix
        self.routes = [[] for _ in matrix.cars]
        self.steps = [0] * len(matrix.cars)
        self.burn = [0] * len(matrix.cars)
        self.left = dict(matrix.stock)
        self.budget = budget
        self.fuel_budget = fuel_budget

    def copy(self) -> "Plan":
        other = Plan.__new__(Plan)
        other.matrix = self.matrix
        other.routes = [list(route) for route in self.routes]
        other.steps = list(self.steps)
        other.burn = list(self.burn)
        other.left = dict(self.left)
        other.budget = self.budget
        other.fuel_budget = self.fuel_budget
        return other

    # -- 採点 ------------------------------------------------------------

    def score(self) -> tuple:
        """(系列の数, 訪問の数, -歩数の合計)。大きいほど良い。

        系列をいちばん上に置くのは、公式の順位が
        種類 → 日別累計 → 玉数 の順だからです。
        """

        brands = set()
        visits = 0
        for route in self.routes:
            visits += len(route)
            for node in route:
                brands.add(self.matrix.brand[node])
        return (len(brands), visits, -sum(self.steps))

    def signature(self) -> tuple:
        return tuple(tuple(route) for route in self.routes)

    # -- 差し込みと取り外し ----------------------------------------------

    def _leg(self, car: int, at: int, node: int) -> tuple[int, int]:
        """car の route の at の位置に node を入れたときの増分 (歩数, 燃料)。"""

        route = self.routes[car]
        steps = self.matrix.steps
        burn = self.matrix.fuel
        before = route[at - 1] if at > 0 else car
        if steps[before][node] >= FAR:
            return (FAR, FAR)
        if at < len(route):
            after = route[at]
            if steps[node][after] >= FAR:
                return (FAR, FAR)
            return (steps[before][node] + steps[node][after]
                    - steps[before][after],
                    burn[before][node] + burn[node][after]
                    - burn[before][after])
        return (steps[before][node], burn[before][node])

    def can_insert(self, car: int, at: int, node: int) -> bool:
        if self.left.get(node, 0) <= 0:
            return False
        if node in self.routes[car]:
            return False
        step_add, fuel_add = self._leg(car, at, node)
        if step_add >= FAR:
            return False
        return (self.steps[car] + step_add <= self.budget
                and self.burn[car] + fuel_add <= self.fuel_budget[car])

    def insert(self, car: int, at: int, node: int) -> None:
        step_add, fuel_add = self._leg(car, at, node)
        self.routes[car].insert(at, node)
        self.steps[car] += step_add
        self.burn[car] += fuel_add
        self.left[node] = self.left.get(node, 0) - 1

    def remove(self, car: int, at: int) -> int:
        route = self.routes[car]
        node = route[at]
        route.pop(at)
        # 抜いたあとの歩数は入れ直して計算し直します（ずれを残さない）。
        self._recount(car)
        self.left[node] = self.left.get(node, 0) + 1
        return node

    def _recount(self, car: int) -> None:
        route = self.routes[car]
        steps = self.matrix.steps
        burn = self.matrix.fuel
        here = car
        total = 0
        used = 0
        for node in route:
            total += steps[here][node]
            used += burn[here][node]
            here = node
        self.steps[car] = total
        self.burn[car] = used

    def feasible(self, car: int) -> bool:
        return (self.steps[car] <= self.budget
                and self.burn[car] <= self.fuel_budget[car])

    # -- 組み立て --------------------------------------------------------

    def fill(self, brand_first: bool = True) -> None:
        """いちばん安いところへ、入るかぎり差し込み続けます。

        `brand_first` のときは、まだ押さえていない系列の店を先に
        入れます（日別累計は玉数より上位の得点だからです）。

        **1回ごとに全部を見直しません。** 店ごとの「いちばん安い
        差し込み先」を覚えておき、入れた車のぶんだけ測り直します。
        差し込みで値段が変わるのは、その車の順路だけだからです。
        32x32（32店・巡回6台）で、1回の埋め直しが4.0ms→0.6msに
        なりました。答えは変わりません（同じ順に同じ場所へ入ります）。
        """

        matrix = self.matrix
        cars = range(len(self.routes))
        have = set()
        for route in self.routes:
            for node in route:
                have.add(matrix.brand[node])
        # 店 → (増える歩数, 車, 位置)。車ごとの最良だけ持ちます。
        spot_best: dict[int, dict[int, tuple[int, int]]] = {}

        def measure(node: int, car: int) -> None:
            """node を car に入れるときの、いちばん安い位置を覚える。"""

            if node in self.routes[car]:
                spot_best[node].pop(car, None)
                return
            found = None
            for at in range(len(self.routes[car]) + 1):
                step_add, fuel_add = self._leg(car, at, node)
                if step_add >= FAR:
                    continue
                if self.steps[car] + step_add > self.budget:
                    continue
                if self.burn[car] + fuel_add > self.fuel_budget[car]:
                    continue
                if found is None or step_add < found[0]:
                    found = (step_add, at)
            if found is None:
                spot_best[node].pop(car, None)
            else:
                spot_best[node][car] = found

        for node, room in self.left.items():
            if room <= 0:
                continue
            spot_best[node] = {}
            for car in cars:
                measure(node, car)
        while True:
            best = None
            for node, places in spot_best.items():
                if self.left.get(node, 0) <= 0 or not places:
                    continue
                fresh = 1 if (brand_first
                              and matrix.brand[node] not in have) else 0
                for car, (step_add, at) in places.items():
                    rank = (-fresh, step_add, car, at, node)
                    if best is None or rank < best[0]:
                        best = (rank, car, at, node)
            if best is None:
                return
            _rank, car, at, node = best
            self.insert(car, at, node)
            have.add(matrix.brand[node])
            if self.left.get(node, 0) <= 0:
                spot_best.pop(node, None)
            # 値段が変わったのは、入れた車の順路だけです。
            for other in list(spot_best):
                measure(other, car)

    # -- 磨き ------------------------------------------------------------

    def two_opt(self, car: int) -> bool:
        """区間をひっくり返して短くなるなら採用。1台ぶん。"""

        route = self.routes[car]
        if len(route) < 3:
            return False
        steps = self.matrix.steps
        moved = False
        for left in range(len(route) - 1):
            for right in range(left + 1, len(route)):
                before = route[left - 1] if left > 0 else car
                after = route[right + 1] if right + 1 < len(route) else None
                old = steps[before][route[left]]
                new = steps[before][route[right]]
                if after is not None:
                    old += steps[route[right]][after]
                    new += steps[route[left]][after]
                if new >= FAR:
                    continue
                # 区間の中身は向きが変わるだけなので、両端だけ見れば足ります
                # （歩数は向きで変わらない前提。山地も道路も出る側のマスで
                #   決まるため厳密には変わりますが、ここは見立てなので
                #   入れ直して数え直します）。
                if new < old:
                    route[left:right + 1] = route[left:right + 1][::-1]
                    self._recount(car)
                    if not self.feasible(car):
                        route[left:right + 1] = route[left:right + 1][::-1]
                        self._recount(car)
                        continue
                    moved = True
        return moved

    def relocate(self) -> bool:
        """1店を別の場所（同じ車でも別の車でも）へ移して短くなるなら採用。"""

        moved = False
        for car in range(len(self.routes)):
            at = 0
            while at < len(self.routes[car]):
                node = self.routes[car][at]
                before_steps = sum(self.steps)
                self.remove(car, at)
                best = None
                for other in range(len(self.routes)):
                    for where in range(len(self.routes[other]) + 1):
                        if other == car and where == at:
                            continue
                        if not self.can_insert(other, where, node):
                            continue
                        step_add, _fuel = self._leg(other, where, node)
                        if best is None or step_add < best[0]:
                            best = (step_add, other, where)
                if best is None:
                    # 戻せないなら元に戻す
                    self.insert(car, at, node)
                    at += 1
                    continue
                _add, other, where = best
                self.insert(other, where, node)
                if sum(self.steps) < before_steps:
                    moved = True
                    if other == car and where <= at:
                        at += 1
                    continue
                at += 1
        return moved

    def polish(self, rounds: int = 2) -> None:
        for _ in range(rounds):
            changed = False
            for car in range(len(self.routes)):
                if self.two_opt(car):
                    changed = True
            if self.relocate():
                changed = True
            if not changed:
                break


def _ruin(plan: Plan, size: int, turn: int) -> None:
    """順路から `size` 店を抜く。どこを抜くかは `turn` で決め打ちします。"""

    spots = [(car, at) for car in range(len(plan.routes))
             for at in range(len(plan.routes[car]))]
    if not spots:
        return
    # 決め打ちで散らすため、turn を種にした等間隔で選びます。
    step = max(1, len(spots) // max(1, size))
    picked = []
    for number in range(size):
        picked.append(spots[(turn * 7 + number * step) % len(spots)])
    # うしろから抜く（番号がずれないように）
    for car, at in sorted(set(picked), reverse=True):
        if at < len(plan.routes[car]):
            plan.remove(car, at)


def search(
    problem: Problem,
    finder,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    refuels: int = 1,
    rounds: int = 200,
    want: int = 4,
) -> list[dict[int, list[int]]]:
    """歩数だけを見て順路を探し、良かった形を `want` 通り返す。

    返すのは `{車の番号: [店の位置, ...]}` で、`PlannedRoutePolicy` に
    そのまま渡せます。採用するかどうかは、呼んだ側が公式シミュレータに
    通して決めてください。
    """

    matrix = DayMatrix(problem, finder, kinds, positions, road_status)
    if not matrix.cars:
        return []
    fuel_budget = [fuels[index] + problem.fuel_limit * refuels
                   for index in matrix.cars]
    best: list[tuple[tuple, tuple]] = []
    seen: set[tuple] = set()

    def keep(plan: Plan) -> None:
        mark = plan.signature()
        if mark in seen:
            return
        seen.add(mark)
        best.append((plan.score(), mark))
        best.sort(key=lambda item: item[0], reverse=True)
        del best[want * 4:]

    start = Plan(matrix, day_steps, fuel_budget)
    start.fill()
    start.polish()
    keep(start)
    plain = Plan(matrix, day_steps, fuel_budget)
    plain.fill(brand_first=False)
    plain.polish()
    keep(plain)

    current = start.copy()
    best_score = current.score()
    for turn in range(rounds):
        trial = current.copy()
        _ruin(trial, RUIN_SIZES[turn % len(RUIN_SIZES)], turn)
        trial.fill()
        trial.polish(rounds=1)
        keep(trial)
        got = trial.score()
        if got >= best_score:
            current = trial
            best_score = got

    out: list[dict[int, list[int]]] = []
    for _score, mark in best[:want]:
        routes: dict[int, list[int]] = {}
        for seat, car in enumerate(matrix.cars):
            routes[car] = [matrix.spot_at[node] for node in mark[seat]]
        out.append(routes)
    return out


def rendezvous(problem: Problem, finder, kinds, positions, fuels,
               road_status, routes) -> list[tuple[int, int]]:
    """**燃料が切れる手前の店を割り出して、補給車の迎え順を作る。**

    手元の燃料だけで組ませると（`refuels=0`）系列は落ちませんが、
    1日の歩数を使い切れません。準決勝32x32は燃料80・96歩なので、
    平地でも**80歩ぶん＝83%**しか動けません。

    そこで満タン1杯ぶんを見込んで順路を組み、各巡回車が
    **燃料切れになる直前にいる店**を計算して、到着の早い順に
    補給車へ渡します。待ち合わせの時刻は決めません。燃料が切れた
    車はその場で止まって待つので、**順番だけ合っていれば必ず
    拾えます**（`PlannedRoutePolicy` の supply_route と同じ約束）。

    第37回では鹿児島51・沖縄12・石川8・サレジオ53・久留米48が
    この「先回り」の形を書いています。

    返すのは `[(待ち受ける店, 車の番号), ...]`。
    """

    spot_cells = {spot.position for spot in problem.spots}
    rows: list[tuple[int, int, int]] = []
    for car, line in routes.items():
        if car >= len(kinds) or kinds[car] != AgentKind.TOURING:
            continue
        if not line:
            continue
        here = positions[car]
        fuel = fuels[car]
        step = 0
        meet = here if here in spot_cells else None
        for spot in line:
            table = finder.table(here, road_status)
            need = table.fuel.get(spot)
            gap = table.steps.get(spot)
            if need is None or gap is None:
                break
            if fuel - need < 0:
                # この先へは行けません。ここで待ち合わせます。
                break
            fuel -= need
            step += gap
            here = spot
            meet = spot
        if meet is None:
            meet = line[-1]
        rows.append((step, meet, car))
    rows.sort()
    return [(cell, car) for _step, cell, car in rows]
