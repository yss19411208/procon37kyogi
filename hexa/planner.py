"""1日分の行動計画を「前向きシミュレーション」で組み立てるプランナー。

方針:
  行動計画をあとから検証するのではなく、公式と同じ処理順で1ステップずつ
  前に進めながら行動を決めます。こうすると「不正な回答」が原理的に作れません。
  燃料の判定もその場の実燃料で行うので、補給を受けたあとの移動も自然に扱えます。

AI（方策）が決めるのは「どのスポットを狙うか」「どの巡回車を補給しに行くか」
だけで、そこへ至る経路は経路探索アルゴリズムが最短で作ります。
これが「学習」と「アルゴリズム」の分担です。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hexa.pathfind import PathFinder, fuel_weight
from hexa.problem import Problem
from hexa.rules import AgentKind, RoadStatus, Terrain, fuel_cost, move_steps


@dataclass
class TouringChoice:
    """巡回車の1回の目標選択。学習の1サンプルになる。"""

    agent_index: int
    step: int
    features: list[float]
    target_position: int | None
    candidate_features: list[list[float]]
    recipe_index: int = -1
    reward: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class SupplyChoice:
    """補給車の1回の目標選択。学習の1サンプルになる。"""

    agent_index: int
    step: int
    features: list[float]
    target_agent: int | None
    target_position: int | None
    candidate_features: list[list[float]] = field(default_factory=list)
    reward: float = 0.0


@dataclass
class DayPlan:
    action_plan: list[list[int]]
    touring_choices: list[TouringChoice]
    supply_choices: list[SupplyChoice]
    acquisitions: list[tuple[int, int, int]]     # (step, agent, brand)
    refuels: list[tuple[int, int]]               # (step, refuelled agent)
    end_positions: list[int]
    end_fuels: list[int]


@dataclass
class _AgentPlanState:
    route: list[int] = field(default_factory=list)   # 残りの移動方向
    target: int | None = None                        # 目標セル
    target_agent: int | None = None
    # 今日これまでに待機したステップ数。上限を超えたら待てなくなる。
    waited: int = 0                  # 補給車が狙う巡回車
    had_options: bool = False                        # 直前の判断で行き先候補があったか
    free_at: int = 0


class DayPlanner:
    def __init__(self, problem: Problem, finder: PathFinder) -> None:
        self.problem = problem
        self.finder = finder

    def plan_day(
        self,
        kinds: tuple[AgentKind, ...],
        positions: list[int],
        fuels: list[int],
        road_status: dict[int, RoadStatus],
        day_steps: int,
        policy,
        acquired_brands: set[int],
        day_index: int = 0,
        day_count: int = 1,
        congestion_risk: dict[int, float] | None = None,
        road_heat: dict[int, float] | None = None,
    ) -> DayPlan:
        problem = self.problem
        agent_count = len(kinds)
        positions = list(positions)
        fuels = list(fuels)
        actions: list[list[int]] = [[] for _ in range(agent_count)]
        plan_state = [_AgentPlanState() for _ in range(agent_count)]
        pending: list[tuple[int, str, int, int] | None] = [None] * agent_count

        spot_by_position = {spot.position: spot for spot in problem.spots}
        stock = {spot.position: spot.stocks for spot in problem.spots}
        acquired_today: list[set[int]] = [set() for _ in range(agent_count)]
        reserved: dict[int, int] = {}     # スポット -> 予約している味方の台数

        acquisitions: list[tuple[int, int, int]] = []
        refuels: list[tuple[int, int]] = []
        touring_choices: list[TouringChoice] = []
        supply_choices: list[SupplyChoice] = []
        brands_today: set[int] = set()

        def release_reservation(position: int | None) -> None:
            if position is None:
                return
            if reserved.get(position):
                reserved[position] -= 1
                if reserved[position] <= 0:
                    reserved.pop(position, None)

        def emit_wait(index: int, length: int) -> None:
            if length <= 0:
                return
            actions[index].append(-length)

        def reflection(step: int) -> None:
            for index in range(agent_count):
                entry = pending[index]
                if entry is None or entry[0] != step:
                    continue
                _complete, kind, destination, cost = entry
                if kind == "move":
                    fuels[index] -= cost
                    positions[index] = destination
                pending[index] = None

            for index in range(agent_count):
                if kinds[index] != AgentKind.TOURING:
                    continue
                spot = spot_by_position.get(positions[index])
                if spot is None or spot.position in acquired_today[index]:
                    continue
                if stock[spot.position] <= 0:
                    continue
                stock[spot.position] -= 1
                acquired_today[index].add(spot.position)
                brands_today.add(spot.brand)
                acquisitions.append((step, index, spot.brand))

            supply_positions = {
                positions[index]
                for index in range(agent_count)
                if kinds[index] == AgentKind.SUPPLY
            }
            if supply_positions:
                for index in range(agent_count):
                    if kinds[index] != AgentKind.TOURING:
                        continue
                    if positions[index] not in supply_positions:
                        continue
                    if fuels[index] >= problem.fuel_limit:
                        continue
                    fuels[index] = problem.fuel_limit
                    refuels.append((step, index))

        def next_hop_cost(index: int, direction: int) -> tuple[int, int, int] | None:
            """(所要ステップ, 燃料消費, 移動先) を返す。移動できないなら None。"""

            origin = positions[index]
            terrain = problem.terrain_at(origin)
            if terrain == Terrain.POND:
                return None
            destination = None
            for candidate_direction, candidate in self.finder.neighbors[origin]:
                if candidate_direction == direction:
                    destination = candidate
                    break
            if destination is None:
                return None
            duration = move_steps(terrain, road_status.get(origin, RoadStatus.SMOOTH))
            cost = fuel_cost(terrain) if kinds[index] == AgentKind.TOURING else 0
            return (duration, cost, destination)

        def start_route(index: int, target: int, remaining: int = 0) -> bool:
            # 巡回車は燃料を考えた経路。補給車は燃料を使わないので最短経路。
            weight = (
                fuel_weight(remaining, fuels[index])
                if kinds[index] == AgentKind.TOURING
                else 0
            )
            table = self.finder.table(positions[index], road_status, weight)
            if not table.reachable(target):
                return False
            plan_state[index].route = table.directions_to(target)
            plan_state[index].target = target
            return True

        def action_phase(step: int) -> None:
            remaining = day_steps - step
            for index in range(agent_count):
                if pending[index] is not None:
                    continue
                state = plan_state[index]

                # 目標に着いた、または目標がないなら、方策に次の目標を聞く。
                if not state.route:
                    if kinds[index] == AgentKind.TOURING:
                        release_reservation(state.target)
                        state.target = None
                        choice = policy.choose_touring_target(
                            agent_index=index,
                            step=step,
                            remaining=remaining,
                            position=positions[index],
                            fuel=fuels[index],
                            stock=stock,
                            reserved=reserved,
                            acquired_today=acquired_today[index],
                            acquired_brands=acquired_brands,
                            brands_today=brands_today,
                            road_status=road_status,
                            finder=self.finder,
                            kinds=kinds,
                            positions=positions,
                            day_index=day_index,
                            day_count=day_count,
                            congestion_risk=congestion_risk,
                            road_heat=road_heat,
                            # 今日の4割まで待てる。それを超えたら必ず動く。
                            allow_wait=(
                                state.waited < day_steps * TOURING_WAIT_BUDGET
                            ),
                            teammate_targets=[
                                other.target if kinds[i] == AgentKind.TOURING else None
                                for i, other in enumerate(plan_state)
                            ],
                        )
                        touring_choices.append(
                            TouringChoice(
                                agent_index=index,
                                step=step,
                                features=choice.features,
                                target_position=choice.target,
                                candidate_features=choice.candidate_features,
                                recipe_index=getattr(choice, "recipe_index", -1),
                            )
                        )
                        state.had_options = len(choice.candidate_features) > 1
                        if choice.target is not None and start_route(index, choice.target, remaining):
                            reserved[choice.target] = reserved.get(choice.target, 0) + 1
                        else:
                            state.target = None
                    else:
                        state.target = None
                        state.target_agent = None
                        supply_claims: dict[int, int] = {}
                        for other_index in range(agent_count):
                            if other_index == index:
                                continue
                            claimed = plan_state[other_index].target_agent
                            if claimed is not None:
                                supply_claims[claimed] = supply_claims.get(claimed, 0) + 1
                        choice = policy.choose_supply_target(
                            agent_index=index,
                            step=step,
                            remaining=remaining,
                            position=positions[index],
                            kinds=kinds,
                            positions=positions,
                            fuels=fuels,
                            plan_targets=[s.target for s in plan_state],
                            supply_claims=supply_claims,
                            road_status=road_status,
                            finder=self.finder,
                            day_index=day_index,
                            day_count=day_count,
                            road_heat=road_heat,
                        )
                        supply_choices.append(
                            SupplyChoice(
                                agent_index=index,
                                step=step,
                                features=choice.features,
                                target_agent=choice.target_agent,
                                target_position=choice.target,
                                candidate_features=choice.candidate_features,
                            )
                        )
                        state.had_options = len(choice.candidate_features) > 1
                        if choice.target is not None and start_route(index, choice.target, remaining):
                            state.target_agent = choice.target_agent
                        else:
                            state.target = None

                # 経路があるなら次の1手を出す。出せないなら待機。
                if state.route:
                    hop = next_hop_cost(index, state.route[0])
                    if hop is not None:
                        duration, cost, destination = hop
                        if step + duration <= day_steps and cost <= fuels[index]:
                            state.route.pop(0)
                            actions[index].append(_direction_action(index, destination, positions, self.finder))
                            pending[index] = (step + duration, "move", destination, cost)
                            continue
                    # 進めないので目標を捨てて待機する。
                    release_reservation(state.target)
                    state.route = []
                    state.target = None

                wait_length = _wait_length(
                    index=index,
                    kinds=kinds,
                    remaining=remaining,
                    plan_state=plan_state,
                )
                emit_wait(index, wait_length)
                # 今日これまでに待った量を数える。上限に達したら待てなくなる。
                state.waited += wait_length
                pending[index] = (step + wait_length, "wait", positions[index], 0)

        action_phase(0)
        _swap_assignments(
            kinds, positions, fuels, plan_state, reserved, stock,
            acquired_today, self.finder, road_status, day_steps, start_route,
        )
        _swap_supply_assignments(
            kinds, positions, fuels, plan_state, self.finder, road_status,
            day_steps, self.problem.fuel_limit,
        )
        for step in range(1, day_steps + 1):
            reflection(step)
            if step < day_steps:
                action_phase(step)

        return DayPlan(
            action_plan=actions,
            touring_choices=touring_choices,
            supply_choices=supply_choices,
            acquisitions=acquisitions,
            refuels=refuels,
            end_positions=positions,
            end_fuels=fuels,
        )


def _direction_action(index: int, destination: int, positions: list[int], finder: PathFinder) -> int:
    origin = positions[index]
    for direction, candidate in finder.neighbors[origin]:
        if candidate == destination:
            return direction
    raise RuntimeError("隣接していないセルへの移動が組み立てられました")


# 巡回車が1日に待機してよい割合。0 なら待機を学習に任せません。
#
# 「燃料を温存して明日に回す」判断を学習に任せられるようにしましたが、
# **実測では日別累計が必ず下がりました**（未見12問）。
#
#   待機なし         : 日別 76.50 / 玉 234.8  ← 採用
#   待機あり 2割まで : 日別 75.83 / 玉 239.8
#   待機あり 4割まで : 日別 75.42 / 玉 238.8
#
# 「今日まだ取っていない系列が1店でも残っているなら待たせない」
# という歯止めも入れましたが、それでも 75.58 までしか戻りませんでした。
# 待った巡回車のぶん、その日に回れる店が減るためだと見ています。
#
# 日別累計は公式の第2優先で玉数より上です。実戦で1位を取った2試合は
# どちらも日別累計で勝っているので、ここを譲ると勝ち筋を失います。
# 玉数が+2%増えても採用しません。
#
# 仕組みは残してあるので、学習後に試すときは 0.2〜0.4 にしてください。
# そのときは玉数だけでなく**日別累計も必ず確認**してください。
TOURING_WAIT_BUDGET = 0.0


def _swap_assignments(
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    plan_state: list["_AgentPlanState"],
    reserved: dict[int, int],
    stock: dict[int, int],
    acquired_today: list[set[int]],
    finder: PathFinder,
    road_status: dict[int, RoadStatus],
    remaining: int,
    start_route,
) -> None:
    """巡回車どうしで目標を交換し、全体の合計が良くなるなら入れ替える。

    **なぜ必要か**

    エージェントは番号の若い順に目標を決め、他が予約した店を避けるだけでした。
    先に決めた者勝ちなので、次のような無駄が残ります。

        0番が店Aを予約したが、実は1番のほうがAに近く、
        0番は店Bのほうが近かった。入れ替えれば両方が得をする。

    番号が若いというだけの理由で0番がAを取っていました。
    ここで全ペアを試し、合計の移動歩数が減るなら交換します。

    **判定の基準**

    公式の優先順位どおり、種類数と日別累計を落とす交換は採用しません。
    どちらも変わらないときだけ、移動歩数の合計で比べます。
    歩数が減れば、その分だけ多くの店を回れます。
    """

    touring = [
        index for index in range(len(kinds))
        if kinds[index] == AgentKind.TOURING and plan_state[index].target is not None
    ]
    if len(touring) < 2:
        return

    def cost_of(agent: int, target: int) -> int | None:
        """その巡回車がその店へ行くのにかかる歩数。届かなければ None。"""

        table = finder.table(
            positions[agent], road_status, fuel_weight(remaining, fuels[agent])
        )
        steps = table.steps.get(target)
        if steps is None or steps > remaining:
            return None
        if table.fuel[target] > fuels[agent]:
            return None
        return steps

    # 何度か回して、改善がなくなったら止める。
    for _round in range(4):
        improved = False
        for first in range(len(touring)):
            for second in range(first + 1, len(touring)):
                a = touring[first]
                b = touring[second]
                target_a = plan_state[a].target
                target_b = plan_state[b].target
                if target_a is None or target_b is None or target_a == target_b:
                    continue
                # すでに今日取った店には行かせない。
                if target_b in acquired_today[a] or target_a in acquired_today[b]:
                    continue

                now_a = cost_of(a, target_a)
                now_b = cost_of(b, target_b)
                if now_a is None or now_b is None:
                    continue
                swap_a = cost_of(a, target_b)
                swap_b = cost_of(b, target_a)
                if swap_a is None or swap_b is None:
                    continue

                if swap_a + swap_b >= now_a + now_b:
                    continue

                # 実際に経路を引き直せるときだけ入れ替える。
                plan_state[a].target = target_b
                plan_state[b].target = target_a
                if start_route(a, target_b, remaining) and start_route(b, target_a, remaining):
                    improved = True
                else:
                    # 引き直せなかったら元に戻す。
                    plan_state[a].target = target_a
                    plan_state[b].target = target_b
                    start_route(a, target_a, remaining)
                    start_route(b, target_b, remaining)
        if not improved:
            break


def _cover_missing_brands(
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    plan_state: list["_AgentPlanState"],
    reserved: dict[int, int],
    stock: dict[int, int],
    acquired_today: set[int],
    brands_today: set[int],
    problem: Problem,
    finder: PathFinder,
    road_status: dict[int, RoadStatus],
    remaining: int,
    start_route,
) -> None:
    """今日まだ誰も向かっていない系列があれば、担当を割り当てる。

    **いまは呼んでいません。実測で逆効果だったためです。**

        カバーなし: 日別累計107 / 玉数202  ← 採用
        カバーあり: 日別累計103 / 玉数186

    理由は、この処理が「その日の最初」にしか働かないことです。
    4体で14系列あるとき、最初に狙えるのは4系列だけで、残り10系列は
    日中に何度も選び直しながら回収します。最初の時点で
    「誰も向かっていない系列」を数えると10個もあり、
    そこへ無理に割り当てると、本来の良い道すじを壊すだけになります。

    直すなら、日中の選び直しのたびに効く形にする必要があります。
    ただし「今日まだ取っていない系列を優先する」判断は、すでに
    貪欲法の並べ替えと作戦「未獲得系列」に入っています。
    残る差は優先順位ではなく、到達できるかどうかの問題です。

    **なぜ必要か**

    日別累計は公式の第2優先で、玉数より上です。満点を取るには
    「その日に取れる系列を全部そろえる」必要があります。

    ところが、いまの仕組みには**誰がどの系列を担当するかを決める処理が
    ありませんでした。** 各エージェントが自分の価値判断で道すじを選び、
    カバーできるかどうかは結果論です。全員が同じ方面へ向かって、
    辺境の1店が残る、ということが起きます。

    実戦の24x24マップで調べると、取り逃しの9割が
    「その系列の店が1つしかない」ものでした。14系列のうち7つが1店だけで、
    そこへ毎日行かないと満点126になりません。1位botはそれをやっています。

    ここでは、向かう先が決まっていない系列を洗い出し、
    **一番安く行けるエージェントに担当させます。** 歩数と燃料が足りる
    範囲だけで、足りなければ諦めます。

    いま向かっている系列を空にしてしまう奪い方はしません。
    片方を埋めて別を空けるだけになるからです。
    """

    touring = [
        index for index in range(len(kinds)) if kinds[index] == AgentKind.TOURING
    ]
    if not touring:
        return

    brand_of = {spot.position: spot.brand for spot in problem.spots}

    def cost_of(agent: int, target: int) -> int | None:
        table = finder.table(
            positions[agent], road_status, fuel_weight(remaining, fuels[agent])
        )
        steps = table.steps.get(target)
        if steps is None or steps > remaining:
            return None
        if table.fuel[target] > fuels[agent]:
            return None
        return steps

    for _round in range(len(touring)):
        # いまどの系列へ向かっているか
        heading: dict[int, int] = {}
        for index in touring:
            target = plan_state[index].target
            if target is None:
                continue
            brand = brand_of.get(target)
            if brand is not None:
                heading[brand] = heading.get(brand, 0) + 1

        # 今日まだ取っておらず、誰も向かっていない系列
        pending: list[int] = []
        for spot in problem.spots:
            if stock.get(spot.position, 0) <= 0:
                continue
            if spot.position in acquired_today:
                continue
            if spot.brand in brands_today or spot.brand in heading:
                continue
            pending.append(spot.position)
        if not pending:
            break

        best: tuple[int, int, int] | None = None
        for target in pending:
            for agent in touring:
                if target in acquired_today:
                    continue
                current = plan_state[agent].target
                if current is not None:
                    current_brand = brand_of.get(current)
                    # いま向かっている系列を空にしてしまうなら奪わない
                    if current_brand is not None and heading.get(current_brand, 0) <= 1:
                        continue
                steps = cost_of(agent, target)
                if steps is None:
                    continue
                if best is None or steps < best[0]:
                    best = (steps, target, agent)
        if best is None:
            break

        _steps, target, agent = best
        previous = plan_state[agent].target
        plan_state[agent].target = target
        if not start_route(agent, target, remaining):
            plan_state[agent].target = previous
            if previous is not None:
                start_route(agent, previous, remaining)
            break

        if previous is not None and previous != target and reserved.get(previous):
            reserved[previous] -= 1
            if reserved[previous] <= 0:
                reserved.pop(previous, None)
        reserved[target] = reserved.get(target, 0) + 1


def _swap_supply_assignments(
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    plan_state: list["_AgentPlanState"],
    finder: PathFinder,
    road_status: dict[int, RoadStatus],
    remaining: int,
    fuel_limit: int,
) -> None:
    """補給車どうしで担当する巡回車を交換し、全体が良くなるなら入れ替える。

    巡回車と同じく、補給車も番号の若い順に担当を決めていました。
    「どの補給車がどの巡回車を担当すると全体で一番得か」は解いていません。

    ここでは「合流までにかかる歩数の合計」で比べます。
    早く着くほど、その巡回車が動けない時間が短くなるからです。
    ただし**燃料が尽きている巡回車を後回しにする交換は採用しません。**
    燃料0の巡回車は補給されるまで0玉のままなので、そこが最優先です。
    """

    supply = [
        index for index in range(len(kinds))
        if kinds[index] == AgentKind.SUPPLY
        and plan_state[index].target_agent is not None
    ]
    if len(supply) < 2:
        return

    def arrival(agent: int, target_agent: int) -> int | None:
        table = finder.table(positions[agent], road_status, 0)
        steps = table.steps.get(positions[target_agent])
        if steps is None or steps > remaining:
            return None
        return steps

    def urgency(target_agent: int) -> int:
        # 燃料が少ないほど先に助ける。0 なら最優先。
        return fuels[target_agent]

    for _round in range(4):
        improved = False
        for first in range(len(supply)):
            for second in range(first + 1, len(supply)):
                a = supply[first]
                b = supply[second]
                to_a = plan_state[a].target_agent
                to_b = plan_state[b].target_agent
                if to_a is None or to_b is None or to_a == to_b:
                    continue

                now_a = arrival(a, to_a)
                now_b = arrival(b, to_b)
                swap_a = arrival(a, to_b)
                swap_b = arrival(b, to_a)
                if None in (now_a, now_b, swap_a, swap_b):
                    continue
                if swap_a + swap_b >= now_a + now_b:
                    continue

                # 立ち往生している巡回車の到着が遅くなる交換はしない。
                stranded = [
                    target for target in (to_a, to_b) if urgency(target) <= 0
                ]
                if stranded:
                    before = {to_a: now_a, to_b: now_b}
                    after = {to_b: swap_a, to_a: swap_b}
                    if any(after[target] > before[target] for target in stranded):
                        continue

                plan_state[a].target_agent = to_b
                plan_state[b].target_agent = to_a
                improved = True
        if not improved:
            break


def _wait_length(
    index: int,
    kinds: tuple[AgentKind, ...],
    remaining: int,
    plan_state: list[_AgentPlanState],
) -> int:
    """待機するステップ数。

    待機を1回選んだだけでその日が終わってしまうと、判断を1度でも外したときの
    損害が大きすぎます。まだ行ける店が残っているうちは短く待って、
    在庫や道路の状況が変わったところで選び直せるようにします。
    """

    if kinds[index] == AgentKind.TOURING:
        awaited = any(state.target_agent == index for state in plan_state)
        if awaited:
            # 補給を待っている巡回車は、補給された次のステップで動けるようにする。
            return 1
        if plan_state[index].had_options:
            # 行ける店があるのに待つと決めた場合は、日の終わりまで固定しない。
            return max(1, min(remaining, max(1, remaining // 8)))
        # 行ける店が1つもない＝たいていは燃料切れ。ここで日の終わりまで
        # 待つと決めてしまうと、**途中で補給されても動き出せません。**
        # 実戦で「燃料0の巡回車が5日間ずっと止まったまま、在庫は60以上残る」
        # という状態を確認しました。こまめに選び直せるようにします。
        return max(1, min(remaining, max(1, remaining // 12)))
    elif plan_state[index].had_options:
        # 補給車も同じ。合流先の候補があるうちは、こまめに選び直す。
        return max(1, min(remaining, max(1, remaining // 8)))
    return max(1, remaining)
