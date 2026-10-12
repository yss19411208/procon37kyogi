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

from hexa.pathfind import FUEL_WEIGHT_STEPS, PathFinder, fuel_weight
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


# `plan_targets` も同じように「読むときに取り出す」形を試しましたが、
# **遅くなりました**（第1958の32x32で20秒あたり6138手→5972手）。
# 補給車側が `plan_targets[other]` と添字で何度も引くので、
# メソッド呼び出しのほうがリストを作るより高くつきます。
# 読む回数が少ない `teammate_targets` だけが得をする形でした。


class _TeammateView:
    """味方がいま向かっている先を、**なめるときに**読む並び。

    `for index, target in enumerate(teammate_targets or [])` のように
    使われるので、`__len__` と `__iter__` だけあれば足ります。
    リストを作らないので、読まない方策では値段が0になります。
    """

    __slots__ = ("plan_state", "touring")

    def __init__(self, plan_state, touring):
        self.plan_state = plan_state
        self.touring = touring

    def __len__(self):
        return len(self.plan_state)

    def __iter__(self):
        touring = self.touring
        for index, other in enumerate(self.plan_state):
            yield other.target if touring[index] else None


@dataclass
class DayPlan:
    action_plan: list[list[int]]
    touring_choices: list[TouringChoice]
    supply_choices: list[SupplyChoice]
    # (歩数, 号車, 店のセル, 系列)
    acquisitions: list[tuple[int, int, int, int]]
    refuels: list[tuple[int, int]]               # (step, refuelled agent)
    end_positions: list[int]
    end_fuels: list[int]
    # 日の終わりに残った在庫（セル -> 数）。
    # **なぜ持つか**
    # 計画を作る過程でこちらも1歩ずつ前に進めているので、公式
    # シミュレータに通し直さなくても結果が分かります。磨きでは
    # 同じ日を何万回も再現するので、2度目のシミュレートを省ける
    # ようにこの形で返します（採用する案だけ通し直します）。
    remaining_stocks: dict[int, int] = field(default_factory=dict)


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
        collect_choices: bool = True,
    ) -> DayPlan:
        """その日の行動計画を作る。

        `collect_choices` を False にすると、学習用の「どの候補から
        どれを選んだか」の記録を残しません。**磨きの最中はこの記録を
        誰も読まない**のに、1日の再現で何百個も作っていました。
        記録以外の答えは何も変わりません。
        """
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

        # 1マスぶんの移動の値は、道路状態ごとに1回だけ作って使い回します
        # （`PathFinder.move_tables` に理由を書いてあります）。
        cell_steps, cell_fuel, hop_table = self.finder.move_tables(road_status)
        touring = [kind == AgentKind.TOURING for kind in kinds]
        # **席の一覧を1日1回だけ作ります。**
        #
        # `reflection` は毎歩すべての車をなめて「巡回車か補給車か」を
        # 判定し直していました。第1958（32x32・7体・99歩）では1手に
        # 99回×7台で、`reflection` だけで全体の8%を使っていました。
        # 種別は1日のあいだ変わらないので、席を先に分けておきます。
        # なめる順番は番号の小さい順のままなので、答えは変わりません。
        touring_seats = [index for index, got in enumerate(touring) if got]
        supply_seats = [index for index in range(agent_count)
                        if kinds[index] == AgentKind.SUPPLY]

        # **誰がどの店を予約したか**を車ごとに覚えます。
        #
        # 以前は「位置」だけで解放していました。予約を増やすのは
        # `choice.target` で経路を引けた1か所だけなのに、`state.target` を
        # 直に書く機構（救援・待つ前の拾い直し・立ち往生時の合流など）が
        # いくつもあり、それらは**増やさないのに解放で減らします**。
        # 減った先が他車の予約だと、その店の予約が消えて3台目が向かえる
        # ようになります。在庫1の店に2台向かえば、片方は丸損です。
        claimed: list[int | None] = [None] * agent_count

        def release_reservation(index: int) -> None:
            """その車の予約を1つだけ返します（持っていなければ何も）。"""

            if not RESERVE_PER_AGENT:
                # 以前の形（位置だけで減らす）。測り比べるために残します。
                position = plan_state[index].target
                if position is not None and reserved.get(position):
                    reserved[position] -= 1
                    if reserved[position] <= 0:
                        reserved.pop(position, None)
                return
            position = claimed[index]
            if position is None:
                return
            claimed[index] = None
            if reserved.get(position):
                reserved[position] -= 1
                if reserved[position] <= 0:
                    reserved.pop(position, None)

        def claim_reservation(index: int, position: int) -> None:
            """その車の予約を1つ立てます（前の予約は返します）。"""

            release_reservation(index)
            reserved[position] = reserved.get(position, 0) + 1
            if RESERVE_PER_AGENT:
                claimed[index] = position

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

            for index in touring_seats:
                spot = spot_by_position.get(positions[index])
                if spot is None or spot.position in acquired_today[index]:
                    continue
                if stock[spot.position] <= 0:
                    continue
                stock[spot.position] -= 1
                acquired_today[index].add(spot.position)
                brands_today.add(spot.brand)
                acquisitions.append(
                    (step, index, spot.position, spot.brand)
                )

            if supply_seats:
                supply_positions = {positions[index]
                                    for index in supply_seats}
                limit = problem.fuel_limit
                for index in touring_seats:
                    if positions[index] not in supply_positions:
                        continue
                    if fuels[index] >= limit:
                        continue
                    fuels[index] = limit
                    refuels.append((step, index))

        def next_hop_cost(index: int, direction: int) -> tuple[int, int, int] | None:
            """(所要ステップ, 燃料消費, 移動先) を返す。移動できないなら None。"""

            origin = positions[index]
            hops = hop_table[origin]
            if hops is None:
                return None
            destination = hops.get(direction)
            if destination is None:
                return None
            return (cell_steps[origin],
                    cell_fuel[origin] if touring[index] else 0,
                    destination)

        def _detour_through_spots(index: int, route: list[int],
                                  remaining: int) -> list[int]:
            """途中の1区間ずつ、隣の店を踏むように差し替える。

            車は歩いた先の店を無条件で拾うので、**踏むだけで玉**です。
            隣のマスに店があり、そこを経由しても次のマスへ行けるなら、
            増える歩数は「その店のマスを出る歩数」だけです。

            歩数も燃料も足りる範囲でしか差し替えません。今日すでに
            その車が取った店と、在庫の切れた店は数えません。
            """

            if not route or PATH_SPOT_DETOUR <= 0:
                return route
            # いまの経路でかかる歩数と燃料を数えます。
            here = positions[index]
            used_steps = 0
            used_fuel = 0
            cells = [here]
            for direction in route:
                hops = hop_table[here]
                if hops is None:
                    return route
                destination = hops.get(direction)
                if destination is None:
                    return route
                used_steps += cell_steps[here]
                used_fuel += cell_fuel[here] if touring[index] else 0
                here = destination
                cells.append(here)
            spare = remaining - used_steps
            spare_fuel = fuels[index] - used_fuel
            if spare <= 0:
                return route
            mine = acquired_today[index]
            out: list[int] = []
            for at, direction in enumerate(route):
                out.append(direction)
                if spare <= 0:
                    continue
                origin = cells[at]
                goal = cells[at + 1]
                best = None
                for way, middle in self.finder.neighbors[origin]:
                    if middle == goal or middle in cells:
                        continue
                    spot = spot_by_position.get(middle)
                    if spot is None or spot.position in mine:
                        continue
                    if stock.get(spot.position, 0) <= 0:
                        continue
                    add_steps = cell_steps[middle]
                    add_fuel = cell_fuel[middle] if touring[index] else 0
                    if add_steps > spare or add_fuel > spare_fuel:
                        continue
                    back = None
                    for other, after in self.finder.neighbors[middle]:
                        if after == goal:
                            back = other
                            break
                    if back is None:
                        continue
                    if best is None or add_steps < best[0]:
                        best = (add_steps, add_fuel, way, back)
                if best is None:
                    continue
                add_steps, add_fuel, way, back = best
                # 直進の1手を「寄ってから行く」2手に差し替えます。
                out[-1] = way
                out.append(back)
                spare -= add_steps
                spare_fuel -= add_fuel
            return out

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
            route = table.directions_to(target)
            if PATH_SPOT_DETOUR > 0 and touring[index]:
                route = _detour_through_spots(index, route, remaining)
            plan_state[index].route = route
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
                        release_reservation(index)
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
                            # **要るときだけ作ります。**
                            #
                            # 使うのは貪欲（`GreedyPolicy`）だけで、
                            # 設計した順路をなぞる `PlannedRoutePolicy`
                            # は見ません。それなのに毎回リストを作って
                            # いました。第1958（32x32・7体）の実測で
                            # 1日ぶんに56万回呼ばれ、そのうち使うのは
                            # 21万回だけです。
                            #
                            # 1歩のなかでも車が順に決めていくので、
                            # 呼ばれた時点の中身でなければいけません。
                            # だから「作っておく」ではなく「なめる
                            # ときに読む」形にします。
                            teammate_targets=_TeammateView(plan_state,
                                                           touring),
                        )
                        if collect_choices:
                            touring_choices.append(
                                TouringChoice(
                                    agent_index=index,
                                    step=step,
                                    features=choice.features,
                                    target_position=choice.target,
                                    candidate_features=choice.candidate_features,
                                    recipe_index=getattr(choice, "recipe_index",
                                                         -1),
                                )
                            )
                        state.had_options = len(choice.candidate_features) > 1
                        # **燃料が尽きる前に、補給車のほうへ向かいます。**
                        #
                        # 実戦1356の7日目で、車0は歩20に燃料2で止まり、
                        # 補給が来たのは歩92。**72歩も立ち往生**しました。
                        # 全体では336歩（予算の7%）が立ち往生で、
                        # 玉数にして約30の損失です。
                        #
                        # 補給車が探しに来るのを待つのではなく、
                        # 動けるうちに自分から近づきます。
                        rescue = None
                        if (MEET_SUPPLY_EARLY
                                and fuels[index] <= self.problem.fuel_limit
                                * MEET_SUPPLY_RATIO):
                            helper = _nearest_supply(
                                index, kinds, positions, fuels, remaining,
                                self.finder, road_status,
                            )
                            if helper is not None:
                                table = self.finder.table(
                                    positions[index], road_status
                                )
                                here = table.steps.get(positions[helper])
                                there = (
                                    table.steps.get(choice.target)
                                    if choice.target is not None else None
                                )
                                # 店より補給車が近いなら、そちらへ向かう。
                                if here is not None and (
                                    there is None or here < there
                                ):
                                    rescue = positions[helper]
                        if rescue is not None and start_route(
                            index, rescue, remaining
                        ):
                            state.target = rescue
                        elif choice.target is not None and start_route(index, choice.target, remaining):
                            claim_reservation(index, choice.target)
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
                        if collect_choices:
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
                    release_reservation(index)
                    state.route = []
                    state.target = None

                # **補給を待つ前に、まだ行ける店がないか見ます。**
                #
                # 実戦1356の3日目で、車0は歩47から待ち始め補給は歩108。
                # **61歩も待って**いました（187歩の日の3分の1）。
                # その間、7歩先に取れる店があったのにです。
                if (WAIT_SMARTER and kinds[index] == AgentKind.TOURING
                        and remaining > 0
                        and any(st.target_agent == index for st in plan_state)):
                    picked = _salvage_target(
                        index, positions[index], fuels[index], remaining,
                        stock, acquired_today[index], reserved,
                        self.finder, road_status,
                    )
                    if (picked is not None
                            and start_route(index, picked, remaining)
                            and state.route):
                        state.target = picked
                        hop = next_hop_cost(index, state.route[0])
                        if hop is not None:
                            duration, cost, destination = hop
                            if (step + duration <= day_steps
                                    and cost <= fuels[index]):
                                state.route.pop(0)
                                actions[index].append(_direction_action(
                                    index, destination, positions, self.finder))
                                pending[index] = (step + duration, "move",
                                                  destination, cost)
                                continue
                        release_reservation(index)
                        state.route = []
                        state.target = None

                # **待つ前に、すぐ行ける店がないか見ます。**
                #
                # 実戦1346の1日目で、車2が14歩も待っていました。燃料は
                # 116残り、7歩先に取れる店があったのに行っていません。
                # 作戦が「良い店」を見つけられなかっただけで、
                # 近くに1玉でも残っていれば拾うほうが得です。
                # **行き先がないと判断したときだけ拾います。**
                #
                # 実戦1447の1日目で、車3は歩46で最後の店に着いたあと
                # 14歩も止まっていました。燃料23が残り、同じ店に1玉、
                # 4歩先に2玉、7歩先に2玉が残っていたのにです。
                # その日の終了時、まだ27玉が15店に残っていました。
                #
                # 以前これを「待機の途中」に割り込ませて失敗しました
                # （玉163→161）。待機には「次の良い店へ向かうタイミング
                # 待ち」も含まれ、そこを邪魔したからです。
                # 今回は **had_options が False のとき**、つまり作戦が
                # 行き先を見つけられなかったときだけに限ります。
                if (SALVAGE_NEARBY and kinds[index] == AgentKind.TOURING
                        and remaining > 0
                        and not plan_state[index].had_options):
                    picked = _salvage_target(
                        index, positions[index], fuels[index], remaining,
                        stock, acquired_today[index], reserved,
                        self.finder, road_status,
                    )
                    if (picked is not None
                            and start_route(index, picked, remaining)
                            and state.route):
                        state.target = picked
                        hop = next_hop_cost(index, state.route[0])
                        if hop is not None:
                            duration, cost, destination = hop
                            if (step + duration <= day_steps
                                    and cost <= fuels[index]):
                                state.route.pop(0)
                                actions[index].append(_direction_action(
                                    index, destination, positions, self.finder))
                                pending[index] = (step + duration, "move",
                                                  destination, cost)
                                continue
                        release_reservation(index)
                        state.route = []
                        state.target = None

                # **補給車は、同じマスの巡回車に同行します。**
                #
                # ここに来た補給車は「行くべき先がない」か「行き先が
                # 自分のいるマス」です。つまり迎えた相手の上にいます。
                # 相手が出ていくと離れてしまい、また追いかけることに
                # なります。同じマスから出るなら**同じ歩数で同じ先に
                # 着ける**ので、ついて行けばずっと満タンに保てます
                # （`SUPPLY_FOLLOWS` に他高専の解法つきの理由）。
                if (SUPPLY_FOLLOWS and kinds[index] == AgentKind.SUPPLY
                        and remaining > 0):
                    mate = _follow_partner(
                        index, kinds, positions, plan_state, pending,
                        self.finder,
                    )
                    if mate is not None and start_route(
                            index, mate[0], remaining) and state.route:
                        hop = next_hop_cost(index, state.route[0])
                        if hop is not None:
                            duration, cost, destination = hop
                            if step + duration <= day_steps:
                                state.route.pop(0)
                                state.target = mate[0]
                                state.target_agent = mate[1]
                                actions[index].append(_direction_action(
                                    index, destination, positions, self.finder))
                                pending[index] = (step + duration, "move",
                                                  destination, cost)
                                continue
                        state.route = []
                        state.target = None

                # **その日の歩が足りないだけなら、明日の店へ歩きます。**
                #
                # 燃料で届く店があるのに、残り歩では着けないとき。
                # 止まっていても明日は近づきません（`DRIFT_TO_SPOT`）。
                if (DRIFT_TO_SPOT and kinds[index] == AgentKind.TOURING
                        and remaining > 0 and fuels[index] > 0):
                    cell = _drift_toward_spot(
                        index=index, position=positions[index],
                        fuel=fuels[index], remaining=remaining,
                        stock=stock, taken_today=acquired_today[index],
                        problem=self.problem, finder=self.finder,
                        road_status=road_status,
                    )
                    if (cell is not None
                            and start_route(index, cell, remaining)
                            and state.route):
                        hop = next_hop_cost(index, state.route[0])
                        if hop is not None:
                            duration, cost, destination = hop
                            if (step + duration <= day_steps
                                    and cost <= fuels[index]):
                                state.route.pop(0)
                                state.target = None
                                actions[index].append(_direction_action(
                                    index, destination, positions, self.finder))
                                pending[index] = (step + duration, "move",
                                                  destination, cost)
                                continue
                        state.route = []
                        state.target = None

                # **待つくらいなら、補給車のほうへ歩きます。**
                #
                # ここに来た巡回車は、店にも補給車にも届きません
                # （上の2つで店は探し終えています）。止まっていても
                # 燃料は増えないので、届くところまで近づきます。
                # 満タンにしてもらうので、ここで使う燃料は戻ります。
                # 理由の内訳は `MEET_SUPPLY_WHEN_STUCK` に。
                if (MEET_SUPPLY_WHEN_STUCK
                        and kinds[index] == AgentKind.TOURING
                        and remaining > 0 and fuels[index] > 0):
                    cell = _approach_supply_with_fuel(
                        index=index, kinds=kinds, positions=positions,
                        fuels=fuels, remaining=remaining,
                        finder=self.finder, road_status=road_status,
                    )
                    if (cell is not None
                            and start_route(index, cell, remaining)
                            and state.route):
                        hop = next_hop_cost(index, state.route[0])
                        if hop is not None:
                            duration, cost, destination = hop
                            if (step + duration <= day_steps
                                    and cost <= fuels[index]):
                                state.route.pop(0)
                                state.target = cell
                                actions[index].append(_direction_action(
                                    index, destination, positions, self.finder))
                                pending[index] = (step + duration, "move",
                                                  destination, cost)
                                continue
                        state.route = []
                        state.target = None

                wait_length = _wait_length(
                    index=index,
                    kinds=kinds,
                    remaining=remaining,
                    plan_state=plan_state,
                    day_steps=day_steps,
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
            remaining_stocks=dict(stock),
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


# **予約を「車ごと」に記帳するか。** False で以前の形（位置だけで減らす）。
# 理由は `release_reservation` の上に。
RESERVE_PER_AGENT = False

# 燃料切れで動けないときに待つ歩数。短いほど、補給された直後に動き出せます。
#
# 以前は `remaining // 12` としていたので、132歩の日は16歩も待っていました。
# 実測（実戦から復元した4問）:
#
#   1歩: 玉831 / 2歩: 玉812 / **3歩: 玉844** / 4歩: 玉843
#   5歩: 玉819 / 8歩: 玉818 / 以前(11〜16歩): 玉835
#
# 短すぎても判断の回数が増えて重くなり、長すぎると補給に気づきません。
# 待つ前に、近くの取れる店を拾いに行くか。
#
# **実測で効かなかったので無効にしています。**
#
# 実戦1346の1日目で車2が14歩も待っており、7歩先に取れる店が
# ありました。拾えば得に見えますが、実測では
#
#   16x16: 玉163 → 161
#   24x24: 玉266 → 262
#
# と**どちらも悪化**しました。待っているのは「次の良い店へ向かう
# タイミングを待っている」ためで、近くの1玉を拾うと、そのあとの
# 動きが崩れるようです。
# 行き先がないときに、近くの店を拾いに行くか。
#
# **実測で効かなかったので無効にしています。**
#
# 実戦1447の1日目で、車3は歩46から14歩止まり、4歩先に2玉が
# 残っていました。拾えば得に見えますが、届く範囲を変えても
#
#   拾わない: 1233=92% / 1228=89%
#   30%以内 : 1233=92% / 1228=89%  （変わらず）
#   50%以内 : 1233=89% / 1228=88%  （悪化）
#
# 変わらないか悪化します。止まっている車が拾える玉は、
# すでに別の車が取りに行っているためです。
SALVAGE_NEARBY = False
SALVAGE_REACH = 0.3

# 補給を待っている巡回車が、まだ行ける店へ向かうか。
#
# **実測で逆効果だったので無効にしています。**
#
# 実戦1356の3日目で、車0は歩47から待ち始め補給は歩108。61歩も
# 待っており、その間7歩先に取れる店がありました。動けば得に
# 見えますが、実測では玉327→318と**悪化**します。
#
# 燃料の少ない車がさらに燃料を使うと、補給後に遠くまで行けなく
# なるためです。**待つのは正しい判断**でした。
WAIT_SMARTER = False

# 燃料がこの割合を切ったら、店より補給車を先に目指すか。
#
# **実測で効かなかったので無効にしています。**
#
# 実戦1356の7日目で、巡回車3台の開始燃料が[11, 2, 27]。補給車までは
# 50〜55歩で燃料45〜53が必要でした。つまり**尽きた車は補給車へ行く
# ことすらできません**。向かおうとすると途中で止まり、立ち往生が
# 171歩→210歩に増えました（玉48→46）。
#
# 直すべきは「前日に燃料を使い切らないこと」です。
# **店を踏むために、1区間だけ遠回りしてよい歩数。0 で止めます。**
#
# 車は歩いた先の店を無条件で拾います（泊まる必要はありません）。
# 実測で、玉のうち**通りがかりで拾ったもの**がこれだけありました
# （記録された渋滞つき・10秒磨いたあと）:
#
#   第1958 (32x32) 1日目 玉85 のうち通りがかり10（12%）
#   第1958 (32x32) 3日目 玉59 のうち通りがかり**25（42%）**
#   第1891 (24x24) 1日目 玉49 のうち通りがかり12（24%）
#   第1981 (16x16) 3日目 玉20 のうち通りがかり 4（20%）
#
# つまり「どの店を順路に入れるか」より「どのマスを踏むか」が
# 点数を決めています。順路を縮めると**かえって玉が減りました**
# （第1958の3日目で歩556→535に縮めたら玉59→55）。踏んでいた店を
# 通らなくなるからです。
#
# そこで逆向きに、**少し遠回りして店を踏む**ようにします。隣の
# マスに店があり、そこを経由しても次のマスへ行けるなら、増える
# 歩数は「その店のマスを出る歩数」だけです（平地2歩・道路1〜4歩）。
PATH_SPOT_DETOUR = 0

# **行き先がない車は、待たずに補給車へ歩きます。**
#
# 第1703（32x32）で、巡回車の歩のうち待ちが19.7%ありました。その
# 理由を分けると
#
#   燃料が足りず店に行けない : 340歩 (39%)
#   燃料切れ                 : 271歩 (31%)
#   歩数が足りない           : 180歩 (21%)
#   その他                   :  70歩 (8%)
#   在庫のある店が残っていない:  0歩 (0%)
#
# **7割が燃料です。** しかも燃料の総量は足りていて（9日で2374入れ、
# 使い切るのに要るのは2172）、足りないのは「空の車と補給車が
# 落ち合うこと」でした。
#
# 立ち止まっている車の燃料は、どうせ店には届きません。だったら
# 補給車のほうへ歩いたほうが得に見えます（満タンになるので、使った
# 燃料は戻ります）。`MEET_SUPPLY_EARLY` と違い、燃料の残りではなく
# 「行き先がない」ことで判断します。
#
# **ところが実測で大きく悪化しました（False で止めています）。**
# 第1703（32x32・1日6秒）:
#
#   切る : 日別224 玉438（13位）
#   入れる: 日別210 玉351（22位）
#
# 理由は2つです。**補給車は巡回車のいまの位置へ向かいます**から、
# 巡回車が動くと待ち合わせ先が動き続けて、すれ違います。それに
# 最後の燃料で店から離れるので、満タンになった場所から店が遠く
# なります。待つほうが、迎えに来てもらえて店の近くにいられます。
MEET_SUPPLY_WHEN_STUCK = False


# **その日の歩が足りないだけの車は、明日の店へ歩き出します。**
#
# 待ちの理由のうち「歩数が足りない」が180歩（待ちの21%、全体の4%）
# ありました。燃料は足りていて、店も届くのに、**その日の残り歩では
# 着けない**という状態です。
#
# 止まっていても明日が近づくわけではありません。届くところまで
# 歩いておけば、翌朝そのぶん早く着きます（店は毎日同じ場所にあり、
# 在庫も毎朝戻ります）。
# **実測で入れませんでした（False で止めています）。**
# 貪欲だけの29試合では+3位ぶんですが、これだけ入れると順位合計550
# といちばん悪くなります（ほかの燃料の直しとそろって初めて効く）。
DRIFT_TO_SPOT = False


# **補給車は、迎えた巡回車に「同行」します。**
#
# 第37回の解法紹介（他高専）を読むと、最多数派は**ペアを作って
# 同行させ、燃料制約そのものを消す**やり方でした。
#
#   長岡（35 Unfixed）: 「巡回車と補給車が同ルートを走行して常に
#     給油するため、**燃料の残量を考慮する必要をなくした**」
#   仙台広瀬（1 おぉ）: 「合流させて同時に動かすことで、**燃料無限の
#     巡回車**として扱うことができる」
#   小山（55）: 1対1のペアを作り、間隔を最小にして補給車は追従
#   津山（4）: 巡回車は基本的に補給車から離れないように動かす
#
# こちらの実装は補給車が**往復して配る**形でした。第1703（32x32）の
# 実測では、巡回車の待ち歩の7割が燃料で、燃料の総量は足りていて
# **空の車と落ち合えていないだけ**でした。同行なら落ち合い続けます。
#
# 補給車は燃料を1滴も使わず、同じマスから出るので移動にかかる歩数も
# 同じです。だから**いちど合流したら、ずっと一緒に動けます**。
#
# 「ほかに行くべき車がないとき」だけ同行します。巡回車が3台で補給車が
# 1台なら、満タンにした相手から離れて次の車へ向かいます（その車の
# 不足が大きいので、下の採点でそちらが選ばれる）。巡回車2台・補給車
# 2台なら相手がいないので、そのまま同行し続けます。
# **実測で入れませんでした（False で止めています）。**
# 貪欲だけの29試合では+9位ぶん効きますが、磨きを入れると
# 燃料の仕掛け一式（予定・セル探索・先回り）でまとめて測って
# 日別が落ちました（733→729）。
SUPPLY_FOLLOWS = False


MEET_SUPPLY_EARLY = False
MEET_SUPPLY_RATIO = 0.25


def _nearest_supply(index, kinds, positions, fuels, remaining, finder,
                    road_status):
    """いちばん近い補給車。届かなければ None。"""

    table = finder.table(positions[index], road_status)
    best = None
    best_steps = None
    for other, kind in enumerate(kinds):
        if kind != AgentKind.SUPPLY:
            continue
        spot = positions[other]
        steps = table.steps.get(spot)
        if steps is None or steps > remaining:
            continue
        if table.fuel[spot] > fuels[index]:
            continue
        if best_steps is None or steps < best_steps:
            best = other
            best_steps = steps
    return best


def _follow_partner(index, kinds, positions, plan_state, pending, finder):
    """同じマスにいる巡回車の**次の行き先**を返す。なければ None。

    戻り値は (行き先のセル, その車)。

    **なぜ「次の行き先」か。** 補給車がいまの位置を追いかけると、
    1マスずつ遅れて永久に離れます。同じマスにいるなら、相手が
    これから出る先へ**同じ歩数で**着けます（出発するマスが同じ
    なので移動にかかる歩数も同じ）。補給車は燃料を使わないので、
    これをずっと続けられます。

    相手がまだ今歩の行き先を決めていない（番号が後ろ）ときは、
    決めてある順路の先頭から読みます。どちらもないときは、
    その場に留まるのが正しいので None を返します。
    """

    here = positions[index]
    for other, kind in enumerate(kinds):
        if kind != AgentKind.TOURING or positions[other] != here:
            continue
        entry = pending[other]
        if entry is not None and entry[1] == "move":
            return (entry[2], other)
        route = plan_state[other].route
        if route:
            for direction, next_position in finder.neighbors[here]:
                if direction == route[0]:
                    return (next_position, other)
        return None
    return None


def _approach_supply_with_fuel(*, index, kinds, positions, fuels, remaining,
                               finder, road_status):
    """**燃料の範囲で**補給車にいちばん近づけるセル。なければ None。

    `_nearest_supply` は「補給車まで届く」ことを求めるので、空に
    近い車では使えません。ここは**届かなくてもかまいません**。
    近づいておけば迎えが早く済み、翌日の朝に拾われることもあります。
    """

    supply = [other for other, kind in enumerate(kinds)
              if kind == AgentKind.SUPPLY]
    if not supply:
        return None
    fuel = fuels[index]
    if fuel <= 0:
        return None
    here = positions[index]
    # 燃料をいちばん重く見た経路で測ります（1歩でも遠くへ行くため）。
    table = finder.table(here, road_status, FUEL_WEIGHT_STEPS)
    best = None
    for other in supply:
        goal = positions[other]
        if goal == here:
            return None
        goal_table = finder.table(goal, road_status)
        for cell, steps in table.steps.items():
            if steps > remaining or table.fuel[cell] > fuel:
                continue
            gap = goal_table.steps.get(cell)
            if gap is None:
                continue
            key = (gap, -steps)
            if best is None or key < best[0]:
                best = (key, cell)
    if best is None or best[1] == here:
        return None
    return best[1]


def _drift_toward_spot(*, index, position, fuel, remaining, stock,
                       taken_today, problem, finder, road_status):
    """燃料では届くが今日の歩では着けない店へ、**近づける先**を返す。

    理由は `DRIFT_TO_SPOT` に。店そのものには着けないので、
    「その店にいちばん近づけるセル」を返します。
    """

    table = finder.table(position, road_status, FUEL_WEIGHT_STEPS)
    goal = None
    for spot, left in stock.items():
        if left <= 0 or spot in taken_today:
            continue
        cost = table.fuel.get(spot)
        if cost is None or cost > fuel:
            continue
        gap = table.steps.get(spot)
        if gap is None or gap <= remaining:
            # 今日のうちに着けるなら、ここの仕事ではありません。
            continue
        if goal is None or gap < goal[0]:
            goal = (gap, spot)
    if goal is None:
        return None
    goal_table = finder.table(goal[1], road_status)
    best = None
    for cell, steps in table.steps.items():
        if steps > remaining or table.fuel[cell] > fuel:
            continue
        away = goal_table.steps.get(cell)
        if away is None:
            continue
        key = (away, -steps)
        if best is None or key < best[0]:
            best = (key, cell)
    if best is None or best[1] == position:
        return None
    return best[1]


def _salvage_target(index, position, fuel, remaining, stock,
                    taken_today, reserved, finder, road_status):
    """待つくらいなら行ける店。一番近いものを返します。

    **残り歩数の半分以内で着ける店だけ**を対象にします。
    ぎりぎりの店へ向かうと、着いた先で身動きが取れなくなり、
    かえって損をするためです。
    """

    table = finder.table(position, road_status)
    best = None
    best_steps = None
    for spot, left in stock.items():
        if left <= 0 or spot in taken_today:
            continue
        if reserved.get(spot, 0) >= left:
            continue
        if not table.reachable(spot):
            continue
        steps = table.steps[spot]
        if steps > remaining * SALVAGE_REACH or table.fuel[spot] > fuel:
            continue
        if best_steps is None or steps < best_steps:
            best = spot
            best_steps = steps
    return best


STRANDED_WAIT = 3

# **行き先を見失った補給車が、1度に待つ歩数。**
#
# 補給車は燃料を使わないので、止まっている歩はそのまま無駄です。
# 以前はここが「その日の終わりまで」で、16x16の1日目に32歩のうち
# 24歩を座って過ごしていました（`_wait_length` に実測つきの理由）。
SUPPLY_IDLE_WAIT = 1

# 行ける店があるのに待つときの長さ（残り歩数をこの数で割る）。
#
# **各サイズで最も成績の悪かった試合で測り直しました**（1位の玉数に対する率）。
#
#   1/16: 16x16 94% / 24x24 93% / 32x32 83%
#   **1/24: 16x16 96% / 24x24 93% / 32x32 93%**  ← 採用
#   1/32: 16x16 93% / 24x24 91% / 32x32 92%
#   1/48: 16x16 95% / 24x24 89% / 32x32 88%
#
# 32x32の3日目は、燃料が12〜18しかないのに全車が冒頭で8歩待っていました
# （128歩 ÷ 16）。待機が予算の54%を占め、獲得は上限122に対し38玉。
# 短く待って選び直すほど、補給や在庫の変化に追いつけます。
OPTION_WAIT_DIVISOR = 24

# この歩数以上の日は、待ちをさらに短くします。
#
# **実測で効かなかったので無効（999）にしています。**
#
# 実戦1476の1日目（99歩）で車2が19歩待って10玉、他は11〜13玉。
# 長い日は待ちを短くすべきに見えました。ところが
#
#   一律1/48  : 1468の日別が95→94
#   90歩以上  : 1476の日別が144→142
#   130歩以上 : 1476の日別は144のまま、玉が86%→75%
#   **無効**  : 1476は日別144・玉86%  ← 最良
#
# 待っているのは「次の良い店へ向かうタイミング待ち」で、
# 急がせると近場の1玉に飛びついて全体が崩れます。
LONG_DAY_STEPS = 999
LONG_DAY_WAIT_DIVISOR = 48


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

    # **(車, 店) ごとに1回だけ測ります。**
    #
    # 位置・燃料・残り歩数はこの関数のあいだ変わらないので、同じ
    # 組み合わせの答えは必ず同じです。ところがペアを全部試すので、
    # 「その車が自分の店へ行く値段」は何度も引き直していました。
    # 第1958（32x32・7体）の実測で1日ぶんに55万回。覚えておけば
    # 引き直しが消えます（答えは変わりません）。
    measured: dict[tuple[int, int], int | None] = {}
    tables: dict[int, object] = {}

    def cost_of(agent: int, target: int) -> int | None:
        """その巡回車がその店へ行くのにかかる歩数。届かなければ None。"""

        mark = (agent, target)
        if mark in measured:
            return measured[mark]
        table = tables.get(agent)
        if table is None:
            table = finder.table(
                positions[agent], road_status,
                fuel_weight(remaining, fuels[agent])
            )
            tables[agent] = table
        steps = table.steps.get(target)
        if steps is None or steps > remaining:
            measured[mark] = None
            return None
        if table.fuel[target] > fuels[agent]:
            measured[mark] = None
            return None
        measured[mark] = steps
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
    day_steps: int = 0,
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
            #
            # **ただし、待つだけで日が終わることがあります。**
            # 実戦1356の3日目で、車0は歩47から待ち始め、補給は歩108。
            # **61歩も待って**いました。187歩の日の3分の1です。
            #
            # 補給車がすぐ来ないなら、残った燃料で行ける店へ向かう
            # ほうが得です。待機は「動けないとき」の最後の手段に
            # すべきで、ここで長く止まるのは設計の誤りでした。
            return 1
        if plan_state[index].had_options:
            # 行ける店があるのに待つと決めた場合は、日の終わりまで固定しない。
            # 実測で 1/8 より 1/16 のほうが良い（玉844→857）。
            # 短く待って選び直すほど、状況の変化に追いつけます。
            # **歩数の多い日は、待ちを短くします。**
            #
            # 実戦1476の1日目（99歩）で、車2が19歩も待って10玉。
            # 他の車は11〜13玉でした。長い日ほど、待っている間に
            # 回れたはずの店が多くなります。
            #
            # ただし一律に短くすると日別を落とします（1468で95→94）。
            # **1日の歩数が多いときだけ**短くします。
            divisor = OPTION_WAIT_DIVISOR
            if day_steps >= LONG_DAY_STEPS:
                divisor = LONG_DAY_WAIT_DIVISOR
            return max(1, min(remaining, max(1, remaining // divisor)))
        # 行ける店が1つもない＝たいていは燃料切れ。ここで日の終わりまで
        # 待つと決めてしまうと、**途中で補給されても動き出せません。**
        #
        # 実戦1025では、132歩の日に全車が冒頭で16歩も待っていました
        # （remaining // 12 = 11〜16歩）。その間に補給車が来ても動けません。
        # 待機は予算の31〜45%にのぼります。
        # **短く待って、補給されたらすぐ動けるようにします。**
        return max(1, min(remaining, STRANDED_WAIT))
    elif plan_state[index].had_options:
        # 補給車も同じ。合流先の候補があるうちは、こまめに選び直す。
        return max(1, min(remaining, max(1, remaining // 8)))
    # **行き先が見つからなかった補給車は、短く待ちます。**
    #
    # ここは以前「その日の終わりまで」待っていました。ところが
    # `had_options` は方策が候補の数を返したときだけ真になり、
    # `GreedyPolicy` も `PlannedRoutePolicy` も `candidate_features=[]`
    # を返します（学習ずみの方策だけが返します）。つまり
    # **貪欲な補給車は、行き先を1度見失うとその日ずっと凍りました。**
    #
    # 第1627（16x16・32歩）の1日目の実測:
    #
    #   補給車の行動計画 [4, 4, 3, 4, 1, 1, -24]
    #   → 歩8で「残り24歩待機」を出し、以後何もしていません。
    #     歩15に巡回車0が来て補給は起きましたが、そのあとも
    #     17歩ただ座っています。その日の終わりに巡回車3は燃料2、
    #     巡回車2は燃料14。翌日から**2台が燃料切れで立ち往生**し、
    #     3日目は3台のうち2台が1歩も動けませんでした。
    #
    # 補給車は燃料を1滴も使わないので、長く待つのは必ず損です。
    # 短く待って選び直せば、車が動いた・燃料が減ったところで
    # すぐ向かい直せます。
    return max(1, min(remaining, SUPPLY_IDLE_WAIT))
