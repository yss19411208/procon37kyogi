"""公式の処理順に従うシミュレータ。

処理順は 質問回答集その1 Q6 および 補足資料(行動詳細) に完全に従います。

  0 ステップ目        : アクションフェーズのみ
  1 ～ N-1 ステップ目 : 反映フェーズ → アクションフェーズ
  N (最終) ステップ目 : 反映フェーズのみ

  反映フェーズの内訳: 燃料の消費 → 移動の反映 → うどんの獲得
                      → 燃料の補給 → 交通量の更新
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hexa.problem import Problem, walkable_neighbors
from hexa.rules import AgentKind, RoadStatus, Terrain, fuel_cost, move_steps


class InvalidAnswer(ValueError):
    """公式サーバーがリジェクトする回答に相当する例外。"""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__(" / ".join(reasons))
        self.reasons = reasons


@dataclass(frozen=True)
class Acquisition:
    step: int
    agent_index: int
    spot_position: int
    brand: int


@dataclass(frozen=True)
class Refuel:
    step: int
    agent_index: int
    position: int
    fuel_before: int
    fuel_after: int


@dataclass
class DayResult:
    positions: list[int]
    fuels: list[int]
    acquisitions: list[Acquisition]
    refuels: list[Refuel]
    stay_counts: dict[int, int]           # セル -> 全エージェントの滞在ステップ数合計
    remaining_stocks: dict[int, int]
    wasted_spot_stays: int                # 獲得済みスポットに留まった回数（学習用の指標）
    step_positions: list[list[int]] = field(default_factory=list)
    step_fuels: list[list[int]] = field(default_factory=list)


def _parse_action(raw: object) -> tuple[str, int]:
    """行動値を ('move', 方向) または ('wait', ステップ数) に変換する。

    フォーマット資料 p.3: -1 以下は待機（絶対値がステップ数）、0～5 は移動方向。
    """

    if type(raw) is not int:
        raise InvalidAnswer([f"行動は整数で指定してください: {raw!r}"])
    if raw <= -1:
        return ("wait", -raw)
    if 0 <= raw <= 5:
        return ("move", raw)
    raise InvalidAnswer([f"行動は -1 以下 または 0～5 です: {raw}"])


def simulate_day(
    problem: Problem,
    agent_kinds: tuple[AgentKind, ...],
    start_positions: tuple[int, ...],
    start_fuels: tuple[int, ...],
    action_plan: list[list[int]],
    day_steps: int,
    road_status: dict[int, RoadStatus],
    record_trace: bool = False,
) -> DayResult:
    """1日分を進める。1体でも不正なら回答全体をリジェクトする。"""

    agent_count = len(agent_kinds)
    if len(action_plan) != agent_count:
        raise InvalidAnswer(["行動計画の要素数がエージェント数と一致しません"])

    neighbors = walkable_neighbors(problem)
    positions = list(start_positions)
    fuels = list(start_fuels)
    kinds = list(agent_kinds)

    # 予約中の行動: (完了ステップ, 種類, 移動先, 燃料消費)
    pending: list[tuple[int, str, int, int] | None] = [None] * agent_count
    cursors = [0] * agent_count

    spot_by_position = {spot.position: spot for spot in problem.spots}
    stock = {spot.position: spot.stocks for spot in problem.spots}
    acquired_today: list[set[int]] = [set() for _ in range(agent_count)]

    acquisitions: list[Acquisition] = []
    refuels: list[Refuel] = []
    stay_counts: dict[int, int] = {}
    wasted_spot_stays = 0
    step_positions: list[list[int]] = []
    step_fuels: list[list[int]] = []

    def action_phase(step: int) -> None:
        for index in range(agent_count):
            if pending[index] is not None:
                continue
            plan = action_plan[index]
            if cursors[index] >= len(plan):
                raise InvalidAnswer(
                    [f"エージェント{index}: 行動計画が1日のステップ数より短いです"]
                )
            kind, value = _parse_action(plan[cursors[index]])
            cursors[index] += 1
            if kind == "wait":
                if step + value > day_steps:
                    raise InvalidAnswer(
                        [f"エージェント{index}: 待機がその日のステップ数を超えます"]
                    )
                pending[index] = (step + value, "wait", positions[index], 0)
                continue

            origin = positions[index]
            terrain = problem.terrain_at(origin)
            if terrain == Terrain.POND:
                raise InvalidAnswer([f"エージェント{index}: 池からは移動できません"])
            destination = None
            for direction, next_position in neighbors[origin]:
                if direction == value:
                    destination = next_position
                    break
            if destination is None:
                raise InvalidAnswer(
                    [f"エージェント{index}: 移動先が池またはマップ外です（方向{value}）"]
                )
            duration = move_steps(terrain, road_status.get(origin, RoadStatus.SMOOTH))
            if step + duration > day_steps:
                raise InvalidAnswer(
                    [f"エージェント{index}: 移動に必要なステップ数が不足しています"]
                )
            # 燃料の判定は「消費が起きる反映フェーズ」で行う。
            # 質問回答集その1 Q6 の 1.1 で消費、1.4 で補給という順序のため、
            # 移動の途中で補給を受けて足りるようになる場合がある。
            # （補足資料 巡回車A: セル1で燃料1のまま移動を予約し、
            #   2ステップ目に補給車から補給されて3になり、3ステップ目に2を消費する）
            cost = fuel_cost(terrain) if kinds[index] == AgentKind.TOURING else 0
            pending[index] = (step + duration, "move", destination, cost)

    def reflection_phase(step: int) -> None:
        nonlocal wasted_spot_stays

        # 1.1 燃料の消費 / 1.2 移動の反映
        for index in range(agent_count):
            entry = pending[index]
            if entry is None or entry[0] != step:
                continue
            _complete_step, kind, destination, cost = entry
            if kind == "move":
                if cost > fuels[index]:
                    raise InvalidAnswer(
                        [f"エージェント{index}: 燃料が不足しています（ステップ{step}）"]
                    )
                fuels[index] -= cost
                positions[index] = destination
            pending[index] = None

        # 1.3 うどんの獲得
        # 質問回答集その2 Q26: 在庫が不足する場合はリスト内の順番が若いエージェントが優先。
        for index in range(agent_count):
            if kinds[index] != AgentKind.TOURING:
                continue
            spot = spot_by_position.get(positions[index])
            if spot is None:
                continue
            if spot.position in acquired_today[index]:
                wasted_spot_stays += 1
                continue
            if stock[spot.position] <= 0:
                continue
            stock[spot.position] -= 1
            acquired_today[index].add(spot.position)
            acquisitions.append(
                Acquisition(step, index, spot.position, spot.brand)
            )

        # 1.4 燃料の補給
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
                refuels.append(
                    Refuel(step, index, positions[index], fuels[index], problem.fuel_limit)
                )
                fuels[index] = problem.fuel_limit

        # 1.5 交通量の更新（移動反映後の位置を数える / 質問回答集その2 Q27）
        for index in range(agent_count):
            position = positions[index]
            stay_counts[position] = stay_counts.get(position, 0) + 1

    action_phase(0)
    if record_trace:
        step_positions.append(list(positions))
        step_fuels.append(list(fuels))

    for step in range(1, day_steps + 1):
        reflection_phase(step)
        if record_trace:
            step_positions.append(list(positions))
            step_fuels.append(list(fuels))
        if step < day_steps:
            action_phase(step)

    reasons: list[str] = []
    for index in range(agent_count):
        if pending[index] is not None:
            reasons.append(f"エージェント{index}: 行動がその日の終わりに完了しません")
        if cursors[index] < len(action_plan[index]):
            reasons.append(f"エージェント{index}: 行動計画に余分な要素があります")
    if reasons:
        raise InvalidAnswer(reasons)

    return DayResult(
        positions=positions,
        fuels=fuels,
        acquisitions=acquisitions,
        refuels=refuels,
        stay_counts=stay_counts,
        remaining_stocks=stock,
        wasted_spot_stays=wasted_spot_stays,
        step_positions=step_positions,
        step_fuels=step_fuels,
    )


def road_status_for_day(
    problem: Problem,
    stay_counts_by_day: list[dict[int, int]],
) -> dict[int, RoadStatus]:
    """翌日の道路状態を求める。

    募集要項 p.9:
      交通量 = 各道路セルにおける全エージェントの前日と前々日の滞在ステップ数を
               全チーム分合算し、チーム数で割った値。
      1日目は全て順調、2日目は1日目の交通量のみで決まる。
      交通量 < 混雑基準値 → 順調
      混雑基準値 <= 交通量 < 渋滞基準値 → 混雑
      渋滞基準値 <= 交通量 → 渋滞

    浮動小数点の境界誤差を避けるため、割り算ではなく整数の掛け算で判定します。
    (合算 / players >= T) は (合算 >= T * players) と同値です。
    """

    recent = stay_counts_by_day[-2:]
    statuses: dict[int, RoadStatus] = {}
    busy_line = problem.busy_threshold * problem.players
    jammed_line = problem.jammed_threshold * problem.players
    for position in range(problem.cell_count):
        if problem.terrain_at(position) != Terrain.ROAD:
            continue
        total = sum(day_counts.get(position, 0) for day_counts in recent)
        if total >= jammed_line:
            statuses[position] = RoadStatus.JAMMED
        elif total >= busy_line:
            statuses[position] = RoadStatus.BUSY
        else:
            statuses[position] = RoadStatus.SMOOTH
    return statuses


@dataclass
class TeamAnswer:
    agent_kinds: tuple[AgentKind, ...]
    action_plans_by_day: list[list[list[int]]]


@dataclass
class TeamMatchResult:
    day_results: list[DayResult]
    road_status_by_day: list[dict[int, RoadStatus]]


def simulate_match(
    problem: Problem,
    answers: list[TeamAnswer],
    record_trace: bool = False,
) -> list[TeamMatchResult]:
    """全チームを同時に進める。道路状態は全チーム共通（募集要項 p.9）。

    スポットの在庫はチームごとに独立（募集要項 p.9）なので、日ごとの
    シミュレーションはチーム別に行い、交通量だけを合算します。
    """

    team_count = len(answers)
    if team_count == 0:
        raise ValueError("チームが1つもありません")

    positions = [list(problem.agent_positions) for _ in range(team_count)]
    fuels = [
        [
            problem.fuel_limit if kind == AgentKind.TOURING else 0
            for kind in answers[team].agent_kinds
        ]
        for team in range(team_count)
    ]
    results = [TeamMatchResult(day_results=[], road_status_by_day=[]) for _ in range(team_count)]
    combined_stay_by_day: list[dict[int, int]] = []

    for day_index, day_steps in enumerate(problem.day_steps):
        status = road_status_for_day(problem, combined_stay_by_day)
        combined: dict[int, int] = {}
        for team in range(team_count):
            answer = answers[team]
            day_result = simulate_day(
                problem=problem,
                agent_kinds=answer.agent_kinds,
                start_positions=tuple(positions[team]),
                start_fuels=tuple(fuels[team]),
                action_plan=answer.action_plans_by_day[day_index],
                day_steps=day_steps,
                road_status=status,
                record_trace=record_trace,
            )
            positions[team] = list(day_result.positions)
            fuels[team] = list(day_result.fuels)
            results[team].day_results.append(day_result)
            results[team].road_status_by_day.append(status)
            for cell, count in day_result.stay_counts.items():
                combined[cell] = combined.get(cell, 0) + count
        combined_stay_by_day.append(combined)

    return results
