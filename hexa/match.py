"""4体のAIで1試合を進める。

流れは公式と同じです。
  1. 各AIがエージェントの種類（巡回車 / 補給車）を決める。
  2. 各日の開始時に、全チーム共通の道路状態が決まる。
  3. 各AIがその日の行動計画を作る。
  4. 公式と同じシミュレータで検証しながら1日進める。
  5. 次の日の道路状態は、全チームの滞在ステップ数を合算して決める。

プランナーが作った計画は、必ず公式シミュレータ simulate_day で再検証します。
プランナー側にバグがあれば InvalidAnswer で必ず落ちるので、
「不正な行動をこっそり出す」ことが起きません。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from hexa.ai import Brain, Policy
from hexa.pathfind import PathFinder
from hexa.planner import DayPlanner
from hexa.problem import Problem
from hexa.rules import AgentKind, RoadStatus
from hexa.opponents import OpponentModel
from hexa.scoring import Score, rank_scores, score_days
from hexa.simulator import DayResult, road_status_for_day, simulate_day

REWARD_DISCOUNT = 0.9

# 日をまたぐときの割引。0 なら「その日の中だけ」で報酬を締める。
#
# 1日を超えて価値を伝えれば「今日は取れないが明日のために動く」も学習できる、
# という狙いで 0.85 を試しましたが、実測では弱くなりました。
# 報酬の大きさが日数ぶん積み上がって桁が変わり、同じ学習率では
# 重みが振れすぎるのが原因だと考えています。
#
#   DAY_DISCOUNT = 0.0  : AIだけで玉170.2（21特徴量・300エピソード）
#   DAY_DISCOUNT = 0.85 : AIだけで玉154.4（68特徴量・390エピソード）
#
# 原因は報酬の桁でした。日数ぶん積み上がると1日だけのときの数倍になり、
# 同じ学習率では重みが振れすぎます。いまは下の DAY_RETURN_SCALE で
# 桁を1日分に戻してから学習させるようにしてあります。
DAY_DISCOUNT = 0.0


def _day_return_scale() -> float:
    """日をまたいだ報酬を、1日分と同じ桁に戻す係数。

    DAY_DISCOUNT を上げると報酬が 1/(1-DAY_DISCOUNT) 倍くらいに膨らむので、
    その分だけ縮めます。これをやらないと学習率を下げ直す必要が出ます。
    """

    if DAY_DISCOUNT <= 0.0:
        return 1.0
    return 1.0 - DAY_DISCOUNT
SUPPLY_SUCCESS_SCALE = 3.0
SUPPLY_FAILURE_PENALTY = -0.1


@dataclass
class TeamRun:
    brain_name: str
    agent_kinds: tuple[AgentKind, ...]
    action_plans_by_day: list[list[list[int]]]
    day_results: list[DayResult] = field(default_factory=list)
    score: Score = field(default_factory=lambda: Score(0, 0, 0))
    rank: int = 0
    touring_samples: list[tuple[list[float], list[float]]] = field(default_factory=list)
    # (エージェント, 日, ステップ, 特徴量, その場で得た報酬)
    touring_pending: list[tuple[int, int, int, list[float], list[float]]] = field(
        default_factory=list
    )
    recipe_results: list[tuple[int, float]] = field(default_factory=list)
    supply_samples: list[tuple[list[float], float]] = field(default_factory=list)
    effective_refuels: int = 0
    wasted_refuels: int = 0
    supply_count: int = 0


@dataclass
class MatchResult:
    problem: Problem
    teams: list[TeamRun]
    road_status_by_day: list[dict[int, RoadStatus]]


def choose_agent_kinds(
    brain: Brain,
    problem: Problem,
    rng: random.Random,
    explore: float,
) -> tuple[AgentKind, ...]:
    """補給車の台数を学習済みの統計から決め、後ろのエージェントを補給車にする。"""

    supply_count = brain.best_supply_count(problem.agent_count, rng, explore)
    supply_count = max(0, min(problem.agent_count - 1, supply_count))
    kinds = [AgentKind.TOURING] * problem.agent_count
    for index in range(problem.agent_count - supply_count, problem.agent_count):
        kinds[index] = AgentKind.SUPPLY
    return tuple(kinds)


def run_match(
    problem: Problem,
    brains: list[Brain],
    seed: int,
    epsilon: float = 0.0,
    kind_explore: float = 0.0,
    record_trace: bool = False,
) -> MatchResult:
    finder = PathFinder(problem)
    planner = DayPlanner(problem, finder)
    team_count = len(brains)
    rngs = [random.Random(seed + 104729 * (index + 1)) for index in range(team_count)]

    def scaled(value: float, index: int) -> float:
        return max(0.0, min(0.9, value * brains[index].explore_scale))

    kinds_by_team = [
        choose_agent_kinds(brains[index], problem, rngs[index], scaled(kind_explore, index))
        for index in range(team_count)
    ]
    policies = [
        Policy(brains[index], problem, rngs[index], scaled(epsilon, index))
        for index in range(team_count)
    ]

    runs = [
        TeamRun(
            brain_name=brains[index].name,
            agent_kinds=kinds_by_team[index],
            action_plans_by_day=[],
            supply_count=sum(1 for k in kinds_by_team[index] if k == AgentKind.SUPPLY),
        )
        for index in range(team_count)
    ]

    positions = [list(problem.agent_positions) for _ in range(team_count)]
    fuels = [
        [problem.fuel_limit if kind == AgentKind.TOURING else 0 for kind in kinds_by_team[index]]
        for index in range(team_count)
    ]
    acquired_brands = [set() for _ in range(team_count)]
    combined_stays: list[dict[int, int]] = []
    road_status_by_day: list[dict[int, RoadStatus]] = []
    # チームごとの滞在数の履歴。自分の分と相手の分を分けて持つ。
    stays_by_team: list[list[dict[int, int]]] = [[] for _ in range(team_count)]
    models = [OpponentModel(problem) for _ in range(team_count)]

    for day_index, day_steps in enumerate(problem.day_steps):
        road_status = road_status_for_day(problem, combined_stays)
        road_status_by_day.append(road_status)
        combined: dict[int, int] = {}

        for team in range(team_count):
            own_window = stays_by_team[team][-2:]
            if day_index > 0:
                models[team].observe_status(road_status, own_window)
                models[team].observe_positions(
                    [
                        position
                        for other in range(team_count)
                        if other != team
                        for position in positions[other]
                    ]
                )
            congestion_risk = models[team].congestion_risk(own_window)
            road_heat = {
                position: value
                for position, value in models[team].hot_roads(limit=60)
            }
            day_plan = planner.plan_day(
                kinds=kinds_by_team[team],
                positions=positions[team],
                fuels=fuels[team],
                road_status=road_status,
                day_steps=day_steps,
                policy=policies[team],
                acquired_brands=acquired_brands[team],
                day_index=day_index,
                day_count=problem.day_count,
                congestion_risk=congestion_risk,
                road_heat=road_heat,
            )
            # 公式シミュレータで再検証する。ここで落ちたらプランナーのバグ。
            day_result = simulate_day(
                problem=problem,
                agent_kinds=kinds_by_team[team],
                start_positions=tuple(positions[team]),
                start_fuels=tuple(fuels[team]),
                action_plan=day_plan.action_plan,
                day_steps=day_steps,
                road_status=road_status,
                record_trace=record_trace,
            )
            runs[team].action_plans_by_day.append(day_plan.action_plan)
            runs[team].day_results.append(day_result)

            _collect_samples(
                run=runs[team],
                day_plan=day_plan,
                day_result=day_result,
                acquired_brands=acquired_brands[team],
                day_steps=day_steps,
                fuel_limit=problem.fuel_limit,
                day_index=day_index,
            )

            for event in day_result.acquisitions:
                acquired_brands[team].add(event.brand)
            positions[team] = list(day_result.positions)
            fuels[team] = list(day_result.fuels)
            stays_by_team[team].append(dict(day_result.stay_counts))
            for cell, count in day_result.stay_counts.items():
                combined[cell] = combined.get(cell, 0) + count

        combined_stays.append(combined)

    for team in range(team_count):
        _finalise_touring_samples(runs[team])

    scores = []
    for team in range(team_count):
        runs[team].score = score_days(runs[team].day_results)
        scores.append(runs[team].score)
    for team, rank in enumerate(rank_scores(scores)):
        runs[team].rank = rank

    return MatchResult(problem=problem, teams=runs, road_status_by_day=road_status_by_day)


def _finalise_touring_samples(run: TeamRun) -> None:
    """試合全体を通した割引報酬を、エージェントごとに後ろから計算する。

    1日の中は REWARD_DISCOUNT、日をまたぐところは DAY_DISCOUNT で割り引きます。
    こうすると「今日は取れないが、明日以降のために動いておく」という判断にも
    値がつくので、日をまたいだ戦い方が学習できます。
    """

    by_agent: dict[int, list] = {}
    for entry in run.touring_pending:
        by_agent.setdefault(entry[0], []).append(entry)

    for entries in by_agent.values():
        entries.sort(key=lambda item: (item[1], item[2]))
        running = [0.0, 0.0, 0.0]
        previous_day = None
        for agent_index, day_index, step, features, immediate in reversed(entries):
            if previous_day is not None and day_index != previous_day:
                for head in range(3):
                    running[head] *= DAY_DISCOUNT
            previous_day = day_index
            for head in range(3):
                running[head] = immediate[head] + REWARD_DISCOUNT * running[head]
            # 玉数は種類数より桁が大きくなりやすいので、頭を揃えて正規化する。
            scale = _day_return_scale()
            run.touring_samples.append(
                (
                    features,
                    [
                        running[0] * scale,
                        running[1] * 0.5 * scale,
                        running[2] * 0.2 * scale,
                    ],
                )
            )
    run.touring_pending.clear()


def _collect_samples(
    *,
    run: TeamRun,
    day_plan,
    day_result: DayResult,
    acquired_brands: set[int],
    day_steps: int,
    fuel_limit: int,
    day_index: int,
) -> None:
    """1日分の判断に、その場で起きた結果を記録する。

    ここでは「その判断の直後に何が取れたか」だけを残します。
    試合全体を通した価値は、試合が終わってから
    _finalise_touring_samples がまとめて計算します。
    """

    # ---- 巡回車 ----
    by_agent: dict[int, list] = {}
    for choice in day_plan.touring_choices:
        by_agent.setdefault(choice.agent_index, []).append(choice)

    acquisitions_by_agent: dict[int, list] = {}
    for event in day_result.acquisitions:
        acquisitions_by_agent.setdefault(event.agent_index, []).append(event)

    known_brands = set(acquired_brands)
    brands_today: set[int] = set()

    for agent_index, choices in by_agent.items():
        choices.sort(key=lambda c: c.step)
        events = sorted(acquisitions_by_agent.get(agent_index, []), key=lambda e: e.step)
        boundaries = [choice.step for choice in choices] + [day_steps + 1]

        immediate: list[list[float]] = []
        for index in range(len(choices)):
            low, high = boundaries[index], boundaries[index + 1]
            window = [event for event in events if low < event.step <= high]
            new_brand = 0.0
            new_daily = 0.0
            for event in window:
                if event.brand not in known_brands:
                    new_brand += 1.0
                    known_brands.add(event.brand)
                if event.brand not in brands_today:
                    new_daily += 1.0
                    brands_today.add(event.brand)
            immediate.append([new_brand, new_daily, float(len(window))])

        for choice, reward in zip(choices, immediate):
            run.touring_pending.append(
                (choice.agent_index, day_index, choice.step, choice.features, reward)
            )
            # 作戦ごとの成績。悪い結果もそのまま残す。
            index = getattr(choice, "recipe_index", -1)
            if index >= 0:
                run.recipe_results.append((index, reward[2]))

    # ---- 補給車 ----
    refuels_by_agent: dict[int, list[int]] = {}
    for event in day_result.refuels:
        refuels_by_agent.setdefault(event.agent_index, []).append(event.step)

    supply_by_agent: dict[int, list] = {}
    for choice in day_plan.supply_choices:
        supply_by_agent.setdefault(choice.agent_index, []).append(choice)

    for agent_index, choices in supply_by_agent.items():
        choices.sort(key=lambda c: c.step)
        boundaries = [choice.step for choice in choices] + [day_steps + 1]
        for index, choice in enumerate(choices):
            if choice.target_agent is None:
                run.supply_samples.append((choice.features, 0.0))
                continue
            low, high = boundaries[index], boundaries[index + 1]
            refuel_steps = [
                step
                for step in refuels_by_agent.get(choice.target_agent, [])
                if low < step <= high
            ]
            if not refuel_steps:
                # 合流できなかった場合、相手が本当に動けなくなっていた（stuck）
                # ときだけ減点する。単に近くへ寄っただけの動きは、燃料も使わず
                # 損もしていないので0にする。ここを一律に減点すると、
                # 補給車が「動かないほうが得」を学んで一日じゅう止まる。
                was_stuck = len(choice.features) > 6 and choice.features[6] >= 1.0
                penalty = SUPPLY_FAILURE_PENALTY if was_stuck else 0.0
                run.supply_samples.append((choice.features, penalty))
                run.wasted_refuels += 1
                continue
            first_refuel = min(refuel_steps)
            gained = sum(
                1
                for event in day_result.acquisitions
                if event.agent_index == choice.target_agent and event.step > first_refuel
            )
            if gained > 0:
                run.effective_refuels += 1
            else:
                run.wasted_refuels += 1
            reward = min(1.0, gained / SUPPLY_SUCCESS_SCALE)
            run.supply_samples.append((choice.features, reward))
