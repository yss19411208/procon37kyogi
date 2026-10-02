"""公式サーバーに接続して1試合を戦う。

毎日、サーバーから配られた状態（自チームの位置・燃料・道路状態）をそのまま使って
行動計画を作り、提出します。

そのうえで、翌日にサーバーが返してきた位置・燃料と、こちらのシミュレータが予測した
値を必ず突き合わせます。ズレたらその場で警告を出します。ズレは「こちらの実装が
公式と違う」ことの証拠なので、見逃してはいけません。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from hexa.ai import Brain, Policy
from hexa.client import MatchState, ProconClient
from hexa.match import choose_agent_kinds
from hexa.pathfind import PathFinder
from hexa.planner import DayPlanner
from hexa.problem import Problem
from hexa.rules import AgentKind
from hexa.scoring import Score
from hexa.simulator import simulate_day


@dataclass
class OnlineDayReport:
    day: int
    plan: list[list[int]]
    revision: int
    plan_seconds: float
    predicted_positions: list[int]
    predicted_fuels: list[int]
    acquisitions: int
    refuels: int
    mismatch: str | None = None


@dataclass
class OnlineMatchReport:
    problem: Problem
    agent_kinds: tuple[AgentKind, ...]
    days: list[OnlineDayReport] = field(default_factory=list)
    score: Score = field(default_factory=lambda: Score(0, 0, 0))
    mismatches: list[str] = field(default_factory=list)


def _describe_mismatch(
    day: int,
    predicted_positions: list[int],
    predicted_fuels: list[int],
    state: MatchState,
) -> str | None:
    actual_positions = [agent.position for agent in state.agents]
    actual_fuels = [agent.fuel for agent in state.agents]
    problems: list[str] = []
    if actual_positions != predicted_positions:
        problems.append(f"位置 予測{predicted_positions} / 実際{actual_positions}")
    if actual_fuels != predicted_fuels:
        problems.append(f"燃料 予測{predicted_fuels} / 実際{actual_fuels}")
    if not problems:
        return None
    return f"{day + 1}日目終了時: " + " / ".join(problems)


def play_match(
    client: ProconClient,
    brain: Brain,
    seed: int = 20260910,
    log=print,
) -> OnlineMatchReport:
    rng = random.Random(seed)

    ok, message = client.check_connection()
    log(f"通信確認: {message}")
    if not ok:
        raise ConnectionError(message)
    problem, raw = client.wait_for_setting(log=log)
    log(f"試合設定を取得しました: {problem.height}x{problem.width} / "
        f"エージェント{problem.agent_count}体 / {problem.day_count}日 / players={problem.players}")

    kinds = choose_agent_kinds(brain, problem, rng, explore=0.0)
    supply_count = sum(1 for kind in kinds if kind == AgentKind.SUPPLY)
    client.post_agent_kinds(kinds)
    log(f"エージェント種別を提出しました: 巡回車{problem.agent_count - supply_count}台 / "
        f"補給車{supply_count}台")

    report = OnlineMatchReport(problem=problem, agent_kinds=kinds)

    finder = PathFinder(problem)
    planner = DayPlanner(problem, finder)
    policy = Policy(brain, problem, rng, epsilon=0.0)
    acquired_brands: set[int] = set()
    brand_total = 0
    daily_brand_total = 0
    noodle_total = 0

    for day_index, day_steps in enumerate(problem.day_steps):
        state = client.wait_for_day(day_index)
        positions = [agent.position for agent in state.agents]
        fuels = [agent.fuel for agent in state.agents]

        started = time.monotonic()
        day_plan = planner.plan_day(
            kinds=kinds,
            positions=positions,
            fuels=fuels,
            road_status=state.road_status,
            day_steps=day_steps,
            policy=policy,
            acquired_brands=acquired_brands,
            day_index=day_index,
            day_count=problem.day_count,
        )
        # 提出前に必ず自前のシミュレータで検証する。ここで落ちたら送らない。
        day_result = simulate_day(
            problem=problem,
            agent_kinds=kinds,
            start_positions=tuple(positions),
            start_fuels=tuple(fuels),
            action_plan=day_plan.action_plan,
            day_steps=day_steps,
            road_status=state.road_status,
        )
        plan_seconds = time.monotonic() - started

        revision = client.post_actions(day_plan.action_plan)
        day_brands = {event.brand for event in day_result.acquisitions}
        new_brands = day_brands - acquired_brands
        acquired_brands |= day_brands
        brand_total = len(acquired_brands)
        daily_brand_total += len(day_brands)
        noodle_total += len(day_result.acquisitions)

        log(
            f"{day_index + 1}日目: 提出しました（revision={revision}, {plan_seconds:.2f}秒）"
            f" 獲得{len(day_result.acquisitions)}玉 / 新しい系列{len(new_brands)} /"
            f" 補給{len(day_result.refuels)}回"
        )

        report.days.append(
            OnlineDayReport(
                day=day_index,
                plan=day_plan.action_plan,
                revision=revision,
                plan_seconds=plan_seconds,
                predicted_positions=list(day_result.positions),
                predicted_fuels=list(day_result.fuels),
                acquisitions=len(day_result.acquisitions),
                refuels=len(day_result.refuels),
            )
        )

        if day_index + 1 < problem.day_count:
            next_state = client.wait_for_day(day_index + 1)
            mismatch = _describe_mismatch(
                day_index,
                list(day_result.positions),
                list(day_result.fuels),
                next_state,
            )
            if mismatch:
                report.days[-1].mismatch = mismatch
                report.mismatches.append(mismatch)
                log("  ずれを検出しました → " + mismatch)

    report.score = Score(brand_total, daily_brand_total, noodle_total)
    return report
