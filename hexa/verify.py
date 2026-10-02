"""公式の簡易版回答用サーバーと、このプログラムのシミュレータを突き合わせる。

4体のAIで実際に1試合を戦い、毎日2つのことを確認します。

  1. 翌日にサーバーが返す自チームの位置・燃料が、こちらの予測と一致するか
     → 移動時間・燃料消費・補給・うどん獲得の判定が公式と同じかの検証
  2. 翌日にサーバーが返す道路状態が、こちらの予測と一致するか
     → 交通量の数え方（滞在ステップ数の合算 ÷ チーム数）が公式と同じかの検証

1件でもズレたら、それはこのプログラムのバグです。必ず直してください。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from hexa.ai import Brain, Policy
from hexa.client import ProconClient
from hexa.match import choose_agent_kinds
from hexa.pathfind import PathFinder
from hexa.planner import DayPlanner
from hexa.problem import Problem
from hexa.rules import AgentKind, RoadStatus
from hexa.scoring import Score
from hexa.simulator import road_status_for_day, simulate_day

import random


@dataclass
class VerifyReport:
    problem: Problem
    day_count: int
    state_checks: int = 0
    traffic_checks: int = 0
    mismatches: list[str] = field(default_factory=list)
    scores: list[tuple[str, Score]] = field(default_factory=list)
    plan_seconds: list[float] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches


def verify_against_server(
    base_url: str,
    tokens: list[str],
    brains: list[Brain],
    problem_hint: Problem | None = None,
    seed: int = 20260910,
    log=print,
) -> VerifyReport:
    clients = [ProconClient(base_url, token) for token in tokens]
    problem, _raw = clients[0].get_setting()
    report = VerifyReport(problem=problem, day_count=problem.day_count)

    if problem_hint is not None and problem_hint.cells != problem.cells:
        report.mismatches.append("サーバーの問題が渡した設定と違います")

    rngs = [random.Random(seed + 977 * (index + 1)) for index in range(len(brains))]
    kinds = [
        choose_agent_kinds(brains[index], problem, rngs[index], explore=0.0)
        for index in range(len(brains))
    ]
    for index, client in enumerate(clients):
        client.post_agent_kinds(kinds[index])
    log("エージェント種別を提出しました。")

    finder = PathFinder(problem)
    planner = DayPlanner(problem, finder)
    policies = [
        Policy(brains[index], problem, rngs[index], epsilon=0.0)
        for index in range(len(brains))
    ]
    acquired_brands: list[set[int]] = [set() for _ in brains]
    totals = [[0, 0, 0] for _ in brains]
    combined_stays: list[dict[int, int]] = []
    predicted_end: list[tuple[list[int], list[int]]] = [([], []) for _ in brains]

    # 1日ぶんの回答時間の3倍まで待って進まなければ、止まったとみなして落とす。
    # だんまりで固まるより、原因が分かる形で失敗したほうがよい。
    day_timeout = max(30.0, max(problem.day_seconds) * 3.0 + 30.0)

    for day_index, day_steps in enumerate(problem.day_steps):
        states = [
            client.wait_for_day(day_index, timeout=day_timeout) for client in clients
        ]

        # --- 検証1: 前日の終わりの位置・燃料 ---
        if day_index > 0:
            for team, state in enumerate(states):
                actual_positions = [agent.position for agent in state.agents]
                actual_fuels = [agent.fuel for agent in state.agents]
                expected_positions, expected_fuels = predicted_end[team]
                report.state_checks += 1
                if actual_positions != expected_positions:
                    report.mismatches.append(
                        f"{day_index}日目終了時 {brains[team].name} の位置: "
                        f"予測{expected_positions} / サーバー{actual_positions}"
                    )
                if actual_fuels != expected_fuels:
                    report.mismatches.append(
                        f"{day_index}日目終了時 {brains[team].name} の燃料: "
                        f"予測{expected_fuels} / サーバー{actual_fuels}"
                    )

        # --- 検証2: 道路状態 ---
        expected_road = road_status_for_day(problem, combined_stays)
        actual_road = states[0].road_status
        report.traffic_checks += 1
        differences = []
        for position, expected in sorted(expected_road.items()):
            actual = actual_road.get(position, RoadStatus.SMOOTH)
            if actual != expected:
                differences.append(f"セル{position} 予測{expected.name}/サーバー{actual.name}")
        if differences:
            report.mismatches.append(
                f"{day_index + 1}日目の道路状態: " + " , ".join(differences[:6])
                + (f" ほか{len(differences) - 6}件" if len(differences) > 6 else "")
            )
        for team in range(1, len(states)):
            if states[team].road_status != actual_road:
                report.mismatches.append(
                    f"{day_index + 1}日目: チームによって道路状態が違います"
                    f"（{brains[team].name}）"
                )

        # --- その日を解いて提出する ---
        combined: dict[int, int] = {}
        for team, state in enumerate(states):
            positions = [agent.position for agent in state.agents]
            fuels = [agent.fuel for agent in state.agents]
            started = time.monotonic()
            day_plan = planner.plan_day(
                kinds=kinds[team],
                positions=positions,
                fuels=fuels,
                road_status=state.road_status,
                day_steps=day_steps,
                policy=policies[team],
                acquired_brands=acquired_brands[team],
                day_index=day_index,
                day_count=problem.day_count,
            )
            day_result = simulate_day(
                problem=problem,
                agent_kinds=kinds[team],
                start_positions=tuple(positions),
                start_fuels=tuple(fuels),
                action_plan=day_plan.action_plan,
                day_steps=day_steps,
                road_status=state.road_status,
            )
            report.plan_seconds.append(time.monotonic() - started)
            clients[team].post_actions(day_plan.action_plan)

            predicted_end[team] = (list(day_result.positions), list(day_result.fuels))
            day_brands = {event.brand for event in day_result.acquisitions}
            acquired_brands[team] |= day_brands
            totals[team][1] += len(day_brands)
            totals[team][2] += len(day_result.acquisitions)
            for cell, count in day_result.stay_counts.items():
                combined[cell] = combined.get(cell, 0) + count

        combined_stays.append(combined)
        log(f"{day_index + 1}日目 提出しました。")

    for team, brain in enumerate(brains):
        report.scores.append(
            (brain.name, Score(len(acquired_brands[team]), totals[team][1], totals[team][2]))
        )
    return report
