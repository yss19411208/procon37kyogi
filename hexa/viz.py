"""ビューア用データの書き出し。

ビューアは viewer/index.html です。ここで作った JSON を
viewer/data/ に置き、viewer/data/index.js に一覧を書き出します。
ブラウザで file:// で開いても読めるように、JSON と JS の両方を出します。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from hexa.match import MatchResult
from hexa.rules import AgentKind


def build_visualization(result: MatchResult, title: str) -> dict:
    problem = result.problem
    spot_brand = {spot.position: spot.brand for spot in problem.spots}

    teams = []
    for team in result.teams:
        days = []
        for day_index, day_result in enumerate(team.day_results):
            road = result.road_status_by_day[day_index]
            days.append(
                {
                    "roadStatus": [
                        {"pos": position, "status": int(status)}
                        for position, status in sorted(road.items())
                    ],
                    "steps": [
                        {"positions": positions, "fuels": fuels}
                        for positions, fuels in zip(
                            day_result.step_positions, day_result.step_fuels
                        )
                    ],
                    "acquisitions": [
                        {
                            "step": event.step,
                            "agent": event.agent_index,
                            "pos": event.spot_position,
                            "brand": event.brand,
                        }
                        for event in day_result.acquisitions
                    ],
                    "refuels": [
                        # **入った量も残します。** 日ごとの一覧で
                        # 「その日に何燃料配れたか」を出すのに使います。
                        # 準決勝32x32の解析では、1位に並ぶのに要るのは
                        # 1台1日あたり20燃料だけでした。
                        {"step": event.step, "agent": event.agent_index,
                         "pos": event.position,
                         "before": event.fuel_before,
                         "after": event.fuel_after}
                        for event in day_result.refuels
                    ],
                    "stayCounts": [
                        {"pos": position, "count": count}
                        for position, count in sorted(day_result.stay_counts.items())
                    ],
                }
            )
        teams.append(
            {
                "name": team.brain_name,
                "kinds": [int(kind) for kind in team.agent_kinds],
                "supplyCount": team.supply_count,
                "effectiveRefuels": team.effective_refuels,
                "wastedRefuels": team.wasted_refuels,
                "score": {
                    "brand": team.score.brand_count,
                    "dailyBrand": team.score.daily_brand_total,
                    "noodles": team.score.noodle_count,
                },
                "rank": team.rank,
                "days": days,
            }
        )

    return {
        "schemaVersion": 1,
        "title": title,
        "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "problem": {
            "height": problem.height,
            "width": problem.width,
            "cells": list(problem.cells),
            "spots": [
                {"pos": spot.position, "brand": spot.brand, "stocks": spot.stocks}
                for spot in problem.spots
            ],
            "agents": list(problem.agent_positions),
            "fuelLimit": problem.fuel_limit,
            "players": problem.players,
            "busyThreshold": problem.busy_threshold,
            "jammedThreshold": problem.jammed_threshold,
            "daySteps": list(problem.day_steps),
        },
        "brandCount": len(set(spot_brand.values())) if spot_brand else 0,
        "teams": teams,
        "overlapRate": route_overlap_rate(result),
    }


def route_overlap_rate(result: MatchResult) -> float:
    """チーム間で、同じ日・同じステップ・同じ号車が同じセルにいた割合。

    4体のAIが本当に別々に動いているかを、印象ではなく数字で確認するための指標です。
    """

    teams = result.teams
    if len(teams) < 2:
        return 0.0
    same = 0
    total = 0
    for day_index in range(len(teams[0].day_results)):
        traces = [team.day_results[day_index].step_positions for team in teams]
        if any(not trace for trace in traces):
            continue
        step_count = min(len(trace) for trace in traces)
        agent_count = min(len(trace[0]) for trace in traces)
        for step in range(step_count):
            for agent in range(agent_count):
                values = [trace[step][agent] for trace in traces]
                for i in range(len(values)):
                    for j in range(i + 1, len(values)):
                        total += 1
                        if values[i] == values[j]:
                            same += 1
    if total == 0:
        return 0.0
    return same / total


def write_visualization(payload: dict, viewer_dir: Path, slug: str) -> Path:
    data_dir = viewer_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / f"{slug}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    js_path = data_dir / f"{slug}.js"
    with open(js_path, "w", encoding="utf-8") as handle:
        handle.write("window.HEXA_DATASETS = window.HEXA_DATASETS || {};\n")
        handle.write(f"window.HEXA_DATASETS[{json.dumps(slug)}] = ")
        json.dump(payload, handle, ensure_ascii=False)
        handle.write(";\n")
    _rewrite_index(data_dir)
    return json_path


def _rewrite_index(data_dir: Path) -> None:
    entries = []
    for path in sorted(data_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        entries.append(
            {
                "slug": path.stem,
                "title": payload.get("title", path.stem),
                "createdAt": payload.get("createdAt", ""),
            }
        )
    entries.sort(key=lambda item: item["createdAt"], reverse=True)
    with open(data_dir / "index.js", "w", encoding="utf-8") as handle:
        handle.write("window.HEXA_INDEX = ")
        json.dump(entries, handle, ensure_ascii=False)
        handle.write(";\n")
    with open(data_dir / "index.json", "w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False)
