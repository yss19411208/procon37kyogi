"""公式フォーマットの回答を書き出す。

フォーマット資料 p.3:
  ・エージェント種別の回答は 0/1 の配列（例 [0, 1, 0, 1]）
  ・行動計画の回答は 全エージェント分の行動列の配列
"""

from __future__ import annotations

import json
from pathlib import Path

from hexa.match import MatchResult


def write_answers(result: MatchResult, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for team in result.teams:
        team_dir = output_dir / team.brain_name
        team_dir.mkdir(parents=True, exist_ok=True)

        kinds_path = team_dir / "agent_kinds.json"
        with open(kinds_path, "w", encoding="utf-8") as handle:
            json.dump([int(kind) for kind in team.agent_kinds], handle)
        written.append(kinds_path)

        for day_index, plan in enumerate(team.action_plans_by_day):
            path = team_dir / f"day_{day_index + 1}_actions.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(plan, handle)
            written.append(path)

        summary_path = team_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "name": team.brain_name,
                    "rank": team.rank,
                    "supplyCount": team.supply_count,
                    "score": {
                        "brand": team.score.brand_count,
                        "dailyBrand": team.score.daily_brand_total,
                        "noodles": team.score.noodle_count,
                    },
                    "effectiveRefuels": team.effective_refuels,
                    "wastedRefuels": team.wasted_refuels,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        written.append(summary_path)
    return written
