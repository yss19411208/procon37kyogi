"""練習場の試合を、あとから調べられるように記録する。

**本番モードでは使いません。** 1日60秒しかないので、記録に時間を
使うわけにいかないからです。メニュー15（練習場）専用です。

試合中は配列に貯めるだけで、**終わってから1回だけ書き出します。**
探索の速度には影響しません。

書き出し先は `logs/match_<試合ID>.json`。
1試合1ファイルなので、あとで欲しい試合だけ送れます。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from hexa.problem import Problem
from hexa.rules import AgentKind

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class MatchLog:
    """1試合分の記録。失敗しても試合は止めない。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.header: dict = {}
        self.days: list[dict] = []
        self.trials: list[dict] = []
        self.started = time.time()

    # ---------------- 記録 ----------------

    def start(self, problem: Problem, kinds: tuple[AgentKind, ...],
              match_id: object, trials: int, trial_seconds: float,
              best_label: str, best_score: tuple) -> None:
        """試合の始めに1回。事前試走の結果もここに残す。"""

        if not self.enabled:
            return
        self.started = time.time()
        self.days = []
        self.header = {
            "matchId": match_id,
            "height": problem.height,
            "width": problem.width,
            "dayCount": problem.day_count,
            "daySteps": list(problem.day_steps),
            "players": problem.players,
            "fuelLimit": problem.fuel_limit,
            "busyThreshold": problem.busy_threshold,
            "jammedThreshold": problem.jammed_threshold,
            "brandCount": len({spot.brand for spot in problem.spots}),
            "spots": [
                {"pos": spot.position, "brand": spot.brand, "stocks": spot.stocks}
                for spot in problem.spots
            ],
            "cells": list(problem.cells),
            "agentKinds": [int(kind) for kind in kinds],
            # 事前試走（補給車の台数決め）の様子
            "trialCount": trials,
            "trialSeconds": round(trial_seconds, 1),
            "trialBest": best_label,
            "trialScore": list(best_score) if best_score else None,
        }

    def note_trial(self, label: str, score: tuple) -> None:
        """探索の1件。上位だけ残すので、全部渡して構いません。"""

        if not self.enabled:
            return
        self.trials.append({"label": label, "score": list(score)})

    def finish_day(
        self,
        day_index: int,
        label: str,
        candidates: int,
        resubmissions: int,
        day_steps: int,
        parallel_plans: int,
        start_positions: list[int],
        start_fuels: list[int],
        positions: list[int],
        fuels: list[int],
        acquisitions: list,
        refuels: list,
        tracks: list[list[int]],
        road_status: dict,
        problem: Problem,
        kinds: tuple[AgentKind, ...],
        finder,
        missed: list[dict],
        brand_total: int,
        daily_total: int,
        noodle_total: int,
        elapsed: float,
        enemies: list[list[int]] | None = None,
    ) -> None:
        """1日が終わったところで呼ぶ。"""

        if not self.enabled:
            return
        top = sorted(self.trials, key=lambda item: item["score"], reverse=True)[:5]
        self.trials = []
        # **その日がどれだけ厳しかったか**を数字で残します。
        # あとから「なぜ取れなかったか」を推測せずに読めるようにするためです。
        jammed = sum(1 for value in road_status.values() if int(value) >= 2)
        busy = sum(1 for value in road_status.values() if int(value) == 1)
        touring = [i for i, kind in enumerate(kinds) if kind == AgentKind.TOURING]
        # **その日に取りうる上限。**
        #
        # 以前は「各店の在庫 × 巡回車の数」で数えていましたが、
        # それは**全店を全台が回れる前提**で、現実には届きません。
        # 実戦で「上限33／獲得14」と出ていたのに、1台は1日に5〜6店しか
        # 回れていませんでした。
        #
        # 取れる量は次の2つのうち小さいほうで決まります。
        #   在庫の都合: 各店の在庫と巡回車の数の小さいほう（1台1日1玉まで）
        #   時間の都合: 1台が1日に回れる店の数 × 台数
        stock_cap = sum(min(spot.stocks, len(touring)) for spot in problem.spots)
        capacity = stock_cap
        # 1台が1日に回れる店の数を、店どうしの距離から見積もる。
        reachable_cap = stock_cap
        try:
            spots = [spot.position for spot in problem.spots]
            if len(spots) >= 2:
                gaps = []
                for origin in spots[:8]:
                    table = finder.table(origin, road_status, 0)
                    for other in spots:
                        if other != origin and table.reachable(other):
                            gaps.append(table.steps[other])
                if gaps:
                    gaps.sort()
                    # **下位10%の距離を使います。**
                    #
                    # 中央値で割ると実際の獲得を大きく下回りました
                    # （上限24に対し獲得60）。良い計画は「近い店を
                    # 連続で回る」ので、平均的な距離では進みません。
                    # 下位25%でもまだ足りず（上限52に対し獲得60）、
                    # 下位10%にしてようやく実際を下回らなくなります。
                    typical = gaps[max(0, len(gaps) // 10)]
                    per_car = max(1, day_steps // max(1, typical))
                    reachable_cap = min(stock_cap, per_car * len(touring))
        except Exception:
            reachable_cap = stock_cap
        capacity = reachable_cap

        # 店どうしの距離を、渋滞のあり/なしで測る。
        gap_clear = gap_now = 0
        try:
            spots = [spot.position for spot in problem.spots]
            if len(spots) >= 2:
                clear_gaps = []
                now_gaps = []
                for origin in spots[:6]:
                    free = finder.table(origin, {}, 0)
                    jam = finder.table(origin, road_status, 0)
                    for other in spots:
                        if other == origin:
                            continue
                        if free.reachable(other):
                            clear_gaps.append(free.steps[other])
                        if jam.reachable(other):
                            now_gaps.append(jam.steps[other])
                if clear_gaps:
                    clear_gaps.sort()
                    gap_clear = clear_gaps[len(clear_gaps) // 2]
                if now_gaps:
                    now_gaps.sort()
                    gap_now = now_gaps[len(now_gaps) // 2]
        except Exception:
            gap_clear = gap_now = 0

        # 自チームが道路に滞在した歩数（翌日の渋滞を自分で作っていないか）
        own_road = 0
        try:
            for index, row in enumerate(tracks):
                if index not in touring:
                    continue
                for cell in row:
                    if int(problem.cells[cell]) == 1:
                        own_road += 1
        except Exception:
            own_road = 0

        severity = {
            "jammedRoads": jammed,
            "busyRoads": busy,
            # **渋滞が何を奪ったか。**
            #
            # 店どうしの距離が渋滞でどれだけ伸びたかを記録します。
            # 実戦1341では、渋滞で距離が13歩→25歩に倍増し、
            # 1台が回れる店が半分になっていました。この数字があれば
            # 「渋滞のせいか、判断のせいか」を推測せずに切り分けられます。
            "spotGapClear": gap_clear,
            "spotGapNow": gap_now,
            # 自チームが前日に道路へ滞在した歩数（翌日の渋滞の一因）
            "ownRoadSteps": own_road,
            "roadCells": sum(1 for cell in problem.cells if int(cell) == 1),
            # 上限1玉あたり使える歩数。これが達成率をほぼ決めます。
            "stepsPerNoodle": round(
                day_steps * len(touring) / max(1, capacity), 1
            ),
            "capacity": capacity,
            # 在庫だけで数えた上限（時間を考えない理論値）
            "stockCapacity": stock_cap,
            "achieved": len(acquisitions),
            # 燃料がどれだけ残っていたか（低いほど苦しい）
            "startFuelRatio": round(
                sum(start_fuels[i] for i in touring)
                / max(1, len(touring) * problem.fuel_limit), 2
            ),
            # 1歩ごとの位置から、同じマスに留まった歩数を数える
            # **並列がどれだけ働いたか。**
            # 子プロセスが作った案の数と、親が作った数を分けて残します。
            # CPUが上下する原因を、あとから数字で追えるようにするためです。
            "parallelPlans": parallel_plans,
            "ownPlans": max(0, candidates - parallel_plans),
            "stillSteps": sum(
                sum(1 for s in range(len(row) - 1) if row[s] == row[s + 1])
                for index, row in enumerate(tracks) if index in touring
            ),
        }
        self.days.append({
            "day": day_index + 1,
            "label": label,
            "candidates": candidates,
            "resubmissions": resubmissions,
            "daySteps": day_steps,
            "seconds": round(elapsed, 1),
            "startPositions": list(start_positions),
            "startFuels": list(start_fuels),
            "positions": list(positions),
            "fuels": list(fuels),
            "acquired": [
                {"step": item.step, "agent": item.agent_index,
                 "pos": item.spot_position, "brand": item.brand}
                for item in acquisitions
            ],
            "refuelEvents": [
                {"step": item.step, "agent": item.agent_index, "pos": item.position,
                 "before": item.fuel_before, "after": item.fuel_after}
                for item in refuels
            ],
            "tracks": [list(row) for row in tracks],
            "roads": {str(key): int(value) for key, value in road_status.items()},
            # **ほかのチームの位置も残します。**
            # 渋滞は全チームの前2日の滞在から決まるので、これが
            # 無いと「なぜその道が混んだか」をあとから追えません。
            "enemies": [list(row) for row in (enemies or [])],
            # その日に取れなかった店と、その理由
            "missed": missed,
            "severity": severity,
            "topCandidates": top,
            "brandTotal": brand_total,
            "dailyTotal": daily_total,
            "noodleTotal": noodle_total,
        })

    # ---------------- 書き出し ----------------

    def save(self, score_text: str, mismatches: list[str]) -> Path | None:
        """試合が終わってから1回だけ書く。"""

        if not self.enabled or not self.header:
            return None
        payload = {
            "savedAt": time.time(),
            "elapsed": round(time.time() - self.started, 1),
            "score": score_text,
            "mismatches": list(mismatches),
            "header": self.header,
            "days": self.days,
        }
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            name = f"match_{self.header.get('matchId', 'unknown')}.json"
            path = LOG_DIR / name
            path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            return path
        except Exception:
            # 記録に失敗しても試合の結果には関係ありません。
            return None


def missed_spots(
    problem: Problem,
    kinds: tuple[AgentKind, ...],
    finder,
    road_status: dict,
    positions: list[int],
    fuels: list[int],
    day_steps: int,
    acquired: list,
) -> list[dict]:
    """その日に取れなかった店と、取れなかった理由。

    理由は、各巡回車から見て一番ましなものを選びます。

      届いた   : 歩数も燃料も足りていたのに行かなかった（判断の問題）
      歩数不足 : どの車からも歩数が足りない
      燃料不足 : 歩数は足りるが燃料が足りない
      在庫切れ : その日もう取れない

    **「届いたのに行かなかった」が多いなら、改善の余地があります。**
    """

    taken: dict[int, int] = {}
    for item in acquired:
        taken[item.spot_position] = taken.get(item.spot_position, 0) + 1
    touring = [i for i, kind in enumerate(kinds) if kind == AgentKind.TOURING]
    result: list[dict] = []
    for spot in problem.spots:
        got = taken.get(spot.position, 0)
        room = min(spot.stocks, len(touring)) - got
        if room <= 0:
            continue
        reason = "在庫切れ"
        best_steps = None
        for index in touring:
            table = finder.table(positions[index], road_status, 0)
            steps = table.steps.get(spot.position)
            if steps is None:
                continue
            if steps > day_steps:
                if reason == "在庫切れ":
                    reason = "歩数不足"
                continue
            if table.fuel[spot.position] > fuels[index]:
                reason = "燃料不足"
                continue
            reason = "届いた"
            if best_steps is None or steps < best_steps:
                best_steps = steps
        result.append({
            "pos": spot.position, "brand": spot.brand, "stocks": spot.stocks,
            "got": got, "room": room, "reason": reason,
            "steps": best_steps,
        })
    return result
