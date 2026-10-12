"""本番モードの様子をリアルタイムで見るための書き出し。

**処理と監視は完全に分けます。**

本番は1日60秒しかありません。監視のために1ミリ秒でも待たされては
いけないので、ここでやるのは**軽いJSONを1つ書くだけ**です。
書き込みは1日1回、数ミリ秒で終わります。

ブラウザのビューアがそのファイルを1秒ごとに読んで表示します。
ビューアを開いても閉じても、落ちても、試合には何の影響もありません。

書き込みに失敗しても試合を止めません。監視は「あれば嬉しい」もので、
本番の点数には一切関係しないからです。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from hexa.problem import Problem
from hexa.rules import AgentKind

# 既定の書き出し先。ビューアはここを読む。
LIVE_DIR = Path(__file__).resolve().parent.parent / "viewer" / "live"
LIVE_NAME = "state.json"
# **探索中だけ書く、小さいほうのファイル。**
#
# `state.json` にはその試合の全部（マップ・全日の1歩ごとの位置）が
# 入ります。10日目には87KBになり、1回書くのに9.1ミリ秒かかりました。
# 0.2秒ごとに書くと**探索の4.5%**を監視に使ってしまいます。
#
# 過ぎた日の中身は、その日のあいだ変わりません。だから探索中は
# 「いまの様子＋いま作っている1日ぶん」だけを書きます。10分の1
# 以下の大きさになり、1ミリ秒で済みます。ビューアはこちらを速く、
# `state.json` は日が変わったときだけ読みます。
NOW_NAME = "now.json"


# **探索中の書き出しの間隔。**
#
# ビューアは届いた瞬間に映すので、ここがそのまま「どれだけ新しい絵が
# 見られるか」の下限になります。探索中に書くのは小さいほうの
# ファイル（`NOW_NAME`）だけなので、0.2秒でも探索の0.5%ほどです。
SEARCH_WRITE_GAP = 0.2


class LiveMonitor:
    """試合の様子を書き出す。失敗しても試合を止めない。"""

    def __init__(self, directory: Path | None = None, enabled: bool = True) -> None:
        self.directory = Path(directory) if directory else LIVE_DIR
        self.enabled = enabled
        self.started_at = time.time()
        self.days: list[dict] = []
        self.header: dict = {}
        self._last_search = 0.0
        # 連続して書けなかった回数。たまの失敗では止めません。
        self._fails = 0
        if self.enabled:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enabled = False

    # ---------------- 記録 ----------------

    def waiting(self, reason: str, linked: bool = True) -> None:
        """**試合が始まるのを待っていることを知らせる。**

        以前は試合が始まるまで何も書きませんでした。そのため
        ビューアは前の試合のファイルを見るか、読めないままで、
        **開始3分前なのに「つながりません」**と出ていました。
        読めないのはビューアとファイルの間の話で、サーバーとの
        接続とは別のことです。ここで軽いほうだけを書き換えます。
        """

        if not self.enabled:
            return
        self._put(NOW_NAME, {
            "updatedAt": time.time(),
            "elapsed": round(time.time() - self.started_at, 1),
            "status": {"phase": "試合を待っています",
                       "reason": reason, "linked": bool(linked)},
            "dayCount": len(self.days),
            "preview": None,
        })

    def start_match(self, problem: Problem, kinds: tuple[AgentKind, ...]) -> None:
        """試合の始めに1回だけ呼ぶ。マップは大きいのでここでだけ書く。"""

        self.started_at = time.time()
        self._last_search = 0.0
        self.days = []
        self.header = {
            "startedAt": self.started_at,
            "height": problem.height,
            "width": problem.width,
            "cells": list(problem.cells),
            "dayCount": problem.day_count,
            "daySteps": list(problem.day_steps),
            "players": problem.players,
            "fuelLimit": problem.fuel_limit,
            "busyThreshold": problem.busy_threshold,
            "jammedThreshold": problem.jammed_threshold,
            "spots": [
                {"pos": spot.position, "brand": spot.brand, "stocks": spot.stocks}
                for spot in problem.spots
            ],
            "agentKinds": [int(kind) for kind in kinds],
            "brandCount": len({spot.brand for spot in problem.spots}),
        }
        self._write({"phase": "種別を決めました", "day": 0})

    def searching(
        self,
        day_index: int,
        candidates: int,
        best_label: str,
        seconds_left: float,
        trying: str = "",
        best_score: tuple | None = None,
        resubmissions: int = 0,
        recent: list | None = None,
        preview=None,
    ) -> None:
        """探索中の様子。回答時間のあいだ、何度でも呼べる。

        `SEARCH_WRITE_GAP` 秒に1回だけ書きます（1回0.4ミリ秒なので
        探索の速さには響きませんが、無駄に書いても意味がないため）。
        """

        now = time.monotonic()
        if now - self._last_search < SEARCH_WRITE_GAP:
            return
        self._last_search = now
        # **盤面は書くときだけ作ります。**
        #
        # 呼ぶ側は候補1件ごとにここへ来ますが、書くのは
        # `SEARCH_WRITE_GAP` 秒に1回です。ところが Python は引数を
        # 先に評価するので、盤面を dict で渡すと**書かない回でも
        # 作られます**。関数で渡してもらい、ここで呼びます。
        if callable(preview):
            preview = preview()
        self._write_now({
            "phase": "探索中",
            "day": day_index + 1,
            "candidates": candidates,
            "bestLabel": best_label,
            "bestScore": list(best_score) if best_score else None,
            "trying": trying,
            "resubmissions": resubmissions,
            "recent": recent or [],
            "secondsLeft": round(max(0.0, seconds_left), 1),
        }, preview)

    def finish_day(
        self,
        day_index: int,
        label: str,
        candidates: int,
        resubmissions: int,
        revision: int,
        day_steps: int,
        start_positions: list[int],
        start_fuels: list[int],
        positions: list[int],
        fuels: list[int],
        acquisitions: list,
        refuels: list,
        action_plan: list[list[int]],
        tracks: list[list[int]] | None,
        road_status: dict,
        enemy_positions: list[list[int]] | None,
        brand_total: int,
        daily_total: int,
        noodle_total: int,
    ) -> None:
        """1日が終わったところで呼ぶ。ここが主な書き出し。"""

        self.days.append({
            "day": day_index + 1,
            "label": label,
            "candidates": candidates,
            "resubmissions": resubmissions,
            "revision": revision,
            # ビューアが1歩ごとの様子を描くのに必要な一式。
            "daySteps": day_steps,
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
            "actions": [list(row) for row in action_plan],
            # 1歩ごとに実際にいたマス。ビューアはこれをそのまま描くので、
            # 方向の解釈を間違えて軌跡がずれる、ということが起きません。
            "tracks": [list(row) for row in (tracks or [])],
            "roads": {str(key): int(value) for key, value in road_status.items()},
            "enemies": enemy_positions or [],
            "brandTotal": brand_total,
            "dailyTotal": daily_total,
            "noodleTotal": noodle_total,
        })
        self._write({"phase": "提出しました", "day": day_index + 1})

    def finish_match(self, score_text: str, mismatches: list[str]) -> None:
        self._write({
            "phase": "試合終了",
            "day": len(self.days),
            "score": score_text,
            "mismatches": list(mismatches),
        })

    # ---------------- 書き出し ----------------

    def _write_now(self, status: dict, preview: dict | None = None) -> None:
        """**探索中だけの、小さい書き出し。**

        過ぎた日の中身は入れません（その日のあいだ変わらないので、
        ビューアは `state.json` から一度読めば足ります）。入れると
        10日目で87KB・9.1ミリ秒になり、探索の4.5%を食います。
        """

        if not self.enabled:
            return
        payload = {
            "updatedAt": time.time(),
            "elapsed": round(time.time() - self.started_at, 1),
            "status": status,
            "dayCount": len(self.days),
            "preview": preview,
        }
        self._put(NOW_NAME, payload)

    def _write(self, status: dict, preview: dict | None = None) -> None:
        """いまの状態を書き出す。

        preview が渡されたときは、**まだ提出していない暫定の最良案**を
        その日の分として一緒に出します。こうすると探索中でも盤面が動き、
        「いまどんな計画が最良か」を見ながら待てます。
        提出が終われば finish_day が本物に差し替えます。
        """

        if not self.enabled:
            return
        days = self.days
        if preview is not None:
            days = self.days + [preview]
        payload = {
            "updatedAt": time.time(),
            "elapsed": round(time.time() - self.started_at, 1),
            "status": status,
            "header": self.header,
            "days": days,
        }
        self._put(LIVE_NAME, payload)
        # 小さいほうも同時に置き換えます。ビューアはこちらを速く
        # 読むので、段階（試合終了など）がすぐ伝わります。
        self._put(NOW_NAME, {
            "updatedAt": payload["updatedAt"],
            "elapsed": payload["elapsed"],
            "status": status,
            "dayCount": len(self.days),
            "preview": preview,
        })

    def _put(self, name: str, payload: dict) -> None:
        """1つのファイルを、途中が見えない形で置き換える。

        途中まで書いたファイルをビューアに読ませないため、一時
        ファイルに書いてから名前を変える（差し替えは一瞬で終わる）。

        **1回の失敗で監視を止めてはいけません。**
        Windows では、ブラウザやウイルス対策がファイルを開いている
        一瞬に os.replace が失敗することがあります。以前はそこで
        enabled を False にしていたため、**一度詰まると二度と更新
        されません**でした。実戦で「途中から止まる」と報告された
        症状がこれです。数回続けて失敗したときだけ諦めます。
        """

        temporary = None
        try:
            handle, temporary = tempfile.mkstemp(
                dir=str(self.directory), prefix=".live", suffix=".tmp"
            )
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False)
            for attempt in range(3):
                try:
                    os.replace(temporary, str(self.directory / name))
                    temporary = None
                    self._fails = 0
                    return
                except OSError:
                    if attempt < 2:
                        time.sleep(0.02)
            raise OSError("差し替えに失敗しました")
        except Exception:
            self._fails += 1
            if self._fails >= 20:
                # さすがに続かないので止める（書けない場所が指定された等）。
                self.enabled = False
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
