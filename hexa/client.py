"""公式回答システムとの通信。

API は「回答システムに関する情報」(redoc-static.html) の OpenAPI 定義に従います。

  GET  /setting   試合設定取得（マップ構成）
  POST /agent     エージェント種別の提出（0:巡回車 1:補給車 の配列）
  GET  /          各日開始時点の試合状態（自チーム・他チーム・道路状態）
  POST /          その日の行動計画の提出

認証はトークンです。ヘッダー `Procon-Token` かクエリ `token` のどちらでも通ります。
このクライアントはヘッダーを使います。

通信は標準ライブラリの urllib だけで行います。競技当日の PC に何もインストール
しなくて済むこと、HTTP/1.1 しか使わないこと（質問回答集その2 A41）が理由です。
リクエストは秒間5回程度が目安なので（同 A42）、最短間隔を入れています。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from hexa.problem import Problem, problem_from_json_dict
from hexa.rules import AgentKind, RoadStatus

MIN_REQUEST_INTERVAL = 0.21     # 秒間5リクエストを超えないための最短間隔

# **提出は何度でも試します。**
#
# 準決勝の32x32（第1791641357試合）で、3日目の提出が1回失敗した
# だけで**その日は7台が1歩も動きませんでした**。ログの「実際の
# 3日目終了時の燃料」は開始時と1ビットも同じで、日別は
# 22+22+**0**+22+22+22 に落ちています（手元の予測は132、実際は112）。
#
# `post_actions` には再試行が1つもなく、通信が一瞬切れただけで
# 1日を捨てる作りでした。会場の回線は落ちるものとして扱います。
POST_RETRIES = 6
# 再試行の待ち（秒）。回数ごとに倍にしていきます。
POST_RETRY_WAIT = 0.3
# これ以上は待たない（1日の持ち時間を食いつぶさないため）。
POST_RETRY_WAIT_MAX = 2.0
# **やり直しても意味がないサーバーの答え。**
# 計画そのものが悪い（長さ違い・池へ移動など）ので、同じものを
# 投げ直しても通りません。呼んだ側が別の計画を出すべきです。
FATAL_POST_CODES = frozenset({
    "ValidationError", "FormatError", "BadRequest",
    "AccessTimeError", "NotFound", "Unauthorized", "Forbidden",
})


class ProconApiError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class AgentState:
    kind: AgentKind
    position: int
    fuel: int


@dataclass(frozen=True)
class MatchState:
    ends_at: int
    day: int
    agents: tuple[AgentState, ...]
    others: tuple[tuple[int, tuple[AgentState, ...]], ...]
    road_status: dict[int, RoadStatus]


def _parse_agents(raw: list[dict]) -> tuple[AgentState, ...]:
    return tuple(
        AgentState(
            kind=AgentKind(int(item["kind"])),
            position=int(item["pos"]),
            fuel=int(item["fuel"]),
        )
        for item in raw
    )


class ProconClient:
    def __init__(self, base_url: str, token: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._last_request_at = 0.0

    # ---------------- 低レベル ----------------

    def _request(self, method: str, path: str, payload: object | None = None) -> object:
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

        url = self.base_url + path
        body = None
        headers = {
            "Procon-Token": self.token,
            "Accept": "application/json",
            # 名乗りを普通のブラウザに合わせる。
            # Python の標準ライブラリは "Python-urllib/3.11" と名乗るため、
            # Cloudflare を挟んだサイトでは 1010 Access denied で弾かれます
            # （非公式練習場で実際に発生）。公式サーバーでも害はありません。
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
            ),
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", errors="replace")
            code, message = "HTTPError", text
            try:
                parsed = json.loads(text)
                code = parsed.get("code", code)
                message = parsed.get("message", message)
            except json.JSONDecodeError:
                pass
            raise ProconApiError(error.code, code, message) from None
        if not text.strip():
            return None
        return json.loads(text)

    # ---------------- 各API ----------------

    def get_setting(self) -> tuple[Problem, dict]:
        raw = self._request("GET", "/setting")
        if not isinstance(raw, dict):
            raise ProconApiError(200, "FormatError", "試合設定が JSON オブジェクトではありません")
        return problem_from_json_dict(raw), raw

    def post_agent_kinds(self, kinds: list[AgentKind] | tuple[AgentKind, ...]) -> None:
        self._request("POST", "/agent", [int(kind) for kind in kinds])

    def get_state(self) -> MatchState:
        raw = self._request("GET", "/")
        if not isinstance(raw, dict):
            raise ProconApiError(200, "FormatError", "試合状態が JSON オブジェクトではありません")
        return MatchState(
            ends_at=int(raw["endsAt"]),
            day=int(raw["day"]),
            agents=_parse_agents(raw["agents"]),
            others=tuple(
                (int(entry["id"]), _parse_agents(entry["agents"]))
                for entry in raw.get("others", [])
            ),
            road_status={
                int(entry["pos"]): RoadStatus(int(entry["status"]))
                for entry in raw.get("traffics", [])
            },
        )

    def post_actions(self, action_plan: list[list[int]], log=None) -> int:
        """行動計画を出す。**通信の失敗は何度でも試し直します。**

        理由は `POST_RETRIES` に。計画そのものが悪いときだけは、
        同じものを投げ直しても通らないので、すぐ上へ返します。
        """

        wait = POST_RETRY_WAIT
        last: Exception | None = None
        for attempt in range(1, POST_RETRIES + 1):
            try:
                raw = self._request("POST", "/", action_plan)
                if isinstance(raw, dict) and "revision" in raw:
                    return int(raw["revision"])
                return 0
            except ProconApiError as error:
                last = error
                if error.code in FATAL_POST_CODES:
                    raise
                # 429（投げすぎ）や5xx（サーバー側）は待てば通ります。
            except (OSError, TimeoutError, urllib.error.URLError) as error:
                # 回線が切れた・応答が遅い。待って投げ直します。
                last = error
            if attempt < POST_RETRIES:
                if log is not None:
                    log(f"  提出をやり直します（{attempt}回目が失敗:"
                        f" {type(last).__name__} {last}）")
                time.sleep(wait)
                wait = min(POST_RETRY_WAIT_MAX, wait * 2)
        if isinstance(last, ProconApiError):
            raise last
        raise ProconApiError(0, "NetworkError", str(last))

    # ---------------- 補助 ----------------

    def wait_for_setting(
        self,
        poll_interval: float = 1.0,
        timeout: float = 3600.0,
        log=None,
        waiting=None,
    ) -> tuple[Problem, dict]:
        """試合設定が取れるようになるまで待つ。

        公式の進行では「試合開始と同時に問題データが取得可能になります」。
        つまり試合開始前にアクセスするとエラーが返ります。
        接続そのものは事前に確認しておきたいので、取れるまで待ち続けます。

        待っている間に落ちないよう、通信エラーも握って再試行します。
        会場の回線が一時的に切れても、復帰すればそのまま続きます。
        """

        deadline = time.monotonic() + timeout
        notified = False
        while True:
            try:
                problem, raw = self.get_setting()
                if int(raw.get("startsAt", 0)) != 0 or raw.get("map"):
                    return problem, raw
                reason = "試合開始時刻がまだ確定していません"
                linked = True
            except ProconApiError as error:
                if error.code not in ("AccessTimeError", "NotFound"):
                    raise
                # **サーバーには届いています。** 試合前に返る答えです。
                reason = f"まだ試合前です（{error.code}）"
                linked = True
            except (OSError, TimeoutError) as error:
                reason = f"通信に失敗しました（{error}）"
                linked = False

            # **待っているあいだも監視に知らせます。**
            #
            # 以前は試合が始まるまで何も書かなかったので、ビューアは
            # 前の試合のファイルを見るか、読めないままでした。開始
            # 3分前なのに「つながりません」と出ていたのがこれです。
            if waiting is not None:
                try:
                    waiting(reason, linked)
                except Exception:
                    pass

            if log is not None and not notified:
                log(f"  {reason}。取得できるまで待ちます…")
                notified = True
            if time.monotonic() > deadline:
                raise TimeoutError("試合設定を取得できませんでした")
            time.sleep(poll_interval)

    def check_connection(self) -> tuple[bool, str]:
        """通信が通るかだけを確かめる。試合前でも使える。

        試合前は AccessTimeError が返りますが、それは「サーバーに届いていて
        トークンも通っている」証拠なので成功とみなします。
        """

        try:
            self.get_setting()
            return (True, "接続できました。試合設定も取得できています。")
        except ProconApiError as error:
            if error.code in ("AccessTimeError", "NotFound"):
                return (True, f"接続できました。まだ試合前です（{error.code}）。")
            text = str(error)
            if "1010" in text or "browser_signature" in text:
                return (
                    False,
                    "サイトの前段（Cloudflare）に弾かれました。"
                    "トークンではなく、通信の名乗り方の問題です。",
                )
            if error.status in (401, 403):
                return (False, f"トークンが通りませんでした（{error}）")
            return (False, f"サーバーがエラーを返しました（{error}）")
        except (OSError, TimeoutError) as error:
            return (False, f"サーバーに届きませんでした（{error}）")

    def wait_for_day(self, day: int, poll_interval: float = 0.4, timeout: float = 900.0) -> MatchState:
        """指定した日の状態が配られるまで待つ。day は初日が0。"""

        deadline = time.monotonic() + timeout
        while True:
            try:
                state = self.get_state()
                if state.day >= day:
                    return state
            except ProconApiError as error:
                if error.code != "AccessTimeError":
                    raise
            if time.monotonic() > deadline:
                raise TimeoutError(f"{day + 1}日目の状態が配られませんでした")
            time.sleep(poll_interval)
