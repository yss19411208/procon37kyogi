"""敵チームの道路の使い方を推定する。

公式ルール上、チーム同士が干渉する経路は道路の交通量だけです
（スポットの在庫はチームごとに独立 / 募集要項 p.10）。
なので「敵を読む」というのは、実質「敵がどの道路をどれだけ踏んでいるか」を
当てにいくことになります。

使える手がかりは2つです。

**手がかり1: 道路状態からの逆算**
  交通量は「全チームの前日・前々日の滞在ステップ数の合算 ÷ チーム数」で、
  その結果の状態（順調・混雑・渋滞）だけがサーバーから返ってきます
  （数値そのものは配られない / 質問回答集その1 Q12）。
  自分の滞在数は正確に分かるので、

      敵の滞在数 = 状態から分かる合算の範囲 − 自分の滞在数

  という不等式が、道路セルごとに毎日1本ずつ手に入ります。

**手がかり2: 各日開始時点の敵の位置**
  GET / の others で、他チームの全エージェントの位置が見えます。
  そこが道路セルなら、少なくともその1ステップは踏んでいた証拠になります。
  1日の途中は見えないので弱い情報ですが、確実な下限です。

この2つを混ぜて、道路セルごとに「敵が1日あたり何ステップ踏むか」の
見積もりを持ち続けます。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hexa.problem import Problem
from hexa.rules import RoadStatus, Terrain

# 渋滞は上限がないので、基準値の何倍とみなすか。控えめに見積もる。
JAMMED_ESTIMATE_FACTOR = 1.3
# 新しい観測をどれくらい強く反映するか。
SMOOTHING = 0.4


@dataclass
class OpponentModel:
    problem: Problem
    daily_estimate: dict[int, float] = field(default_factory=dict)
    sightings: dict[int, int] = field(default_factory=dict)
    observed_days: int = 0

    def __post_init__(self) -> None:
        self.road_cells = [
            position
            for position in range(self.problem.cell_count)
            if self.problem.terrain_at(position) == Terrain.ROAD
        ]

    # ---------------- 観測 ----------------

    def observe_positions(self, other_positions: list[int]) -> None:
        """各日開始時点の敵の位置から、道路セルの目撃数を数える。"""

        for position in other_positions:
            if self.problem.terrain_at(position) == Terrain.ROAD:
                self.sightings[position] = self.sightings.get(position, 0) + 1

    def observe_status(
        self,
        status: dict[int, RoadStatus],
        own_stays_window: list[dict[int, int]],
    ) -> None:
        """道路状態と、その状態を作った自分の滞在数から敵の滞在数を逆算する。

        own_stays_window はその状態の根拠になった日（1日または2日）の
        自分の滞在数です。
        """

        if not own_stays_window:
            return
        players = max(1, self.problem.players)
        busy_line = self.problem.busy_threshold * players
        jammed_line = self.problem.jammed_threshold * players
        days = len(own_stays_window)

        for position in self.road_cells:
            own = sum(day.get(position, 0) for day in own_stays_window)
            state = status.get(position, RoadStatus.SMOOTH)
            if state == RoadStatus.SMOOTH:
                total_guess = max(0.0, (busy_line - 1) / 2.0)
            elif state == RoadStatus.BUSY:
                total_guess = (busy_line + jammed_line - 1) / 2.0
            else:
                total_guess = jammed_line * JAMMED_ESTIMATE_FACTOR
            others_per_day = max(0.0, (total_guess - own) / days)
            previous = self.daily_estimate.get(position)
            if previous is None:
                self.daily_estimate[position] = others_per_day
            else:
                self.daily_estimate[position] = (
                    previous * (1.0 - SMOOTHING) + others_per_day * SMOOTHING
                )
        self.observed_days += 1

    # ---------------- 利用 ----------------

    def estimated_daily(self, position: int) -> float:
        """その道路セルに敵が1日あたり何ステップ滞在するかの見積もり。"""

        base = self.daily_estimate.get(position, 0.0)
        # 目撃は確実な下限なので、見積もりが小さすぎるときは引き上げる。
        seen = self.sightings.get(position, 0)
        if self.observed_days > 0:
            base = max(base, seen / self.observed_days)
        return base

    def hot_roads(self, limit: int = 40) -> list[tuple[int, float]]:
        """敵がよく踏んでいる道路セルを、多い順に返す。"""

        scored = [
            (position, self.estimated_daily(position))
            for position in self.road_cells
        ]
        scored = [item for item in scored if item[1] > 0.0]
        scored.sort(key=lambda item: -item[1])
        return scored[:limit]

    def predict_status(
        self,
        own_stays_window: list[dict[int, int]],
        extra_own: dict[int, int] | None = None,
    ) -> dict[int, RoadStatus]:
        """翌日の道路状態を、発表される前に見積もる。

        own_stays_window は翌日の判定に使われる自分の滞在数（直近1〜2日）。
        extra_own は「これから出す計画で追加される自分の滞在数」です。
        """

        players = max(1, self.problem.players)
        busy_line = self.problem.busy_threshold * players
        jammed_line = self.problem.jammed_threshold * players
        days = max(1, len(own_stays_window))
        predicted: dict[int, RoadStatus] = {}

        for position in self.road_cells:
            own = sum(day.get(position, 0) for day in own_stays_window)
            if extra_own:
                own += extra_own.get(position, 0)
            total = own + self.estimated_daily(position) * days
            if total >= jammed_line:
                predicted[position] = RoadStatus.JAMMED
            elif total >= busy_line:
                predicted[position] = RoadStatus.BUSY
            else:
                predicted[position] = RoadStatus.SMOOTH
        return predicted

    def congestion_risk(self, own_stays_window: list[dict[int, int]]) -> dict[int, float]:
        """道路セルごとに「明日どれくらい混みそうか」を 0〜1 で返す。

        0 は明日も順調、1 は渋滞見込み。経路選びの材料に使います。
        """

        players = max(1, self.problem.players)
        busy_line = self.problem.busy_threshold * players
        jammed_line = self.problem.jammed_threshold * players
        days = max(1, len(own_stays_window))
        risk: dict[int, float] = {}
        for position in self.road_cells:
            own = sum(day.get(position, 0) for day in own_stays_window)
            total = own + self.estimated_daily(position) * days
            if total >= jammed_line:
                risk[position] = 1.0
            elif total >= busy_line:
                span = max(1.0, jammed_line - busy_line)
                risk[position] = 0.5 + 0.5 * (total - busy_line) / span
            else:
                risk[position] = 0.5 * total / max(1.0, busy_line)
        return risk
