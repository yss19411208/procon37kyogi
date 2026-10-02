"""本番用の戦い方。

学習したAIのうち1体だけを使うのはもったいないので、本番では毎日
次のことをします。

  1. 全AIぶんの行動計画と、学習に依存しない貪欲な行動計画を作る
  2. 全部を公式と同じシミュレータにかけて、その日の成果を測る
  3. 公式の優先順位（新しい系列 → その日の系列数 → 玉数）で一番良いものを提出する
  4. 回答時間がまだ残っていれば、探索を少し入れた計画をさらに試し、
     良いものが見つかったら出し直す（再提出は認められています）

こうすると「一番強いAI」ではなく「その日一番良かった判断」を毎日選べます。
貪欲な計画を必ず候補に入れてあるので、AIが変な判断をした日でも下限が保証されます。

重みを混ぜて1体の「合体AI」を作っているわけではありません。各AIはそれぞれ
別のまま残り、日ごとに採用する案を選び直します。前の日にどのAIが選ばれたかは、
翌日の選択に影響しません。

安全側の作りとして、
  ・提出する計画は必ず公式と同じシミュレータで検証してから送る
  ・どの候補も作れなかった場合は「全員その日の終わりまで待機」を送る
  ・回答時間の締切には必ず間に合わせる（残り時間を見て打ち切る）
  ・翌日にサーバーが返す状態と自分の予測が食い違ったら、その場で警告する
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from hexa.ai import Brain, Policy, SupplyDecision, TouringDecision
from hexa.client import MatchState, ProconApiError, ProconClient
from hexa.opponents import OpponentModel
from hexa.livemonitor import LiveMonitor
from hexa.pathfind import PathFinder, fuel_weight
from hexa.planner import DayPlanner
from hexa.problem import Problem
from hexa.rules import AgentKind, RoadStatus
from hexa.scoring import Score
from hexa.simulator import DayResult, InvalidAnswer, road_status_for_day, simulate_day

SAFETY_MARGIN_SECONDS = 2.0     # 締切のこれだけ手前で必ず打ち切る

# 募集要項 p.12「試合に関する注意事項」:
#   (5) 制限時間内であれば再提出が可能ですが、試合の進行を妨げるほどの回数や
#       容量を送信すると妨害行為とみなし失格となる可能性があります。
#   (6) 最後に受理された有効な回答を採用します。
# 良くなったときだけ出し直し、さらに1日あたりの回数に上限を設ける。
MAX_RESUBMISSIONS_PER_DAY = 15


class GreedyPolicy:
    """学習を使わない、素直な方策。

    本番で必ず候補に入れる下限です。AIがどれだけ変な判断をしても、
    これより悪い計画が選ばれることはありません。

      巡回車: まだ取っていない系列の店を最優先。次に今日まだ取っていない系列。
              同じ条件なら近い店。行ける店がある限り待機しない。
      補給車: 燃料の不足が一番大きい巡回車のところへ向かう。
    """

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.fuel_limit = max(1, problem.fuel_limit)
        self.spot_brand = {spot.position: spot.brand for spot in problem.spots}
        self.spot_positions = [spot.position for spot in problem.spots]

    def choose_touring_target(
        self, *, agent_index, step, remaining, position, fuel, stock, reserved,
        acquired_today, acquired_brands, brands_today, road_status, finder,
        kinds=(), positions=None, day_index=0, day_count=1, congestion_risk=None,
        teammate_targets=None, road_heat=None, allow_wait=True,
    ) -> TouringDecision:
        table = finder.table(position, road_status, fuel_weight(remaining, fuel))
        best = None
        for spot_position in self.spot_positions:
            if spot_position in acquired_today or stock.get(spot_position, 0) <= 0:
                continue
            if not table.reachable(spot_position):
                continue
            steps = table.steps[spot_position]
            need = table.fuel[spot_position]
            if steps > remaining or need > fuel:
                continue
            brand = self.spot_brand[spot_position]
            # 在庫が残っていれば、他の車が向かっていても構いません。
            # ただし今日まだ取っていない系列の店は1台だけにします
            # （2台目を向けると、別の系列を取る車が減るため）。
            claimed = reserved.get(spot_position, 0)
            # 今日まだ取っていない系列の店は1台だけ。
            limit = 1 if brand not in brands_today else stock.get(spot_position, 0)
            if claimed >= limit:
                continue
            key = (
                0 if brand not in acquired_brands else 1,
                0 if brand not in brands_today else 1,
                claimed,
                steps,
            )
            if best is None or key < best[0]:
                best = (key, spot_position)
        target = best[1] if best else None
        return TouringDecision(target=target, features=[], candidate_features=[[]] * 2)

    def choose_supply_target(
        self, *, agent_index, step, remaining, position, kinds, positions, fuels,
        plan_targets, supply_claims, road_status, finder,
        day_index=0, day_count=1, road_heat=None,
    ) -> SupplyDecision:
        table = finder.table(position, road_status)
        best = None
        for other in range(len(kinds)):
            if kinds[other] != AgentKind.TOURING:
                continue
            deficit = self.fuel_limit - fuels[other]
            for meeting in {positions[other], plan_targets[other]}:
                if meeting is None or not table.reachable(meeting):
                    continue
                steps = table.steps[meeting]
                if steps > remaining:
                    continue
                # 味方がすでに担当している巡回車は後回しにする。
                key = (supply_claims.get(other, 0), -deficit, steps)
                if best is None or key < best[0]:
                    best = (key, meeting, other)
        if best is None:
            if road_heat:
                # 合流先がないときは、敵がよく踏む道路で待つ（燃料を使わないのでタダ）。
                parking = None
                parking_key = None
                for candidate_position, heat in road_heat.items():
                    if heat <= 0.0:
                        continue
                    steps = table.steps.get(candidate_position)
                    if steps is None or steps > remaining:
                        continue
                    key = (heat, -steps)
                    if parking_key is None or key > parking_key:
                        parking_key = key
                        parking = candidate_position
                if parking is not None:
                    return SupplyDecision(target=parking, target_agent=None,
                                          features=[], candidate_features=[])
            return SupplyDecision(target=None, target_agent=None, features=[],
                                  candidate_features=[])
        return SupplyDecision(target=best[1], target_agent=best[2], features=[],
                              candidate_features=[])


class BrandFirstPolicy(GreedyPolicy):
    """その日の系列数（日別累計）を最優先で取りにいく方策。

    **なぜ必要か**

    日別累計は公式の第2優先で、玉数より上です。実戦で1位を取った試合は
    どちらも日別累計で勝っていました。ところが最近は1〜2点負けることが増え、
    調べると「届くのに行かなかった」取り逃しが24日中29件ありました。

    以前、**すべての判断で**新しい系列を優先させたら、日別累計まで
    下がりました（77.90→73.00）。遠い店へ無理に向かうと、近場を回れば
    取れたはずの別の系列を落とすからです。

    そこでこの方策は、**候補の1つとして加えるだけ**にします。
    最終的にどの計画を提出するかは公式の優先順位で選ぶので、
    この方策が勝つのは「本当に日別累計が増えるとき」だけです。
    **候補が増えるだけなので、悪くなることは構造的にありません。**

    貪欲法との違いは、同じ条件のときに
    「近い店」ではなく「今日まだ取っていない系列のうち一番近い店」を
    強く優先することです。系列を1つでも多くそろえる方向に倒します。
    """

    def choose_touring_target(
        self, *, agent_index, step, remaining, position, fuel, stock, reserved,
        acquired_today, acquired_brands, brands_today, road_status, finder,
        kinds=(), positions=None, day_index=0, day_count=1, congestion_risk=None,
        teammate_targets=None, road_heat=None, allow_wait=True,
    ) -> TouringDecision:
        table = finder.table(position, road_status, fuel_weight(remaining, fuel))
        # 味方が向かっている系列は、そちらに任せる。
        mates = set()
        for index, target in enumerate(teammate_targets or []):
            if index != agent_index and target is not None:
                brand = self.spot_brand.get(target)
                if brand is not None:
                    mates.add(brand)

        best = None
        for spot_position in self.spot_positions:
            if spot_position in acquired_today or stock.get(spot_position, 0) <= 0:
                continue
            if not table.reachable(spot_position):
                continue
            steps = table.steps[spot_position]
            need = table.fuel[spot_position]
            if steps > remaining or need > fuel:
                continue
            brand = self.spot_brand[spot_position]
            # 在庫が残っていれば、他の車が向かっていても構いません。
            # ただし今日まだ取っていない系列の店は1台だけにします
            # （2台目を向けると、別の系列を取る車が減るため）。
            claimed = reserved.get(spot_position, 0)
            # 今日まだ取っていない系列の店は1台だけ。
            limit = 1 if brand not in brands_today else stock.get(spot_position, 0)
            if claimed >= limit:
                continue
            key = (
                0 if brand not in acquired_brands else 1,
                0 if brand not in brands_today else 1,
                1 if brand in mates else 0,
                claimed,
                steps,
            )
            if best is None or key < best[0]:
                best = (key, spot_position)
        target = best[1] if best else None
        return TouringDecision(target=target, features=[], candidate_features=[[]] * 2)


class NoisyGreedyPolicy(GreedyPolicy):
    """貪欲な方策に、たまに二番手を選ばせたもの。

    貪欲は毎回まったく同じ答えを出すので、そのままでは1案しか作れません。
    少しだけ揺らすと、貪欲の近くを探すことができます。
    学習を使わないので、AIが未熟なうちでも上積みになります。
    """

    def __init__(self, problem: Problem, rng: random.Random, noise: float) -> None:
        super().__init__(problem)
        self.rng = rng
        self.noise = noise

    def choose_touring_target(self, **kwargs) -> TouringDecision:
        decision = super().choose_touring_target(**kwargs)
        if decision.target is None or self.rng.random() >= self.noise:
            return decision
        table = kwargs["finder"].table(kwargs["position"], kwargs["road_status"])
        stock = kwargs["stock"]
        acquired_today = kwargs["acquired_today"]
        remaining = kwargs["remaining"]
        fuel = kwargs["fuel"]
        options = [
            spot_position
            for spot_position in self.spot_positions
            if spot_position not in acquired_today
            and stock.get(spot_position, 0) > 0
            and table.reachable(spot_position)
            and table.steps[spot_position] <= remaining
            and table.fuel[spot_position] <= fuel
        ]
        if len(options) <= 1:
            return decision
        return TouringDecision(
            target=self.rng.choice(options), features=[], candidate_features=[[]] * 2
        )


@dataclass
class Candidate:
    label: str
    action_plan: list[list[int]]
    day_result: DayResult
    key: tuple[int, int, int, int]


@dataclass
class ProductionDayReport:
    day: int
    chosen: str
    revision: int
    considered: int
    plan_seconds: float
    resubmissions: int
    day_score: tuple[int, int, int]
    refuels: int
    predicted_positions: list[int] = field(default_factory=list)
    predicted_fuels: list[int] = field(default_factory=list)
    mismatch: str | None = None


@dataclass
class ProductionReport:
    problem: Problem
    agent_kinds: tuple[AgentKind, ...]
    days: list[ProductionDayReport] = field(default_factory=list)
    score: Score = field(default_factory=lambda: Score(0, 0, 0))
    mismatches: list[str] = field(default_factory=list)


def _fallback_plan(agent_count: int, day_steps: int) -> list[list[int]]:
    """必ず有効な最後の砦。全員その日の終わりまで待機する。"""

    return [[-day_steps] for _ in range(agent_count)]


# 明日以降に「この日数ぶん」は動けるようにしておきたい、という目安。
# 1 なら翌日ぶんだけ、2 なら翌々日ぶんまで見ます。
FUEL_LOOKAHEAD_DAYS = 2

# 先読みを効かせる条件。燃料の余裕が「この倍率」以上のときだけ使います。
#
# **なぜ条件を付けるか**
#
# 実戦9試合で測ると、先読みは燃料に余裕のある試合でだけ日別累計を
# 増やし、余裕のない試合では玉数を落とすだけでした。
#
#   試合839 余裕1.16倍: 日別51→52 玉75→78  ← 効く
#   試合836 余裕0.15倍: 日別189→189 玉782→647  ← 損だけ
#
# 余裕がない試合では、どのみち燃料は足りません。残そうとすると
# 今日の玉を捨てるだけで、明日も結局動けません。
# 余裕がある試合でだけ「使いすぎないようにする」のが正しい形です。
FUEL_LOOKAHEAD_MARGIN = 1.0


def _fuel_margin(problem: Problem, supply_count: int = 0) -> float:
    """燃料の余裕。1.0 以上なら、計画しだいで全日動ける。

    平地は2歩で燃料1なので、必要な燃料はおよそ『総歩数の半分』です。

    **補給車がいる場合は、その分を足します。** 補給車の燃料は無限で、
    合流すれば満タンにできます。1日に1台ぶんは渡せると見て、
    `燃料上限 × 日数` を上乗せします。

    これを入れないと、補給車がいるのに「余裕なし」と判定され、
    燃料を残す判断が働きません。実際、試合833は見かけの余裕0.28ですが、
    補給車1台を数えると2.4倍になります。
    """

    need = sum(problem.day_steps) / 2.0
    if need <= 0:
        return 10.0
    # 補給車1台につき、1日1回の補給で燃料上限ぶんを渡せると見積もる。
    supplied = problem.fuel_limit * supply_count * max(0, problem.day_count - 1)
    return (problem.fuel_limit + supplied) / need


def _fuel_shortfall(
    day_result: DayResult,
    kinds: tuple[AgentKind, ...],
    fuel_limit: int,
    steps_ahead: int,
) -> int:
    """明日以降に動けなくなる巡回車が何台いるか。

    **なぜ必要か**

    実戦で、3日目の開始時点に燃料が [96, 11, 6, 0] まで枯れていました。
    そこから3304案を試しても13系列中12しか取れません。
    **当日の探索では絶対に解決できない**状態です。

    原因は、1日ずつ「種類→日別→玉」で最良を選んでいたこと。
    今日の点数だけを見るので、明日の燃料が残りません。
    終了時の燃料は同点のときしか効かず、歯止めになっていませんでした。

    ここでは「明日その車が動けるか」を数えます。平地1マスは
    2歩で燃料1なので、必要な燃料はおよそ『歩数の半分』です。
    足りない車が少ないほど良い計画、と評価します。
    """

    if steps_ahead <= 0:
        return 0
    # 平地は2歩で燃料1なので、歩数の半分の燃料があれば全部動けます。
    # 以前は「その半分（÷4）あれば十分」としていましたが、実測では
    # 甘すぎました。試合833の7日目（32歩）は開始燃料15で系列を落とします。
    # ÷2（つまり全部動ける量）を目安にすると満点に届きました。
    need = max(1, steps_ahead // 2)
    short = 0
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        if day_result.fuels[index] < need:
            short += 1
    return short


# 明日の歩数のうち、どれだけ以内に店がほしいか。
# 0.3 なら「明日の歩数の3割以内に店がある」ことを目指します。
#
# **0 にして無効にしてあります。実測で1点も変わりませんでした。**
#
# 試合838の3日目（34歩）が14系列に届かないのは「前日の終わり方」が
# 原因だと見立てて実装しましたが、9試合すべてで結果が同一でした。
# 理由は、計画がすでに店の近くで1日を終えているからです。
# 最後に訪れるのが店なので、終了位置は自然と店の上か隣になります。
#
# 計算を増やすだけなので 0 にしています。試すときは 0.30 あたりから。
POSITION_REACH_RATIO = 0.0


def _position_penalty(
    day_result: DayResult,
    kinds: tuple[AgentKind, ...],
    problem: Problem,
    finder,
    next_steps: int,
) -> int:
    """明日いい場所から始められない巡回車が何台いるか。

    **なぜ必要か**

    いまの計画は「その日に取れる玉」だけを見ていて、翌日どこから
    始まるかを考えていません。燃料は先読みするようにしましたが、
    位置は見ていませんでした。

    実戦の試合838は、3日目が34歩しかなく14系列に届きませんでした。
    開始燃料を満タンにしても届かないので、**前日の終わり方**が原因です。
    店から遠い場所で1日を終えると、短い日は移動だけで終わります。

    ここでは「終了位置から一番近い店まで、明日の歩数の3割以内か」を
    数えます。遠い車が少ないほど良い計画と評価します。
    """

    if next_steps <= 0 or POSITION_REACH_RATIO <= 0:
        return 0
    limit = max(1, int(next_steps * POSITION_REACH_RATIO))
    # 店から限界以内のマスを1回だけ集めて使い回す。
    # 候補1つごとに経路表を引き直すと重すぎます。
    near = _near_spot_cells(problem, finder, limit)
    far = 0
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        if day_result.positions[index] not in near:
            far += 1
    return far


_NEAR_CACHE: dict[tuple[int, int], frozenset] = {}


def _near_spot_cells(problem: Problem, finder, limit: int) -> frozenset:
    """どれかの店から limit 歩以内にあるマスの集合。

    店の側から1回だけ探索して作ります。候補ごとに車の位置から
    探索し直すと、1日に何百回も経路表を作ることになり重すぎます。
    """

    key = (id(problem), limit)
    cached = _NEAR_CACHE.get(key)
    if cached is not None:
        return cached
    cells: set[int] = set()
    for spot in problem.spots:
        table = finder.table(spot.position, {}, 0)
        for position, steps in table.steps.items():
            if steps <= limit:
                cells.add(position)
    result = frozenset(cells)
    if len(_NEAR_CACHE) > 32:
        _NEAR_CACHE.clear()
    _NEAR_CACHE[key] = result
    return result


def _candidate_key(
    day_result: DayResult,
    acquired_brands: set[int],
    fuel_limit: int,
    kinds: tuple[AgentKind, ...] = (),
    steps_ahead: int = 0,
    problem: Problem | None = None,
    finder=None,
    next_steps: int = 0,
) -> tuple[int, int, int, int, int]:
    """公式の優先順位でその日の計画を評価する。

    (新しく増えた系列数, その日の系列数, 明日動けない車の少なさ,
     玉数, 終了時の燃料合計)

    3番目に「明日動けない巡回車の少なさ」を入れてあります。
    **日別累計（2番目）より下、玉数（4番目）より上**です。
    今日の系列を1つ落としてまで燃料を残すことはせず、
    玉数を少し削ってでも明日の系列は守る、という順番にしています。
    """

    day_brands = {event.brand for event in day_result.acquisitions}
    new_brands = len(day_brands - acquired_brands)
    short = _fuel_shortfall(day_result, kinds, fuel_limit, steps_ahead)
    if problem is not None and finder is not None and next_steps > 0:
        short += _position_penalty(day_result, kinds, problem, finder, next_steps)
    return (
        new_brands,
        len(day_brands),
        -short,
        len(day_result.acquisitions),
        sum(day_result.fuels),
    )


def _build_candidate(
    label: str,
    problem: Problem,
    planner: DayPlanner,
    kinds: tuple[AgentKind, ...],
    positions: list[int],
    fuels: list[int],
    road_status: dict[int, RoadStatus],
    day_steps: int,
    policy,
    acquired_brands: set[int],
    day_index: int = 0,
    congestion_risk: dict[int, float] | None = None,
    road_heat: dict[int, float] | None = None,
    steps_ahead: int = 0,
    next_steps: int = 0,
) -> Candidate | None:
    try:
        day_plan = planner.plan_day(
            kinds=kinds,
            positions=list(positions),
            fuels=list(fuels),
            road_status=road_status,
            day_steps=day_steps,
            policy=policy,
            acquired_brands=acquired_brands,
            day_index=day_index,
            day_count=problem.day_count,
            congestion_risk=congestion_risk,
            road_heat=road_heat,
        )
        day_result = simulate_day(
            problem=problem,
            agent_kinds=kinds,
            start_positions=tuple(positions),
            start_fuels=tuple(fuels),
            action_plan=day_plan.action_plan,
            day_steps=day_steps,
            road_status=road_status,
        )
    except (InvalidAnswer, ValueError, KeyError, RuntimeError):
        return None
    return Candidate(
        label=label,
        action_plan=day_plan.action_plan,
        day_result=day_result,
        key=_candidate_key(
            day_result, acquired_brands, problem.fuel_limit, kinds, steps_ahead,
            problem, planner.finder, next_steps,
        ),
    )


# 本選の「タイプ選択時間」（本選マップに関するお知らせ）。
# 16x16=60秒 / 24x24=90秒 / 32x32=120秒。
# 締切を過ぎると種別を提出できず、全エージェントが巡回車になります
# （フォーマット資料 p.3 / 質問回答集その2 A53）。補給車が一切使えなくなるので、
# 全部を試し切るより「必ず間に合わせる」ほうを優先します。
KIND_TIME_TABLE: tuple[tuple[int, float], ...] = (
    (16, 60.0),
    (24, 90.0),
    (32, 120.0),
)
KIND_TIME_DEFAULT = 60.0
# 締切のどれだけ手前で打ち切るか。提出と通信の時間を残す。
KIND_SAFETY_RATIO = 0.70
KIND_SAFETY_FLOOR = 8.0
# 試合開始前の試走で使う作戦の数。本番の1日の計画では12個使いますが、
# 試走は台数を決めるためのものなので、回数を稼ぐほうが効きます。
TRIAL_ROUTE_BUDGET = 4
# 1日の計画づくりで1回の判断に試す作戦の数。
#
# 12個から6個に減らしても結果は変わらず、作れる案が1割増えました
# （32x32・235歩で20秒間: 作戦12個78案 / 6個85案、どちらも玉104）。
# 1日の時間は決まっているので、同じ結果なら軽いほうを採ります。
DAY_ROUTE_BUDGET = 6


def kind_time_budget(problem: Problem, announced_seconds: float | None = None) -> float:
    """種別選択に使ってよい秒数。

    announced_seconds が分かっていればそれを、なければマップの大きさから
    本選の表を引きます。どちらの場合も余裕を残して短めに使います。
    """

    if announced_seconds is None:
        side = max(problem.height, problem.width)
        announced_seconds = KIND_TIME_DEFAULT
        for threshold, seconds in KIND_TIME_TABLE:
            if side <= threshold:
                announced_seconds = seconds
                break
        else:
            announced_seconds = KIND_TIME_TABLE[-1][1]
    usable = announced_seconds * KIND_SAFETY_RATIO
    return max(1.0, min(usable, announced_seconds - KIND_SAFETY_FLOOR))


def choose_production_kinds(
    problem: Problem,
    brains: list[Brain],
    budget_seconds: float | None = None,
    log=print,
) -> tuple[AgentKind, ...]:
    """補給車の台数を、実際に1試合ぶん試して決める。

    相手チームの動きは読めないので、道路状態は「全チームが自分と同じように動く」
    と仮定して求めます。学習した統計だけに頼るより、その問題そのものを
    解いてみたほうが確実です。

    ただし本選には「タイプ選択時間」があり、32x32 で全部を試すと
    実測124秒かかって120秒の制限を超えます。そこで、

      1. まず貪欲な方策で「補給車0台〜最大」を一通り試す（台数の決定に直結）
      2. 残り時間があれば、学習したAIで同じ台数を試して上書きする

    という順にし、締切が近づいたらその時点の最良を返します。
    全部試し切れなくても、台数の候補は必ず一周しています。
    """

    started = time.monotonic()
    budget = budget_seconds if budget_seconds is not None else kind_time_budget(problem)

    finder = PathFinder(problem)
    planner = DayPlanner(problem, finder)
    supply_counts = list(range(0, problem.agent_count))

    # 試す順番。台数を一通り押さえることを最優先にする。
    trials: list[tuple[str, object, int]] = []
    greedy = GreedyPolicy(problem)
    for supply_count in supply_counts:
        trials.append(("貪欲", greedy, supply_count))
    # 試走は「補給車の台数を決める」ためのものなので、
    # 1回1回の精度より回数を優先する。作戦を絞ると1試走が軽くなり、
    # 同じ持ち時間で何倍も試せます。
    for index, brain in enumerate(brains):
        policy = Policy(brain, problem, random.Random(31 + index), 0.0,
                        route_budget=TRIAL_ROUTE_BUDGET)
        for supply_count in supply_counts:
            trials.append((brain.name, policy, supply_count))

    seed_base = 104729
    best_kinds = tuple(AgentKind.TOURING for _ in range(problem.agent_count))
    best_key: tuple[int, int, int] | None = None
    best_label = ""
    done = 0
    longest = 0.0

    for label, policy, supply_count in trials:
        elapsed = time.monotonic() - started
        # 1回の試走にかかった最長時間を見て、次が入らないなら打ち切る。
        if done > 0 and elapsed + longest > budget:
            break
        kinds = tuple(
            AgentKind.SUPPLY if index >= problem.agent_count - supply_count else AgentKind.TOURING
            for index in range(problem.agent_count)
        )
        trial_started = time.monotonic()
        score = _rollout_match(problem, planner, kinds, policy)
        longest = max(longest, time.monotonic() - trial_started)
        done += 1
        if score is None:
            continue
        if best_key is None or score > best_key:
            best_key = score
            best_kinds = kinds
            best_label = f"{label} / 補給車{supply_count}台"

    # 一通り試して時間が余っていたら、探索を入れた試走を続ける。
    # 補給車の台数は試合開始時に1回しか決められないので、
    # 持ち時間はできるだけ使い切ったほうが良い。
    # 以前は44通りで打ち切って30秒以上余らせていた。
    attempt = 0
    while time.monotonic() - started + longest < budget and attempt < 100000:
        attempt += 1
        supply_count = supply_counts[attempt % len(supply_counts)]
        if brains:
            brain = brains[attempt % len(brains)]
            noise = (0.05, 0.15, 0.35, 0.55)[(attempt // max(1, len(brains))) % 4]
            policy = Policy(brain, problem, random.Random(seed_base + attempt), noise,
                            route_budget=TRIAL_ROUTE_BUDGET)
            label = f"{brain.name}+探索{noise:.2f}"
        else:
            policy = GreedyPolicy(problem)
            label = "貪欲"
        kinds = tuple(
            AgentKind.SUPPLY if index >= problem.agent_count - supply_count else AgentKind.TOURING
            for index in range(problem.agent_count)
        )
        trial_started = time.monotonic()
        score = _rollout_match(problem, planner, kinds, policy)
        longest = max(longest, time.monotonic() - trial_started)
        done += 1
        if score is not None and (best_key is None or score > best_key):
            best_key = score
            best_kinds = kinds
            best_label = f"{label} / 補給車{supply_count}台"

    elapsed = time.monotonic() - started
    if best_key is not None:
        log(
            f"  事前試走 {done}通り（{elapsed:.1f}秒 / 持ち時間 {budget:.0f}秒）"
            f" 最良: {best_label} → 種類{best_key[0]} / 日別累計{best_key[1]} / 玉数{best_key[2]}"
        )
    else:
        log(f"  事前試走が1つも成立しませんでした。全て巡回車で進めます。")
    return best_kinds


def _rollout_match(
    problem: Problem,
    planner: DayPlanner,
    kinds: tuple[AgentKind, ...],
    policy,
) -> tuple[int, int, int] | None:
    """自分だけで1試合を通しでシミュレートし、公式スコアを返す。"""

    positions = list(problem.agent_positions)
    fuels = [
        problem.fuel_limit if kind == AgentKind.TOURING else 0
        for kind in kinds
    ]
    acquired: set[int] = set()
    daily_total = 0
    noodles = 0
    stays: list[dict[int, int]] = []

    for day_index, day_steps in enumerate(problem.day_steps):
        # 明日から数日ぶんの歩数。燃料を残すかどうかの判断に使う。
        # 燃料に余裕がない試合では、残そうとしても損なだけなので使いません。
        margin = _fuel_margin(
            problem, sum(1 for k in kinds if k == AgentKind.SUPPLY)
        )
        steps_ahead = (
            sum(problem.day_steps[day_index + 1: day_index + 1 + FUEL_LOOKAHEAD_DAYS])
            if margin >= FUEL_LOOKAHEAD_MARGIN
            else 0
        )
        # 明日の歩数。短い日の前ほど、店の近くで終えておく価値があります。
        next_steps = (
            problem.day_steps[day_index + 1]
            if day_index + 1 < len(problem.day_steps) else 0
        )
        road_status = road_status_for_day(problem, stays)
        try:
            day_plan = planner.plan_day(
                kinds=kinds, positions=positions, fuels=fuels,
                road_status=road_status, day_steps=day_steps,
                policy=policy, acquired_brands=acquired,
                day_index=day_index, day_count=problem.day_count,
            )
            day_result = simulate_day(
                problem=problem, agent_kinds=kinds,
                start_positions=tuple(positions), start_fuels=tuple(fuels),
                action_plan=day_plan.action_plan, day_steps=day_steps,
                road_status=road_status,
            )
        except (InvalidAnswer, ValueError, KeyError, RuntimeError):
            return None
        day_brands = {event.brand for event in day_result.acquisitions}
        acquired |= day_brands
        daily_total += len(day_brands)
        noodles += len(day_result.acquisitions)
        positions = list(day_result.positions)
        fuels = list(day_result.fuels)
        # 全チームが自分と同じように動くと仮定して交通量を見積もる。
        stays.append(
            {cell: count * problem.players for cell, count in day_result.stay_counts.items()}
        )

    return (len(acquired), daily_total, noodles)


def play_production_match(
    client: ProconClient,
    brains: list[Brain],
    seed: int = 20260910,
    log=print,
) -> ProductionReport:
    ok, message = client.check_connection()
    log(f"通信確認: {message}")
    if not ok:
        raise ConnectionError(message)

    problem, _raw = client.wait_for_setting(log=log)
    log(f"試合設定: {problem.height}x{problem.width} / エージェント{problem.agent_count}体 / "
        f"{problem.day_count}日 / players={problem.players} / 燃料上限{problem.fuel_limit}")
    log(f"混雑基準{problem.busy_threshold} / 渋滞基準{problem.jammed_threshold} / "
        f"各日のステップ数 {list(problem.day_steps)}")

    budget = kind_time_budget(problem)
    log(f"エージェント種別を決めています（持ち時間 {budget:.0f}秒）…")
    kinds = choose_production_kinds(problem, brains, budget_seconds=budget, log=log)
    supply_count = sum(1 for kind in kinds if kind == AgentKind.SUPPLY)
    client.post_agent_kinds(kinds)
    log(f"提出しました: 巡回車{problem.agent_count - supply_count}台 / 補給車{supply_count}台")

    report = ProductionReport(problem=problem, agent_kinds=kinds)
    # 監視は本番の処理とは別。書き出しに失敗しても試合は止まりません。
    monitor = LiveMonitor()
    monitor.start_match(problem, kinds)
    finder = PathFinder(problem)
    planner = DayPlanner(problem, finder)
    acquired_brands: set[int] = set()
    daily_total = 0
    noodle_total = 0
    predicted: tuple[list[int], list[int]] | None = None
    model = OpponentModel(problem)
    own_stays: list[dict[int, int]] = []

    for day_index, day_steps in enumerate(problem.day_steps):
        # 明日から数日ぶんの歩数。燃料を残すかどうかの判断に使う。
        # 燃料に余裕がない試合では、残そうとしても損なだけなので使いません。
        margin = _fuel_margin(
            problem, sum(1 for k in kinds if k == AgentKind.SUPPLY)
        )
        steps_ahead = (
            sum(problem.day_steps[day_index + 1: day_index + 1 + FUEL_LOOKAHEAD_DAYS])
            if margin >= FUEL_LOOKAHEAD_MARGIN
            else 0
        )
        # 明日の歩数。短い日の前ほど、店の近くで終えておく価値があります。
        next_steps = (
            problem.day_steps[day_index + 1]
            if day_index + 1 < len(problem.day_steps) else 0
        )
        try:
            state = client.wait_for_day(day_index)
        except Exception as error:
            # 状態が取れなくても、その日の回答は出さなければなりません。
            # 前日の位置から全員待機を送って、次の日へ進みます。
            log(f"  {day_index + 1}日目の状態を取れませんでした（{type(error).__name__}）。")
            _submit(client, _fallback_plan(problem.agent_count, day_steps), log)
            predicted = None
            continue

        if predicted is not None:
            mismatch = _describe_mismatch(day_index - 1, predicted, state)
            if mismatch:
                report.mismatches.append(mismatch)
                log("  ずれを検出: " + mismatch)

        positions = [agent.position for agent in state.agents]
        fuels = [agent.fuel for agent in state.agents]

        # 敵の動きを読む材料を更新する。
        model.observe_positions(
            [agent.position for other in state.others for agent in other[1]]
        )
        if day_index > 0:
            model.observe_status(state.road_status, own_stays[-2:])
        congestion_risk = model.congestion_risk(own_stays[-2:])
        road_heat = dict(model.hot_roads(limit=60))
        if road_heat:
            hottest = max(road_heat.values())
            log(f"  敵がよく使う道路 {len(road_heat)}セルを把握（最大 {hottest:.1f}歩/日）")

        started = time.monotonic()
        deadline_monotonic = started + _seconds_until_deadline(
            state.ends_at, problem.day_seconds[day_index]
        )

        candidates: list[Candidate] = []
        base_policies: list[tuple[str, object]] = [("貪欲", GreedyPolicy(problem))]
        for index, brain in enumerate(brains):
            base_policies.append(
                (brain.name, Policy(brain, problem, random.Random(seed + index), 0.0,
                                    route_budget=DAY_ROUTE_BUDGET))
            )
        for label, policy in base_policies:
            candidate = _build_candidate(
                label, problem, planner, kinds, positions, fuels,
                state.road_status, day_steps, policy, acquired_brands, day_index,
                congestion_risk, road_heat, steps_ahead, next_steps,
            )
            if candidate is not None:
                candidates.append(candidate)

        if candidates:
            best = max(candidates, key=lambda item: item.key)
        else:
            plan = _fallback_plan(problem.agent_count, day_steps)
            day_result = simulate_day(
                problem=problem, agent_kinds=kinds,
                start_positions=tuple(positions), start_fuels=tuple(fuels),
                action_plan=plan, day_steps=day_steps, road_status=state.road_status,
            )
            best = Candidate("全員待機", plan, day_result,
                             _candidate_key(day_result, acquired_brands,
                                            problem.fuel_limit, kinds, steps_ahead))
            log("  有効な計画を作れなかったので、全員待機を提出します。")

        revision = _submit(client, best.action_plan, log)
        # サーバーが受理した案。予測はここから取る。
        accepted = best if revision >= 0 else None
        recent: list[dict] = []
        if accepted is None:
            log("  最初の提出に失敗しました。次の提出まで予測を出しません。")
        plan_seconds = time.monotonic() - started

        # ここから、締切まで時間いっぱい探す。
        # うまくいっている間は best の近くを細かく探し、しばらく改善がないと
        # 探索を強めて別の形を試す。改善が出たらまた細かい探索に戻す。
        resubmissions = 0
        attempt = 0
        since_improvement = 0
        noise_levels = (0.05, 0.10, 0.20, 0.35, 0.55)
        noise_index = 0

        while time.monotonic() < deadline_monotonic and attempt < 100000:
            attempt += 1
            noise = noise_levels[min(noise_index, len(noise_levels) - 1)]
            source = attempt % (len(brains) + 2)
            if source == len(brains) + 1:
                label = "系列優先"
                policy = BrandFirstPolicy(problem)
            elif source == len(brains):
                label = f"貪欲+ゆらぎ{noise:.2f}"
                policy = NoisyGreedyPolicy(
                    problem, random.Random(seed + 104729 * attempt), noise
                )
            else:
                brain = brains[source]
                label = f"{brain.name}+探索{noise:.2f}"
                policy = Policy(
                    brain, problem, random.Random(seed + 7919 * attempt), epsilon=noise
                )
            try:
                candidate = _build_candidate(
                    label, problem, planner, kinds, positions, fuels,
                    state.road_status, day_steps, policy, acquired_brands, day_index,
                    congestion_risk, road_heat, steps_ahead, next_steps,
                )
            except MemoryError:
                # メモリが足りないときは、貯めた候補を捨てて探索を打ち切る。
                # **提出済みの回答は残るので0点にはなりません。**
                log("  メモリが足りなくなったので、この日の探索を終わります。")
                candidates = candidates[-1:]
                import gc

                gc.collect()
                break
            except Exception as error:
                # どんな失敗でも試合を止めない。1つの候補を捨てて次へ進む。
                log(f"  候補づくりに失敗しました（{type(error).__name__}）。次を試します。")
                continue
            if candidate is None:
                continue

            # 監視へ「いま何を試しているか」を流す。
            # 0.5秒に1回しか書かないので、探索の速度は落ちません。
            # key は (新しい系列, 日別, 明日の余力, 玉数, 燃料合計)。
            # 燃料の項目を足したとき、ここの位置をずらし忘れて
            # ビューアに「玉0」と出ていました。
            recent.append({
                "label": label,
                "brands": candidate.key[0],
                "daily": candidate.key[1],
                "noodles": candidate.key[3],
                "best": candidate.key > best.key,
            })
            if len(recent) > 12:
                del recent[:-12]
            monitor.searching(
                day_index=day_index,
                candidates=len(candidates),
                best_label=best.label,
                seconds_left=deadline_monotonic - time.monotonic(),
                trying=label,
                best_score=(best.key[0], best.key[1], best.key[3]),
                resubmissions=resubmissions,
                recent=list(reversed(recent)),
                # 暫定の最良案を盤面に出す。探索中も動きが見えます。
                preview=_preview_day(
                    monitor, problem, kinds, positions, fuels, state,
                    best, day_index, day_steps, len(candidates),
                    resubmissions, revision, acquired_brands,
                    daily_total, noodle_total,
                ),
            )
            candidates.append(candidate)
            if candidate.key > best.key:
                if resubmissions < MAX_RESUBMISSIONS_PER_DAY:
                    sent = _submit(client, candidate.action_plan, log)
                    resubmissions += 1
                    if sent < 0:
                        # **送れなかった案を採用したことにしてはいけません。**
                        # サーバーが持っているのは最後に受理された回答なので、
                        # ここで best を進めると予測と実際がずれます。
                        # 実戦で毎試合3〜5件の「ずれを検出」が出ていた原因です。
                        continue
                    revision = sent
                    best = candidate
                    accepted = candidate
                else:
                    # 上限に達したので、これ以上は送らない。
                    # 採用されるのは最後に受理された回答なので、探索も打ち切る。
                    log("  再提出の上限に達したので、この日の探索を終わります。")
                    break
                since_improvement = 0
                noise_index = 0
            else:
                since_improvement += 1
                # 同じ強さで探し続けても出てこないなら、揺らぎを大きくする。
                if since_improvement >= 25 * (noise_index + 1):
                    noise_index = min(noise_index + 1, len(noise_levels) - 1)

        day_brands = {event.brand for event in best.day_result.acquisitions}
        new_brands = day_brands - acquired_brands
        acquired_brands |= day_brands
        daily_total += len(day_brands)
        noodle_total += len(best.day_result.acquisitions)
        # 予測は**サーバーが受理した案**から取る。best ではない。
        final = accepted if accepted is not None else best
        monitor.finish_day(
            day_index=day_index,
            label=final.label,
            candidates=len(candidates),
            resubmissions=resubmissions,
            revision=revision,
            day_steps=day_steps,
            start_positions=list(positions),
            start_fuels=list(fuels),
            positions=list(final.day_result.positions),
            fuels=list(final.day_result.fuels),
            acquisitions=list(final.day_result.acquisitions),
            refuels=list(final.day_result.refuels),
            action_plan=final.action_plan,
            tracks=_tracks_of(problem, kinds, positions, fuels,
                              state.road_status, final.action_plan, day_steps),
            road_status=state.road_status,
            enemy_positions=[
                [agent.position for agent in others]
                for _team, others in getattr(state, "others", [])
            ],
            brand_total=len(acquired_brands),
            daily_total=daily_total,
            noodle_total=noodle_total,
        )
        predicted = (list(final.day_result.positions), list(final.day_result.fuels))
        own_stays.append(dict(final.day_result.stay_counts))

        log(
            f"{day_index + 1}日目: 「{best.label}」を採用（候補{len(candidates)}件 / "
            f"出し直し{resubmissions}回 / revision={revision}）"
            f" 獲得{len(best.day_result.acquisitions)}玉 / 新しい系列{len(new_brands)} / "
            f"補給{len(best.day_result.refuels)}回"
        )

        report.days.append(
            ProductionDayReport(
                day=day_index,
                chosen=best.label,
                revision=revision,
                considered=len(candidates),
                plan_seconds=plan_seconds,
                resubmissions=resubmissions,
                day_score=(len(new_brands), len(day_brands),
                           len(best.day_result.acquisitions)),
                refuels=len(best.day_result.refuels),
                predicted_positions=list(best.day_result.positions),
                predicted_fuels=list(best.day_result.fuels),
            )
        )

    monitor.finish_match(
        f"種類{len(acquired_brands)} / 日別累計{daily_total} / 玉数{noodle_total}",
        report.mismatches,
    )
    report.score = Score(len(acquired_brands), daily_total, noodle_total)
    return report


def _seconds_until_deadline(ends_at: int, day_seconds: int) -> float:
    """締切までの残り秒数。

    サーバーの endsAt は UNIX 時間なので、手元の時計がずれていると当てになりません。
    その日の回答時間からありえない値になっていたら、回答時間のほうを信じます。
    どちらの場合も安全マージンを引きます。
    """

    budget = day_seconds - SAFETY_MARGIN_SECONDS
    remaining = ends_at - time.time() - SAFETY_MARGIN_SECONDS
    if remaining <= 0 or remaining > day_seconds * 2:
        return max(0.5, budget)
    return max(0.5, remaining)


def _preview_day(monitor, problem, kinds, positions, fuels, state, best,
                 day_index, day_steps, candidates, resubmissions, revision,
                 acquired_brands, daily_total, noodle_total):
    """探索中に見せる、暫定の最良案。

    重い処理はしません。軌跡は 1.5 秒に1回だけ作り直します
    （1回およそ数ミリ秒ですが、探索の邪魔をしないため）。
    """

    now = time.monotonic()
    cached = getattr(monitor, "_preview_cache", None)
    if cached is not None and cached[0] == best.label and now - cached[1] < 1.5:
        return cached[2]

    day_brands = {event.brand for event in best.day_result.acquisitions}
    preview = {
        "day": day_index + 1,
        "label": best.label + "（暫定）",
        "candidates": candidates,
        "resubmissions": resubmissions,
        "revision": revision,
        "daySteps": day_steps,
        "startPositions": list(positions),
        "startFuels": list(fuels),
        "positions": list(best.day_result.positions),
        "fuels": list(best.day_result.fuels),
        "acquired": [
            {"step": e.step, "agent": e.agent_index,
             "pos": e.spot_position, "brand": e.brand}
            for e in best.day_result.acquisitions
        ],
        "refuelEvents": [
            {"step": r.step, "agent": r.agent_index, "pos": r.position,
             "before": r.fuel_before, "after": r.fuel_after}
            for r in best.day_result.refuels
        ],
        "actions": [list(row) for row in best.action_plan],
        "tracks": _tracks_of(problem, kinds, positions, fuels,
                             state.road_status, best.action_plan, day_steps),
        "roads": {str(k): int(v) for k, v in state.road_status.items()},
        "enemies": [],
        "brandTotal": len(acquired_brands | day_brands),
        "dailyTotal": daily_total + len(day_brands),
        "noodleTotal": noodle_total + len(best.day_result.acquisitions),
    }
    monitor._preview_cache = (best.label, now, preview)
    return preview


def _tracks_of(problem, kinds, start_positions, start_fuels,
               road_status, action_plan, day_steps) -> list[list[int]]:
    """1歩ごとに各エージェントがいたマスを並べる。

    ビューアに渡して軌跡を描くためのものです。行動値から方向を
    解釈し直すと間違えるので、**公式シミュレータが出した結果を
    そのまま使います。**
    """

    try:
        result = simulate_day(
            problem, kinds, tuple(start_positions), tuple(start_fuels),
            action_plan, day_steps, road_status, record_trace=True,
        )
    except Exception:
        return []
    trace = result.step_positions
    if not trace:
        return []
    # trace[step][agent] を agent ごとの並びに入れ替える
    return [[row[index] for row in trace] for index in range(len(kinds))]


def _submit(client: ProconClient, action_plan: list[list[int]], log) -> int:
    try:
        return client.post_actions(action_plan)
    except ProconApiError as error:
        log(f"  提出に失敗しました: {error}")
        return -1


def _describe_mismatch(
    day: int,
    predicted: tuple[list[int], list[int]],
    state: MatchState,
) -> str | None:
    actual_positions = [agent.position for agent in state.agents]
    actual_fuels = [agent.fuel for agent in state.agents]
    parts = []
    if actual_positions != predicted[0]:
        parts.append(f"位置 予測{predicted[0]} / 実際{actual_positions}")
    if actual_fuels != predicted[1]:
        parts.append(f"燃料 予測{predicted[1]} / 実際{actual_fuels}")
    if not parts:
        return None
    return f"{day + 1}日目終了時: " + " / ".join(parts)


# ---------------- 練習場を回し続ける ----------------
#
# 非公式練習場（procon37arena.online）は15分ごとに試合が開かれ、
# 開始3分前からトークン付きでエンドポイントを叩くと参加表明になります。
# 試合が終わるとクライアントは切断されるので、次の受付までに
# もう一度つなぎ直せば、放置で回し続けられます。
#
# プロトコルは公式の回答システムと同一なので、接続先を変えるだけです。

ARENA_RETRY_SECONDS = 20.0


def play_matches_forever(
    client: ProconClient,
    brains: list[Brain],
    seed: int = 20260910,
    max_matches: int = 0,
    log=print,
) -> list[ProductionReport]:
    """試合が開かれるたびに参加し続ける。

    max_matches が0なら止めるまで続けます。
    途中でどんな失敗が起きても、次の試合まで待ってやり直します。
    1試合が落ちても放置運転が止まらないようにするためです。
    """

    reports: list[ProductionReport] = []
    played = 0
    while max_matches <= 0 or played < max_matches:
        try:
            report = play_production_match(client, brains, seed=seed + played, log=log)
            reports.append(report)
            played += 1
            log("")
            log(f"=== {played}試合目が終わりました: {report.score} ===")
            if report.mismatches:
                log(f"  予測とのズレ {len(report.mismatches)}件")
        except KeyboardInterrupt:
            log("")
            log("中断しました。")
            break
        except ProconApiError as error:
            log(f"  サーバーがエラーを返しました: {error}")
        except (ConnectionError, TimeoutError, OSError) as error:
            log(f"  通信に失敗しました: {error}")
        except Exception as error:  # 放置運転を止めないための最後の砦
            log(f"  想定外の失敗: {error!r}")

        if max_matches > 0 and played >= max_matches:
            break
        log(f"  次の試合の受付まで待ちます（{ARENA_RETRY_SECONDS:.0f}秒ごとに確認）…")
        try:
            time.sleep(ARENA_RETRY_SECONDS)
        except KeyboardInterrupt:
            log("")
            log("中断しました。")
            break
    return reports
