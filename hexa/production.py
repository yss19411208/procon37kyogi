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
import json
import time
from pathlib import Path
from dataclasses import dataclass, field

from hexa.ai import Brain, Policy, SupplyDecision, TouringDecision
from hexa.client import MatchState, ProconApiError, ProconClient
from hexa.opponents import OpponentModel
from hexa.livemonitor import LiveMonitor
from hexa.improve import RoutePolisher
from hexa.matchlog import MatchLog, missed_spots
from hexa.pathfind import FUEL_WEIGHT_STEPS, PathFinder, fuel_weight
from hexa.planner import DayPlanner
from hexa.problem import Problem
from hexa.rules import AgentKind, RoadStatus, fuel_cost
from hexa.scoring import Score
from hexa.simulator import DayResult, InvalidAnswer, road_status_for_day, simulate_day

# **締切のこれだけ手前で必ず打ち切ります。**
#
# 2.0秒にしていましたが、1日60秒のうち実際に探したのは57.4秒
# （ログの `seconds`）でした。日の切れ目のたびに2.6秒、7日なら
# 18秒ぶん探索を捨てていたことになります。
#
# **短くしても「その日が0点」にはなりません。** 良くなるたびに
# 出し直しているので（要項 p.12 (6)「最後に受理された有効な回答を
# 採用します」）、締切ぎりぎりの1回が落ちても、ひとつ前の回答が
# そのまま残ります。margin が守っているのは「最後の1回の送信」
# だけです。送信は数十ミリ秒なので、1.0秒でも十分な余裕があります。
SAFETY_MARGIN_SECONDS = 1.0

# 募集要項 p.12「試合に関する注意事項」:
#   (5) 制限時間内であれば再提出が可能ですが、試合の進行を妨げるほどの回数や
#       容量を送信すると妨害行為とみなし失格となる可能性があります。
#   (6) 最後に受理された有効な回答を採用します。
# 良くなったときだけ出し直し、さらに1日あたりの回数に上限を設ける。
MAX_RESUBMISSIONS_PER_DAY = 15


# 合流先が見つからないとき、補給車を遊ばせずに近づけるか。
SUPPLY_NEVER_IDLES = True


def _approach_toward(*, remaining, position, goal, road_status, finder, table):
    """今日の歩数で、goal にいちばん近づけるセルを返す（なければ None）。"""

    if goal == position:
        return None
    goal_table = finder.table(goal, road_status)
    best = None
    for cell, steps in table.steps.items():
        if steps > remaining:
            continue
        gap = goal_table.steps.get(cell)
        if gap is None:
            continue
        key = (gap, -steps)
        if best is None or key < best[0]:
            best = (key, cell)
    if best is None or best[1] == position:
        return None
    return best[1]


def _approach_lowest_fuel(*, remaining, position, kinds, positions, fuels,
                          road_status, finder, table):
    """いちばん燃料の少ない巡回車へ、今日の歩数で近づける先を返す。

    戻り値は (向かうセル, その車) か None。

    合流は今日のうちに終わらなくてもかまいません。近づいておけば
    **翌日の最初の反映フェーズで拾える**ことがあり、少なくとも
    翌日の迎えが早くなります。補給車は燃料を使わないので、
    近づくこと自体に損はありません。
    """

    target_agent = None
    lowest = None
    for other, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        if fuels[other] >= remaining:
            # 今日の歩数ぶんの燃料があるなら、急いで迎える相手ではない。
            continue
        if lowest is None or fuels[other] < lowest:
            lowest = fuels[other]
            target_agent = other
    if target_agent is None:
        return None
    cell = _approach_toward(
        remaining=remaining, position=position,
        goal=positions[target_agent], road_status=road_status,
        finder=finder, table=table,
    )
    if cell is None:
        return None
    return (cell, target_agent)


# **待ち合わせは店の上でやります。**
#
# 補給車は燃料を1滴も使わず（simulator の `cost` が巡回車だけ）、
# 満タンにするのも無料です。つまり燃料の費用は**待ち合わせの歩数
# だけ**です。だったら待ち合わせの場所を店にすれば、来た巡回車は
# ついでに玉を1つ取れます。
#
# いまどこで補給しているかを数えると（第1958・32x32・9日・10秒磨き）
#
#   補給63回 うち店の上 21回（33%）
#
# 残りの42回は、ただのマスで燃料を渡しているだけでした。これを店に
# 寄せられれば、1日あたり4〜5玉ぶんの拾い所があります（1日平均50玉
# なので1割）。
#
# `MEET_ON_SPOT_STEPS` は「待ち合わせを店にするために、巡回車が
# 余分に歩いてよい歩数」です。
#
# **実測で入れませんでした（0 で止めています）。** 第1958の9日を
# 1日5秒ずつ磨いて通すと:
#
#   | 寄り道 | 玉 | 日別 | 補給 | うち店の上 |
#   |---|---|---|---|---|
#   | 0歩（切る） | 451 | 220 | 44回 | 18回(41%) |
#   | 2歩 | 456 | 221 | 56回 | 19回(34%) |
#   | 4歩 | 436 | 221 | 51回 | 15回(29%) |
#   | 8歩 | 421 | 218 | 45回 | 14回(31%) |
#
# **店の上で渡せた割合はむしろ下がります。** 補給車が店へ寄ると、
# 巡回車は「さっきまで補給車がいた場所」へ向かっているので、
# 落ち合う場所が店からずれるためです。待ち合わせを店にするには、
# 両方の行き先を同時に決める必要があります。
MEET_ON_SPOT_STEPS = 0


# **在庫の多い店を、歩数いくつぶん近いものとして扱うか。** 0 で止めます。
#
# 大島商船（解説集 p23下・玉52）は店の評価式に **−3×在庫数** を
# 置いています（小さいほど優先なので、在庫が多いほど得）。八戸
# （p6上・玉51）は「1店から取れる最大量は min(在庫, 巡回車数)」として
# 各店にその台数分の担当を割り当てています。詳しくは `docs/RIVALS.md`。
GREEDY_STOCK_WEIGHT = 0.0


# **その日に届く車が少ない店を、歩数いくつぶん近いものとして扱うか。**
# 0 で止めます。理由は `_reach_rarity` の上に（鹿児島・解説集p30上）。
REACH_RARITY = 0.0


# **行き先がないときに「近づく」か。**
#
# 準決勝32x32（本物の渋滞・巡回6台）を1歩ずつ数えると、こうでした。
#
#   | 日 | 移動 | 燃料切れ | **燃料はあるのに止まる** |
#   |---|---|---|---|
#   | 1 | 93.2% | 0.0% | 6.8% |
#   | 3 | 46.0% | 0.0% | **54.0%** |
#   | 5 | 47.4% | 0.0% | **52.6%** |
#
# **燃料切れは1歩もありません。** 止まっている歩のほぼ全部が
# 「方策が目標を返さなかった」で、その理由を数えると
#
#   遠すぎる（残り歩数では着けない）: 40〜57%
#   燃料不足（燃料が足りない）      : 22〜45%
#   在庫切れ                        : 7〜25%
#
# でした。32x32は店が32軒しかなく広いので、1日の終わりには
# 「どの店にも今日は着けない」状態になります。そこで車は**その場で
# 立ったまま1日を終えます**。
#
# けれど店は毎朝在庫が戻り、場所も変わりません。**今日20歩ぶん
# 近づいておけば、明日その店に20歩早く着きます**。立っているのは
# まるごと損です。鍵には翌日の届き具合（`tomorrow_reach`）も
# 入っているので、採点はちゃんと見ます。
APPROACH_TOMORROW = False
# 近づくときに残しておく燃料。0なら使い切ってもかまいません。
#
# 使い切ると翌朝1歩も動けないことがあります。鍵には動けない車の数
# （`-immobile`）と立ち往生（`-stranded`）が入っているので採点は
# 見ますが、取り返しがつかない形は作らないほうが安全です。
APPROACH_KEEP_FUEL = 0


def _approach_spot(*, remaining, position, goal, fuel, road_status, finder,
                   table):
    """今日の歩数と燃料で、goal にいちばん近づけるセルを返す。

    補給車用の `_approach_toward` と違い、**燃料も見ます**（巡回車は
    燃料を使うので、届かない先を目指すと途中で立ち往生します）。

    **盤を全部なめてはいけません。** 最初はそう書きましたが、
    32x32では1回1024セル、しかも行き先からの経路表をもう1枚
    引くので、1日に200回呼ばれると1案の採点が3msから100ms近くに
    なります（探索数が30分の1）。

    `table.previous` は最短経路の木なので、**行き先から始点へ
    さかのぼれば**経路上のセルが並びます。その道の上では歩数も
    燃料も単調に増えるので、**最初に予算に収まったセルが
    いちばん遠くまで行ける点**です。長さは経路のぶんだけ（数十）。
    """

    if goal == position:
        return None
    room = fuel - APPROACH_KEEP_FUEL
    steps_of = table.steps
    fuel_of = table.fuel
    previous = table.previous
    cell = goal
    while cell != position:
        steps = steps_of.get(cell)
        need = fuel_of.get(cell)
        if steps is None or need is None:
            return None
        if steps <= remaining and need <= room:
            return cell
        entry = previous.get(cell)
        if entry is None:
            return None
        cell = entry[0]
    return None


class GreedyPolicy:
    """学習を使わない、素直な方策。

    本番で必ず候補に入れる下限です。AIがどれだけ変な判断をしても、
    これより悪い計画が選ばれることはありません。

      巡回車: まだ取っていない系列の店を最優先。次に今日まだ取っていない系列。
              同じ条件なら近い店。行ける店がある限り待機しない。
      補給車: 燃料の不足が一番大きい巡回車のところへ向かう。
    """

    def __init__(self, problem: Problem, brand_first: bool = True) -> None:
        self.problem = problem
        self.fuel_limit = max(1, problem.fuel_limit)
        self.spot_brand = {spot.position: spot.brand for spot in problem.spots}
        self.spot_positions = [spot.position for spot in problem.spots]
        # **系列を先に見るか、近い店から拾うか。**
        #
        # ベンチ29試合を通すと、種類と日別累計は1位とほぼ並んでいて、
        # 負けているのは玉数だけです（32x32で1位の76〜85%）。
        # ところがこの方策は常に「まだ取っていない系列」を最優先に
        # するので、系列をそろえ終わったあとも遠くの店へ車を
        # 走らせます。日別が足りている日には、それがそのまま玉数の
        # 損になります。
        #
        # そこで「近い店から拾うだけ」の形も作れるようにします。
        # 採点は公式の順（種類→日別→玉数）なので、**日別が下がる日
        # には選ばれません**。候補が増えるだけです。
        self.brand_first = brand_first
        # 待ち合わせ先が店かを一発で見るための集合。
        self.spot_cells = frozenset(self.spot_positions)
        # **その系列の店が何軒あるか**（`RARITY_STEPS` で使います）。
        count: dict[int, int] = {}
        for spot in problem.spots:
            count[spot.brand] = count.get(spot.brand, 0) + 1
        self.brand_spots = {spot.position: count[spot.brand]
                            for spot in problem.spots}

    def _reach_rarity(self, positions, kinds, finder, road_status, remaining):
        """店ごとの「その日に届く巡回車が少ないほど大きい」おまけ。

        鹿児島高専（解説集 p30上）:

        > スポットの希少性は、**その日の時間と燃料で到達できる巡回車数**
        > から求める。到達できる巡回車が少ないスポットを優先することで、
        > 取得機会を失いやすいうどんを回収する。

        こちらの希少性（`RARITY_STEPS`）は「その系列の店が何軒あるか」
        だけで、**その日に何台が届くか**を見ていません。第1791437153の
        日別の損10個のうち8個は「届いていたのに別の店を選んだ」でした。
        1台しか届かない店を後回しにすると、その系列は丸ごと落ちます。

        **1日1回だけ数えます**（最初に聞かれたときの位置で数え、以後は
        使い回す）。毎歩数え直すと、1歩に店の数×台数の経路表引きが
        増えます。燃料は方策に全車ぶんが渡ってこないので、ここは
        **歩数だけ**で数えます（届く台数の目安としては十分です）。
        """

        if not REACH_RARITY or positions is None or not kinds:
            return None
        got = getattr(self, "_reach_cache", None)
        if got is not None:
            return got
        cars = [index for index, kind in enumerate(kinds)
                if kind == AgentKind.TOURING]
        count: dict[int, int] = {}
        for car in cars:
            if car >= len(positions):
                continue
            table = finder.table(positions[car], road_status)
            for cell in self.spot_positions:
                steps = table.steps.get(cell)
                if steps is not None and steps <= remaining:
                    count[cell] = count.get(cell, 0) + 1
        most = max(count.values()) if count else 1
        self._reach_cache = {
            cell: REACH_RARITY * (most - reached)
            for cell, reached in count.items()
        }
        return self._reach_cache

    def choose_touring_target(
        self, *, agent_index, step, remaining, position, fuel, stock, reserved,
        acquired_today, acquired_brands, brands_today, road_status, finder,
        kinds=(), positions=None, day_index=0, day_count=1, congestion_risk=None,
        teammate_targets=None, road_heat=None, allow_wait=True,
    ) -> TouringDecision:
        table = finder.table(position, road_status, fuel_weight(remaining, fuel))
        reach_bonus = self._reach_rarity(positions, kinds, finder, road_status,
                                        remaining)
        best = None
        best_key = None
        # 余力を残せる店が1つもなかったときの控え（`KEEP_FUEL_RATIO`）。
        loose = None
        loose_key = None
        # **巡回車が少ない盤面でだけ余力を残します**（`few_touring`）。
        # 本選1回戦6試合（16x16・巡回3台）で日別218→**221**・
        # 玉273→**277**。いっぽう生成した逼迫マップ4つ（巡回4台と
        # 6台）では玉785→**761** と落ちました。
        reserve = (int(self.fuel_limit * KEEP_FUEL_RATIO)
                   if KEEP_FUEL_RATIO > 0.0 and few_touring(kinds) else 0)
        # **「補給車まで戻れるぶん」を控えにするか**（`KEEP_FUEL_REACH`）。
        #
        # 一定の割合だと、燃料1〜4で店に着いてそこから何もできない形が
        # 残ります。本選1回戦第1試合では、燃料が足りず動けなかった歩が
        # 5日で588歩のうち80歩（13.6%）ありました。控えを
        # 「その店から補給車までの燃料」にすれば、**自力で迎えに行けます**。
        reach = None
        if KEEP_FUEL_REACH and positions is not None:
            for other, other_kind in enumerate(kinds):
                if other_kind != AgentKind.SUPPLY:
                    continue
                reach = finder.table(positions[other], road_status).fuel
                break
        # **引き物を手元に置きます。**
        #
        # ここは1日ぶんの再現で何十万回も回ります（第1958の32x32で
        # 21万回 × 店32軒）。`self.` や `table.` を毎回たどるのと、
        # `reachable()` をメソッドで呼ぶのが、そのまま時間になります。
        # 中身は変えていないので答えは1ビットも変わりません
        # （`plan_day` の指紋300件で確認）。
        steps_of = table.steps
        fuel_of = table.fuel
        brand_of = self.spot_brand
        stock_get = stock.get
        reserved_get = reserved.get
        rare_of = self.brand_spots.get
        rarity = RARITY_STEPS
        # **燃料が減ったら、補給車の側へ寄ります**（`SUPPLY_PULL`）。
        pull = None
        if (SUPPLY_PULL > 0.0 and positions is not None
                and fuel < self.fuel_limit * SUPPLY_PULL_RATIO):
            for other, other_kind in enumerate(kinds):
                if other_kind != AgentKind.SUPPLY:
                    continue
                pull = finder.table(positions[other], road_status).steps
                break
        # **今日は届かない店**の控え。全部外れたら、ここへ近づきます。
        far = None
        for spot_position in self.spot_positions:
            if spot_position in acquired_today:
                continue
            room = stock_get(spot_position, 0)
            if room <= 0:
                continue
            steps = steps_of.get(spot_position)
            if steps is None or steps > remaining or fuel_of[spot_position] > fuel:
                # **立ち止まるより近づきます**（理由は `APPROACH_TOMORROW`）。
                if APPROACH_TOMORROW and steps is not None:
                    brand = brand_of[spot_position]
                    mark = (0 if brand not in acquired_brands else 1,
                            0 if brand not in brands_today else 1, steps)
                    if far is None or mark < far[0]:
                        far = (mark, spot_position)
                continue
            brand = brand_of[spot_position]
            # 在庫が残っていれば、他の車が向かっていても構いません。
            # ただし今日まだ取っていない系列の店は1台だけにします
            # （2台目を向けると、別の系列を取る車が減るため）。
            claimed = reserved_get(spot_position, 0)
            # 今日まだ取っていない系列の店は1台だけ。
            fresh_today = brand not in brands_today
            # 系列を見ない形では、**在庫が残っているかぎり何台でも**
            # 向かわせます（玉数だけを狙う候補なので）。
            limit = 1 if (fresh_today and self.brand_first) else room
            if claimed >= limit:
                continue
            # **在庫の多い店を近いものとして扱います**
            # （`GREEDY_STOCK_WEIGHT`）。
            #
            # 大島商船（解説集 p23下・玉52）は店の評価式を数値で
            # 公開していて、**在庫数に −3 の係数**を置いています
            # （小さいほど優先なので、在庫が多いほど得）。八戸
            # （p6上・玉51）は「1店から取れる最大量は min(在庫,
            # 巡回車数) なので、各店にその台数分の担当を割り当てる」
            # と書いています。
            #
            # こちらは設計（`INSERT_STOCK_WEIGHTS` /
            # `PLANNED_STOCK_WEIGHTS`）では在庫を見ていますが、
            # **貪欲の鍵には在庫が入っていませんでした**。貪欲の案は
            # しばしば最良になるので、ここにも入れてみます。
            bonus = GREEDY_STOCK_WEIGHT * (room - 1) if GREEDY_STOCK_WEIGHT else 0
            if reach_bonus is not None:
                # 届く台数が少ないほど大きく引く（＝近いものとして扱う）。
                bonus += reach_bonus.get(spot_position, 0.0)
            if self.brand_first:
                key = (
                    0 if brand not in acquired_brands else 1,
                    0 if fresh_today else 1,
                    claimed,
                    # 店が少ない系列を近いものとして扱う（取り返しが
                    # つかないので）。1軒ならおまけ0。
                    steps + rarity * (rare_of(spot_position, 1) - 1) - bonus,
                )
            else:
                # 近い店から拾うだけ。系列は見ません。
                key = (claimed, steps - bonus)
            if pull is not None:
                gap = pull.get(spot_position)
                if gap is not None:
                    # 歩数に「補給車からの遠さ」を足して選び直します。
                    key = key[:-1] + (steps + SUPPLY_PULL * gap,)
            want = reserve
            if reach is not None:
                # 補給車の側から測った燃料（地形は移動元で決まるので
                # 向きで厳密には違いますが、見立てとしては十分です）。
                back = reach.get(spot_position)
                if back is not None:
                    want = max(want, int(back * KEEP_FUEL_REACH))
            if want and fuel_of[spot_position] + want > fuel:
                # 着いたら身動きが取れない店。控えに回す。
                if loose_key is None or key < loose_key:
                    loose_key = key
                    loose = spot_position
                continue
            if best_key is None or key < best_key:
                best_key = key
                best = spot_position
        if best is None:
            best = loose
        if best is None and far is not None:
            # **今日どの店にも着けない。** それでも立っているのは
            # まるごと損なので、明日いちばん欲しい店へ近づきます。
            best = _approach_spot(
                remaining=remaining, position=position, goal=far[1],
                fuel=fuel, road_status=road_status, finder=finder,
                table=table)
        return TouringDecision(target=best, features=[],
                               candidate_features=[[]] * 2)

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
            # **どの店にも届かない車か。** 届かないなら、その車は
            # 待つしかありません。先に迎えます。
            stuck = 0
            if STUCK_FIRST:
                car_table = finder.table(positions[other], road_status)
                car_fuel = car_table.fuel
                cheapest = None
                for spot_position in self.spot_positions:
                    cost = car_fuel.get(spot_position)
                    if cost is None:
                        continue
                    if cheapest is None or cost < cheapest:
                        cheapest = cost
                if cheapest is None or cheapest > fuels[other]:
                    stuck = 1
            meetings = {positions[other], plan_targets[other]}
            # **ただのマスで渡すより、店の上で渡すほうが得です。**
            # 巡回車がそこまで `MEET_ON_SPOT_STEPS` 歩で行ける店を
            # 待ち合わせ先の候補に足します（下の key で店を優先）。
            if (MEET_ON_SPOT_STEPS > 0
                    and positions[other] not in self.spot_cells):
                near_table = finder.table(positions[other], road_status)
                near_steps = near_table.steps
                closest = None
                for spot_position in self.spot_positions:
                    gap = near_steps.get(spot_position)
                    if gap is None or gap > MEET_ON_SPOT_STEPS:
                        continue
                    if closest is None or gap < closest[0]:
                        closest = (gap, spot_position)
                if closest is not None:
                    meetings.add(closest[1])
            for meeting in meetings:
                if meeting is None or not table.reachable(meeting):
                    continue
                steps = table.steps[meeting]
                if steps > remaining:
                    continue
                # **これから行く店に燃料が足りるか**を最優先で見ます。
                #
                # 以前は「燃料が最も減っている車」を優先していました。
                # ところが実戦958では、山地に囲まれた店へ片道31燃料かかり、
                # 上限46では往復できません（62必要）。補給が要るのは
                # 「いま減っている車」ではなく「これから足りなくなる車」です。
                # この店を4日も取り逃していました。
                need = 0
                target = plan_targets[other]
                if target is not None:
                    other_table = finder.table(positions[other], road_status)
                    if other_table.reachable(target):
                        need = other_table.fuel[target]
                short = 1 if need > fuels[other] else 0
                # **動けなくなる寸前の車を最優先にします。**
                #
                # 実戦1356の7日目で、開始燃料が [11, 2, 27, 98] なのに
                # 補給が歩50・74・92・118と順番でした。最後の車は
                # 118歩まで待たされ、172歩のうち96歩を無駄にしています。
                #
                # ただし**これは貪欲法だけの話**で、本番で採用される案は
                # ほぼ学習済みAIが作ります。実測では玉327→318〜328で
                # 変わりませんでした。CRITICAL_FUEL=0 で無効にしています。
                critical = (
                    0 if (CRITICAL_FUEL > 0
                          and fuels[other] < self.fuel_limit * CRITICAL_FUEL)
                    else 1
                )
                # 同じ車に向かうなら、**店の上で渡せる行き先**を選びます。
                on_spot = 0 if meeting in self.spot_cells else 1
                # **「行き先に燃料が足りない車」を空の車より先にするか。**
                #
                # 行き先が決まっていない車（どこにも行けない車を含む）は
                # need が0になるので short=0 です。つまり**空の車が
                # 「足りている」と見えて後回しになります**。
                # 実戦1791641357（32x32・巡回6台）の5日目・6日目で、
                # 5号車は燃料5〜10のまま96歩のうち4回しか動けていません。
                # `SUPPLY_SHORT_FIRST=False` で、不足よりも
                # 「どれだけ空か」を先に見ます。
                head = ((supply_claims.get(other, 0), -stuck, critical, -short)
                        if SUPPLY_SHORT_FIRST
                        else (supply_claims.get(other, 0), -stuck, critical))
                if SUPPLY_BY_RATE:
                    # 1歩あたり何燃料入れられるか（多いほうが先）。
                    rate = -(deficit / max(1, steps))
                    key = head + (rate, on_spot, steps)
                else:
                    key = head + (-deficit, on_spot, steps)
                if best is None or key < best[0]:
                    best = (key, meeting, other)
        if best is None:
            # **補給車が立ち止まるのは必ず損です。**
            #
            # 補給車は燃料を1滴も使いません（simulator の `cost` が
            # 巡回車だけ）。動くこと自体にコストがないので、
            # 止まっている歩数はそのまま無駄です。
            #
            # 実戦1687の1日目で、補給車は74歩のうち**5歩しか動かず**、
            # 補給は2回だけでした。翌日の開始燃料は46%まで落ちています。
            # 実戦1685でも4日目に39%、5日目に56%を止まって過ごし、
            # 補給は1回と5回。開始燃料は0.61→0.49と枯れていきました。
            #
            # 今日のうちに合流できなくても、**近づいておけば翌日すぐ
            # 拾えます**。いちばん燃料の少ない車へ、今日の歩数で
            # 届くところまで進みます。
            if SUPPLY_NEVER_IDLES:
                approach = _approach_lowest_fuel(
                    remaining=remaining, position=position, kinds=kinds,
                    positions=positions, fuels=fuels,
                    road_status=road_status, finder=finder, table=table,
                )
                if approach is not None:
                    return SupplyDecision(
                        target=approach[0], target_agent=approach[1],
                        features=[], candidate_features=[],
                    )
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


# 残りがこれ以下しか入らない車は、迎えに行かずに飛ばします。
#
# **実測で入れませんでした（0 で止めています）。**
#
# 実戦6試合を記録された渋滞つきで再現（手数3000）:
#
#   飛ばさない : 玉1495（32x32は455）
#   残り2以下を飛ばす: 玉1464（32x32は424）
#
# 補給車はほかに使い道がないので、満タンに近い車でも寄って
# 入れておくほうが、あとの燃料切れを防げるということです。
# 以前 `SUPPLY_SKIP_RATIO` で測ったときと同じ結論でした。
SUPPLY_SKIP_ROOM = 0


# **燃料が減った車は、補給車の側の店を選びます。**
#
# 第1703（32x32・6台+補給1台・9日）で上限を見積もると
#
#   満タンのつもり        : 726玉
#   持っている燃料のまま  : 469玉   ← 実際は442玉（上限の94%）
#   1位                   : 611玉
#
# **回り方はもう上限近くまで出ています。足りないのは燃料です。**
# 1歩2歩で燃料1なので、85歩を使い切るには1台1日43燃料要ります。
# 6台9日で2295燃料。最初に持っているのは678だけなので、残りは
# 補給で入れるしかありません。補給車は燃料を使わず満タンにできる
# ので**入れられる総量に上限はなく、落ち合えるかだけ**が問題です。
#
# ところが落ち合う回数を数えると9日で44〜63回もあります。つまり
# 回数ではなく**1回あたりの量**が小さい（ほぼ満タンの車に入れて
# いる）。空に近い車と落ち合えていません。
#
# そこで、燃料が減った車は「補給車に近い店」を選ぶようにします。
# `SUPPLY_PULL` は補給車への距離をどれだけ重く見るかです。0 で止めます。
#
# **入れました。** 準決勝32x32（第1791641357試合・巡回6台・燃料80・
# 歩96・記録どおりの渋滞）で玉230→**240**。この盤面は
#
#   | 条件 | 玉の上限（厳密） |
#   |---|---|
#   | 手持ちの燃料だけ | **243** |
#   | 1台1日0.5回の補給 | 356 |
#   | 1位の実績 | 304 |
#
# で、こちらは補給なしの上限に張り付いていました（探索を10倍に
# しても230→231）。**足りないのは探索ではなく燃料の配達**で、
# 待ち合わせを安くする手だけが効きます。
#
# 16x16では悪くなりません（準決勝16x16で玉59のまま、本選1回戦
# 6試合で日別218・玉272対273）。生成した24x24・32x32の12マップ
# では玉3791→3820でした。
SUPPLY_PULL = 0.5
# **補給車は「身動きの取れない車」を先に迎えます。**
#
# これまでの優先は「燃料の減りがいちばん大きい車」でした。ところが
# 残り10で3燃料先に店がある車は困っていません。困っているのは
# **どの店にも届かない車**です。第1703で待ちの理由を数えると、
# 「燃料が足りず店に行けない」340歩と「燃料切れ」271歩で7割でした。
# **入れました。** 貪欲だけの29試合では順位合計512のまま動きません
# でしたが（`SUPPLY_BY_RATE` が同じ判断を含むため）、磨きつきで
# 測ると本選1回戦6試合（16x16・燃料40）で
#
#   切る : 日別222 玉278
#   入れる: 日別**223** 玉277
#
# 日別は玉数より上位の得点なので入れます。生成12マップでは
# 日別・玉ともに1つも変わりませんでした（巡回4〜6台では、どの店にも
# 届かない車がそもそも出にくい）。
#
# **なぜ必要か。** 補給車の優先は `-short`（これから行く店に燃料が
# 足りない車）を見ますが、**どこにも行けない車は行き先が None で
# need が0になり、「足りている」と見えます**。第1試合の2日目、
# セル17で燃料1の2号車がまさにこれで、補給車は2マス隣から出発して
# 盤面の反対側へ行きました。
STUCK_FIRST = True
# **補給の優先で「行き先に燃料が足りない車」を空の車より先にするか。**
# 理由は `choose_supply_target` の中に。
SUPPLY_SHORT_FIRST = True
# **1歩あたりどれだけ入れられるかで迎え先を選ぶ。**
#
# これまでは「減りがいちばん大きい車」を選んでいました。第1627の
# 2日目がその失敗そのものです。補給車はマス71、巡回車は
#
#   車0 マス122（残り23・約10歩）
#   車2 マス 13（残り14・約12歩）
#   車3 マス224（残り 2・約24歩） ← 減りが最大なのでこちらへ
#
# 36歩の日に24歩かけて車3だけを満タンにし、車0と車2は残り2と0まで
# 枯れました。翌日は3台のうち2台が1歩も動けず、玉5・日別5。
#
# 近い車から順に満タンにすれば、同じ36歩で2台に入れられます。
# そこで「入れられる量 ÷ かかる歩数」で選びます。0 で止めます。
# **実測で入れませんでした（False で止めています）。**
# 貪欲だけの29試合では+25位ぶん効きますが、磨きを入れた29試合では
# 確認できていません。`RARITY_STEPS` と同じ性質（貪欲の受け持ちを
# 変える）なので、同じ理由で打ち消される見込みです。
SUPPLY_BY_RATE = False
# **店が1軒しかない系列を先に取る**（鹿児島高専の「スポットの希少性」）。
#
# 第37回の解法紹介で、鹿児島高専は「スポットの希少性は、その日の
# 時間と燃料で到達できる巡回車数から求める。**到達できる巡回車が
# 少ないスポットを優先**することで、取得機会を失いやすいうどんを
# 回収する」と書いています。
#
# 第1627（16x16）の盤面はまさにそれでした。**系列10個のうち7個は
# 店が1軒だけ**（系列0はマス0、1は10、4は79、6は144、7は192、
# 8は224、9は245）。残り3系列は2〜5軒あります。1軒しかない系列を
# 落とすと取り返せません。実戦では左上の角にある系列0と1を
# 2・3・4日目に落として日別36でした（1位は44）。
#
# 毎歩「何台が届くか」を数え直すのは重いので（1日に21万回呼ばれる）、
# **その系列の店の数**で代わりにします。こちらは試合中ずっと変わり
# ません。
#
# **鍵の桁として入れるのはやりすぎでした。** 順位の条件にしてしまうと、
# 近くに別の未取得系列があっても、1軒しかない系列へ盤面の端まで車を
# 走らせます。そこで「**何歩ぶん近いものとして扱うか**」という
# 歩数のおまけにします。店が1軒なら0、2軒なら `RARITY_STEPS` 歩、
# 5軒なら4倍ぶん遠く見ます。0 で止めます。
RARITY_STEPS = 0
# **設計の迎え先（燃料が切れる地点）を優先して待ち受けるか。**
# **集合場所へ帰り始める余裕（歩）。**
#
# 渋滞は1歩ごとに変わるので、歩数表どおりに着くとは限りません。
# これだけ早く出ます。間に合わないと満タン1台ぶんが丸ごと消えます。
HOME_MARGIN = 2


# **補給車は「追いかける」のではなく「待ち伏せる」か。**
#
# 準決勝32x32（本物の渋滞）で、補給車が何歩使って何回配れたかを
# 数えるとこうでした。
#
#   | 日 | 補給の回数 | 配った燃料 | 補給車の歩 |
#   |---|---|---|---|
#   | 1 | **1回** | 50 | 59/64 |
#   | 2 | **1回** | 80 | 79/80 |
#   | 3 | 5回 | 248 | 95/96 |
#   | 6 | 9回 | 404 | 94/96 |
#
# **補給車は1日の歩をほぼ全部使って動いているのに、初日と2日目は
# 1回しか配れていません。** 必要なのは1日120燃料（6台×20。厳密な
# 上限から、1位に並ぶのに要るのは1台1日20燃料）。そして3日目の朝の
# 燃料は **[80, 22, 0, 1, 69, 74]** で、2台が1歩も動けませんでした。
#
# 理由は**追いかけているから**です。迎え先を「その車のいる場所」や
# 「その車の次の行き先」にすると、着いたころには車は先へ行っています。
# 相対速度がほぼ0なので、1日かけて1回しか会えません。
#
# そこで**待ち伏せ**に変えます。`PlannedRoutePolicy` はその車の
# 残りの順路を全部持っているので、
#
#   1. その車が順路の各店に**いつ着くか**を積み上げて出す
#   2. 補給車がそこへ**何歩で行けるか**を引く
#   3. 補給車のほうが先に着ける店のうち、**いちばん後ろの店**を選ぶ
#
# とすれば、先回りして待てます。いちばん後ろを選ぶのは、そのとき車が
# いちばん空だからです（1回の補給量が大きいほど得）。
SUPPLY_INTERCEPT = True
# 待ち伏せの余裕（歩）。渋滞で遅れても間に合うように早く出ます。
SUPPLY_INTERCEPT_MARGIN = 2
# **待ち伏せ先を「いちばん手前」にするか**（既定は「いちばん後ろ」）。
#
# いちばん後ろは1回の補給量が大きい代わりに、会うまでが長く、磨きで
# 順路が変わると待ち合わせが外れます。いちばん手前なら**会う回数が
# 増える**かわりに、そのとき車はまだ空いていません。
SUPPLY_INTERCEPT_FIRST = True
# **困っている車がいるときは待ち伏せをやめて救援に回すか。**
# 理由は `choose_supply_target` の待ち伏せの中に。
SUPPLY_INTERCEPT_RESCUE_FIRST = False
# 順路の先を何店まで見るか（深くすると経路表を引く回数が増えます）。
SUPPLY_INTERCEPT_DEPTH = 6


# **夜の合流先へ向かい始める余裕（歩）。**
#
# 渋滞は1歩ごとに変わる可能性があるので、歩数表どおりに着くとは
# 限りません。`night_margin` ぶん早く出ます。0だと間に合わない日が
# 出て、満タンが1台ぶん丸ごと消えます。
NIGHT_PARK_MARGIN = 3


# 理由は `PlannedRoutePolicy.choose_supply_target` の中に。
# **実測で入れませんでした（False で止めています）。**
# 32x32の4試合を手数1500で通すと、歩の使用率は87.8%→89.7%、玉は
# 2223→2241 と上がりますが、**日別が733→729 と落ちます**。日別は
# 玉より上位の得点なので採用できません。
SUPPLY_PREFER_HINT = False
# **時刻つきの約束に、何歩の遅れまで許すか。**
# 渋滞で歩数表どおりに進まないことがあるので、少し緩めます。
SUPPLY_PLAN_SLACK = 3
# **着いた先で身動きが取れなくなる店は、後回しにします。**
#
# 行き先の判定は「その店まで燃料が足りるか」だけでした。だから
# ぎりぎり届く店へ行き、着いたところで次のどこにも行けなくなります。
# そこで**着いたあとにこれだけ残る店**を先に選びます。残る店が
# 1つもないときだけ、ぎりぎりの店へ行きます。
#
# 値は燃料上限に対する割合です。0 で止めます。
#
# **巡回車が3台以下の盤面でだけ効かせます**（`few_touring`）。
# 本選1回戦の実問題6つ（16x16・燃料40・巡回3台）を渋滞つき・
# 手数800で通すと
#
#   0（切る）  : 日別218 玉273
#   **0.1**    : 日別**221** 玉**277**
#   0.2        : 日別215 玉272
#
# いっぽう生成した逼迫マップ4つ（24x24は巡回4台・32x32は巡回6台）
# では 日別430/430 のまま 玉785→**761** と落ちました。台数が多い
# ほど1台の立ち往生が軽いので、余力を残す損のほうが勝ちます。
KEEP_FUEL_RATIO = 0.1
# **「足りない系列を拾う」直しは、日別が本当に増えるときだけ採る。**
#
# `cover_missing_brands` は、足りない系列の店を車の**先頭**に
# 差し込みます。必ず着く代わりに、後ろの店を落とすので玉数は
# 減ります。日別累計は玉数より上位の得点なので、**日別が増えるなら**
# それでいいのですが、採否は鍵全体の大小で見ていました。
#
# 第1627（16x16）の実戦ログ:
#
#   1日目「足りない系列を拾いました（日別 7 → 7）」
#   2日目「足りない系列を拾いました（日別 5 → 5）」
#
# **どちらも日別は増えていません。** 鍵の別の要素（明日届かない
# 系列・立ち往生・燃料の余力）が上がったので採られ、玉数だけが
# 減っていました。それらは磨きの仕事で、こちらの差し込みは
# 「先頭に置く」という乱暴な手しか持っていません。
#
# そこで**日別が増えるときだけ**採ります。
BRAND_FIX_DAILY_ONLY = True
# この割合より燃料が減った車だけが引き寄せられます。
SUPPLY_PULL_RATIO = 0.35
# **設計どおりに歩くとき、歩数最短の経路を使うか。**
#
# **実測では入れませんでした（False）。** 設計と実行で経路の見方が
# 5%食い違っている（下の `choose_touring_target` に数字）のは本当
# ですが、素直に合わせると悪くなります:
#
#   過去29試合（手数2000・順位合計） 318 → **328**（2勝5敗）
#   準決勝32x32（1試合）             240 → 238
#
# 実行側の「燃料を考えた経路」は歩が5%伸びるかわり燃料が7%安く、
# **燃料のほうが縛っている**ので、そちらが正しい選択でした。
#
# 食い違いを直すなら、実行を設計に合わせるのではなく**設計を実行に
# 合わせる**（`route_cost` を燃料こみの重みで測る）のが筋です。
PLANNED_USES_SHORTEST = False
# **控えを「その店から補給車まで戻れる燃料」の何倍にするか。**
# 0 で止めます（`KEEP_FUEL_RATIO` の一定割合だけになる）。
KEEP_FUEL_REACH = 0.0


class PlannedRoutePolicy:
    """前もって決めた順路を、そのままなぞる方策。

    **1日の初めに全車の行き先をまとめて決め**、その並びどおりに
    訪問します。1手ずつ「次に良い店」を選ぶのではなく、
    1日を通した配分を先に設計するのが違いです。

    実戦の渋滞で測ると（1位の玉数に対する率）

        いまの実装   : 1535=89% / 1533=88%
        **まとめて設計**: 1535=92% / 1533=92%

    補給車の動きは貪欲法に任せます（巡回車の順路だけを設計）。
    """

    def __init__(self, problem, routes: dict[int, list[int]],
                 supply_route: list | None = None,
                 brand_first: bool = True, park: int | None = None,
                 night_park: dict[int, int] | None = None,
                 home=None, supply_cells: list[int] | None = None,
                 supply_plan: list[tuple[int, int, int]] | None = None):
        self.problem = problem
        # **巡回車が夜を越すマス（全車共通）。**
        #
        # 順路の最後に入れるだけでは**着きません**。`choose_touring_target`
        # は「もう取った店」と「在庫の切れた店」を飛ばすので、集合場所を
        # 店にすると、

        #   - 朝そのマスに立っている車はその場で取ってしまう
        #   - 在庫は巡回車の台数より少ないことが多い
        #
        # のどちらかで**飛ばされます**。準決勝32x32で集合場所を店683に
        # して通すと、1日目は6台中4〜6台そろうのに2日目以降は0台でした。
        # 全車が集合場所から朝を始めるので、全車がそこを飛ばしたのです。
        #
        # そこで「帰る」は在庫も取得済みも見ない別の決まりにします。
        # 残り歩数が帰るのにぎりぎりになったら、そこへ向かいます。
        # 1マスを渡せば全車共通、辞書なら**車ごと**です。帰る車を
        # 選べることが大事です。準決勝32x32で**全車を帰らせる**と
        # 夜の合流は6台中4〜6台まとまりましたが、往復に歩を使うので
        # 日別が132→101に落ちました（日別は玉数より上位なので負け）。
        # 遠くの系列を持つ車は帰らせず、近い車だけ帰すのが得です。
        self.home = home
        self.home_margin = HOME_MARGIN
        # **夜に補給車を置くマス（補給車ごと）。**
        #
        # `park`（ハブ）との違いは**出発する時刻**です。ハブは1日の
        # 最初からそこに居座るので、日中の補給が1つの点でしか
        # できません。こちらは**日中はふつうに迎えに回り、届くだけの
        # 歩数が残っているぎりぎりになってから**合流先へ向かいます。
        #
        # なぜ夜に合流するのか。`simulator` の補給（1.4）は歩0の
        # 行動段階でも走るので、同じマスで夜を越した巡回車は**翌朝
        # 1歩も動かないうちに満タン**になります。巡回車の歩も
        # 補給車の歩も使わない、いちばん安い満タンです。
        #
        # 準決勝32x32を厳密な上限で切ると、実際の朝の燃料だと玉243
        # ですが**毎朝満タンなら373**。1位は304で、こちらの実戦は
        # 216でした。**1位の304は、こちらの「実際の燃料での上限243」
        # より上**です。つまり1位は燃料をこちらより多く手にしていて、
        # 勝負はそこしかありません。
        # **補給車が通るマスの並び**（`design_loop` の逆回り用）。
        #
        # `supply_route` は「どの車を迎えるか」の並びで、満タンに近い車は
        # 飛ばします。逆回りは**車ではなくマスを順にたどる**ので、別の
        # 入れ物にします。空なら何も変わりません。
        self.supply_cells = list(supply_cells or [])
        # **時刻つきの迎え先の列**（歩, セル, 車）。`plan_supply_meetings` が
        # 作ります。理由はその関数の上に。
        self.supply_plan = list(supply_plan or [])
        self.night_park = dict(night_park or {})
        # 合流先へ向かい始める余裕（歩）。ぎりぎりだと渋滞で落ちます。
        self.night_margin = NIGHT_PARK_MARGIN
        self.routes = {index: list(spots) for index, spots in routes.items()}
        # **補給車の「迎える順番」です。**
        #
        # 中身は車の番号か、(迎え先のセル, 車の番号) の組です。
        # 組のほうは `design_with_supply` が作る「燃料が切れる地点」で、
        # 迎えに行く手がかりになります。
        #
        # 待ち合わせの時刻は決めません。燃料が切れた車はその場で
        # 止まって待つので、**順番だけ合っていれば必ず拾えます**。
        self.supply_route = list(supply_route or [])
        # **補給車を置く店（ハブ）。** None なら動かします。
        #
        # 置く理由は `hexa/allocate.py` の `design_hub` に。要点は、
        # 6台を毎日満タンにするには補給車の持ち歩では往復が間に合わず、
        # **全車の周回が通る一点に置けば移動ゼロで配れる**ことです。
        self.park = park
        # 順路を使い切ったあとの受け持ち。**ここが効きます。**
        # 設計した店にはほとんど着いていないことがあります（燃料が
        # 少ない日は `MEET_SUPPLY_EARLY` で補給車へ向かうため）。
        # つまり玉の大半はこの受け持ちが取っています。
        self._fallback = GreedyPolicy(problem, brand_first=brand_first)

    def _home_of(self, agent_index):
        """その車の集合場所（なければ None）。"""

        if self.home is None:
            return None
        if isinstance(self.home, dict):
            return self.home.get(agent_index)
        return self.home

    def _go_home(self, cell):
        """集合場所へ向かう決定。"""

        return TouringDecision(target=cell, features=[],
                               candidate_features=[[]] * 2)

    def _can_return(self, spot, home, kwargs) -> bool:
        """その店へ寄ってから集合場所へ**帰れるか**。

        **ここが夜合流の生死を分けます。** 順路の最後に集合場所を
        入れても、planner は順路を使い切ると貪欲に任せるので、
        そのあいだに燃料も歩数も使い切ってしまいます。準決勝32x32で
        集合場所を決めて通すと、1日目は6台中5〜6台そろうのに
        2日目以降は0〜2台しか帰って来ませんでした。

        そこで**1歩進むごとに「ここへ寄ってもまだ帰れるか」を
        確かめ**、帰れなくなる一歩手前で引き返します。
        """

        finder = kwargs["finder"]
        road_status = kwargs["road_status"]
        here = finder.table(kwargs["position"], road_status)
        leg_steps = here.steps.get(spot)
        leg_fuel = here.fuel.get(spot)
        if leg_steps is None or leg_fuel is None:
            return False
        there = finder.table(spot, road_status)
        back_steps = there.steps.get(home)
        back_fuel = there.fuel.get(home)
        if back_steps is None or back_fuel is None:
            return False
        if leg_steps + back_steps + self.home_margin > kwargs["remaining"]:
            return False
        return leg_fuel + back_fuel <= kwargs["fuel"]

    def choose_touring_target(self, *, agent_index, acquired_today, stock,
                              **kwargs) -> TouringDecision:
        # **帰る時刻になったら、在庫を見ずに集合場所へ向かいます。**
        #
        # 在庫も取得済みも見ないのが肝です。集合場所を店にすると、
        # 朝そこに立っている車がその場で取ってしまい、在庫も
        # 巡回車の台数より少ないことが多いので、ふつうの決まりでは
        # **飛ばされてしまいます**。
        home = self._home_of(agent_index)
        if home is not None and kwargs["position"] != home:
            table = kwargs["finder"].table(kwargs["position"],
                                           kwargs["road_status"])
            steps = table.steps.get(home)
            need = table.fuel.get(home)
            if steps is None or need is None or need > kwargs["fuel"]:
                # 帰れないので、集合場所のことは忘れて玉を取ります。
                home = None
            elif steps + self.home_margin >= kwargs["remaining"]:
                return self._go_home(home)
        queue = self.routes.get(agent_index)
        while queue:
            spot = queue[0]
            # すでに取った店や、在庫の切れた店は飛ばす。
            if spot in acquired_today or stock.get(spot, 0) <= 0:
                queue.pop(0)
                continue
            if home is not None and not self._can_return(spot, home, kwargs):
                # 寄ったら帰れない。順路を切り上げて帰ります。
                if kwargs["position"] != home:
                    return self._go_home(home)
                queue.pop(0)
                continue
            return TouringDecision(target=spot, features=[],
                                   candidate_features=[[]] * 2)
        # 順路を使い切ったら、ふつうの貪欲に任せる。
        got = self._fallback.choose_touring_target(
            agent_index=agent_index, acquired_today=acquired_today,
            stock=stock, **kwargs
        )
        if (home is not None and got.target is not None
                and got.target != home
                and not self._can_return(got.target, home, kwargs)
                and kwargs["position"] != home):
            return self._go_home(home)
        return got

    def _intercept(self, car, position, table, remaining, fuels, positions,
                   finder, road_status):
        """**その車の順路のうち、先回りして待てるいちばん後ろの店。**

        理由と実測は `SUPPLY_INTERCEPT` の上に。要点は、迎え先を
        「その車のいる場所」にすると追いかけることになり、1日に1回しか
        会えないことです。

        その車がいつ着くかは、順路を順にたどって歩数を積み上げます。
        経路は**その車と同じ見方**（燃料を考えた重み）で引きます。
        補給車は燃料を1滴も使わない（要項 p.11）ので、自分の側は
        歩数最短で引きます。
        """

        line = self.routes.get(car)
        if not line:
            return None
        from hexa.pathfind import fuel_weight

        weight = fuel_weight(remaining, fuels[car] if car < len(fuels) else 0)
        here = positions[car]
        clock = 0
        best = None
        for spot in line[:SUPPLY_INTERCEPT_DEPTH]:
            leg = finder.table(here, road_status, weight).steps.get(spot)
            if leg is None:
                break
            clock += leg
            if clock > remaining:
                break
            mine = table.steps.get(spot)
            if (mine is not None
                    and mine + SUPPLY_INTERCEPT_MARGIN <= clock
                    and mine <= remaining):
                # 先に着ける。既定はいちばん後ろ（上書きし続ける）。
                best = spot
                if SUPPLY_INTERCEPT_FIRST:
                    return spot
            here = spot
        return best

    def choose_supply_target(self, **kwargs):
        """決めておいた順に迎えに行く。ハブがあるならそこに居続ける。

        **迎え先は「その車の燃料が切れる場所」です。**
        切れた車はそこで止まって待つので、時刻を合わせる必要が
        ありません。順番だけ守れば必ず拾えます。

        ただし設計どおりに進まないこともあるので、

          - すでに満タンの車は飛ばす
          - その車の**いまいる場所**が迎え先より近ければそちらへ向かう

        という2点だけ、現場の様子を見て直します。
        """

        # **夜の合流先が決まっているなら、間に合う時刻に向かいます。**
        if self.night_park:
            cell = self.night_park.get(kwargs["agent_index"])
            if cell is not None:
                table = kwargs["finder"].table(kwargs["position"],
                                               kwargs["road_status"])
                steps = table.steps.get(cell)
                if steps is not None:
                    if kwargs["position"] == cell:
                        # 着いています。動かずに夜を待ちます。
                        return SupplyDecision(target=cell, target_agent=None,
                                              features=[],
                                              candidate_features=[])
                    if steps + self.night_margin >= kwargs["remaining"]:
                        return SupplyDecision(target=cell, target_agent=None,
                                              features=[],
                                              candidate_features=[])

        # **時刻つきの迎え先の列があるなら、それに従います。**
        #
        # `supply_route`（迎える車の順番）や `hint`（設計が出した「燃料が
        # 尽きる地点」）と違い、ここは**何歩目にどのセルで会うか**まで
        # 決まっています。理由と実測は `plan_supply_meetings` の上に。
        while self.supply_plan:
            when, cell, car = self.supply_plan[0]
            step = kwargs["step"]
            if when < step - SUPPLY_PLAN_SLACK:
                # 約束の時刻を過ぎました。次の約束へ。
                self.supply_plan.pop(0)
                continue
            table = kwargs["finder"].table(kwargs["position"],
                                           kwargs["road_status"])
            need = table.steps.get(cell)
            if need is None or need > kwargs["remaining"]:
                self.supply_plan.pop(0)
                continue
            if kwargs["position"] == cell:
                if when <= step:
                    # 会えました（補給は反映フェーズで走ります）。
                    self.supply_plan.pop(0)
                    continue
                # 早く着いたので、約束の時刻まで待ちます。
                return SupplyDecision(target=cell, target_agent=car,
                                      features=[], candidate_features=[])
            if need <= (when - step) + SUPPLY_PLAN_SLACK:
                return SupplyDecision(target=cell, target_agent=car,
                                      features=[], candidate_features=[])
            # 間に合いません。この約束は捨てます。
            self.supply_plan.pop(0)

        # **逆回りのマスの並びがあるなら、それをたどります。**
        # 理由と実測は `hexa/allocate.py` の `design_loop` の上に。
        while self.supply_cells:
            cell = self.supply_cells[0]
            if cell == kwargs["position"]:
                self.supply_cells.pop(0)
                continue
            table = kwargs["finder"].table(kwargs["position"],
                                           kwargs["road_status"])
            steps = table.steps.get(cell)
            if steps is None or steps > kwargs["remaining"]:
                self.supply_cells.pop(0)
                continue
            return SupplyDecision(target=cell, target_agent=None,
                                  features=[], candidate_features=[])

        # **ハブがあるなら、そこから動きません。**
        # 着いたあとは `start_route` が空の順路を返すので、planner は
        # 待ちに回します。`SUPPLY_IDLE_WAIT` が短いので毎歩ここへ
        # 聞きに来て、またハブを返す（＝居座る）形になります。
        if self.park is not None:
            return SupplyDecision(target=self.park, target_agent=None,
                                  features=[], candidate_features=[])
        queue = self.supply_route
        fuels = kwargs["fuels"]
        positions = kwargs["positions"]
        plan_targets = kwargs["plan_targets"]
        remaining = kwargs["remaining"]
        finder = kwargs["finder"]
        position = kwargs["position"]
        road_status = kwargs["road_status"]
        table = finder.table(position, road_status)
        while queue:
            entry = queue[0]
            if isinstance(entry, tuple):
                hint, car = entry
            else:
                hint, car = None, int(entry)
            # **ほぼ満タンの車は飛ばします。**
            #
            # 合流すると満タンになりますが、残り2しか入らない車に
            # 寄るのは補給車の歩数の無駄です。32x32（巡回6台）では
            # 補給車の歩数が足りず、燃料切れで止まった歩数が208歩
            # ありました。寄り先を1つ飛ばせば、そのぶん本当に
            # 空いている車へ回せます。
            room = self.problem.fuel_limit - fuels[car] if car < len(fuels) else 0
            if car >= len(fuels) or room <= SUPPLY_SKIP_ROOM:
                queue.pop(0)
                continue
            # **設計の迎え先（燃料が切れる地点）を優先します。**
            #
            # 以前は「いる場所・行き先・設計の迎え先」のうち
            # **いちばん近いところ**で待ち受けていました。ところが
            # 近いところで落ち合うと、その車はまだ燃料が残っている
            # ので**入る量が小さい**のです。
            #
            # 第1703（32x32）で補給を数えると9日で44〜63回もあるのに
            # 1回あたり9〜111燃料とばらばらで、ほぼ満タンの車に入れて
            # いる回が多くありました。歩を100%使うには1日4回ぶんの
            # 満タンが要るので、**回数ではなく1回の量**が足りません。
            #
            # 設計の迎え先は `design_with_supply` が出した「その車の
            # 燃料が切れる地点」です。そこで落ち合えば満タンぶん
            # 入ります。鹿児島高専（51）の「先回り」と同じ考えです。
            # **待ち伏せ**（`SUPPLY_INTERCEPT`）。理由と実測は上に。
            if SUPPLY_INTERCEPT:
                spot = self._intercept(car, position, table, remaining,
                                       fuels, positions, finder, road_status)
                if spot is not None:
                    if spot == position:
                        # もう待ち伏せ先にいる。ここで待ちます。
                        return SupplyDecision(
                            target=position, target_agent=car, features=[],
                            candidate_features=[])
                    return SupplyDecision(target=spot, target_agent=car,
                                          features=[], candidate_features=[])
            if SUPPLY_PREFER_HINT and hint is not None:
                steps = table.steps.get(hint)
                if steps is not None and steps <= remaining:
                    if hint == position:
                        queue.pop(0)
                        continue
                    return SupplyDecision(target=hint, target_agent=car,
                                          features=[], candidate_features=[])
            # いる場所と、その車がこれから向かう店と、設計の迎え先。
            # いちばん近いところで待ち受けます。
            best = None
            for candidate in (positions[car], plan_targets[car], hint):
                if candidate is None:
                    continue
                steps = table.steps.get(candidate)
                if steps is None or steps > remaining:
                    continue
                if best is None or steps < best[0]:
                    best = (steps, candidate)
            if best is not None:
                if best[1] == position:
                    # もう着いているので、次の車へ。
                    queue.pop(0)
                    continue
                return SupplyDecision(target=best[1], target_agent=car,
                                      features=[], candidate_features=[])
            # 今日のうちに届かないなら、**届くところまで近づきます。**
            approach = _approach_toward(
                remaining=remaining, position=position, goal=positions[car],
                road_status=road_status, finder=finder, table=table,
            )
            if approach is not None:
                return SupplyDecision(target=approach, target_agent=car,
                                      features=[], candidate_features=[])
            queue.pop(0)
        # **迎える順番が空のときも待ち伏せます。**
        #
        # ここが抜けていました。設計（`design_day` / `design_cover` /
        # 周回ソルバなど）は巡回車の順路だけを作り、補給車の順番は
        # 作りません。つまり `supply_route` が空なので上の while には
        # 1度も入らず、そのまま貪欲に落ちていました。**実戦で選ばれる
        # 案のほとんどがこれ**なので、待ち伏せが働く機会がありません。
        #
        # 空のときは、全部の巡回車について待ち伏せ先を出し、
        # **いちばん多く入る車**（＝いちばん空の車）を選びます。
        if SUPPLY_INTERCEPT and self.routes:
            limit = self.problem.fuel_limit
            # **困っている車がいるなら、待ち伏せより救援を先にします。**
            #
            # 手数2000・15試合で待ち伏せを入れると順位合計は163→151に
            # なりますが、**日別が1495→1491に落ちます**。第1791437153では
            # 系列を3つ落として8位→10位でした。日別は玉数より上位なので、
            # ここは放っておけません。
            #
            # 理由は、補給車が待ち伏せに専念すると**燃料が切れた別の車を
            # 助けに行かなくなる**ことです。助けられなかった車は、その日の
            # 残りを立ったまま過ごし、受け持ちの系列を落とします。
            #
            # そこで「次の店にも届かない車」が1台でもいれば、待ち伏せを
            # やめて貪欲（`STUCK_FIRST` つき＝困っている車から迎える）に
            # 任せます。全車が自力で動けている日だけ待ち伏せます。
            if SUPPLY_INTERCEPT_RESCUE_FIRST:
                for car, line in self.routes.items():
                    if not line or car >= len(fuels):
                        continue
                    here = finder.table(positions[car], road_status)
                    need = here.fuel.get(line[0])
                    if need is not None and need > fuels[car]:
                        # この車は次の店に届きません。救援が先です。
                        return self._fallback.choose_supply_target(**kwargs)
            pick = None
            for car, line in self.routes.items():
                if not line or car >= len(fuels):
                    continue
                room = limit - fuels[car]
                if room <= SUPPLY_SKIP_ROOM:
                    continue
                spot = self._intercept(car, position, table, remaining,
                                       fuels, positions, finder, road_status)
                if spot is None:
                    continue
                if pick is None or room > pick[0]:
                    pick = (room, spot, car)
            if pick is not None:
                return SupplyDecision(target=pick[1], target_agent=pick[2],
                                      features=[], candidate_features=[])
        return self._fallback.choose_supply_target(**kwargs)


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
        # **設計どおりに歩くときは、設計と同じ見方（歩数最短）で経路を
        # 引きます。**
        #
        # 設計（`hexa/allocate.py`）は `route_cost` で費用を測りますが、
        # これは**歩数最短**の経路です。いっぽう実行はここで
        # `fuel_weight` をかけた「燃料を考えた経路」を引いていました。
        # 準決勝32x32の1日目で測ると、同じ6店を回るのに
        #
        #   歩数最短（設計の見方）: 歩212 燃料205
        #   燃料こみ（実行の見方）: 歩**223** 燃料190
        #
        # と**5%ずれます**。設計が「96歩で入る」と言った順路が実際には
        # 101歩かかり、**最後の店が落ちます**。1台1日1軒なら6台6日で
        # 36玉の損です。設計は燃料の足りることも確かめてあるので、
        # ここは設計と同じ経路を歩きます。
        weight = (0 if PLANNED_USES_SHORTEST
                  else fuel_weight(remaining, fuel))
        table = finder.table(position, road_status, weight)
        # 味方が向かっている系列は、そちらに任せる。
        mates = set()
        for index, target in enumerate(teammate_targets or []):
            if index != agent_index and target is not None:
                brand = self.spot_brand.get(target)
                if brand is not None:
                    mates.add(brand)

        best = None
        # **今日は届かない店**の控え。全部外れたら、ここへ近づきます。
        far = None
        for spot_position in self.spot_positions:
            if spot_position in acquired_today or stock.get(spot_position, 0) <= 0:
                continue
            if not table.reachable(spot_position):
                continue
            steps = table.steps[spot_position]
            need = table.fuel[spot_position]
            if steps > remaining or need > fuel:
                if APPROACH_TOMORROW:
                    brand = self.spot_brand[spot_position]
                    key = (0 if brand not in acquired_brands else 1,
                           0 if brand not in brands_today else 1, steps)
                    if far is None or key < far[0]:
                        far = (key, spot_position)
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
        if target is None and far is not None:
            # **立ち止まらずに、明日の店へ近づきます。** 理由は
            # `APPROACH_TOMORROW` に。
            target = _approach_spot(
                remaining=remaining, position=position, goal=far[1],
                fuel=fuel, road_status=road_status, finder=finder,
                table=table)
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
    # **補給車を置く店。** 磨きを越えて残す必要があります。
    #
    # 磨きの採点は `PlannedRoutePolicy(problem, routes, order)` を
    # 作り直すので、`park` を渡していませんでした。つまり
    # **ハブや夜合流の案は、1手磨いた瞬間に置き場所が消えます**。
    # 準決勝32x32で、夜合流の設計を入れてもハブを入れても玉が
    # 1つも動かなかった（240のまま）のはこれが理由です。
    park: int | None = None
    # 夜に補給車を置くマス（補給車の番号 → マス）。
    night_park: dict | None = None
    # 夜に帰らせる巡回車（車の番号 → マス）。
    home: dict | None = None
    # 時刻つきの待ち合わせ（歩, セル, 車）の列。
    supply_plan: list | None = None


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

def _short_limit(problem, kinds) -> int:
    """何台まで燃料不足を許して玉を優先するか。

    **一律1台です。**

    以前は「巡回3台以下なら1台、4台以上は0台」と分けていましたが、
    測り直すと**4台以上でも1台まで許すほうが良い**と分かりました。

      1320（巡回4台）: 0台で玉358 → 1台で玉363
      1233（巡回4台）: 0台で玉187 → 1台で玉190
      1236（巡回3台）: 0台で玉134 → 1台で玉137

    以前「4台以上で日別が落ちる」と判断したのは乱数の揺れでした。
    2台まで許すと今度は落ちます（1233で184、1236で136）ので、
    1台がちょうどよい線です。

    1台の不足は補給で間に合うことが多く、それより玉を取りに行く
    ほうが得だということです。
    """

    return FUEL_SHORT_LIMIT


def _short_limit_for_day(problem, kinds, steps_ahead: int,
                         next_steps: int = 0) -> int:
    """その日に、何台まで燃料不足を許すか。

    **明日の必要量が燃料上限を超える日は、1台まで許します。**

    実戦1228の4日目で、翌日が151歩（必要燃料75）に対し上限120、
    開始燃料が[120, 17, 29, 26]という状態でした。どう計画しても
    何台かは足りません。そこで

        玉32（余力0） が 玉39（余力-1） に勝つ

    という逆転が起き、7玉を捨てていました。満タンにできない日に
    「1台も欠けないこと」を求めると、玉を取りに行けなくなります。
    """

    # **翌日が短い日なら、1台も欠けさせません。**
    #
    # 実戦1533の3日目（79歩）で、開始燃料が [17, 124, 124, 124]。
    # 1台が動けず、取れる上限が61玉から45玉に下がりました。
    #
    # 短い日は1台の欠けが重く効きます。歩数が多い日なら補給で
    # 取り返せますが、79歩では取り返す時間がありません。
    #
    # 逆に長い日は、少しの不足を許して玉を取りに行くほうが得です
    # （1228で玉32→39の逆転があった）。
    # **歩数の多い試合では、不足を広く許します。**
    #
    # 実戦1533（歩数平均101）と1535（125）では、許容を1台から3台に
    # 広げると玉数が伸びました（1533で249→259、1535で421→431）。
    # 歩数があれば、燃料が減っても補給で取り返せるためです。
    #
    # 逆に1233（歩数平均60）のような短い試合では、1台の欠けを
    # 取り返す時間がありません（許容3台で93%→87%に悪化）。
    base = _short_limit(problem, kinds)
    if problem is not None and problem.day_steps:
        average = sum(problem.day_steps) // max(1, len(problem.day_steps))
        if average >= LONG_MATCH_STEPS:
            return LONG_MATCH_SHORT_LIMIT
    return base


# 明日動けない巡回車がこの台数までなら、玉数を優先します。
# これを超えると、余力のほうを優先して身を守ります。
#
# 実測（4試合・1位の成績に対して）:
#   0台: 1233=195玉 / 1234=169玉（1位超） / 1235=598玉（1位超）
#   1台: 1233=179玉 / 1234=169玉（1位超）
#
# **0台**、つまり「1台も欠けないときだけ玉を優先する」が最良でした。
# 1台でも足りない日は、翌日の系列を落とす危険があるということです。
FUEL_SHORT_LIMIT = 1

# 1日の平均歩数がこれ以上なら「歩数の多い試合」とみなします。
LONG_MATCH_STEPS = 90
# そのときに許す燃料不足の台数。
LONG_MATCH_SHORT_LIMIT = 3

# 燃料の不足を「台数」ではなく「深刻さ」で数えるか。
#
# **実測で日別が落ちたので無効にしています。**
#
# 玉数は伸びます（1233で87%→94%）が、**1219で日別が138→136**と
# 落ちました。日別は玉数より重い得点なので採用できません。
# 深刻さで数えると、少しの不足を許しすぎて翌日に系列を落とします。
WEIGHTED_SHORTFALL = False

# **盤面の燃料の逼迫度で、数え方を切り替えます。**
#
# 満タン1杯が1日の歩数をまかなえない盤面では、「何台足りないか」で
# はなく「**どれだけ足りないか**」で見るべきです。3台しかない盤面で
# 台数で数えると 0/1/2/3 の4段階しかなく、乱暴すぎます。
#
# 本選1回戦の実問題6つ（16x16・燃料40・平均39.2歩 → 比 1.02）を
# 渋滞つきで通すと
#
#   台数で数える   : 日別212 玉265
#   深刻さで数える : 日別**218** 玉**273**
#
# 過去問29試合を比のしきい値で分けると、符号がきれいに変わります
# （磨きつき・手数2000・順位合計）:
#
#   | | 試合 | 台数 | 深刻さ |
#   |---|---|---|---|
#   | 逼迫（比 ≤ 1.1） |  7 | 92 | **89** |
#   | 余裕（比 > 1.1） | 22 | **262** | 281 |
#
# **過去問（練習場）と本選でパラメータが違う**ので、盤面サイズでは
# なく**その盤面の数字**で決めます。
WEIGHTED_WHEN_TIGHT = True
# 満タン ÷ 1日の平均歩数 がこれ以下なら「逼迫」とみなします。
TIGHT_FUEL_RATIO = 1.1
# **巡回車がこれより多い盤面では、深刻さでは数えません。**
#
# 台数で数えるのが乱暴なのは「段階が少ないから」です。巡回車が
# 3台なら 0/1/2/3 の4段階しかありませんが、6台なら7段階あり、
# それだけで足ります。実測でも、生成した逼迫マップ4つ
# （24x24は巡回4台・32x32は巡回6台）では符号が逆でした
# （手数600・渋滞つき）:
#
#   深刻さで数える : 日別427/430 玉745
#   台数で数える   : 日別**430/430** 玉**785**
#
# いっぽう本選1回戦（16x16・巡回3台）では深刻さが
# 日別218/玉273 対 212/265 で勝っています。
# 盤面の幅と台数は 16→4台 / 24→5台 / 32→7台 と連動しているので、
# 「巡回3台以下＝16x16」とほぼ同じ意味になります。
TIGHT_MAX_TOURING = 3
# 逼迫しているときの許容（深刻さの合計）。実測で3が最良でした
# （本選6試合で 上限1:日別217 / 2:218 / **3:218玉273** / 4:217 / 6:216）。
TIGHT_WEIGHTED_LIMIT = 3

# **「明日ぶんの燃料が足りている」と見る基準。燃料1を何歩ぶんと
# 見るか。**
#
# 平地は燃料1で2歩なので 2.0 が「全部平地」の見立てです。山地は
# 燃料2で3歩（1.5歩）、道路は燃料2で1〜4歩（0.5〜2歩）なので、
# 道路や山地を通る日は 2.0 では甘すぎます。小さくすると「足りて
# いる」の基準が上がり、燃料を残す案が通りやすくなります。
SHORTFALL_STEPS_PER_FUEL = 2.0


def few_touring(kinds) -> bool:
    """**巡回車が少ない盤面か**（`TIGHT_MAX_TOURING` 台以下）。

    台数が少ないほど、1台を失う損が大きくなります。3台なら1台で
    3分の1、6台なら6分の1です。「不足を深刻さで数える」ことと
    「着いた先で動けなくなる店を避ける」ことは、どちらも**少ない
    台数のときだけ**得でした。実測は `TIGHT_MAX_TOURING` と
    `KEEP_FUEL_RATIO` に。
    """

    if not kinds:
        return False
    return sum(1 for kind in kinds
               if kind == AgentKind.TOURING) <= TIGHT_MAX_TOURING


_TIGHT_CACHE: dict[tuple, bool] = {}


def fuel_is_tight(problem) -> bool:
    """**満タン1杯が1日の歩数をまかなえない盤面か。**

    理由は `WEIGHTED_WHEN_TIGHT` に。1日に何十万回も呼ばれるので
    覚えておきます。
    """

    if problem is None or not problem.day_steps:
        return False
    token = (problem.fuel_limit, problem.day_steps)
    got = _TIGHT_CACHE.get(token)
    if got is None:
        average = sum(problem.day_steps) / len(problem.day_steps)
        got = problem.fuel_limit <= average * TIGHT_FUEL_RATIO
        if len(_TIGHT_CACHE) > 64:
            _TIGHT_CACHE.clear()
        _TIGHT_CACHE[token] = got
    return got
# 深刻さで数えるときの許容値。
# 実測（1位の玉数に対する率）:
#   2: 1233=87% / 1228=86%
#   4: 1233=88% / 1228=90%
#   **6: 1233=94% / 1228=89%**
#   8: 1233=92% / 1228=89%
WEIGHTED_LIMIT = 6

# 通信が切れたときに、状態の取得を何回まで試すか。
# 1日は60秒あるので、20回（およそ30秒）粘っても回答は間に合います。
NETWORK_RETRIES = 20

# 燃料がこの割合を切った巡回車は、補給の優先順位を上げます。0 で無効。
CRITICAL_FUEL = 0.0


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


# 明日の歩数のうち、どれだけ以内に店がほしいか。0 で無効。
#
# **実測で1点も変わらなかったので無効にしています。** 計画はすでに
# 店の近くで1日を終えるため（最後に訪れるのが店なので）、
# わざわざ評価に入れる意味がありませんでした。
# 同じマスで終わる巡回車1台につき、評価をどれだけ下げるか。0 で無効。
#
# **実測で逆効果だったので 0 にしています。**
# 実戦1189では10日中7日で複数台が同じマスで終わっており、翌日の
# 出発点が重なると分散できないように見えました。ところが罰を入れると
# 実戦4問で 日別544→535・玉924→884 と悪化します。
# 同じマスで終わるのは「そこに良い店がある」結果であって、
# 原因ではなかったということです。
OVERLAP_PENALTY = 0

# 補給車が燃料の少ない車から遠いときの罰。
#
# **実測で逆効果だったので 0 にしています。**
#
# 実戦1356の7日目で、補給車は満タンの車と同じマスにいて、空の3台
# からは50〜55歩離れていました。マス535なら合計47歩で済みます。
# そこで「近くで終える」計画を選ぶようにしたところ、
#
#   罰0: 玉327 / 罰1: 玉307 / 罰2: 玉302
#
# と**悪化**しました。補給車の終了位置を縛ると、その日の補給が
# 回らなくなります。翌日の都合より、今日の補給のほうが重いのです。
SUPPLY_HOME_WEIGHT = 0
# 燃料がこの割合を切った巡回車を「近くにいてほしい相手」とみなす。
SUPPLY_HOME_RATIO = 0.4

POSITION_REACH_RATIO = 0.0

_NEAR_CACHE: dict[tuple[int, int], frozenset] = {}


def _near_spot_cells(problem: Problem, finder, limit: int) -> frozenset:
    """どれかの店から limit 歩以内にあるマスの集合。"""

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


# 明日すぐ行ける店を数えるときに、明日の歩数のどれだけを見込むか。
#
# 以前これを 0.5 で入れたら 11玉悪くなり（実戦1685・1687で
# 玉372→361）、0 で止めていました。「店の多いあたりに車が寄って
# しまう」と見立てていましたが、**数え方に2つ誤りがありました**。
#
#  1. **今日の売れ残り在庫**を見ていた。在庫は**毎日満タンに戻る**
#     ので（`simulate_day` が毎日作り直します）、これは明日の話に
#     なっていません。今日取り切った良い店ほど0と数えられ、
#     「今日あまり取らなかったあたり」へ車を誘導していました。
#  2. 1つの店に2台が届くと **2** と数えていた。在庫1の店に2台
#     寄っても明日取れるのは1玉です。寄り集まるほど得に見えるので、
#     見立てどおり車が固まります。
#
# 直した数え方は「明日の朝に取れる玉の数」です。店ごとに
# min(在庫, 届く台数) を足します。
#
# **それでも実測では入りませんでした（0 で止めています）。**
#
# 16試合を記録された渋滞つきで再現（手数3000）:
#
#   使わない      : 玉3998 / 日別1683
#   直して0.25で使う: 玉3943 / 日別1685
#
# 玉が55落ちて、日別は2しか増えません。理由は置き場所です。この項は
# **余力（終了時の燃料合計）より上**にあるので、玉が同点の案のうち
# 「燃料を残す案」ではなく「店に近い案」を選ぶようになります。
# 燃料を残すほうが翌日の歩数に直結するぶん強い、ということです。
#
# 数え方の誤りは直したまま残してあります（上の2点）。順位を
# 余力より下に置けば効く余地はありますが、今回は測っていません。
TOMORROW_REACH_RATIO = 0.0
# **明日の到達を、歩数だけでなく燃料でも絞るか。**
#
# 本選1回戦の実問題を渋滞つきで通すと、2日目・3日目・5日目の
# 開始時に巡回車の最低燃料が **1・1・4** でした。燃料1では平地1回
# （2歩）しか動けないのに、`_tomorrow_reach` は歩数だけで数えて
# いたので「明日そこまで行ける」と点が付き、燃料を使い切る案が
# 選ばれていました。厳密上限（燃料無視・渋滞あり）と比べると、
# 第1試合の2日目は日別4に対し上限7で、**3系列を燃料で落として
# いました**。
TOMORROW_REACH_FUEL = True


def _tomorrow_reach(day_result, kinds, problem, finder, next_steps,
                    road_status=None) -> int:
    """終わった場所から、明日すぐ取りに行ける店の数の合計。

    **なぜ必要か**

    採点がここまで同点になる案は毎日たくさん出ます。以前は最後に
    燃料の合計だけを見ていたので、同点なら「どこで終わるか」は
    選ばれていませんでした。

    車は夜のあいだ店の上に留まるので、翌日の1歩目で1玉入ります
    （実戦のログでも2日目以降は全車が歩1で獲得しています）。
    その先どれだけ続けられるかは、終わった場所の周りに店が
    どれだけあるかで決まります。
    """

    if next_steps <= 0 or finder is None or problem is None:
        return 0
    if TOMORROW_REACH_RATIO <= 0:
        return 0
    limit = max(1, int(next_steps * TOMORROW_REACH_RATIO))
    roads = road_status if road_status is not None else {}
    cars = sum(1 for kind in kinds if kind == AgentKind.TOURING)
    # **補給車と同じマスで終えた車は、明日の0歩目で満タンになります**
    #（`simulator` の 1.4 は歩0でも走ります）。燃料で絞るときに
    # ここを忘れると、いちばん良い終わり方を罰してしまいます。
    parked = {day_result.positions[index]
              for index, kind in enumerate(kinds)
              if kind == AgentKind.SUPPLY}
    # 店ごとに「何台が届くか」を数えます。
    reach: dict[int, int] = {}
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        table = finder.table(day_result.positions[index], roads)
        # **燃料も見ます。** 歩数だけで数えていたので、燃料1で終えた
        # 車にも「明日そこまで行ける」と点を付けていました。燃料1では
        # 平地1回ぶん（2歩）しか動けません。
        fuel_left = (day_result.fuels[index] if TOMORROW_REACH_FUEL
                     else None)
        if fuel_left is not None and day_result.positions[index] in parked:
            fuel_left = None
        for spot in problem.spots:
            steps = table.steps.get(spot.position)
            if steps is None or steps > limit:
                continue
            if fuel_left is not None:
                need = table.fuel.get(spot.position)
                if need is not None and need > fuel_left:
                    continue
            reach[spot.position] = reach.get(spot.position, 0) + 1
    # **明日の朝に取れる玉の数**にします。
    #
    # 在庫は毎日満タンに戻るので、見るのは今日の売れ残りではなく
    # もとの在庫です。そして1つの店から出るのは在庫のぶんだけなので、
    # 届く台数が在庫を超えても増えません。こうしないと「寄り集まる
    # ほど得」に見えて、車が固まります。
    stocks = {spot.position: min(spot.stocks, cars)
              for spot in problem.spots}
    return sum(min(count, stocks.get(spot, 0))
               for spot, count in reach.items())


def _position_penalty(
    day_result: DayResult,
    kinds: tuple[AgentKind, ...],
    problem: Problem,
    finder,
    next_steps: int,
) -> int:
    """明日いい場所から始められない巡回車の数。既定では常に0です。"""

    if next_steps <= 0 or POSITION_REACH_RATIO <= 0 or finder is None:
        return 0
    limit = max(1, int(next_steps * POSITION_REACH_RATIO))
    near = _near_spot_cells(problem, finder, limit)
    far = 0
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        if day_result.positions[index] not in near:
            far += 1
    return far


def _fuel_shortfall_weighted(
    day_result: DayResult,
    kinds: tuple[AgentKind, ...],
    fuel_limit: int,
    steps_ahead: int,
) -> int:
    """明日どれだけ動けないか。**台数ではなく深刻さで数えます。**

    実戦1233の1日目で、玉35（1台不足）が玉40（2台不足）に勝って
    いました。許容が1台なので2台不足は弾かれます。

    ところが「不足」の中身はさまざまです。必要67に対し

      残り66の車 → ほぼ足りている
      残り 3の車 → ほとんど動けない

    これを同じ1台と数えるのは乱暴です。**足りない割合**で数えれば、
    少しの不足は許し、深刻な枯渇だけ避けられます。
    """

    if steps_ahead <= 0:
        return 0
    need = max(1, int(steps_ahead / SHORTFALL_STEPS_PER_FUEL))
    # 夜に補給車と同室の車は翌朝満タン（`_fuel_shortfall` と同じ理由）。
    parked = set()
    if SHORTFALL_COUNTS_PARKED:
        parked = {day_result.positions[index]
                  for index, kind in enumerate(kinds)
                  if kind == AgentKind.SUPPLY}
    score = 0
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        if parked and day_result.positions[index] in parked:
            continue
        left = day_result.fuels[index]
        if left >= need:
            continue
        # 0.0（ちょうど足りない）〜1.0（空）を4段階で数える
        gap = (need - left) / need
        score += int(gap * 4) + 1
    return score


def _fuel_shortfall(
    day_result: DayResult,
    kinds: tuple[AgentKind, ...],
    fuel_limit: int,
    steps_ahead: int,
) -> int:
    """明日以降に動けない度合い。小さいほど良い。

    **なぜ必要か**

    実戦で、3日目の開始時に燃料が [108, 108, 8, 108, 26, 3] まで枯れ、
    取り逃した5系列すべてが「燃料不足」でした。前日に使い切ったのが原因です。

    **数えるのは「足りない車の数」です。**

    「足りない量の合計」で数える案も試しましたが、**実測で大きく
    悪化しました**（32x32が93%→80%）。燃料を温存しすぎて、
    その日の玉を取りに行かなくなります。

    必要量を燃料上限で頭打ちにする案も試しました。24x24 は 88%→93% に
    上がりますが、32x32 が 93%→80% に落ちます。**元の形が最良**でした。

      そのまま    : 16x16 96% / 24x24 93% / 32x32 93% / 1067 92%
      量で数える  : 16x16 89% / 24x24 90% / 32x32 80% / 1067 81%
      上限で頭打ち: 16x16 96% / 24x24 86% / 32x32 80% / 1067 89%
    """

    if steps_ahead <= 0:
        return 0
    # 平地は2歩で燃料1なので、歩数の半分の燃料があれば全部動けます。
    need = max(1, int(steps_ahead / SHORTFALL_STEPS_PER_FUEL))
    # **補給車と同じマスで夜を越した車は、翌朝ほぼ無料で満タンです。**
    #
    # `_stranded_count` / `_tomorrow_coverage` / `_tomorrow_reach` /
    # `_immobile_count` の4つはこれを数えているのに、**ここだけ素の
    # 燃料を見ていました**。この桁は玉数より**上**（鍵の5桁目に対して
    # 玉数は7桁目）なので、**夜に合流する案は上の桁で罰されて、下の桁
    # でしか報われていませんでした**。
    #
    # つまり今日「夜合流は15試合で190（基準163）」と測った数字は、
    # 鍵が夜合流を罰している状態で取ったものです。
    parked = set()
    if SHORTFALL_COUNTS_PARKED:
        parked = {day_result.positions[index]
                  for index, kind in enumerate(kinds)
                  if kind == AgentKind.SUPPLY}
    short = 0
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        if parked and day_result.positions[index] in parked:
            # 翌朝は満タンなので、足りない車には数えません。
            continue
        if day_result.fuels[index] < need:
            short += 1
    return short


# **燃料不足を数えるとき、夜に補給車と同室の車を満タン扱いするか。**
# 理由は `_fuel_shortfall` の中に。
SHORTFALL_COUNTS_PARKED = False


# 立ち往生を 0 にすると、この仕組みを止められます（実測用）。
STRAND_WEIGHT = 1

# 立ち往生を数えるときに見込む「補給車が迎えに行ける距離」。
# 翌日の歩数のうち、これだけの割合までなら迎えに行けるとみなします。
STRAND_RESCUE_RATIO = 0.25


# **明日どの系列にも手が届かなくなることの重さ。0 で止めます。**
#
# 実戦1960（24x24）の3日目は、足りない系列5と12の店が**50〜51歩先**に
# あり、その日の48歩ではどう動いても届きませんでした。その日のうちに
# 直すのは不可能で、**前日の終わり方**が原因です。
#
# 日別累計は玉数より上位の得点なので、「明日1系列落とす終わり方」は
# 「今日2〜3玉少ない終わり方」よりずっと悪い。だから採点の位置も
# 玉数より上、立ち往生（`_stranded_count`）と同じ並びに置きます。
BRAND_REACH_WEIGHT = 1

# **明日の歩数のうち、どこまでを「届く」と見なすか。**
#
# 明日の渋滞は分かりません。今日の状態を当てにすると甘く出ます。
# 実戦1960の2日目の終わりから見ると、足りなくなった系列5・12は
# **今日の渋滞では42歩・40歩**（その日は48歩）で届くように見え、
# **明日の実際の渋滞では51歩・50歩**で届きませんでした。
#
# 渋滞が悪化するぶんの余裕を持たせます。
BRAND_REACH_MARGIN = 0.8

# **採点の並びのうち、玉数が何番目か。**
#
# 項目を足すたびに番号がずれます。以前も1つ足したときにずらし忘れ、
# ビューアに「玉0」と出ていました。名前で持ちます。
KEY_NOODLES = 6


def _unreachable_brands(
    day_result: DayResult,
    kinds: tuple[AgentKind, ...],
    problem: Problem | None,
    finder,
    next_steps: int,
    road_status: dict | None = None,
) -> int:
    """明日、手が届かない系列の数を (きつい, ゆるい) で返す。

    **きつい** : 明日の歩数いっぱい使っても届かない系列。確実に
      1つ落とします。これは**玉数より上**に置きます。
    **ゆるい** : 明日の歩数の `BRAND_REACH_MARGIN` 倍までに届かない
      系列。渋滞が悪化すると落ちる危うい形です。こちらは**玉数より
      下**に置くので、玉を1つも減らしません（同点のときだけ効きます）。

    届く範囲は「明日の歩数」と「いま残っている燃料で歩ける数」の
    小さいほうです（1歩に燃料は最大でも2分の1しか要らないので、
    燃料×2が歩ける上限）。道路の状態は今日のものを使います。交通量は
    前2日の滞在から決まるので、今日混んでいる道は明日も混んでいる
    可能性が高く、いちばん手近な見立てになります。

    **甘く見ます。** 燃料も補給も渋滞の変化も良いほうに見て、それでも
    届かない系列だけを数えます。見込み違いで玉を捨てないためです。
    """

    if next_steps <= 0 or finder is None or problem is None:
        return (0, 0)
    roads = road_status if road_status is not None else {}
    spots_by_brand: dict[int, list[int]] = {}
    for spot in problem.spots:
        spots_by_brand.setdefault(spot.brand, []).append(spot.position)
    # 系列ごとに、いちばん近い車からの歩数
    nearest: dict[int, int] = {}
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        can_walk = 2 * day_result.fuels[index]
        if can_walk <= 0:
            continue
        table = finder.table(day_result.positions[index], roads)
        for brand, spots in spots_by_brand.items():
            for spot in spots:
                steps = table.steps.get(spot)
                if steps is None or steps > can_walk:
                    continue
                if brand not in nearest or steps < nearest[brand]:
                    nearest[brand] = steps
    hard = 0
    soft = 0
    room = max(1, int(next_steps * BRAND_REACH_MARGIN))
    for brand in spots_by_brand:
        gap = nearest.get(brand)
        if gap is None or gap > next_steps:
            hard += 1
            soft += 1
        elif gap > room:
            soft += 1
    return (hard, soft)


# **明日の系列を何台で回れるかを見積もるかどうか。0 で止めます。**
#
# 実戦1981（16x16・13系列）は、日別累計75で8位でした。1位は満点の78。
# 落としたのは**5日目だけ**（系列5・6・10）で、その日は36歩。
# 原因は前日の終わり方です。巡回3台が **29・12・13**、つまり
# 右上の2x2のなかに固まって終わっていました。
#
# `_unreachable_brands` では捕まりません。足りない系列の店は
# 26歩・31歩先で、36歩なら**1台だけなら届く**からです。届かないのは
# 「13系列を3台で**同時に**回ること」でした。
#
# その日を90秒・14万手かけて解き直しても10系列止まりでした。探索の
# 不足ではありません。出発点を散らすと同じ36歩で13系列に届きます。
#
#   出発 29,12,188,13（実戦）  → 日別10
#   出発 12,115,178,188        → 日別13
#
# そこで、終わった場所から**明日を1台ずつ順に拾わせて**、何系列まで
# 回れるかを数えます。固まった終わり方はここで低く出ます。
BRAND_COVER_WEIGHT = 1

# **1系列ぶんの見込みを、玉何個ぶんと見るか。**
#
# 見積りをそのまま玉数より上に置くと（0 がその形）、系列の見込みを
# 1つ上げるために玉を**何個でも**捨てます。日別累計は玉数より上位の
# 得点なので筋は通っていますが、見積りが当たらない盤面では損だけが
# 残ります。小さいベンチの実測（1日15秒・記録された渋滞つき）:
#
#   第1703 (32x32) 日別225=満点 / 玉454  ← 入れない 225 / 433  **+21**
#   第1958 (32x32) 日別**198**=満点 / 玉440  ← 入れない 196 / 444
#   第1981 (16x16) 日別**78**=満点 / 玉181  ← 入れない 75 / 181
#   第1891 (24x24) 日別 98=満点 / 玉246  ← 入れない  98 / 249
#   第1915 (24x24) 日別180=満点 / 玉341  ← 入れない 180 / 347
#   第1960 (24x24) 日別138（1位140）/ 玉204  ← 入れない 138 / **217**
#
# 満点の日別が取れる盤面では玉の損もわずかで、1958と1981では
# **満点に届きました**。一方1960は日別が動かず玉だけ13減りました。
#
# そこで「玉3個ぶんまでなら払う」という上限を試しましたが、
# **測ると逆に悪くなりました**。第1703（32x32・巡回6+補給1）:
#
#   上限なし（辞書順）: 日別**225（満点）** / 玉**463** → 11位/30
#   玉3個ぶんまで      : 日別 222        / 玉 446  → 17位/30
#
# 上限を付けると満点の日別を取りこぼし、玉も増えませんでした。
# 日別累計は玉数より上位の得点なので、**払うべきときは払う**のが
# 正しい形です。上限は 0（辞書順）にしています。
BRAND_COVER_BONUS = 0

# **明日の歩数のうち、どこまでを見積りに使うか。**
#
# **1.0（そのまま）が実測で最良です。** 厳しくすると悪化します。
#
# きっかけは第1981を本番のループで通したときでした。4日目が
# [210,157,15] で終わり、5日目（36歩）は12系列。その日を42,406手
# かけて解き直しても12系列止まりなので本当に届かないのに、見積りは
# 12と出ていて、13系列取れる終わり方と同じ値でした。歩数に0.9倍を
# 掛けると分かれます（[210,157,15]→9 / [188,77,242]→11 /
# [12,115,178]→12 / 実戦の[29,12,13]→7）。
#
# ところが**過去問で測ると逆でした**（16x16の11試合＋32x32の4試合・
# 1日15秒・補給1台）:
#
#   余裕なし1.0倍 : 日別**1524** / 玉**3927**
#   余裕 0.9倍    : 日別 1523  / 玉 3817   ← 110玉も減る
#
#   第1703 (32x32) 玉458 → 402（日別も225→224）
#   第1958 (32x32) 玉504 → 472
#
# 厳しくすると見積りが**満ちることがなくなり**、ずっと「散れ」と
# 要求し続けて玉を捨てます。上限に届いたあとは同じ見込みのなかで
# 玉数を競う、という形が崩れるためです。1例の盤面に合わせて全体を
# 悪くしていました。1.0 に戻してあります。
BRAND_COVER_MARGIN = 1.0
# **鍵の2番目を「今日の系列数＋明日回れる系列数×この重み」にするか。**
# 0 で今日だけ（もとの形）。理由は `_candidate_key` の中に。
DAILY_PLUS_COVER = 0
# **燃料1でどれだけ歩けると見るか。**
#
# 平地なら燃料1で2歩ですが、山地は燃料2で3歩（1.5歩）、道路は
# 燃料2で1〜4歩（0.5〜2歩）です。2.0 は「全部平地」の見立てで、
# いちばん甘い数字です。甘く見ると今日の玉を捨てずに済む反面、
# 燃料を使い切る案が通ります。
COVER_STEPS_PER_FUEL = 2.0

# **見積りの覚え書き。** 同じ終わり方は何度も出てきます（磨きは経路の
# 途中だけを入れ替えるので、終わりの位置と燃料はよく一致します）。
# 1手ごとに巡回車のぶんだけ経路表を引き直すのは無駄なので、
# 「道路の状態・明日の歩数・巡回車の位置と燃料」で覚えておきます。
#
# 入力が同じなら答えも同じなので、結果は変わりません（乱数なし）。
_COVER_CACHE: dict[tuple, int] = {}
_COVER_CACHE_LIMIT = 40_000


def _tomorrow_coverage(
    day_result: DayResult,
    kinds: tuple[AgentKind, ...],
    problem: Problem | None,
    finder,
    next_steps: int,
    road_status: dict | None = None,
) -> int:
    """明日、終わった場所から**何系列まで回れるか**の見積り。

    各巡回車を順番に1回ずつ動かし（持ち回り）、まだ押さえていない
    系列のうちいちばん近い店へ向かわせます。歩数の残りが尽きた車は
    抜けます。押さえた系列の数を返します。

    **なぜ持ち回りか**

    1台に全部拾わせると、近くに固まった系列を1台が取り切る形になり、
    「固まって終わる」ことの不利が出ません。1台ずつ交代で拾うほうが
    実際の走り方に近く、散らばった終わり方が高く出ます。

    **甘く見ます。** 渋滞は今日のもの、燃料は満タン換算（1歩に燃料は
    2分の1までしか要らない）で見ます。見積りが厳しすぎると、今日の
    玉を無駄に捨てるからです。
    """

    if next_steps <= 0 or finder is None or problem is None:
        return 0
    if BRAND_COVER_WEIGHT <= 0:
        return 0
    roads = road_status if road_status is not None else {}
    # 1台につき「いまの場所・残り歩数」。燃料で歩ける上限も見ます。
    cars: list[tuple[int, int]] = []
    # **補給車と同じマスで終えた車は、明日の0歩目で満タンです**
    #（`simulator` の補給は歩0でも走ります）。燃料で歩ける上限を
    # かけるときにここを忘れると、いちばん良い終わり方を罰します。
    parked = {day_result.positions[index]
              for index, kind in enumerate(kinds)
              if kind == AgentKind.SUPPLY}
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        fuel = day_result.fuels[index]
        if day_result.positions[index] in parked:
            fuel = problem.fuel_limit
        room = min(int(next_steps * BRAND_COVER_MARGIN),
                   int(COVER_STEPS_PER_FUEL * fuel))
        if room <= 0:
            continue
        cars.append((day_result.positions[index], room))
    if not cars:
        return 0
    token = (finder.road_id(roads), next_steps, tuple(cars))
    remembered = _COVER_CACHE.get(token)
    if remembered is not None:
        return remembered
    spots = problem.spots
    taken: set[int] = set()
    alive = True
    while alive:
        alive = False
        for seat, (position, room) in enumerate(cars):
            if room <= 0:
                continue
            table = finder.table(position, roads)
            pick = None
            for spot in spots:
                if spot.brand in taken:
                    continue
                steps = table.steps.get(spot.position)
                if steps is None or steps > room:
                    continue
                if pick is None or steps < pick[0]:
                    pick = (steps, spot.position, spot.brand)
            if pick is None:
                cars[seat] = (position, 0)
                continue
            steps, where, brand = pick
            taken.add(brand)
            cars[seat] = (where, room - steps)
            alive = True
    if len(_COVER_CACHE) >= _COVER_CACHE_LIMIT:
        _COVER_CACHE.clear()
    _COVER_CACHE[token] = len(taken)
    return len(taken)


# **終わり方だけで決まる項目の覚え書き。**
#
# `_unreachable_brands` と `_stranded_count` が見るのは「終わりの位置と
# 燃料」だけです。磨きは順路の途中を入れ替えるので、終わり方が同じに
# なる手がたくさん出ます。第1958（32x32・7体）の実測で、採点は1手
# あたり0.85ms＝全体の11%を使っていました。
#
# 入力が同じなら答えも同じ（乱数なし）なので、結果は変わりません。
_REACH_CACHE: dict[tuple, tuple[int, int]] = {}
_STRAND_CACHE: dict[tuple, int] = {}
_SIDE_CACHE_LIMIT = 40_000


def _cached_unreachable_brands(day_result, kinds, problem, finder, next_steps,
                               road_status, touring_seats):
    """`_unreachable_brands` を、終わり方で覚えながら呼ぶ。"""

    if next_steps <= 0 or finder is None or problem is None:
        return (0, 0)
    roads = road_status if road_status is not None else {}
    # ここも全車ぶんで持ちます。見るのは巡回車だけですが、鍵を
    # 絞って速くなる幅より、取り違えの危険のほうが大きいからです。
    token = (finder.road_id(roads), next_steps, tuple(day_result.positions),
             tuple(day_result.fuels))
    got = _REACH_CACHE.get(token)
    if got is not None:
        return got
    got = _unreachable_brands(day_result, kinds, problem, finder, next_steps,
                              road_status)
    if len(_REACH_CACHE) >= _SIDE_CACHE_LIMIT:
        _REACH_CACHE.clear()
    _REACH_CACHE[token] = got
    return got


def _cached_stranded_count(day_result, kinds, problem, finder, steps_ahead,
                           touring_seats):
    """`_stranded_count` を、終わり方で覚えながら呼ぶ。

    **鍵は全車ぶんです。** 巡回車だけにしていたら、テストが
    取り違えを捕まえました。`_stranded_count` は「補給車と同じマスで
    終われば立ち往生にならない」と数えるので、**補給車の位置も
    答えを変えます**。巡回車が同じ場所・同じ燃料でも、補給車の
    居場所が違えば答えは違います。
    """

    if steps_ahead <= 0 or problem is None or finder is None:
        return 0
    token = (steps_ahead, tuple(day_result.positions),
             tuple(day_result.fuels))
    got = _STRAND_CACHE.get(token)
    if got is not None:
        return got
    got = _stranded_count(day_result, kinds, problem, finder, steps_ahead)
    if len(_STRAND_CACHE) >= _SIDE_CACHE_LIMIT:
        _STRAND_CACHE.clear()
    _STRAND_CACHE[token] = got
    return got


def _cover_rank(day_result, kinds, problem, finder, next_steps,
                road_status=None) -> int:
    """採点に並べる「明日回れる系列」の値。

    `BRAND_COVER_BONUS` が 0 なら見積りそのまま（玉数より上の辞書順で、
    玉を何個でも払う）。1以上なら**玉数に足した形**にするので、
    見込みを1つ上げるために払う玉はその数までに収まります。
    """

    cover = _tomorrow_coverage(day_result, kinds, problem, finder,
                               next_steps, road_status) * BRAND_COVER_WEIGHT
    if not BRAND_COVER_BONUS:
        return cover
    return len(day_result.acquisitions) + BRAND_COVER_BONUS * cover


def _stranded_count(
    day_result: DayResult,
    kinds: tuple[AgentKind, ...],
    problem: Problem | None,
    finder,
    steps_ahead: int,
) -> int:
    """**一歩も動けなくなった巡回車の数**。少ないほど良い。

    **なぜ「燃料不足の台数」と分けて数えるか**

    `_fuel_shortfall` は「明日ぶんの燃料が足りない台数」です。
    燃料7で明日50歩の車と、燃料0の車が、どちらも1台として
    同じ扱いになります。ところがこの2台はまったく違います。

    燃料が少しでもあれば、自力で補給車に近づけます。
    **0 だと自力では永久に動けません。** 補給車が来るまで、
    その車は毎日1玉も取れません。

    実戦1646（16x16・燃料60）のログでは、3号車が3日目に燃料2で
    止まり、**4日目と5日目は一歩も動いていません**（1歩も動かず
    34歩と42歩を消費）。試合終了まで復帰せず、玉数は78。
    1位は95でした。差17玉のほとんどがこの1台です。

    実戦1650の3日目でも、4号車が燃料7で始まり0歩で終わっています
    （50歩を無駄にし、その日の獲得は1玉）。

    `FUEL_SHORT_LIMIT = 1` は「1台までの燃料不足は許す」という
    設定です。これは**自力で動ける不足**を前提にしていました。
    立ち往生は1台でも許してはいけないので、別に数えます。

    **補給車と同じマスで終われば立ち往生になりません。**
    翌日の最初の反映フェーズで満タンになるためです。
    補給車がその日のうちに迎えに行ける距離なら、同じく数えません。
    """

    if steps_ahead <= 0 or problem is None or finder is None:
        return 0
    supply = [index for index, kind in enumerate(kinds)
              if kind == AgentKind.SUPPLY]
    stranded = 0
    reach = int(steps_ahead * STRAND_RESCUE_RATIO)
    spot_positions = [spot.position for spot in problem.spots]
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        position = day_result.positions[index]
        fuel = day_result.fuels[index]
        # そのマスを出るのに必要な燃料（移動元のセルで決まる / Q10・Q25）
        if fuel >= fuel_cost(problem.terrain_at(position)):
            # 動けるだけでは足りません。**店にも補給車にも届かない**
            # なら、自力では何もできません。燃料最優先の経路で測ります。
            table = finder.table(position, {}, weight=FUEL_WEIGHT_STEPS)
            need = min(
                (table.fuel[target]
                 for target in spot_positions
                 if target in table.fuel),
                default=None,
            )
            for other in supply:
                cost = table.fuel.get(day_result.positions[other])
                if cost is not None and (need is None or cost < need):
                    need = cost
            if need is None or fuel >= need:
                continue
        rescued = False
        for other in supply:
            if day_result.positions[other] == position:
                rescued = True
                break
            table = finder.table(day_result.positions[other], {})
            steps = table.steps.get(position)
            if steps is not None and steps <= reach:
                rescued = True
                break
        if not rescued:
            stranded += 1
    return stranded


IMMOBILE_TIEBREAK = True


# **その場を出られないまま夜を越す巡回車1台を、玉いくつの損と見るか。**
# 0 で止めます（いまは玉数の下の桁でだけ効かせています）。
#
# 準決勝32x32の3日目は、朝の燃料が **[80, 22, 0, 1, 69, 74]** でした。
# 2台が0と1で、**1歩も動けないまま夜明けを迎えています**。その日の
# 玉は21〜26で、ほかの日（31〜40）の3分の2しかありません。
#
# 原因は前の晩で、**その日の玉を1つ増やすために燃料を使い切った**
# ことです。鍵では「動けない車の少なさ」が玉数より**下**にあるので、
# 探索はその取引をいつでも選びます。ところが動けない車は翌日まるごと
# 損で、1台あたり玉6つぶん（1台1日の実績）に近い値段です。
#
# 玉数の桁から引く形にすれば、**その日の玉をいくつまで払うか**を
# この数で決められます。辞書順で上に置くのと違い、払いすぎません。
IMMOBILE_BALLS = 0


NIGHT_REFUEL_GAIN = True


# **夜に補給車と同じマスで越す車の「ただの満タン」を玉いくつと見るか。**
# 0 で止めます。
#
# `FUEL_BALL_RATE`（燃料をそのまま玉に換算する）は実測で負けました
# （32x32で玉230→225/217/220/217）。貯め込むほど今日の玉を捨てる
# からです。こちらは**貯め込みではなく位置の選択**で、同じマスに
# 何台いてもよく、巡回車の歩も補給車の歩も使いません。
#
# 準決勝32x32の厳密な上限:
#
#   実際の朝の燃料・補給なし  : 玉**243**（こちらは240で張り付き）
#   **毎朝満タン・補給なし**  : 玉**373**
#   1位の実績                 : 玉304
#
# **毎朝満タンなら日中の補給は1回も要りません**（燃料80は96歩を
# まかなう）。だから「夜に集まる」ことだけを玉の桁で評価します。
# 明日使い切れないぶんは数えません。
NIGHT_BALL_RATE = 0.0


def _night_balls(day_result, kinds, fuel_limit: int, steps_ahead: int) -> int:
    """夜合流でただで入る燃料を、玉いくつぶんとして返す。理由は上に。"""

    if NIGHT_BALL_RATE <= 0.0 or steps_ahead <= 0:
        return 0
    supply = {day_result.positions[index]
              for index, kind in enumerate(kinds)
              if kind == AgentKind.SUPPLY}
    if not supply:
        return 0
    # **明日使えるぶんだけ数えます**（貯め込みを得にしないため）。
    # 道路は1歩で燃料2なので、歩数と同じだけの燃料を使い切れます
    # （平地は2歩で燃料1なので歩数の半分）。1位は1移動あたりの燃料が
    # 2.0に近い走り方（＝道路づかい）をしているはずなので、歩数ぶん
    # まで数えます。
    room = max(1, steps_ahead)
    total = 0
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        if day_result.positions[index] not in supply:
            continue
        # 明日使えるぶんだけ数えます（貯め込みを得にしないため）。
        total += max(0, min(fuel_limit, room) - day_result.fuels[index])
    return int(NIGHT_BALL_RATE * total)


def _night_refuel_gain(day_result, kinds, fuel_limit: int) -> int:
    """**補給車と同じマスで夜を越した巡回車が、翌朝ただで入る燃料。**

    `simulator` の補給（1.4）は歩0の行動段階でも走るので、同じマスに
    いる巡回車は翌日1歩も動かないうちに満タンになります。巡回車を
    1歩も動かさず、補給車の歩も使わない、いちばん安い補給です。

    本選1回戦第1試合を渋滞つきで通すと、燃料が足りずに動けなかった歩が
    5日で588歩のうち**80歩（13.6%）**ありました。配った燃料は246、
    初期120で合計366に対し、全歩を平地で使うのに必要なのは294なので、
    **量は余っていて届く時刻だけが合っていません**。

    鍵のいちばん下の桁（燃料の合計）に足して、「今夜の残り」ではなく
    「明日の朝の残り」を見る形にします。上の桁は何も動きません。
    """

    if not NIGHT_REFUEL_GAIN:
        return 0
    supply = {day_result.positions[index]
              for index, kind in enumerate(kinds)
              if kind == AgentKind.SUPPLY}
    if not supply:
        return 0
    gain = 0
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        if day_result.positions[index] in supply:
            gain += max(0, fuel_limit - day_result.fuels[index])
    return gain


# **明日使える燃料を「玉いくつぶん」と見るか。** 0 で止めます。
#
# 準決勝32x32（第1791641357試合・巡回6台・燃料80・歩96）を
# 厳密な上限で切り分けると、こうなりました:
#
#   | 条件 | 玉の上限 |
#   |---|---|
#   | 手持ちの燃料だけ（補給なし） | **243** |
#   | 1台1日0.5回の補給 | 356 |
#   | 1台1日1回の補給 | 373 |
#   | 1位の実績 | 304 |
#   | こちらの実績 | 216（手元の台で230〜236） |
#
# **こちらは「補給なしの上限」に張り付いています。** 探索を10倍
# （手数2000→20000）にしても230→231で動きません。足りないのは
# 探索ではなく**燃料の配達**です。
#
# ところが鍵は玉数（7番目）を燃料（8番目以降）より上に置いて
# いるので、「今日の玉を1つ減らして満タンにする」案は絶対に
# 選ばれません。ハブ（補給車を店に置いて全車が寄る形）を候補に
# 入れても、玉数で負けて一度も採用されませんでした（玉230のまま）。
#
# そこで**燃料を玉に換算して玉数の桁に足します**。実測では
# 1移動に燃料1.28、1玉に約4移動なので、**玉1個は燃料5ほど**です。
# 明日使えないぶん（明日の歩数の半分より多い燃料）は数えません。
# 貯め込むほど得にならないようにするためです。
FUEL_BALL_RATE = 0.0


def _fuel_as_balls(day_result, kinds, steps_ahead: int) -> int:
    """明日使える燃料を、玉いくつぶんとして返す。理由は上に。"""

    if FUEL_BALL_RATE <= 0.0 or steps_ahead <= 0:
        return 0
    # 明日1日で使い切れる量まで（平地なら1歩に燃料0.5）。
    room = max(1, steps_ahead // 2)
    total = 0
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        total += min(day_result.fuels[index], room)
    return int(FUEL_BALL_RATE * total)


def _immobile_count(day_result, kinds, problem) -> int:
    """**そのマスを出られないまま夜を越す巡回車の数。**

    `_stranded_count` は「補給車が明日迎えに行ける距離なら立ち往生と
    数えない」という作りです。正しい見立てですが、迎えに行く朝のぶん
    （巡回車の歩と補給車の歩の両方）は必ず失われます。

    本選1回戦第1試合の2日目、2号車は**セル17で燃料1のまま37歩を
    1歩も動かず**終わりました（補給車は2マス隣から出発して盤面の
    反対側へ行き、迎えには来ていません）。前日の計画が燃料1を
    残したのが始まりです。

    そこで**玉数より下の同点決着**として入れます。玉は1つも減らさず、
    同じ玉数なら「燃料2を残す」案を選びます。
    """

    # **巡回車が少ない盤面でだけ効かせます**（`few_touring`）。
    # 台数が多いと1台の立ち往生は軽く、押し込むほうが得でした。
    # 生成12マップ（巡回4〜6台）で 日別1634→1632・玉3831→3824 と
    # 落ち、本選6試合（巡回3台）では 日別221→222・玉277→278 と
    # 上がります。
    if not IMMOBILE_TIEBREAK or problem is None or not few_touring(kinds):
        return 0
    parked = {day_result.positions[index]
              for index, kind in enumerate(kinds)
              if kind == AgentKind.SUPPLY}
    count = 0
    for index, kind in enumerate(kinds):
        if kind != AgentKind.TOURING:
            continue
        position = day_result.positions[index]
        if position in parked:
            continue
        if day_result.fuels[index] < fuel_cost(problem.terrain_at(position)):
            count += 1
    return count


def _supply_home_penalty(day_result, kinds, problem, finder, fuel_limit) -> int:
    """補給車が、燃料の少ない車から遠いほど大きくなる値。

    翌日すぐ補給に向かえる位置で終われば0に近づきます。
    """

    supply = [i for i, k in enumerate(kinds) if k == AgentKind.SUPPLY]
    if not supply:
        return 0
    low = [
        index for index, kind in enumerate(kinds)
        if kind == AgentKind.TOURING
        and day_result.fuels[index] < fuel_limit * SUPPLY_HOME_RATIO
    ]
    if not low:
        return 0
    try:
        table = finder.table(day_result.positions[supply[0]], {}, 0)
    except Exception:
        return 0
    total = 0
    for index in low:
        spot = day_result.positions[index]
        steps = table.steps.get(spot)
        if steps is None:
            total += problem.width + problem.height
        else:
            total += steps
    # 1台あたり何歩離れているかを、評価の単位に落とす。
    span = max(1, (problem.width + problem.height) // 4)
    return (total // max(1, len(low))) // span * SUPPLY_HOME_WEIGHT


def _candidate_key(
    day_result: DayResult,
    acquired_brands: set[int],
    fuel_limit: int,
    kinds: tuple[AgentKind, ...] = (),
    steps_ahead: int = 0,
    problem: Problem | None = None,
    finder=None,
    next_steps: int = 0,
    road_status: dict | None = None,
) -> tuple[int, ...]:
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
    # 覚え書きの鍵に使う巡回車の席。
    touring_seats = tuple(index for index, kind in enumerate(kinds)
                          if kind == AgentKind.TOURING)
    # **燃料が逼迫した盤面では「深刻さ」で数えます**
    #（`WEIGHTED_WHEN_TIGHT` に実測つきの理由）。
    weighted = WEIGHTED_SHORTFALL or (
        WEIGHTED_WHEN_TIGHT
        and len(touring_seats) <= TIGHT_MAX_TOURING
        and fuel_is_tight(problem))
    short = (
        _fuel_shortfall_weighted(day_result, kinds, fuel_limit, steps_ahead)
        if weighted
        else _fuel_shortfall(day_result, kinds, fuel_limit, steps_ahead)
    )
    if problem is not None and finder is not None and next_steps > 0:
        short += _position_penalty(day_result, kinds, problem, finder, next_steps)
    # **補給車が、燃料の少ない車の近くで終わっているか。**
    #
    # 実戦1356の7日目で、補給車は満タンの車と同じマスにいて、
    # 空の3台からは50〜55歩離れていました。172歩のうち最初の
    # 4割を補給待ちに費やしています。
    # マス535なら3台への合計47歩で済みました（実際は156歩）。
    #
    # 翌日の冒頭を無駄にしないよう、近くで終える計画を選びます。
    if SUPPLY_HOME_WEIGHT and problem is not None and finder is not None:
        short += _supply_home_penalty(
            day_result, kinds, problem, finder, fuel_limit
        )

    # **同じマスで終わる巡回車を数えます。**
    #
    # 実戦1189では10日中7日で複数台が同じマスで終わっていました
    # （3日目は3台とも同じ）。翌日の出発点が重なると分散できず、
    # その日の獲得が落ちます（実戦26玉に対し、散っていれば35玉）。
    if OVERLAP_PENALTY and kinds:
        seen: set[int] = set()
        for index, kind in enumerate(kinds):
            if kind != AgentKind.TOURING:
                continue
            spot = day_result.positions[index]
            if spot in seen:
                short += OVERLAP_PENALTY
            seen.add(spot)
    stranded = (
        _cached_stranded_count(day_result, kinds, problem, finder,
                               steps_ahead, touring_seats)
        if STRAND_WEIGHT else 0
    )
    # **明日どの車からも手が届かない系列の数。**
    #
    # 実戦1960（24x24）の3日目は、足りない系列の店が50〜51歩先に
    # あり、その日の48歩ではどう動いても届きませんでした。その日の
    # うちに直すのは不可能で、原因は**前日の終わり方**です。
    hard_miss, soft_miss = (
        _cached_unreachable_brands(day_result, kinds, problem, finder,
                                   next_steps, road_status, touring_seats)
        if BRAND_REACH_WEIGHT else (0, 0)
    )
    # **今日の系列だけでなく、明日回れる系列も足して見ます。**
    #
    # 採点は「日別累計＝毎日の系列数の合計」です。ところが鍵の
    # 2番目は**今日の系列数**だけなので、今日1系列増やすために
    # 明日3系列落とす計画が通ります。本選第1試合の1日目を手数6000で
    # 解くと、日別8の代わりに **4号車を燃料2・3号車を燃料3** まで
    # 使い切る案が選ばれ、2日目は日別5でした（手数300で解いた
    # ゆるい案のほうが通算では上）。**探すほど悪くなる**形です。
    #
    # `_tomorrow_coverage` の見積りは、部分集合DPで出した厳密な上限と
    # 8/8・6/6・6/6 とほぼ一致します（`an/ceil2.py`）。信頼できるので、
    # 2日ぶんの和にして転がします。0 で止めます。
    cover_value = (
        _tomorrow_coverage(day_result, kinds, problem, finder, next_steps,
                           road_status)
        if DAILY_PLUS_COVER else 0
    )
    return (
        new_brands,
        len(day_brands) + DAILY_PLUS_COVER * cover_value,
        # **明日1系列を落とす終わり方は、今日の玉より重く見ます。**
        #
        # 日別累計は玉数より上位の得点です。実戦1958では1位から12位
        # までの全員が日別満点（198）で、こちらは197。玉448でも、
        # 玉442のチームより下に置かれました。1系列は玉では償えません。
        #
        # 甘く見積もっているので（燃料も渋滞も良いほうに見る）、
        # ここに挙がるのは本当に届かない系列だけです。
        -hard_miss * BRAND_REACH_WEIGHT,
        # **一歩も動けない車を作らないことは、玉数より重く見ます。**
        #
        # 立ち往生した車は、その日だけでなく**試合の終わりまで**
        # 1玉も取れません。今日の玉を数玉あきらめても割に合います。
        # 系列（上の2つ）より下に置くのは、系列が公式の上位条件だからです。
        -stranded * STRAND_WEIGHT,
        # **燃料の余力は「足りないときだけ」玉数より優先します。**
        #
        # 以前は常に玉数より前に置いていました。そのため
        # 「余力-1で玉36」が「余力-3で玉43」に勝ってしまい、
        # 実戦1233では5日間で27玉を取り逃していました。
        #
        # かといって常に後ろへ下げると、燃料が枯れて翌日に系列を
        # 落とします（1234で日別104→102）。日別は玉数より重い得点
        # なので、それは許容できません。
        #
        # そこで**不足が FUEL_SHORT_LIMIT 台までなら玉数を優先**し、
        # それを超えたら余力を優先します。少しの不足は補給で間に合い、
        # 多くの車が動けない日だけ身を守る、という形です。
        0 if short <= (
            (TIGHT_WEIGHTED_LIMIT if not WEIGHTED_SHORTFALL
             else WEIGHTED_LIMIT) if weighted
            else _short_limit_for_day(problem, kinds, steps_ahead, next_steps)
        ) else -short,
        # **明日、何系列まで回れる終わり方か。**
        #
        # 実戦1981（16x16・13系列）の4日目は、巡回3台が **29・12・13**、
        # 右上の2x2のなかに固まって終わりました。翌日は36歩しかなく、
        # 13系列のうち3つ（5・6・10）を落として日別75で8位。1位は満点
        # の78でした。その日を90秒・14万手かけて解き直しても10系列
        # 止まりで、探索の不足ではありません。出発点を散らすと同じ
        # 36歩で13系列に届きます（12,115,178 → 13系列）。
        #
        # **玉数より上に置きます。** 日別累計は玉数より上位の得点で、
        # 1系列は玉では償えません。1981でも玉182のチームが日別73で
        # 最下位、玉181・日別75のこちらが8位でした。
        #
        # **ただし払う上限を決めます。** `BRAND_COVER_BONUS` が
        # 「1系列ぶんの見込みは玉何個ぶんか」です。玉数に足した形で
        # 並べるので、見込みを1つ上げるのに払う玉はその数までです。
        # 0 にすると辞書順（玉を何個でも払う）に戻ります。
        _cover_rank(day_result, kinds, problem, finder, next_steps,
                    road_status),
        # **玉数。** 上に置くか下に置くかで `IMMOBILE_BALLS` が効きます。
        len(day_result.acquisitions) + _fuel_as_balls(
            day_result, kinds, steps_ahead) + _night_balls(
            day_result, kinds, fuel_limit, steps_ahead)
        - IMMOBILE_BALLS * _immobile_count(day_result, kinds, problem),
        -short,
        # **そのマスを出られないまま夜を越す巡回車の少なさ。**
        # 玉数より下なので玉は1つも減りません。理由は
        # `_immobile_count` の中に。
        -_immobile_count(day_result, kinds, problem),
        # **明日すぐ取りに行ける店の数。**
        #
        # ここまでが同点になる案は毎日たくさん出ます。そのとき
        # 「明日の出だしがよい終わり方」を選びます。
        #
        # 道路の状態は今日のものを使います。交通量は前2日の滞在から
        # 決まるので、今日混んでいる道は明日も混んでいる可能性が
        # 高く、いちばん手近な見立てになります。
        # **危うい系列の少なさ。玉数より下なので、玉は1つも減りません。**
        # 同点の案が並んだとき、「明日どの系列にも余裕をもって手が
        # 届く終わり方」を選びます。実戦1960の3日目は、前日の
        # 終わりから足りない系列まで42歩（その日は48歩）で、
        # 余裕がありませんでした。明日の渋滞が悪化して51歩になり、
        # 届かなくなっています。
        -soft_miss,
        _tomorrow_reach(day_result, kinds, problem, finder, next_steps,
                        road_status),
        # **明日の朝の燃料の合計。**（今夜の合計ではありません）
        #
        # 補給は `simulator` の**歩0の行動段階でも走る**ので、補給車と
        # 同じマスで夜を越した巡回車は、翌日1歩も動かないうちに満タンに
        # なります。いちばん安い補給です。今夜の残りだけを見ていると、
        # この終わり方が見えません。
        sum(day_result.fuels) + _night_refuel_gain(day_result, kinds,
                                                   fuel_limit),
    )


@dataclass(frozen=True)
class _Event:
    """磨きの採点に必要な最小限の出来事。

    **補給の記録にも使い回しています。** そのときは店も系列も
    ありませんが、記録を書き出す側（`livemonitor`）は本物の
    `Refuel` と同じ名前で読むので、別名でも引けるようにします。
    こうしておかないと、磨きの結果がそのまま1日の答えになったときに
    `AttributeError` で試合が止まります（実際に止まりました）。
    """

    step: int
    agent_index: int
    spot_position: int
    brand: int

    @property
    def position(self) -> int:
        return self.spot_position

    @property
    def fuel_before(self) -> int:
        return -1

    @property
    def fuel_after(self) -> int:
        return -1


@dataclass
class _FastResult:
    """**公式シミュレータに通し直さずに作った1日の結果。**

    計画を作る過程で planner も1歩ずつ前に進めているので、結果は
    そこから分かります。磨きは同じ日を何万回も再現するため、
    2度目のシミュレートはそのまま2割の無駄になります。

    採点に使うのは、系列・玉数・終わりの位置と燃料・残り在庫だけ
    です。**採用する案だけは必ず公式シミュレータに通し直します**
    （`_build_candidate` を verify=True で呼び直す）。
    """

    positions: tuple
    fuels: tuple
    acquisitions: list
    refuels: list
    remaining_stocks: dict
    stay_counts: dict = field(default_factory=dict)
    wasted_spot_stays: int = 0
    step_positions: list = field(default_factory=list)
    step_fuels: list = field(default_factory=list)


def _fast_result(day_plan) -> _FastResult:
    return _FastResult(
        positions=tuple(day_plan.end_positions),
        fuels=tuple(day_plan.end_fuels),
        acquisitions=[_Event(step, agent, position, brand)
                      for step, agent, position, brand
                      in day_plan.acquisitions],
        refuels=[_Event(step, agent, -1, -1)
                 for step, agent in day_plan.refuels],
        remaining_stocks=day_plan.remaining_stocks,
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
    verify: bool = True,
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
            # 学習用の記録はここでは誰も読みません（磨きで何千回も
            # 呼ぶので、作らないぶんだけ多くの手を試せます）。
            collect_choices=False,
        )
        if verify:
            day_result = simulate_day(
                problem=problem,
                agent_kinds=kinds,
                start_positions=tuple(positions),
                start_fuels=tuple(fuels),
                action_plan=day_plan.action_plan,
                day_steps=day_steps,
                road_status=road_status,
            )
        else:
            # **磨きの最中は通し直しません**（2割の無駄になるため）。
            # 採用するときに必ず verify=True で作り直します。
            day_result = _fast_result(day_plan)
    except (InvalidAnswer, ValueError, KeyError, RuntimeError):
        return None
    return Candidate(
        label=label,
        action_plan=day_plan.action_plan,
        day_result=day_result,
        key=_candidate_key(
            day_result, acquired_brands, problem.fuel_limit, kinds, steps_ahead,
            problem, planner.finder, next_steps, road_status,
        ),
        # **補給車の置き場所は案といっしょに持ち回します。**
        # 磨きの採点で作り直すときに渡さないと消えてしまいます。
        park=getattr(policy, "park", None),
        night_park=dict(getattr(policy, "night_park", None) or {}) or None,
        home=(dict(getattr(policy, "home", None))
              if isinstance(getattr(policy, "home", None), dict) else None),
        supply_plan=(list(getattr(policy, "supply_plan", None) or []) or None),
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
# 1日の計画づくりで1回の判断に試す作戦の数。
#
# **学習済みモデルで測り直して 6 → 12 に変えました。**
#
# 以前は「12個でも6個でも結果が同じ」という測定から6個にしていました。
# しかしそれは**未学習のモデル**での結果です。学習済みでは作戦ごとの
# 質に差がつくので、多く試すほど良い組み合わせに当たります。
#
# 実戦から復元した4問（学習済み30体）:
#   6個: 日別534 / 玉881
#   10個: 日別537 / 玉917
#   **12個: 日別538 / 玉918**  ← 採用
# まとめて設計するとき、在庫をどれだけ重く見るか。
# 0 は「近い店優先」、大きいほど「在庫の多い店優先」。
# 実測（実戦の渋滞・1位の玉数に対する率）:
#   設計なし: 1535=89% / 1533=88%
#   4通り   : 1535=91% / 1533=90%
#   **8通り**: 1535=91% / 1533=92%  ← 採用
PLANNED_STOCK_WEIGHTS = (0.0, 0.5, 1.0, 2.0)

# 系列をそろえることをどれだけ重く見るか。0 は「玉数だけ」。
#
# 実戦1627の1日目（系列11必要）での実測:
#   重み0 : 最大9系列  → 候補にならない
#   重み3 : 最大10系列 → 候補にならない
#   **重み8**: 11系列すべて（玉16）
PLANNED_BRAND_WEIGHTS = (0.0, 8.0)

# 差し込みで組む設計の重み。1通り0.03〜0.1秒なので、組み合わせは
# 絞ります（1日の初めに1回だけ作ります）。
INSERT_STOCK_WEIGHTS = (0.0, 1.0)
# **補給の見込みは「満タン何杯ぶん」で、小数も入れられます。**
#
# 準決勝32x32（第1791641357・巡回6台・補給1台・燃料80）の厳密な
# 上限を、見込みを振って出すとこうなりました。
#
#   | 見込み | 1台1日あたり | 上限の玉 |
#   |---|---|---|
#   | 0杯 | +0 | 243 |
#   | **0.25杯** | **+20** | **310** |
#   | 0.5杯 | +40 | 356 |
#   | 1杯 | +80 | 373（在庫で頭打ち） |
#
# **1位は304です。** つまり1位に並ぶのに要るのは**1台1日あたり
# 20燃料**だけで、こちらは実際に1日150〜358配っています
# （6台で120あれば足りる）。**燃料は足りていました。**
#
# 足りていなかったのは**見込み**です。採用されていた案はどれも
# 「周回0x0」「系列網羅x0」——**補給を1滴も見込まない設計**で、
# だから順路が短く切られていました。いっぽう見込み1杯（+80）は
# 過大で、途中で燃料が切れて系列を落とします（`ORIENTEER_REFUELS`
# の表）。**その間の 0.25 杯がちょうど実力に合う**という見立てです。
#
# 小数を入れられるよう、既定を浮動小数の組にしてあります
# （実験で `KNOBS` から差し替えるときに型が合うように）。
INSERT_REFUELS = (0, 1)

# **同行できる構成では「燃料は気にしない」設計も作ります。**
#
# 補給車の台数が巡回車の台数以上なら、1台ずつ付き添えます。付き
# 添われた巡回車は毎歩満タンに戻るので、**燃料が制約でなくなります**
# （長岡35「同ルート走行で常時給油するため燃料残量を考慮する必要を
# なくした」、仙台広瀬1「燃料無限の巡回車として扱う」）。
#
# ところが設計は `fuels[index] + fuel_limit * refuels` を予算にする
# ので、refuels が小さいと順路が短く切られます。同行できるときは
# 大きめの補給回数でも作ります。
ESCORT_REFUELS = 6

DAY_ROUTE_BUDGET = 12


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


# 事前試走を複数のコアで走らせるか。Windows でも動きます。
PARALLEL_TRIALS = True


_WARM_FILES: tuple[str, str] | None = None


def _warm_up_workers(problem, brains, log) -> None:
    """子プロセスを前もって起こす。

    **プールは hexa.parallel が1つだけ持ち、プログラムが終わるまで
    使い回します。** ここでは問題と重みをファイルに書き出し、
    起動だけ済ませます。失敗しても何もしません。
    """

    global _WARM_FILES
    try:
        import tempfile

        from hexa.ai import save_population
        from hexa.parallel import start_pool, usable_workers

        workers = usable_workers()
        if workers <= 1:
            return
        folder = tempfile.mkdtemp(prefix="hexa_warm_")
        problem_path = str(Path(folder) / "problem.json")
        Path(problem_path).write_text(
            json.dumps(problem.to_json_dict(), ensure_ascii=False), encoding="utf-8"
        )
        population_path = ""
        if brains:
            population_path = str(Path(folder) / "population.json")
            save_population(brains, Path(population_path))
        started = time.monotonic()
        pool = start_pool(problem_path, population_path, workers)
        if pool is not None:
            _WARM_FILES = (problem_path, population_path)
            log(f"  並列の準備ができました（{workers}プロセス / "
                f"{time.monotonic() - started:.1f}秒）")
        else:
            _WARM_FILES = None
    except Exception:
        _WARM_FILES = None


# **通りがかりで踏む店まで数えた設計を候補に足すか。**
#
# 車は歩いた先の店を無条件で拾います。実測で、玉のうち通りがかりで
# 拾ったものが8〜42%ありました（第1958の32x32・3日目は42%）。
# ところが `design_day` と `design_insert` は歩数だけで組み立てていて、
# この拾いを数えていません。
#
# `design_passing` は「その差し込みで**新たに踏む店が何軒増えるか**」で
# 選びます。日によって従来より強い形が出ます（第1958・10秒磨き）:
#
#   1日目 従来 玉85/日別22   通りがかり 玉77/日別22
#   3日目 従来 玉64/日別21   通りがかり 玉63/**日別22**  ← 採点では勝ち
#   5日目 従来 玉56/日別22   通りがかり 玉51/日別22
#
# 置き換えずに**候補に足す**だけなので、良いほうが採点で選ばれます。
# 1日0.2秒ほどで、持ち時間57秒のうち0.4%です。
PASSING_ON = True
PASSING_DESIGNS = ((True, 0.0), (True, 8.0), (False, 0.0))

# **縄張りの設計**（`design_territory`）を候補に入れるか。
# **実測で入れませんでした（False で止めています）。**
# 本選1回戦の6試合（渋滞つき・手数800）で
#   縄張りなし : 日別218 玉273
#   縄張りあり : 日別215 玉263
# 割り振りそのものは正しく出ます（第6試合で系列7,8,9が店192・224・
# 245の縦の密集に、系列0,3,2が上側に、残りが右側にまとまる）。
# ところが候補に1つ足すと、磨きの登り直しの出発点が入れ替わり、
# かえって悪くなりました。**周回の形を直接磨かせる作りにしないと
# 活きません**（`pick_territory` は残してあります）。
TERRITORY_DESIGN = False
TERRITORY_REFUELS = (0, 1)

# **補給車を動かさない設計**（`design_hub`）を候補に入れるか。
#
# **実測で入れませんでした（False で止めています）。**
#
# 32x32の4試合を手数1500で通すと
#
#   ハブなし : 日別735 玉**2246** 歩の使用率89.0%
#   ハブあり : 日別735 玉 2230  歩の使用率89.9%
#
# 歩の使用率は上がるのに玉は減ります。理由は**寄り道**です。ハブは
# 店の群れの中心（medoid）に置きますが、こちらの設計は各車に近い店を
# まとめて割り当てるので、車の周回は盤面の隅に寄ります。中心まで
# 往復する寄り道のほうが、満タンで得た歩より高くつきました。
#
# 算数そのものは正しい（6台を毎日満タンにするには往復では間に合わ
# ない）ので、**周回そのものを中心を通る形に割り当て直す**必要が
# あります（秋田23の「補給車を中心とした円」、弓削37のクラスタ分割、
# 宇部22の大ループ逆走がその形）。次の一手として残します。
HUB_DESIGN = False
# ハブで満タンになるぶんを何回ぶん見込んで順路を組むか。
HUB_REFUELS = (1, 2)

# `design_cover` を候補に入れるか、補給を何回見込んで作るか。
# 磨きつき29試合（手数2000）で、候補3種（系列網羅・通りがかり・
# 種まき）を足すと **日別3135・玉7414** とどちらも最良になりました
# （順位合計は391で基準388とほぼ互角）。本番の手数はこの50〜100倍
# あるので、候補が多いほうが活きると見て入れています。
COVER_DESIGN = True
# 小数も入れられます（理由は `INSERT_REFUELS` の上に）。
# **逆回りループの設計を作るか**（宇部型）。理由は `design_loop` に。
LOOP_DESIGN = False
LOOP_REFUELS = (0, 1)


COVER_DESIGN_REFUELS = (0, 1)
# **歩数と燃料だけを見る高速ソルバを使うか。**
#（理由は `_planned_route_policies` の中に実測つきで）
ORIENTEER_DESIGN = True
# 崩して埋め直す回数。32x32で400回が0.8秒。
ORIENTEER_ROUNDS = 400
# 返してもらう形の数。
ORIENTEER_WANT = 3
# **燃料の見込み（満タン何杯ぶんを足して考えるか）。0 杯が最良。**
#
# 準決勝32x32（本物の渋滞・本物の開始状態）で、ソルバの形を公式
# シミュレータに通すと:
#
#   | 日 | 見込み0杯 | 見込み1杯 | いまの最良(磨きなし) |
#   |---|---|---|---|
#   | 1日目 | 玉**39** 系列22 | 玉39 系列22 | 玉34 系列22 |
#   | 2日目 | 玉**36** 系列22 | 玉33 系列**18** | 玉35 系列22 |
#   | 3日目 | 玉**40** 系列22 | 玉39 系列**19** | 玉32 系列22 |
#
# 補給を見込むと、途中で燃料が切れて**系列を落とします**。
# 手元の燃料だけで組ませるのが正解でした。
# 小数も入れられます（理由は `INSERT_REFUELS` の上に）。
ORIENTEER_REFUELS = (0,)

# **その日の上限を測って画面に出すか。**
# 32x32で1日2.2秒（持ち時間の4%弱）。理由は `hexa/ceiling.py` に。
CEILING_REPORT = True
# **1台ぶんの順路を厳密に解く設計を作るか。**
# 理由は `hexa/allocate.py` の `design_exact` の上に。
# **実測では入れませんでした（False で止めています）。**
# 準決勝32x32（記録どおりの渋滞・手数2000）で
#
#   | 厳密 | 夜合流 | 玉 |
#   |---|---|---|
#   | なし | なし | **240** |
#   | あり（夜合流つきの形も作る） | なし | 208 |
#   | なし | あり | **240** |
#   | あり（夜合流つきの形は作らない） | なし | **240** |
#
# 中立か悪化です。悪化するのは「最後に補給車のマスへ戻る」形で、
# 帰り道のぶん1移動あたりの燃料が1.46→1.56に上がります。
# 中立のものも1日1.5秒かかるので、既定では作りません。
#
# **ただし分かったことは大きい**ので仕掛けは残します。
# `design_exact` は1日48軒の順路を作れます（こちらの実績32〜41、
# 在庫上限65、1位は約51軒）。負けるのは**系列**のほうで、
# 軒数だけで組むと1軒しかない系列を落として日別で即負けします。
EXACT_DESIGN = False
# その設計の燃料予算（満タン何杯ぶんを足して考えるか）。
EXACT_REFUELS = (0,)
# 「最後に補給車のマスへ戻る」形も作るか。
EXACT_NIGHT = False
# **夜に補給車のマスへ全車が集まる設計を作るか。**
# 理由は `hexa/allocate.py` の `design_night` の上に。
#
# **実測では入れませんでした（False）。** 準決勝32x32で玉240のまま
# 動きません。強いて他の設計を切って夜合流だけにすると、待ちは
# 16〜40%→8〜17%に下がり、配った燃料も1300→1644に増えるのに、
# 玉は231でした。帰り道で道路や山地を通るぶん、1移動あたりの
# 燃料が1.28→1.50に上がって打ち消されます。
#
# **厳密な上限では、この形がいちばん高い**ので、やり方を変えて
# もう一度挑む価値があります:
#
#   実際の朝の燃料・補給なし        : 玉243（こちらは240）
#   毎朝満タン・補給なし            : 玉**373**
#   毎朝満タン・夜に1店へ戻る条件つき: 玉**354**（1位は304）
NIGHT_DESIGN = False
# その設計の燃料予算（満タン何杯ぶんを足して考えるか）。
# 0 は「日中の補給をあてにしない」という意味で、これが要点です。
NIGHT_REFUELS = (0, 1)
# **明日の陣形で終わる設計を作るか。**（理由は `pick_formation` の上）
FORMATION_DESIGN = True
# 明日の歩数のうち、どこまで「届く」と見るか。
FORMATION_MARGINS = (0.5, 1.0)
# 設計の燃料予算（満タン何杯ぶんを足して考えるか）。
# 小数も入れられます（理由は `INSERT_REFUELS` の上に）。
FORMATION_REFUELS = (0, 1)
# **補給車にも陣形を教えるか。** True なら「燃料が少ない車から順に、
# その車の陣形の店で待ち受ける」順路を補給車に渡します。補給は歩0の
# 行動段階でも走るので、同じマスで夜を越せば翌朝ただで満タンです。
#
# **実測では入れませんでした。** 本選6試合（手数2500）では
# 日別224/玉287 で**まったく同じ**。生成12マップ（手数600）では
#
#   補給つきも作る（8通り）: 日別1632/1634 玉3823
#   作らない（4通り）      : 日別**1634/1634** 玉**3836**
#
# と落ちます。案が倍になって磨きが薄まるぶんが、現場で決める
# `choose_supply_target`（`STUCK_FIRST` つき）より良くなりません。
# 仕掛けは残してあります。
FORMATION_SUPPLY = (False,)


# **設計が失敗した回数。** 黙って捨てないための覚え書き。
#
# この日の作業で2度やられました。`design_cover`（系列網羅）の中の
# `best_insert` に書き足した変数が未定義で `NameError` になり、
# 家族ごとの `except Exception` が握りつぶして**系列網羅の設計が
# 丸ごと消えていました**。過去29試合で順位318→321・玉7612→7554
# と落ちていたのがそれです。
#
# 消えても試合は続くので気づきません。1日の終わりに数を出します。
DESIGN_FAILURES: dict[str, int] = {}


def _design_failed(name: str) -> None:
    """設計が1つ失敗したことを数える。"""

    DESIGN_FAILURES[name] = DESIGN_FAILURES.get(name, 0) + 1


def _planned_route_policies(problem, finder, kinds, positions, fuels,
                            road_status, day_steps, acquired_brands=None,
                            next_steps=0):
    """順路をまとめて設計した方策を、重みを変えて何通りか返す。

    **系列をそろえる重みも振ります。**

    実戦1627の1日目で、玉数だけを見た設計は11系列中9系列しか
    押さえられず、候補になりませんでした。系列は日別の得点に
    直結するので、そちらを優先する案も作ります。
    """

    out = []
    try:
        from hexa.allocate import (SUPPLY_REFUELS, design_cover, design_day,
                                   design_exact, design_formation, design_hub,
                                   design_insert, design_night, design_passing,
                                   design_territory, design_with_supply,
                                   pick_night_cell, polish_exact)

        # **1台ぶんの順路を厳密に解く設計**（`design_exact`）。
        #
        # 準決勝32x32で縛っているものを順に剥がすと、最後に残ったのは
        # **順路の質**でした。1店あたりの移動がこちら6.25回に対し、
        # 上限の形は4.4回。厳密に解くと**1日48軒**（実績32〜41、
        # 在庫上限65、1位は約51軒）で、1台1.5秒で解けます。
        for refuels in (EXACT_REFUELS if EXACT_DESIGN else ()):
            for night in (True, False) if EXACT_NIGHT else (False,):
                cell = None
                if night:
                    try:
                        cell = pick_night_cell(
                            problem, finder, road_status, positions,
                            [i for i, k in enumerate(kinds)
                             if k == AgentKind.TOURING], day_steps)
                    except Exception:
                        _design_failed("夜のマス")
                        cell = None
                    if cell is None:
                        continue
                try:
                    # **系列網羅の上に、並びを厳密にして足します。**
                    # 「軒数だけ」で組むと1軒しかない系列を落とし、
                    # 日別で即負けします（日別は玉数より上）。
                    base = design_cover(
                        problem, finder, kinds, positions, fuels, road_status,
                        day_steps, acquired_brands=acquired_brands,
                        refuels=refuels)
                    routes = polish_exact(
                        problem, finder, kinds, positions, fuels, road_status,
                        day_steps, base, refuels=refuels, end=cell)
                except Exception:
                    _design_failed("厳密")
                    routes = {}
                if not any(routes.values()):
                    continue
                label = "厳密"
                if refuels:
                    label += f"x{refuels}"
                if night:
                    label += "夜合流"
                out.append((label, PlannedRoutePolicy(
                    problem, routes, park=cell)))

        # **夜に補給車のマスへ全車が集まる設計**（`design_night`）。
        #
        # 準決勝32x32を厳密な上限で切り分けると、**毎朝満タンで
        # 始まれば日中の補給は1回も要らず**、上限は玉373（補給なし・
        # 実際の朝の燃料だと243）。理由と数字は `design_night` の上に。
        for refuels in (NIGHT_REFUELS if NIGHT_DESIGN else ()):
            try:
                routes, cell = design_night(
                    problem, finder, kinds, positions, fuels, road_status,
                    day_steps, acquired_brands=acquired_brands,
                    refuels=refuels)
            except Exception:
                _design_failed("夜合流")
                routes, cell = {}, None
            if cell is not None and any(routes.values()):
                label = "夜合流"
                if refuels:
                    label += f"x{refuels}"
                out.append((label, PlannedRoutePolicy(
                    problem, routes, park=cell)))

        # **歩数と燃料だけを見る高速ソルバ**（`hexa/orienteering.py`）。
        #
        # 磨きは1手ごとに1日を丸ごと再現するので、32x32では1手3ms。
        # いっぽう歩数表を引くだけの検査は10マイクロ秒で**300倍速い**。
        # そこで「崩して埋め直す」探索を歩数・燃料だけで回し、良かった
        # 形を磨きの出発点に渡します。
        #
        # 準決勝の2試合（本物の渋滞・本物の開始状態）で、1日0.1〜0.8秒
        # 回しただけの見立て:
        #
        #   | | 実戦(1.49.0) | ソルバ | 1位 |
        #   |---|---|---|---|
        #   | 16x16 玉 | 65 | **73** | 71 |
        #   | 32x32 玉 | 216 | **290** | 304 |
        #
        # 見立てが外れても、採否は公式シミュレータに通した正確な採点が
        # 決めるので点数は下がりません（候補が増えるだけ）。
        if ORIENTEER_DESIGN:
            from hexa.orienteering import search as orienteer
            for refuels in ORIENTEER_REFUELS:
                try:
                    plans = orienteer(
                        problem, finder, kinds, positions, fuels, road_status,
                        day_steps, refuels=refuels, rounds=ORIENTEER_ROUNDS,
                        want=ORIENTEER_WANT)
                except Exception:
                    _design_failed("周回")
                    plans = []
                for number, routes in enumerate(plans):
                    if not any(routes.values()):
                        continue
                    label = f"周回{number}"
                    if refuels != 1:
                        label += f"x{refuels}"
                    out.append((label, PlannedRoutePolicy(problem, routes)))

        # **明日の陣形で終わる設計**（`design_formation`）。
        #
        # 厳密な上限（部分集合DP）と比べると、本選1回戦第1試合は
        # 「届かない系列なし・出発点が自由なら毎日満点」でした。
        # 差は**終わり方（位置と燃料）**だけです。理由と数字は
        # `hexa/allocate.py` の `pick_formation` の上に。
        for margin in (FORMATION_MARGINS if FORMATION_DESIGN else ()):
            for refuels in FORMATION_REFUELS:
                for with_supply in FORMATION_SUPPLY:
                    try:
                        got = design_formation(
                            problem, finder, kinds, positions, fuels,
                            road_status, day_steps, next_steps=next_steps,
                            acquired_brands=acquired_brands, refuels=refuels,
                            margin=margin, with_supply=with_supply)
                    except Exception:
                        _design_failed("陣形")
                        got = ({}, []) if with_supply else {}
                    routes, supply_route = got if with_supply else (got, None)
                    if not any(routes.values()):
                        continue
                    label = f"陣形{margin:.1f}"
                    if refuels != 1:
                        label += f"x{refuels}"
                    if with_supply:
                        label += "補給"
                    out.append((label, PlannedRoutePolicy(
                        problem, routes, supply_route)))

        # **逆回りループの設計**（`design_loop`）。既定はオフ。
        # 理由と実測は `hexa/allocate.py` の `design_loop` の上に。
        for refuels in (LOOP_REFUELS if LOOP_DESIGN else ()):
            try:
                from hexa.allocate import design_loop
                routes, cells = design_loop(
                    problem, finder, kinds, positions, fuels, road_status,
                    day_steps, acquired_brands=acquired_brands,
                    refuels=refuels)
            except Exception:
                _design_failed("逆回り")
                routes, cells = {}, []
            if any(routes.values()):
                label = "逆回り" + (f"x{refuels}" if refuels != 1 else "")
                out.append((label, PlannedRoutePolicy(
                    problem, routes, supply_cells=cells)))

        # **系列をまず全部押さえる設計**（`design_cover`）。
        #
        # 第1627（16x16）では系列10個のうち7個が1軒しかなく、
        # 日別満点にはその7マスを毎日踏むしかありません。玉数から
        # 組むとその高い店が切り捨てられ、実戦では2・3・4日目に
        # 系列0と1を落としました（日別36、1位は44）。
        for refuels in (COVER_DESIGN_REFUELS if COVER_DESIGN else ()):
            try:
                routes = design_cover(
                    problem, finder, kinds, positions, fuels, road_status,
                    day_steps, acquired_brands=acquired_brands,
                    refuels=refuels,
                )
            except Exception:
                _design_failed("系列網羅")
                routes = {}
            if any(routes.values()):
                label = "系列網羅"
                if refuels != 1:
                    label += f"x{refuels}"
                out.append((label, PlannedRoutePolicy(problem, routes)))

        # **店を順路の途中に差し込んで組む設計も作ります。**
        #
        # `design_day` は末尾に足すだけなので、2店のあいだに割り込む
        # 形が作れません。実戦1703（32x32）では1玉あたり9.5歩で、
        # 3位は8.0歩でした。立ち止まりは10%しかないので、差は
        # 回り方そのものにあります。
        # 同行できる構成（補給車 >= 巡回車）では、燃料を気にしない
        # 予算でも設計します（`ESCORT_REFUELS`）。
        supply_cars = sum(1 for kind in kinds if kind == AgentKind.SUPPLY)
        touring_cars = sum(1 for kind in kinds if kind == AgentKind.TOURING)
        insert_refuels = tuple(INSERT_REFUELS)
        if supply_cars and supply_cars >= touring_cars:
            insert_refuels += (ESCORT_REFUELS,)
        for weight in INSERT_STOCK_WEIGHTS:
            for brand in PLANNED_BRAND_WEIGHTS:
                for refuels in insert_refuels:
                    routes = design_insert(
                        problem, finder, kinds, positions, fuels, road_status,
                        day_steps, stock_weight=weight, brand_weight=brand,
                        acquired_brands=acquired_brands, refuels=refuels,
                    )
                    if any(routes.values()):
                        label = f"差し込み{weight:.2f}"
                        if refuels:
                            label += f"x{refuels}"
                        if brand:
                            label += f"系列{brand:.0f}"
                        out.append((label, PlannedRoutePolicy(problem, routes)))

        # **縄張りの設計**（`design_territory`）。
        #
        # 系列を車に割り振って、毎日同じ周回を繰り返す形です。本選
        # 1回戦では日別が勝負（1位でも44/50）で、こちらは日ごとに
        # 組み直すため1日目に燃料を使い切り、2〜3日目に2台が
        # 動けなくなっていました。理由の数字は `design_territory` に。
        if TERRITORY_DESIGN:
            for refuels in TERRITORY_REFUELS:
                try:
                    routes = design_territory(
                        problem, finder, kinds, positions, fuels, road_status,
                        day_steps, acquired_brands=acquired_brands,
                        refuels=refuels)
                except Exception:
                    _design_failed("縄張り")
                    routes = {}
                if any(routes.values()):
                    label = "縄張り" + (f"x{refuels}" if refuels != 1 else "")
                    out.append((label, PlannedRoutePolicy(problem, routes)))

        # **補給車を動かさない設計**（`design_hub`）。
        #
        # 全車の周回が通る一点（店）に補給車を置き、巡回車はそこを
        # 順路に含めます。補給車は移動ゼロで全車を満タンにできます。
        # 理由の数字は `hexa/allocate.py` の `design_hub` に。
        if HUB_DESIGN:
            for refuels in HUB_REFUELS:
                try:
                    routes, hub = design_hub(
                        problem, finder, kinds, positions, fuels, road_status,
                        day_steps, acquired_brands=acquired_brands,
                        refuels=refuels)
                except Exception:
                    _design_failed("ハブ")
                    routes, hub = {}, None
                if hub is not None and any(routes.values()):
                    label = "ハブ" + (f"x{refuels}" if refuels != 1 else "")
                    out.append((label, PlannedRoutePolicy(
                        problem, routes, None, park=hub)))

        # **通りがかりで踏む店まで数えた設計**（`PASSING_DESIGNS`）。
        for per_step, brand in (PASSING_DESIGNS if PASSING_ON else ()):
            try:
                routes = design_passing(
                    problem, finder, kinds, positions, fuels, road_status,
                    day_steps, brand_weight=brand,
                    acquired_brands=acquired_brands, refuels=1,
                    per_step=per_step,
                )
            except Exception:
                _design_failed("通りがかり")
                routes = {}
            if any(routes.values()):
                label = "通りがかり" + ("歩" if per_step else "軒")
                if brand:
                    label += f"系列{brand:.0f}"
                out.append((label, PlannedRoutePolicy(problem, routes)))

        for weight in PLANNED_STOCK_WEIGHTS:
            for brand in PLANNED_BRAND_WEIGHTS:
                routes = design_day(
                    problem, finder, kinds, positions, fuels, road_status,
                    day_steps, stock_weight=weight, brand_weight=brand,
                    acquired_brands=acquired_brands,
                )
                if any(routes.values()):
                    label = f"設計{weight:.2f}"
                    if brand:
                        label += f"系列{brand:.0f}"
                    out.append((label, PlannedRoutePolicy(problem, routes)))
                # **補給を見込んだ設計も作ります。**
                # 燃料が足りないせいで順路が短くなる試合で効きます。
                for refuels in SUPPLY_REFUELS:
                    routes, supply_route = design_with_supply(
                        problem, finder, kinds, positions, fuels, road_status,
                        day_steps, stock_weight=weight, brand_weight=brand,
                        acquired_brands=acquired_brands, refuels=refuels,
                    )
                    if any(routes.values()):
                        label = f"補給設計{weight:.2f}x{refuels}"
                        if brand:
                            label += f"系列{brand:.0f}"
                        out.append((
                            label,
                            PlannedRoutePolicy(problem, routes, supply_route),
                        ))
    except Exception:
        _design_failed("設計ぜんぶ")
        return []
    return out


# **登り直しの出発点に「玉の多い案」「日別の多い案」も混ぜるか。**
#
# 採点の順（日別 → 玉数）だけで並べると、玉の多い案は下位に沈んで
# 出発点に選ばれません。ところが磨いた結果はそちらのほうが良いことが
# あります（`diverse_seeds` に実測つきの理由）。
SEED_OTHER_CRITERIA = True


# **形のちがいを何で測るか。**
#
#   "routes"     : 車ごとの店の並びが完全一致したときだけ同じとみなす
#   "brands_end" : **その日に取った系列の集合と、終わりの位置**で測る
#
# `routes` は**ほぼ絶対に衝突しません**。1店でも違えば別扱いなので、
# 重複排除が何もせず、`diverse_seeds` は実質「鍵の上位 limit 件」を
# 返していました。上位数件は中身がほとんど同じ案なので、**出発点を
# 5か所に分けているつもりで1か所から登っていた**ことになります。
#
# 今日、設計の家族を6つ作って15試合で測ったら全部負けました
# （近づく手211／夜合流190／待ち伏せ178／逆回り201／在庫の重み191。
# 基準163）。どの家族も出発点に選ばれていないので、**そもそも
# 「その設計から磨いたらどうなるか」を1度も測れていません**。
#
# 沖縄高専（解説集 p10下）はここを明文化しています。
#
# > その際、候補が類似した経路に集中しないよう、**取得できる系列の
# > 組合せと終了位置の多様性を考慮する**。
#
# 系列の集合と終わりの位置が同じ案は、磨きの出発点としては同じ谷です。
# そこで既定をそちらにします。
SEED_SIGNATURE = "routes"


def _seed_signature(candidate, routes, kinds):
    """磨きの出発点として「同じ谷か」を測る印。理由は上に。"""

    if SEED_SIGNATURE != "brands_end":
        return tuple(sorted(
            (index, tuple(spots)) for index, spots in routes.items()
        ))
    result = candidate.day_result
    brands = frozenset(event.brand for event in result.acquisitions)
    ends = tuple(result.positions[index] for index, kind in enumerate(kinds)
                 if kind == AgentKind.TOURING)
    return (brands, ends)


def diverse_seeds(candidates, kinds, limit: int):
    """**形のちがう良い案**を、良い順に `limit` 個まで返す。

    **なぜ必要か**

    磨きは登り切ると止まります。実戦1891（24x24）の盤面で測ると、
    貪欲1つから磨いた場合:

      20秒  : 玉46 / 40 / 30 / 36（1〜4日目）
      100秒 : 玉46 / 40 / 30 / 36  ← **5倍かけても1玉も動かない**

    手数は18,403手まで伸びているのに結果は同じでした。同じ谷を
    登り切っているので、時間を足しても意味がありません。

    ところが**出発点を5か所に分けて20秒ずつ**にすると:

      1日目 玉46 → **玉50** / 3日目 玉30 → **玉31**

    と越えられます。だから子プロセスには「いまの最良」ばかりでなく、
    **形のちがう案**を配ります。同じ形を配ると、別の子が同じ谷を
    登り直すだけになるので、車ごとの店の並びで重複を省きます。
    """

    # **採点の順だけで並べません。**
    #
    # 採点は日別累計を玉数より上に見ます（公式の順位がそうなので正解
    # です）。ところが**磨きの出発点としては、玉の多い案のほうが良い
    # ことがあります**。実測（第1958の32x32・3日目・10秒磨き）:
    #
    #   採点で最良の設計（玉45/日別22）から磨く → 玉59/日別22
    #   玉の多い設計（玉65/日別20）から磨く     → **玉62**/日別21
    #
    # 採点の順だと玉の多い案は下位に沈むので、出発点に選ばれません。
    # そこで「玉がいちばん多い案」と「日別がいちばん多い案」を先頭に
    # 混ぜます。採点そのものは変えないので、採用の判断は今までどおり
    # 公式の順位です。
    pool = sorted(candidates, key=lambda item: item.key, reverse=True)
    if SEED_OTHER_CRITERIA and len(pool) > 1:
        extra = []
        try:
            extra.append(max(candidates,
                             key=lambda item: item.key[KEY_NOODLES]))
            extra.append(max(candidates, key=lambda item: item.key[1]))
        except Exception:
            extra = []
        # 先頭の次に差し込みます（いちばん良い案は必ず残します）。
        head = pool[:1]
        rest = [item for item in pool[1:] if item not in extra]
        pool = head + [item for item in extra if item not in head] + rest
    seeds = []
    seen: set = set()
    for candidate in pool:
        try:
            routes = routes_from_result(candidate.day_result, kinds)
        except Exception:
            continue
        if not routes:
            continue
        signature = _seed_signature(candidate, routes, kinds)
        if signature in seen:
            continue
        seen.add(signature)
        seeds.append(candidate)
        if len(seeds) >= limit:
            break
    return seeds


# **足りない系列を1つずつ差し込んで直す回数の上限。**
#
# 1つの系列につき、いちばん安い差し込み方をこの数だけ試します。
BRAND_FIX_TRIES = 3
# 横ばいの手を挟んでよい回数（これを超えたら元に戻します）。
BRAND_FIX_CHAIN = 4


# **採用する前に、並びだけ最善にして余った歩で店を足す。**
#
# 過去29試合を通すと、**日別は27試合で1位と同じ**なのに玉数だけが
# 5〜25%足りません（16x16で5〜12玉、24x24で20〜70玉、32x32で
# 100〜140玉）。盤面が大きいほど差が開くので、原因は順路の質です。
#
# 準決勝32x32で測ると、1店あたりの移動がこちらの設計で6.25回、
# 厳密に解いた形で4.4回でした。**並びだけで3割ぶんの歩が浮きます。**
#
# `polish_exact` は**店の集合を変えません**。集合が同じなら系列も
# 同じなので、**日別は1つも減りません**。浮いた歩で店を足すので
# 玉だけが増えます。12店までは部分集合DPで厳密に並べ替えます。
#
# **実測では入れませんでした（False）。** 過去29試合（磨きつき・
# 手数2000）で順位合計 318→**321**、玉7612→7606。1試合だけ悪化し、
# 良くなった試合は1つもありませんでした。
#
# 理由は、磨きがすでに並びを詰めているからです。`design_cover` の
# 生の順路では1店あたり6.25移動でも、磨きを通した案では既に
# 短くなっていて、ここで並べ替える余地が残っていません。
#
# **仕掛けは残します。** `polish_exact` は「店の集合を変えずに
# 並びだけ厳密にする」ので、設計の段階で使えば効く見込みがあります。
EXACT_POLISH = False


def improve_by_exact_order(problem, finder, kinds, positions, road_status,
                           best, evaluate, supply_order, fuels=None,
                           log=None):
    """並びを最善にして店を足した案を返す。良くならなければ None。"""

    if not EXACT_POLISH or problem is None or finder is None:
        return None
    try:
        from hexa.allocate import polish_exact

        routes = routes_from_result(best.day_result, kinds)
        if not any(routes.values()):
            return None
        start = list(positions)
        room = list(fuels) if fuels is not None else [
            problem.fuel_limit for _ in kinds]
        fixed = polish_exact(
            problem, finder, kinds, start, room, road_status,
            _day_steps_of(best), routes)
        if not fixed or fixed == routes:
            return None
        got = evaluate(fixed, supply_order, verify=True)
    except Exception:
        return None
    if got is None or got.key <= best.key:
        return None
    if log is not None:
        log(f"  並びを直して玉が増えました（{best.key[KEY_NOODLES]} → "
            f"{got.key[KEY_NOODLES]}）")
    return got


def _day_steps_of(candidate) -> int:
    """その案が使った歩数（行動の列の長さから読む）。"""

    plan = candidate.action_plan or []
    if not plan:
        return 0
    return max(sum(abs(int(value)) for value in line) for line in plan)


# **夜の合流を使うか。**
NIGHT_PARK_ON = False
# **巡回車が何台までなら使うか。**
#
# 盤の幅で台数は決まります（16→4体・24→5体・32→7体）。つまり
# 「巡回3台まで」は16x16だけ、「5台まで」は24x24まで、という意味です。
#
# 大きい盤では往復が高くつきます。準決勝32x32で全車を帰らせると
# 日別が132→101に落ちました（満点は132。日別は玉数より上位なので
# 即負け）。小さい盤は往復が安いので、別の話になります。
NIGHT_PARK_MAX_TOURING = 99
# 帰らせる台数の上限。
NIGHT_PARK_SIZES = 6
# 試す集合場所の数（それぞれ台数を0〜6で振るので、1日最大42回通す。
# 32x32で1回3msなので0.13秒）。
NIGHT_PARK_CELLS = 6
# 寄り道1歩を燃料いくつの損と見るか。玉に換算した理由は関数の中に。
NIGHT_PARK_STEP_COST = 3.6


def park_supply_for_night(problem, finder, kinds, positions, road_status,
                          day_steps, best, evaluate, supply_order):
    """**夜を同じマスで越させて、翌朝ただで満タンにする。**

    **なぜこれがいちばん大事なのか。** 準決勝32x32（第1791641357・
    巡回6台・補給1台・燃料80・歩96）を厳密な上限（部分集合DP）で
    切り分けると

      実際の朝の燃料・補給なし : 玉**243**
      毎朝満タン・在庫で頭打ち : 玉**373**
      こちらの実戦(1.49.0)     : 玉216
      1位                      : 玉**304**

    **1位の304は、こちらの「実際の燃料での上限243」より上**です。
    順路をどれだけ磨いても243は越えられないので、1位は燃料をこちらより
    多く手にしています。残っている道はここだけです。

    `simulator` の補給（要項1.4）は**歩0の行動段階でも走る**ので、
    補給車と同じマスで夜を越した巡回車は翌朝1歩も動かないうちに
    満タンになります。巡回車の歩も補給車の歩も使いません。しかも
    1マスに何台でも入れるので、補給車1台で何台ぶんでも配れます。

    **ただし全車を帰らせると負けます。** 集合場所を店683に決めて
    6日通すと夜の合流は4〜6台まとまりましたが、往復に歩を使うので

      日別 130 → **101**（満点132） / 玉 216 → 206

    と、日別で大きく負けました（日別は玉数より上位の得点）。遠くの
    系列を受け持つ車を帰らせると、その系列が落ちるからです。

    そこで**帰らせる車を選びます**。

      1. 合流先の候補を作る（巡回車が夜を越す予定のマスと、
         補給車から届く範囲でいちばん多くの車が集まれるマス）
      2. 候補ごとに、寄り道の安い車から順に「帰る車」を増やす
      3. それぞれ実際に通して、点数が上がったものを採る

    採点は日別が玉数より上なので、**日別を落とす合流案は自動的に
    捨てられます**。1台ぶんでも翌朝の満タンが増えれば、その日の
    玉を1つ落としてでも採る価値があることは鍵の下の桁
    （`_night_refuel_gain`）が見ます。
    """

    if not NIGHT_PARK_ON or best is None:
        return best
    supply = [index for index, kind in enumerate(kinds)
              if kind == AgentKind.SUPPLY]
    touring = [index for index, kind in enumerate(kinds)
               if kind == AgentKind.TOURING]
    if not supply or not touring:
        return best
    if len(touring) > NIGHT_PARK_MAX_TOURING:
        return best

    limit = problem.fuel_limit
    result = best.day_result
    ends = {index: result.positions[index] for index in touring}
    room = {index: max(0, limit - result.fuels[index]) for index in touring}
    # 補給車がどこまで届くか。補給車は**燃料を1滴も使いません**
    # （要項 p.11）。歩数だけが縛りです。
    reach = {index: finder.table(positions[index], road_status)
             for index in supply}
    # 巡回車が夜を越す場所から、どの集合場所へいくら寄り道になるか。
    from_end = {index: finder.table(ends[index], road_status)
                for index in touring}

    # 1. 集合場所の候補を点数で選びます。
    #
    #    どのマスでもよく、店でなくてもかまいません（店にすると在庫の
    #    縛りを受けるので、むしろ平地のほうが素直です）。
    #
    #    点数は**玉に換算**します。翌朝ただで入る燃料80は、1台の
    #    1日の上限を玉6.75→11に上げるので**玉4つぶん**（厳密な上限
    #    から。燃料80で玉243、燃料160で玉373）。いっぽう寄り道1歩は
    #    玉0.18ぶんの損です（1位もこちらも1移動あたり0.18玉）。
    #    つまり**寄り道1歩は燃料3.6ぶんの損**で、燃料80を取りに
    #    行けるのは寄り道22歩まで、という計算になります。
    best_cells: list[tuple[float, int, int]] = []     # (点, マス, 補給車)
    for cell in range(problem.width * problem.height):
        car = None
        for index in supply:
            steps = reach[index].steps.get(cell)
            if steps is None or steps > day_steps:
                continue
            if car is None or steps < car[0]:
                car = (steps, index)
        if car is None:
            continue
        cost = []
        for index in touring:
            steps = from_end[index].steps.get(cell)
            if steps is not None:
                cost.append((steps, index))
        if not cost:
            continue
        cost.sort()
        score = gain = spent = 0
        for steps, index in cost[:NIGHT_PARK_SIZES]:
            gain += room[index]
            spent += steps
            score = max(score, gain - NIGHT_PARK_STEP_COST * spent)
        best_cells.append((score, cell, car[1]))
    if not best_cells:
        return best
    best_cells.sort(key=lambda item: -item[0])
    # **巡回車が夜を越す予定のマスも必ず試します。** 寄り道0の車が
    # 1台はいるので、損のない候補です。点数では埋もれることがあります。
    head = []
    for index in touring:
        cell = ends[index]
        car = min((other for other in supply
                   if (reach[other].steps.get(cell) or 10 ** 9) <= day_steps),
                  key=lambda other: reach[other].steps.get(cell, 10 ** 9),
                  default=None)
        if car is not None:
            head.append((0.0, cell, car))
    cells: list[tuple[int, int]] = []
    for _score, cell, car in head + best_cells:
        if (cell, car) in cells:
            continue
        cells.append((cell, car))
        if len(cells) >= NIGHT_PARK_CELLS:
            break

    # 2. 候補ごとに、帰らせる台数を増やしながら実際に通します。
    current = best
    routes = routes_from_result(best.day_result, kinds)
    for cell, car in cells:
        cost = sorted((from_end[index].steps.get(cell, 10 ** 9), index)
                      for index in touring)
        cost = [item for item in cost if item[0] < 10 ** 9]
        for size in range(0, min(NIGHT_PARK_SIZES, len(cost)) + 1):
            home = {index: cell for _steps, index in cost[:size]}
            try:
                # **空の辞書も渡せます。** None にすると磨き用の控え
                # （`polish_home`）が効いて、前の帰り先を引きずります。
                got = evaluate(routes, supply_order, night_park={car: cell},
                               home=home)
            except TypeError:
                return current
            if got is not None and got.key > current.key:
                current = got
    return current


# **巡回路を凍結してから、燃料が尽きる直前に補給車を横切らせる。**
#
# 久留米高専（解説集 p28下）がこう書いています。
#
# > 各巡回車における燃料残量についての問題は、**巡回車の移動を事前に
# > 確定させ、燃料が尽きるステップの直前に補給車を横切らせる**ことで
# > 解決する。
#
# 一関（p11上）は「経路をセル単位・時刻単位に展開して合流可否を確認」、
# 沖縄（p10下）は「分枝限定で全体を組み、補給計画の実現可能性を動的
# 計画法で判定」。**玉数で上位のチームは、巡回車と補給車を一緒に
# 解いています。**
#
# こちらの補給車は1歩ずつ貪欲に決めていました。待ち伏せ
# （`SUPPLY_INTERCEPT`）はその考えのいちばん簡単な形で、それだけで
# 15試合の玉が3836→3975（順位合計195→174）になりました。ここは
# **時刻まで合わせた版**です。
#
# やること。
#
#   1. いまの最良案を1歩ずつ再現して、**各巡回車がいつどこにいて燃料が
#      いくらか**を全部出す（`step_positions` / `step_fuels`）
#   2. 車ごとに「**もう次の1歩が払えなくなる歩**」を見つける（なければ
#      燃料がいちばん少なくなる歩）。そこが**入る量がいちばん大きい
#      待ち合わせ地点**
#   3. 時刻の早い順に並べ、補給車がその歩までに着けるものだけを拾って
#      約束の列にする
#   4. 公式シミュレータに通して、点数が上がったら採る
#
# 約束は `PlannedRoutePolicy.supply_plan` に (歩, セル, 車) で渡します。
SUPPLY_PLAN_ON = False
# 試す並べ方の数（時刻順・入る量順・近い順）。
SUPPLY_PLAN_TRIES = 3
# **「空に近い」とみなす燃料の割合。** 上限のこの割合まで落ちた歩で会います。
SUPPLY_PLAN_FLOOR = 0.25


def plan_supply_meetings(problem, finder, kinds, positions, fuels,
                         road_status, day_steps, best, evaluate, supply_order):
    """**時刻つきの待ち合わせ**を組んで、良くなったら返す。理由は上に。"""

    if not SUPPLY_PLAN_ON or best is None:
        return best
    supply = [index for index, kind in enumerate(kinds)
              if kind == AgentKind.SUPPLY]
    touring = [index for index, kind in enumerate(kinds)
               if kind == AgentKind.TOURING]
    if len(supply) != 1 or not touring:
        # 補給車が1台の盤だけ扱います（2台以上は約束の割り振りが別問題）。
        return best

    # 1. 1歩ごとの軌跡。`_build_candidate` の結果には入っていないので、
    #    ここで1回だけ通し直します（1日1回なので値段はありません）。
    try:
        trace = simulate_day(
            problem=problem, agent_kinds=kinds,
            start_positions=tuple(positions), start_fuels=tuple(fuels),
            action_plan=best.action_plan, day_steps=day_steps,
            road_status=road_status, record_trace=True)
    except Exception:
        return best
    if not trace.step_fuels or not trace.step_positions:
        return best

    limit = problem.fuel_limit
    last = min(len(trace.step_fuels), len(trace.step_positions)) - 1
    if last <= 0:
        return best

    # 2. 車ごとの「**燃料が尽きる直前**の歩」。
    #
    # **「燃料がいちばん少なくなる歩」を選んではいけません。** それは
    # たいてい1日の終わりなので、補給車が1日待ってしまい、入る量は
    # 大きくても**その車はもう走れません**。久留米の記述も「燃料が
    # 尽きる**直前**」です。1試合で測ると、終わりで会う形は玉168→160
    # と落ちました。
    #
    # そこで**最初に空に近づく歩**を採ります。そこで満タンにすれば、
    # その車は残りの歩を走り切れます。
    floor = max(1, int(limit * SUPPLY_PLAN_FLOOR))
    wants = []
    for car in touring:
        pick = None
        for step in range(last + 1):
            tank = trace.step_fuels[step][car]
            if tank <= floor:
                pick = (step, trace.step_positions[step][car], limit - tank)
                break
        if pick is None:
            # 1日ずっと余裕がある車。いちばん減った歩で会います
            # （入る量が小さければ下で落とします）。
            low = min(range(last + 1),
                      key=lambda s: trace.step_fuels[s][car])
            pick = (low, trace.step_positions[low][car],
                    limit - trace.step_fuels[low][car])
        if pick[2] <= SUPPLY_SKIP_ROOM:
            continue
        wants.append((pick[0], pick[1], car, pick[2]))

    if not wants:
        return best

    # 3. 並べ方を何通りか作り、補給車が間に合うものだけ拾います。
    source = positions[supply[0]]
    orders = [
        sorted(wants, key=lambda item: item[0]),            # 時刻の早い順
        sorted(wants, key=lambda item: -item[3]),           # 入る量の多い順
    ]
    table = finder.table(source, road_status)
    orders.append(sorted(
        wants, key=lambda item: table.steps.get(item[1], 1 << 30)))

    plans = []
    for want_order in orders[:SUPPLY_PLAN_TRIES]:
        plan = []
        here = source
        clock = 0
        for when, cell, car, _room in want_order:
            hop = finder.table(here, road_status).steps.get(cell)
            if hop is None:
                continue
            arrive = clock + hop
            if arrive > when + SUPPLY_PLAN_SLACK:
                # 約束の時刻に間に合いません。
                continue
            if arrive > day_steps:
                break
            plan.append((max(when, arrive), cell, car))
            # 約束の時刻まで待つので、次の出発はその時刻から。
            clock = max(when, arrive)
            here = cell
        if plan and plan not in plans:
            plans.append(plan)
    if not plans:
        return best

    # 4. 通して、良くなったら採る。
    routes = routes_from_result(best.day_result, kinds)
    current = best
    for plan in plans:
        try:
            got = evaluate(routes, supply_order, supply_plan=plan)
        except TypeError:
            return current
        if got is not None and got.key > current.key:
            current = got
    return current


def cover_missing_brands(problem, finder, kinds, positions, road_status,
                         best, evaluate, supply_order):
    """**その日に取れていない系列を、差し込みで拾いに行く。**

    **実戦で分かったこと。** 上位陣は日別累計が満点です。

      実戦1958（32x32・9日・系列22）: 1位も12位も日別198（=22x9）。
        こちらは**197**。9日目だけ21系列で、それだけで順位が4つ下。
      実戦1960（24x24・7日・系列20）: 上位は日別140（=20x7）。
        こちらは**137**。3日目に2つ、5日目に1つ落として13位。

    日別累計は**玉数より上位の得点**なので、1つ落とすと玉で挽回でき
    ません。落としたのはどれも**歩数の短い日**（68歩・48歩・57歩）で、
    玉を最大にする形だと遠い系列が切り捨てられます。

    採点は日別（key[1]）を玉数（KEY_NOODLES）より上に置いてあるので、
    **全系列そろえた案が候補に1つでもあれば必ず選ばれます**。
    つまり足りないのは採点ではなく、その案を作ることでした。

    ここでは、いまの最良案に対して

      1. 今日まだ取れていない系列を数える
      2. その系列の店を、いちばん安く差し込める車と位置を探す
      3. 実際に通して、点数が上がったら採る

    を系列ごとに繰り返します。差し込みは安い順に `BRAND_FIX_TRIES`
    通りだけ試すので、1回あたり十数回の再現で済みます。
    """

    if best is None:
        return best
    spots_by_brand: dict[int, list] = {}
    brand_of: dict[int, int] = {}
    for spot in problem.spots:
        spots_by_brand.setdefault(spot.brand, []).append(spot.position)
        brand_of[spot.position] = spot.brand
    touring = [index for index, kind in enumerate(kinds)
               if kind == AgentKind.TOURING]
    if not touring:
        return best

    current = best
    for _round in range(len(spots_by_brand)):
        have = {event.brand for event in current.day_result.acquisitions}
        missing = [brand for brand in spots_by_brand if brand not in have]
        if not missing:
            break
        routes = routes_from_result(current.day_result, kinds)
        tries = []
        for brand in missing:
            for spot in spots_by_brand[brand]:
                for car in touring:
                    line = routes.get(car, [])
                    if spot in line:
                        continue
                    table = finder.table(positions[car], road_status)
                    for at in range(len(line) + 1):
                        before = (positions[car] if at == 0
                                  else line[at - 1])
                        head = finder.table(before, road_status)
                        gap = head.steps.get(spot)
                        if gap is None:
                            continue
                        if at < len(line):
                            back = finder.table(spot, road_status)
                            tail = back.steps.get(line[at])
                            was = head.steps.get(line[at])
                            if tail is None or was is None:
                                continue
                            delta = gap + tail - was
                        else:
                            delta = gap
                        tries.append((delta, car, at, spot))
        # **まず「その車の最初の目的地にする」形を試します。**
        #
        # 途中に差し込むと順路が長くなり、planner がそこへ着く前に
        # 歩数を使い切ります。実測でも、安い順に差し込むだけでは
        # 系列は1つも増えませんでした（32x32の9日目・24x24の3日目
        # と5日目のいずれも）。
        #
        # 先頭に置けば**必ず着きます**。そのぶん後ろの店を落として
        # 玉は減りますが、日別累計は玉数より上位の得点なので、
        # 採点が正しく判断します。
        front = []
        for brand in missing:
            for spot in spots_by_brand[brand]:
                for car in touring:
                    if spot in routes.get(car, []):
                        continue
                    table = finder.table(positions[car], road_status)
                    gap = table.steps.get(spot)
                    if gap is not None:
                        front.append((gap, car, 0, spot))
        front.sort()
        tries.sort()
        if not front and not tries:
            break
        picked = None
        level = None
        for _delta, car, at, spot in (front[:BRAND_FIX_TRIES]
                                      + tries[:BRAND_FIX_TRIES]):
            trial = {index: list(line) for index, line in routes.items()}
            trial.setdefault(car, [])
            trial[car] = [item for item in trial[car] if item != spot]
            trial[car] = trial[car][:at] + [spot] + trial[car][at:]
            got = evaluate(trial, supply_order, verify=True)
            if got is None:
                continue
            if got.key > current.key and not (
                    BRAND_FIX_DAILY_ONLY
                    and got.key[1] <= current.key[1]):
                picked = got
                break
            # **横ばいの手も1つだけ覚えておきます。**
            #
            # 実測で分かったこと（32x32の9日目）。足りない系列0の店へ
            # 車1を先に行かせると**系列0は取れます**が、その車が
            # 受け持っていた別の系列を落とし、日別は21のまま動きません。
            # 1手では越えられない壁です。
            #
            # そこで「系列の数は同じだが、足りない系列が入れ替わった」
            # 形をいったん受け取り、**次の周でその入れ替わったほうを
            # 拾いに行きます**。最後に元より良くなっていなければ
            # 元を返すので、損はしません。
            if (level is None and got.key[1] >= current.key[1]
                    and brand_of.get(spot) in missing):
                after = {event.brand
                         for event in got.day_result.acquisitions}
                if len(set(missing) - after) < len(missing):
                    level = got
        if picked is None:
            picked = level
        if picked is None:
            break
        current = picked
        if current.key < best.key:
            # 横ばいを挟んでいる途中。深追いはしません。
            if _round >= BRAND_FIX_CHAIN:
                break
    if current.key <= best.key:
        return best
    if BRAND_FIX_DAILY_ONLY and current.key[1] <= best.key[1]:
        # 日別が増えていないなら、ここでは直しません（玉数が減るだけ）。
        return best
    return current


def routes_from_result(day_result, kinds) -> dict[int, list[int]]:
    """できあがった1日の結果から、「実際に回った店の並び」を取り出す。

    **なぜ必要か**

    順路を磨く仕組みは、出発点として「車ごとの店の並び」を必要と
    します。ところが実戦でいちばん強い案は、設計ではなく
    「揺らぎのある貪欲」や学習済みAIが作ります。その案は並びの形を
    持っていません。

    結果から取り出せば、**どの方策が作った案でも磨けます**。
    """

    routes: dict[int, list[int]] = {
        index: [] for index, kind in enumerate(kinds)
        if kind == AgentKind.TOURING
    }
    for event in sorted(day_result.acquisitions, key=lambda item: item.step):
        if event.agent_index in routes:
            routes[event.agent_index].append(event.spot_position)
    return routes


def supply_order_from_result(day_result, kinds) -> list[int]:
    """その日の結果から、補給車が実際に迎えた順番を取り出す。

    **なぜ必要か**

    順路を磨く仕組みは巡回車しか見ていませんでした。補給車の
    動きは貪欲法まかせで、**誰も手を入れていませんでした。**

    実戦1687の1日目で、補給車は74歩のうち5歩しか動かず、補給は
    2回だけ。実戦1685でも4日目に39%、5日目に56%を止まって
    過ごしています。補給車は燃料を使わないので、止まっている
    歩数はそのまま無駄です。翌日の開始燃料は0.61→0.49と枯れ、
    玉数も27・30と落ちました。

    迎えた順番を取り出せば、**その順番も磨けます**。
    """

    order: list[int] = []
    for event in sorted(day_result.refuels, key=lambda item: item.step):
        if order and order[-1] == event.agent_index:
            continue
        order.append(event.agent_index)
    return order


# 設計を磨くのに使う「試す手」の数。1手ごとに1日ぶんの再現を回します。
IMPROVE_BUDGET = 400
# 磨く設計の数。素の点数が高いものだけを選びます。
IMPROVE_SOURCES = 4

# 親プロセスが「磨き」に使う割合。
#
# 50 なら50回のうち49回を磨きに使います（attempt % 50 が 0 以外のとき）。
#
# **1件あたりの重さが2桁違います。** 実戦1703（32x32・7体）で
# 測ると
#
#   学習済みAIの案を1件作る : 約600ミリ秒
#   磨きの手を1つ試す       : **5ミリ秒**（毎秒200手）
#
# 32x32の実戦では1日の候補が863〜1590件しか作れていませんでした。
# 同じ時間を磨きに回せば万単位の手を試せます。実際、記録された
# 渋滞つきの再現では
#
#   磨きなし          : 玉366
#   磨き400手         : 玉405  ← 実戦の390を超える
#
# でした。広く探すほうは子プロセス11個がやっているので、親は
# ほぼ磨きに専念させます。
#
# 手を出し切ったらふつうの探索に戻るので、無駄に回りません。
# 0 にすると磨きを止めます。
POLISH_EVERY = 50

# 子プロセス1つに任せる「磨き」の手数と、何個に分けるか。
#
# 子は11個あります。最良案が変わるたびに、開始位置をずらした
# 仕事を POLISH_WORKER_JOBS 個だけ割り込ませます。1個 60手なら
# 24個で1,440手。親だけなら締切までに400手ほどなので、3倍以上に
# なります。実測では手数がそのまま効きます。
#
#   手数400 : 玉1379 / 日別757
#   手数800 : 玉1422 / 日別761
#   手数3200: 玉1440 / 日別762
# **1本を長くします。**
#
# 磨きは「少し粘って出てこなければ崩して登り直す」形になりました
# （`hexa/improve.py`）。粘る長さは形の大きさに比例し、32x32なら
# 約340手です。**1本400手では崩せて1回だけ**で、登り直しの効果が
# ほとんど出ません。
#
# 同じ総手数でも、長い1本のほうが強いことを実測しました
# （実戦6試合・記録された渋滞つき・総手数6000）:
#
#   1500手×4本 : 玉1458
#   6000手×1本 : 玉1486
#   3000手×2本 : 玉1495
#
# そこで1本2500手（32x32で約12秒）にし、本数を16に減らします。
# 締切が来たら途中でも、そこまでの最良を返します。
POLISH_WORKER_BUDGET = 2500
# 手の種類は8つあります。種類の数で割り切れる数にして、
# どの種類にも同じだけ子をあてます。崩し方の始まりもずらします。
POLISH_WORKER_JOBS = 16
# 1回の割り込みで 32 × 400 = 12,800手。手の種類8つに4つずつ
# あたります。

# **列に「磨き」をこれだけ残しておきます。**
#
# 子の列は、1日のはじめに積んだ「ゆらぎ探索」でいっぱいです。
# 磨きは最良案が変わったときだけ割り込ませていたので、最良案が
# 落ち着くと列から磨きが消え、**子は残りの時間ずっとゆらぎ探索**を
# していました。実戦ログでも、採用された案は7日のうち6日が「磨き」
# なのに、24x24の1日の候補は610件（16x16は8278件）しかありません。
#
# 同じ持ち時間でどちらが得かを、実戦1891（24x24）の盤面で測りました
# （1通り20秒・親1つ）:
#
#         出発点   ゆらぎ探索      磨き
#   1日目  玉35    玉43(7945回)  **玉46**(15607手)
#   2日目  玉35    玉37(9411回)  **玉40**(14581手)
#   3日目  玉25    玉27(11959回) **玉30**(8874手)
#   4日目  玉32    玉33(9339回)  **玉36**(14657手)
#
# **どの日も磨きが+3玉**。1手あたりも磨き1.36ms に対しゆらぎ7.67ms
# と5.6倍安いので、同じ秒数で何倍も試せます。
#
# そこで、回収のたびに列の磨きを数え、少なければ継ぎ足します。
# ゆらぎ探索は列の後ろに残るので、磨きが尽きたときだけ走ります。
POLISH_QUEUE_FLOOR = 12
# 継ぎ足すときの本数。手の種類8つに同じだけあてます。
POLISH_QUEUE_REFILL = 16
# **ゆらぎ探索も少しだけ残します。**
#
# 磨きは出発点の近くしか探しません。実戦1894の2日目は
# 「S1-AI-1+探索0.55」が採用されており、まれに当たります。
#
# ただし**仕事の長さがまるで違います**。磨き1本は2500手（24x24で
# 約3.4秒）、ゆらぎ1かたまりは4案（約0.03秒）。1本ずつ交互に
# しても時間の99%は磨きになります。1手あたりの比（磨き1.36ms /
# ゆらぎ7.67ms）から、磨き1本につきゆらぎ25かたまりを挟むと、
# 時間でおよそ 8:2 になります。盤面が重くなると両方とも同じ
# ように遅くなるので、この比は盤面によらずほぼ保たれます。
POLISH_NOISE_GAP = 25
# **子に配る「形のちがう案」の数。**
#
# 磨きは登り切ると止まります。実戦1891（24x24）の盤面で、貪欲1つ
# から磨くと20秒でも100秒でも玉46/40/30/36 と**まったく同じ**
# （手数は18,403手まで伸びています）。同じ谷を登り切っているので、
# 時間を足しても1玉も増えません。
#
# 出発点を5か所に分けて20秒ずつにすると、1日目は玉46→**50**、
# 3日目は玉30→**31** と越えられました。そこで最良案だけでなく、
# 車ごとの店の並びが**ちがう**良い案を順に配ります。
POLISH_SEEDS = 5
# 形を選ぶときに見る、直近の案の数。全部を並べ替えると1日に何千件
# にもなるので、新しいほうから見ます。
POLISH_SEED_POOL = 400
# **設計で作った案も、磨きの出発点の候補に必ず入れるか。**
#
# `candidates[-POLISH_SEED_POOL:]`（直近400件）から出発点を選んでいま
# したが、**設計の案は1日の最初に作られる**ので、候補が400件を超えた
# 時点で**出発点の候補から丸ごと外れます**。32x32の1日は候補が900件を
# 超えるので、実質いつも外れていました。
#
# 今日、設計の家族を6つ作って29試合で測ったら全部負けました
# （近づく手211／夜合流190／待ち伏せ178／逆回り201／在庫の重み191。
# 基準163）。けれど**どの家族も出発点には選ばれていません**。つまり
# 「その設計から磨いたらどうなるか」は、まだ1度も測れていません。
#
# 設計の案は1日に数十件しかないので、全部足しても出発点選びは重く
# なりません。
SEED_KEEPS_DESIGNS = False

# **昨日いちばん良かった順路を、今日の候補に加えるか。**
#
# 店の場所は毎日同じで、在庫も毎朝満タンに戻ります。だから昨日
# 1日ぶんの時間をかけて磨いた「車ごとの店の並び」は、今日もそのまま
# 良い形です。毎朝それを捨てて素案から登り直すのは、1手3msかかる
# 32x32ではとくに損でした。
#
# 出発点が違っても計画は作れます（planner はいまいる場所から順路を
# たどるだけ）。良くなければ採点で落ちるので、候補に1つ増えるだけの
# 害のない足し方です。
WARM_START_ROUTES = True


def make_repair(problem, finder, kinds, positions, fuels, road_status,
                day_steps, acquired_brands, refuels=0):
    """崩したあとの順路を、差し込みで埋め直す関数を作る。

    **実測で使っていません。** 崩したあとは `PlannedRoutePolicy` の
    貪欲な埋め戻しに任せるほうが良かったためです。

    16試合を記録された渋滞つきで再現（手数3000・学習済みAIなし）:

      貪欲で埋め直す     : 玉3857 / 日別1683
      差し込みで埋め直す : 玉3849 / 日別1679

    内訳を見ると、燃料の少ない24x24では良くなり（1791440423_7は
    286→293）、**16x16では悪くなります**（1791435518_1は92→84、
    1791439853_6は73→66）。店が近い地図では、差し込みで歩数いっぱいまで
    詰めるより、その場で近い店を拾うほうが取りこぼしが少ないようです。

    仕組みは残してあります。地図の大きさで使い分ければ効く余地が
    あります（`RoutePolisher(repair=...)` に渡すだけで有効になります）。
    """

    from hexa.allocate import design_insert

    def repair(routes):
        return design_insert(
            problem, finder, kinds, positions, fuels, road_status, day_steps,
            brand_weight=PLANNED_BRAND_WEIGHTS[-1],
            acquired_brands=acquired_brands, refuels=refuels, start=routes,
        )

    return repair


# 先に弾くときに許す「歩数の超過」。1.0 でぴったり、1.5 で1.5倍まで通す。
PREFILTER_SLACK = 1.25


def make_prefilter(problem, finder, kinds, positions, fuels, road_status,
                   day_steps):
    """**歩数が足りない順路を、再現する前に弾く関数を作る。**

    1日ぶんの再現は1件5ミリ秒ですが、隣どうしの最短距離を足して
    歩数を見るだけなら10マイクロ秒です。500倍ちがうので、先に
    弾けばそのぶん本物の手を試せます。

    戻り値は「通るなら合計歩数、通らないなら None」です。合計歩数は
    「訪問数が増えないのに歩数が増える手」を捨てるのにも使います。

    **燃料は見ません。**

    補給で満タンになるので、順路だけでは足りるかどうか決まりません。
    「満タン2回ぶんを超えたら無理」という上限で弾いてみましたが、
    歩数の多い日（1日177歩など）には**正しい順路まで弾いて**
    しまいました。実戦1709では日別累計が84から79に落ちています。

    そもそも1歩で使う燃料は最大2（道路）なので、必要な燃料は
    必ず「歩数×2」以下です。歩数の検査を通った順路は燃料の上限も
    自動的に満たすので、この検査は何も足しません。
    """

    from hexa.allocate import route_cost

    cache: dict[int, object] = {}
    # **車ごとの順路の歩数を覚えます。**
    #
    # 磨きの手はたいてい1台の順路しか変えません。それなのに毎回
    # 全車ぶんを数え直していました。第1958（32x32・巡回6台）の実測で
    # 1手に41回 `route_cost` を呼び、1手の7%を使っています。
    #
    # 鍵は (車の番号, 店の並び) です。中身が同じなら歩数も同じなので、
    # 答えは変わりません。
    known: dict[tuple, int] = {}

    # **少しの超過は通します。**
    #
    # 順路が歩数を超えていても、planner は途中で打ち切って貪欲に
    # 切り替えるので、使える計画になることがあります。ぴったりで
    # 弾くと、そういう案を捨ててしまいます（実戦1709で日別累計が
    # 84→79に落ちました）。
    limit = int(day_steps * PREFILTER_SLACK)

    def feasible(routes):
        """通るなら合計歩数、通らないなら None を返す。"""

        total = 0
        for index, spots in routes.items():
            if not spots:
                continue
            mark = (index, tuple(spots))
            cost = known.get(mark)
            if cost is None:
                cost = route_cost(finder, road_status, positions[index],
                                  spots, cache)
                if len(known) >= 60_000:
                    known.clear()
                known[mark] = cost
            if cost > limit:
                return None
            total += cost
        return total

    return feasible


def improved_candidates(problem, finder, planner, kinds, positions, fuels,
                        road_status, day_steps, acquired_brands, day_index=0,
                        steps_ahead=0, next_steps=0, budget=IMPROVE_BUDGET,
                        bases=None):
    """設計を実際の点数で磨いた候補を返す。

    出発点は `_planned_route_policies` が作った設計です。磨く手は
    `hexa/improve.py` にあります。
    """

    from hexa.improve import improve_routes

    found = []
    sources = bases
    if sources is None:
        sources = []
        for label, policy in _planned_route_policies(
            problem, finder, kinds, positions, fuels, road_status, day_steps,
            acquired_brands, next_steps=next_steps,
        ):
            if isinstance(policy, PlannedRoutePolicy):
                sources.append((label, policy.routes, policy.supply_route))
    # **まず素の設計を採点し、良いものだけを磨きます。**
    # 設計は16通り作りますが、全部を磨くと1日ぶんが何千回の再現に
    # なります。素の点数が低い設計を磨いても追いつきません。
    if len(sources) > IMPROVE_SOURCES:
        scored = []
        for label, routes, supply_route in sources:
            candidate = _build_candidate(
                label, problem, planner, kinds, positions, fuels, road_status,
                day_steps, PlannedRoutePolicy(problem, routes, supply_route),
                acquired_brands, day_index=day_index,
                steps_ahead=steps_ahead, next_steps=next_steps,
            )
            if candidate is not None:
                scored.append((candidate.key, label, routes, supply_route))
        scored.sort(key=lambda item: item[0], reverse=True)
        sources = [(label, routes, supply_route)
                   for _key, label, routes, supply_route
                   in scored[:IMPROVE_SOURCES]]
    for label, routes, supply_route in sources:
        def evaluate(trial, order, label=label, verify=False):
            return _build_candidate(
                f"磨き{label}", problem, planner, kinds, positions, fuels,
                road_status, day_steps,
                PlannedRoutePolicy(problem, trial, order),
                acquired_brands, day_index=day_index,
                steps_ahead=steps_ahead, next_steps=next_steps,
                verify=verify,
            )

        polished, polished_supply, candidate = improve_routes(
            problem, kinds, routes, evaluate, budget=budget,
            supply_order=supply_route,
            prefilter=make_prefilter(problem, finder, kinds, positions,
                                     fuels, road_status, day_steps),
        )
        if candidate is None:
            continue
        # **採用する案は必ず公式シミュレータに通し直します。**
        # 磨きの最中は通していないので、ここで確かめます。
        exact = evaluate(polished, polished_supply, verify=True)
        found.append(exact if exact is not None else candidate)
    return found


# 1日先まで見て選び直す案の数。**0 で止めてあります**（理由は
# `_rollout_pick` の説明に実測つきで書いてあります）。
ROLLOUT_CANDIDATES = 0


def _rollout_pick(problem, planner, kinds, candidates, road_status,
                  next_steps, acquired_brands, day_index, best):
    """**完全に同点の案だけ**、1日先まで見て選び直す。

    **実測で止めました（`ROLLOUT_CANDIDATES = 0`）。**

    玉数まで含めて同点の案は毎日たくさん出ます。その中から
    「終わりの燃料が多い案」ではなく「明日の出だしが良い案」を
    選べば得なはずでした。16試合を記録された渋滞つきで再現
    （手数3000）:

      使わない           : 玉3998 / 日別1683
      同点のときだけ使う : 玉3998 / 日別1683  ← **1点も動かない**
      玉数を今日＋明日に : 玉3975（32x32と1709は伸びたが、ほかの
                           9試合で落ち、1699では日別も144→143）

    同点のときだけに絞ると、16試合・140日を通して**結果が1つも
    変わりませんでした**。明日の見積りは貪欲1回なので、実際に明日
    打つ手（磨き）よりずっと弱く、同点の案のあいだで差がつかない
    のだと見ています。

    そして**効き目が0なら、入れるのは損です**。本番の持ち時間は
    時計で決まるので、1日の再現に使った時間はそのまま磨きの手数を
    減らします。手数を2〜3倍にすると玉3998→4035になる（頭打ちでは
    ない）ことが分かっているので、時間は磨きに回します。
    """

    if best is None:
        return None
    # 玉数・余力まで同点の案だけを見ます（公式の順位は崩しません）。
    same = [candidate for candidate in candidates
            if candidate.key[:8] == best.key[:8]]
    top = sorted(same, key=lambda item: item.key,
                 reverse=True)[:ROLLOUT_CANDIDATES]
    if len(top) < 2:
        return None
    scored = []
    for candidate in top:
        brands = acquired_brands | {
            event.brand for event in candidate.day_result.acquisitions
        }
        gain = 0
        try:
            ahead = _build_candidate(
                "先読み", problem, planner, kinds,
                list(candidate.day_result.positions),
                list(candidate.day_result.fuels), road_status, next_steps,
                GreedyPolicy(problem), brands, day_index=day_index + 1,
                verify=False,
            )
            if ahead is not None:
                gain = len(ahead.day_result.acquisitions)
        except Exception:
            gain = 0
        scored.append(((gain, candidate.key[10]), candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    winner = scored[0][1]
    return winner if winner is not best else None


def _candidate_from_plan(problem, kinds, positions, fuels, road_status,
                         day_steps, action_plan, label, acquired_brands,
                         steps_ahead, next_steps, finder=None):
    """子プロセスが返した行動計画を、こちらで採点し直して候補にする。"""

    try:
        day_result = simulate_day(
            problem, kinds, tuple(positions), tuple(fuels),
            action_plan, day_steps, road_status, record_trace=True,
        )
    except Exception:
        return None
    return Candidate(
        label, action_plan, day_result,
        # **採点の材料は親の案とそろえます。**
        # finder を渡し忘れると「明日すぐ行ける店の数」が常に0に
        # なり、子の案だけ不利に採点されます。
        _candidate_key(day_result, acquired_brands, problem.fuel_limit,
                       kinds, steps_ahead, problem, finder, next_steps,
                       road_status),
    )


def _start_parallel_day(problem, brains, kinds, positions, fuels, state,
                        day_steps, day_index, acquired_brands,
                        steps_ahead, next_steps, deadline, log,
                        seed_for_day: int = 0):
    """1日分の計画づくりを子プロセスに投げる。すぐ戻ります。

    **親を待たせません。** 投げたあと親は自分の探索を続け、
    その合間に `harvest()` で終わったぶんを受け取ります。
    こうすると締切まで全プロセスが動き続けます。
    """

    if not _WARM_FILES:
        return None
    try:
        from hexa.parallel import start_day_search, usable_workers

        workers = usable_workers()
        if workers <= 1:
            return None
        if deadline - time.monotonic() < 1.0:
            return None
        payload = {
            "kinds": [int(k) for k in kinds],
            "positions": list(positions),
            "fuels": list(fuels),
            "roads": {str(k): int(v) for k, v in state.road_status.items()},
            "acquired": sorted(acquired_brands),
            "day_steps": day_steps,
            "day_index": day_index,
            "steps_ahead": steps_ahead,
            "next_steps": next_steps,
        }
        # **探索の強さを混ぜて配ります。**
        #
        # 以前は「方策を一巡してから次の強さへ」という順でした。
        # AI が30体なら最初の32件がすべて探索0.00になり、
        # 学習済みの方策をそのまま使った**似た案ばかり**が並びます。
        # 実戦で候補4,176件あったのに、手元で強さを混ぜた400案より
        # 8玉少ない、という結果になっていました。
        #
        # 1件ごとに強さを変えて、最初から幅広く探します。
        # **強さは順番に決め打ちします。**
        #
        # 以前は1件ごとに乱数で選んでいました。そのため同じ条件でも
        # 結果が変わり、実測で日別が55〜56、玉が82〜91と揺れます。
        # 本番は一度きりなので、運に左右される設計は誤りです。
        #
        # 順番に割り当てれば、何案作れたかで結果が決まり、
        # 同じ案数なら必ず同じ結果になります。
        noise_levels = (0.0, 0.05, 0.10, 0.20, 0.35, 0.55)
        jobs = []
        # **仕事は多めに作ります。**
        # 実戦で並列が 6000 ちょうどで頭打ちになっていました
        # （16x16の短い日）。締切で打ち切られるので余らせても害はなく、
        # 足りないと手の空いた子が遊びます。
        for index in range(40000):
            # 最初の一巡だけは素の方策（探索なし）を必ず試す。
            if index < len(brains) + 2:
                noise = 0.0
            else:
                # 方策を一巡するごとに、次の強さへ進む。
                noise = noise_levels[
                    (index // max(1, len(brains) + 2)) % len(noise_levels)
                ]
            slot = index % (len(brains) + 2)
            if slot == len(brains) + 1:
                jobs.append(("系列優先", -2, 0.0, 7919 * index))
            elif slot == len(brains):
                label = "貪欲" if noise == 0 else f"貪欲+ゆらぎ{noise:.2f}"
                jobs.append((label, -1, noise, 104729 * index))
            else:
                label = f"{brains[slot].name}+探索{noise:.2f}"
                jobs.append((label, slot, noise, 7919 * index))
        problem_path, population_path = _WARM_FILES
        # 締切の0.5秒前まで回す（提出の時間だけ残す）。
        return start_day_search(
            problem_path, population_path, payload, jobs, deadline - 0.5, workers
        )
    except Exception:
        return None


def _run_parallel_trials(problem, brains, supply_counts, started, budget,
                         seed_base, log):
    """事前試走の一部を子プロセスに投げる。

    戻り値は (試した数, 最良の (スコア, 種別, ラベル) か None)。
    失敗したら (0, None) を返すので、呼び出し側はそのまま続けられます。
    """

    import tempfile

    from hexa.ai import save_population
    from hexa.parallel import run_trials, usable_workers

    workers = usable_workers()
    if workers <= 1:
        log("  並列: このPCは1コアなので使いません。")
        return (0, None)

    # 持ち時間の8割を並列に使います。親は並列の結果を待たずに
    # 自分の試走を進めるので、長く取っても無駄になりません。
    deadline = started + budget * 0.80
    left = deadline - time.monotonic()
    if left < 3.0:
        log(f"  並列: 残り{left:.1f}秒では足りないので使いません。")
        return (0, None)

    # 組み合わせは多めに作ります。締切が来たところで打ち切られるので、
    # 余らせても害はありません。少ないと並列の手が空きます。
    jobs = []
    for index in range(4000):
        supply_count = supply_counts[index % len(supply_counts)]
        if brains:
            brain_index = index % len(brains)
            noise = (0.0, 0.05, 0.15, 0.35)[(index // max(1, len(brains))) % 4]
            label = f"{brains[brain_index].name}+探索{noise:.2f}"
        else:
            brain_index = -1
            noise = 0.0
            label = "貪欲"
        # 台数を1周するごとに席を1つずらします（席も探すため）。
        seat_list = supply_seat_choices(problem.agent_count, supply_count)
        seats = seat_list[(index // max(1, len(supply_counts)))
                          % len(seat_list)]
        jobs.append((label, brain_index, supply_count, noise,
                     seed_base + index, seats))

    try:
        # **すでに起こしてあるプールのファイルをそのまま使います。**
        # 毎回 tempfile を作ると、プールも作り直しになってしまいます。
        if _WARM_FILES:
            problem_path, population_path = _WARM_FILES
        else:
            folder = tempfile.mkdtemp(prefix="hexa_trial_")
            problem_path = str(Path(folder) / "problem.json")
            Path(problem_path).write_text(
                json.dumps(problem.to_json_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            population_path = ""
            if brains:
                population_path = str(Path(folder) / "population.json")
                save_population(brains, Path(population_path))
        results = run_trials(
            problem_path, population_path, jobs, deadline, workers
        )
    except Exception as error:
        log(f"  並列の事前試走は使えませんでした（{type(error).__name__}）。")
        return (0, None)

    if not results:
        log("  並列: 結果が返りませんでした（子プロセスの起動に失敗した可能性）。")
        return (0, None)

    usable = [item for item in results if len(item) >= 4]
    if not usable:
        log("  並列: 使える結果がありませんでした。")
        return (0, None)
    best = max(usable, key=lambda item: item[0])
    score, label, supply_count, seats = best[:4]
    kinds = kinds_for_seats(problem.agent_count, seats)
    seat_note = (
        "" if supply_count <= 0 else
        " / 補給車の席" + ",".join(
            str(problem.agent_positions[index]) for index in seats)
    )
    log(f"  並列で {len(results)}通り（{workers}プロセス）")
    return (len(results),
            (tuple(score), kinds,
             f"{label} / 補給車{supply_count}台{seat_note}"))



def kinds_for_seats(agent_count: int, seats) -> tuple:
    """`seats` の席を補給車、残りを巡回車にした種別の並びを作る。

    **なぜ席を選べるのか**

    種別は車ごとに提出します（`POST /agent` に `[0,1,0,1]` のような
    配列）。並びは配られた初期位置に対応し、初期位置は全チーム共通で
    動かせません（Q2 A38）。つまり**どの出発点を補給車にするかは
    こちらの自由**です。

    これまでは必ず**最後の席**を補給車にしていました。台数だけ探して
    席は探していなかったので、たまたま当たっているだけでした。

    実測（第1981・16x16・4体・記録された渋滞つき・1日8秒）:

      出発点252（0番） 日別78 玉179
      出発点 66（1番） 日別78 玉183
      出発点180（2番） 日別78 玉178
      出発点118（3番） 日別78 **玉184**  ← これまでの置き方

    同じ日別で**玉に6個（3.3%）の差**がありました。
    """

    picked = set(seats)
    return tuple(AgentKind.SUPPLY if index in picked else AgentKind.TOURING
                 for index in range(agent_count))


# **4体の盤面では、席の試走に予測力がありません。**
#
# 本選1回戦の実問題6つ（16x16・4体）で、席を1つずつ固定して
# 磨きつき（手数800）で通した本当の成績:
#
#   | 試合 | 席0 | 席1 | 席2 | 席3 |
#   |---|---|---|---|---|
#   | 第1 | 32 | 33 | 34 | **35** |
#   | 第2 | 39 | **42** | 40 | **42** |
#   | 第3 | **37** | **37** | 32 | 35 |
#   | 第4 | 37 | **39** | 36 | 36 |
#   | 第5 | 36 | 36 | **37** | **37** |
#   | 第6 | 35 | 36 | 37 | **38** |
#
# これに対し、選び方ごとの日別合計:
#
#   | 選び方 | 日別合計 |
#   |---|---|
#   | 貪欲試走（本番がいま使う形） | 216 |
#   | 案そろえ試走（毎日全家族＋磨き1手） | 216 |
#   | でたらめに選ぶ（4席の平均） | 219.5 |
#   | **後ろ詰めに固定** | **223** |
#   | 席1に固定 | 223 |
#   | 毎試合の最良（届かない上限） | 228 |
#
# **試走はでたらめより悪い**ので、4体では席を振りません。
# 生成した24x24・32x32（5体・7体）では逆に、探したほうが
# 日別1634/1634・玉3826で、後ろ詰め固定（1624/3776）より上でした。
# 席の差が大きい盤面では試走が効き、小さい盤面では雑音が勝ちます。
SEAT_TRIAL_MIN_AGENTS = 5


def supply_seat_choices(agent_count: int, supply_count: int,
                        limit: int = 12) -> list[tuple]:
    """試す「補給車の席」の組み合わせ。

    1台なら席の数だけ（4体で4通り・7体で7通り）。2台以上は組み合わせ
    が増えるので、**端から順に散らした形**に絞ります（全部試すと
    事前試走の時間に入りません）。

    **体が少ない盤面では後ろ詰めだけ**にします（理由は
    `SEAT_TRIAL_MIN_AGENTS` に実測つきで）。
    """

    import itertools

    if supply_count <= 0:
        return [()]
    if supply_count >= agent_count:
        return [tuple(range(agent_count))]
    if agent_count < SEAT_TRIAL_MIN_AGENTS:
        return [tuple(range(agent_count - supply_count, agent_count))]
    whole = list(itertools.combinations(range(agent_count), supply_count))
    if len(whole) <= limit:
        return whole
    # 多いときは「後ろ詰め」「前詰め」「等間隔」を優先して選びます。
    picked: list[tuple] = []

    def add(got):
        if got not in picked:
            picked.append(got)

    add(tuple(range(agent_count - supply_count, agent_count)))
    add(tuple(range(supply_count)))
    step = max(1, agent_count // supply_count)
    add(tuple(sorted((index * step) % agent_count
                     for index in range(supply_count))))
    for got in whole:
        if len(picked) >= limit:
            break
        add(got)
    return picked[:limit]


def _trial_better(score, supply_count, best_key, best_supply,
                  margin=None) -> bool:
    """事前試走で、この結果を採用すべきか。

    **同点なら巡回車が多いほうを選びます。**

    実戦1218で、補給1台と2台がどちらも日別139で並びました。
    玉数は1台が213、2台が175です。ところが単純な `>` 比較だと
    「先に見つけたほう」が残るため、2台が選ばれていました。

    本番では補給1台なら日別144/玉272（1位は144/276）が出るのに、
    2台では玉227まで落ちます。**台数を外すと全日に響きます。**

    玉で差がつくならそれで決まりますが、玉まで同じなら
    巡回車が多いほうが取りに行けるので有利です。
    """

    if best_key is None:
        return True
    left = _trial_discount(score, supply_count, margin)
    right = _trial_discount(best_key, best_supply, margin)
    if left != right:
        return left > right
    # 完全に同点なら、補給車が少ない（＝巡回車が多い）ほうを採る。
    return supply_count < best_supply


# **補給車の台数を1から動かすときに要求する上積み（玉数の割合）。**
#
# 事前試走は**貪欲**で回します。貪欲は燃料を無駄に使うので、
# 補給車が多いほど得に見えます。ところが本番は磨いた順路で走るため
# 燃料の無駄が小さく、補給車は1台で足ります。この食い違いで、
# 実戦1958（32x32・7体）は巡回5+補給2を選んでいました。
#
# 記録された渋滞つきで、台数だけ変えて9日通した実測:
#
#   実戦1958 (32x32 燃料137)  巡回6+補給1 **玉485/日別198**
#                             巡回5+補給2   玉426/日別197  ← 実戦の選択
#                             巡回4+補給3   玉349/日別195
#   実戦1703 (32x32 燃料113)  巡回6+補給1 **玉413/日別221**
#                             巡回5+補給2   玉322/日別220
#                             巡回4+補給3   玉311/日別208
#   実戦1960 (24x24 燃料147)  巡回4+補給1 **玉179/日別137**
#                             巡回3+補給2   玉160/日別124
#                             巡回2+補給3   玉114/日別105
#
# **3試合とも1台が圧勝**で、差は11〜22%ありました。巡回車が1台減る
# と取りに行く力がそのまま減るので当然です。
#
# **0台はもっと悪い。** 同じ1958の盤面で巡回7+補給0を通すと
# **玉173/日別92**（1台なら485/198）。補給がないと数日で全車が
# 燃料切れになり、試合が終わります。
#
# つまり山は**1台に鋭く尖って**います。ところが事前試走は実戦で
# 2台を、学習済みAIなしの手元では0台を選びました。**推定器そのものが
# 当てになりません。** そこで1台を既定とし、そこから動かすときは
# どちらの向きにも「これだけ上回らなければ選ばない」という上積みを
# 課します。実測の差（11%以上）より少し内側の15%にしました。
#
# 種類と日別は割り引きません。公式の上位条件なので、台数の見立て
# 違いで捨ててよいものではありません。
#
# **「燃料が逼迫する盤面では2台」という例外は、実測で否定されました。**
#
# 実戦1970（32x32・燃料124・1日191歩の日あり）だけ2台が勝つように
# 見え、いちど「逼迫度で切り替える」仕掛けを入れました。ところが
# **手数を増やすと逆転します**。
#
#   1970  手数 500 : 巡回6+補給1 玉394 / 巡回5+補給2 玉410
#   1970  手数1500 : 巡回6+補給1 **玉452** / 巡回5+補給2 玉407
#
# **巡回車が多い構成ほど、良くするのに探索が要ります。** だから
# 手数の少ない比べ方は、巡回車の少ない構成を不当に有利に見せます。
# 事前試走（貪欲1回）はその極端な場合で、これが台数を外す本当の
# 理由でした。燃料の逼迫は関係ありません。
#
# 5盤面すべてで1台が正解です（1960 / 1958 / 1703 / 1966 / 1970）。
# 例外なしで1台を既定にします。
SUPPLY_EXTRA_MARGIN = 0.15


def supply_margin_for(problem) -> float:
    """その盤面で、台数を1から動かすのに要求する上積み。

    盤面によらず一定です。「燃料が逼迫していれば2台」という例外を
    試しましたが、手数を増やすと消えました（上の実測）。
    """

    return SUPPLY_EXTRA_MARGIN


def _trial_discount(score, supply_count, margin=None):
    """補給車が1台でない結果を、上積みのぶんだけ割り引く。

    `score` は (種類, 日別累計, 玉数) です。割り引くのは玉数だけに
    します。種類と日別は公式の上位条件で、台数の見立て違いで
    捨ててよいものではありません。
    """

    if margin is None:
        margin = SUPPLY_EXTRA_MARGIN
    extra = abs(supply_count - 1)
    if not extra or not margin or len(score) < 3:
        return score
    return (score[0], score[1],
            score[2] * (1.0 - margin * extra)) + tuple(score[3:])


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
    # (名前, 方策, 補給車の台数, 補給車の席)
    trials: list[tuple[str, object, int, tuple]] = []
    greedy = GreedyPolicy(problem)
    # **席も試します。** 種別は車ごとに提出するので、どの出発点を
    # 補給車にするかは自由です（`kinds_for_seats` に実測つきの理由）。
    # まず貪欲で「台数×席」を一通り押さえます。
    for supply_count in supply_counts:
        for seats in supply_seat_choices(problem.agent_count, supply_count):
            trials.append(("貪欲", greedy, supply_count, seats))
    # 試走は「補給車の台数を決める」ためのものなので、
    # 1回1回の精度より回数を優先する。作戦を絞ると1試走が軽くなり、
    # 同じ持ち時間で何倍も試せます。
    for index, brain in enumerate(brains):
        policy = Policy(brain, problem, random.Random(31 + index), 0.0,
                        route_budget=TRIAL_ROUTE_BUDGET)
        for supply_count in supply_counts:
            # 学習済みAIは「これまでの置き方（後ろ詰め）」だけにします。
            # 席の総当たりは貪欲で済ませてあるので、ここは台数の確認です。
            trials.append((brain.name, policy, supply_count,
                           tuple(range(problem.agent_count - supply_count,
                                       problem.agent_count))))

    seed_base = 104729
    # 盤面ごとの上積み（燃料が逼迫していれば0）。
    margin = supply_margin_for(problem)
    best_kinds = tuple(AgentKind.TOURING for _ in range(problem.agent_count))
    best_key: tuple[int, int, int] | None = None
    # 同点のときに巡回車の多いほうを選ぶため、採用中の補給車の数も覚える。
    best_supply = 99
    best_label = ""
    done = 0
    longest = 0.0

    # **並列はいちばん最初に動かします。**
    #
    # 以前は総当たりのあとに置いていたため、呼ばれる時点で持ち時間を
    # ほぼ使い切っており（42秒の試合で残り2.0秒）、一度も動きませんでした。
    # 子プロセスは同じ問題を自分で解くので、先に投げておけば
    # そのあいだ親は単体の試走を続けられます。
    if PARALLEL_TRIALS:
        extra, found = _run_parallel_trials(
            problem, brains, supply_counts, started, budget, seed_base, log
        )
        done += extra
        found_supply = (
            sum(1 for k in found[1] if k == AgentKind.SUPPLY)
            if found is not None else 0
        )
        if found is not None and _trial_better(
            found[0], found_supply, best_key, best_supply, margin
        ):
            best_key = found[0]
            best_supply = found_supply
            best_kinds = found[1]
            best_label = found[2]

    for label, policy, supply_count, seats in trials:
        elapsed = time.monotonic() - started
        # 1回の試走にかかった最長時間を見て、次が入らないなら打ち切る。
        if done > 0 and elapsed + longest > budget:
            break
        kinds = kinds_for_seats(problem.agent_count, seats)
        trial_started = time.monotonic()
        score = _rollout_match(problem, planner, kinds, policy)
        longest = max(longest, time.monotonic() - trial_started)
        done += 1
        if score is None:
            continue
        if _trial_better(score, supply_count, best_key, best_supply,
                         margin):
            best_key = score
            best_kinds = kinds
            best_supply = supply_count
            seat_note = (
                "" if supply_count <= 0 else
                " / 補給車の席" + ",".join(
                    str(problem.agent_positions[index]) for index in seats)
            )
            best_label = f"{label} / 補給車{supply_count}台{seat_note}"

    # 一通り試して時間が余っていたら、探索を入れた試走を続ける。
    # 補給車の台数は試合開始時に1回しか決められないので、
    # 持ち時間はできるだけ使い切ったほうが良い。
    # 以前は44通りで打ち切って30秒以上余らせていた。
    # **ここだけ複数のコアを使います。**
    #
    # 1日の計画づくりは「25案を超えると結果が変わらない」と実測済みなので
    # 並列にしても意味がありません。一方この事前試走は補給車の台数を
    # 決める1回限りの判断で、決勝では231通りのうち38通りしか試せて
    # いませんでした。**台数を外すと全日に響く**ので、ここを厚くします。
    #
    # Windows は fork が使えず spawn なので、子プロセスごとに Python を
    # 起動し直します。問題と重みはファイル経由で渡し、1つの子に
    # いくつかの組み合わせをまとめて渡して起動の回数を減らしています。
    # 失敗したら黙って1プロセスのまま続けます。
    attempt = 0
    while time.monotonic() - started + longest < budget and attempt < 100000:
        attempt += 1
        supply_count = supply_counts[attempt % len(supply_counts)]
        # **席も回します。** 台数を1周するごとに席を1つずらすので、
        # 時間が余るほど「台数×席」が満遍なく試されます。
        seat_list = supply_seat_choices(problem.agent_count, supply_count)
        seats = seat_list[(attempt // max(1, len(supply_counts)))
                          % len(seat_list)]
        if brains:
            brain = brains[attempt % len(brains)]
            noise = (0.05, 0.15, 0.35, 0.55)[(attempt // max(1, len(brains))) % 4]
            policy = Policy(brain, problem, random.Random(seed_base + attempt), noise,
                            route_budget=TRIAL_ROUTE_BUDGET)
            label = f"{brain.name}+探索{noise:.2f}"
        else:
            policy = GreedyPolicy(problem)
            label = "貪欲"
        kinds = kinds_for_seats(problem.agent_count, seats)
        trial_started = time.monotonic()
        score = _rollout_match(problem, planner, kinds, policy)
        longest = max(longest, time.monotonic() - trial_started)
        done += 1
        if score is not None and _trial_better(
            score, supply_count, best_key, best_supply
        ):
            best_key = score
            best_kinds = kinds
            best_supply = supply_count
            seat_note = (
                "" if supply_count <= 0 else
                " / 補給車の席" + ",".join(
                    str(problem.agent_positions[index]) for index in seats)
            )
            best_label = f"{label} / 補給車{supply_count}台{seat_note}"

    elapsed = time.monotonic() - started
    if best_key is not None:
        log(
            f"  事前試走 {done}通り（{elapsed:.1f}秒 / 持ち時間 {budget:.0f}秒）"
            f" 最良: {best_label} → 種類{best_key[0]} / 日別累計{best_key[1]} / 玉数{best_key[2]}"
        )
    else:
        log(f"  事前試走が1つも成立しませんでした。全て巡回車で進めます。")
    # 記録のために、試した数と最良の中身も返す。
    return best_kinds, done, elapsed, best_label, best_key


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
    record_log: bool = False,
    match_id: object = None,
) -> ProductionReport:
    # **監視は通信の確認より先に起こします。**
    # 待っているあいだの様子もビューアに流すためです。
    monitor = LiveMonitor()
    monitor.waiting("サーバーにつないでいます", True)
    ok, message = client.check_connection()
    monitor.waiting(message, ok)
    log(f"通信確認: {message}")
    if not ok:
        raise ConnectionError(message)

    # **待っているあいだも、ビューアに様子を流します。**
    # 以前は試合が始まるまで何も書かなかったので、開始3分前なのに
    # 「つながりません」と出ていました。
    problem, _raw = client.wait_for_setting(
        log=log,
        waiting=(monitor.waiting if monitor is not None else None),
    )
    log(f"試合設定: {problem.height}x{problem.width} / エージェント{problem.agent_count}体 / "
        f"{problem.day_count}日 / players={problem.players} / 燃料上限{problem.fuel_limit}")
    log(f"混雑基準{problem.busy_threshold} / 渋滞基準{problem.jammed_threshold} / "
        f"各日のステップ数 {list(problem.day_steps)}")

    # **子プロセスは事前試走より先に起こします。**
    #
    # Windows は spawn なので、11プロセスの起動に10秒かかります。
    # 試走のあとに起こしていたため、試走では毎回作り直しになり、
    # 1日の探索でも作り直していました。タスクマネージャーで
    # 「1日が終わるとCPUが落ち、しばらくして上がる」のはこれが原因です。
    #
    # ここで1つ作れば、事前試走も毎日の探索も同じものを使い回せます。
    # **子プロセスの起動にかかった時間も持ち時間から引きます。**
    #
    # 2026-10-11の準決勝で `agent kind submission is closed` を踏みました。
    # 24x24の持ち時間は90秒で、こちらが試走に使うのは63秒（70%）。
    # ところが**子プロセスの起動（Windowsで11個・約10秒）がこの63秒に
    # 入っていませんでした**。さらに試走は「次を始める前」に締切を見るので、
    # 最後の1回が長引くと数十秒はみ出します。合計すると90秒に迫ります。
    #
    # 起動にかかった実時間を引けば、そのぶん安全になります。
    warm_started = time.monotonic()
    if PARALLEL_TRIALS:
        _warm_up_workers(problem, brains, log)
    warm_seconds = time.monotonic() - warm_started

    budget = max(1.0, kind_time_budget(problem) - warm_seconds)
    if warm_seconds >= 1.0:
        log(f"  子プロセスの起動に{warm_seconds:.0f}秒かかりました"
            f"（そのぶん試走の持ち時間から引きます）")
    log(f"エージェント種別を決めています（持ち時間 {budget:.0f}秒）…")
    kinds, trial_count, trial_seconds, trial_label, trial_score = (
        choose_production_kinds(problem, brains, budget_seconds=budget, log=log)
    )
    supply_count = sum(1 for kind in kinds if kind == AgentKind.SUPPLY)
    # **種類の提出に失敗しても試合を続けます。**
    #
    # 2026-10-11の準決勝で、ここが包まれていなかったために
    # `HTTP 403 AccessTimeError: agent kind submission is closed`
    # で**試合そのものが止まりました**。種類の提出時間（90秒）は
    # 行動の提出（1日60秒）とは別の枠なので、**種類が間に合わなくても
    # 試合は戦えます**。要項（書式p.3 / Q2 A53）により、送らなかった
    # 場合はサーバー側で全員巡回車になります。
    #
    # こちらの思い込みで続けるとずれるので、**1日目の状態でサーバーが
    # 返してくる本当の種類に合わせ直します**（下の `state.agents` の
    # `kind`）。だからここでは止めずに進めるのが正解です。
    try:
        client.post_agent_kinds(kinds)
        log(f"提出しました: 巡回車{problem.agent_count - supply_count}台 / "
            f"補給車{supply_count}台")
    except ProconApiError as error:
        log(f"  ⚠ 種類を提出できませんでした（{error.code}）。"
            f"**試合は続けます。** 1日目にサーバーが返す種類に合わせます。")
    except Exception as error:
        log(f"  ⚠ 種類の提出で失敗しました（{type(error).__name__}）。"
            f"**試合は続けます。** 1日目にサーバーが返す種類に合わせます。")

    report = ProductionReport(problem=problem, agent_kinds=kinds)
    # 監視は本番の処理とは別。書き出しに失敗しても試合は止まりません。
    # （入れ物は、待ちに入る前にもう作ってあります。）
    monitor.start_match(problem, kinds)
    # 練習場のときだけ記録する。本番では1日60秒しかないので使わない。
    recorder = MatchLog(enabled=record_log)
    recorder.start(
        problem, kinds, match_id if match_id is not None else int(time.time()),
        trials=trial_count, trial_seconds=trial_seconds,
        best_label=trial_label, best_score=trial_score,
    )
    finder = PathFinder(problem)
    planner = DayPlanner(problem, finder)
    acquired_brands: set[int] = set()
    daily_total = 0
    noodle_total = 0
    predicted: tuple[list[int], list[int]] | None = None
    model = OpponentModel(problem)
    own_stays: list[dict[int, int]] = []
    # 昨日いちばん良かった順路と補給の順番（今日の出発点に使う）
    carried: tuple[dict[int, list[int]], list] | None = None

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
        # **通信が切れても諦めません。**
        #
        # 実戦で、4日目に名前解決が一瞬失敗しただけで試合を落としました
        # （日別144が取れるはずの試合で64）。無線や会場のネットワークは
        # 数秒で戻ることが多いので、間を置いて何度も試します。
        state = None
        for attempt in range(NETWORK_RETRIES):
            try:
                state = client.wait_for_day(day_index)
                break
            except Exception as error:
                last = error
                if attempt == 0:
                    log(f"  {day_index + 1}日目の状態を取れません"
                        f"（{type(error).__name__}）。繰り返し試します。")
                time.sleep(min(2.0, 0.3 * (attempt + 1)))
        if state is None:
            # どうしても取れないときだけ、全員待機を送って次の日へ。
            log(f"  {day_index + 1}日目は状態を取れませんでした。待機を送ります。")
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

        # **サーバーが返す種類に合わせ直します。**
        #
        # 種類の提出が間に合わなかった・通らなかったときは、サーバー側は
        # こちらの思っている構成と違います（送らなければ全員巡回車）。
        # 状態には1体ごとの `kind` が入っているので、それが正です。
        # 違ったまま進めると、補給車のつもりの車にうどんを取らせる計画を
        # 作って**毎日ずれ続けます**。
        told = tuple(agent.kind for agent in state.agents)
        if len(told) == len(kinds) and told != tuple(kinds):
            was = sum(1 for k in kinds if k == AgentKind.SUPPLY)
            now = sum(1 for k in told if k == AgentKind.SUPPLY)
            log(f"  ⚠ サーバーの種類がこちらの予定と違います"
                f"（補給車 {was}台 → **{now}台**）。"
                f"サーバーに合わせて組み直します。")
            kinds = told
            report.agent_kinds = kinds
            try:
                monitor.start_match(problem, kinds)
            except Exception:
                pass

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
        day_started = started
        deadline_monotonic = started + _seconds_until_deadline(
            state.ends_at, problem.day_seconds[day_index]
        )

        # **子プロセスを、親が考えはじめる前に走らせます。**
        #
        # これまでは、親が素案（貪欲・設計8通り・学習済みAI）を
        # 作り終えて最初の提出を済ませてから子に仕事を渡していました。
        # 設計だけで0.27秒かかるので、その間 11個の子は何もしません。
        # 1日あたり数コア秒を捨てていたことになります。
        #
        # 子に渡すのに要るのは「いまの盤面」だけで、親の素案は
        # 要りません。先に走らせます。
        parallel = None
        if PARALLEL_TRIALS and _WARM_FILES:
            parallel = _start_parallel_day(
                problem, brains, kinds, positions, fuels, state,
                day_steps, day_index, acquired_brands, steps_ahead,
                next_steps, deadline_monotonic, log,
                seed_for_day=seed + day_index * 7919,
            )

        candidates: list[Candidate] = []
        # 子に配った磨きの通し番号（崩し方の始まりをずらすのに使う）
        seed_turn = 0
        # 設計の失敗はこの日のぶんだけ出したいので、前の数を控えます。
        failures_before = dict(DESIGN_FAILURES)
        base_policies: list[tuple[str, object]] = [("貪欲", GreedyPolicy(problem))]
        # **全車の順路をまとめて設計した案も加えます。**
        #
        # 1手ずつ「次に良い店」を選ぶのではなく、1日の配分を先に
        # 決める方式です。実戦の渋滞で測ると、1位の玉数に対して
        # いまの実装が88〜89%、こちらは92%でした。
        #
        # 在庫をどれだけ重く見るかを変えて何通りか作ります。
        # 1通り0.02秒ほどなので、探索の時間をほとんど使いません。
        base_policies.extend(
            _planned_route_policies(problem, finder, kinds, positions,
                                    fuels, state.road_status, day_steps,
                                    acquired_brands, next_steps=next_steps)
        )
        # **昨日の順路を、今日の出発点として持ち越します。**
        #
        # 店は毎日同じ場所にあり、在庫も毎朝戻ります。だから昨日
        # 1日ぶんの時間をかけて磨いた順路は、今日もそのまま良い形
        # です。毎朝それを捨てて素案から登り直すのは、32x32では
        # とくに損でした（1手3msかかるので、登り直しに何千手も
        # 要ります）。
        #
        # 出発点が違っても計画は作れます（planner はいまいる場所から
        # 順路をたどるだけ）。良くなければ採点で落ちるので、
        # 候補に1つ加えるだけで害はありません。
        if WARM_START_ROUTES and carried is not None:
            base_policies.append(
                ("昨日の順路", PlannedRoutePolicy(problem, carried[0],
                                               carried[1]))
            )
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

        # 設計でできた案の数（あとで「子から何案届いたか」を引き算で
        # 出すのに使います）。
        designed_count = len(candidates)
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

        # **磨いている案の「補給車の置き場所」。**
        # 磨きは順路と迎える順番だけを動かすので、置き場所はその案の
        # あいだずっと同じです。採点のたびに渡さないと消えてしまい、
        # ハブや夜合流の案が1手で壊れます。
        polish_park = [None]
        # **夜の合流先も磨きを越えて残します。**
        # 残さないと、1手磨いた瞬間に翌朝の満タンが消えます
        # （`Candidate.park` と同じ罠）。
        polish_night = [None]
        # **帰らせる車も磨きを越えて残します。**
        polish_home = [None]
        # **時刻つきの待ち合わせも磨きを越えて残します。**
        polish_plan = [None]

        def polish_evaluate(trial, order, verify=False, night_park=None,
                            home=None, supply_plan=None):
            """磨いた順路を採点する。

            **磨きの最中は公式シミュレータに通し直しません。**
            計画を作る過程で planner も1歩ずつ前に進めているので、
            結果はそこから分かります。同じ日を何万回も再現するので、
            2度目のシミュレートはそのまま2割の無駄になります。
            採用するときだけ verify=True で通し直します。
            """

            return _build_candidate(
                "磨き", problem, planner, kinds, positions, fuels,
                state.road_status, day_steps,
                PlannedRoutePolicy(problem, trial, order,
                                   park=polish_park[0],
                                   night_park=(polish_night[0]
                                               if night_park is None
                                               else night_park),
                                   home=(polish_home[0] if home is None
                                         else home),
                                   supply_plan=(polish_plan[0]
                                                if supply_plan is None
                                                else supply_plan)),
                acquired_brands,
                day_index, congestion_risk, road_heat, steps_ahead, next_steps,
                verify=verify,
            )

        def seed_pool():
            """**磨きの出発点を選ぶ元の集まり。**

            直近 `POLISH_SEED_POOL` 件に、**設計で作った案**を必ず
            足します。理由は `SEED_KEEPS_DESIGNS` の上に。
            """

            recent = candidates[-POLISH_SEED_POOL:]
            if not SEED_KEEPS_DESIGNS:
                return recent
            designs = candidates[:designed_count]
            seen = {id(item) for item in recent}
            return [item for item in designs if id(item) not in seen] + recent

        def verified(item):
            """**公式シミュレータに通し直した案**を返す。

            磨きの採点は速さのために通し直しません（`_FastResult`）。
            ところが `cover_missing_brands` や
            `park_supply_for_night` は既定のまま採点を呼ぶので、
            **通していない結果がそのまま1日の答えになることが
            ありました**。1日目の系列拾いは長らく `NameError` で
            黙って落ちていたので表に出ませんでしたが、そこを直したら
            すぐ `AttributeError` で試合が止まりました。

            翌日の予測もこの結果から取るので、ここは必ず本物に
            します。1日1〜2回なので値段はありません。
            """

            if item is None or not isinstance(item.day_result, _FastResult):
                return item
            polish_park[0] = item.park
            polish_night[0] = item.night_park
            polish_home[0] = item.home
            got = polish_evaluate(
                routes_from_result(item.day_result, kinds),
                supply_order_from_result(item.day_result, kinds),
                verify=True)
            return got if got is not None else item

        # **提出する前に、足りない系列を拾いに行きます。**
        # 日別累計は玉数より上位の得点で、上位陣は満点です
        # （`cover_missing_brands` に実測つきの理由）。
        try:
            fixed = verified(cover_missing_brands(
                problem, finder, kinds, positions, state.road_status,
                best, polish_evaluate, supply_order_from_result(
                    best.day_result, kinds)))
            if fixed is not None and fixed.key > best.key:
                log(f"  足りない系列を拾いました（日別 {best.key[1]} → "
                    f"{fixed.key[1]} / 玉 {best.key[KEY_NOODLES]} → "
                    f"{fixed.key[KEY_NOODLES]}）")
                best = fixed
                candidates.append(fixed)
        except Exception:
            pass

        # **巡回路を凍結して、燃料が尽きる直前に補給車を横切らせます。**
        # 理由と実測は `plan_supply_meetings` の上に。
        try:
            planned = verified(plan_supply_meetings(
                problem, finder, kinds, positions, fuels, state.road_status,
                day_steps, best, polish_evaluate,
                supply_order_from_result(best.day_result, kinds)))
            if planned is not None and planned.key > best.key:
                log(f"  待ち合わせを時刻ごと組みました"
                    f"（玉 {best.key[KEY_NOODLES]} → "
                    f"{planned.key[KEY_NOODLES]}）")
                best = planned
                polish_plan[0] = planned.supply_plan
                candidates.append(planned)
        except Exception:
            _design_failed("時刻つき待ち合わせ")

        # **夜に補給車を巡回車のところへ送ります。**
        # 翌朝の満タンは歩を1つも使わない補給です。理由と上限の数字は
        # `park_supply_for_night` に。
        try:
            polish_park[0] = best.park
            parked = verified(park_supply_for_night(
                problem, finder, kinds, positions, state.road_status,
                day_steps, best, polish_evaluate,
                supply_order_from_result(best.day_result, kinds)))
            if parked is not None and parked.key > best.key:
                stayed = sum(
                    1 for index, kind in enumerate(kinds)
                    if kind == AgentKind.TOURING
                    and parked.day_result.positions[index] in {
                        parked.day_result.positions[other]
                        for other, other_kind in enumerate(kinds)
                        if other_kind == AgentKind.SUPPLY})
                log(f"  夜の合流を組みました（同じマスで夜を越す巡回車"
                    f"{stayed}台 / 翌朝ただで入る燃料"
                    f"{_night_refuel_gain(parked.day_result, kinds, problem.fuel_limit)}"
                    f" / 玉 {best.key[KEY_NOODLES]} → "
                    f"{parked.key[KEY_NOODLES]}）")
                best = parked
                polish_night[0] = parked.night_park
                polish_home[0] = parked.home
                candidates.append(parked)
        except Exception:
            _design_failed("夜に送る")

        revision, landed = _submit_or_wait(
            client, problem, kinds, positions, fuels, state.road_status,
            day_steps, best, log)
        if landed is not None and landed is not best:
            # 待機に差し替わったので、今日の案もそちらに合わせます。
            best = landed
            candidates.append(landed)
        # サーバーが受理した案。予測はここから取る。
        accepted = landed if revision >= 0 else None
        recent: list[dict] = []
        if accepted is None:
            log("  最初の提出に失敗しました。次の提出まで予測を出しません。")
        plan_seconds = time.monotonic() - started

        # ここから、締切まで時間いっぱい探す。
        # うまくいっている間は best の近くを細かく探し、しばらく改善がないと
        # 探索を強めて別の形を試す。改善が出たらまた細かい探索に戻す。
        resubmissions = 0
        attempt = 0
        # **探索数の覚え書き。**
        #
        # 「探索数が少なかった」という指摘を数で見られるようにします。
        # 磨き係は別の形から登り直すたびに作り直されるので、捨てる
        # 前にその手数を足しておきます。
        polish_tried = 0
        polish_wins = 0
        polish_seeds = 0
        # 磨きを別の形から始め直した回。作り直しの間隔をあけるのに使う。
        last_restart = -1000
        since_improvement = 0
        noise_levels = (0.05, 0.10, 0.20, 0.35, 0.55)
        noise_index = 0

        # **1日の探索も複数のコアで回します。**
        #
        # 事前試走は並列になりましたが、ここは単体のままでした。
        # タスクマネージャーで見ると探索中のCPUは20〜30%しか動きません。
        # 子プロセスに計画を作らせ、返ってきた行動計画をこちらで
        # 採点し直して候補に加えます（採点はこちらでやるので、
        # 子から重い中身を送り返す必要がありません）。
        # **子プロセスに投げて、親はすぐ自分の探索に入ります。**
        # 以前は並列が全部終わるまで親が待っていたため、
        # 締切の7割で並列が終わったあとは親1つだけが動いていました。
        # **出し直しは「改善ごと」ではなく「間隔ごと」にします。**
        #
        # 要項 p.12 (5) は、試合の進行を妨げるほどの回数を送ると
        # 妨害行為とみなされると書いています。そこで1日の回数を
        # MAX_RESUBMISSIONS_PER_DAY 回までに抑えています。
        #
        # ところが「磨き」は小さな改善を何十回も見つけます。
        # 改善ごとに送っていると上限にすぐ届き、**そこで探索を
        # 打ち切っていました**。良くなった案は手元に貯めておき、
        # 残り時間を残り回数で割った間隔ごとに、そのときの最良だけを
        # 送ります。締切の直前には必ず最後の1回を送ります。
        pending: Candidate | None = None
        last_submit = time.monotonic()

        def flush(force: bool = False) -> None:
            """貯めておいた最良案を送る。"""

            nonlocal pending, revision, accepted, resubmissions, last_submit
            if pending is None:
                return
            if resubmissions >= MAX_RESUBMISSIONS_PER_DAY:
                return
            now = time.monotonic()
            if not force:
                left = max(0.0, deadline_monotonic - now)
                room = MAX_RESUBMISSIONS_PER_DAY - resubmissions
                if now - last_submit < left / max(1, room):
                    return
            sent = _submit(client, pending.action_plan, log)
            resubmissions += 1
            last_submit = time.monotonic()
            if sent < 0:
                return
            revision = sent
            accepted = pending
            pending = None

        def take_parallel() -> None:
            """子から届いたぶんを取り込む。"""

            nonlocal best, pending, seed_turn
            if parallel is None:
                return
            for _key, label, plan in parallel.harvest():
                rebuilt = _candidate_from_plan(
                    problem, kinds, positions, fuels, state.road_status,
                    day_steps, plan, label, acquired_brands,
                    steps_ahead, next_steps, finder,
                )
                if rebuilt is None:
                    continue
                # **子が作った案も記録に残します。**
                # 以前は親の案だけ記録していたので、ログを見ても
                # 「どんな案があったか」が分かりませんでした。
                if record_log:
                    recorder.note_trial(label, rebuilt.key)
                candidates.append(rebuilt)
                if best is None or rebuilt.key > best.key:
                    best = rebuilt
                    pending = rebuilt
            # **列に磨きを切らさないようにします。**
            # 切らすと、子は残りの時間ずっとゆらぎ探索をします
            # （`POLISH_QUEUE_FLOOR` に実測つきの理由）。
            #
            # 配るのは**形のちがう案**です。最良案ばかり配ると、
            # 子が全員で同じ谷を登り直します（`diverse_seeds`）。
            try:
                if (best is not None
                        and parallel.polish_backlog() < POLISH_QUEUE_FLOOR):
                    # 設計の案も出発点に入れます（`seed_pool`）。
                    seeds = diverse_seeds(
                        seed_pool(), kinds, POLISH_SEEDS) or [best]
                    share = max(1, POLISH_QUEUE_REFILL // len(seeds))
                    for seed in seeds:
                        seed_turn += share
                        parallel.add_polish_jobs(
                            routes_from_result(seed.day_result, kinds),
                            POLISH_WORKER_BUDGET, share,
                            supply_order_from_result(seed.day_result, kinds),
                            offset=seed_turn,
                            gap=POLISH_NOISE_GAP,
                        )
            except Exception:
                pass

        polisher = None
        # **設計の案からも磨かせます。**
        #
        # 親が磨くのは「いまの最良案」からだけです。最良案はたいてい
        # 貪欲や学習済みAIが作ったもので、1台ずつ近い店を選んだ形を
        # しています。**全車の受け持ちをまとめて決めた設計**は、
        # そこから何手かけても届かない別の形です。
        #
        # 広いマップではこの差が大きく出ます。実戦1703（32x32・
        # 7体）を記録された渋滞つきで再現すると
        #
        #   実戦                            : 玉390
        #   設計からも磨く（手数3000）      : 玉436 / 日別225（1位と同じ）
        #
        # 子プロセスに投げるので、親の持ち時間は減りません。
        if parallel is not None:
            # **設計は2系統あるので、それぞれの best を磨かせます。**
            #
            #   「設計／補給設計」: いまいる場所から次に良い店へ、
            #                       **末尾に足して**組む
            #   「差し込み」      : どの車のどの位置に入れるのが
            #                       いちばん安いかを見て、**途中に
            #                       割り込んで**組む
            #
            # 作り方が違うので、1手ずつ直してもお互いの形には
            # 移れません。別々に登らせたほうが遠くまで行けます
            # （実戦6試合で 玉1495 → 1501）。
            picked: dict[str, Candidate] = {}
            for candidate in candidates:
                if candidate.label.startswith("差し込み"):
                    family = "差し込み"
                elif candidate.label.startswith(("設計", "補給設計")):
                    family = "設計"
                else:
                    continue
                if (family not in picked
                        or candidate.key > picked[family].key):
                    picked[family] = candidate
            share = max(1, POLISH_WORKER_JOBS // max(1, len(picked)))
            for designed in picked.values():
                try:
                    parallel.add_polish_jobs(
                        routes_from_result(designed.day_result, kinds),
                        POLISH_WORKER_BUDGET, share,
                        supply_order_from_result(designed.day_result, kinds),
                    )
                except Exception:
                    pass
        last_harvest = time.monotonic()
        while time.monotonic() < deadline_monotonic and attempt < 100000:
            attempt += 1
            # **回数ではなく時間で区切って取り込みます。**
            #
            # 長い日は親の1案に123ミリ秒かかるので、10回に1回だと
            # 1.2秒も回収が止まります。その間に子は仕事を終えて
            # 次をもらえず遊びます。CPUが上下して見えた一因です。
            now = time.monotonic()
            if now - last_harvest >= 0.2:
                last_harvest = now
                take_parallel()
            flush()
            # **いちばん良い案を、1手ずつ磨きます。**
            #
            # 実戦7試合を渋滞つきで再現した実測（学習済みAIなし）:
            #
            #   磨かない          : 玉1317 / 日別744
            #   設計だけ磨く      : 玉1337 / 日別758
            #   **最良案を磨く**  : 玉1440 / 日別762
            #
            # 1位との差が 78% → 87% に縮み、日別は7試合のうち6試合で
            # 1位と並びました。磨く手は「店を別の車に渡す」「2台で
            # 入れ替える」「まだ行かない店を差し込む」「並びを
            # ほどく」「行くのをやめる」の5種類で、**実際に1日ぶんを
            # 再現して点数が上がったものだけ**残します。
            #
            # 出発点はその時点の最良案です。設計でもAIでも、できた
            # 結果から「車ごとの店の並び」を取り出せるので、どの方策の
            # 案でも磨けます。
            #
            # 1手ずつ進めるので、締切のどこで止めても安全です。
            if POLISH_EVERY and attempt % POLISH_EVERY:
                # **中の点数ではなく、作ったときの点数で比べます。**
                # 崩して登り直すあいだ中の点数は下がるので、
                # それで作り直すと登り直しが毎回つぶれます。
                if polisher is None or polisher.base_key != best.key:
                    if polisher is not None:
                        polish_tried += polisher.tried
                        polish_wins += polisher.improvements
                    base_routes = routes_from_result(best.day_result, kinds)
                    base_supply = supply_order_from_result(
                        best.day_result, kinds
                    )
                    polish_park[0] = best.park
                    polish_night[0] = best.night_park
                    polish_home[0] = best.home
                    polisher = RoutePolisher(
                        problem, kinds, base_routes,
                        polish_evaluate, key=best.key,
                        supply_order=base_supply,
                        prefilter=make_prefilter(
                            problem, finder, kinds, positions, fuels,
                            state.road_status, day_steps,
                        ),
                    )
                    # **子プロセスにも同じ順路を磨かせます。**
                    # 親は1コアしか使えません。子ごとに試す手の
                    # 開始位置をずらすので、重ならずに11倍の手を
                    # 試せます。
                    if parallel is not None:
                        try:
                            parallel.add_polish_jobs(
                                base_routes, POLISH_WORKER_BUDGET,
                                POLISH_WORKER_JOBS, base_supply,
                            )
                        except Exception:
                            pass
                # **登り切ったら、別の形から登り直します。**
                #
                # 以前はここでふつうの探索（ゆらぎ）に戻していました。
                # けれど実測では、同じ出発点から磨きを100秒続けても
                # 20秒と1玉も変わりません（玉46/40/30/36のまま）。
                # 登り切ったあとに要るのは時間ではなく**別の出発点**で、
                # 5か所に分けると1日目は玉46→50になりました。
                #
                # ゆらぎ探索は1手7.67ms、磨きは1.36msで5.6倍ちがいます。
                # 余った時間はゆらぎではなく、別の形の磨きに回します。
                if polisher.exhausted and attempt - last_restart >= 50:
                    # 作り直しは軽くありません（案を並べ替えます）。
                    # 新しい形もすぐ登り切る盤面で毎回やらないよう、
                    # 間隔をあけます。
                    last_restart = attempt
                    # **ここで転んでも試合は止めません。**
                    # 作り直しは「あれば得」なだけなので、失敗したら
                    # いまの磨き係のまま次へ進みます。
                    try:
                        spare = [item for item in diverse_seeds(
                            seed_pool(), kinds, POLISH_SEEDS)
                            if item.key != polisher.base_key]
                        if spare:
                            seed_turn += 1
                            polish_seeds += 1
                            polish_tried += polisher.tried
                            polish_wins += polisher.improvements
                            pick = spare[seed_turn % len(spare)]
                            polish_park[0] = pick.park
                            polish_night[0] = pick.night_park
                            polish_home[0] = pick.home
                            polisher = RoutePolisher(
                                problem, kinds,
                                routes_from_result(pick.day_result, kinds),
                                polish_evaluate, key=pick.key,
                                supply_order=supply_order_from_result(
                                    pick.day_result, kinds),
                                prefilter=make_prefilter(
                                    problem, finder, kinds, positions, fuels,
                                    state.road_status, day_steps,
                                ),
                            )
                            polisher.shake = seed_turn
                            # 親の比較用。これを best にすると、次の回で
                            # すぐ作り直されてしまいます。
                            polisher.base_key = best.key
                    except Exception:
                        pass
                if not polisher.exhausted:
                    polished = polisher.step()
                    if polished is not None:
                        # **採用する前に公式シミュレータに通し直します。**
                        exact = polish_evaluate(
                            polisher.best_routes, polisher.best_supply,
                            verify=True,
                        )
                        if exact is None or exact.key <= best.key:
                            # 通し直したら良くなっていなかった。進めません。
                            polisher.base_key = best.key
                            continue
                        polished = exact
                        candidates.append(polished)
                        if record_log:
                            recorder.note_trial(polished.label, polished.key)
                        best = polished
                        pending = polished
                        # **磨き係を作り直させません。**
                        #
                        # 作り直すと、崩して登り直す途中の状態
                        # （いちばん良かった形・崩し方の順番）が
                        # 毎回消えます。せっかく自分で見つけた
                        # 改善で、自分の積み上げを捨てることに
                        # なっていました。
                        polisher.base_key = polished.key
                        since_improvement = 0
                        noise_index = 0
                    continue

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
            if record_log:
                recorder.note_trial(label, candidate.key)
            recent.append({
                "label": label,
                "brands": candidate.key[0],
                "daily": candidate.key[1],
                # 玉数の位置は KEY_NOODLES で持ちます。番号を
                # ずらし忘れると、ビューアに「玉0」と出ます。
                "noodles": candidate.key[KEY_NOODLES],
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
                best_score=(best.key[0], best.key[1], best.key[KEY_NOODLES]),
                resubmissions=resubmissions,
                recent=list(reversed(recent)),
                # 暫定の最良案を盤面に出す。探索中も動きが見えます。
                # **関数で渡します。** 候補1件ごとにここへ来ますが、
                # ビューアへ書くのは0.2秒に1回です。dict で渡すと
                # 書かない回でも作られてしまいます（軌跡の作り直しは
                # 1日ぶんの再現なので、そのまま探索の邪魔になる）。
                preview=lambda: _preview_day(
                    monitor, problem, kinds, positions, fuels, state,
                    best, day_index, day_steps, len(candidates),
                    resubmissions, revision, acquired_brands,
                    daily_total, noodle_total,
                ),
            )
            candidates.append(candidate)
            if candidate.key > best.key:
                best = candidate
                pending = candidate
                since_improvement = 0
                noise_index = 0
            else:
                since_improvement += 1
                # 同じ強さで探し続けても出てこないなら、揺らぎを大きくする。
                if since_improvement >= 25 * (noise_index + 1):
                    noise_index = min(noise_index + 1, len(noise_levels) - 1)

        # 締切の直前に、子から届いているぶんを最後に取り込む。
        take_parallel()
        # **上位の案だけ、1日先まで見て選び直します。**
        #
        # 採点は1日単位なので、「今日は同じ玉数でも、明日の出だしが
        # まるで違う」案を見分けられません。終わった位置と燃料から
        # 翌日を貪欲で1回走らせれば、その差が数字になります。
        #
        # 全部に先読みすると重いので、上位だけにします。実測
        # （実戦7試合・記録された渋滞つき・手数3000）:
        #
        #   先読みなし  : 玉1866（32x32は502）
        #   上位12件    : 玉1875（32x32は511 / 1709は251→263）
        #   上位40件    : 玉1875（12件と同じ。上位に入っていれば足りる）
        #
        # 翌日の道路状態は今日のものを使います。交通量は前2日の
        # 滞在から決まるので、今日混んでいる道は明日も混んでいる
        # 可能性が高く、いちばん手近な見立てです。
        if ROLLOUT_CANDIDATES and next_steps > 0 and len(candidates) > 1:
            picked = _rollout_pick(
                problem, planner, kinds, candidates, state.road_status,
                next_steps, acquired_brands, day_index, best,
            )
            if picked is not None and picked is not best:
                log(f"  1日先まで見て「{picked.label}」に差し替えます。")
                best = picked
                pending = picked
        # **締切の前にもう一度、足りない系列を拾いに行きます。**
        # 磨きは玉数を伸ばす方向に動くので、途中で系列を落とすことが
        # あります。日別は玉数より上位なので、最後に必ず見ます。
        try:
            fixed = verified(cover_missing_brands(
                problem, finder, kinds, positions, state.road_status,
                best, polish_evaluate, supply_order_from_result(
                    best.day_result, kinds)))
            if fixed is not None and fixed.key > best.key:
                log(f"  足りない系列を拾いました（日別 {best.key[1]} → "
                    f"{fixed.key[1]} / 玉 {best.key[KEY_NOODLES]} → "
                    f"{fixed.key[KEY_NOODLES]}）")
                best = fixed
                pending = fixed
        except Exception:
            pass

        # **巡回路を凍結して、燃料が尽きる直前に補給車を横切らせます。**
        # 理由と実測は `plan_supply_meetings` の上に。
        try:
            planned = verified(plan_supply_meetings(
                problem, finder, kinds, positions, fuels, state.road_status,
                day_steps, best, polish_evaluate,
                supply_order_from_result(best.day_result, kinds)))
            if planned is not None and planned.key > best.key:
                log(f"  待ち合わせを時刻ごと組みました"
                    f"（玉 {best.key[KEY_NOODLES]} → "
                    f"{planned.key[KEY_NOODLES]}）")
                best = planned
                polish_plan[0] = planned.supply_plan
                pending = planned
        except Exception:
            _design_failed("時刻つき待ち合わせ")

        # **夜に補給車を巡回車のところへ送ります。**
        # 翌朝の満タンは歩を1つも使わない補給です。理由と上限の数字は
        # `park_supply_for_night` に。
        try:
            polish_park[0] = best.park
            parked = verified(park_supply_for_night(
                problem, finder, kinds, positions, state.road_status,
                day_steps, best, polish_evaluate,
                supply_order_from_result(best.day_result, kinds)))
            if parked is not None and parked.key > best.key:
                stayed = sum(
                    1 for index, kind in enumerate(kinds)
                    if kind == AgentKind.TOURING
                    and parked.day_result.positions[index] in {
                        parked.day_result.positions[other]
                        for other, other_kind in enumerate(kinds)
                        if other_kind == AgentKind.SUPPLY})
                log(f"  夜の合流を組みました（同じマスで夜を越す巡回車"
                    f"{stayed}台 / 翌朝ただで入る燃料"
                    f"{_night_refuel_gain(parked.day_result, kinds, problem.fuel_limit)}"
                    f" / 玉 {best.key[KEY_NOODLES]} → "
                    f"{parked.key[KEY_NOODLES]}）")
                best = parked
                polish_night[0] = parked.night_park
                polish_home[0] = parked.home
                pending = parked
        except Exception:
            _design_failed("夜に送る")

        # **貯めた最良案を必ず送ります。**
        flush(force=True)
        if pending is not None:
            # 送れなかったので、サーバーが持っているのは1つ前の案です。
            # **ここで best を進めると、翌日の予測が実際とずれます。**
            if accepted is not None:
                log("  最後の出し直しが通らなかったので、受理済みの案を採用します。")
                best = accepted
        if parallel is not None:
            parallel.stop()

        day_brands = {event.brand for event in best.day_result.acquisitions}
        new_brands = day_brands - acquired_brands
        acquired_brands |= day_brands
        daily_total += len(day_brands)
        noodle_total += len(best.day_result.acquisitions)
        # 予測は**サーバーが受理した案**から取る。best ではない。
        final = accepted if accepted is not None else best
        # 明日の出発点にする順路を取っておきます（`WARM_START_ROUTES`）。
        if WARM_START_ROUTES:
            try:
                carried = (routes_from_result(final.day_result, kinds),
                           supply_order_from_result(final.day_result, kinds))
            except Exception:
                carried = None
        recorder.finish_day(
            day_index=day_index,
            label=final.label,
            candidates=len(candidates),
            resubmissions=resubmissions,
            day_steps=day_steps,
            parallel_plans=(parallel.collected if parallel is not None else 0),
            start_positions=list(positions),
            start_fuels=list(fuels),
            positions=list(final.day_result.positions),
            fuels=list(final.day_result.fuels),
            acquisitions=list(final.day_result.acquisitions),
            refuels=list(final.day_result.refuels),
            tracks=_tracks_of(problem, kinds, positions, fuels,
                              state.road_status, final.action_plan, day_steps),
            road_status=state.road_status,
            # ほかのチームの位置（渋滞のもと）も記念に残します。
            enemies=[[agent.position for agent in group]
                     for _name, group in state.others],
            problem=problem,
            kinds=kinds,
            finder=planner.finder,
            missed=missed_spots(
                problem, kinds, planner.finder, state.road_status,
                list(positions), list(fuels), day_steps,
                list(final.day_result.acquisitions),
            ) if record_log else [],
            brand_total=len(acquired_brands),
            daily_total=daily_total,
            noodle_total=noodle_total,
            elapsed=time.monotonic() - day_started,
        )
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
        # **画面に出すだけの処理は、失敗しても試合を止めません。**
        #
        # ここから下は「人が見るための数字」だけで、提出にも翌日の予測にも
        # 使いません。それなのに素のまま置くと、表示の書き間違い1つで
        # その日が落ちます（実際に `_Event` に `position` がなくて
        # `AttributeError` で試合が止まりました）。まとめて包みます。
        try:
            # **補給の内訳を出します。**
            #
            # 準決勝32x32の解析で、ここがいちばん効く数字でした。補給車は
            # 1日の歩をほぼ全部使って動いているのに、初日と2日目は**1回
            # しか配れていません**（50燃料と80燃料）。必要なのは1日120燃料
            # （6台×20。厳密な上限から、1位に並ぶのに要るのは1台1日20燃料）。
            # そして3日目の朝の燃料は [80, 22, 0, 1, 69, 74] で、2台が1歩も
            # 動けませんでした。**初日の飢えが3日目を壊す**という筋です。
            #
            # 回数だけでは分かりません（満タンに近い車に寄ると回数は増えて
            # 量は増えない）。**量と1回の平均**を並べて出します。
            poured = [event.fuel_after - event.fuel_before
                      for event in best.day_result.refuels]
            supply_cars = sum(1 for kind in kinds if kind == AgentKind.SUPPLY)
            touring_cars = sum(1 for kind in kinds if kind == AgentKind.TOURING)
            if supply_cars:
                want = 20 * touring_cars
                log(f"  補給: {len(poured)}回 / 合計{sum(poured)}燃料"
                    f"（1回平均{sum(poured) / max(1, len(poured)):.0f}"
                    f" / 20以下{sum(1 for x in poured if x <= 20)}回）"
                    f" → 1位に並ぶ目安は1日{want}燃料"
                    f"（1台20×{touring_cars}台）")
            if touring_cars:
                dry = [best.day_result.fuels[index]
                       for index, kind in enumerate(kinds)
                       if kind == AgentKind.TOURING]
                stuck = sum(1 for value in dry if value <= 1)
                log(f"  明日の朝の燃料 {dry}"
                    + (f" ← **{stuck}台が動けません**" if stuck else ""))

            # **探索数を出します。**
            #
            # 準決勝のあと「探索数が少なかった」と言われました。実際に
            # どれだけ手を試せているのかが見えないと、速くしても効いたか
            # 分かりません。親の試行・磨きの手数・当たり数・子から届いた
            # 案・設計の失敗を1行にまとめます。
            if polisher is not None:
                polish_tried += polisher.tried
                polish_wins += polisher.improvements
            got_parallel = len(candidates) - attempt - designed_count
            log(f"  探索: 親{attempt}手 / 磨き{polish_tried}手"
                f"（当たり{polish_wins} / 出発点{polish_seeds + 1}か所）"
                f" / 子から{max(0, got_parallel)}案 / 候補{len(candidates)}件"
                f" / {plan_seconds:.1f}秒で初回提出")
            # **設計が黙って消えるのを見逃さないための行です。**
            # 2026-10-10に `design_cover` が `NameError` で丸ごと消え、
            # 過去29試合で順位318→321に落ちていました。
            today_failures = {
                name: count - failures_before.get(name, 0)
                for name, count in DESIGN_FAILURES.items()
                if count - failures_before.get(name, 0) > 0
            }
            if today_failures:
                lost = " / ".join(f"{name}{count}回"
                                  for name, count in sorted(today_failures.items()))
                log(f"  ⚠ 設計が失敗しました: {lost}")

            # **その日の上限と、どこまで取れたかを出します。**
            #
            # 準決勝では上位18チームが日別満点で、勝負は玉数だけでした。
            # 「あと何玉あるのか」と「何が縛っているのか（歩数・燃料・
            # 在庫）」が分かれば、次の日の作戦を変えられます。
            # 理由と実測は `hexa/ceiling.py` に。
            if CEILING_REPORT:
                try:
                    from hexa import ceiling as _ceiling

                    got = _ceiling.day_ceiling(
                        problem, finder, kinds, positions, fuels,
                        state.road_status, day_steps)
                    took = len(best.day_result.acquisitions)
                    rate = 100 * took / max(1, got["balls"])
                    log(f"  {_ceiling.describe(got)}"
                        f" → 取れたのは{took}（{rate:.0f}%）")
                except Exception:
                    pass
        except Exception as error:      # 表示だけなので、黙って続けます
            log(f"  （この日の集計表示に失敗しました: {error}）")

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

    saved = recorder.save(
        f"種類{len(acquired_brands)} / 日別累計{daily_total} / 玉数{noodle_total}",
        report.mismatches,
    )
    if saved is not None:
        log(f"  行動の記録を残しました: {saved}")
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
    """行動計画を出す。**通信の失敗は `client` 側で何度も試し直します。**

    理由は `hexa/client.py` の `POST_RETRIES` に。ここで返す -1 は
    「何度やっても通らなかった」という意味です。
    """

    try:
        return client.post_actions(action_plan, log=log)
    except ProconApiError as error:
        log(f"  提出に失敗しました: {error}")
        return -1
    except Exception as error:               # 想定外でも1日を捨てない
        log(f"  提出に失敗しました（{type(error).__name__}）: {error}")
        return -1


def _submit_or_wait(client: ProconClient, problem: Problem, kinds, positions,
                    fuels, road_status, day_steps: int, best, log):
    """計画を出す。**通らなければ全員待機を出して、予測もそれに合わせる。**

    準決勝の32x32（第1791641357試合）では、3日目の提出が通らず
    **7台が1歩も動きませんでした**（手元の予測は日別132・玉216、
    実際は日別112）。何も受理されない日を作らないために、最後は
    必ず「必ず有効な計画」（全員待機）を出します。待機でも、店の
    上で夜を越した車は翌朝の1歩目で1玉入ります。

    返すのは (受理番号, 実際に受理された案)。案が差し替わったら
    予測もそちらに合わせるので、翌日の計画が狂いません。
    """

    revision = _submit(client, best.action_plan, log)
    if revision >= 0:
        return revision, best
    log("  計画が通らないので、全員待機を出し直します"
        "（何も受理されない日を作らないため）")
    plan = _fallback_plan(problem.agent_count, day_steps)
    revision = _submit(client, plan, log)
    if revision < 0:
        return -1, None
    try:
        day_result = simulate_day(
            problem=problem, agent_kinds=kinds,
            start_positions=tuple(positions), start_fuels=tuple(fuels),
            action_plan=plan, day_steps=day_steps, road_status=road_status,
        )
    except Exception:
        return revision, None
    return revision, Candidate("全員待機", plan, day_result, best.key)


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
            # 練習場では行動を記録する（本番モードでは記録しない）。
            report = play_production_match(
                client, brains, seed=seed + played, log=log,
                record_log=True, match_id=f"{int(time.time())}_{played + 1}",
            )
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
