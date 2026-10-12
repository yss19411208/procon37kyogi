"""設計した順路を、**公式シミュレータで採点しながら**少しずつ直す。

**なぜ必要か**

`hexa/allocate.py` の設計は「近い店から順に配る」だけなので、
机上の見積り（店どうしの最短距離の合計）で良く見えても、実際に
走らせると違うことが起きます。

  - 通り道にある店の玉は、狙っていなくても入る（公式 1.3）
  - 在庫は全車で取り合うので、2台目が空振りすることがある
  - 渋滞した道を通ると、見積りより大きく遅れる

実戦7試合を渋滞つきで再現すると、設計が採用されたのは最初の
数日だけで、あとは「揺らぎのある貪欲」に負けていました。
設計の形は良いのに、**その形を実際の点数で磨いていない**のが
原因です。

**どう直すか**

いちばん良かった設計から出発して、

  1. ある車の1店を、別の車に渡す
  2. 2台の店を入れ替える
  3. まだ誰も行かない店を、どこかの車に差し込む
  4. 取りに行くのをやめて歩数を空ける

という4種類の手を順に試し、**実際にシミュレータに通して**
点数が上がったものだけ残します。見積りではなく本物の点数で
選ぶので、上の3つの「机上では見えないこと」が自動的に入ります。

手の数は決め打ちです。乱数は使いません。同じ入力なら必ず
同じ答えになります。
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

from hexa.problem import Problem, walkable_neighbors
from hexa.rules import AgentKind


# 「取り替える手」（`_phase_exchange`）で、1店につき何店まで
# 候補にするか。**この手は実測で入れませんでした**（下の表）。
NEAR_SPOTS = 6


@lru_cache(maxsize=8)
def _near_spots(problem: Problem, count: int = NEAR_SPOTS) -> dict:
    """店ごとに、近い順の別の店を返す。

    池を避けたマス数（地形の重みなしの幅優先）で測ります。地形や
    渋滞まで入れた最短歩数のほうが正確ですが、ここは「取り替える
    候補を絞る」だけなので、近さの目安で足ります。
    """

    neighbors = walkable_neighbors(problem)
    spots = [spot.position for spot in problem.spots]
    spot_set = set(spots)
    table: dict[int, tuple[int, ...]] = {}
    for start in spots:
        seen = {start: 0}
        queue = deque([start])
        found: list[int] = []
        while queue and len(found) < count:
            cell = queue.popleft()
            for _direction, next_cell in neighbors[cell]:
                if next_cell in seen:
                    continue
                seen[next_cell] = seen[cell] + 1
                queue.append(next_cell)
                if next_cell in spot_set:
                    found.append(next_cell)
                    if len(found) >= count:
                        break
        table[start] = tuple(found)
    return table


def _insertions(spots: list[int], spot: int) -> list[list[int]]:
    """その店を、並びのどこに差し込むかの候補をすべて返す。"""

    return [spots[:at] + [spot] + spots[at:] for at in range(len(spots) + 1)]


# **何手つづけて失敗したら崩すか。**
#
# 手の種類をすべて出し切るまで待つと、1日の持ち時間では一度も
# 崩せません（1種類で何百通りもあるため）。少し粘って出てこなければ
# 崩したほうが、同じ手数でも遠くまで行けます。
#
# **固定の数にすると当たり外れが大きい**ので、形の大きさに
# 比例させます。手の数は「全車が回る店の数」でほぼ決まるためです。
# 固定値で測ると 100で1410 / 180で1421 / 250で1474 / 320で1431 /
# 450で1429 と、250だけ当たりのような形になりました。これは
# この6試合に合わせただけの数字なので採りません。
# **崩す強さの上限。1 で「重ねがけしない」＝実測でこちらが上。**
#
# 崩し方を一巡しても最高記録が動かないとき、崩し方を重ねがけして
# もっと遠くへ飛ばす、という案を試しました。理屈は通っています
# （1つだけ崩しても同じ谷に戻ってしまう）。
#
# 実戦4試合を記録された渋滞つきで再現（1日20秒・親1つ）:
#
#   重ねがけしない(1) : 玉849 / 日別422
#   3段まで重ねる(3)  : 玉845 / 日別421
#
# **わずかに下がりました。** 理由ははっきりしています。呼んだ側は
# 「出し切った」を合図に**別の形から登り直す**ようになっており
# （`production.diverse_seeds`）、そちらのほうが強いからです。
# 重ねがけは、同じ谷にとどまる時間を延ばすだけでした。
#
# 手は残してあります。別の形の持ち合わせが無い場面（出発点が
# 1つしか作れない盤面）では、また意味を持つかもしれません。
_SHAKE_STRENGTH = 1

POLISH_PATIENCE_FLOOR = 120
POLISH_PATIENCE_PER_SPOT = 8

# 1回の step() のなかで、安い検査で弾き続ける上限。
# ここで打ち切らないと、全部弾かれる場合に1回が長くなりすぎます。
# **安く見積もって、有望な手から試すか。**
#
# 32x32（巡回6台・32店）の手の総数は数万あり、1手4msでは1日57秒でも
# **近傍を一巡すらできません**（実測で20秒6,504手・「登り切った」0回）。
# つまり近傍のごく一部しか見ていないので、**どの手から試すか**が
# そのまま点数になります。
#
# 歩数の差は `prefilter` の覚え書きで1マイクロ秒で出ます。本物の
# 評価（4ms）の4000分の1なので、まとめて見積もってから良い順に
# 評価すれば、同じ秒数で当たりの濃い手を試せます。
#
# 0 にすると、これまでどおり出てきた順に試します。
BATCH_SIZE = 0
# まとめたなかから、本物の評価に回す数。**0 なら全部**（並べ替えだけ）。
#
# 少なく絞ると、残りは次に崩すまで試されません。近傍の取りこぼしに
# なるので、既定は「捨てずに並べ替えるだけ」にします。
BATCH_TAKE = 0

# 見積りの基準。
#   "both"   : 訪問数が多く、歩数が短い順
#   "steps"  : 歩数が短い順だけ
#   "visits" : 訪問数が多い順だけ
BATCH_SCORE = "both"

SKIP_LIMIT = 2000

# 覚えておく「すでに再現した案」の数。超えたら忘れます。
#
# 1日に再現するのは数千件なので、これだけあれば足ります。
# 20万件まで覚えると、32x32では指紋だけで100MBを超えて
# メモリが足りなくなりました。
SEEN_LIMIT = 30_000

# 訪問数が増えないのに歩数が増える手を捨てるか。
#
# **実測で入れませんでした（False で止めています）。**
#
# 実戦7試合を記録された渋滞つきで再現（手数3000）:
#
#   捨てない : 玉1866（32x32は502 / 1709の日別84）
#   捨てる   : 玉1839（32x32は477 / 1709の日別82）
#
# 「同じ訪問数で歩数が増えるのは損」という見立ては外れでした。
# 余分に歩いた先で planner が別の店を拾ったり、終わる場所が
# 良くなったりするためだと見ています。
DROP_COSTLIER_MOVES = False

# 同点の手を続けて何回まで許すか（横歩きの上限）。
#
# **実測で入れませんでした（0 で止めています）。**
#
# 実戦6試合を記録された渋滞つきで再現（手数3000）:
#
#   許さない : 玉1495（32x32は455）
#   30回まで : 玉1437（32x32は411）
#
# 同点の手に持ち時間を使うぶん、崩して登り直す回数が減ります。
# 平らな場所から出る役目は「崩して登り直す」ほうが上でした。
SIDEWAYS_LIMIT = 0


class RoutePolisher:
    """1手ずつ磨く係。締切のある本番でも、好きなところで止められます。

    `step()` を1回呼ぶと、手を1つ試して1日ぶんを再現します。
    良くなったら中の順路を置き換え、その候補を返します。
    """

    def __init__(self, problem, kinds, routes, evaluate, key=None,
                 supply_order=None, repair=None, prefilter=None,
                 keep_end=None):
        # **終点を動かさない縛り**（`keep_end` にセル番号）。
        #
        # 準決勝32x32を厳密な上限で切り分けると、「毎朝満タンで
        # 始まる」形がいちばん高い（玉373。実際の朝の燃料だと243）。
        # 補給は `simulator` の歩0の行動段階でも走るので、**全車が
        # 夜に補給車のマスへ集まれば6台ぶんの満タンがただで入ります**。
        # 夜に1店へ戻る条件をつけても上限は354で、1位の304より上です。
        #
        # ところが磨きは順路を自由に動かすので、**夜合流の形は数手で
        # 壊れます**。集まる店を6通り変えて測っても玉231で1つも
        # 動かなかったのはこれが理由でした（縛らない既定は240）。
        #
        # そこで、この縛りがあるあいだは「どの車も最後の店を
        # `keep_end` のまま保つ」手だけを採ります。
        self.keep_end = keep_end
        self.problem = problem
        self.touring = [index for index, kind in enumerate(kinds)
                        if kind == AgentKind.TOURING]
        self.routes = {index: list(routes.get(index, []))
                       for index in self.touring}
        # 補給車が迎える順番。中身は車の番号か (迎え先のセル, 車) の組。
        #
        # **以前はセルを捨てていました。** 「迎え先はその場で選び直す」
        # という作りで、`PlannedRoutePolicy` が
        # min(その車のいる場所, その車の行き先, 設計の指定) から
        # 貪欲に選んでいました。
        #
        # ところが**待ち合わせの場所は、燃料が切れるまでの時間と
        # 巡回車の寄り道量を決めるいちばん重要な変数**です。それが
        # 探索の外にあったので、鹿児島高専（51）や沖縄（12）がやって
        # いる「先回り」の形が原理的に作れませんでした。
        #
        # 第1703（32x32）で磨きに9倍の時間をかけても玉が3個しか
        # 増えなかったのはこれが理由です（手数は足りていて、動かせる
        # 変数が足りていなかった）。いまはセルも持って、下の
        # `_supply_moves` で動かします。
        self.supply = [
            (int(car[0]), int(car[1])) if isinstance(car, tuple)
            else int(car)
            for car in (supply_order or [])
        ]
        self.evaluate = evaluate
        # **崩したあとの埋め直し**（渡されなければ貪欲まかせ）。
        #
        # 崩して空いた歩数を、`PlannedRoutePolicy` の貪欲な埋め戻しに
        # 任せると「その場で次に近い店」しか選べません。差し込みで
        # 組み直すと、2店のあいだに割り込む形まで作れます。
        self.repair = repair
        # **再現する前に弾く関数**（渡さなければ全部再現します）。
        #
        # 1日ぶんの再現は1件5ミリ秒ですが、「歩数が足りない順路」かを
        # 見るだけなら隣どうしの距離を足すだけで10マイクロ秒です。
        # 500倍ちがうので、先に弾けばそのぶん本物の手を試せます。
        self.prefilter = prefilter
        self.skipped = 0
        # いまの形の「訪問数」と「合計歩数」。手を捨てる目安に使います。
        self.visits = sum(len(spots) for spots in self.routes.values())
        self.cost = (prefilter(self.routes)
                     if prefilter is not None else None)
        # すでに再現した案の指紋。
        self.seen: set = set()
        self.all_spots = [spot.position for spot in problem.spots]
        # 店ごとの「近い店」。取り替える手の候補を絞るのに使います。
        self.near = _near_spots(problem)
        # 店ごとの在庫（1日に何台まで取れるか）。
        self.stocks = {spot.position: min(spot.stocks, len(self.touring))
                       for spot in problem.spots}
        # 呼び出した側が持っていた点数。これは変えません（本番の親は
        # これを見て「作り直すか」を決めるので、中で動かすと毎回
        # 作り直しになってしまいます）。
        self.base_key = key
        # いま登っている最中の点数
        self.key = key
        # **これまでで一番良かったもの**。崩して登り直すので、
        # 途中で点数が下がります。良いものは別に取っておきます。
        self.best_key = key
        self.best_routes = {index: list(spots)
                            for index, spots in self.routes.items()}
        self.best_supply = list(self.supply)
        # その盤面で試す手の一覧（小さい盤面では増えます）。
        self.phases = phases_for(problem)
        # 手の種類ごとの (試した数, 当たった数)。
        self.stats = [[0, 0] for _ in self.phases]
        self.supply_stat = [0, 0]
        self.phase = 0
        self.shake = 0
        self.shakes = 0
        # 崩す強さ（重ねがけする崩し方の数）。行き止まるたびに上げます。
        self.strength = 1
        # 崩し方を一巡する前の最高記録。一巡して動かなければ行き止まり。
        self._cycle_key = key
        self.sideways = 0
        # まとめて見積もった手の控え（`BATCH_SIZE` が0なら使いません）
        self.batch: list = []
        self.batch_drained = False
        self.batch_skipped = 0
        self.moves = self._new_moves()
        self.tried = 0
        self.improvements = 0
        self.fails = 0
        self.sideways = 0
        self.exhausted = False
        entries = sum(len(spots) for spots in self.routes.values())
        if not entries:
            entries = len(self.all_spots)
        self.patience = max(POLISH_PATIENCE_FLOOR,
                            POLISH_PATIENCE_PER_SPOT * entries)

    def _order(self):
        """試す手の種類の並び。**1つずつずらして一巡します。**

        **当たり率の高い順に並べる案は、実測で大きく悪化しました。**

        実戦1703（32x32）の4日分で種類ごとの当たり率を数えると、
        200倍の開きがあります。

          取りに行くのをやめる  : 4.26%（282通り中12件）
          1台のなかで並べ替え   : 1.42%（773通り中11件）
          2台で1店ずつ入れ替え  : 0.22%
          2台の後半を交換       : 0.19%
          まだ行かない店を足す  : 0.18%
          1店を別の車へ渡す     : 0.15%
          補給車の順番          : 0.11%
          続いた2〜3店を渡す    : 0.07%
          3台で1店ずつ回す      : 0.02%（45,330通り中7件）

        それでも当たり率で並べると 玉1866 → 1738（32x32は502→424、
        日別も225→220）と落ちました。当たりやすい「やめる」手に
        寄ってしまい、訪問数が減るためだと見ています。種類を順に
        一巡するほうが、形が散って遠くまで行けます。

        数字は `self.stats` に残してあるので、あとから見られます。
        """

        count = len(self.phases)
        return [(self.phase + offset) % count for offset in range(count)]

    def _new_moves(self):
        return _all_moves(self.touring, self.routes, self.supply,
                          self.all_spots, self._order(), self.stocks,
                          self.near, self.phases)

    def _restart(self, routes, supply, key) -> None:
        # 土台が変わると見積りが古くなるので、控えは捨てます。
        self.batch = []
        self.batch_drained = False
        self.routes = {index: list(spots) for index, spots in routes.items()}
        self.supply = list(supply)
        self.key = key
        # **次は違う種類の手から始めます。**
        # 毎回1番から試し直すと、1番だけで持ち時間が尽きて、
        # 後ろの種類に一度もたどり着けませんでした。
        self.phase += 1
        self.moves = self._new_moves()
        self.fails = 0
        self.exhausted = False
        self.visits = sum(len(spots) for spots in self.routes.values())
        if self.prefilter is not None:
            self.cost = self.prefilter(self.routes)

    def _shake(self) -> bool:
        """**手が尽きたら、いちばん良かった形を少し崩して登り直します。**

        **なぜ必要か**

        1手ずつ良くする登り方（山登り）は、どの1手を試しても
        下がる形に着くとそこで止まります。実測でもはっきり出ました。

          手数 800 : 玉365
          手数3000 : 玉365  ← 増やしても1玉も動かない

        手の種類を増やしても、いずれ同じことが起きます。
        そこで**わざと崩して**、別の谷から登り直します。崩し方は
        決め打ちの順番なので、乱数は使いません。良かった形は
        `best_*` に取ってあるので、崩して悪くなっても失いません。
        """

        # **崩し方は一巡したら、また最初から使います。**
        #
        # 一巡で打ち切ると、持ち時間が増えてもそこで止まります。
        # 崩すのは毎回「いちばん良かった形」なので、同じ崩し方でも
        # 土台が変わっていれば別の谷に落ちます。
        #
        # そのうえで、**一巡まわって最高記録が1つも動かなければ**
        # 打ち切ります。そこが本当の行き止まりです。
        if self.shakes and self.shake % len(_SHAKES) == 0:
            if self.best_key == self._cycle_key:
                # **一巡して駄目でも、すぐには諦めません。**
                #
                # 崩し方を1つだけ当てる形では、崩しても同じ谷に
                # 戻ってしまいます。実測でも、同じ出発点から磨くと
                # 20秒でも100秒でも玉46/40/30/36 と**1玉も動きません**
                # でした（手数は18,403手まで伸びています）。
                #
                # そこで行き止まったら**崩す強さを1段あげます**。
                # 2段目は崩し方を2つ重ねがけ、3段目は3つ、という形で、
                # 同じ谷から出られるまで遠くへ飛ばします。
                # `_SHAKE_STRENGTH` まで上げても最高記録が動かなければ、
                # そこが本当の行き止まりです。
                if self.strength >= _SHAKE_STRENGTH:
                    return False
                self.strength += 1
            self._cycle_key = self.best_key
        routes, supply = self.best_routes, self.best_supply
        for extra in range(self.strength):
            routes, supply = _SHAKES[(self.shake + extra) % len(_SHAKES)](
                self.touring, routes, supply, self.problem
            )
        # **埋め直しは1回おきにします。**
        #
        # 差し込みで埋め直すと、小さい地図では良くなりますが
        # 32x32では悪くなりました（玉1501→1478 / 32x32は455→427）。
        # 広い地図では「その場で次に近い店」を積む埋め方のほうが
        # 合うことがあるので、両方を交互に使って形を散らします。
        if self.repair is not None:
            try:
                routes = self.repair(routes)
            except Exception:
                pass
        self.shake += 1
        self.shakes += 1
        self.routes = {index: list(spots) for index, spots in routes.items()}
        self.supply = list(supply)
        # 崩した形も「もう見た」に入れておきます（あとで同じ形が
        # 手として出てきても、もう一度再現しないように）。
        self.seen.add((tuple(tuple(self.routes[i]) for i in self.touring),
                       tuple(self.supply)))
        found = self.evaluate(self.routes, self.supply)
        self.tried += 1
        # 崩した直後の点数から登り直します（下がっていてもかまいません）。
        self.key = found.key if found is not None else None
        self.phase = 0
        self.moves = self._new_moves()
        self.fails = 0
        self.visits = sum(len(spots) for spots in self.routes.values())
        if self.prefilter is not None:
            self.cost = self.prefilter(self.routes)
        return True

    def _score_move(self, routes, supply):
        """安い見積り。小さいほど有望。通らない手は None。

        見るのは2つだけです。

          1. 訪問数が増えるか（増えるほうが玉になりやすい）
          2. 歩数の合計（短いほど、余りで拾える玉が増える）

        `prefilter` が車ごとの歩数を覚えているので、1台だけ変えた手
        なら計算はほとんど起きません。
        """

        if self.prefilter is None:
            return (0, 0)
        total = self.prefilter(routes)
        if total is None:
            return None
        base_cost = self.cost if self.cost is not None else total
        if BATCH_SCORE == "steps":
            return (total - base_cost, 0)
        visits = sum(len(spots) for spots in routes.values())
        base_visits = self.visits
        if BATCH_SCORE == "visits":
            return (base_visits - visits, 0)
        return (base_visits - visits, total - base_cost)

    def _fill_batch(self):
        """手をまとめて取り出し、見積りの良い順に並べて控えに置く。"""

        picked = []
        for _ in range(BATCH_SIZE):
            try:
                routes, supply, kind = next(self.moves)
            except StopIteration:
                self.batch_drained = True
                break
            mark = (tuple(tuple(routes[i]) for i in self.touring),
                    tuple(supply))
            if mark in self.seen:
                self.skipped += 1
                continue
            score = self._score_move(routes, supply)
            if score is None:
                self.seen.add(mark)
                self.skipped += 1
                continue
            picked.append((score, routes, supply, kind, mark))
        picked.sort(key=lambda item: item[0])
        # `BATCH_TAKE` が 0 なら捨てません（並べ替えだけ）。
        keep = picked if not BATCH_TAKE else picked[:BATCH_TAKE]
        self.batch = list(keep)
        self.batch.reverse()          # pop() で良い順に取り出す
        self.batch_skipped += len(picked) - len(keep)

    def step(self):
        """手を1つ試す。これまでで一番良くなった候補を返す（なければ None）。"""

        if self.exhausted:
            return None
        if self.patience and self.fails >= self.patience:
            # **粘っても出てこないので崩します。**
            if self._shake():
                return None
            self.exhausted = True
            return None
        # **まとめて見積もる形。** 有望な手から試します。
        if BATCH_SIZE:
            if not self.batch:
                if self.batch_drained:
                    if self._shake():
                        return None
                    self.exhausted = True
                    return None
                self._fill_batch()
                if not self.batch:
                    return None
            _score, routes, supply, kind, mark = self.batch.pop()
            self.seen.add(mark)
            if len(self.seen) > SEEN_LIMIT:
                self.seen.clear()
            return self._try(routes, supply, kind)

        # **再現するまで、安い検査で弾き続けます。**
        # 1回の step() が1件の再現にあたるようにします。
        routes = supply = None
        kind = SUPPLY_PHASE
        for _ in range(SKIP_LIMIT):
            try:
                routes, supply, kind = next(self.moves)
            except StopIteration:
                if self._shake():
                    return None
                self.exhausted = True
                return None
            # **同じ案を2度再現しません。**
            #
            # 手の種類をまたぐと同じ形が何度も出てきます（崩して
            # 登り直すとさらに増えます）。1件5ミリ秒の再現を
            # 繰り返すのは丸損なので、指紋で覚えておきます。
            mark = (tuple(tuple(routes[i]) for i in self.touring),
                    tuple(supply))
            if mark in self.seen:
                self.skipped += 1
                routes = None
                continue
            if self.prefilter is not None:
                total = self.prefilter(routes)
                if total is None:
                    self.seen.add(mark)
                    self.skipped += 1
                    routes = None
                    continue
                # **訪問数が増えないのに歩数が増える手は捨てます。**
                #
                # 同じ数の店を回るのに歩数が増えれば、余りが減るだけ
                # です。planner が余りで拾う玉も減ります。
                #
                # これは「ほぼ損」という見立てなので、厳密ではありま
                # せん（終わる場所が良くなることはある）。安い検査で
                # 落として、そのぶん本物の手を試すほうが得でした。
                if DROP_COSTLIER_MOVES and self.cost is not None:
                    visits = sum(len(v) for v in routes.values())
                    if visits <= self.visits and total > self.cost:
                        self.seen.add(mark)
                        self.skipped += 1
                        routes = None
                        continue
            self.seen.add(mark)
            if len(self.seen) > SEEN_LIMIT:
                self.seen.clear()
            break
        if routes is None:
            return None
        return self._try(routes, supply, kind)

    def _keeps_end(self, routes) -> bool:
        """どの巡回車も、最後の店が `keep_end` のままか。"""

        if self.keep_end is None:
            return True
        for index in self.touring:
            line = routes.get(index) or []
            if line and line[-1] != self.keep_end:
                return False
        return True

    def _try(self, routes, supply, kind):
        """1つの手を本物の採点にかけ、良ければ採る。"""

        if not self._keeps_end(routes):
            # 終点が動く手は見ません（理由は `keep_end` に）。
            self.fails += 1
            return None
        self.tried += 1
        slot = self.supply_stat if kind == SUPPLY_PHASE else self.stats[kind]
        slot[0] += 1
        found = self.evaluate(routes, supply)
        if found is None:
            self.fails += 1
            return None
        if self.key is not None and not (found.key > self.key):
            # **同点の手は、少しだけ許します（横歩き）。**
            #
            # 採点は「系列 → 立ち往生 → 余力 → 玉数 → …」の順で
            # 比べるので、**同点になる手がとても多い**です。同点を
            # 捨てていると、その平らな場所から出られません。
            #
            # 横に歩いたぶんは最高記録に数えないので、悪くなること
            # はありません。同じところを回り続けないよう、続けて
            # 歩ける回数に上限を置きます。
            if (found.key == self.key and self.sideways < SIDEWAYS_LIMIT):
                self.sideways += 1
                self.fails += 1
                self._restart(routes, supply, found.key)
                return None
            self.fails += 1
            return None
        self.improvements += 1
        slot[1] += 1
        self.sideways = 0
        self._restart(routes, supply, found.key)
        if self.best_key is not None and not (found.key > self.best_key):
            # 登り直しの途中。まだ最高記録には届いていません。
            return None
        self.best_key = found.key
        self.best_routes = {index: list(spots)
                            for index, spots in self.routes.items()}
        self.best_supply = list(self.supply)
        return found


SUPPLY_PHASE = -1


def _all_moves(touring, routes, supply, all_spots, order=None, stocks=None,
               near=None, phases=None):
    """巡回車の順路と、補給車の迎える順番の両方を動かす手を出す。

    戻り値は (順路, 迎える順番, 手の種類の番号) の組です。
    `order` は「どの種類から順に試すか」の並びです。
    `phases` を渡すと、その一覧の手だけを試します（小さい盤面では
    `phases_for` が手を増やします）。
    """

    if phases is None:
        phases = _MOVE_PHASES

    # **補給車の順番を先に試します。**
    # 手の数が巡回車より何桁も少ないのに、効き方は大きいからです
    # （補給車は燃料を使わないので、止まっている歩数は丸損）。
    for trial in _supply_moves(touring, supply, routes):
        yield (routes, trial, SUPPLY_PHASE)
    if order is None:
        order = range(len(phases))
    for index in order:
        maker = phases[index]
        # 近い店の表を要る手にだけ渡します（ほかの手の形は変えません）。
        trials = (maker(touring, routes, all_spots, near)
                  if maker in _NEAR_PHASES
                  else maker(touring, routes, all_spots))
        for trial in trials:
            if stocks is not None and not _within_stock(trial, stocks):
                continue
            yield (trial, supply, index)


def _within_stock(routes, stocks) -> bool:
    """**在庫より多くの車が同じ店を狙っていないか。**

    公式の処理順では、在庫が足りないときリストの若い車が先に取り、
    あとの車は**何も取れません**（質問回答集その2 Q26）。つまり
    在庫1の店を2台が狙うと、片方はその往復を丸ごと無駄にします。

    実戦1703（32x32）で測ると、取った順を最短でなぞれば1玉8.1歩で
    済むのに、実際は9.6歩かかっていました。3位は8.0歩です。
    **差は配り方ではなく、寄り道と立ち止まり**でした。
    在庫を超えて狙う手を最初から捨てれば、その寄り道が減ります。
    """

    wanted: dict[int, int] = {}
    for spots in routes.values():
        for spot in spots:
            wanted[spot] = wanted.get(spot, 0) + 1
    for spot, count in wanted.items():
        if count > stocks.get(spot, 0):
            return False
    return True


def _car_of(entry):
    """迎え先の指定から、車の番号だけを取り出す。"""

    return entry[1] if isinstance(entry, tuple) else entry


def _supply_moves(touring, supply, routes=None):
    """補給車の「迎える順番と迎え先」を動かす手。

    **補給車は誰も手を入れていませんでした。**

    燃料を1滴も使わず、合流すれば相手は満タンになります。
    止まっている歩数はそのまま無駄です。それでも実戦では
    1日の4〜9割を止まって過ごしていました。

    待ち合わせの時刻は決めません。燃料が切れた車はその場で
    止まって待つので、**順番だけ合っていれば必ず拾えます**。
    だから動かすのは順番だけで足ります。
    """

    # 1. 隣どうしを入れ替える
    for i in range(len(supply) - 1):
        out = list(supply)
        out[i], out[i + 1] = out[i + 1], out[i]
        yield out

    # 2. 1台を先頭へ持ってくる（いちばん効く手なので早めに試す）
    for i in range(1, len(supply)):
        out = list(supply)
        out.insert(0, out.pop(i))
        yield out

    # 3. 迎える車を増やす（どの位置にも差し込む）
    for car in touring:
        for at in range(len(supply) + 1):
            out = list(supply)
            out.insert(at, car)
            yield out

    # 4. 迎えるのをやめる
    for i in range(len(supply)):
        out = list(supply)
        del out[i]
        yield out

    # 5. 離れた位置どうしを入れ替える
    for i in range(len(supply)):
        for j in range(i + 2, len(supply)):
            out = list(supply)
            out[i], out[j] = out[j], out[i]
            yield out

    # 6. **迎え先のセルを、その車が回る店に変える。**
    #
    # 待ち合わせを巡回車の順路の上でやれば、巡回車の寄り道が
    # ゼロになり、しかもその店の玉も入ります。どの店で落ち合うのが
    # いちばん良いかは盤面と燃料で変わるので、**探索に決めさせます**。
    #
    # 以前これを現場の貪欲（補給車が店へ寄る）でやって失敗しました。
    # 補給車が店へ動くと、巡回車は「さっきまで補給車がいた場所」へ
    # 向かっているのですれ違います。**両方の行き先を同時に決める**
    # 必要があり、それはここでしかできません。
    if routes:
        for i in range(len(supply)):
            car = _car_of(supply[i])
            for spot in routes.get(car, ())[:SUPPLY_MEET_SPOTS]:
                if supply[i] == (spot, car):
                    continue
                out = list(supply)
                out[i] = (spot, car)
                yield out

    # 7. 迎え先の指定をやめる（その車のいる場所で落ち合う）。
    for i in range(len(supply)):
        if not isinstance(supply[i], tuple):
            continue
        out = list(supply)
        out[i] = _car_of(supply[i])
        yield out


def improve_routes(
    problem: Problem,
    kinds: tuple[AgentKind, ...],
    routes: dict[int, list[int]],
    evaluate,
    budget: int = 400,
    supply_order=None,
    repair=None,
    prefilter=None,
) -> tuple[dict[int, list[int]], list, object]:
    """順路を実際の点数で磨く。

    戻り値は (磨いた順路, 補給車の迎える順番, その候補)。

    **順番も返します。** 返し忘れていたせいで、呼び出し側が
    公式シミュレータに通し直すときに「補給車の予定なし」で
    採点していました（実戦7試合で玉1866→1741）。

    `evaluate(routes, supply_order)` は候補を返す関数です。候補は
    `key` を持ち、大きいほど良いものとします。返せないときは
    None を返します。
    """

    polisher = RoutePolisher(problem, kinds, routes, evaluate,
                             supply_order=supply_order, repair=repair,
                             prefilter=prefilter)
    base = evaluate(polisher.routes, polisher.supply)
    if base is None:
        return polisher.routes, polisher.supply, None
    polisher.base_key = polisher.key = polisher.best_key = base.key
    best_candidate = base
    for _ in range(max(0, budget - 1)):
        if polisher.exhausted:
            break
        found = polisher.step()
        if found is not None:
            best_candidate = found
    return polisher.best_routes, polisher.best_supply, best_candidate


def _moves(touring, routes, all_spots):
    """試す手を順に出す。すべて決め打ちで、乱数は使いません。"""

    for phase in _MOVE_PHASES:
        yield from phase(touring, routes, all_spots)


def _phase_transfer(touring, routes, all_spots):

    # 1. ある車の1店を、別の車のどこかへ渡す
    for giver in touring:
        for position in range(len(routes[giver])):
            spot = routes[giver][position]
            for taker in touring:
                if taker == giver:
                    continue
                if spot in routes[taker]:
                    continue
                for order in _insertions(routes[taker], spot):
                    out = {index: list(spots) for index, spots in routes.items()}
                    del out[giver][position]
                    out[taker] = order
                    yield out

def _phase_relocate(touring, routes, all_spots):
    """同じ車のなかで、1店を別の位置へ動かす。

    **実測で入れませんでした（手の一覧から外してあります）。**

    実戦6試合を記録された渋滞つきで再現（手数3000）:

      入れない          : 玉1495（32x32は455）
      1番目に入れる     : 玉1453（32x32は417）
      2番目に入れる     : 玉1458（32x32は416）

    手の種類が増えると1種類あたりの持ち時間が減り、崩して登り直す
    回数も減ります。並べ替えは `allocate.improve_order` の2-optと
    planner の最短経路がすでにかなり詰めているので、この手に
    持ち時間を割くと差し引きで損でした。

    1台が1日に10店以上回るマップ（32x32は32店を6台で回ります）では、
    回る順番そのものが歩数を決めます。これまでは「別の車に渡す」
    「2台で入れ替える」が先にあり、同じ車のなかの並べ替えは
    2-opt（区間をひっくり返す）だけでした。2-optは区間の向きしか
    変えられないので、「1店だけ離れた位置へ移す」形には届きません。
    """

    for owner in touring:
        spots = routes[owner]
        for i in range(len(spots)):
            rest = spots[:i] + spots[i + 1:]
            for at in range(len(rest) + 1):
                if at == i:
                    continue
                out = {index: list(items) for index, items in routes.items()}
                out[owner] = rest[:at] + [spots[i]] + rest[at:]
                yield out


def _phase_or_opt(touring, routes, all_spots):
    """同じ車のなかで、続いた2〜3店をまとめて別の位置へ動かす。

    **実測で入れませんでした**（`_phase_relocate` と同じ理由）。
    """

    for owner in touring:
        spots = routes[owner]
        for length in (2, 3):
            for i in range(len(spots) - length + 1):
                segment = spots[i:i + length]
                rest = spots[:i] + spots[i + length:]
                for at in range(len(rest) + 1):
                    if at == i:
                        continue
                    out = {index: list(items)
                           for index, items in routes.items()}
                    out[owner] = rest[:at] + segment + rest[at:]
                    yield out


def _phase_tail_swap(touring, routes, all_spots):
    # 2. **2台の後半を丸ごと交換する（2-opt*）**
    #
    # 1店ずつ動かす手だけだと、持ち時間をいくら増やしても同じ形の
    # 近くから出られません。実測（実戦1685・1687を渋滞つきで再現）:
    #
    #   手数 800 : 玉365
    #   手数3000 : 玉365  ← 増やしても動かない
    #
    # 2台の受け持ちをまとめて取り替えると、「どの車がどのあたりを
    # 回るか」という大きな形が変わります。1店ずつでは何手もかけて
    # 通らないと届かない形に、1手で移れます。
    for first in touring:
        for second in touring:
            if second <= first:
                continue
            left = routes[first]
            right = routes[second]
            for i in range(len(left) + 1):
                for j in range(len(right) + 1):
                    head_a, tail_a = left[:i], left[i:]
                    head_b, tail_b = right[:j], right[j:]
                    if not tail_a and not tail_b:
                        continue
                    new_a = head_a + tail_b
                    new_b = head_b + tail_a
                    # 同じ車が同じ店を2回回っても2玉目は入りません。
                    if len(set(new_a)) != len(new_a):
                        continue
                    if len(set(new_b)) != len(new_b):
                        continue
                    out = {index: list(spots)
                           for index, spots in routes.items()}
                    out[first] = new_a
                    out[second] = new_b
                    yield out

def _phase_segment(touring, routes, all_spots):
    # 3. **続いた2〜3店をまとめて別の車へ渡す（Or-opt）**
    for giver in touring:
        spots = routes[giver]
        for length in (2, 3):
            for start in range(len(spots) - length + 1):
                segment = spots[start:start + length]
                rest = spots[:start] + spots[start + length:]
                for taker in touring:
                    if taker == giver:
                        continue
                    if any(spot in routes[taker] for spot in segment):
                        continue
                    for at in range(len(routes[taker]) + 1):
                        out = {index: list(items)
                               for index, items in routes.items()}
                        out[giver] = list(rest)
                        out[taker] = (routes[taker][:at] + segment
                                      + routes[taker][at:])
                        yield out

def _phase_swap_one(touring, routes, all_spots):
    # 4. 2台の店を1つずつ入れ替える
    for first in touring:
        for second in touring:
            if second <= first:
                continue
            for i, left in enumerate(routes[first]):
                for j, right in enumerate(routes[second]):
                    if right in routes[first] or left in routes[second]:
                        continue
                    out = {index: list(spots) for index, spots in routes.items()}
                    out[first][i] = right
                    out[second][j] = left
                    yield out

def _phase_insert(touring, routes, all_spots):
    # 5. まだ行かない店を差し込む
    planned = {spot for spots in routes.values() for spot in spots}
    for spot in all_spots:
        for taker in touring:
            if spot in routes[taker]:
                continue
            if spot in planned and len(touring) > 1:
                # すでに誰かが行く店でも、別の車が行けば1玉増えます。
                pass
            for order in _insertions(routes[taker], spot):
                out = {index: list(spots) for index, spots in routes.items()}
                out[taker] = order
                yield out

def _phase_two_opt(touring, routes, all_spots):
    # 6. 1台のなかで並びを入れ替える（2-opt）
    #
    # 遠回りをほどく手です。並びが良くなると歩数が余るので、
    # そのぶんは planner が勝手に次の店へ回します。
    for owner in touring:
        spots = routes[owner]
        for i in range(len(spots)):
            for j in range(i + 2, len(spots) + 1):
                out = {index: list(items) for index, items in routes.items()}
                out[owner][i:j] = list(reversed(out[owner][i:j]))
                if out[owner] != spots:
                    yield out

def _phase_drop(touring, routes, all_spots):
    # 7. 取りに行くのをやめて歩数を空ける
    for giver in touring:
        for position in range(len(routes[giver])):
            out = {index: list(spots) for index, spots in routes.items()}
            del out[giver][position]
            yield out


def _phase_exchange(touring, routes, all_spots, near=None):
    """**行く店を1つ、行かない店と取り替える。**

    **実測で入れませんでした（手の一覧から外してあります）。**

    理屈は通っていました。遠い店を近い店に替えるには「やめる
    （drop）」と「差し込む（insert）」の2手が必要で、**その途中の形は
    必ず玉が1つ少ない**ので、山登りはそこを通れません。1手で
    入れ替えれば通れます。32x32の内訳でも、無駄な歩（寄り道・
    立ち止まり・燃料切れ）は全体の15%しかなく、玉1つにかかる移動は
    7.4歩（3位と並ぶには6.75歩）。つまり残っている差は「どの店を
    どの順に回るか」そのもので、ここが効く場所のはずでした。

    16試合を記録された渋滞つきで再現（手数3000）:

      入れない            : 玉3998 / 日別1683
      近い4店と取り替える : 玉3986 / 日別1678
      近い6店と取り替える : 玉4012 / 日別1682
      近い8店と取り替える : 玉3966 / 日別1685
      全部の店と取り替える: 玉3985 / 日別1683

    **候補の数に対して単調ではありません。** 6店のときだけ玉が
    +14ですが、4店と8店では下がります。1試合ずつ見ても、同じ手で
    1437153は149〜168、1454913は178〜200と大きく振れていました。
    これは手の良し悪しではなく、**初日の選び方が変わると9日目まで
    別の道をたどる**という揺れです（日別も1678〜1685に振れます）。

    つまりこの手の効き目は揺れの中に埋もれていて、16試合では
    「効く」と言えません。**日別は玉数より上位の得点**なので、玉が
    +14でも日別が-1になる形は採れません。手の種類を増やすと1種類
    あたりの持ち時間が減るぶん確実に損をするので、外しました。

    （差し込む位置は元の場所のままにしてあります。並べ替えは2-optと
    planner の最短経路が後から直すので、位置まで広げると手の数が
    店の数の2乗になります。）
    """

    for owner in touring:
        spots = routes[owner]
        for position in range(len(spots)):
            here = set(spots)
            target = spots[position]
            pool = all_spots if near is None else near.get(target, ())
            for spot in pool:
                if spot in here:
                    continue
                out = {index: list(items) for index, items in routes.items()}
                out[owner][position] = spot
                yield out


def _phase_rotate3(touring, routes, all_spots):
    """3台のあいだで1店ずつ回す（巡回交換）。

    **なぜ必要か**

    2台のあいだで渡す・入れ替える手だけでは、「AがBの近くへ、
    BがCの近くへ、CがAの近くへ」とずれている形から出られません。
    どの1手も単独では点数を下げるので、山登りが止まります。

    3台で輪にして1店ずつ回すと、1手でその形を解けます。
    置き換えは同じ位置にするので、手の数は増えすぎません。
    """

    if len(touring) < 3:
        return
    for first in touring:
        for second in touring:
            if second == first:
                continue
            for third in touring:
                if third == first or third == second:
                    continue
                if not (first < second and first < third):
                    # 同じ輪を3通り数えないよう、いちばん小さい番号から
                    continue
                left, middle, right = (routes[first], routes[second],
                                       routes[third])
                for i in range(len(left)):
                    for j in range(len(middle)):
                        for k in range(len(right)):
                            a, b, c = left[i], middle[j], right[k]
                            if a == b or b == c or c == a:
                                continue
                            if c in left or a in middle or b in right:
                                continue
                            out = {index: list(items)
                                   for index, items in routes.items()}
                            out[first][i] = c
                            out[second][j] = a
                            out[third][k] = b
                            yield out


# 試す手の種類。**毎回1番から試し直すと、1番だけで持ち時間が尽きます。**
#
# 良くなるたびに最初から試し直していたので、後ろの種類（受け持ちの
# 大きな入れ替えなど）には一度もたどり着けませんでした。実測でも
# 手数を800から3000に増やして玉数が1つも動きませんでした。
#
# いまは**始める種類を1つずつずらして**一巡します。どの種類も
# 必ず順番が回ってきます。
_MOVE_PHASES = (
    _phase_transfer,
    _phase_tail_swap,
    _phase_segment,
    _phase_swap_one,
    _phase_insert,
    _phase_two_opt,
    _phase_rotate3,
    _phase_drop,
)

# **小さい盤面だけで足す手。**
#
# `_phase_relocate` と `_phase_or_opt` を外した理由は「手の種類が
# 増えると1種類あたりの持ち時間が減る」でした。これは持ち時間が
# 足りない盤面（32x32は1手3ms）の話です。
#
# 16x16は**登り切って止まります**。同じ盤面（第1981）を1日2秒で
# 回しても玉178、20秒でも玉181で、時間を10倍にして3玉しか動きません。
# 止まっている盤面では、減らす持ち時間がありません。そこで小さい
# 盤面にかぎって手を足します。
_SMALL_MOVE_PHASES = _MOVE_PHASES + (_phase_relocate, _phase_or_opt)

# ここまでのマス数を「小さい盤面」とみなします（16x16=256マス、
# 24x24=576マス）。
SMALL_BOARD_CELLS = 400


def phases_for(problem) -> tuple:
    """その盤面で試す手の一覧。

    小さい盤面は登り切って止まるので、手の種類を増やします
    （`_SMALL_MOVE_PHASES` に実測つきの理由）。
    """

    cells = getattr(problem, 'cell_count', None)
    if cells is not None and cells <= SMALL_BOARD_CELLS:
        return _SMALL_MOVE_PHASES
    return _MOVE_PHASES

# 近い店の表（`_near_spots`）を受け取る手。
_NEAR_PHASES = frozenset({_phase_exchange})


# ---------------------------------------------------------------------------
# 崩し方（ruin & recreate）
#
# **なぜ必要か**
#
# 山登りは「どの1手を試しても下がる形」に着くと止まります。手数を
# 800から3000に増やしても玉数が1つも動かなかったのがその証拠です。
#
# そこで、いちばん良かった形をわざと崩し、別の谷から登り直します。
# 崩したぶんは `PlannedRoutePolicy` が貪欲に埋め直すので、
# 「崩して作り直す」形になります。
#
# 崩し方は決め打ちの順番で、乱数は使いません。同じ入力なら必ず
# 同じ答えになります。
# ---------------------------------------------------------------------------

def _drop_last(touring, routes, supply, problem=None):
    """全車の最後の1店をやめる。余った歩数は貪欲が埋めます。"""

    out = {i: list(routes.get(i, [])) for i in touring}
    for i in touring:
        if out[i]:
            out[i].pop()
    return out, list(supply)


def _drop_first(touring, routes, supply, problem=None):
    """全車の最初の1店をやめる。出だしの形が変わります。"""

    out = {i: list(routes.get(i, [])) for i in touring}
    for i in touring:
        if out[i]:
            out[i].pop(0)
    return out, list(supply)


def _drop_half(touring, routes, supply, problem=None):
    """全車の後半をやめる。大きく作り直します。"""

    out = {}
    for i in touring:
        spots = list(routes.get(i, []))
        out[i] = spots[: max(1, len(spots) // 2)]
    return out, list(supply)


def _shift_routes(touring, routes, supply, problem=None):
    """**受け持ちを丸ごと隣の車へ回す。**

    広いマップで効きます。1店ずつ動かす手では、何十手かけないと
    この形には移れません。
    """

    out = {}
    for n, i in enumerate(touring):
        out[i] = list(routes.get(touring[(n + 1) % len(touring)], []))
    return out, list(supply)


def _reverse_all(touring, routes, supply, problem=None):
    """全車の回る向きを逆にする。"""

    out = {i: list(reversed(routes.get(i, []))) for i in touring}
    return out, list(supply)


def _thin_out(touring, routes, supply, problem=None):
    """全車で1つ飛ばしにやめる。骨だけ残して埋め直します。"""

    out = {i: list(routes.get(i, []))[::2] for i in touring}
    return out, list(supply)


def _clear_supply(touring, routes, supply, problem=None):
    """補給車の順番を白紙に戻し、貪欲に任せる。"""

    return {i: list(routes.get(i, [])) for i in touring}, []


def _split_pairs(touring, routes, supply, problem=None):
    """隣どうしの車で受け持ちを半分ずつ交換する。"""

    out = {i: list(routes.get(i, [])) for i in touring}
    for n in range(0, len(touring) - 1, 2):
        a, b = touring[n], touring[n + 1]
        ha, hb = len(out[a]) // 2, len(out[b]) // 2
        left, right = out[a][:ha] + out[b][hb:], out[b][:hb] + out[a][ha:]
        # 同じ車が同じ店を2回回っても2玉目は入らないので、重複を取る
        out[a] = list(dict.fromkeys(left))
        out[b] = list(dict.fromkeys(right))
    return out, list(supply)


def _slice_by_row(touring, routes, supply, problem=None):
    """**地図を上から順に切って、受け持ちを引き直す。**

    いま回っている形をいじるのではなく、**まったく別の分け方**から
    やり直します。広いマップで効きます（32x32は32店を6台で回ります）。

    店を行・列の順に並べ、台数ぶんの帯に等分します。1店ずつ動かす手
    では、何十手かけてもこの形には移れません。
    """

    if problem is None or not touring:
        return {i: list(routes.get(i, [])) for i in touring}, list(supply)
    order = sorted(problem.spots, key=lambda s: (s.position // problem.width,
                                                 s.position % problem.width))
    return _share(touring, [s.position for s in order]), list(supply)


def _slice_by_angle(touring, routes, supply, problem=None):
    """地図の中心から見た向きで切って、受け持ちを引き直す。

    上下の帯とは別の分け方になります。中心にまとまった地図では
    こちらのほうが短く回れます。
    """

    if problem is None or not touring:
        return {i: list(routes.get(i, [])) for i in touring}, list(supply)
    import math

    cx = problem.width / 2.0
    cy = problem.height / 2.0

    def angle(spot):
        x = spot.position % problem.width
        y = spot.position // problem.width
        return math.atan2(y - cy, x - cx)

    order = sorted(problem.spots, key=angle)
    return _share(touring, [s.position for s in order]), list(supply)


def _share(touring, positions):
    """並べた店を、台数ぶんにできるだけ均等に配る。"""

    out = {i: [] for i in touring}
    if not positions:
        return out
    size = max(1, (len(positions) + len(touring) - 1) // len(touring))
    for n, i in enumerate(touring):
        out[i] = positions[n * size:(n + 1) * size]
    return out


# **迎え先の候補にする、順路の先頭から何店まで見るか。**
#
# その車が回る店はたいてい数軒なので、全部見ても手数は増えません。
# **実測では 0（止める）のままにしています。**
# 32x32の4試合を手数600で通すと 日別725/玉2036 → 日別724/玉2051、
# 手数2000では 玉2088 と玉は増えますが、燃料の仕掛け一式で測ると
# 日別が落ちました。1巡あたりの手が増えるぶん、1つの形を登り切る
# のに手数がかかるのが効いています。**考え方（待ち合わせの場所を
# 探索に入れる）は正しい**ので、手の数を増やさない形に作り直す
# 価値があります。
SUPPLY_MEET_SPOTS = 0


_SHAKES = (
    _drop_last,
    _shift_routes,
    _drop_half,
    _thin_out,
    _drop_first,
    _split_pairs,
    _reverse_all,
    _slice_by_row,
    _clear_supply,
    _slice_by_angle,
)
