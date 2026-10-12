"""経路探索。AIの「アルゴリズム側」の土台。

同じ日の中では地形も道路状態も変わらないので、ある始点からの最短経路表は
一度だけ計算してキャッシュします。これをやらないと、1手を選ぶたびに
マップ全体のダイクストラを何万回も回すことになり、学習が実用速度に届きません。

コストは (ステップ数, 燃料) の辞書順です。ステップ数が同じなら燃料の少ない
経路を選びます。どちらも移動元のセルで決まるため（質問回答集その1 Q10 /
その2 Q25）、通常のダイクストラで正しく解けます。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush

from hexa.problem import Problem, walkable_neighbors
from hexa.rules import RoadStatus, Terrain, fuel_cost, move_steps


# 「燃料をどれだけ重く見るか」の段階数。細かくするとキャッシュが効かなくなる。
FUEL_WEIGHT_STEPS = 4

# **燃料をどれだけ早めに心配するか。**
#
# `fuel_weight` は「今日の残り歩数」と「残り燃料」だけを見ます。
# ところが燃料は試合をまたいで持ち越すので、**今日は足りていても
# 試合全体では足りない**ことがあります。
#
# 実戦1703（32x32・9日・巡回6台・燃料上限113）では、1台が試合中に
# 728歩を動きます。平地だけでも364燃料、実際は500以上必要なので、
# タンク4〜5杯ぶんの補給が要ります。ところや1日単位で見ると
# 「97歩に対し燃料113」なので余裕があるように見え、燃料を節約
# しません。結果、燃料切れで止まった歩数が208歩ありました。
#
# この値を1より大きくすると、同じ状況でも早めに燃料を節約します。
#
# **実測では1.0（元のまま）が最良でした。**
#
# 実戦6試合を記録された渋滞つきで再現（手数3000）:
#
#   1.0 : 玉1495（32x32は455）
#   1.5 : 玉1476（32x32は440）
#
# 早く節約すると道路を避けるので歩数が伸び、取れる店が減ります。
# 燃料切れで止まる歩数より、遠回りで失う歩数のほうが大きいという
# ことです。数字は残しておきます。
FUEL_URGENCY = 1.0

# **歩数と燃料の重みを、書いてあるとおりに釣り合わせるか。**
#
# `fuel_weight` の説明には「1マスのコストを ステップ数/S + 燃料/F で
# 測れば、足りないほうの資源が自動的に重くなる」と書いてあります。
# ところが実装は `round(W × min(1, S/F))` で、**S=F のときに W（＝
# 燃料だけ最小化）を返していました**。コストの形が
# `歩数×(W-w) + 燃料×w` なので、その式と同じ向きにするには
# `w = W × S/(S+F)` で、S=F なら W/2（釣り合い）が正しい値です。
#
# **本選の盤面はまさにそこでした。** 1日の歩数40・燃料上限40なので
# 毎朝 S=F=40 → 重み4 → 道路を一切使わない。道路は1歩・燃料2、
# 平地は2歩・燃料1なので、燃料だけ見ると道路は4倍高く見えます。
# ところが1日目は全道路が順調（1歩）で**2倍速い**のに、それを
# 丸ごと捨てていました。16x16には道路が60セルあります。
FUEL_WEIGHT_BALANCED = True


def fuel_weight(remaining_steps: int, remaining_fuel: int) -> int:
    """残りステップと残り燃料から、燃料の重み（0〜4）を決める。

    1マス進むのに必要なものは地形で違います。

        道路(順調): 1ステップ / 燃料2
        平地      : 2ステップ / 燃料1

    道路は2倍速いぶん、**燃料は4倍**かかります。
    ステップ最短の道だけを選ぶと道路ばかり通り、燃料が先に尽きて動けません。
    実戦のマップで測ると、最短経路の燃料消費は1ステップあたり1.75
    （道路の2.0にほぼ張り付き）でした。燃料110・総ステップ268の試合では、
    燃料が全体の4割ぶんしか持たない計算になります。

    残りステップ S と残り燃料 F のとき、1マスのコストを

        ステップ数 / S + 燃料 / F

    で測れば、足りないほうの資源が自動的に重くなります。
    """

    if remaining_fuel <= 0:
        return FUEL_WEIGHT_STEPS
    want = FUEL_URGENCY * remaining_steps
    if FUEL_WEIGHT_BALANCED:
        # **釣り合う形**。コストは `歩数×(W-w) + 燃料×w` なので、
        # 「歩数/S + 燃料/F」と同じ向きにするには
        #
        #     w / (W - w) = S / F   →   w = W × S / (S + F)
        #
        # でなければいけません。S=F（どちらも同じだけ残っている）なら
        # w = W/2 で釣り合います。
        ratio = want / max(1e-9, want + float(remaining_fuel))
    else:
        # **もとの形（取り違えていました）。** S=F で w=W を返すので、
        # 「歩数を完全に無視して燃料だけ最小化」になります。
        ratio = FUEL_URGENCY * remaining_steps / max(1.0,
                                                     float(remaining_fuel))
    ratio = max(0.0, min(1.0, ratio))
    return int(round(ratio * FUEL_WEIGHT_STEPS))


# 同点のときに見る「おまけ」の桁を分けるための倍率。
#
# **なぜ必要か**
#
# 重み0（ステップ最短）では、平地と「混雑した道路」がどちらも2歩で
# 並びます。ところが燃料は平地1・道路2です。並んだときにどちらを
# 選ぶかが決まっていないので、**同じ歩数なのに燃料を倍使う道**を
# 選ぶことがありました。
#
# 実戦7試合のログを数えると、混雑した道路の移動が485回ありました。
# 1回あたり燃料1の無駄なので、最大で485燃料（上限73の試合なら
# 6.6タンク分）を捨てていた計算になります。
#
# そこでコストを「本命 × 倍率 + おまけ」の形にします。おまけは
# 歩数と燃料の和です。重み0なら歩数が本命で決まっているので
# おまけは燃料の比較になり、重み4（燃料最優先）なら逆に歩数の
# 比較になります。倍率はおまけが本命を覆さない大きさにします
# （1マスの歩数は最大4・燃料は最大2、経路長はその日の歩数以下）。
TIE_BREAK_SCALE = 1 << 24

# **同点なら、店を通る道を選びます。**
#
# 公式の処理順では、巡回車はそのマスに「いる」だけで玉を取ります
# （1.3 うどんの獲得）。狙った店でなくても、通り道にある店の玉は
# そのまま入ります。ところが経路探索は店を見ていなかったので、
# 同じ歩数・同じ燃料なら、店を通らない道を選ぶこともありました。
#
# 1 にすると「歩数 → 燃料 → 店の多さ」の順、2 にすると
# 「歩数 → 店の多さ → 燃料」の順で選びます。0 で止めます。
SPOT_TIE_BREAK = 1
# 桁を分ける倍率。経路は同じマスを通らないので、長さはマス数
# （最大 32*32=1024）以下です。1マスの「歩数+燃料」は最大6なので、
# 合計は 6144 を超えません。店の数は 1024 を超えません。
SPOT_SCALE = 1 << 11
AWAY_SCALE = 1 << 13


def _blend(steps: int, fuel: int, weight: int) -> int:
    """1マス分のコスト。weight が大きいほど燃料を重く見る。"""

    return steps * (FUEL_WEIGHT_STEPS - weight) + fuel * weight


def _cell_cost(steps: int, fuel: int, weight: int, is_spot: bool) -> int:
    """1マスを出るときの総合コスト。

    「本命 → おまけ1 → おまけ2」を1つの整数に詰めます。桁が混ざらない
    ように、下の桁の最大値より大きい倍率をかけています。
    """

    main = _blend(steps, fuel, weight)
    away = 0 if is_spot else 1
    if SPOT_TIE_BREAK == 2:
        return main * TIE_BREAK_SCALE + away * AWAY_SCALE + steps + fuel
    if SPOT_TIE_BREAK == 1:
        return main * TIE_BREAK_SCALE + (steps + fuel) * SPOT_SCALE + away
    return main * TIE_BREAK_SCALE + steps + fuel


@dataclass(frozen=True)
class PathTable:
    """ある始点からの最短経路表。"""

    origin: int
    steps: dict[int, int]                  # セル -> 到達に要するステップ数
    fuel: dict[int, int]                   # セル -> 到達に要する燃料
    previous: dict[int, tuple[int, int]]   # セル -> (直前セル, その時の方向)
    # 行き先 -> 方向の列（作り直しを避けるための覚え書き）
    _ways: dict = field(default_factory=dict, compare=False, repr=False)
    # 行き先 -> その経路の途中で踏む店（ビット列）
    _passed: dict = field(default_factory=dict, compare=False, repr=False)

    def reachable(self, position: int) -> bool:
        return position in self.steps

    def directions_to(self, position: int) -> list[int]:
        """始点から position までの移動方向の列を返す。

        **同じ行き先は何度も聞かれます。** 1日ぶんの再現で、同じ経路を
        引き直すだけで全体の7%を使っていました（第1958の32x32で
        1手101回）。表そのものは道路状態と始点ごとに覚えてあるので、
        方向の列もここに置けます。
        呼んだ側が `pop(0)` で削るので、**写しを返します**。
        """

        if position == self.origin:
            return []
        got = self._ways.get(position)
        if got is not None:
            return list(got)
        directions: list[int] = []
        current = position
        while current != self.origin:
            entry = self.previous.get(current)
            if entry is None:
                raise KeyError(f"セル{position}へは到達できません")
            previous_position, direction = entry
            directions.append(direction)
            current = previous_position
        directions.reverse()
        self._ways[position] = directions
        return list(directions)



    def path_spots(self, position: int, spot_bits: dict[int, int]) -> int:
        """始点から position までの経路で**踏む店**（ビット列）。

        **なぜ要るか**

        車は歩いた先の店を無条件で拾います。実測で、玉のうち通りがかりで
        拾ったものが8〜42%ありました（第1958の32x32・3日目は42%）。
        ところが順路の設計は歩数だけで組み立てていて、この拾いを
        数えていませんでした。だから「訪問数を増やす」「歩数を縮める」
        という見立てがどれも外れていました。

        経路の途中で踏む店が分かれば、差し込みの損得を正しく出せます。

            得 = |店(a→x) ∪ 店(x→b) ∪ {x}| − |店(a→b)|
            損 = 増える歩数

        `spot_bits` は「マス → そのマスのビット」の表です。始点のマスは
        含めません（そこはもう居る場所なので、改めて拾う話ではない）。
        """

        # **別のビット表で作った覚え書きは使いません。**
        # 覚え書きの鍵は行き先だけなので、渡される `spot_bits` が
        # 違えば答えも違います。表そのものは道路状態と始点ごとに
        # 使い回されるので、ここを見ないと取り違えが起きます。
        # 覚え書きの中に「どのビット表で作ったか」を鍵 None で置きます
        # （行き先はマス番号なので None とはぶつかりません）。この表は
        # 凍らせてあるので、入れ替えではなく中身を書き換えます。
        if self._passed.get(None) is not spot_bits:
            self._passed.clear()
            self._passed[None] = spot_bits
        got = self._passed.get(position)
        if got is not None:
            return got
        bits = 0
        current = position
        while current != self.origin:
            bits |= spot_bits.get(current, 0)
            entry = self.previous.get(current)
            if entry is None:
                bits = 0
                break
            current = entry[0]
        self._passed[position] = bits
        return bits


def road_key(road_status: dict[int, RoadStatus]) -> tuple[tuple[int, int], ...]:
    """道路状態を辞書からハッシュ可能なキーに変換する。順調は省略して短くする。"""

    return tuple(
        sorted(
            (position, int(status))
            for position, status in road_status.items()
            if status != RoadStatus.SMOOTH
        )
    )


# 経路表のキャッシュに使ってよい「セル数」の合計。
# 1つの表はマップのセル数に比例したメモリを使い、実測で 1セルあたり約150バイト。
# 20万セルなら 1プロセスあたり約30MBに収まります。
#
# ここを表の「個数」で制限すると事故ります。32x32 のマップでは1つの表が
# 150KB あるので、4096個まで貯めると1プロセスで610MB、64並列なら38GBになり、
# メモリ不足でプロセスごと殺されます（実際に起きました）。
# 経路表をいくつ貯めるか。マスの数で割って決めます。
#
# **実測で 200,000 → 650,000 に増やしました。**
#
# 32x32（1024マス）だと上限195件しか貯まらず、1案を作るあいだに
# **4,646回も作り直して**いました。実戦でも32x32の長い日は候補が
# 90〜500件しか作れず、短い日の5,000件と比べて桁が違いました。
#
#   上限 195: 1案446ミリ秒 / 作り直し4,646回
#   上限 600: 1案268ミリ秒 / 作り直し  598回  ← 1.7倍速い
#
# 経路表1つは15KB程度なので、600件でも9MB。子プロセス11個でも
# 100MB弱に収まります。
CACHE_CELL_BUDGET = 650_000


class PathFinder:
    """1つの問題に対する経路探索器。日ごとに道路状態が変わるので状態別にキャッシュする。"""

    def __init__(self, problem: Problem, cache_limit: int | None = None) -> None:
        self.problem = problem
        self.neighbors = walkable_neighbors(problem)
        self.terrains = tuple(problem.terrain_at(p) for p in range(problem.cell_count))
        self._cache: dict[tuple[tuple[tuple[int, int], ...], int], PathTable] = {}
        # 店のあるマス。同点のとき「店を通る道」を選ぶために使います。
        self._spot_cells = frozenset(spot.position for spot in problem.spots)
        # **店のビット表は1つだけ作ります。**
        # `PathTable.path_spots` の覚え書きは渡された表と同じ物かを
        # 見るので、呼ぶ側が毎回作り直すと覚え書きが毎回捨てられます。
        self.spot_bits: dict[int, int] = {
            spot.position: 1 << number
            for number, spot in enumerate(problem.spots)}
        if cache_limit is None:
            # マップが大きいほど1つの表が重いので、貯める数を減らす。
            cache_limit = max(64, CACHE_CELL_BUDGET // max(1, problem.cell_count))
        self._cache_limit = cache_limit
        self.compute_count = 0
        self.query_count = 0
        # 直前に使った道路状態とそのキー（作り直しを避けるため）
        self._last_road_object: object = None
        self._last_road_key: tuple = ()
        # 道路状態の中身 → 小さな番号。
        #
        # **キャッシュの鍵にタプルをそのまま使うと重い。** 道路状態の
        # 鍵は「混んでいるマスと度合い」の並びで、32x32では80要素
        # ほどになります。辞書を引くたびにその長さぶんハッシュを
        # 計算するので、1回8マイクロ秒かかっていました（1手に178回
        # 引くので、それだけで全体の18%）。
        #
        # 中身から作った鍵に**番号**を振り、鍵は整数3つにします。
        # 同じ中身なら必ず同じ番号なので、取り違えは起きません。
        self._road_ids: dict[tuple, int] = {}
        self._last_road_id: int = -1
        # 1マスのコスト表（道路状態と重みごと）
        self._cost_cache: dict[tuple[int, int], list] = {}
        # 1マスぶんの移動の値（道路状態ごと）。`move_tables` が使います。
        self._move_cache: dict[tuple, tuple] = {}

    def road_token(self, road_status: dict[int, RoadStatus]) -> tuple:
        """道路状態の鍵。**中身から作るので、使い回しても食い違いません。**

        `road_key` は毎回ソートし直すので、1手ごとに呼ぶと重くなります。
        1日のあいだ道路状態の辞書は同じ物なので、その物を**参照ごと**
        覚えておいて使い回します。参照を持つので id が別の辞書に
        使い回されることはありません（以前、`id()` だけで覚えていて
        古いコスト表を引く不具合がありました）。

        鍵と**番号**（`road_id`）はここで必ず一緒に更新します。
        別々に覚えると、片方だけ古いまま残って取り違えます。
        """

        if road_status is self._last_road_object:
            return self._last_road_key
        token = road_key(road_status)
        number = self._road_ids.get(token)
        if number is None:
            number = len(self._road_ids)
            self._road_ids[token] = number
        self._last_road_object = road_status
        self._last_road_key = token
        self._last_road_id = number
        return token

    def road_id(self, road_status: dict[int, RoadStatus]) -> int:
        """道路状態の**番号**。中身から作るので取り違えません。

        キャッシュの鍵に使います。鍵をタプルのまま使うと、引くたびに
        その長さぶんハッシュを計算します。道路状態の鍵は32x32で
        80要素ほどになり、1回8マイクロ秒かかっていました（1手に
        178回引くので、それだけで全体の18%）。整数にすれば一定です。
        """

        if road_status is not self._last_road_object:
            self.road_token(road_status)
        return self._last_road_id

    def table(
        self,
        origin: int,
        road_status: dict[int, RoadStatus],
        weight: int = 0,
    ) -> PathTable:
        """始点からの経路表。weight=0 ならステップ最短、大きいほど燃料節約。"""

        if weight < 0:
            weight = 0
        elif weight > FUEL_WEIGHT_STEPS:
            weight = FUEL_WEIGHT_STEPS

        # road_key は毎回ソートし直すので、同じ道路状態なら使い回す。
        # 道路状態は1日のあいだ変わらないので、ほぼ必ず当たります。
        # 1案を作るのに8万回も引くため、ここが効きます。
        #
        # **ここだけは `road_id` を呼ばずに直に見ます。** 1手に178回
        # 通る道なので、関数呼び出し1段がそのまま時間になります。
        # 中身は `road_id` と同じで、違う辞書が来たときだけ任せます。
        if road_status is self._last_road_object:
            road = self._last_road_id
        else:
            road = self.road_id(road_status)

        key = (road, origin, weight)
        self.query_count += 1
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        table = self._compute(origin, road_status, weight)
        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[key] = table
        return table

    def move_tables(self, road_status: dict[int, RoadStatus]) -> tuple:
        """1マスぶんの移動の値。(歩数, 燃料, 方向→移動先) を全マス分返す。

        池のマスは歩数0・燃料0・移動先の表 None です。

        **なぜここに置くか**

        `planner.plan_day` の `next_hop_cost` は、呼ばれるたびに
        「地形を引く → 渋滞を引く → 歩数を計算する → 隣をなめて
        方向を探す」をやり直していました。1日の再現で14万回以上
        呼ばれます。地形も渋滞もその日のあいだ変わらないので、
        1回作れば使い回せます。

        本番の持ち時間は時計で決まるので、**1手が速いほど多くの手を
        試せます**。16試合で手数を2〜3倍にすると玉3998→4035に
        なりました（頭打ちではありません）。速さはそのまま点数です。
        答えは1つも変わりません。

        作り直しを避ける仕組みは `table` と同じで、まず**同じ辞書の
        ままか**を見て、違えば `road_key` で引き直します。磨きの
        最中は同じ辞書が何千回も渡ってくるので、ほぼ必ず当たります。
        """

        road = self.road_id(road_status)
        cached = self._move_cache.get(road)
        if cached is not None:
            return cached

        pond = int(Terrain.POND)
        count = self.problem.cell_count
        cell_steps = [0] * count
        cell_fuel = [0] * count
        hops: list[dict[int, int] | None] = [None] * count
        for position, terrain in enumerate(self.terrains):
            if terrain == pond:
                continue
            kind = Terrain(terrain)
            cell_steps[position] = move_steps(
                kind, road_status.get(position, RoadStatus.SMOOTH))
            cell_fuel[position] = fuel_cost(kind)
            hops[position] = {
                direction: destination
                for direction, destination in self.neighbors[position]
            }
        tables = (cell_steps, cell_fuel, hops)
        if len(self._move_cache) > 8:
            self._move_cache.clear()
        self._move_cache[road] = tables
        return tables

    def _cost_table(
        self,
        road_status: dict[int, RoadStatus],
        weight: int,
    ) -> list[tuple[int, int, int] | None]:
        """各マスを出るときの (歩数, 燃料, 混ぜたコスト)。池は None。

        同じ道路状態と重みなら使い回します。探索1回ごとに作り直すと、
        それだけで全体の2割を食います。

        **キーは `id()` ではいけません。** 以前は `id(road_status)` を
        キーにしていましたが、辞書そのものは覚えていないので、その
        辞書が捨てられたあと別の辞書が同じ番地に置かれると、**別の
        道路状態の表をそのまま返して**しまいます。1日の途中で道の
        混み具合が入れ替わった表を使えば、経路も獲得も変わります。
        実際、同じ入力で再現しても試合ごとの玉数がぶれていました。
        `table` と同じ `road_key`（中身から作る鍵）に直します。
        """

        if road_status is self._last_road_object:
            fingerprint = self._last_road_key
        else:
            fingerprint = self.road_token(road_status)
        key = (fingerprint, weight)
        cached = self._cost_cache.get(key)
        if cached is not None:
            return cached

        pond = int(Terrain.POND)
        road = int(Terrain.ROAD)
        table: list[tuple[int, int, int] | None] = []
        # 道路以外は道路状態に左右されないので、地形ごとに1回だけ作る。
        plain: dict[tuple[int, bool], tuple[int, int, int]] = {}
        for position, terrain in enumerate(self.terrains):
            if terrain == pond:
                table.append(None)
                continue
            is_spot = position in self._spot_cells
            if terrain == road:
                status = road_status.get(position, RoadStatus.SMOOTH)
                steps = move_steps(terrain, status)
                burn = fuel_cost(terrain)
                table.append(
                    (steps, burn, _cell_cost(steps, burn, weight, is_spot))
                )
                continue
            entry = plain.get((terrain, is_spot))
            if entry is None:
                steps = move_steps(terrain, RoadStatus.SMOOTH)
                burn = fuel_cost(terrain)
                entry = (steps, burn, _cell_cost(steps, burn, weight, is_spot))
                plain[(terrain, is_spot)] = entry
            table.append(entry)

        if len(self._cost_cache) > 24:
            self._cost_cache.clear()
        self._cost_cache[key] = table
        return table

    def _compute(
        self,
        origin: int,
        road_status: dict[int, RoadStatus],
        weight: int = 0,
    ) -> PathTable:
        """始点から全セルへの最短経路。

        ここが処理時間の9割を占めるので、辞書ではなくリストで持ちます。
        セル番号は 0 から cell_count-1 の連番なので、リストで引けます。
        辞書の get を1回の試走で900万回呼んでいたのを無くします。
        """

        self.compute_count += 1
        neighbors = self.neighbors
        terrains = self.terrains
        cell_count = self.problem.cell_count
        unreached = 1 << 60

        best_cost = [unreached] * cell_count
        best_steps = [0] * cell_count
        best_fuel = [0] * cell_count
        previous_cell = [-1] * cell_count
        previous_direction = [-1] * cell_count
        settled = [False] * cell_count

        best_cost[origin] = 0
        frontier: list[tuple[int, int]] = [(0, origin)]
        pond = int(Terrain.POND)

        # 1マスを出るときの「歩数・燃料・コスト」は地形と道路状態だけで決まります。
        # 探索の中で60万回も計算していたので、先に表にしておきます。
        # 道路以外は道路状態に左右されないので、地形ごとに1回で済みます。
        cost_table = self._cost_table(road_status, weight)

        while frontier:
            cost, position = heappop(frontier)
            if settled[position] or cost > best_cost[position]:
                continue
            settled[position] = True
            entry = cost_table[position]
            if entry is None:
                continue
            step_cost, burn, blended = entry
            next_cost = cost + blended
            next_steps = best_steps[position] + step_cost
            next_fuel = best_fuel[position] + burn
            for direction, next_position in neighbors[position]:
                if settled[next_position]:
                    continue
                if next_cost >= best_cost[next_position]:
                    # 同じコストなら、ステップ数が少ないほうを採る。
                    if next_cost > best_cost[next_position]:
                        continue
                    if next_steps >= best_steps[next_position]:
                        continue
                best_cost[next_position] = next_cost
                best_steps[next_position] = next_steps
                best_fuel[next_position] = next_fuel
                previous_cell[next_position] = position
                previous_direction[next_position] = direction
                heappush(frontier, (next_cost, next_position))

        settled_steps = {
            index: best_steps[index] for index in range(cell_count) if settled[index]
        }
        settled_fuel = {
            index: best_fuel[index] for index in range(cell_count) if settled[index]
        }
        previous = {
            index: (previous_cell[index], previous_direction[index])
            for index in range(cell_count)
            if settled[index] and previous_cell[index] >= 0
        }
        return PathTable(
            origin=origin,
            steps=settled_steps,
            fuel=settled_fuel,
            previous=previous,
        )
