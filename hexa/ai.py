"""学習するAI本体。

構成:
  ・目標選択は線形モデルで価値を予測する。予測は公式の優先順位に合わせて
    3本（種類数 / 日別種類数 / 玉数）に分けて持つ。1本のスカラーに潰さないので、
    「玉数のために種類数を捨てる」ような歪みが起きない。
  ・経路そのものは学習しない。最短経路アルゴリズムに任せる。
    これで「速さ」はアルゴリズム、「どこを狙うか」は学習、と役割が分かれる。
  ・補給車は別の価値関数を持ち、「補給したあとにその巡回車が実際に増やした玉数」
    だけを報酬にする。動いただけ・補給しただけでは報酬が入らない。
  ・経験は models/ に貯まり続け、学習を回すほどサンプルが増えて予測が良くなる。

4体のAIは、初期重み・乱数・経験バッファがすべて独立です。同じ問題を解いても
別の判断に育つように、初期重みだけ小さくばらつかせています。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from hexa.hexgrid import neighbor
from hexa.pathfind import PathFinder, fuel_weight
from hexa.routes import (
    CRITERIA_COUNT,
    RECIPE_POOL_LIMIT,
    SEED_RECIPES,
    blend_recipes,
    build_routes,
    default_recipes,
    mutate_recipe,
)
from hexa.rules import AgentKind, RoadStatus, Terrain

# 素の材料の数。ここに「材料どうしの掛け算」を足したものが実際の特徴量になる。
# 今日まだ取っていない系列を優先して選ぶか。
#
# **実測で否定されたので False です。**（未見10問）
#
#   優先しない: 日別 77.90 / 玉 251.0  ← 採用
#   優先する  : 日別 73.00 / 玉 185.3
#
# 日別累計まで下がったのが意外でした。理由はこうだと見ています。
# 「今日まだ取っていない系列」へ無理に向かうと遠い店へ行くことになり、
# その1系列は取れても、近場を回っていれば取れたはずの**別の系列を
# 落とします。** 1歩あたりで見ると、近い店を数多く回るほうが
# 結果的に多くの系列をそろえられる、ということです。
#
# 実戦で「届くのに行かなかった」取り逃しが24日中29件ありましたが、
# それは判断の誤りではなく、**そちらへ行かないほうが得だった**可能性が高い。
_PREFER_FRESH_BRANDS = False

# 補給の合流先に「相手の経路の途中」を入れるか。
#
# 実戦1025の6日目（190歩）では、巡回車が120〜134歩も補給待ちで
# 止まっていました。補給車1台で4台を回るため、現在地まで迎えに行くと
# 順番待ちが長くなります。相手が進む先で落ち合えば待ち時間が縮みます。
#
#   途中で合流なし: 日別364 / 玉829
#   途中で合流あり: 日別364 / 玉835  ← 採用
_MEET_ON_THE_WAY = True

# 燃料がこの割合以上ある車は、補給の相手に選びません。
#
# **1.01（事実上の無効）が最良でした。**
#
# 実戦のログでは、補給830回のうち637回が「ほぼ満タンの車」に対する
# もので、無駄に見えました（燃料262で差9〜33など）。そこで追いかけるのを
# やめたところ、**玉数が大きく落ちました**。
#
#   0.80: 16x16 95% / 24x24 90% / 32x32 87% / 1067 84%
#   0.95: 16x16 94% / 24x24 89% / 32x32 84% / 1067 92%
#   1.01: 16x16 96% / 24x24 90% / 32x32 93% / 1067 92%  ← 採用
#
# 満タンに近い車への補給は無駄ではなく、**先回りして満タンを保つ**
# 動きでした。補給車はどのみち暇なので、ついでに満たしておくほうが
# あとで効きます。
SUPPLY_SKIP_RATIO = 1.01

# 残り歩数に余裕があるとき、未取得系列の店にも2台目を送るか。
#
# **実測で効かなかったので 0 にしています。**
# 1356で 玉327 → 322（条件1.0）→ 313（条件2.0）と悪化しました。
STACK_WHEN_ROOMY = 0.0

TOURING_BASE_COUNT = 40
SUPPLY_BASE_COUNT = 18

# 「材料どうしの積」の組み合わせ。AIを作るときに1つ選び、モデルに保存されます。
#
# 線形モデルは「新しい系列なら遠くても行く、既出の系列なら近いときだけ行く」
# のような条件付きの判断が書けません。材料どうしの積を足せば書けるようになり、
# 覚えられる判断が細かくなります。
#
# ただし重みが増えるぶん、同じ経験量では当てずっぽうも増えます。実測:
#   simple ・300エピソード : AIだけで玉170.2（貪欲は163.4）
#   cross  ・390エピソード : AIだけで玉154.4
#   cross  ・640エピソード : AIだけで玉149.9
# 短い学習では simple が勝ちます。交差する点は確認できていません。
CAPACITY_PRESETS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    # 素の材料だけ。巡回車23・補給車13の重み。
    "simple": ((), ()),
    # よく効く材料どうしの積を足す。巡回車68・補給車34の重み。
    # 作戦の重み（18〜25）と、マップの素性（28〜34）を掛け合わせる。
    # 「山地だらけなら燃料節約」「密集地なら周回」のような使い分けが書ける。
    "cross": ((2, 3, 7, 10, 13, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 34),
              (1, 3, 4, 6, 13, 15, 16)),
    # 素の材料をほぼ全部掛け合わせる。巡回車176・補給車68の重み。
    # 何十万エピソードも回せるとき向け。
    "large": (
        (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
         37, 38, 39),
        (1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17),
    ),
}
DEFAULT_CAPACITY = "simple"


def touring_feature_count(cross: tuple[int, ...]) -> int:
    return TOURING_BASE_COUNT + _cross_count(cross)


def supply_feature_count(cross: tuple[int, ...]) -> int:
    return SUPPLY_BASE_COUNT + _cross_count(cross)


def _cross_count(indexes: tuple[int, ...]) -> int:
    size = len(indexes)
    return size * (size + 1) // 2


def _step_scale(features: list[float]) -> float:
    """一歩の大きさを材料の大きさで割るための係数。

    1.0 を足しているのは、材料がすべて0に近いときに割り算が
    暴れないようにするためです（ゼロ除算の防止も兼ねます）。
    """

    total = 1.0
    for value in features:
        total += value * value
    return 1.0 / total


def _expand(base: list[float], indexes: tuple[int, ...]) -> list[float]:
    """素の材料に、指定した材料どうしの積を足した特徴量を返す。"""

    expanded = list(base)
    for position, first in enumerate(indexes):
        value = base[first]
        for second in indexes[position:]:
            expanded.append(value * base[second])
    return expanded


# 掛け算の重みは0から始め、学習の効きも弱くする。
# こうすると学習の最初は素の材料だけの素直なモデルとして動き、
# 経験が貯まってきたところで細かい使い分けが少しずつ乗ってくる。
# 最初から掛け算を強く動かすと、材料が増えたぶん当てずっぽうが増えて
# かえって弱くなる（実測で確認済み）。
CROSS_LEARNING_SCALE = 0.25
HEAD_COUNT = 3          # 種類数 / 日別種類数 / 玉数
# 予測値を比べるときの刻み幅。これより小さい差は「同じ」とみなして次の優先順位へ送る。
# 線形モデルの予測には必ず誤差が乗るので、刻みが細かすぎると誤差だけで
# 上位の優先順位が決まってしまい、下位の優先順位が一切見られなくなる。
LEXICO_TOLERANCE = 0.12


@dataclass
class TouringDecision:
    target: int | None
    features: list[float]
    candidate_features: list[list[float]]
    recipe_index: int = -1


@dataclass
class SupplyDecision:
    target: int | None
    target_agent: int | None
    features: list[float]
    candidate_features: list[list[float]]


def _dot(weights: list[float], features: list[float]) -> float:
    total = 0.0
    for weight, value in zip(weights, features):
        total += weight * value
    return total


@dataclass
class Brain:
    """1体分のAIの重みと経験。"""

    name: str
    seed: int
    # 探索の強さ。4体で違う値にしてあるので、同じ経験からでも別の癖に育つ。
    # 小さい個体は手堅く、大きい個体は変わった手を試してから収束する。
    explore_scale: float = 1.0
    # このAIが使う「材料どうしの積」の組み合わせ。
    # モデルの形そのものなので、AIごとに保存して読み書きします。
    touring_cross: tuple[int, ...] = ()
    supply_cross: tuple[int, ...] = ()
    # 手持ちの作戦。学習の中で変異し、増えていく。
    recipes: list[list[float]] = field(default_factory=default_recipes)
    # 作戦ごとの成績 [使われた回数, 報酬の合計, これまでの最高]。
    # **平均が悪くても消しません。** ある作戦が平均で負けていても、
    # 特定の地形や特定の日だけ刺さることがあるからです。
    recipe_stats: list[list[float]] = field(default_factory=list)
    touring_heads: list[list[float]] = field(default_factory=list)
    supply_weights: list[float] = field(default_factory=list)
    # "エージェント数:補給車数" -> [試行回数, 種類数合計, 日別種類数合計, 玉数合計]
    kind_stats: dict[str, list[float]] = field(default_factory=dict)
    trained_episodes: int = 0
    touring_samples: int = 0
    supply_samples: int = 0
    history: list[dict] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        name: str,
        seed: int,
        explore_scale: float = 1.0,
        capacity: str = DEFAULT_CAPACITY,
    ) -> "Brain":
        touring_cross, supply_cross = CAPACITY_PRESETS.get(
            capacity, CAPACITY_PRESETS[DEFAULT_CAPACITY]
        )
        rng = random.Random(seed)
        heads = [
            [
                rng.uniform(-0.05, 0.05) if index < TOURING_BASE_COUNT else 0.0
                for index in range(touring_feature_count(touring_cross))
            ]
            for _ in range(HEAD_COUNT)
        ]
        supply = [
            rng.uniform(-0.05, 0.05) if index < SUPPLY_BASE_COUNT else 0.0
            for index in range(supply_feature_count(supply_cross))
        ]
        return cls(
            name=name, seed=seed, explore_scale=explore_scale,
            touring_cross=touring_cross, supply_cross=supply_cross,
            touring_heads=heads, supply_weights=supply,
        )

    def predict_touring(self, base: list[float]) -> tuple[float, float, float]:
        """素の材料を受け取り、内部で掛け算を展開してから予測する。

        呼び出し側は素の材料（23個）だけを持てばよく、経験としてそれを
        貯めておけます。large では展開後が176個になるので、
        展開後を貯めると7.6倍のメモリを食います。
        """

        # **特徴を作らずに、そのまま内積を取ります。**
        #
        # 以前は「素の材料に積を足したリスト」を作ってから内積を
        # 3回取っていました。1案を作るあいだに `append` が527万回
        # 呼ばれ、_expand だけで全体の2割を占めます。
        #
        # 掛け算の項は作った端から足せるので、リストは要りません。
        # 16x16 で1案32ミリ秒 → 候補を増やす余地が生まれます。
        heads = self.touring_heads
        first = heads[0]
        second = heads[1]
        third = heads[2]
        limit = len(first)
        count = min(len(base), limit)
        total0 = total1 = total2 = 0.0
        for index in range(count):
            value = base[index]
            if value:
                total0 += first[index] * value
                total1 += second[index] * value
                total2 += third[index] * value
        at = len(base)
        cross = self.touring_cross
        for position in range(len(cross)):
            left = base[cross[position]]
            for other in range(position, len(cross)):
                if at >= limit:
                    return (total0, total1, total2)
                if left:
                    product = left * base[cross[other]]
                    if product:
                        total0 += first[at] * product
                        total1 += second[at] * product
                        total2 += third[at] * product
                at += 1
        return (total0, total1, total2)

    def predict_supply(self, base: list[float]) -> float:
        # 巡回車と同じく、特徴を作らずに内積を取ります。
        #
        # 重みが特徴より短いことがあるので（学習の途中で材料が
        # 増えた場合）、短いほうで打ち切ります。`_dot` が zip で
        # 同じことをしていました。
        weights = self.supply_weights
        limit = len(weights)
        total = 0.0
        count = min(len(base), limit)
        for index in range(count):
            value = base[index]
            if value:
                total += weights[index] * value
        at = len(base)
        cross = self.supply_cross
        for position in range(len(cross)):
            left = base[cross[position]]
            for other in range(position, len(cross)):
                if at >= limit:
                    return total
                if left:
                    product = left * base[cross[other]]
                    if product:
                        total += weights[at] * product
                at += 1
        return total

    def update_touring(self, base: list[float], rewards: list[float], alpha: float) -> None:
        """重みを更新する。材料の大きさで割ってから進む（正規化）。

        **なぜ割るのか**

        以前は `重み += 学習率 × 誤差 × 材料` とそのまま足していました。
        この形が安定するのは `学習率 < 2 / |材料|^2` のときだけです。
        材料の数が増えると |材料|^2 も増えるので、同じ学習率でも破綻します。

            simple  材料  40個 → |材料|^2 = 15.2  → 上限 0.132（0.08 は安全）
            cross   材料 211個 → |材料|^2 = 30.2  → 上限 0.066（0.08 は危険）
            large   材料 781個 → |材料|^2 = 121.5 → 上限 0.017（0.08 は危険）

        実際、large で60万エピソード回したログでは、2万回目から最後まで
        ほぼ毎ラウンド「重みが壊れたAIがいます」が出続け、そのたびに
        最良へ巻き戻されて**61時間かけて一歩も進みませんでした。**

        |材料|^2 で割れば、材料が何個でも一歩の大きさが揃います。
        学習率は「誤差のどれだけを一度に埋めるか」の意味になり、
        1未満なら必ず収束します。
        """

        features = _expand(base, self.touring_cross)
        scale = _step_scale(features)
        cross_alpha = alpha * CROSS_LEARNING_SCALE
        for head_index in range(HEAD_COUNT):
            weights = self.touring_heads[head_index]
            error = rewards[head_index] - _dot(weights, features)
            step = error * scale
            for i, value in enumerate(features):
                if value == 0.0:
                    continue
                rate = alpha if i < TOURING_BASE_COUNT else cross_alpha
                weights[i] += rate * step * value
        self.touring_samples += 1

    def update_supply(self, base: list[float], reward: float, alpha: float) -> None:
        """補給車側の更新。考え方は update_touring と同じ。"""

        features = _expand(base, self.supply_cross)
        scale = _step_scale(features)
        cross_alpha = alpha * CROSS_LEARNING_SCALE
        error = reward - _dot(self.supply_weights, features)
        step = error * scale
        for i, value in enumerate(features):
            if value == 0.0:
                continue
            rate = alpha if i < SUPPLY_BASE_COUNT else cross_alpha
            self.supply_weights[i] += rate * step * value
        self.supply_samples += 1

    def grow(self, capacity: str) -> None:
        """学習を引き継いだまま、AIを大きくする。

        新しく増える重みはすべて0から始めます。掛け算の項に0を掛けると
        必ず0なので、**広げた直後の予測値はまったく変わりません**。
        素の材料23個に対する重みはそのまま残るからです。

            広げる前: sum(w[i] * x[i])                 （i は素の材料）
            広げた後: sum(w[i] * x[i]) + sum(0 * 積)   = 同じ値

        つまり「大きくしたせいで弱くなる」ことが原理的に起きません。
        そこから学習を続けると、掛け算の重みだけが0から動き出します。
        """

        touring_cross, supply_cross = CAPACITY_PRESETS[capacity]
        if len(touring_cross) < len(self.touring_cross):
            raise ValueError("いまより小さい大きさには変えられません")

        new_touring = touring_feature_count(touring_cross)
        new_supply = supply_feature_count(supply_cross)
        for head_index in range(HEAD_COUNT):
            head = self.touring_heads[head_index]
            # 素の材料の重みだけを残し、掛け算の重みは0から作り直す。
            base_weights = head[:TOURING_BASE_COUNT]
            self.touring_heads[head_index] = base_weights + [0.0] * (
                new_touring - TOURING_BASE_COUNT
            )
        base_supply = self.supply_weights[:SUPPLY_BASE_COUNT]
        self.supply_weights = base_supply + [0.0] * (new_supply - SUPPLY_BASE_COUNT)
        self.touring_cross = touring_cross
        self.supply_cross = supply_cross

    def record_recipe_result(self, index: int, reward: float) -> None:
        """作戦の成績を記録する。悪い結果もそのまま残す。

        悪かったからといって作戦を消すことはしません。
        「この地形では負けるが、別の地形では勝つ」作戦は普通にあります。
        平均だけ見て切り捨てると、その勝てる場面ごと失います。
        """

        while len(self.recipe_stats) < len(self.recipes):
            self.recipe_stats.append([0.0, 0.0, 0.0])
        if not 0 <= index < len(self.recipe_stats):
            return
        entry = self.recipe_stats[index]
        entry[0] += 1.0
        entry[1] += reward
        entry[2] = max(entry[2], reward)

    def working_recipes(self, limit: int = 12) -> list[list[float]]:
        """1回の判断で試す作戦。出発点の8つ＋成績の良かったものを少し。

        手持ちは増え続けますが（最大48）、毎回全部を試すと判断が重くなり、
        1日に検討できる候補の数が激減します。ここで数を絞ります。
        絞るのは「試す対象」だけで、手持ちからは何も消えません。
        """

        seeds = len(SEED_RECIPES)
        chosen = [list(item) for item in self.recipes[:seeds]]
        extras = []
        for index in range(seeds, len(self.recipes)):
            if index < len(self.recipe_stats):
                used, total, best = self.recipe_stats[index]
            else:
                used, total, best = 0.0, 0.0, 0.0
            # 使われたことのないものも一定の確率で混ぜたいので、
            # 最高記録を第一、平均を第二に見る。
            extras.append(((best, total / used if used else 0.0), index))
        extras.sort(reverse=True)
        for _key, index in extras[: max(0, limit - seeds)]:
            chosen.append(list(self.recipes[index]))
        return chosen

    def remember_recipe(self, recipe: list[float]) -> int:
        """その場で作った作戦を手持ちに加える。

        上限に達したら、平均が悪いものではなく
        **一度も良い結果を出していないもの**を置き換えます。
        出発点の作戦は守ります。
        """

        while len(self.recipe_stats) < len(self.recipes):
            self.recipe_stats.append([0.0, 0.0, 0.0])
        for index, existing in enumerate(self.recipes):
            if all(abs(a - b) < 1e-6 for a, b in zip(existing, recipe)):
                return index
        if len(self.recipes) < RECIPE_POOL_LIMIT:
            self.recipes.append(list(recipe))
            self.recipe_stats.append([0.0, 0.0, 0.0])
            return len(self.recipes) - 1

        protected = len(SEED_RECIPES)
        worst = None
        worst_key = None
        for index in range(protected, len(self.recipes)):
            used, total, best = self.recipe_stats[index]
            key = (best, total / used if used else 0.0)
            if worst_key is None or key < worst_key:
                worst_key = key
                worst = index
        if worst is None:
            return 0
        self.recipes[worst] = list(recipe)
        self.recipe_stats[worst] = [0.0, 0.0, 0.0]
        return worst

    def record_kind_result(
        self, agent_count: int, supply_count: int, score: tuple[int, int, int]
    ) -> None:
        key = f"{agent_count}:{supply_count}"
        entry = self.kind_stats.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        entry[0] += 1.0
        entry[1] += score[0]
        entry[2] += score[1]
        entry[3] += score[2]

    def best_supply_count(self, agent_count: int, rng: random.Random, explore: float = 0.0) -> int:
        """学習した平均成績から補給車の台数を選ぶ。台数0も候補に含める。"""

        options = list(range(0, agent_count))     # 巡回車を最低1台は残す
        if explore > 0.0 and rng.random() < explore:
            return rng.choice(options)
        untried = [
            supply_count
            for supply_count in options
            if self.kind_stats.get(f"{agent_count}:{supply_count}", [0.0])[0] == 0
        ]
        if untried:
            # 未経験の構成は必ず一度は試す。0台に固まって補給車を学ばない事故を防ぐ。
            return rng.choice(untried)

        best_option = options[0]
        best_value: tuple[float, float, float] | None = None
        for supply_count in options:
            entry = self.kind_stats[f"{agent_count}:{supply_count}"]
            trials = entry[0]
            # 公式の優先順位のまま平均を比べる。1つのスカラーには潰さない。
            value = (entry[1] / trials, entry[2] / trials, entry[3] / trials)
            if best_value is None or value > best_value:
                best_value = value
                best_option = supply_count
        return best_option

    def to_json_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "exploreScale": self.explore_scale,
            "recipes": [list(recipe) for recipe in self.recipes],
            "recipeStats": [list(item) for item in self.recipe_stats],
            "touringCross": list(self.touring_cross),
            "supplyCross": list(self.supply_cross),
            "touringHeads": self.touring_heads,
            "supplyWeights": self.supply_weights,
            "kindStats": self.kind_stats,
            "trainedEpisodes": self.trained_episodes,
            "touringSamples": self.touring_samples,
            "supplySamples": self.supply_samples,
            "history": self.history[-200:],
        }

    @classmethod
    def from_json_dict(cls, raw: dict) -> "Brain":
        brain = cls(
            name=raw["name"],
            seed=int(raw["seed"]),
            explore_scale=float(raw.get("exploreScale", 1.0)),
            recipes=[list(map(float, r)) for r in raw.get("recipes", default_recipes())],
            recipe_stats=[list(map(float, r)) for r in raw.get("recipeStats", [])],
            touring_cross=tuple(int(v) for v in raw.get("touringCross", ())),
            supply_cross=tuple(int(v) for v in raw.get("supplyCross", ())),
            touring_heads=[list(map(float, head)) for head in raw["touringHeads"]],
            supply_weights=[float(value) for value in raw["supplyWeights"]],
            kind_stats={
                key: [float(value) for value in values]
                for key, values in raw.get("kindStats", {}).items()
                if len(values) == 4
            },
            trained_episodes=int(raw.get("trainedEpisodes", 0)),
            touring_samples=int(raw.get("touringSamples", 0)),
            supply_samples=int(raw.get("supplySamples", 0)),
            history=list(raw.get("history", [])),
        )
        expected_touring = touring_feature_count(brain.touring_cross)
        expected_supply = supply_feature_count(brain.supply_cross)
        if len(brain.touring_heads) != HEAD_COUNT or any(
            len(head) != expected_touring for head in brain.touring_heads
        ):
            raise ValueError(f"{brain.name}: 保存済みモデルの特徴量数が現在の実装と違います")
        if len(brain.supply_weights) != expected_supply:
            raise ValueError(f"{brain.name}: 保存済みモデルの補給特徴量数が現在の実装と違います")
        return brain


class Policy:
    """1体のAIを、プランナーが呼べる方策として包むクラス。"""

    def __init__(
        self,
        brain: Brain,
        problem,
        rng: random.Random,
        epsilon: float = 0.0,
        route_budget: int = 12,
        recipes: list[list[float]] | None = None,
    ) -> None:
        # その試合用に差し替えた作戦。None なら学習済みのものを使います。
        # 本番では、良い結果を出した作戦を少し変えながら探すので、
        # ここに毎回違うものが入ります。
        self.override_recipes = recipes
        # 1回の判断で試す作戦の数。試合開始前の試走のように
        # 「たくさん回すこと自体に意味がある」場面では小さくします。
        self.route_budget = route_budget
        self.brain = brain
        self.problem = problem
        self.rng = rng
        self.epsilon = epsilon
        self.fuel_limit = max(1, problem.fuel_limit)
        # 今日の長さ。判断のたびに更新される（日ごとに変わるため）。
        self.today_steps = max(1, problem.day_steps[0])
        self.day_length = 0.0
        # マップの大きさで割って、8x8 でも 32x32 でも同じ意味になる尺度にする。
        # ここを「100ステップ」「40セル」のような固定値にすると、
        # 大きいマップでは全部1に張り付いて材料として働かなくなる。
        self.map_span = max(1, problem.width + problem.height)
        self.brand_count = max(1, len({spot.brand for spot in problem.spots}))
        self.spot_positions = [spot.position for spot in problem.spots]
        self.brand_by_position = {spot.brand: spot.brand for spot in problem.spots}
        self.spot_brand = {spot.position: spot.brand for spot in problem.spots}
        self.road_cells = {
            position
            for position in range(problem.cell_count)
            if problem.terrain_at(position) == Terrain.ROAD
        }
        self.spots_by_brand: dict[int, list[int]] = {}
        for spot in problem.spots:
            self.spots_by_brand.setdefault(spot.brand, []).append(spot.position)

        # ---- その問題そのものを表す材料 ----
        # これがないと、AIは 8x8 と 32x32 の区別がつきません。
        # 材料がすべて「マップ基準の割合」に正規化されているので、
        # 大きさも地形の構成も、盤面からは見えないためです。
        # 「大きいマップではこう、山地だらけならこう」という
        # 場面に応じた戦い方を学習させるために、問題の素性を直接渡します。
        cells = problem.cells
        total_cells = max(1, len(cells))
        self.map_features = [
            # 広さ。公式の範囲は 16〜64（縦+横）なので、そこで正規化する。
            min(1.0, max(0.0, (self.map_span - 16) / 48)),
            # 縦横の偏り。細長いマップでは経路の選択肢が変わる。
            abs(problem.height - problem.width) / max(1, problem.height + problem.width),
            sum(1 for c in cells if c == int(Terrain.POND)) / total_cells,
            sum(1 for c in cells if c == int(Terrain.ROAD)) / total_cells,
            sum(1 for c in cells if c == int(Terrain.MOUNTAIN)) / total_cells,
            # 燃料の余裕。1日目のステップ数の1〜3倍なので、そこで正規化。
            min(1.0, max(0.0, (problem.fuel_limit / max(1, problem.day_steps[0]) - 1.0) / 2.0)),
            # 渋滞のなりやすさ。基準値が小さいほど1に近づく。
            1.0 - min(1.0, (problem.jammed_threshold - 2) / 8.0),
            # 店の多さ（エージェント1体あたり何店あるか）。
            min(1.0, len(problem.spots) / max(1, problem.agent_count * 4)),
            # 系列の細かさ。1なら玉数勝負、店数と同じなら種類数勝負。
            len(self.spots_by_brand) / max(1, len(problem.spots)),
        ]

    # ---------------- 巡回車 ----------------

    def _route_features(
        self,
        *,
        route,
        recipe: list[float] | None,
        remaining: int,
        fuel: int,
        days_left_ratio: float,
        fuel_per_day: float,
        spots_left_ratio: float,
        teammate_overlap: float,
        is_wait: bool,
    ) -> list[float]:
        """1つの作戦（数店を連ねた道すじ）を表す材料。

        作戦を「種類番号」で表すと、そこにない作戦を評価できません。
        **重みの数値そのものを材料にする**ので、学習の中で生まれた
        新しい作戦も同じ物差しで評価できます。ここが天井をなくす鍵です。
        """

        remaining = max(1, remaining)
        limit = self.fuel_limit
        spots = len(route.spots) if route is not None else 0
        steps = route.steps if route is not None else 0
        need = route.fuel if route is not None else 0
        roads = route.road_cells if route is not None else 0
        tactic = list(recipe) if (recipe and not is_wait) else [0.0] * CRITERIA_COUNT

        base = [
            1.0,                                             # 0 バイアス
            1.0 if is_wait else 0.0,                         # 1 待機か
            0.0 if is_wait else 1.0,                         # 2 作戦を実行するか
            min(1.0, steps / remaining),                     # 3 使う時間の割合
            1.0 - min(1.0, steps / remaining),               # 4 残る余裕
            min(1.0, need / max(1, fuel)),                   # 5 使う燃料の割合
            max(0.0, fuel - need) / limit,                   # 6 走り終えた後の燃料
            min(1.0, spots / 3.0),                           # 7 回れる店の数
            min(1.0, (route.new_brands if route else 0) / 3.0),    # 8 新しい系列
            min(1.0, (route.new_today if route else 0) / 3.0),     # 9 今日初の系列
            min(1.0, (route.stock_total if route else 0) / 12.0),  # 10 押さえる在庫
            min(1.0, roads / max(1, steps)),                 # 11 道路を踏む割合
            min(1.0, (route.congestion if route else 0.0)),  # 12 明日の混雑
            min(1.0, (route.enemy_road if route else 0.0) / 200.0),  # 13 敵の幹線を踏む量
            min(1.0, steps / max(1, spots) / 30.0),          # 14 1店あたりの手間
            min(1.0, fuel / limit),                          # 15 いまの燃料
            days_left_ratio,                                 # 16 試合の残り日数
            fuel_per_day,                                    # 17 残り1日あたりの燃料
            *tactic,                                         # 18〜25 どんな作戦か
            spots_left_ratio,                                # 26 今日まだ取れる店
            teammate_overlap,                                # 27 味方と狙いが重なる度合い
            # 28 その道すじが渋滞に弱いか（1マス幅の道路に頼っていないか）
            min(1.0, (route.detour if route else 0) / max(1, remaining)),
            # 29 今日はどれだけ長い日か（マップの広さに対する今日のステップ数）
            # これまでは「残りに対する割合」しか見ておらず、
            # 32歩の日と246歩の日を絶対値で区別できませんでした。
            # 短い日は近場を確実に、長い日は遠征、という使い分けに必要です。
            self.day_length,
            # 30 今日のうちどれだけ使ったか
            1.0 - min(1.0, remaining / max(1, self.today_steps)),
        ]
        base.extend(self.map_features)                       # 31〜39 マップの素性
        # 展開（掛け算の追加）は predict / update の側で行う。
        # ここで展開すると二重になり、重みの数と合わなくなる。
        return base

    def choose_touring_target(
        self,
        *,
        agent_index: int,
        step: int,
        remaining: int,
        position: int,
        fuel: int,
        stock: dict[int, int],
        reserved: dict[int, int],
        acquired_today: set[int],
        acquired_brands: set[int],
        brands_today: set[int],
        road_status: dict[int, RoadStatus],
        finder: PathFinder,
        kinds: tuple[AgentKind, ...] = (),
        positions: list[int] | None = None,
        day_index: int = 0,
        day_count: int = 1,
        congestion_risk: dict[int, float] | None = None,
        teammate_targets: list[int | None] | None = None,
        road_heat: dict[int, float] | None = None,
        allow_wait: bool = True,
    ) -> TouringDecision:
        days_left = max(1, day_count - day_index)
        days_left_ratio = days_left / max(1, day_count)
        fuel_per_day = min(1.0, (fuel / self.fuel_limit) / days_left)

        # 今日の長さを覚える。質問回答集その1 Q20 より、各日のステップ数は
        # W+H の1〜4倍なので、そこを 0〜1 に写す。
        if 0 <= day_index < len(self.problem.day_steps):
            self.today_steps = max(1, self.problem.day_steps[day_index])
        span = max(1, self.problem.width + self.problem.height)
        self.day_length = max(0.0, min(1.0, (self.today_steps / span - 1.0) / 3.0))

        # **在庫の数だけ、同じ店に向かえるようにします。**
        #
        # 以前は「1台でも予約していれば他の車は行けない」でした。
        # そのため在庫7の店に1台しか行かず、6玉を取り逃していました。
        # 実戦の1日目を調べると、32店すべてを訪問しているのに
        # 取り切ったのは6店だけ。取り逃し65玉のうち53玉が在庫5以上の店でした。
        #
        # 1台が1日に取れるのは1店につき1玉なので、在庫7の店には
        # 7台まで行く意味があります（巡回車の数が上限）。
        # **在庫の数だけ、同じ店に向かえるようにします。**
        #
        # 以前は「1台でも予約していれば他の車は行けない」でした。
        # そのため在庫7の店に1台しか行かず、6玉を取り逃していました。
        #
        # ただし**今日まだ誰も取っていない系列の店は、1台だけ**にします。
        # そこに2台目を向かわせると、別の系列を取りにいく車が減り、
        # 日別累計（玉数より上の指標）を落とすためです。
        # 実測で、無条件に重ねると日別が4つ減りました。
        # **台数の配分で決めます。**
        #
        # 今日まだ取っていない系列の数より、手の空いている車のほうが多いなら、
        # 余った台数は在庫の多い店へ重ねて構いません。
        # 少ないなら全台を系列集めに出します。
        #
        # 店ごとに「1台だけ」と決めていたときは、余力があるのに重ねられず
        # 玉を取り逃していました。逆に無条件に重ねると系列を落とします。
        # **全体の台数で見れば、系列に必要な分を確保したうえで余力だけ回せます。**
        # **台数の配分で決めます。**
        #
        # 今日まだ取っていない系列の数より、手の空いている車のほうが多いなら、
        # 余った台数は在庫の多い店へ重ねて構いません。
        #
        # 「1日に何回訪問できるか」で配分する案も試しましたが、
        # **実測で悪化しました**（玉470→457、制限なしでは399）。
        # 早い段階で在庫の多い店に重ねると、遠い系列へ向かう車が足りなくなり、
        # 結局そこへ戻るのに歩数を使うためだと見ています。
        fresh_brands = {
            spot.brand for spot in self.problem.spots
            if spot.brand not in brands_today
            and spot.position not in acquired_today
            and stock.get(spot.position, 0) > 0
        }
        busy = sum(1 for target in (teammate_targets or []) if target is not None)
        touring_total = sum(1 for k in kinds if k == AgentKind.TOURING) or 1
        free = max(1, touring_total - busy)
        # 系列集めに必要な台数より車が多いときだけ、未取得系列の店にも重ねる。
        fresh_limit = 1 if free <= len(fresh_brands) else 1 + (free - len(fresh_brands))
        # **残り歩数に余裕があれば、早い段階でも重ねます。**
        #
        # 実戦1356では日別152（満点）なのに玉344、1位は427でした。
        # 系列19を取るのに19訪問、残りが「2台目以降」です。
        # 私たちは24訪問、1位は34訪問。重ね方が足りません。
        #
        # 未取得の系列が残っていても、残り歩数でまだ回れるなら
        # 重ねたほうが得です。
        if STACK_WHEN_ROOMY > 0 and fresh_brands:
            span = max(4, (self.problem.width + self.problem.height) // 6)
            rounds = remaining // span
            if rounds > len(fresh_brands) * STACK_WHEN_ROOMY:
                fresh_limit = max(fresh_limit, 2)

        available = {}
        for spot in self.problem.spots:
            left = stock.get(spot.position, 0)
            if left <= 0 or spot.position in acquired_today:
                continue
            claimed = reserved.get(spot.position, 0)
            limit = left if spot.brand in brands_today else fresh_limit
            if claimed >= limit:
                continue
            available[spot.position] = left
        spots_left_ratio = len(available) / max(1, len(self.spot_positions))
        mate_targets = {
            target
            for index, target in enumerate(teammate_targets or [])
            if target is not None and index != agent_index
        }

        wait_features = self._route_features(
            route=None, recipe=None, remaining=remaining, fuel=fuel,
            days_left_ratio=days_left_ratio, fuel_per_day=fuel_per_day,
            spots_left_ratio=spots_left_ratio, teammate_overlap=0.0, is_wait=True,
        )
        candidates: list[tuple[int | None, list[float]]] = [(None, wait_features)]

        # 手持ちの作戦が増えても、1回の判断で試す数は一定に保つ。
        # 全部試すと、作戦48個で1判断20ミリ秒（8個なら5ミリ秒）。
        # 実戦で1日の候補が数千件から20件まで落ちていた原因がこれ。
        # 出発点の8つは常に入れ、残りは成績の良かったものから選ぶ。
        recipes = (
            self.override_recipes
            if self.override_recipes is not None
            else self.brain.working_recipes(self.route_budget)
        )
        if self.epsilon > 0.0 and len(recipes) >= 2:
            for _ in range(2):
                blended = blend_recipes(
                    self.rng.choice(recipes), self.rng.choice(recipes), self.rng
                )
                recipes.append(mutate_recipe(blended, self.rng))

        routes = build_routes(
            self.problem, finder, road_status, position, fuel, remaining,
            available, acquired_brands, brands_today, recipes,
            congestion_risk, road_heat,
        )
        for route in routes:
            overlap = sum(1 for spot in route.spots if spot in mate_targets)
            recipe = recipes[route.recipe_index] if route.recipe_index < len(recipes) else None
            candidates.append(
                (
                    route.target,
                    self._route_features(
                        route=route, recipe=recipe, remaining=remaining, fuel=fuel,
                        days_left_ratio=days_left_ratio, fuel_per_day=fuel_per_day,
                        spots_left_ratio=spots_left_ratio,
                        teammate_overlap=min(1.0, overlap / max(1, len(route.spots))),
                        is_wait=False,
                    ),
                )
            )

        # 待つかどうかは学習に決めさせます。燃料を温存して明日に回すほうが
        # 得な場面は実際にあるからです（燃料上限が厳しい試合で顕著）。
        #
        # ただし**待機を無制限に許すと、学習前の重みしだいで
        # 一日じゅう一歩も動かない巡回車が出ます**（実戦で確認）。
        # そこで「今日どれだけ待ったか」に上限を設け、
        # 使い切ったら待機を候補から外します。
        # 学習は上限の内側で自由に判断でき、事故は構造的に起きません。
        # **今日まだ取っていない系列に届くなら、待機は選ばせません。**
        # 日別累計は公式の第2優先で、玉数より上です。実測でも、
        # 無条件に待たせると玉数は増えるが日別累計が下がりました
        # （日別76.50→75.42 / 玉234.8→238.8）。
        # 実戦で1位を取った2試合はどちらも日別累計で勝っているので、
        # ここを譲るわけにはいきません。
        #
        # 今日の系列を取り切ったあとなら、待って燃料を温存するかどうかは
        # 学習に任せます。そこでは日別累計は減りません。
        # 道すじに含まれるかではなく、**まだ残っている店そのもの**で判定します。
        # 道すじは作戦しだいで新しい系列を含まないことがあるので、
        # そこで判定すると取りこぼしが起きます。
        # 味方が予約済みの店も数に入れます。予約したからといって
        # そこへ着ける保証はないので、予約を理由に待たせると
        # その系列を丸ごと落とすことがあります。
        fresh_brand_reachable = any(
            spot.brand not in brands_today
            and spot.position not in acquired_today
            and stock.get(spot.position, 0) > 0
            for spot in self.problem.spots
        )
        if fresh_brand_reachable or not allow_wait:
            movable = [item for item in candidates if item[0] is not None]
            if movable:
                candidates = movable

        # **今日まだ取っていない系列に届く道すじがあるなら、そこから選びます。**
        #
        # 日別累計は公式の第2優先で、玉数より上です。1系列を今日そろえれば
        # 日別累計が +1 されるので、玉数が何玉増えるより価値があります。
        #
        # 実戦で「届くのに行かなかった」取り逃しが24日中29件ありました
        # （届かなかったのは0件）。距離の問題ではなく、選び方の問題です。
        # 学習前の重みでは、せっかく良い候補が出ていても選べません。
        #
        # どの系列を取りにいくか、何歩かけるかは、引き続き学習が決めます。
        if _PREFER_FRESH_BRANDS and routes:
            fresh = [
                item for item, route in zip(candidates[1:], routes)
                if route.new_today > 0
            ]
            if fresh:
                candidates = fresh

        target, features = self._select_lexicographic(candidates)
        chosen = -1
        for route in routes:
            if route.target == target and target is not None:
                recipe = recipes[route.recipe_index] if route.recipe_index < len(recipes) else None
                if recipe is not None:
                    # その場で作った変異が選ばれたら、手持ちに加える。
                    # 良い作戦が定着し、学習を重ねるほど手札が増えていく。
                    chosen = self.brain.remember_recipe(recipe)
                break
        return TouringDecision(
            target=target,
            features=features,
            candidate_features=[item[1] for item in candidates],
            recipe_index=chosen,
        )

    def _congestion_on_path(
        self,
        table,
        target: int,
        congestion_risk: dict[int, float] | None,
    ) -> float:
        """経路上の道路セルが、明日どれくらい混みそうかの平均。

        今日その道を通ると自分も交通量を増やすので、明日また同じ道が要るなら
        避けたほうがよい、という判断の材料になります。
        """

        if not congestion_risk:
            return 0.0
        try:
            directions = table.directions_to(target)
        except KeyError:
            return 0.0
        if not directions:
            return 0.0
        position = table.origin
        total = 0.0
        for direction in directions:
            total += congestion_risk.get(position, 0.0)
            next_position = neighbor(position, direction, self.problem.width, self.problem.height)
            if next_position is None:
                break
            position = next_position
        return min(1.0, total / len(directions))

    def _supply_distance_ratio(
        self,
        target: int,
        supply_positions: list[int],
        finder: PathFinder,
        road_status: dict[int, RoadStatus],
        remaining: int,
    ) -> float:
        """目標から最寄りの補給車までの距離。近いほど遠出しやすい。"""

        if not supply_positions:
            return 1.0
        best = None
        for supply_position in supply_positions:
            supply_table = finder.table(supply_position, road_status)
            steps = supply_table.steps.get(target)
            if steps is None:
                continue
            if best is None or steps < best:
                best = steps
        if best is None:
            return 1.0
        return min(1.0, best / max(1, remaining))

    def _select_lexicographic(
        self,
        candidates: list[tuple[int | None, list[float]]],
    ) -> tuple[int | None, list[float]]:
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return self.rng.choice(candidates)

        best_index = 0
        best_key: tuple[tuple[int, int, int], int] | None = None
        for index, (target, features) in enumerate(candidates):
            # 予測が横並びのときは、待機よりも実際に店へ向かうほうを選ぶ。
            # 学習前や、どの店も似た価値に見える場面で、
            # 「とりあえず待つ」に倒れて一歩も動かなくなるのを防ぐため。
            key = (self._bucket(self.brain.predict_touring(features)), 0 if target is None else 1)
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        return candidates[best_index]

    @staticmethod
    def _bucket(values: tuple[float, float, float]) -> tuple[int, int, int]:
        """予測値を比較用の階段値に変換する。

        報酬は「その選択のあとに増えた種類数・日別種類数・玉数」なので、
        真の値は必ず0以上です。線形モデルは平気で負の値を出すので、0で切ります。

        これを忘れると、獲得済みの系列しか残っていない場面で
        「行く」の第1優先（種類数）だけがわずかに負になり、
        待機（ほぼ0）に第1優先で負けて、第2・第3優先が一度も見られなくなります。
        結果としてエージェントが一歩も動かなくなります。
        """

        return tuple(
            int(math.floor(max(0.0, value) / LEXICO_TOLERANCE)) for value in values
        )

    # ---------------- 補給車 ----------------

    def _supply_features(
        self,
        *,
        travel_steps: int,
        remaining: int,
        deficit_ratio: float,
        projected_fuel_ratio: float,
        target_is_stuck: bool,
        reachable_spots: int,
        arrives_in_time: bool,
        margin_ratio: float,
        already_claimed: bool,
        days_left_ratio: float,
        target_spot_distance: float,
        is_wait: bool,
    ) -> list[float]:
        remaining = max(1, remaining)
        base = [
            1.0,                                          # 0 バイアス
            0.0 if is_wait else 1.0,                      # 1 補給に行く選択か
            1.0 if is_wait else 0.0,                      # 2 待機の選択か
            min(1.0, travel_steps / remaining),           # 3 合流までの時間
            deficit_ratio,                                # 4 合流時点で不足する燃料
            projected_fuel_ratio,                         # 5 合流時点の相手の残燃料
            1.0 if target_is_stuck else 0.0,              # 6 相手が動けない状態か
            # 店の数はマップによって違うので、その問題の店数で割る。
            min(1.0, reachable_spots / max(1, len(self.spot_positions))),  # 7 狙える店の割合
            1.0 if arrives_in_time else 0.0,              # 8 相手より先に着けるか
            margin_ratio,                                 # 9 どれだけ先に着けるか
            1.0 if already_claimed else 0.0,              # 10 味方の補給車が担当済みか
            days_left_ratio,                              # 11 試合の残り日数の割合
            target_spot_distance,                         # 12 相手が次に狙える店までの距離
            # 13〜17 問題の素性のうち、補給の判断に効くものだけ
            self.map_features[0],                         # 13 マップの広さ
            self.map_features[3],                         # 14 道路の割合
            self.map_features[4],                         # 15 山地の割合
            self.map_features[5],                         # 16 燃料の余裕
            self.map_features[6],                         # 17 渋滞のなりやすさ
        ]
        return base

    def choose_supply_target(
        self,
        *,
        agent_index: int,
        step: int,
        remaining: int,
        position: int,
        kinds: tuple[AgentKind, ...],
        positions: list[int],
        fuels: list[int],
        plan_targets: list[int | None],
        supply_claims: dict[int, int],
        road_status: dict[int, RoadStatus],
        finder: PathFinder,
        day_index: int = 0,
        day_count: int = 1,
        road_heat: dict[int, float] | None = None,
    ) -> SupplyDecision:
        table = finder.table(position, road_status)
        days_left_ratio = max(1, day_count - day_index) / max(1, day_count)
        wait_features = self._supply_features(
            travel_steps=0, remaining=remaining, deficit_ratio=0.0,
            projected_fuel_ratio=1.0, target_is_stuck=False, reachable_spots=0,
            arrives_in_time=False, margin_ratio=0.0, already_claimed=False,
            days_left_ratio=days_left_ratio, target_spot_distance=1.0, is_wait=True,
        )
        candidates: list[tuple[int | None, int | None, list[float]]] = [
            (None, None, wait_features)
        ]

        for other in range(len(kinds)):
            if kinds[other] != AgentKind.TOURING:
                continue
            # **燃料が足りている車は追いかけません。**
            #
            # 実戦のログで、補給830回のうち637回が「ほぼ満タンの車」に
            # 対するものでした（燃料262で差9〜33など）。補給車が同じマスに
            # 居るだけで自動的に起きるので、追いかけるだけ歩数の無駄です。
            # その間、本当に困っている車を待たせています。
            if fuels[other] >= self.fuel_limit * SUPPLY_SKIP_RATIO:
                continue
            # 合流地点は、相手の現在地か、相手が向かっている目標セル。
            #
            # **その途中のセルも候補に入れます。**
            # 実戦1025の6日目（190歩）では、巡回車が120〜134歩も
            # 補給待ちで止まっていました。補給車1台で4台を回るので、
            # 現在地まで迎えに行くと順番待ちが長くなります。
            # 相手が進む先で落ち合えば、待ち時間を縮められます。
            meeting_options = {positions[other]}
            target = plan_targets[other]
            if target is not None:
                meeting_options.add(target)
                if _MEET_ON_THE_WAY:
                    path = finder.table(positions[other], road_status)
                    if path.reachable(target):
                        half = path.steps[target] // 2
                        if half > 0:
                            for cell, steps in path.steps.items():
                                if steps == half:
                                    meeting_options.add(cell)
                                    break
            other_table = finder.table(positions[other], road_status)
            reachable_spots = sum(
                1
                for spot_position in self.spot_positions
                if other_table.reachable(spot_position)
                and other_table.fuel[spot_position] <= self.fuel_limit
                and other_table.steps[spot_position] <= remaining
            )
            stuck = not any(
                other_table.reachable(spot_position)
                and other_table.fuel[spot_position] <= fuels[other]
                and other_table.steps[spot_position] <= remaining
                for spot_position in self.spot_positions
            )
            nearest_spot = min(
                (
                    other_table.steps[spot_position]
                    for spot_position in self.spot_positions
                    if other_table.reachable(spot_position)
                ),
                default=None,
            )
            spot_distance = (
                1.0 if nearest_spot is None else min(1.0, nearest_spot / max(1, remaining))
            )
            for meeting in meeting_options:
                if not table.reachable(meeting):
                    continue
                travel_steps = table.steps[meeting]
                if travel_steps > remaining:
                    continue
                # 相手が合流地点に着いた時点で残っている燃料で判断する。
                # 「いま満タンだから用がない」と切ってしまうと、補給車が
                # 一日じゅう止まったままになり、相手が空になってから慌てて動く
                # ことになる。先回りできるように、着いた時点の燃料で見る。
                other_steps = other_table.steps.get(meeting)
                other_fuel_cost = other_table.fuel.get(meeting, 0)
                projected_fuel = max(0, fuels[other] - other_fuel_cost)
                deficit = max(0, self.fuel_limit - projected_fuel)
                # 不足が0でも候補には残す。補給車は燃料を使わないので、
                # 近くで待っておくこと自体に損がない。近くにいるかどうかが、
                # あとで補給できるかどうかを決める。
                if other_steps is None:
                    other_steps = travel_steps
                arrives_in_time = travel_steps <= other_steps
                margin = (other_steps - travel_steps) / max(1, remaining)
                candidates.append(
                    (
                        meeting,
                        other,
                        self._supply_features(
                            travel_steps=travel_steps,
                            remaining=remaining,
                            deficit_ratio=deficit / self.fuel_limit,
                            projected_fuel_ratio=projected_fuel / self.fuel_limit,
                            target_is_stuck=stuck,
                            reachable_spots=reachable_spots,
                            arrives_in_time=arrives_in_time,
                            margin_ratio=max(-1.0, min(1.0, margin)),
                            already_claimed=supply_claims.get(other, 0) > 0,
                            days_left_ratio=days_left_ratio,
                            target_spot_distance=spot_distance,
                            is_wait=False,
                        ),
                    )
                )

        # 立ち往生している巡回車がいるなら、そこへ行くことを最優先にする。
        # 燃料が尽きて1店も回れない巡回車は、補給されるまで0玉のままです。
        # 実戦で「7体中2体が5日間ずっと止まったまま、在庫は60以上残っている」
        # という状態を確認しました。学習に任せていると、この最優先が
        # 身につくまで何万エピソードも無駄になります。
        # どの立ち往生車へ行くかは、引き続き学習が決めます。
        stranded = [
            item for item in candidates
            if item[0] is not None and len(item[2]) > 6 and item[2][6] >= 1.0
        ]
        if stranded:
            candidates = stranded

        # 合流できる相手が1体でもいるなら、待機は候補から外す。
        # 補給車は燃料を使わず、うどんも獲得できないので、その場に留まって
        # 得することが何もない。近くにいるかどうかだけが、あとで補給できるか
        # どうかを決める。どの相手を選ぶかは学習に任せる。
        movable = [item for item in candidates if item[0] is not None]
        if not movable and road_heat:
            # 合流する相手がいない補給車は、敵がよく踏む道路の上で待つ。
            # 補給車は燃料を使わず、うどんも獲得できないので、待つ場所を
            # 変えるだけなら自分の損はゼロ。そこに居座ると交通量が増えて、
            # 翌日その道を使う敵の移動時間だけが伸びる。
            parking = self._pick_blocking_cell(table, road_heat, remaining)
            if parking is not None:
                return SupplyDecision(
                    target=parking, target_agent=None,
                    features=wait_features, candidate_features=[wait_features],
                )
        if movable:
            # 味方の補給車がまだ担当していない巡回車があるなら、そちらを優先する。
            # 2台が同じ巡回車に張り付くと、片方の働きがまるごと無駄になる。
            unclaimed = [
                item for item in movable if supply_claims.get(item[1], 0) == 0
            ]
            candidates = unclaimed if unclaimed else movable

        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            meeting, other, features = self.rng.choice(candidates)
        else:
            best = max(
                candidates,
                key=lambda item: (self.brain.predict_supply(item[2]),
                                  0 if item[0] is None else 1),
            )
            meeting, other, features = best
        return SupplyDecision(
            target=meeting,
            target_agent=other,
            features=features,
            candidate_features=[item[2] for item in candidates],
        )

    def _nearest_teammate_distance(
        self,
        table,
        target: int,
        mate_targets: list[int],
        remaining: int,
    ) -> float:
        """この目標から、一番近い味方の目標までの距離。

        近すぎるなら味方と重なっている、遠いなら手分けできている、という
        判断の材料になります。味方がいなければ1（重なりなし）を返します。
        """

        if not mate_targets:
            return 1.0
        best = None
        for mate_target in mate_targets:
            if mate_target == target:
                return 0.0
            steps = table.steps.get(mate_target)
            if steps is None:
                continue
            own = table.steps.get(target, 0)
            gap = abs(steps - own)
            if best is None or gap < best:
                best = gap
        if best is None:
            return 1.0
        return min(1.0, best / max(1, remaining))

    def _pick_blocking_cell(
        self,
        table,
        road_heat: dict[int, float],
        remaining: int,
    ) -> int | None:
        """敵の通行量が多く、たどり着ける道路セルを選ぶ。

        補給車が合流する相手を持たないときの待機場所です。補給車は燃料を
        使わず、うどんも獲得できないので、待つ場所を変えるだけなら自分の損は
        ゼロ。そこに居座ると交通量が増えて、翌日その道を使う敵の移動時間だけが
        伸びます。
        """

        best = None
        best_key = None
        for position, heat in road_heat.items():
            if heat <= 0.0:
                continue
            steps = table.steps.get(position)
            if steps is None or steps > remaining:
                continue
            # 熱いほど良い。同じ熱さなら近いほうが良い。
            key = (heat, -steps)
            if best_key is None or key > best_key:
                best_key = key
                best = position
        return best


# ---------------- 保存と読み込み ----------------

DEFAULT_POPULATION_SIZE = 4
DEFAULT_AI_NAMES = ("AI-1", "AI-2", "AI-3", "AI-4")

# 「性格」。探索の強さを変えることで、同じルールから別の戦い方に育つ。
# 体数を増やすときは、この範囲を等間隔に割り当てる。
MIN_EXPLORE_SCALE = 0.55
MAX_EXPLORE_SCALE = 1.90
DEFAULT_EXPLORE_SCALES = (0.55, 0.85, 1.30, 1.90)


def explore_scales(count: int) -> tuple[float, ...]:
    """体数に合わせて、探索の強さを均等に割り振る。"""

    if count <= 1:
        return (1.0,)
    span = MAX_EXPLORE_SCALE - MIN_EXPLORE_SCALE
    return tuple(
        round(MIN_EXPLORE_SCALE + span * index / (count - 1), 2)
        for index in range(count)
    )


def new_population(
    seed: int = 20260910,
    names: tuple[str, ...] | None = None,
    count: int = DEFAULT_POPULATION_SIZE,
    capacity: str = DEFAULT_CAPACITY,
) -> list[Brain]:
    if names is None:
        names = tuple(f"AI-{index + 1}" for index in range(count))
    scales = explore_scales(len(names))
    return [
        Brain.new(name, seed + 7919 * (index + 1), scales[index], capacity)
        for index, name in enumerate(names)
    ]


def save_population(brains: list[Brain], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "formatVersion": 2,
        "touringBaseCount": TOURING_BASE_COUNT,
        "supplyBaseCount": SUPPLY_BASE_COUNT,
        "brains": [brain.to_json_dict() for brain in brains],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temporary.replace(path)


def load_population(path: str | Path) -> list[Brain]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("touringBaseCount", TOURING_BASE_COUNT) != TOURING_BASE_COUNT:
        raise ValueError("保存済みモデルの材料の数が現在の実装と違います。新規学習してください。")
    return [Brain.from_json_dict(raw) for raw in payload["brains"]]
