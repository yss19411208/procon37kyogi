"""その試合のあいだだけ、作戦を少しずつ良くしていく。

**いまは使っていません。実測で伸びなかったためです。**

実戦24x24で、1日目に240案を作って改良ありとなしを比べたところ、
どちらも「系列17 / 玉41」で同じでした。理由を調べると、

  * 作戦ごとの成績には差がある（12玉〜41玉）
  * ところが**最良の作戦が最初から種に入っている**
  * その作戦を変異させても、元を超えるものが出ない

という状態でした。さらに、その日の獲得41玉は
「店どうしの距離24歩 × 41訪問 ÷ 4台 = 241歩」が必要な計算なのに、
実際の予算は55歩です。**近い店を連続で回ることで理論値の4倍以上を
すでに達成**しており、作戦の微調整で動く余地がありません。

仕組みは残してあるので、学習が進んで作戦の質が変わったときに
試し直せます。Policy に recipes を渡せば差し替わります。


**なぜ必要か**

いまの探索は「学習済みの作戦をそのまま使って1案ずつ作り、一番良いものを
選ぶ」だけです。良い案が見つかっても、**その近くをさらに探すことをして
いません。**

学習は「どんなマップでもそこそこ良い作戦」を作ります。しかし本番で出会う
のは1つのマップだけです。そのマップに合わせて作戦を少し変えれば、
学習済みのものより良くなる余地があります。

ここでは、その日の探索のあいだに

  1. 良い結果を出した作戦を覚えておく
  2. それを少し変えた作戦を作る（変異）
  3. 2つの良い作戦を混ぜる（交配）
  4. 試して良ければ残す

を繰り返します。**学習そのものは変えません。** 試合が終われば捨てる、
使い捨ての工夫です。
"""

from __future__ import annotations

import random

from hexa.routes import SEED_RECIPES, blend_recipes, mutate_recipe

# 手元に置く作戦の数。多いと1回の判断が重くなります。
POOL_SIZE = 14

# 新しい作戦を作るとき、変異と交配をどの割合で使うか。
BLEND_RATIO = 0.35


class RecipeRefiner:
    """その試合用の作戦の持ち主。

    使い方は単純で、`recipes()` で試す作戦を受け取り、結果が出たら
    `note()` で点数を返すだけです。
    """

    def __init__(self, base: list[list[float]] | None = None,
                 rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random(20261003)
        seeds = [list(weights) for _name, weights in SEED_RECIPES]
        if base:
            seeds = [list(item) for item in base] + seeds
        # 同じものが並ばないよう、先頭から詰める
        self.pool: list[list[float]] = []
        for item in seeds:
            if len(self.pool) >= POOL_SIZE:
                break
            self.pool.append(list(item))
        # 各作戦の成績（試した回数と、最も良かった点数）
        self.tries: list[int] = [0] * len(self.pool)
        self.best: list[float] = [-1e18] * len(self.pool)

    def recipes(self) -> list[list[float]]:
        """いま試す作戦。そのまま Policy に渡せます。"""

        return [list(item) for item in self.pool]

    def note(self, index: int, score: float) -> None:
        """その作戦が出した点数を覚える。"""

        if not (0 <= index < len(self.pool)):
            return
        self.tries[index] += 1
        if score > self.best[index]:
            self.best[index] = score

    def evolve(self) -> int:
        """成績の悪い作戦を1つ、良い作戦から作った新しいものに入れ替える。

        戻り値は入れ替えた位置。入れ替えなかったときは -1。

        **一度も試していない作戦は消しません。** 偶然悪い点数が1回出ただけ
        で捨てると、本当は良い作戦を見失います。
        """

        tried = [i for i, count in enumerate(self.tries) if count >= 2]
        if len(tried) < 4:
            return -1

        ranked = sorted(tried, key=lambda i: self.best[i], reverse=True)
        good = ranked[: max(2, len(ranked) // 3)]
        worst = ranked[-1]

        if self.rng.random() < BLEND_RATIO and len(good) >= 2:
            first, second = self.rng.sample(good, 2)
            fresh = blend_recipes(self.pool[first], self.pool[second], self.rng)
        else:
            parent = self.rng.choice(good)
            fresh = mutate_recipe(self.pool[parent], self.rng, strength=0.30)

        self.pool[worst] = fresh
        self.tries[worst] = 0
        self.best[worst] = -1e18
        return worst

    def summary(self) -> str:
        """ログ用の短い説明。"""

        tried = sum(1 for count in self.tries if count > 0)
        best = max(self.best) if self.best else 0
        return f"作戦{len(self.pool)}個 / 試した{tried}個 / 最高{best:.0f}"
