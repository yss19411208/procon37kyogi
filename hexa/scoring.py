"""公式の勝敗判定（募集要項 p.12）。

  (1) 1試合で獲得したうどんの種類数が多いチームを勝者
  (2) 各日に獲得したうどんの種類数の累積が多いチームを勝者
  (3) 1試合で獲得したうどんの玉数の総和が多いチームを勝者
  (4) 各日の最後に送信した有効な回答の回答時間の累積が少ないチームを勝者
  (5) それでも決まらない場合はサイコロなどで決めるか引き分け

このプログラムはオフラインなので (1)～(3) を主指標、(4) は解答生成時間を
参考値として別に記録します（オフラインでは公式の回答時間と一致しません）。
"""

from __future__ import annotations

from dataclasses import dataclass

from hexa.simulator import DayResult


@dataclass(frozen=True)
class Score:
    brand_count: int              # (1) 総種類数
    daily_brand_total: int        # (2) 日別種類数の累計
    noodle_count: int             # (3) 総玉数

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.brand_count, self.daily_brand_total, self.noodle_count)

    def __str__(self) -> str:
        return (
            f"種類{self.brand_count} / 日別累計{self.daily_brand_total} "
            f"/ 玉数{self.noodle_count}"
        )


def score_days(day_results: list[DayResult]) -> Score:
    brands: set[int] = set()
    daily_total = 0
    noodles = 0
    for day_result in day_results:
        day_brands = {event.brand for event in day_result.acquisitions}
        brands |= day_brands
        daily_total += len(day_brands)
        noodles += len(day_result.acquisitions)
    return Score(len(brands), daily_total, noodles)


def rank_scores(scores: list[Score]) -> list[int]:
    """1位を1とする順位。同点は同順位。"""

    ranks = []
    for score in scores:
        better = sum(1 for other in scores if other.key > score.key)
        ranks.append(better + 1)
    return ranks
