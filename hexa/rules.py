"""公式ルールの定数。

すべての値は公式配布資料が根拠です。根拠は docs/RULES.md に対応表があります。

- 募集要項 p.8-p.12（表1 セルの地形タイプ一覧、道路セルと交通量、スポット、
  エージェント、移動と燃料、試合の進行、勝敗判定）
- 競技部門「ヘキサうどん」のフォーマットについて p.1-p.3
- 質問回答集その1 (Q1, Q6, Q10, Q11, Q13, Q14, Q15, Q16, Q20)
- 質問回答集その2 (Q22, Q25, Q26, Q27, Q30, Q31, Q47, Q49, Q51)
- 質問回答集その3 (Q62)
- 質問回答集 補足資料(行動詳細)
"""

from __future__ import annotations

from enum import IntEnum


class Terrain(IntEnum):
    """フォーマット資料 p.1: cells の値 0:平地 1:道路 2:山地 3:池"""

    PLAIN = 0
    ROAD = 1
    MOUNTAIN = 2
    POND = 3


class AgentKind(IntEnum):
    """フォーマット資料 p.2: kind の値 0:巡回車 1:補給車"""

    TOURING = 0
    SUPPLY = 1


class RoadStatus(IntEnum):
    """フォーマット資料 p.2: traffics.status の値 0:順調 1:混雑 2:渋滞"""

    SMOOTH = 0
    BUSY = 1
    JAMMED = 2


# 募集要項 p.9 表1: 移動時間（ステップ）
# 平地 2 / 山地 3 / 池 進入不可 / 道路 順調1・混雑2・渋滞4
PLAIN_MOVE_STEPS = 2
MOUNTAIN_MOVE_STEPS = 3
ROAD_MOVE_STEPS: dict[RoadStatus, int] = {
    RoadStatus.SMOOTH: 1,
    RoadStatus.BUSY: 2,
    RoadStatus.JAMMED: 4,
}

# 募集要項 p.9 表1: 燃料消費量。道路は状態によらず 2。
# 質問回答集その2 Q28 で「燃量」は燃料の量の誤記と確定。
PLAIN_FUEL_COST = 1
MOUNTAIN_FUEL_COST = 2
ROAD_FUEL_COST = 2

# 質問回答集その1 Q10 / その2 Q25:
# 移動に要するステップ数も燃料も「移動元」のセルを参照する。
MOVE_COST_USES_ORIGIN_CELL = True

# 募集要項 p.8: 1試合は 4～10 日
MIN_DAYS = 4
MAX_DAYS = 10

# 募集要項 p.8: 各マップサイズは縦横それぞれ最大32セル、最小8セル
MIN_MAP_SIDE = 8
MAX_MAP_SIDE = 32

# 募集要項 p.10: エージェントの数は 3～8 体（質問回答集その2 Q50: 1チームあたり）
MIN_AGENTS = 3
MAX_AGENTS = 8

# 質問回答集その1 Q13 / その2 Q30: 混雑基準値 1～5、渋滞基準値 2～10、正の整数
MIN_BUSY_THRESHOLD = 1
MAX_BUSY_THRESHOLD = 5
MIN_JAMMED_THRESHOLD = 2
MAX_JAMMED_THRESHOLD = 10

# 質問回答集その1 Q14 + その3 Q62:
# 燃料積載量上限は「1日目のステップ数」の1～3倍の範囲（整数倍とは限らない/その2 Q60）
FUEL_LIMIT_MIN_RATIO = 1.0
FUEL_LIMIT_MAX_RATIO = 3.0

# 質問回答集その1 Q20: 各日のステップ数は最小 W+H、最大 4*(W+H)
DAY_STEPS_MIN_RATIO = 1
DAY_STEPS_MAX_RATIO = 4

# 質問回答集その1 Q15: スポット数の最小はエージェント数、最大はマップの縦幅・横幅の最大値
# 募集要項 p.9: 最大在庫数の範囲は 1 ～ (1チームのエージェント数)
# 募集要項 p.9: 系列数の範囲は 1 ～ スポット数

# 質問回答集その2 Q47: 0=左上 1=右上 2=右 3=右下 4=左下 5=左
DIRECTION_NAMES: tuple[str, ...] = ("左上", "右上", "右", "右下", "左下", "左")

# 質問回答集その1 Q1: 偶数行が右にずれる形で固定。Q2: トーラスではない。


def move_steps(terrain: Terrain, road_status: RoadStatus | None) -> int:
    """移動元セルの地形から、移動に要するステップ数を返す。"""

    if terrain == Terrain.PLAIN:
        return PLAIN_MOVE_STEPS
    if terrain == Terrain.MOUNTAIN:
        return MOUNTAIN_MOVE_STEPS
    if terrain == Terrain.ROAD:
        return ROAD_MOVE_STEPS[road_status if road_status is not None else RoadStatus.SMOOTH]
    raise ValueError("池からは移動できません")


def fuel_cost(terrain: Terrain) -> int:
    """移動元セルの地形から、巡回車が消費する燃料を返す。"""

    if terrain == Terrain.PLAIN:
        return PLAIN_FUEL_COST
    if terrain == Terrain.MOUNTAIN:
        return MOUNTAIN_FUEL_COST
    if terrain == Terrain.ROAD:
        return ROAD_FUEL_COST
    raise ValueError("池からは移動できません")
