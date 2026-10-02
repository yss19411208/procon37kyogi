"""問題（マップ構成）の生成。

生成される値はすべて公式資料で確認できる範囲に収まります。生成後は必ず
problem.validate_problem() を通し、違反があれば作り直します。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from hexa import rules
from hexa.hexgrid import neighbor_table
from hexa.problem import Problem, Spot, validate_problem
from hexa.rules import Terrain


@dataclass(frozen=True)
class GeneratorSettings:
    height: int
    width: int
    agent_count: int
    spot_count: int
    brand_count: int
    day_count: int
    players: int
    pond_ratio: float = 0.06
    road_ratio: float = 0.22
    mountain_ratio: float = 0.14
    day_seconds: int = 60
    starts_at: int = 1_778_227_200
    # 燃料上限は「1日目のステップ数 × この範囲」から引く（質問回答集その1 Q14 / その3 Q62）。
    fuel_ratio_low: float = 1.0
    fuel_ratio_high: float = 3.0
    # 混雑・渋滞の基準値の範囲（質問回答集その1 Q13 / その2 Q30）。
    busy_low: int = 1
    busy_high: int = 5
    jammed_low: int = 2
    jammed_high: int = 10
    # 各日のステップ数は W+H の何倍か（質問回答集その1 Q20 より 1〜4倍）。
    # 公開された参考マップは 1.0〜1.5倍と、許される範囲の下限寄りでした。
    day_steps_low_ratio: float = 1.0
    day_steps_high_ratio: float = 4.0
    # 地形のかたまり具合。0 なら完全にばらばら、1 に近いほど大きな塊になる。
    # 公開された参考マップは道路も山地も明らかに固まっていました
    # （道路の出現率24%に対し、隣が道路である割合は36%）。
    # ばらばらに置くと道路が1マスずつ孤立し、道路をつないで速く走ることが
    # できません。本選とは別のゲームになってしまいます。
    clump: float = 0.05
    archetype: str = "mixed"

    def describe(self) -> str:
        return (
            f"{self.height}x{self.width} / エージェント{self.agent_count}体 / "
            f"スポット{self.spot_count}個 / 系列{self.brand_count} / "
            f"{self.day_count}日 / players={self.players}"
        )


# マップの「型」。地形の割合と燃料・基準値の厳しさを変えると、
# 効く戦術がまるごと入れ替わります。特定の型だけで学習すると、
# 本番で別の型が来たときに手が出ません。
#
#   街    : 道路が多い。移動は速いが交通量が溜まりやすく、妨害と渋滞回避が効く
#   山岳  : 山地が多い。燃料の消費が激しく、補給車の価値が跳ね上がる
#   群島  : 池が多い。通れる道が細く、渋滞が一点に集中する
#   平原  : ほぼ平地。素直な距離勝負になり、小細工が効かない
#   混在  : 何でもあり
ARCHETYPES: dict[str, dict] = {
    "street": {
        "pond": (0.02, 0.06), "road": (0.35, 0.55), "mountain": (0.03, 0.10),
        "fuel": (1.0, 3.0), "busy": (1, 3), "jammed": (2, 5),
        # 道路が長い回廊になる。
        "clump": (0.05, 0.50),
    },
    "mountain": {
        "pond": (0.02, 0.08), "road": (0.05, 0.15), "mountain": (0.35, 0.50),
        "fuel": (1.0, 1.8), "busy": (2, 5), "jammed": (4, 10),
        # 山脈になる。大きな一枚岩も出す。
        "clump": (0.10, 0.80),
    },
    "islands": {
        "pond": (0.18, 0.30), "road": (0.15, 0.30), "mountain": (0.10, 0.25),
        "fuel": (1.0, 2.5), "busy": (1, 3), "jammed": (2, 6),
        # 島と水路になる。大きな湖も出す。
        "clump": (0.15, 0.85),
    },
    "plain": {
        "pond": (0.02, 0.06), "road": (0.05, 0.12), "mountain": (0.03, 0.10),
        "fuel": (1.5, 3.0), "busy": (3, 5), "jammed": (6, 10),
        "clump": (0.0, 0.30),
    },
    "mixed": {
        "pond": (0.02, 0.12), "road": (0.10, 0.35), "mountain": (0.05, 0.25),
        "fuel": (1.0, 3.0), "busy": (1, 5), "jammed": (2, 10),
        # ばらばらから塊まで、何でも出す。
        "clump": (0.05, 0.75),
    },
    # 公開された参考マップ（sample_16 / 24 / 32）から読み取った比率。
    # 平地21〜35% / 道路24〜30% / 山地18〜31% / 池18〜24%、
    # 燃料は1日目のステップ数の1.25〜1.88倍、混雑1・渋滞3。
    # いままでのどの型よりも通れない場所が多く、燃料も基準値も厳しい設定です。
    "honsen": {
        "pond": (0.17, 0.25), "road": (0.23, 0.31), "mountain": (0.17, 0.32),
        "fuel": (1.2, 1.9), "busy": (1, 2), "jammed": (3, 5),
        # 参考マップは超過分 +0.01〜+0.12 だったが、実際に配られた例では
        # 山地 +0.56 / 池 +0.60 と、巨大な一枚岩になっていた。
        # 迂回できない壁があるマップは経路選択がすべてになるので、
        # ばらばらから一枚岩まで、幅広く経験させる。
        "clump": (0.05, 0.75),
    },
}

# 本選の試合条件（本選マップに関するお知らせ）。
# マップサイズ・エージェント数・スポット数・系列数・タイプ選択時間は公開済み。
# 日数・ステップ数・最大在庫数・燃料積載量上限・混雑/渋滞基準値は非公開。
HONSEN_ROUNDS: dict[str, dict] = {
    "1回戦": {
        "side": 16, "agents": 4, "spots": 16, "brands": (10, 14), "kind_seconds": 60,
    },
    "準決勝": {
        "side": 24, "agents": 5, "spots": 24, "brands": (14, 20), "kind_seconds": 90,
    },
    "決勝": {
        "side": 32, "agents": 7, "spots": 32, "brands": (20, 28), "kind_seconds": 120,
    },
}


def honsen_settings(
    round_name: str,
    rng: random.Random,
    players: int = 10,
    archetype: str = "honsen",
    day_count: int | None = None,
) -> GeneratorSettings:
    """本選と同じ条件の問題設定を作る。

    公開されている条件（大きさ・エージェント数・スポット数・系列数）は固定し、
    非公開の項目（日数・ステップ数・燃料・基準値）は型の範囲から引きます。
    """

    spec = HONSEN_ROUNDS[round_name]
    shape = ARCHETYPES.get(archetype, ARCHETYPES["honsen"])
    brand_low, brand_high = spec["brands"]
    return GeneratorSettings(
        height=spec["side"],
        width=spec["side"],
        agent_count=spec["agents"],
        spot_count=spec["spots"],
        brand_count=rng.randint(brand_low, brand_high),
        day_count=day_count if day_count is not None else rng.randint(4, 10),
        players=players,
        pond_ratio=rng.uniform(*shape["pond"]),
        road_ratio=rng.uniform(*shape["road"]),
        mountain_ratio=rng.uniform(*shape["mountain"]),
        fuel_ratio_low=shape["fuel"][0],
        fuel_ratio_high=shape["fuel"][1],
        busy_low=shape["busy"][0],
        busy_high=shape["busy"][1],
        jammed_low=shape["jammed"][0],
        jammed_high=shape["jammed"][1],
        # 参考マップは1.0〜1.5倍だったが、実際に配られた例では
        # 1.09〜3.91倍と大きく振れていた（125, 108, 106, 108, 35, 82, 37, 78, 107）。
        # 極端に短い日も長い日も経験させる必要がある。
        day_steps_low_ratio=1.0,
        day_steps_high_ratio=4.0,
        clump=rng.uniform(*shape.get("clump", (0.02, 0.10))),
        archetype=archetype,
    )


def random_settings(
    rng: random.Random,
    players: int,
    archetype: str | None = None,
) -> GeneratorSettings:
    """公式の範囲から、マップの大きさも型も丸ごとランダムに作る。

    学習を特定の形に偏らせないための関数です。プリセットの大きさだけ、
    あるいは同じ地形の割合だけで学習すると、
    「このくらいのマップなら使える戦術」を1通りしか覚えません。
    """

    name = archetype or rng.choice(list(ARCHETYPES))
    shape = ARCHETYPES[name]
    height = rng.randint(rules.MIN_MAP_SIDE, rules.MAX_MAP_SIDE)
    width = rng.randint(rules.MIN_MAP_SIDE, rules.MAX_MAP_SIDE)
    agent_count = rng.randint(rules.MIN_AGENTS, rules.MAX_AGENTS)
    spot_count = rng.randint(agent_count, max(agent_count, max(height, width)))
    # 系列1なら玉数勝負、系列=店数なら種類数勝負。どちらも経験させる。
    brand_count = rng.randint(1, spot_count)
    day_count = rng.randint(rules.MIN_DAYS, rules.MAX_DAYS)
    busy_low, busy_high = shape["busy"]
    jammed_low, jammed_high = shape["jammed"]
    return GeneratorSettings(
        height=height,
        width=width,
        agent_count=agent_count,
        spot_count=spot_count,
        brand_count=brand_count,
        day_count=day_count,
        players=players,
        pond_ratio=rng.uniform(*shape["pond"]),
        road_ratio=rng.uniform(*shape["road"]),
        mountain_ratio=rng.uniform(*shape["mountain"]),
        fuel_ratio_low=shape["fuel"][0],
        fuel_ratio_high=shape["fuel"][1],
        busy_low=busy_low,
        busy_high=busy_high,
        jammed_low=jammed_low,
        jammed_high=jammed_high,
        clump=rng.uniform(*shape.get("clump", (0.0, 0.15))),
        archetype=name,
    )


PRESETS: dict[str, GeneratorSettings] = {
    # 学習の序盤用。公式の最小サイズ。
    "small": GeneratorSettings(
        height=8, width=8, agent_count=4, spot_count=6, brand_count=3,
        day_count=4, players=4,
    ),
    # 標準。学習の主力。
    "medium": GeneratorSettings(
        height=12, width=12, agent_count=6, spot_count=10, brand_count=5,
        day_count=6, players=4,
    ),
    # 本番想定の上限寄り。
    "large": GeneratorSettings(
        height=20, width=20, agent_count=8, spot_count=16, brand_count=8,
        day_count=8, players=4,
    ),
    # 公式の最大サイズ。動作確認と最終評価用。
    "max": GeneratorSettings(
        height=32, width=32, agent_count=8, spot_count=32, brand_count=10,
        day_count=10, players=4,
    ),
}


def clamp_settings(settings: GeneratorSettings) -> GeneratorSettings:
    """公式の範囲に収まるように設定値を丸める。"""

    height = max(rules.MIN_MAP_SIDE, min(rules.MAX_MAP_SIDE, settings.height))
    width = max(rules.MIN_MAP_SIDE, min(rules.MAX_MAP_SIDE, settings.width))
    agent_count = max(rules.MIN_AGENTS, min(rules.MAX_AGENTS, settings.agent_count))
    # 質問回答集その1 Q15: スポット数はエージェント数以上、max(縦,横)以下
    spot_count = max(agent_count, min(max(height, width), settings.spot_count))
    brand_count = max(1, min(spot_count, settings.brand_count))
    day_count = max(rules.MIN_DAYS, min(rules.MAX_DAYS, settings.day_count))
    players = max(1, settings.players)
    return GeneratorSettings(
        height=height,
        width=width,
        agent_count=agent_count,
        spot_count=spot_count,
        brand_count=brand_count,
        day_count=day_count,
        players=players,
        pond_ratio=settings.pond_ratio,
        road_ratio=settings.road_ratio,
        mountain_ratio=settings.mountain_ratio,
        day_seconds=settings.day_seconds,
        starts_at=settings.starts_at,
        fuel_ratio_low=settings.fuel_ratio_low,
        fuel_ratio_high=settings.fuel_ratio_high,
        busy_low=settings.busy_low,
        busy_high=settings.busy_high,
        jammed_low=settings.jammed_low,
        jammed_high=settings.jammed_high,
        day_steps_low_ratio=settings.day_steps_low_ratio,
        day_steps_high_ratio=settings.day_steps_high_ratio,
        clump=settings.clump,
        archetype=settings.archetype,
    )


def _grow_regions(
    rng: random.Random,
    candidates: list[int],
    target: int,
    clump: float,
    width: int,
    height: int,
) -> set[int]:
    """候補セルの中から target 個を「かたまり」として選ぶ。

    種をいくつか蒔いて、その周りに広げていきます。
    clump が0に近いほど種が多く（＝ばらばら）、1に近いほど種が少なく
    （＝大きな塊）なります。
    """

    target = max(0, min(target, len(candidates)))
    if target == 0:
        return set()
    table = neighbor_table(width, height)
    pool = set(candidates)
    # 塊ひとつの平均の大きさを clump で決める。
    average_blob = 1.0 + clump * 11.0
    seed_count = max(1, min(target, int(target / average_blob) + 1))

    chosen: set[int] = set()
    order = list(candidates)
    rng.shuffle(order)
    seeds = order[:seed_count]
    frontier: list[int] = []
    for seed in seeds:
        if len(chosen) >= target:
            break
        chosen.add(seed)
        frontier.append(seed)

    cursor = seed_count
    while len(chosen) < target:
        if not frontier:
            # 種が尽きたら、まだ使っていないセルから新しい種を置く。
            while cursor < len(order) and order[cursor] in chosen:
                cursor += 1
            if cursor >= len(order):
                break
            chosen.add(order[cursor])
            frontier.append(order[cursor])
            cursor += 1
            continue
        index = rng.randrange(len(frontier))
        current = frontier[index]
        neighbours = [
            nb for _direction, nb in table[current]
            if nb in pool and nb not in chosen
        ]
        if not neighbours:
            frontier.pop(index)
            continue
        picked = rng.choice(neighbours)
        chosen.add(picked)
        frontier.append(picked)
    return chosen


def _connected_without(cells: list[int], width: int, height: int) -> bool:
    table = neighbor_table(width, height)
    walkable = [p for p, value in enumerate(cells) if value != int(Terrain.POND)]
    if not walkable:
        return False
    seen = {walkable[0]}
    stack = [walkable[0]]
    while stack:
        current = stack.pop()
        for _direction, next_position in table[current]:
            if cells[next_position] == int(Terrain.POND) or next_position in seen:
                continue
            seen.add(next_position)
            stack.append(next_position)
    return len(seen) == len(walkable)


def generate_problem(seed: int, settings: GeneratorSettings) -> Problem:
    settings = clamp_settings(settings)
    rng = random.Random(seed)

    for _attempt in range(200):
        problem = _try_generate(rng, settings)
        if problem is not None and not validate_problem(problem):
            return problem
    raise RuntimeError(
        "この設定では有効な問題を生成できませんでした。"
        "スポット数やエージェント数を減らしてください。"
    )


def _try_generate(rng: random.Random, settings: GeneratorSettings) -> Problem | None:
    height, width = settings.height, settings.width
    cell_count = height * width
    cells = [int(Terrain.PLAIN)] * cell_count

    # 池をかたまりで置く。1マスずつ試し、連結が保たれるものだけ採用する。
    pond_target = max(1, int(cell_count * settings.pond_ratio))
    # 池は**最も大きな塊**になります。練習場の実マップを測ると
    # 池の超過分が +0.63、山地が +0.52 と、道路(+0.28)の2倍以上でした。
    # 以前は分断を恐れて 0.7 倍に抑えていましたが、実際はもっと固まっています。
    # 連結が壊れるものは1マスずつ採用判定で弾いているので、強めて構いません。
    wanted = _grow_regions(
        rng, list(range(cell_count)), pond_target,
        min(1.0, settings.clump * 1.6), width, height,
    )
    ponds = 0
    for position in sorted(wanted, key=lambda _x: rng.random()):
        cells[position] = int(Terrain.POND)
        if _connected_without(cells, width, height):
            ponds += 1
        else:
            cells[position] = int(Terrain.PLAIN)
    if ponds == 0:
        return None

    # 池以外に道路と山地を割り当てる。連結性には影響しない。
    # ここもかたまりで置く。道路は1マスずつ孤立していると、つないで走れない。
    land = [p for p in range(cell_count) if cells[p] != int(Terrain.POND)]
    road_target = max(1, int(len(land) * settings.road_ratio))
    mountain_target = max(1, int(len(land) * settings.mountain_ratio))
    if road_target + mountain_target >= len(land):
        return None

    roads = _grow_regions(rng, land, road_target, settings.clump, width, height)
    for position in roads:
        cells[position] = int(Terrain.ROAD)
    remaining = [position for position in land if position not in roads]
    # 山地も池に次いで大きな塊になります（実マップで +0.52）。
    mountains = _grow_regions(
        rng, remaining, mountain_target,
        min(1.0, settings.clump * 1.3), width, height,
    )
    for position in mountains:
        cells[position] = int(Terrain.MOUNTAIN)

    plains = [p for p in range(cell_count) if cells[p] == int(Terrain.PLAIN)]
    if len(plains) < settings.spot_count + settings.agent_count:
        return None
    rng.shuffle(plains)
    spot_positions = plains[:settings.spot_count]
    agent_positions = plains[settings.spot_count:settings.spot_count + settings.agent_count]

    # 系列は 0..brand_count-1 が必ず1回以上出るように配る。
    brands = list(range(settings.brand_count))
    while len(brands) < settings.spot_count:
        brands.append(rng.randrange(settings.brand_count))
    rng.shuffle(brands)
    spots = tuple(
        Spot(
            brand=brands[index],
            position=position,
            # 募集要項 p.9: 最大在庫数は 1 ～ 1チームのエージェント数
            stocks=rng.randint(1, settings.agent_count),
        )
        for index, position in enumerate(spot_positions)
    )

    # 質問回答集その1 Q20: 各日のステップ数は W+H 以上 4*(W+H) 以下
    span = width + height
    steps_low = max(rules.DAY_STEPS_MIN_RATIO, settings.day_steps_low_ratio)
    steps_high = min(rules.DAY_STEPS_MAX_RATIO, settings.day_steps_high_ratio)
    day_steps = tuple(
        rng.randint(int(span * steps_low), max(int(span * steps_low), int(span * steps_high)))
        for _ in range(settings.day_count)
    )
    # 質問回答集その1 Q14 + その3 Q62: 燃料上限は1日目のステップ数の1〜3倍。
    # 型ごとに厳しさを変えるため、その範囲内でさらに絞り込む。
    fuel_low = max(rules.FUEL_LIMIT_MIN_RATIO, settings.fuel_ratio_low)
    fuel_high = min(rules.FUEL_LIMIT_MAX_RATIO, settings.fuel_ratio_high)
    if fuel_high < fuel_low:
        fuel_high = fuel_low
    fuel_floor = max(day_steps[0], int(day_steps[0] * fuel_low))
    fuel_ceiling = max(fuel_floor, int(day_steps[0] * fuel_high))
    fuel_limit = rng.randint(fuel_floor, fuel_ceiling)

    # 質問回答集その1 Q13: 混雑1〜5、渋滞2〜10。混雑 < 渋滞。
    # 型ごとの範囲を、公式の範囲の内側に収めてから引く。
    busy_low = max(rules.MIN_BUSY_THRESHOLD, settings.busy_low)
    busy_high = min(rules.MAX_BUSY_THRESHOLD, settings.busy_high)
    if busy_high < busy_low:
        busy_high = busy_low
    busy = rng.randint(busy_low, busy_high)

    jammed_low = max(busy + 1, rules.MIN_JAMMED_THRESHOLD, settings.jammed_low)
    jammed_high = min(rules.MAX_JAMMED_THRESHOLD, settings.jammed_high)
    if jammed_high < jammed_low:
        jammed_high = jammed_low
    jammed = rng.randint(jammed_low, min(jammed_high, rules.MAX_JAMMED_THRESHOLD))

    return Problem(
        starts_at=settings.starts_at,
        day_seconds=tuple(settings.day_seconds for _ in range(settings.day_count)),
        day_steps=day_steps,
        height=height,
        width=width,
        cells=tuple(cells),
        spots=spots,
        agent_positions=tuple(agent_positions),
        fuel_limit=fuel_limit,
        players=settings.players,
        busy_threshold=busy,
        jammed_threshold=jammed,
    )
