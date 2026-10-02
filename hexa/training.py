"""自己対戦による学習。

4体のAIが同じ問題を同時に解き、そこで実際に起きたことを教師にして
それぞれの価値関数を更新します。経験は models/ に貯まり続け、
次回の学習でも再利用されるので、回した分だけサンプルが増えて予測が良くなります。

学習が本当に良くなっているかは、固定の検証問題で毎回測って history に残します。
数字は models/ の中に入るので、あとから伸びを確認できます。
"""

from __future__ import annotations

import gc
import gzip
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from hexa.ai import (
    CAPACITY_PRESETS,
    Brain,
    load_population,
    new_population,
    save_population,
)
from hexa.generator import (
    ARCHETYPES,
    HONSEN_ROUNDS,
    GeneratorSettings,
    PRESETS,
    generate_problem,
    honsen_settings,
    random_settings,
)
from hexa.match import run_match
from hexa.problem import Problem, load_problem
from hexa.progress import Progress, format_duration

REPLAY_LIMIT_PER_BRAIN = 60_000

# 経験1件あたりのおおよそのメモリ量（バイト）。
# Python のリストと float は1要素で30バイト前後を使うので、そこから見積もる。
BYTES_PER_FEATURE = 34


def replay_limit_for_budget(
    budget_bytes: int,
    brain_count: int,
    touring_features: int,
    supply_features: int,
) -> int:
    """使ってよいメモリ量から、1体あたりに貯める経験の数を決める。

    ここで効くのはディスクではなくメモリです。経験は学習中ずっと
    メモリに載せておくので、貯めすぎると動かなくなります。
    ディスクにはgzipで書くので、メモリの1/5〜1/10程度で収まります。
    """

    per_sample = (touring_features + supply_features + 8) * BYTES_PER_FEATURE
    per_brain = budget_bytes / max(1, brain_count)
    return max(1_000, int(per_brain / max(1, per_sample)))
VALIDATION_SEED_BASE = 900_000
# 検証の間隔がどれだけ長くても、この秒数ごとには必ず保存する。
# これがないと、エピソード数を大きくしたときに「途中で止めたら全部消える」
# という事故が起きる。
AUTOSAVE_SECONDS = 120.0
# 検証で一番良かった時点のモデルを別に残しておく。
BEST_SUFFIX = "_best"


@dataclass
class TrainingSettings:
    episodes: int
    curriculum: tuple[str, ...] = ("small", "medium", "large")
    epsilon_start: float = 0.35
    epsilon_end: float = 0.03
    alpha_start: float = 0.08
    alpha_end: float = 0.01
    replay_updates_per_episode: int = 64
    validation_every: int = 25
    validation_problems: int = 3
    validation_preset: str = "medium"
    # 学習の後半で、マップの大きさを公式の範囲から丸ごとランダムに振る割合。
    # 決まった大きさだけで学習すると、その間の大きさに弱くなる。
    random_size_ratio: float = 0.6
    # ランダムに振る回のうち、本選と同じ条件（16/24/32・系列多め・
    # ステップ数少なめ）にする割合。本選で実際に来る形を重点的に経験させる。
    honsen_ratio: float = 0.5
    problem_dir: str | None = None
    # 1体あたりに貯める経験の数。None なら REPLAY_LIMIT_PER_BRAIN。
    replay_limit: int | None = None
    # マップの型ごとの出やすさ。苦手な型を多めに出すために使う。
    archetype_bias: dict[str, float] | None = None
    # 長時間学習を小分けにして回すとき、全体の中での位置を伝えるための値。
    # これがないと、1回ごとに探索率と学習率が最初に戻ってしまい、
    # いくら回しても収束しません。
    progress_offset: int = 0
    progress_total: int = 0
    # 1エピソードで戦わせるチーム数。AIの総数より少なくできる。
    # 少なくすると1エピソードが軽くなり、多くのAIを抱えても学習が重くならない。
    # None なら「AIの総数」と「本選の上限10」の小さいほう。
    match_size: int | None = None


def _replay_path(models_dir: Path, brain: Brain) -> Path:
    safe = brain.name.replace("/", "_")
    return models_dir / f"replay_{safe}.json.gz"


def _legacy_replay_path(models_dir: Path, brain: Brain) -> Path:
    safe = brain.name.replace("/", "_")
    return models_dir / f"replay_{safe}.json"


def load_replay(models_dir: Path, brain: Brain) -> tuple[list, list]:
    path = _replay_path(models_dir, brain)
    legacy = _legacy_replay_path(models_dir, brain)
    try:
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        elif legacy.exists():
            with open(legacy, encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            return ([], [])
    except (json.JSONDecodeError, OSError, EOFError):
        return ([], [])
    return (payload.get("touring", []), payload.get("supply", []))


def save_replay(
    models_dir: Path,
    brain: Brain,
    touring: list,
    supply: list,
    limit: int = REPLAY_LIMIT_PER_BRAIN,
) -> None:
    path = _replay_path(models_dir, brain)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"touring": touring[-limit:], "supply": supply[-limit:]}
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle)
    temporary.replace(path)
    legacy = _legacy_replay_path(models_dir, brain)
    if legacy.exists():
        try:
            legacy.unlink()
        except OSError:
            pass


def load_or_create_population(models_dir: Path, model_name: str) -> tuple[list[Brain], bool]:
    path = models_dir / model_name
    if path.exists():
        return (load_population(path), True)
    return (new_population(), False)


def _global_progress(settings: TrainingSettings, done_in_round: int) -> float:
    """学習全体の中で、いまどこまで進んだかを 0〜1 で返す。

    長時間学習は小分けに回すので、1回ぶんの中だけで進み具合を測ると
    探索率も学習率も毎回リセットされ、いつまでも収束しません。
    全体の総数が渡されていれば、そちらを基準にします。
    """

    if settings.progress_total > 0:
        done = settings.progress_offset + done_in_round
        return min(1.0, done / max(1, settings.progress_total - 1))
    return min(1.0, done_in_round / max(1, settings.episodes - 1))


def resolve_match_size(settings: TrainingSettings, population: int) -> int:
    if settings.match_size is None:
        return max(2, min(population, 10))
    return max(2, min(population, settings.match_size))


def _curriculum_settings(
    progress: float,
    curriculum: tuple[str, ...],
    players: int,
    rng: random.Random | None = None,
    random_size_ratio: float = 0.0,
    archetype_bias: dict[str, float] | None = None,
    honsen_ratio: float = 0.0,
) -> GeneratorSettings:
    """その時点で学習に使う問題の設定を決める。

    序盤は小さいマップから順に大きくしていき、慣れてきたら
    公式の範囲（8〜32）から丸ごとランダムに振ります。
    こうしないと、プリセットの大きさ以外に弱いAIになります。
    """

    # 本選も練習場も 16x16 / 24x24 / 32x32 の3種類だけでした。
    # それ以外の大きさを混ぜても本番に効かないので、最初から3種類に絞ります。
    # 地形の型・日数・ステップ数・燃料・基準値は毎回変わるので、
    # 同じマップを繰り返すことにはなりません。
    if rng is not None:
        # 本選の初戦は9チーム戦（津山高専＋8チーム）。
        # 交通量は「全チームの滞在数 ÷ players」なので、players が変われば
        # 道路の混み方も変わります。9でも10でも戦えるよう、両方を経験させます。
        seats = players
        if players >= 9:
            seats = rng.choice((9, 10))
        return honsen_settings(
            _pick_round(rng),
            rng,
            players=seats,
            archetype=_pick_archetype(rng, archetype_bias) or "honsen",
        )

    if rng is not None and random_size_ratio > 0.0:
        # 序盤はプリセットで基礎を作り、進むほどランダムの比率を上げる。
        chance = random_size_ratio * min(1.0, progress * 2.0)
        if rng.random() < chance:
            archetype = _pick_archetype(rng, archetype_bias)
            if rng.random() < honsen_ratio:
                # 本選と同じ条件。大きさ・体数・店数・系列数が固定され、
                # 地形と非公開パラメータだけが振れる。
                return honsen_settings(
                    rng.choice(list(HONSEN_ROUNDS)),
                    rng,
                    players=players,
                    archetype=archetype or "honsen",
                )
            return random_settings(rng, players, archetype)
    return _from_preset(progress, curriculum, players)


# 1エピソードにかかる時間は大きさで大きく違います（5チーム・1コアの実測）。
#
#   1回戦 16x16 : 0.7秒
#   準決勝 24x24 : 2.0秒
#   決勝   32x32 : 8.0秒
#
# 均等に引くと平均3.6秒ですが、小さいほうを多めに引けば同じ時間で
# 多くの経験を積めます。ただし**実戦で一番弱いのが32x32**（1位の63〜78%）
# なので、減らしすぎると弱点がそのまま残ります。
# 時間を約1.4倍稼ぎつつ、決勝の条件も3割は必ず出す配分にしています。
ROUND_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("1回戦", 0.40),
    ("準決勝", 0.30),
    ("決勝", 0.30),
)


def _pick_round(rng: random.Random) -> str:
    roll = rng.random()
    total = 0.0
    for name, weight in ROUND_WEIGHTS:
        total += weight
        if roll < total:
            return name
    return ROUND_WEIGHTS[-1][0]


def _pick_archetype(rng: random.Random, bias: dict[str, float] | None) -> str | None:
    """苦手な型ほど出やすくなるように、重み付きで型を選ぶ。"""

    if not bias:
        return None
    names = [name for name in ARCHETYPES if bias.get(name, 0.0) > 0.0]
    if not names:
        return None
    weights = [bias[name] for name in names]
    total = sum(weights)
    if total <= 0.0:
        return None
    threshold = rng.random() * total
    running = 0.0
    for name, weight in zip(names, weights):
        running += weight
        if running >= threshold:
            return name
    return names[-1]


def _from_preset(
    progress: float,
    curriculum: tuple[str, ...],
    players: int,
) -> GeneratorSettings:

    index = min(len(curriculum) - 1, int(progress * len(curriculum)))
    base = PRESETS[curriculum[index]]
    return GeneratorSettings(
        height=base.height,
        width=base.width,
        agent_count=base.agent_count,
        spot_count=base.spot_count,
        brand_count=base.brand_count,
        day_count=base.day_count,
        players=players,
        pond_ratio=base.pond_ratio,
        road_ratio=base.road_ratio,
        mountain_ratio=base.mountain_ratio,
    )


def build_validation_problems(count: int, preset: str, players: int) -> list[Problem]:
    """検証用の固定問題。大きさをばらけさせて、特定の大きさに偏らないようにする。

    全部同じ大きさで測ると「そのサイズだけ得意なAI」を良いと判定してしまいます。
    小・中・大を混ぜて、平均で判断します。
    """

    order = ("small", "medium", "large", "max")
    shapes = list(ARCHETYPES)
    problems: list[Problem] = []
    for index in range(count):
        base = PRESETS[order[index % len(order)]] if count > 1 else PRESETS[preset]
        shape = ARCHETYPES[shapes[index % len(shapes)]]
        settings = GeneratorSettings(
            height=base.height, width=base.width, agent_count=base.agent_count,
            spot_count=base.spot_count, brand_count=base.brand_count,
            day_count=base.day_count, players=players,
            pond_ratio=sum(shape["pond"]) / 2,
            road_ratio=sum(shape["road"]) / 2,
            mountain_ratio=sum(shape["mountain"]) / 2,
            fuel_ratio_low=shape["fuel"][0], fuel_ratio_high=shape["fuel"][1],
            busy_low=shape["busy"][0], busy_high=shape["busy"][1],
            jammed_low=shape["jammed"][0], jammed_high=shape["jammed"][1],
            archetype=shapes[index % len(shapes)],
        )
        problems.append(generate_problem(VALIDATION_SEED_BASE + index, settings))
    return problems


def validate(
    brains: list[Brain],
    problems: list[Problem],
    match_size: int | None = None,
) -> list[tuple[float, float, float]]:
    """探索なしで固定問題を解き、各AIの平均スコアを返す。

    AIが多いときに全員を1試合に入れると重くなるので、
    match_size ごとの組に分けて順に測ります。
    """

    size = max(2, min(len(brains), match_size or len(brains)))
    groups = [
        list(range(start, min(start + size, len(brains))))
        for start in range(0, len(brains), size)
    ]
    totals = [[0.0, 0.0, 0.0] for _ in brains]
    for index, problem in enumerate(problems):
        for group in groups:
            if len(group) < 2:
                # 1体だけ余ったら、直前の組に混ぜて測る。
                group = list(range(max(0, len(brains) - size), len(brains)))
            result = run_match(
                problem, [brains[i] for i in group],
                seed=VALIDATION_SEED_BASE + index, epsilon=0.0, kind_explore=0.0,
            )
            for slot, team in enumerate(result.teams):
                brain_index = group[slot]
                totals[brain_index][0] += team.score.brand_count
                totals[brain_index][1] += team.score.daily_brand_total
                totals[brain_index][2] += team.score.noodle_count
    count = max(1, len(problems))
    return [(t[0] / count, t[1] / count, t[2] / count) for t in totals]


def train(
    brains: list[Brain],
    settings: TrainingSettings,
    models_dir: Path,
    model_name: str,
    seed: int = 20260910,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    match_size = resolve_match_size(settings, len(brains))
    players = match_size
    replay_limit = settings.replay_limit or REPLAY_LIMIT_PER_BRAIN

    replays = [load_replay(models_dir, brain) for brain in brains]
    bank: list[Problem] = []
    if settings.problem_dir:
        directory = Path(settings.problem_dir)
        files = sorted(directory.glob("*.json"))
        if files:
            print(f"問題バンクを読み込みます: {directory} ({len(files)}問)")
            with Progress("問題バンク読込", total=len(files)) as bar:
                for path in files:
                    bank.append(load_problem(path))
                    bar.step()

    validation_problems = build_validation_problems(
        settings.validation_problems, settings.validation_preset, players
    )

    print()
    print(
        f"学習を開始します。エピソード数 {settings.episodes}"
        f" / AI {len(brains)}体 / 1試合 {match_size}チーム"
    )
    print(f"経験の引き継ぎ: " + ", ".join(
        f"{brain.name}={len(replays[i][0])}件" for i, brain in enumerate(brains)
    ))
    print()

    started = time.monotonic()
    last_autosave = started
    interrupted = False
    best_key: tuple[float, float, float] | None = None
    with Progress("学習中", total=settings.episodes) as bar:
      try:
        for episode in range(settings.episodes):
            progress = _global_progress(settings, episode)
            epsilon = settings.epsilon_end + (settings.epsilon_start - settings.epsilon_end) * (
                1.0 - progress
            ) ** 2
            alpha = settings.alpha_start + (settings.alpha_end - settings.alpha_start) * progress

            if bank and rng.random() < 0.5:
                problem = bank[rng.randrange(len(bank))]
            else:
                problem = generate_problem(
                    rng.randrange(1, 2_000_000_000),
                    _curriculum_settings(
                        progress, settings.curriculum, players,
                        rng, settings.random_size_ratio, settings.archetype_bias,
                        settings.honsen_ratio,
                    ),
                )

            # 全員を毎回戦わせる必要はない。試合ごとに顔ぶれを変えると、
            # 1エピソードが軽いまま、いろいろな相手と当たれる。
            entrants = (
                list(range(len(brains)))
                if match_size >= len(brains)
                else rng.sample(range(len(brains)), match_size)
            )
            result = run_match(
                problem,
                [brains[index] for index in entrants],
                seed=rng.randrange(1, 2_000_000_000),
                epsilon=epsilon,
                kind_explore=max(0.05, epsilon),
            )

            for slot, team in enumerate(result.teams):
                team_index = entrants[slot]
                brain = brains[team_index]
                for features, rewards in team.touring_samples:
                    brain.update_touring(features, rewards, alpha)
                    replays[team_index][0].append([features, rewards])
                for features, reward in team.supply_samples:
                    brain.update_supply(features, reward, alpha)
                    replays[team_index][1].append([features, reward])
                brain.record_kind_result(
                    problem.agent_count, team.supply_count, team.score.key
                )
                for recipe_index, reward in team.recipe_results:
                    brain.record_recipe_result(recipe_index, reward)
                brain.trained_episodes += 1

                _replay_updates(
                    brain, replays[team_index], settings.replay_updates_per_episode, alpha, rng
                )

                if len(replays[team_index][0]) > replay_limit:
                    del replays[team_index][0][:-replay_limit]
                if len(replays[team_index][1]) > replay_limit:
                    del replays[team_index][1][:-replay_limit]

            best = max(result.teams, key=lambda t: t.score.key)
            note = f"最高 {best.brain_name} {best.score}"

            if settings.validation_every and (episode + 1) % settings.validation_every == 0:
                averages = validate(brains, validation_problems, match_size)
                for team_index, brain in enumerate(brains):
                    brain.history.append(
                        {
                            "episodes": brain.trained_episodes,
                            "avgBrand": round(averages[team_index][0], 3),
                            "avgDailyBrand": round(averages[team_index][1], 3),
                            "avgNoodles": round(averages[team_index][2], 3),
                        }
                    )
                _save_all(brains, replays, models_dir, model_name, replay_limit)
                key = _validation_key(averages)
                if best_key is None or key > best_key:
                    best_key = key
                    save_population(brains, _best_path(models_dir, model_name))
                note = "検証: " + " / ".join(
                    f"{brains[i].name} 種{averages[i][0]:.1f} 日{averages[i][1]:.1f} 玉{averages[i][2]:.0f}"
                    for i in range(len(brains))
                )

            if time.monotonic() - last_autosave >= AUTOSAVE_SECONDS:
                _save_all(brains, replays, models_dir, model_name, replay_limit)
                last_autosave = time.monotonic()

            bar.step(note=note)
      except KeyboardInterrupt:
        interrupted = True

    if hasattr(gc, "unfreeze"):
        gc.unfreeze()
    _save_all(brains, replays, models_dir, model_name, replay_limit)
    _finish(brains, models_dir, model_name, validation_problems, match_size,
            best_key, started, interrupted)
    return interrupted
    return interrupted


def _best_path(models_dir: Path, model_name: str) -> Path:
    return models_dir / f"{Path(model_name).stem}{BEST_SUFFIX}.json"


def _validation_key(averages: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """検証結果を、公式の優先順位のまま1つのキーにする。"""

    return (
        sum(item[0] for item in averages),
        sum(item[1] for item in averages),
        sum(item[2] for item in averages),
    )


def _finish(
    brains: list[Brain],
    models_dir: Path,
    model_name: str,
    validation_problems: list[Problem],
    match_size: int,
    best_key: tuple[float, float, float] | None,
    started: float,
    interrupted: bool,
) -> None:
    """学習の締め。良かった時点より悪くなっていたら、そこまで巻き戻す。

    学習の成績は上下しますが、保存されるモデルは必ず一番良かったものになります。
    こうしておくと「回した結果、前より弱くなった」が起きません。
    """

    elapsed = time.monotonic() - started
    print()
    if interrupted:
        print("中断しました。ここまでの学習は保存済みです。")
        print("もう一度「2. AIを学習させる」を選べば、この続きから学習できます。")
    print(f"学習が終わりました。所要時間 {format_duration(elapsed)}")

    averages = validate(brains, validation_problems, match_size)
    final_key = _validation_key(averages)
    best_file = _best_path(models_dir, model_name)

    if best_key is not None and final_key < best_key and best_file.exists():
        print()
        print("最後の状態より、途中のほうが成績が良かったので巻き戻します。")
        print(f"  いまの状態  : 種類{final_key[0]:.2f} / 日別{final_key[1]:.2f} / 玉{final_key[2]:.1f}")
        print(f"  一番良い状態: 種類{best_key[0]:.2f} / 日別{best_key[1]:.2f} / 玉{best_key[2]:.1f}")
        restored = load_population(best_file)
        for index, brain in enumerate(restored):
            if index < len(brains):
                brains[index].touring_heads = brain.touring_heads
                brains[index].supply_weights = brain.supply_weights
                brains[index].kind_stats = brain.kind_stats
        save_population(brains, models_dir / model_name)
        averages = validate(brains, validation_problems, match_size)
    else:
        save_population(brains, best_file)

    print(f"モデル: {models_dir / model_name}")
    print("最終検証（固定問題・探索なし）")
    for index, brain in enumerate(brains):
        brand, daily, noodles = averages[index]
        print(
            f"  {brain.name}: 種類{brand:.2f} / 日別累計{daily:.2f} / 玉数{noodles:.1f}"
            f"  （累計{brain.trained_episodes}エピソード, 経験{brain.touring_samples}件）"
        )


def _save_all(
    brains: list[Brain],
    replays: list[tuple[list, list]],
    models_dir: Path,
    model_name: str,
    limit: int = REPLAY_LIMIT_PER_BRAIN,
) -> None:
    save_population(brains, models_dir / model_name)
    for index, brain in enumerate(brains):
        save_replay(models_dir, brain, replays[index][0], replays[index][1], limit)


def _replay_updates(
    brain: Brain,
    replay: tuple[list, list],
    update_count: int,
    alpha: float,
    rng: random.Random,
) -> None:
    touring, supply = replay
    if update_count <= 0:
        return
    touring_updates = update_count * 3 // 4
    supply_updates = update_count - touring_updates
    if touring:
        for _ in range(touring_updates):
            features, rewards = touring[rng.randrange(len(touring))]
            brain.update_touring(features, rewards, alpha * 0.5)
    if supply:
        for _ in range(supply_updates):
            features, reward = supply[rng.randrange(len(supply))]
            brain.update_supply(features, reward, alpha * 0.5)


# ---------------- 並列学習 ----------------
#
# 自己対戦の1エピソードはCPU1コアしか使いません。学習を回す時間が限られている
# ときは、コア数だけプロセスを立てて同時にエピソードを回すのが一番効きます。
# 標準ライブラリの multiprocessing だけを使うので、追加のインストールは不要です。
#
# 1バッチ（プロセス数と同じ本数）を同時に走らせ、返ってきた経験を親プロセスが
# まとめて重みに反映し、更新後の重みを次のバッチに配ります。

def _episode_worker(task: tuple) -> tuple:
    payload, seed, epsilon, kind_explore, settings_dict, problem_path, entrants = task
    from hexa.ai import Brain

    brains = [Brain.from_json_dict(raw) for raw in payload]
    if problem_path:
        problem = load_problem(problem_path)
    else:
        problem = generate_problem(seed, GeneratorSettings(**settings_dict))
    result = run_match(
        problem, brains, seed=seed, epsilon=epsilon, kind_explore=kind_explore
    )
    return (
        problem.agent_count,
        entrants,
        [
            (
                team.touring_samples,
                team.supply_samples,
                team.supply_count,
                team.score.key,
            )
            for team in result.teams
        ],
    )


def train_parallel(
    brains: list[Brain],
    settings: TrainingSettings,
    models_dir: Path,
    model_name: str,
    worker_count: int,
    seed: int = 20260910,
) -> None:
    import multiprocessing as mp

    models_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    match_size = resolve_match_size(settings, len(brains))
    players = match_size
    replay_limit = settings.replay_limit or REPLAY_LIMIT_PER_BRAIN
    replays = [load_replay(models_dir, brain) for brain in brains]

    bank_paths: list[str] = []
    if settings.problem_dir:
        bank_paths = [str(path) for path in sorted(Path(settings.problem_dir).glob("*.json"))]
        if bank_paths:
            print(f"問題バンクを使います: {settings.problem_dir}（{len(bank_paths)}問）")

    validation_problems = build_validation_problems(
        settings.validation_problems, settings.validation_preset, players
    )

    print()
    print(
        f"学習を開始します。エピソード数 {settings.episodes} / 並列 {worker_count} プロセス"
        f" / AI {len(brains)}体 / 1試合 {match_size}チーム"
    )
    print("経験の引き継ぎ: " + ", ".join(
        f"{brain.name}={len(replays[i][0])}件" for i, brain in enumerate(brains)
    ))
    print()

    started = time.monotonic()
    last_autosave = started
    interrupted = False
    best_key: tuple[float, float, float] | None = None
    completed = 0

    # 子プロセスを作る前に、いま持っている経験を「もう触らない世代」に移す。
    # これをしないと、子プロセス側のガベージコレクションが親から引き継いだ
    # メモリに触れてしまい、コピーが発生して並列数のぶんだけメモリを食います。
    gc.collect()
    if hasattr(gc, "freeze"):
        gc.freeze()

    with mp.Pool(processes=worker_count) as pool:
      try:
        def make_task():
            """1エピソード分の指示を作る。"""

            if True:
                if True:
                    progress = _global_progress(settings, completed)
                    epsilon = settings.epsilon_end + (
                        settings.epsilon_start - settings.epsilon_end
                    ) * (1.0 - progress) ** 2
                    problem_path = ""
                    settings_dict = {}
                    if bank_paths and rng.random() < 0.5:
                        problem_path = bank_paths[rng.randrange(len(bank_paths))]
                    else:
                        curriculum = _curriculum_settings(
                            progress, settings.curriculum, players,
                            rng, settings.random_size_ratio, settings.archetype_bias,
                            settings.honsen_ratio,
                        )
                        settings_dict = curriculum.__dict__.copy()
                    entrants = (
                        list(range(len(brains)))
                        if match_size >= len(brains)
                        else rng.sample(range(len(brains)), match_size)
                    )
                    tasks.append(
                        (
                            [brains[index].to_json_dict() for index in entrants],
                            rng.randrange(1, 2_000_000_000),
                            epsilon,
                            max(0.05, epsilon),
                            settings_dict,
                            problem_path,
                            entrants,
                        )
                    )
            return tasks[-1]

        # **終わった子から次の仕事を渡す形。**
        #
        # 以前は「子の数だけ作って全員終わるまで待つ」形でした。
        # 1エピソードは16x16で1.7秒、32x32で19.1秒と11倍違うので、
        # 20並列でも1つ遅いものが全体を縛り、実効33%まで落ちます
        # （CPU使用率43.9%として実測）。
        #
        # 常に worker_count*2 件を流しておき、終わった順に取り込みます。
        with Progress("学習中", total=settings.episodes) as bar:
            inflight = []
            dispatched = 0
            last_validation = 0
            while completed < settings.episodes:
                while (
                    len(inflight) < worker_count * 2
                    and dispatched < settings.episodes
                ):
                    tasks = []
                    inflight.append(pool.apply_async(_episode_worker, (make_task(),)))
                    dispatched += 1

                ready = None
                while ready is None:
                    for index, pending in enumerate(inflight):
                        if pending.ready():
                            ready = inflight.pop(index)
                            break
                    if ready is None:
                        time.sleep(0.001)

                agent_count, entrants, team_results = ready.get()
                alpha = settings.alpha_start + (
                    settings.alpha_end - settings.alpha_start
                ) * _global_progress(settings, completed)

                if True:
                    for slot, (touring, supply, supply_count, score_key) in enumerate(
                        team_results
                    ):
                        team_index = entrants[slot]
                        brain = brains[team_index]
                        for features, rewards in touring:
                            brain.update_touring(features, rewards, alpha)
                            replays[team_index][0].append([features, rewards])
                        for features, reward in supply:
                            brain.update_supply(features, reward, alpha)
                            replays[team_index][1].append([features, reward])
                        brain.record_kind_result(agent_count, supply_count, score_key)
                        brain.trained_episodes += 1
                        _replay_updates(
                            brain,
                            replays[team_index],
                            settings.replay_updates_per_episode,
                            alpha,
                            rng,
                        )
                        if len(replays[team_index][0]) > replay_limit:
                            del replays[team_index][0][:-replay_limit]
                        if len(replays[team_index][1]) > replay_limit:
                            del replays[team_index][1][:-replay_limit]
                    completed += 1
                    bar.set(completed)

                if (
                    settings.validation_every
                    and completed - last_validation >= settings.validation_every
                ):
                    last_validation = completed
                    averages = validate(brains, validation_problems, match_size)
                    for team_index, brain in enumerate(brains):
                        brain.history.append(
                            {
                                "episodes": brain.trained_episodes,
                                "avgBrand": round(averages[team_index][0], 3),
                                "avgDailyBrand": round(averages[team_index][1], 3),
                                "avgNoodles": round(averages[team_index][2], 3),
                            }
                        )
                    _save_all(brains, replays, models_dir, model_name, replay_limit)
                    key = _validation_key(averages)
                    if best_key is None or key > best_key:
                        best_key = key
                        save_population(brains, _best_path(models_dir, model_name))
                    bar.set(completed, note="検証: " + " / ".join(
                        f"{brains[i].name} 種{averages[i][0]:.1f} 日{averages[i][1]:.1f} "
                        f"玉{averages[i][2]:.0f}"
                        for i in range(len(brains))
                    ))

                if time.monotonic() - last_autosave >= AUTOSAVE_SECONDS:
                    _save_all(brains, replays, models_dir, model_name, replay_limit)
                    last_autosave = time.monotonic()
      except KeyboardInterrupt:
        interrupted = True
        pool.terminate()

    if hasattr(gc, "unfreeze"):
        gc.unfreeze()
    _save_all(brains, replays, models_dir, model_name, replay_limit)
    _finish(brains, models_dir, model_name, validation_problems, match_size,
            best_key, started, interrupted)
    return interrupted


# ---------------- おまかせ長時間学習 ----------------
#
# 「一度回したら放置しておけば、いろいろなマップに強い高精度のAIができている」
# を目指すモードです。人が見ていなくても、次のことを自動でやります。
#
#   1. 一定エピソードずつ学習する
#   2. マップの型ごとに成績を測る
#   3. 苦手な型を次のラウンドで多めに出す
#   4. 伸びが止まったらAIを一段階大きくする（判断は変えずに表現力だけ増やす）
#   5. 一番良かった時点を必ず残す
#   6. 途中経過をログに書き出す

AUTO_ROUND_EPISODES = 2000
AUTO_STAGNANT_ROUNDS = 3
AUTO_LOG_NAME = "auto_train_log.txt"
CAPACITY_ORDER = ("simple", "cross", "large")


# 検証に使う試合。本選の3条件をそのまま使う（本選マップに関するお知らせ）。
#   1回戦・敗者復活戦 16x16 / 4体 / 16店 / 系列10-14
#   準決勝           24x24 / 5体 / 24店 / 系列14-20
#   決勝             32x32 / 7体 / 32店 / 系列20-28
#
# 以前は 8x8〜32x32 の自作プリセットで測っていましたが、系列数が本選の
# 1/2〜1/3しかなく（1系列あたり3店 対 本選1.1〜1.6店）、
# 「取り逃してもリカバリが効く」別のゲームで判断していました。
VALIDATION_SIZE_ORDER = tuple(HONSEN_ROUNDS)

# 検証の日数。参考マップに合わせて5日で固定し、測定時間を揃える。
VALIDATION_DAY_COUNT = 5


def validation_sizes(per_archetype: int) -> list[str]:
    """その細かさで実際に測られる試合の種類。

    per_archetype が3未満だと、決勝の条件（32x32・7体・系列20-28）が
    一度も測られません。3以上にしてください。
    """

    return [
        VALIDATION_SIZE_ORDER[index % len(VALIDATION_SIZE_ORDER)]
        for index in range(per_archetype)
    ]


def build_archetype_validation(
    per_archetype: int,
    players: int,
) -> dict[str, list[Problem]]:
    """型ごとの検証問題。本選の3条件を順に割り当てる。"""

    result: dict[str, list[Problem]] = {}
    for shape_index, name in enumerate(ARCHETYPES):
        problems: list[Problem] = []
        for index in range(per_archetype):
            round_name = VALIDATION_SIZE_ORDER[index % len(VALIDATION_SIZE_ORDER)]
            seed = VALIDATION_SEED_BASE + shape_index * 1000 + index
            settings = honsen_settings(
                round_name,
                random.Random(seed),
                players=players,
                archetype=name,
                day_count=VALIDATION_DAY_COUNT,
            )
            problems.append(generate_problem(seed, settings))
        result[name] = problems
    return result


def validate_by_archetype(
    brains: list[Brain],
    problems_by_archetype: dict[str, list[Problem]],
    match_size: int,
) -> dict[str, tuple[float, float, float]]:
    """型ごとに「その集団の中で一番良かった成績」の平均を返す。

    本番モードは全員の案から一番良いものを選ぶので、
    集団の最大値で測るのが実際の強さに近くなります。
    """

    size = max(2, min(len(brains), match_size))
    result: dict[str, tuple[float, float, float]] = {}
    for name, problems in problems_by_archetype.items():
        totals = [0.0, 0.0, 0.0]
        for index, problem in enumerate(problems):
            best: tuple[int, int, int] | None = None
            for start in range(0, len(brains), size):
                group = brains[start:start + size]
                if len(group) < 2:
                    group = brains[-size:]
                outcome = run_match(
                    problem, group, seed=VALIDATION_SEED_BASE + index,
                    epsilon=0.0, kind_explore=0.0,
                )
                for team in outcome.teams:
                    if best is None or team.score.key > best:
                        best = team.score.key
            if best is not None:
                totals[0] += best[0]
                totals[1] += best[1]
                totals[2] += best[2]
        count = max(1, len(problems))
        result[name] = (totals[0] / count, totals[1] / count, totals[2] / count)
    return result


def _overall_key(scores: dict[str, tuple[float, float, float]]) -> tuple[float, float, float]:
    if not scores:
        return (0.0, 0.0, 0.0)
    count = len(scores)
    return (
        sum(item[0] for item in scores.values()) / count,
        sum(item[1] for item in scores.values()) / count,
        sum(item[2] for item in scores.values()) / count,
    )


def _bias_from_scores(scores: dict[str, tuple[float, float, float]]) -> dict[str, float]:
    """成績の悪い型ほど重くする。差が小さければほぼ均等になる。"""

    if not scores:
        return {}
    values = {name: value[2] for name, value in scores.items()}
    best = max(values.values()) or 1.0
    bias: dict[str, float] = {}
    for name, value in values.items():
        shortfall = max(0.0, best - value) / max(1.0, best)
        # 最低でも1.0（均等）、一番苦手な型で最大4.0までにする。
        bias[name] = 1.0 + 3.0 * shortfall
    return bias


def _weights_are_finite(brain: Brain) -> bool:
    """重みが壊れていないか（無限大や NaN になっていないか）を確かめる。

    何十時間も回すので、万一おかしな値に飛んでも気づけるようにしておきます。
    """

    for head in brain.touring_heads:
        for value in head:
            if value != value or value in (float("inf"), float("-inf")):
                return False
            if abs(value) > 1e6:
                return False
    for value in brain.supply_weights:
        if value != value or value in (float("inf"), float("-inf")):
            return False
        if abs(value) > 1e6:
            return False
    return True


def current_capacity(brain: Brain) -> str:
    for name, (touring_cross, _supply_cross) in CAPACITY_PRESETS.items():
        if tuple(touring_cross) == tuple(brain.touring_cross):
            return name
    return "simple"


def auto_train(
    brains: list[Brain],
    settings: TrainingSettings,
    models_dir: Path,
    model_name: str,
    worker_count: int,
    total_episodes: int,
    round_episodes: int = AUTO_ROUND_EPISODES,
    validation_per_archetype: int = 4,
    allow_growth: bool = True,
    log=print,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    log_path = models_dir / AUTO_LOG_NAME
    match_size = resolve_match_size(settings, len(brains))
    problems = build_archetype_validation(validation_per_archetype, match_size)
    covered = validation_sizes(validation_per_archetype)

    def write_log(line: str) -> None:
        log(line)
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    write_log("")
    write_log("=" * 60)
    write_log(
        f"おまかせ学習を開始します "
        f"({time.strftime('%Y-%m-%d %H:%M:%S')})"
    )
    write_log(
        f"  AI {len(brains)}体 / 1試合 {match_size}チーム / "
        f"大きさ {current_capacity(brains[0])} / 目標 {total_episodes} エピソード"
    )
    write_log(
        "  検証する試合: "
        + " / ".join(dict.fromkeys(covered))
        + f" × {len(ARCHETYPES)}型 = {sum(len(v) for v in problems.values())}問"
        f"（各{VALIDATION_DAY_COUNT}日）"
    )
    if "決勝" not in covered:
        write_log(
            "  ※ 決勝の条件（32x32・7体・系列20-28）が検証に入っていません。"
            "小さい試合だけを見て判断することになります。"
        )

    # 前回までの最高を引き継ぐ。これをしないと、学習を止めて再開したとき
    # 1ラウンド目が無条件で「過去最高」になり、**前回より悪い状態で
    # population_best.json を上書きしてしまいます。**
    best_key: tuple[float, float, float] | None = None
    best_file = _best_path(models_dir, model_name)
    if best_file.exists():
        try:
            previous = load_population(best_file)
            if len(previous[0].touring_heads[0]) == len(brains[0].touring_heads[0]):
                best_key = _overall_key(
                    validate_by_archetype(previous, problems, match_size)
                )
                write_log(
                    f"  前回までの最高を引き継ぎます"
                    f"（種類{best_key[0]:.2f} 日別{best_key[1]:.2f} 玉{best_key[2]:.1f}）"
                )
        except (ValueError, OSError, IndexError):
            pass

    stagnant = 0
    # これまでに回した総数。表示を「今回の分」と取り違えないようにする。
    already = min(brain.trained_episodes for brain in brains) if brains else 0
    done = 0
    round_index = 0
    started = time.monotonic()

    while done < total_episodes:
        round_index += 1
        this_round = min(round_episodes, total_episodes - done)
        round_settings = TrainingSettings(
            episodes=this_round,
            progress_offset=done,
            progress_total=total_episodes,
            curriculum=settings.curriculum,
            epsilon_start=settings.epsilon_start,
            epsilon_end=settings.epsilon_end,
            alpha_start=settings.alpha_start,
            alpha_end=settings.alpha_end,
            replay_updates_per_episode=settings.replay_updates_per_episode,
            validation_every=0,
            validation_problems=settings.validation_problems,
            validation_preset=settings.validation_preset,
            problem_dir=settings.problem_dir,
            match_size=settings.match_size,
            random_size_ratio=settings.random_size_ratio,
            honsen_ratio=settings.honsen_ratio,
            replay_limit=settings.replay_limit,
            archetype_bias=settings.archetype_bias,
        )
        # ラウンドごとに種を変える。同じ種のままだと、毎回まったく同じ
        # 問題列を解き直すことになり、何回まわしても経験が増えません。
        round_seed = 20260910 + round_index * 1_000_003
        if worker_count > 1:
            interrupted = train_parallel(
                brains, round_settings, models_dir, model_name, worker_count,
                seed=round_seed,
            )
        else:
            interrupted = train(
                brains, round_settings, models_dir, model_name, seed=round_seed
            )
        done += this_round

        validation_started = time.monotonic()
        scores = validate_by_archetype(brains, problems, match_size)
        validation_seconds = time.monotonic() - validation_started
        key = _overall_key(scores)
        settings.archetype_bias = _bias_from_scores(scores)

        elapsed = format_duration(time.monotonic() - started)
        write_log("")
        write_log(
            f"[{round_index}] 今回 {done}/{total_episodes} "
            f"（通算 {already + done}） エピソード "
            f"（経過 {elapsed} / 大きさ {current_capacity(brains[0])}"
            f" / 検証に {format_duration(validation_seconds)}）"
        )
        for name in sorted(scores):
            brand, daily, noodles = scores[name]
            weight = settings.archetype_bias.get(name, 1.0)
            write_log(
                f"    {name:9s} 種類{brand:5.2f} 日別{daily:6.2f} 玉{noodles:7.1f}"
                f"   次の出やすさ x{weight:.1f}"
            )

        for brain in brains:
            brain.history.append(
                {
                    "episodes": brain.trained_episodes,
                    "avgBrand": round(key[0], 3),
                    "avgDailyBrand": round(key[1], 3),
                    "avgNoodles": round(key[2], 3),
                    "capacity": current_capacity(brain),
                }
            )

        broken = [brain.name for brain in brains if not _weights_are_finite(brain)]
        if broken:
            write_log(f"    → 重みが壊れたAIがいます（{', '.join(broken)}）。最良の状態に戻します。")
            best_file = _best_path(models_dir, model_name)
            if best_file.exists():
                restored = load_population(best_file)
                for index, brain in enumerate(restored):
                    if index < len(brains):
                        brains[index].touring_heads = brain.touring_heads
                        brains[index].supply_weights = brain.supply_weights
                save_population(brains, models_dir / model_name)
            continue

        if best_key is None or key > best_key:
            best_key = key
            save_population(brains, _best_path(models_dir, model_name))
            stagnant = 0
            write_log("    → 過去最高。この状態を保存しました。")
        else:
            stagnant += 1
            write_log(f"    → 更新なし（{stagnant}回連続）")

        if interrupted:
            write_log("    → 中断を検出しました。ここで終わります。")
            break

        if allow_growth and stagnant >= AUTO_STAGNANT_ROUNDS:
            capacity = current_capacity(brains[0])
            index = CAPACITY_ORDER.index(capacity)
            if index + 1 < len(CAPACITY_ORDER):
                target = CAPACITY_ORDER[index + 1]
                for brain in brains:
                    brain.grow(target)
                best_file = _best_path(models_dir, model_name)
                if best_file.exists():
                    best_brains = load_population(best_file)
                    for brain in best_brains:
                        brain.grow(target)
                    save_population(best_brains, best_file)
                save_population(brains, models_dir / model_name)
                stagnant = 0
                best_key = key
                write_log(
                    f"    → 伸びが止まったので {capacity} から {target} に大きくしました。"
                    "（判断は変わりません）"
                )
            else:
                write_log("    → すでに一番大きい large です。このまま続けます。")
                stagnant = 0

    # 最後に、一番良かった状態に戻す。
    best_file = _best_path(models_dir, model_name)
    if best_file.exists():
        try:
            best_brains = load_population(best_file)
            if len(best_brains[0].touring_heads[0]) == len(brains[0].touring_heads[0]):
                for index, brain in enumerate(best_brains):
                    if index < len(brains):
                        brains[index].touring_heads = brain.touring_heads
                        brains[index].supply_weights = brain.supply_weights
                        brains[index].kind_stats = brain.kind_stats
                save_population(brains, models_dir / model_name)
                write_log("")
                write_log("一番良かった状態を採用しました。")
        except ValueError:
            pass

    write_log(f"おまかせ学習を終わります。所要時間 {format_duration(time.monotonic() - started)}")
    write_log(f"ログ: {log_path}")
