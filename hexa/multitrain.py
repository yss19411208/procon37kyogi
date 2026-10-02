"""AIの集団を複数セット、同時に学習させる。

**なぜこうするか**

学習は「子プロセスが試合を解く → 親プロセスが重みを更新する」流れですが、
親の更新は直列です。64コアあっても親が1コアで詰まるので、
子をいくら増やしても頭打ちになります（実測でCPU使用率1.9%）。

そこで**学習そのものを複数セットに分けて、親を複数立てます。**
セットごとに独立した集団・独立した経験・独立したフォルダを持つので、
親が3つなら3倍、6つなら6倍の速さで学習が進みます。

おまけとして、セットごとに違う個性の集団が育ちます。
本番モードは全AIの案から一番良いものを選ぶので、
バラバラに育った集団を最後に合流させると、案の幅が広がります。

合流するとき、セットごとにAIの大きさ（simple / cross / large）が
違っていても構いません。大きさはAI1体ごとにモデルへ保存されるので、
混ざっていてもそのまま動きます。
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

from hexa.ai import Brain, load_population, new_population, save_population
from hexa.training import (
    AUTO_LOG_NAME,
    TrainingSettings,
    auto_train,
    replay_limit_for_budget,
)
from hexa.ai import supply_feature_count, touring_feature_count

MODEL_NAME = "population.json"


def set_dir(models_dir: Path, index: int) -> Path:
    return models_dir / f"set{index + 1}"


@dataclass
class SetPlan:
    set_count: int
    brains_per_set: int
    episodes: int
    workers_per_set: int
    memory_gb_per_set: int
    match_size: int
    round_episodes: int
    validation_per_archetype: int
    capacity: str
    allow_growth: bool


def _train_one_set(index: int, models_dir_text: str, plan: SetPlan) -> None:
    """1セット分の学習。別プロセスとして動く。"""

    models_dir = Path(models_dir_text)
    directory = set_dir(models_dir, index)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_NAME

    if path.exists():
        brains = load_population(path)
    else:
        # セットごとに種を変える。同じ初期値から始めると個性が育たない。
        brains = new_population(
            seed=20260910 + index * 1_000_003,
            count=plan.brains_per_set,
            capacity=plan.capacity,
        )
        save_population(brains, path)

    replay_limit = replay_limit_for_budget(
        plan.memory_gb_per_set * 1024 ** 3,
        len(brains),
        touring_feature_count(brains[0].touring_cross),
        supply_feature_count(brains[0].supply_cross),
    )
    settings = TrainingSettings(
        episodes=plan.episodes,
        validation_every=0,
        match_size=min(plan.match_size, len(brains)),
        replay_limit=replay_limit,
    )
    auto_train(
        brains,
        settings,
        directory,
        MODEL_NAME,
        worker_count=plan.workers_per_set,
        total_episodes=plan.episodes,
        round_episodes=plan.round_episodes,
        validation_per_archetype=plan.validation_per_archetype,
        allow_growth=plan.allow_growth,
        log=lambda message: None,   # 画面が混ざるので、各セットのログはファイルだけ
    )


def train_sets(models_dir: Path, plan: SetPlan, log=print) -> None:
    """複数セットを同時に学習させる。"""

    models_dir.mkdir(parents=True, exist_ok=True)
    log("")
    log(
        f"{plan.set_count}セットを同時に学習します"
        f"（1セット {plan.brains_per_set}体 / プロセス {plan.workers_per_set} / "
        f"メモリ {plan.memory_gb_per_set}GB）"
    )
    log(f"  合計 {plan.set_count * plan.brains_per_set} 体、"
        f"使うプロセスは親{plan.set_count} + 子{plan.set_count * plan.workers_per_set} 個")
    log("  各セットの進み具合は次のファイルに出ます。")
    for index in range(plan.set_count):
        log(f"    {set_dir(models_dir, index) / AUTO_LOG_NAME}")
    log("  Ctrl+C でいつ止めても、各セットの学習は保存済みです。")
    log("")

    processes: list[mp.Process] = []
    for index in range(plan.set_count):
        process = mp.Process(
            target=_train_one_set,
            args=(index, str(models_dir), plan),
            daemon=False,
        )
        process.start()
        processes.append(process)
        # 同時に起動すると検証が重なって重いので、少しずらす。
        time.sleep(0.5)

    started = time.monotonic()
    try:
        while any(process.is_alive() for process in processes):
            time.sleep(5.0)
    except KeyboardInterrupt:
        log("")
        log("  中断します。各セットの保存を待っています…")
        for process in processes:
            process.terminate()
    finally:
        for process in processes:
            process.join(timeout=60)

    elapsed = time.monotonic() - started
    log("")
    log(f"学習を終わりました（{int(elapsed // 60)}分{int(elapsed % 60)}秒）")
    for index in range(plan.set_count):
        path = set_dir(models_dir, index) / MODEL_NAME
        if not path.exists():
            log(f"  セット{index + 1}: 保存がありません")
            continue
        brains = load_population(path)
        log(f"  セット{index + 1}: {len(brains)}体 / 累計 {brains[0].trained_episodes} エピソード")


def collect_sets(models_dir: Path) -> list[Brain]:
    """全セットのAIを1つの集団にまとめる。

    名前が重ならないように「セット番号-元の名前」に付け替えます。
    大きさが違っても、AI1体ごとに保存されているのでそのまま混ぜられます。
    """

    brains: list[Brain] = []
    index = 0
    while True:
        path = set_dir(models_dir, index) / MODEL_NAME
        if not path.exists():
            break
        for brain in load_population(path):
            brain.name = f"S{index + 1}-{brain.name}"
            brains.append(brain)
        index += 1
    return brains


def merge_sets(models_dir: Path, output_name: str = MODEL_NAME, log=print) -> int:
    """全セットをまとめて models/population.json に書き出す。"""

    brains = collect_sets(models_dir)
    if not brains:
        log("  まとめるセットが見つかりません。")
        return 0
    save_population(brains, models_dir / output_name)
    from collections import Counter

    from hexa.ai import CAPACITY_PRESETS

    sizes: Counter = Counter()
    for brain in brains:
        for name, (touring_cross, _supply) in CAPACITY_PRESETS.items():
            if tuple(touring_cross) == tuple(brain.touring_cross):
                sizes[name] += 1
                break
    log(f"  {len(brains)}体を {models_dir / output_name} にまとめました。")
    log("  大きさの内訳: " + " / ".join(f"{k} {v}体" for k, v in sizes.items()))
    log(f"  累計エピソード: 最小 {min(b.trained_episodes for b in brains)}"
        f" 〜 最大 {max(b.trained_episodes for b in brains)}")
    return len(brains)
