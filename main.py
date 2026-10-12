"""ヘキサうどん AI ── 対話メニュー

引数は使いません。すべてこの画面の質問に答えるだけで動きます。

    python main.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import unittest
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from hexa.ai import (
    CAPACITY_PRESETS,
    load_population,
    new_population,
    save_population,
    supply_feature_count,
    touring_feature_count,
)
from hexa.answers import write_answers
from hexa.generator import GeneratorSettings, PRESETS, generate_problem
from hexa.match import run_match
from hexa.problem import Problem, load_problem, save_problem, validate_problem
from hexa.progress import Progress, format_duration
from hexa.training import (
    AUTO_LOG_NAME,
    validation_sizes,
    TrainingSettings,
    auto_train,
    current_capacity,
    build_validation_problems,
    replay_limit_for_budget,
    train,
    train_parallel,
    validate,
)
from hexa.viz import build_visualization, write_visualization

MODELS_DIR = BASE_DIR / "models"
PROBLEM_DIR = BASE_DIR / "problems"
ANSWER_DIR = BASE_DIR / "answers"
VIEWER_DIR = BASE_DIR / "viewer"
SERVER_DIR = BASE_DIR / "server"
MACHINE_FILE = BASE_DIR / "machine.json"
MODEL_NAME = "population.json"


# ---------------- 入力ヘルパー ----------------

def ask(prompt: str, default: str = "") -> str:
    suffix = f"（既定 {default}）" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        print()
        return default
    return answer or default


def ask_int(prompt: str, default: int, low: int, high: int) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            print(f"  数字を入れてください（{low}～{high}）")
            continue
        if not low <= value <= high:
            print(f"  {low}～{high} の範囲で入れてください")
            continue
        return value


def ask_choice(prompt: str, options: list[str], default: str) -> str:
    while True:
        raw = ask(f"{prompt} [{' / '.join(options)}]", default)
        if raw in options:
            return raw
        print(f"  {', '.join(options)} のどれかを入れてください")


def ask_yes_no(prompt: str, default: bool) -> bool:
    return ask_choice(prompt, ["y", "n"], "y" if default else "n") == "y"


# ---------------- このPCの性能 ----------------

def load_machine() -> dict:
    """学習用PCの性能をおぼえておく。毎回聞かれなくて済む。"""

    if MACHINE_FILE.exists():
        try:
            with open(MACHINE_FILE, encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_machine(data: dict) -> None:
    with open(MACHINE_FILE, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def menu_machine() -> None:
    import multiprocessing

    print()
    print("== このPCの性能を登録する ==")
    print("  一度入れておけば、学習のたびに聞かれなくなります。")
    detected = multiprocessing.cpu_count()
    machine = load_machine()
    print(f"  検出したコア数: {detected}")
    cores = ask_int("使ってよいコア数", machine.get("cores", max(1, detected - 1)), 1, 256)
    memory = ask_int("学習に使ってよいメモリ（GB）", machine.get("memoryGb", 4), 1, 1024)
    reserve = ask_yes_no("フルに使いますか（他の作業をしないなら y）",
                         machine.get("full", False))
    save_machine({"cores": cores, "memoryGb": memory, "full": reserve})
    print()
    print(f"  登録しました: コア{cores} / メモリ{memory}GB / フル使用{'する' if reserve else 'しない'}")
    if not reserve:
        print("  フル使用を n にしたので、実際にはコアを1つ、メモリを2割残して使います。")


def machine_cores() -> int:
    import multiprocessing

    machine = load_machine()
    cores = machine.get("cores", max(1, multiprocessing.cpu_count() - 1))
    if not machine.get("full", False):
        cores = max(1, cores - 1)
    return max(1, int(cores))


def machine_memory_gb() -> int:
    machine = load_machine()
    memory = machine.get("memoryGb", 4)
    if not machine.get("full", False):
        memory = max(1, int(memory * 0.8))
    return max(1, int(memory))


# ---------------- モデル ----------------

def get_population(create_if_missing: bool = True):
    path = MODELS_DIR / MODEL_NAME
    if path.exists():
        try:
            return load_population(path), True
        except ValueError as error:
            print(f"  保存済みモデルを読めませんでした: {error}")
            if not ask_yes_no("  新しく作り直しますか", True):
                raise
    if not create_if_missing:
        raise FileNotFoundError("学習済みモデルがありません")
    return new_population(), False


def settings_from_input(players: int) -> GeneratorSettings:
    preset_name = ask_choice("マップの大きさ", list(PRESETS.keys()), "medium")
    base = PRESETS[preset_name]
    if not ask_yes_no("細かい設定も変えますか", False):
        return GeneratorSettings(
            height=base.height, width=base.width, agent_count=base.agent_count,
            spot_count=base.spot_count, brand_count=base.brand_count,
            day_count=base.day_count, players=players,
        )
    height = ask_int("マップの縦", base.height, 8, 32)
    width = ask_int("マップの横", base.width, 8, 32)
    agent_count = ask_int("1チームのエージェント数", base.agent_count, 3, 8)
    spot_count = ask_int("スポット数", base.spot_count, agent_count, max(height, width))
    brand_count = ask_int("系列数", min(base.brand_count, spot_count), 1, spot_count)
    day_count = ask_int("日数", base.day_count, 4, 10)
    return GeneratorSettings(
        height=height, width=width, agent_count=agent_count,
        spot_count=spot_count, brand_count=brand_count,
        day_count=day_count, players=players,
    )


def ask_players(default: int = 4) -> int:
    print()
    print("  players（1試合の参加チーム数）について")
    print("  道路の交通量は「全チームの滞在ステップ数の合計 ÷ players」で決まります。")
    print("  本選の組合せ（付録7）では1試合 9～10チームです。")
    print("  1試合に出すチーム数と揃えておくのが素直です。")
    print("  道路の混み具合は、合算をチーム数で割った平均なので、")
    print("  チーム数が変わってもほとんど変わりません（実測で4チーム20%、9チーム21%）。")
    return ask_int("players", default, 1, 10)


# ---------------- 各メニュー ----------------

def menu_generate() -> None:
    print()
    print("== 問題を作る ==")
    players = ask_players()
    settings = settings_from_input(players)
    count = ask_int("何問作りますか", 20, 1, 100000)
    seed = ask_int("乱数の種", 1, 0, 2_000_000_000)
    directory = PROBLEM_DIR / ask("保存先フォルダ名", f"{settings.height}x{settings.width}_p{players}")
    directory.mkdir(parents=True, exist_ok=True)

    print()
    started = time.monotonic()
    with Progress("問題を生成中", total=count) as bar:
        for index in range(count):
            problem = generate_problem(seed + index, settings)
            save_problem(problem, directory / f"problem_{index:05d}.json")
            bar.step(note=f"{problem.height}x{problem.width} 日数{problem.day_count}")
    print(f"{count}問を {directory} に保存しました（{format_duration(time.monotonic() - started)}）")


def menu_train() -> None:
    print()
    print("== AIを学習させる ==")
    brains, resumed = get_population()
    if resumed:
        print("  保存済みモデルの続きから学習します。")
        for brain in brains:
            print(f"    {brain.name}: 累計{brain.trained_episodes}エピソード / 経験{brain.touring_samples}件")
    else:
        print("  新しいAIを作ります。")

    print()
    print("  エピソード数の目安。1エピソードは1コアで0.5〜1秒（9チーム戦）なので、")
    print("  コア数で割ったものが実時間です。8コアなら1エピソード0.1秒前後。")
    print()
    print("    5,000      … 8コアで10分ほど")
    print("    50,000     … 8コアで1.5時間ほど")
    print("    500,000    … 8コアで半日〜1日")
    print("    5,000,000  … 8コアで1週間。本選まで1か月なら現実的ではありません")
    print()
    print("  途中で Ctrl+C を押しても保存済みです（2分ごとに自動保存）。")
    print("  もう一度この項目を選べば続きから学習できるので、")
    print("  何回かに分けて回すのが安全です。")
    print()
    episodes = ask_int("エピソード数", 3000, 1, 10_000_000)
    if episodes > 200000:
        seconds = episodes * 0.7 / max(1, machine_cores())
        print()
        print(f"  このPCだと、ざっと {format_duration(seconds)} かかります。")
        print("  AIの大きさが simple のままなら、これだけ回しても伸びません")
        print("  （実測で数百エピソードから先は横ばい）。")
        print("  長時間回すなら、メニュー10で cross か large を選んでください。")
        if not ask_yes_no("  このまま進めますか", True):
            episodes = ask_int("エピソード数", 5000, 1, 10_000_000)
    # 検証は重いので間隔を空けるが、空けすぎると途中経過が見えなくなる。
    default_validation = max(1, min(episodes // 10, 250))
    validation_every = ask_int("何エピソードごとに検証するか", default_validation, 1, 100000)
    print("  （検証の間隔とは別に、2分ごとに自動保存します。")
    print("    途中で Ctrl+C を押しても、そこまでの学習は残ります。）")

    problem_dir = None
    if PROBLEM_DIR.exists() and any(PROBLEM_DIR.iterdir()):
        folders = sorted(p.name for p in PROBLEM_DIR.iterdir() if p.is_dir())
        if folders and ask_yes_no("作り置きの問題を使いますか", True):
            print("  " + " / ".join(folders))
            chosen = ask("フォルダ名", folders[0])
            candidate = PROBLEM_DIR / chosen
            if candidate.is_dir():
                problem_dir = str(candidate)
            else:
                print("  そのフォルダはありません。毎回生成する方式で進めます。")

    default_match_size = min(len(brains), 9)
    match_size = default_match_size
    if len(brains) > 2:
        print()
        print(f"  AIは {len(brains)} 体います。1試合に何チーム出しますか。")
        print("  毎回その人数だけランダムに選ばれて戦います。")
        print("  本選の1試合は9〜10チームです。少なくすると1エピソードが軽くなります。")
        match_size = ask_int("1試合のチーム数", default_match_size, 2, len(brains))

    print()
    print("  貯めておく経験の量を、使ってよいメモリの大きさで決めます。")
    print("  経験は学習中ずっとメモリに載せるので、ここが実質の上限です。")
    print("  ディスクにはgzipで書くので、ファイルはこの1/5〜1/10に収まります。")
    print()
    print("  ここで指定するのは**親プロセス1つぶん**です。学習用のプロセスは")
    print("  別に動くので、その分の余裕を残してください。")
    print("  目安は「PCの搭載メモリ ÷ 4」まで。それ以上にすると、")
    print("  メモリ不足でプロセスごと強制終了されることがあります。")
    memory_gb = ask_int("経験に使ってよいメモリ（GB）",
                        max(1, machine_memory_gb() // 4), 1, 1024)
    replay_limit = replay_limit_for_budget(
        memory_gb * 1024 ** 3,
        len(brains),
        touring_feature_count(brains[0].touring_cross),
        supply_feature_count(brains[0].supply_cross),
    )
    print(f"  1体あたり {replay_limit:,} 件まで貯めます（{len(brains)}体で"
          f" 約{replay_limit * len(brains):,} 件）。")

    # すでに学習済みなら、その続きとして探索率と学習率を進める。
    done_so_far = min(brain.trained_episodes for brain in brains) if brains else 0
    settings = TrainingSettings(
        episodes=episodes,
        validation_every=validation_every,
        problem_dir=problem_dir,
        match_size=match_size,
        replay_limit=replay_limit,
        progress_offset=done_so_far,
        progress_total=done_so_far + episodes,
    )

    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    workers = 1
    if cpu_count > 1:
        print()
        print(f"  このPCのコア数は {cpu_count} です。")
        print("  複数プロセスで同時にエピソードを回すと、その分だけ速く学習できます。")
        workers = ask_int("使うプロセス数（1なら並列にしない）", machine_cores(), 1, cpu_count * 4)

    seed = 20260910 + done_so_far * 7919 + int(time.time()) % 100003
    if workers > 1:
        train_parallel(brains, settings, MODELS_DIR, MODEL_NAME,
                       worker_count=workers, seed=seed)
    else:
        train(brains, settings, MODELS_DIR, MODEL_NAME, seed=seed)


def menu_solve() -> None:
    print()
    print("== 問題をAI全員で解く ==")
    try:
        brains, resumed = get_population(create_if_missing=False)
    except FileNotFoundError:
        print("  学習済みモデルがありません。先に「2. AIを学習させる」を実行してください。")
        print("  未学習のまま解くこともできますが、成績はかなり低くなります。")
        if not ask_yes_no("  未学習のまま進めますか", False):
            return
        brains = new_population()
        resumed = False
    if resumed:
        total = sum(brain.trained_episodes for brain in brains)
        print(f"  学習済みモデルを使います（{len(brains)}体合計 {total} エピソード）。")

    source = ask_choice("問題をどうしますか", ["new", "file"], "new")
    if source == "new":
        players = ask_players(len(brains))
        settings = settings_from_input(players)
        seed = ask_int("乱数の種", 42, 0, 2_000_000_000)
        problem = generate_problem(seed, settings)
        PROBLEM_DIR.mkdir(parents=True, exist_ok=True)
        problem_path = PROBLEM_DIR / f"solved_seed{seed}.json"
        save_problem(problem, problem_path)
        print(f"  問題を {problem_path} に保存しました。")
    else:
        raw_path = ask("問題ファイルのパス", "")
        if not raw_path:
            print("  パスが空です。")
            return
        path = Path(raw_path).expanduser()
        if not path.exists():
            print(f"  ファイルが見つかりません: {path}")
            return
        try:
            problem = load_problem(path)
        except Exception as error:
            print(f"  この JSON は問題ファイルとして読めませんでした: {error}")
            return
        issues = validate_problem(problem)
        if issues:
            print("  この問題は公式の範囲から外れています:")
            for issue in issues:
                print(f"    - {issue}")
            if not ask_yes_no("  それでも解きますか", False):
                return

    slug = ask("この結果の名前", f"solve_{time.strftime('%Y%m%d_%H%M%S')}")

    # players と実際に戦うチーム数を必ず揃える。
    # ずれていると、交通量が「合算 ÷ players」で計算されるので、
    # 道路の混み方が本番とまったく違うものになってしまう。
    team_count = max(2, min(len(brains), problem.players))
    entrants = brains[:team_count]
    print()
    if team_count < len(brains):
        print(f"  この問題は players={problem.players} なので、AI {len(brains)}体のうち"
              f" 先頭{team_count}体で戦います。")
        print("  （交通量は「全チームの滞在数 ÷ players」なので、")
        print("    チーム数と players がずれると道路の混み方が変わってしまいます）")
    print(f"  {team_count}体のAIが同時に解きます。")
    started = time.monotonic()
    result = run_match(problem, entrants, seed=12345, epsilon=0.0, kind_explore=0.0,
                       record_trace=True)
    elapsed = time.monotonic() - started

    print()
    print(f"  解答時間 {elapsed:.2f}秒（{problem.day_count}日分・{team_count}体合計）")
    print()
    print("  順位  AI      種類  日別累計  玉数  補給車  有効な補給  空振り")
    for team in sorted(result.teams, key=lambda t: t.rank):
        print(
            f"   {team.rank}    {team.brain_name:6s} "
            f"{team.score.brand_count:4d} {team.score.daily_brand_total:9d} "
            f"{team.score.noodle_count:6d} {team.supply_count:6d} "
            f"{team.effective_refuels:10d} {team.wasted_refuels:7d}"
        )

    payload = build_visualization(result, title=slug)
    overlap = payload["overlapRate"] * 100
    print()
    print(f"  AI同士の位置一致率 {overlap:.1f}%（低いほど別々の動きをしています）")

    answer_paths = write_answers(result, ANSWER_DIR / slug)
    json_path = write_visualization(payload, VIEWER_DIR, slug)
    print(f"  公式フォーマットの回答: {ANSWER_DIR / slug}（{len(answer_paths)}ファイル）")
    print(f"  ビューア用データ: {json_path}")
    print("  「4. ビューアを開く」で動きを見られます。")


def menu_viewer() -> None:
    print()
    print("== ビューアを開く ==")
    data_dir = VIEWER_DIR / "data"
    if not data_dir.exists() or not list(data_dir.glob("*.json")):
        print("  表示できるデータがありません。先に「3. 問題をAI全員で解く」を実行してください。")
        return

    if ask_yes_no("ローカルサーバーを立てて開きますか", True):
        port = ask_int("ポート番号", 8765, 1024, 65535)
        url = f"http://localhost:{port}/index.html"
        print(f"  {url} を開きます。終了するには Ctrl+C を押してください。")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            subprocess.run(
                [sys.executable, "-m", "http.server", str(port)],
                cwd=str(VIEWER_DIR),
            )
        except KeyboardInterrupt:
            print("\n  サーバーを止めました。")
    else:
        print(f"  次のファイルをブラウザで開いてください:\n    {VIEWER_DIR / 'index.html'}")


def menu_status() -> None:
    print()
    print("== 学習の状況 ==")
    path = MODELS_DIR / MODEL_NAME
    if not path.exists():
        print("  まだ学習していません。")
        return
    brains, _ = get_population()
    for brain in brains:
        print()
        print(f"  {brain.name}")
        print(f"    累計エピソード : {brain.trained_episodes}")
        print(f"    巡回車の経験    : {brain.touring_samples} 件")
        print(f"    補給車の経験    : {brain.supply_samples} 件")
        if brain.kind_stats:
            print("    補給車の台数別の平均成績（種類 / 日別累計 / 玉数）")
            for key in sorted(brain.kind_stats, key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1]))):
                trials, brand, daily, noodles = brain.kind_stats[key]
                agents, supply = key.split(":")
                print(
                    f"      エージェント{agents}体・補給車{supply}台 : "
                    f"{brand / trials:.2f} / {daily / trials:.2f} / {noodles / trials:.1f}"
                    f"  ({int(trials)}回)"
                )
        if brain.history:
            print("    検証スコアの推移（新しい方から5件）")
            for entry in brain.history[-5:][::-1]:
                capacity = entry.get("capacity", "")
                suffix = f" [{capacity}]" if capacity else ""
                print(
                    f"      {entry['episodes']:>7} エピソード時点 : "
                    f"種類{entry['avgBrand']} / 日別累計{entry['avgDailyBrand']}"
                    f" / 玉数{entry['avgNoodles']}{suffix}"
                )

    if ask_yes_no("いま検証を1回走らせますか", False):
        players = ask_int("players", 4, 1, 10)
        count = ask_int("検証問題の数", 3, 1, 50)
        problems = build_validation_problems(count, "medium", players)
        with Progress("検証中", total=1) as bar:
            averages = validate(brains, problems, min(len(brains), players))
            bar.step()
        for index, brain in enumerate(brains):
            brand, daily, noodles = averages[index]
            print(f"  {brain.name}: 種類{brand:.2f} / 日別累計{daily:.2f} / 玉数{noodles:.1f}")


def menu_tests() -> None:
    print()
    print("== 動作テスト ==")
    loader = unittest.TestLoader()
    suite = loader.discover(str(BASE_DIR / "tests"), top_level_dir=str(BASE_DIR))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def _remove_tree(path: Path) -> list[str]:
    """フォルダを消す。消せなかったファイルの一覧を返す。

    Windows では、ビューアをブラウザで開いたままだと data フォルダの
    ファイルが掴まれていて消せないことがあります。読み取り専用属性が
    付いている場合もあるので、そこは外してから消し直します。
    """

    failures: list[str] = []

    def on_error(_function, failed_path, _info) -> None:
        try:
            import stat

            Path(failed_path).chmod(stat.S_IWRITE)
            Path(failed_path).unlink()
        except OSError as error:
            failures.append(f"{failed_path}: {error}")

    try:
        shutil.rmtree(path, onerror=on_error)
    except OSError as error:
        failures.append(f"{path}: {error}")
    return failures


def menu_reset() -> None:
    print()
    print("== 学習をやり直す ==")
    print("  models/ の中身（重みと経験）を消して、AIを新品に戻します。")
    if not ask_yes_no("本当に消しますか", False):
        print("  やめました。")
        return

    print()
    print("  何体のAIを育てますか。")
    print("  体数は、本番モードで毎日作る案の数になります。多いほど当たりを引きやすく")
    print("  なりますが、1体あたりに回る学習回数はその分減ります。")
    print()
    print("  1試合に何チーム出すかは別に決められます（学習のとき、毎回その人数だけ")
    print("  ランダムに選ばれて戦います）。本選の1試合は9〜10チームなので、")
    print("  9か10にしておくと本番に近い条件で学習できます。")
    print()
    print("  実測（12x12・6日・1コア）: 1試合4チーム 0.42秒/エピソード、")
    print("  9チーム 0.66秒/エピソード。1エピソードの重さは体数ではなく")
    print("  「1試合のチーム数」で決まります。")
    print()
    print("  迷ったら 体数10 / 1試合9チーム あたりが無難です。")
    count = ask_int("体数", 10, 2, 32)

    print()
    print("  AIの大きさ（＝覚えられる判断の細かさ）を選びます。")
    print()
    print("  AIは「この店に行くべきか」を、32個の材料（距離・燃料・在庫・未獲得の")
    print("  系列か・残り日数・マップの広さ・地形の割合など）に重みを掛けて")
    print("  足すことで決めています。")
    print("  この形だと材料を足し算でしか使えないので、")
    print("    『未獲得の系列なら遠くても行く、獲得済みなら近いときだけ行く』")
    print("  のような条件付きの判断が書けません。")
    print("  そこで材料どうしの掛け算を足すと書けるようになります。")
    print("  重みの数が増えるほど細かく使い分けられますが、その分たくさん")
    print("  学習しないと当てずっぽうが増えて、かえって弱くなります。")
    print()
    for key in ("simple", "cross", "large"):
        touring_cross, supply_cross = CAPACITY_PRESETS[key]
        print(
            f"    {key:7s} 巡回車{touring_feature_count(touring_cross):4d}個 / "
            f"補給車{supply_feature_count(supply_cross):3d}個の重み"
        )
    print()
    print("    simple … 材料をそのまま使う。〜1万エピソード向け")
    print("    cross  … よく効く材料どうしの掛け算を追加。数万〜数十万向け")
    print("    large  … 材料をほぼ全部掛け合わせる。数十万以上向け（未検証）")
    print()
    print("  実測: 600エピソード程度では simple が cross に勝ちました（玉170対150）。")
    print("  長時間回す予定がないなら simple のままにしてください。")
    capacity = ask_choice("AIの大きさ", ["simple", "cross", "large"], "simple")

    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    brains = new_population(count=count, capacity=capacity)
    save_population(brains, MODELS_DIR / MODEL_NAME)
    print(f"  新しいAIを{count}体作りました（大きさ: {capacity}）。")
    print("  探索の強さ: " + " / ".join(
        f"{brain.name}={brain.explore_scale}" for brain in brains
    ))

    extra = [
        ("problems", PROBLEM_DIR, "生成した問題のJSON"),
        ("answers", ANSWER_DIR, "書き出した回答のJSON"),
        ("viewer/data", VIEWER_DIR / "data", "ビューア用データのJSON"),
    ]
    present = [item for item in extra if item[1].exists() and any(item[1].iterdir())]
    if not present:
        return
    print()
    print("  ほかにも消せるものがあります。")
    for name, path, note in present:
        count_files = sum(1 for _ in path.rglob("*") if _.is_file())
        print(f"    {name}/ … {note}（{count_files}ファイル）")
    if not ask_yes_no("これらも消しますか", False):
        print("  残しました。")
        return
    for name, path, _note in present:
        failures = _remove_tree(path)
        if failures:
            print(f"  {name}/ の一部を消せませんでした（{len(failures)}件）。")
            for failure in failures[:3]:
                print(f"    {failure}")
            print("    ビューアをブラウザで開いたままだと、そのファイルは消せません。")
            print("    タブを閉じてから、もう一度この項目を選んでください。")
            print("    消せなくても学習や解答には影響しません。")
        else:
            print(f"  {name}/ を消しました。")
    data_dir = VIEWER_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "index.js").write_text("window.HEXA_INDEX = [];\n", encoding="utf-8")
    (data_dir / "index.json").write_text("[]", encoding="utf-8")


def _require_server_binary():
    from hexa.localserver import find_server_binary

    binary = find_server_binary(SERVER_DIR)
    if binary is None:
        print(f"  公式サーバーのバイナリが {SERVER_DIR} にありません。")
        print("  https://www.procon.gr.jp/news/2026/08/20/117 から")
        print("  「簡易サーバー.zip」を落として、中身を server/ に置いてください。")
        return None
    try:
        binary.chmod(binary.stat().st_mode | 0o111)
    except OSError:
        pass
    return binary


def menu_verify_server() -> None:
    print()
    print("== 公式サーバーと照合する ==")
    print("  公式の簡易版サーバーを起動してAI全員で実際に1試合戦い、")
    print("  こちらのシミュレータの予測とサーバーの結果が一致するか確かめます。")
    binary = _require_server_binary()
    if binary is None:
        return

    from hexa.localserver import start_server
    from hexa.verify import verify_against_server

    try:
        brains, _ = get_population(create_if_missing=False)
    except FileNotFoundError:
        print("  学習済みモデルがないので、未学習のAIで照合します。")
        brains = new_population()

    match_count = ask_int("何試合ぶん照合しますか", 3, 1, 20)
    day_seconds = ask_int("1日の回答時間（秒）", 6, 3, 60)
    stress = ask_yes_no("道路が混みやすい設定にしますか（渋滞の検証用）", True)

    import dataclasses

    tokens = [f"token-p{index}" for index in range(len(brains))]
    total_state = 0
    total_traffic = 0
    all_mismatches: list[str] = []

    for index in range(match_count):
        base = GeneratorSettings(
            height=10, width=10, agent_count=5, spot_count=8, brand_count=4,
            day_count=5, players=len(brains), day_seconds=day_seconds,
            road_ratio=0.35 if stress else 0.22,
        )
        problem = generate_problem(20260910 + index * 977, base)
        if stress:
            problem = dataclasses.replace(problem, busy_threshold=1, jammed_threshold=2)
        server = start_server(
            binary, problem, tokens, BASE_DIR / "tmp_server",
            names=[brain.name for brain in brains],
            kind_deadline="4s", match_start_delay="3s",
        )
        try:
            print(f"  {index + 1}試合目を進めています…")
            report = verify_against_server(
                server.base_url, tokens, brains, problem, log=lambda message: None
            )
        finally:
            server.stop()
        total_state += report.state_checks
        total_traffic += report.traffic_checks
        all_mismatches.extend(report.mismatches)
        print("    " + " | ".join(f"{name} {score}" for name, score in report.scores))

    print()
    print(f"  位置と燃料の照合 {total_state} 回、道路状態の照合 {total_traffic} 回")
    if all_mismatches:
        print(f"  ずれが {len(all_mismatches)} 件見つかりました:")
        for message in all_mismatches[:20]:
            print(f"    - {message}")
        print("  これはこのプログラムのバグです。docs/RULES.md の出典に当たって直してください。")
    else:
        print("  ずれはありませんでした。シミュレータは公式サーバーと一致しています。")


def menu_serve() -> None:
    print()
    print("== 練習用サーバーを立てる（別のPCから接続する） ==")
    print("  公式の簡易版サーバーを、同じLANの別PCから見える形で起動します。")
    print("  このPCでサーバー、別のPCでメニュー8、という練習ができます。")
    binary = _require_server_binary()
    if binary is None:
        return

    import dataclasses

    from hexa.generator import HONSEN_ROUNDS, honsen_settings
    from hexa.localserver import local_ip, start_server

    round_name = ask_choice("どの試合の条件にしますか", list(HONSEN_ROUNDS), "1回戦")
    team_count = ask_int("参加チーム数", 10, 2, 10)
    day_count = ask_int("日数", 5, 4, 10)
    day_seconds = ask_int("1日の回答時間（秒）", 60, 5, 300)
    kind_seconds = HONSEN_ROUNDS[round_name]["kind_seconds"]
    wait_seconds = ask_int("起動してから試合開始までの秒数", 90, 10, 1800)
    seed = ask_int("乱数の種", 2026, 0, 2_000_000_000)

    settings = honsen_settings(
        round_name, __import__("random").Random(seed),
        players=team_count, day_count=day_count,
    )
    settings = dataclasses.replace(settings, day_seconds=day_seconds)
    problem = generate_problem(seed, settings)
    tokens = [f"team{index + 1}" for index in range(team_count)]

    print()
    print(f"  {round_name}の条件: {problem.height}x{problem.width} / "
          f"エージェント{problem.agent_count}体 / スポット{len(problem.spots)} / "
          f"系列{len({spot.brand for spot in problem.spots})} / {problem.day_count}日")
    print(f"  タイプ選択 {kind_seconds}秒 / 行動回答 {day_seconds}秒")
    print()

    server = start_server(
        binary, problem, tokens, BASE_DIR / "tmp_serve",
        names=[f"チーム{index + 1}" for index in range(team_count)],
        kind_deadline=f"{wait_seconds}s",
        match_start_delay=f"{kind_seconds}s",
        host="0.0.0.0",
    )
    address = server.base_url
    port = address.rsplit(":", 1)[1]
    print("  サーバーを起動しました。")
    print(f"    このPCから  : http://127.0.0.1:{port}")
    print()
    print("  別のPCからは、下のどれかで繋がります（上から試してください）:")
    from hexa.localserver import local_ips

    candidates = [found for found in local_ips() if not found.startswith("127.")]
    if candidates:
        for found in candidates:
            note = ""
            if found.startswith("169.254."):
                note = "  ← DHCPなし。相手も169.254なら繋がります"
            print(f"    http://{found}:{port}{note}")
    else:
        print("    （LANのアドレスが見つかりません。下の確認をしてください）")
    print()
    print("  **繋がらないときの確認（Windows）**")
    print("   1. ネットワークが「識別されていないネットワーク」と出る場合、")
    print("      DHCPがありません。両方のPCに固定IPを入れてください。")
    print("      例: このPC 192.168.50.1 / 相手 192.168.50.2")
    print("          サブネットマスクはどちらも 255.255.255.0")
    print("          （設定 → ネットワーク → アダプターのオプション →")
    print("           IPv4 → 次のIPアドレスを使う）")
    print("   2. ファイアウォールが受信を止めていることがあります。")
    print("      これはこの画面から開けられます（下で聞きます）。")
    print("   3. 相手のPCから届くかは、相手で次を実行すると分かります:")
    for found in (candidates or ["<このPCのIP>"]):
        print(f"        ping {found}")
        print(f"        curl http://{found}:{port}/")
        break
    print()
    opened = False
    if ask_yes_no("ファイアウォールでこのポートの受信を許可しますか", True):
        done, message = open_firewall(int(port))
        print(f"  {'成功' if done else '失敗'}: {message}")
        if done:
            opened = True
        else:
            print("  許可の画面で「いいえ」を押した場合は、もう一度14番を")
            print("  選び直してください。")
    print()
    print("  別のPCで「8. 本番モードで戦う」を選び、上のURLと下のトークンを入れてください。")
    for index, token in enumerate(tokens):
        print(f"    チーム{index + 1}: {token}")
    print()
    print(f"  タイプ選択の受付は起動から{wait_seconds}秒後に締め切られます。")
    print("  それまでに全チームが接続してください。")
    print("  終わるには Ctrl+C を押してください。")
    try:
        while server.process.poll() is None:
            time.sleep(1.0)
        print("  サーバーが終了しました。")
        print(server.read_log(2000))
    except KeyboardInterrupt:
        print()
        print("  サーバーを止めます。")
    finally:
        server.stop()
        if opened and ask_yes_no("ファイアウォールの許可を元に戻しますか", True):
            done, message = close_firewall(int(port))
            print(f"  {'成功' if done else '失敗'}: {message}")




def _firewall_rule_name(port: int) -> str:
    return f"ヘキサうどん {port}"


def open_firewall(port: int) -> tuple[bool, str]:
    """Windowsのファイアウォールで、このポートの受信を許可する。

    **なぜプログラム側でやるか**

    LANでPC同士を繋ぐとき、繋がらない原因のほとんどは固定IPと
    ファイアウォールです。固定IPはWindowsの設定でしか変えられ
    ませんが、ファイアウォールはここから開けられます。

    管理者の権限が要るので、`Start-Process -Verb RunAs` で昇格を
    頼みます。画面に「はい／いいえ」の確認（UAC）が出るので、
    **対話だけで済み、コマンドを打つ必要はありません**。

    Windows以外では何もしません（macOSやLinuxでは、ふつう
    LAN内の受信は止まっていません）。
    """

    import platform
    import subprocess

    if platform.system() != "Windows":
        return (True, "Windowsではないので、何もしていません。")
    name = _firewall_rule_name(port)
    inner = (
        f"netsh advfirewall firewall delete rule name=\\\"{name}\\\" "
        f"> $null 2>&1; "
        f"netsh advfirewall firewall add rule "
        f"name=\\\"{name}\\\" dir=in action=allow protocol=TCP "
        f"localport={port}"
    )
    command = (
        "Start-Process -FilePath powershell -Verb RunAs -Wait "
        f"-ArgumentList '-NoProfile','-Command','{inner}'"
    )
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        return (False, "powershell が見つかりません。")
    except subprocess.TimeoutExpired:
        return (False, "確認の画面が返ってきませんでした。")
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else "理由は分かりません"
        return (False, f"許可できませんでした（{reason}）。")
    return (True, f"ポート{port}の受信を許可しました（規則名「{name}」）。")


def close_firewall(port: int) -> tuple[bool, str]:
    """`open_firewall` で入れた規則を消す。試合が終わったあとに使う。"""

    import platform
    import subprocess

    if platform.system() != "Windows":
        return (True, "Windowsではないので、何もしていません。")
    name = _firewall_rule_name(port)
    inner = (
        f"netsh advfirewall firewall delete rule "
        f"name=\\\"{name}\\\""
    )
    command = (
        "Start-Process -FilePath powershell -Verb RunAs -Wait "
        f"-ArgumentList '-NoProfile','-Command','{inner}'"
    )
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return (False, "消せませんでした。")
    if done.returncode != 0:
        return (False, "消せませんでした。")
    return (True, f"規則「{name}」を消しました。")


def _explain_connection_failure(base_url: str) -> None:
    """繋がらないときに、**どこを見ればよいか**を順に出す。

    LANでPC同士を繋ぐときの失敗は、ほぼこの4つに収まります。
    Windowsが「識別されていないネットワーク」と出すのは1番です。
    """

    import socket
    from urllib.parse import urlparse

    parsed = urlparse(base_url if "//" in base_url else "http://" + base_url)
    host = parsed.hostname or ""
    port = parsed.port or 80
    print()
    print("  == 繋がらないときの確認 ==")
    if host in ("127.0.0.1", "localhost", "::1"):
        print("   このURLは**自分自身**を指しています。別のPCのサーバーに")
        print("   繋ぐなら、相手のPCのLANのアドレスを入れてください。")
        print("   相手のPCの「14. 練習用サーバーを立てる」が表示します。")
        return
    # 1) 名前が引けるか
    try:
        socket.gethostbyname(host)
    except OSError:
        print(f"   1. 「{host}」という名前が引けません。")
        print("      IPアドレスを直に入れてください（例 http://192.168.50.1:8080）。")
    # 2) TCPで届くか
    reach = False
    try:
        with socket.create_connection((host, port), timeout=1.5):
            reach = True
    except OSError as error:
        print(f"   2. {host}:{port} に届きません（{type(error).__name__}）。")
    if reach:
        print(f"   2. {host}:{port} には届いています。")
        print("      → URLの綴り（http:// と ポート番号）とトークンを見直してください。")
        return
    print()
    print("   この順に試してください（Windows）。")
    print("   A. 「識別されていないネットワーク」と出ていませんか。")
    print("      DHCPが無い状態です。両方のPCに固定IPを入れます。")
    print("        サーバー側 192.168.50.1 / こちら 192.168.50.2")
    print("        サブネットマスク どちらも 255.255.255.0")
    print("        （設定 → ネットワークとインターネット → アダプターの")
    print("         オプション → IPv4 → 次のIPアドレスを使う）")
    print("      ※ 無線と有線の両方が繋がっていると、どちらを使うか")
    print("         迷います。試合中は無線を切るのが確実です。")
    print(f"   B. サーバー側のファイアウォールが受信を止めていませんか。")
    print("      サーバー側の管理者PowerShellで:")
    print(f'        New-NetFirewallRule -DisplayName "procon{port}" '
          f'-Direction Inbound -Protocol TCP -LocalPort {port} -Action Allow')
    print(f"   C. 相手まで届くかを確かめます（このPCで）:")
    print(f"        ping {host}")
    print("      ping が通ってポートだけ駄目なら B です。")
    print("      ping も通らなければ A か、ケーブル・ハブの問題です。")
    print("   D. サーバー側が待ち受けているかを、サーバー側のPCで:")
    print(f"        netstat -ano | findstr :{port}")
    print("      0.0.0.0:ポート と出ていれば、外からの受信を待っています。")
    print("      127.0.0.1:ポート しか無ければ、14番で立て直してください。")


def menu_connection_test() -> None:
    print()
    print("== 通信だけ確認する ==")
    print("  試合が始まる前でも使えます。URLとトークンが正しいか、")
    print("  サーバーに届くかだけを確かめます。")
    from hexa.client import ProconClient

    base_url = ask("サーバーのURL", "http://localhost:8080")
    token = ask("トークン", "")
    if not token:
        print("  トークンが空です。")
        return
    client = ProconClient(base_url, token)
    ok, message = client.check_connection()
    print()
    print(f"  {'成功' if ok else '失敗'}: {message}")
    if not ok:
        _explain_connection_failure(base_url)
        return
    try:
        problem, _raw = client.get_setting()
        print(f"  試合設定: {problem.height}x{problem.width} / "
              f"エージェント{problem.agent_count}体 / {problem.day_count}日 / "
              f"players={problem.players}")
    except Exception:
        print("  まだ試合設定は配られていません（試合開始前なら正常です）。")


def menu_online() -> None:
    print()
    print("== 本番モードで戦う ==")
    from hexa.client import ProconApiError, ProconClient
    from hexa.online import play_match
    from hexa.production import play_production_match

    try:
        brains, _ = get_population(create_if_missing=False)
    except FileNotFoundError:
        print("  学習済みモデルがありません。先に「2. AIを学習させる」を実行してください。")
        return

    total = sum(brain.trained_episodes for brain in brains)
    print(f"  学習済みモデルを使います（4体合計 {total} エピソード）。")
    print()
    print("  本番モードは毎日、次のことをします。")
    print("    ・4体ぶんの計画と、学習に頼らない貪欲な計画を作る")
    print("    ・全部を公式と同じシミュレータにかけて成果を測る")
    print("    ・公式の優先順位で一番良いものを提出する")
    print("    ・回答時間が余っていれば、探索を入れた計画をさらに試して出し直す")
    print()

    base_url = ask("サーバーのURL", "http://localhost:8080")
    token = ask("トークン", "")
    if not token:
        print("  トークンが空です。")
        return

    mode = ask_choice("どちらで戦いますか", ["production", "single"], "production")
    if ask_yes_no("リアルタイム監視をブラウザで開きますか", True):
        _server, viewer_url = start_viewer_server()
        if viewer_url:
            print(f"  監視: {viewer_url}")
    client = ProconClient(base_url, token)

    # **繋がるかを先に見ます。**
    #
    # 試合が始まってから「繋がりません」だけ出ても、どこが悪いのか
    # 分かりません。LANでPC同士を繋ぐときは、ほぼ固定IPか
    # ファイアウォールのどちらかです。先に切り分けを出します。
    ok, message = client.check_connection()
    print()
    print(f"  通信: {'成功' if ok else '失敗'}（{message}）")
    if not ok:
        _explain_connection_failure(base_url)
        print()
        if not ask_yes_no("それでもこのまま進みますか", False):
            return

    try:
        if mode == "production":
            report = play_production_match(client, brains)
        else:
            print("  どのAIで戦いますか")
            for index, brain in enumerate(brains):
                print(f"    {index + 1}. {brain.name}（累計{brain.trained_episodes}エピソード）")
            choice = ask_int("番号", 1, 1, len(brains))
            report = play_match(client, brains[choice - 1])
    except ProconApiError as error:
        print(f"  サーバーがエラーを返しました: {error}")
        return
    except (TimeoutError, OSError) as error:
        print(f"  通信に失敗しました: {error}")
        return
    except KeyboardInterrupt:
        print("\n  中断しました。")
        return

    print()
    print(f"  最終成績: {report.score}")
    if report.mismatches:
        print("  サーバーの結果とこちらの予測にずれがありました:")
        for message in report.mismatches:
            print(f"    - {message}")
        print("  これは実装の食い違いです。docs/RULES.md の出典に当たって直してください。")
    else:
        print("  こちらの予測はサーバーの結果と全日一致しました。")


def menu_production_rehearsal() -> None:
    print()
    print("== 本番モードの通し練習 ==")
    print("  公式の簡易版サーバーを起動して、本番モードで実際に1試合戦います。")
    print("  比較のため、他の3枠は単体のAIが同じ試合に参加します。")
    binary = _require_server_binary()
    if binary is None:
        return

    import threading

    from hexa.client import ProconClient
    from hexa.localserver import start_server
    from hexa.online import play_match
    from hexa.production import play_production_match

    try:
        brains, _ = get_population(create_if_missing=False)
    except FileNotFoundError:
        print("  学習済みモデルがありません。先に学習してください。")
        return

    players = ask_int("参加チーム数", 4, 2, 10)
    day_seconds = ask_int("1日の回答時間（秒）", 10, 5, 120)
    settings = settings_from_input(players)
    seed = ask_int("乱数の種", 4242, 0, 2_000_000_000)
    problem = generate_problem(seed, dataclasses_replace_day_seconds(settings, day_seconds))

    tokens = [f"token-p{index}" for index in range(players)]
    labels = ["本番モード"] + [
        brains[index % len(brains)].name + "単体" for index in range(players - 1)
    ]
    server = start_server(
        binary, problem, tokens, BASE_DIR / "tmp_rehearsal",
        names=labels, kind_deadline="25s", match_start_delay="5s",
    )
    results: dict[int, object] = {}

    def run(index: int) -> None:
        client = ProconClient(server.base_url, tokens[index])
        try:
            if index == 0:
                results[index] = play_production_match(client, brains, log=lambda m: None)
            else:
                results[index] = play_match(
                    client, brains[index % len(brains)], log=lambda m: None
                )
        except Exception as error:  # 1チームが落ちても他の結果は見たい
            results[index] = error

    threads = [threading.Thread(target=run, args=(index,)) for index in range(players)]
    print("  試合中です。しばらくお待ちください…")
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=900)
    finally:
        server.stop()

    print()
    print("  結果")
    for index in range(players):
        result = results.get(index)
        if isinstance(result, Exception):
            print(f"    {labels[index]}: 失敗しました（{result}）")
        elif result is None:
            print(f"    {labels[index]}: 応答がありませんでした")
        else:
            note = "ずれなし" if not result.mismatches else f"ずれ{len(result.mismatches)}件"
            print(f"    {labels[index]}: {result.score}  {note}")

    first = results.get(0)
    if first is not None and not isinstance(first, Exception):
        print()
        print("  本番モードが毎日どの計画を選んだか")
        for day in first.days:
            print(
                f"    {day.day + 1}日目: 「{day.chosen}」"
                f"（候補{day.considered}件 / 出し直し{day.resubmissions}回 /"
                f" 新しい系列{day.day_score[0]} 日別{day.day_score[1]} 玉{day.day_score[2]}）"
            )


def dataclasses_replace_day_seconds(settings, day_seconds: int):
    import dataclasses

    return dataclasses.replace(settings, day_seconds=day_seconds)


def menu_auto_train() -> None:
    print()
    print("== おまかせ学習（放置用） ==")
    print("  一度始めれば、あとは見ていなくても勝手に育ちます。")
    print()
    print("  一定エピソードごとに、マップの5つの型（街・山岳・群島・平原・混在）で")
    print("  成績を測り、次のことを自動でやります。")
    print("    ・苦手な型を次のラウンドで多めに出す")
    print("    ・伸びが止まったらAIを一段階大きくする（判断は変えずに表現力だけ増やす）")
    print("    ・一番良かった時点を必ず残し、最後にそこへ戻す")
    print("    ・途中経過を models/" + AUTO_LOG_NAME + " に書き出す")
    print()
    print("  Ctrl+C でいつ止めても、そこまでの学習は残ります。")
    print()

    brains, resumed = get_population()
    if resumed:
        total = sum(brain.trained_episodes for brain in brains)
        print(f"  続きから学習します（{len(brains)}体 / 大きさ {current_capacity(brains[0])}"
              f" / 累計 {total} エピソード）。")
    else:
        print("  新しいAIから始めます。先にメニュー10で体数を決めておくこともできます。")

    episodes = ask_int("回すエピソード数", 50000, 100, 100_000_000)
    round_episodes = ask_int("何エピソードごとに様子を見るか", 2000, 50, 100000)
    allow_growth = ask_yes_no("伸びが止まったらAIを自動で大きくしますか", True)

    print()
    print("  様子を見るとき、どの試合で測るかを決めます。")
    print("  ここで測った成績だけを見て「伸びたか」「どの型が苦手か」")
    print("  「AIを大きくするか」を判断します。")
    print()
    print("  本選の3条件をそのまま使います（本選マップに関するお知らせ）。")
    print("    1回戦  16x16 / 4体 / 16店 / 系列10-14")
    print("    準決勝 24x24 / 5体 / 24店 / 系列14-20")
    print("    決勝   32x32 / 7体 / 32店 / 系列20-28")
    print()
    for count in (3, 6, 9):
        names = list(dict.fromkeys(validation_sizes(count)))
        print(f"    {count} … {' / '.join(names)} × 6型 = {count * 6}問")
    print()
    print("  3未満にすると決勝の条件が一度も測られません。")
    print("  6にすると同じ条件を2問ずつ測るので、ばらつきに強くなります。")
    print("  そのぶん検証に時間がかかります（6なら10分前後）。")
    per_archetype = ask_int("検証の細かさ（型ごとの問題数）", 3, 1, 12)
    if per_archetype < 3:
        print("  ※ 3未満だと決勝の条件（32x32・7体）が測られません。")
        if not ask_yes_no("  それでよいですか", False):
            per_archetype = ask_int("検証の細かさ（型ごとの問題数）", 3, 3, 12)

    default_match = min(len(brains), 9)
    match_size = default_match
    if len(brains) > 2:
        match_size = ask_int("1試合のチーム数", default_match, 2, len(brains))

    print()
    print("  経験に使うメモリは**親プロセス1つぶん**です。学習用のプロセスは")
    print("  別に動くので、その分の余裕を残してください（目安: 搭載メモリ ÷ 4）。")
    memory_gb = ask_int("経験に使ってよいメモリ（GB）",
                        max(1, machine_memory_gb() // 4), 1, 1024)
    replay_limit = replay_limit_for_budget(
        memory_gb * 1024 ** 3,
        len(brains),
        touring_feature_count(brains[0].touring_cross),
        supply_feature_count(brains[0].supply_cross),
    )

    import multiprocessing

    workers = ask_int("使うプロセス数", machine_cores(), 1,
                      multiprocessing.cpu_count() * 4)
    estimate = memory_gb + workers * 0.15
    print(f"  おおよその使用メモリ: 経験 {memory_gb}GB + プロセス {workers}個で"
          f" 約 {estimate:.1f}GB")

    problem_dir = None
    if PROBLEM_DIR.exists() and any(PROBLEM_DIR.iterdir()):
        folders = sorted(p.name for p in PROBLEM_DIR.iterdir() if p.is_dir())
        if folders and ask_yes_no("作り置きの問題も混ぜますか", False):
            print("  " + " / ".join(folders))
            chosen = ask("フォルダ名", folders[0])
            if (PROBLEM_DIR / chosen).is_dir():
                problem_dir = str(PROBLEM_DIR / chosen)

    settings = TrainingSettings(
        episodes=episodes,
        validation_every=0,
        problem_dir=problem_dir,
        match_size=match_size,
        replay_limit=replay_limit,
    )
    auto_train(
        brains, settings, MODELS_DIR, MODEL_NAME, workers,
        total_episodes=episodes, round_episodes=round_episodes,
        validation_per_archetype=per_archetype,
        allow_growth=allow_growth,
    )


# このビルドの版。pull したものが最新かどうかを目で確かめるためのものです。
#
# 番号の意味（大きいほど新しい）
#   1桁目: 作りを大きく変えたとき
#   2桁目: 仕組みを足したり、成績が変わる修正をしたとき
#   3桁目: 表示や記録まわりの小さな修正
#
# 変更の履歴は docs/FINAL_AUDIT.md に書いてあります。
BUILD_VERSION = "1.53.3"
BUILD_NOTE = ("**種類の提出に失敗しても試合が止まりません**"
              "（2026-10-11の準決勝で `AccessTimeError: agent kind "
              "submission is closed` で試合そのものが落ちました）。"
              "種類の提出（90秒）は行動の提出（1日60秒）とは別の枠なので、"
              "間に合わなくても試合は戦えます"
              " / さらに**1日目にサーバーが返す本当の種類に合わせ直します**。"
              "状態には1体ごとの `kind` が入っているのでそれが正です"
              " / **子プロセスの起動時間も種別選択の持ち時間から引きます。**"
              "24x24の持ち時間90秒に対しこちらは63秒を使いますが、"
              "起動（Windowsで11個・約10秒）がその63秒に入っていませんでした"
              " / **補給車を「追いかける」から「待ち伏せる」に変えました。**"
              "その車の順路の先で先回りして待ちます。15試合で"
              "順位合計 195→**174**、玉 3836→**3975**、日別も+3。"
              "手数2000では 163→**151**"
              " / **17通り測って、勝ったのは2つだけ**でした"
              "（待ち伏せと `design_cover` のバグ修正）。負けた15通りは"
              "全部**既定オフで同梱**してあります"
              "（近づく手・夜合流・逆回りループ・在庫の重み・"
              "補給の見込み0.25杯・出発点の多様性・設計を種に入れる・"
              "予約の記帳・救援を先に・余裕0・補給車2台・"
              "**時刻つきの待ち合わせ**・**届く台数の重み**）"
              " / **負け方に共通点があります。** 構造を変えた4つ"
              "（ハブ／夜合流／逆回りループ／時刻つき待ち合わせ）は、"
              "どれも補給車を1か所・1つの約束に縛る形でした。磨きが"
              "順路を変え続けるので、縛るとすぐ古くなります。勝った"
              "待ち伏せだけが**毎歩その場で決め直す**形でした"
              " / **物理的な上限はまだ15〜30%残っています**"
              "（2位まで中央値あと12玉）。頭打ちなのはつまみで、"
              "上限ではありません。次の筋は `docs/POLISH.md` の末尾に"
              " / 準決勝の当日の手順書 `docs/SEMIFINAL.md`、"
              "他高専の解説集34ページの読み取り `docs/RIVALS.md` 入り"
              " / テスト619件")


def build_line() -> str:
    """起動時に出す1行。models の学習量もあれば添えます。"""

    extra = ""
    try:
        import json

        path = BASE_DIR / "models" / "population.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            brains = data.get("brains", [])
            if brains:
                episodes = brains[0].get("trainedEpisodes", 0)
                width = len(brains[0].get("touringHeads", [[]])[0])
                size = {40: "simple", 211: "cross", 781: "large"}.get(width, f"{width}個")
                extra = f" / AI {len(brains)}体 {size} 通算{episodes}エピソード"
    except Exception:
        extra = ""
    return f"版 {BUILD_VERSION}（{BUILD_NOTE}）{extra}"


ENV_NAME = ".env"


def load_env() -> dict[str, str]:
    """.env を読む。無ければ空。

    当日の入力ミスを減らすため、トークンをここに置いておけます。
    中身は `名前=値` を1行ずつ並べるだけです。
    """

    path = BASE_DIR / ENV_NAME
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _sep, value = line.partition("=")
            values[name.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


def save_env(values: dict[str, str]) -> None:
    """.env に書く。既にある項目は書き換え、無い項目は足す。"""

    path = BASE_DIR / ENV_NAME
    existing: list[str] = []
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            existing = []
    done: set[str] = set()
    out: list[str] = []
    for line in existing:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            name = stripped.split("=", 1)[0].strip()
            if name in values:
                out.append(f"{name}={values[name]}")
                done.add(name)
                continue
        out.append(line)
    for name, value in values.items():
        if name not in done:
            out.append(f"{name}={value}")
    try:
        path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    except OSError as error:
        print(f"  .env に書けませんでした（{error}）")


def menu_token() -> None:
    print()
    print("== トークンを登録する ==")
    print("  .env に保存します。本番当日は入力しなくて済みます。")
    print("  このファイルは .gitignore に入れてあるので、push されません。")
    print()
    env = load_env()
    current = env.get("PROCON_TOKEN", "")
    if current:
        print(f"  いま登録されているトークン: {current[:8]}…（{len(current)}文字）")
    token = ask("トークン", current)
    if not token:
        print("  空なので、何もしませんでした。")
        return
    save_env({"PROCON_TOKEN": token})
    print(f"  保存しました: {BASE_DIR / ENV_NAME}")


def menu_tournament() -> None:
    print()
    print("== 本番（大会当日） ==")
    print("  URLを入れるだけで始まります。トークンは .env から読みます。")
    print("  監視ビューアは自動で開きます。")
    print()
    from hexa.client import ProconClient
    from hexa.production import play_production_match

    try:
        brains, _ = get_population(create_if_missing=False)
    except FileNotFoundError:
        print("  学習済みモデルがありません。models/population.json を置いてください。")
        return
    print(f"  AI {len(brains)}体（合計 {sum(b.trained_episodes for b in brains)} エピソード）")

    env = load_env()
    token = env.get("PROCON_TOKEN", "")
    if token:
        print(f"  トークン: {token[:8]}…（.env から読みました）")
    else:
        print("  .env にトークンがありません。")
        token = ask("トークン", "")
        if not token:
            print("  トークンが空です。")
            return
        if ask_yes_no("  このトークンを .env に保存しますか", True):
            save_env({"PROCON_TOKEN": token})

    base_url = ask("サーバーのURL", env.get("PROCON_URL", ""))
    if not base_url:
        print("  URLが空です。")
        return
    base_url = base_url.rstrip("/")
    if ask_yes_no("  このURLを .env に保存しますか", False):
        save_env({"PROCON_URL": base_url})

    client = ProconClient(base_url, token)
    ok, message = client.check_connection()
    print(f"  通信確認: {message}")
    if not ok:
        print("  URL・トークン・ネットワークを確認してください。")
        return

    _server, viewer_url = start_viewer_server()
    if viewer_url:
        print(f"  監視: {viewer_url}")

    print()
    print("  試合の開始を待っています。止めるには Ctrl+C を押してください。")
    try:
        report = play_production_match(client, brains)
    except KeyboardInterrupt:
        print("  中断しました。")
        return
    except Exception as error:
        print(f"  試合が最後まで進みませんでした: {type(error).__name__}: {error}")
        return

    print()
    print(f"  最終成績: {report.score}")
    if report.mismatches:
        print(f"  予測とのズレ {len(report.mismatches)}件 ← 実装の食い違いです。")
    else:
        print("  予測とサーバーの結果は一致しました。")


def start_viewer_server(open_browser: bool = True):
    """監視ビューアを配る小さなサーバーを立てる。

    失敗しても None を返すだけで、呼び出し元は先へ進めます。
    **本番の試合を止めないことが最優先**だからです。
    """

    import threading
    import webbrowser
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    directory = BASE_DIR / "viewer"

    # partial はクラスではないので継承できません。
    # 先に継承してから partial を作るのが正しい順序です。
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def handle_one_request(self):
            # ブラウザが読み込みを途中でやめると、Windows では
            # ConnectionAbortedError が飛んで標準エラーが例外で埋まります。
            # 監視は本番の点数に関係しないので、黙って捨てます。
            try:
                super().handle_one_request()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                self.close_connection = True

        def handle(self):
            try:
                super().handle()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass

    handler = partial(Quiet, directory=str(directory))
    class QuietServer(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # 切断は毎回起きるので、画面に出しません。
            pass

    try:
        server = QuietServer(("127.0.0.1", 0), handler)
    except OSError:
        return (None, "")
    url = f"http://127.0.0.1:{server.server_port}/live.html"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return (server, url)


def menu_live_viewer() -> None:
    print()
    print("== リアルタイム監視を開く ==")
    print("  メニュー8や15で戦っているあいだ、いまの状況を見られます。")
    print("  盤面・探索の様子・提出した行動・日ごとの詳細が1秒ごとに更新されます。")
    print()
    print("  **本番の処理とは完全に別です。** 本番側は軽いファイルを1つ書くだけで")
    print("  （実測0.4ミリ秒）、このビューアを開いても閉じても試合に影響しません。")
    print()
    server, url = start_viewer_server()
    if server is None:
        print("  ビューアを開けませんでした。")
        return
    print(f"  開きました: {url}")
    print("  終わるには Enter を押してください。")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    server.shutdown()
    print("  閉じました。")


def menu_arena() -> None:
    print()
    print("== 練習場で回し続ける ==")
    print("  非公式練習場（procon37arena.online）は15分ごとに試合が開かれます。")
    print("  プロトコルは公式とまったく同じなので、接続先を変えるだけで参加できます。")
    print()
    print("  開始3分前からトークン付きで接続すると参加表明になります。")
    print("  このプログラムは接続できるまで待ち続けるので、いつ起動しても構いません。")
    print("  試合が終わると自動で次の受付につなぎ直します。放置で回せます。")
    print()
    from hexa.client import ProconClient
    from hexa.production import play_matches_forever

    try:
        brains, _ = get_population(create_if_missing=False)
    except FileNotFoundError:
        print("  学習済みモデルがありません。先に学習してください。")
        return
    print(f"  AI {len(brains)}体を使います（合計 "
          f"{sum(b.trained_episodes for b in brains)} エピソード）。")

    base_url = ask("練習場のURL", "https://procon37arena.online")
    token = ask("トークン", "")
    if not token:
        print("  トークンが空です。")
        return
    count = ask_int("何試合やりますか（0なら止めるまで）", 0, 0, 1000)

    client = ProconClient(base_url, token)
    ok, message = client.check_connection()
    print(f"  通信確認: {message}")
    if not ok:
        print("  URL・トークン・ネットワークを確認してください。")
        return

    if ask_yes_no("  リアルタイム監視をブラウザで開きますか", True):
        _server, viewer_url = start_viewer_server()
        if viewer_url:
            print(f"  監視: {viewer_url}")

    print()
    print("  開始します。止めるには Ctrl+C を押してください。")
    reports = play_matches_forever(client, brains, max_matches=count)

    print()
    print(f"  {len(reports)}試合を終えました。")
    if reports:
        best = max(reports, key=lambda r: r.score.key)
        print(f"  一番良かった成績: {best.score}")
        mismatches = sum(len(r.mismatches) for r in reports)
        if mismatches:
            print(f"  予測とのズレ 合計{mismatches}件 ← 実装の食い違いです。要調査。")
        else:
            print("  予測とサーバーの結果は全試合で一致しました。")


def menu_multi_train() -> None:
    print()
    print("== 複数セット同時学習（高速・多数のAI） ==")
    print("  学習は「子プロセスが試合を解く → 親プロセスが重みを更新する」流れですが、")
    print("  親の更新は直列なので、64コアあっても親1つで詰まります。")
    print("  そこで**学習そのものを複数セットに分けて、親を複数立てます。**")
    print("  親が3つなら3倍、6つなら6倍の速さで進みます。")
    print()
    print("  おまけに、セットごとに違う個性の集団が育ちます。")
    print("  最後に「まとめる」を選べば、全部を1つの集団にできます。")
    print("  本番モードは全AIの案から一番良いものを選ぶので、案の幅が広がります。")
    print()
    import multiprocessing

    from hexa.multitrain import SetPlan, train_sets

    cores = machine_cores()
    memory = machine_memory_gb()
    print(f"  登録されているPC性能: コア{cores} / メモリ{memory}GB")
    print()
    print("  ※ 学習用PCと本番用PCで性能が違う場合、ここは**学習用PC**の値です。")
    print("    本番用PCが非力でも、モデルさえ運べば動きます（学習しないため）。")
    set_count = ask_int("セット数（＝同時に動く親プロセスの数）", 3, 1, 16)
    brains_per_set = ask_int("1セットのAI数", 10, 2, 32)
    workers = max(1, (cores - set_count) // max(1, set_count))
    workers_per_set = ask_int("1セットあたりの子プロセス数", workers, 1,
                              multiprocessing.cpu_count())
    memory_per_set = max(1, (memory - set_count * workers_per_set // 6) // max(1, set_count))
    memory_per_set = ask_int("1セットあたりのメモリ（GB）", max(1, memory_per_set), 1, 256)

    total_processes = set_count + set_count * workers_per_set
    total_memory = set_count * memory_per_set + total_processes * 0.15
    print()
    print(f"  合計 {set_count * brains_per_set} 体 / プロセス {total_processes} 個 / "
          f"メモリおよそ {total_memory:.1f}GB")
    if total_memory > memory * 0.9:
        print("  ※ メモリが足りません。セット数かメモリを減らしてください。")
        if not ask_yes_no("  それでも進めますか", False):
            return

    episodes = ask_int("1セットが回すエピソード数", 200000, 100, 100_000_000)
    round_episodes = ask_int("何エピソードごとに様子を見るか", 20000, 50, 1000000)
    per_archetype = ask_int("検証の細かさ（型ごとの問題数）", 3, 1, 12)
    match_size = ask_int("1試合のチーム数", min(10, brains_per_set), 2, brains_per_set)
    capacity = ask_choice("新しく作るAIの大きさ", ["simple", "cross", "large"], "simple")
    allow_growth = ask_yes_no("伸びが止まったら自動で大きくしますか", True)

    plan = SetPlan(
        set_count=set_count,
        brains_per_set=brains_per_set,
        episodes=episodes,
        workers_per_set=workers_per_set,
        memory_gb_per_set=memory_per_set,
        match_size=match_size,
        round_episodes=round_episodes,
        validation_per_archetype=per_archetype,
        capacity=capacity,
        allow_growth=allow_growth,
    )
    train_sets(MODELS_DIR, plan)


def menu_merge_sets() -> None:
    print()
    print("== セットを1つにまとめる ==")
    print("  複数セットで学習したAIを、1つの集団にまとめます。")
    print("  まとめたものが models/population.json になり、")
    print("  解答・本番モード・練習場で使われます。")
    print()
    print("  大きさ（simple / cross / large）が混ざっていても構いません。")
    print("  大きさはAI1体ごとに保存されているので、そのまま動きます。")
    print()
    from hexa.multitrain import merge_sets

    if (MODELS_DIR / MODEL_NAME).exists():
        print("  いまの models/population.json は上書きされます。")
        if not ask_yes_no("  続けますか", True):
            print("  やめました。")
            return
    merge_sets(MODELS_DIR)


def menu_grow() -> None:
    print()
    print("== AIを大きくする（学習を引き継ぐ） ==")
    try:
        brains, _ = get_population(create_if_missing=False)
    except FileNotFoundError:
        print("  まだAIがいません。先に「10. 学習をやり直す」で作ってください。")
        return

    current = "simple"
    for key, (touring_cross, _supply_cross) in CAPACITY_PRESETS.items():
        if tuple(touring_cross) == tuple(brains[0].touring_cross):
            current = key
    total = sum(brain.trained_episodes for brain in brains)
    print(f"  いまの大きさ: {current}（{len(brains)}体 / 累計 {total} エピソード）")
    print()
    print("  大きくしても、いまの判断はまったく変わりません。")
    print("  増える重みはすべて0から始まり、掛け算の項に0を掛けると必ず0なので、")
    print("  広げた直後の予測値は1つも動きません。学習した中身はそのまま残ります。")
    print("  そこから学習を続けると、掛け算の重みだけが0から動き出します。")
    print("  つまり「大きくしたせいで弱くなる」ことが原理的に起きません。")
    print()
    order = ["simple", "cross", "large"]
    choices = order[order.index(current) + 1:]
    if not choices:
        print("  すでに一番大きい large です。")
        return
    for key in choices:
        touring_cross, supply_cross = CAPACITY_PRESETS[key]
        print(
            f"    {key:7s} 巡回車{touring_feature_count(touring_cross):4d}個 / "
            f"補給車{supply_feature_count(supply_cross):3d}個の重み"
        )
    target = ask_choice("どこまで大きくしますか", choices, choices[0])
    if not ask_yes_no(f"  {current} → {target} に広げますか", True):
        print("  やめました。")
        return

    for brain in brains:
        brain.grow(target)
    save_population(brains, MODELS_DIR / MODEL_NAME)
    best = MODELS_DIR / "population_best.json"
    if best.exists():
        # 巻き戻し先も同じ大きさにしておかないと、あとで形が合わなくなる。
        best_brains = load_population(best)
        for brain in best_brains:
            brain.grow(target)
        save_population(best_brains, best)
    print(f"  {target} に広げました。経験（replay）はそのまま使えます。")
    print("  続けて「2. AIを学習させる」で学習してください。")



def _print_static_ip_steps(role: str = "サーバー") -> None:
    """固定IPの入れ方を画面に出す。

    **なぜ画面に出すか**

    Windowsが「識別されていないネットワーク」と出すのは、DHCPが無くて
    アドレスが決まらない状態です。LANケーブル直結やハブだけの接続だと
    必ずこうなります。このときは手でアドレスを決めるしかありません。
    設定の場所は Windows のバージョンで変わるので、画面の順路と
    コマンドの両方を出します。
    """

    first = "192.168.50.1" if role == "サーバー" else "192.168.50.2"
    other = "192.168.50.2" if role == "サーバー" else "192.168.50.1"
    print()
    print("  == 固定IPの入れ方（Windows） ==")
    print()
    print("  決めること（2台で違う番号にします）")
    print(f"    このPC（{role}側） IPアドレス    : {first}")
    print(f"    相手のPC                        : {other}")
    print("    サブネットマスク（両方同じ）     : 255.255.255.0")
    print("    デフォルトゲートウェイ           : 空のまま")
    print("    DNS                              : 空のまま")
    print("  ※ ゲートウェイを空にすると「インターネットなし」と出ますが、")
    print("     LAN内だけの対戦なら問題ありません。")
    print()
    print("  画面から入れる場合")
    print("    1. 設定 → ネットワークとインターネット")
    print("    2. 「ネットワークの詳細設定」→「ネットワーク アダプター")
    print("       オプションの詳細」（古い版では「アダプターのオプションを")
    print("       変更する」）")
    print("    3. 使っている有線アダプター（イーサネット）を右クリック →")
    print("       プロパティ")
    print("    4. 「インターネット プロトコル バージョン 4 (TCP/IPv4)」を")
    print("       選んで「プロパティ」")
    print("    5. 「次のIPアドレスを使う」を選び、上の数字を入れる")
    print("    6. OK → OK で閉じる")
    print()
    print("  コマンドで入れる場合（管理者のPowerShellで1行ずつ）")
    print("    まずアダプターの名前を調べます:")
    print("      Get-NetAdapter | Format-Table Name,Status,LinkSpeed")
    print('    名前が「イーサネット」なら:')
    print(f'      netsh interface ip set address name="イーサネット" '
          f'static {first} 255.255.255.0')
    print("    元に戻す（DHCPに戻す）とき:")
    print('      netsh interface ip set address name="イーサネット" dhcp')
    print()
    print("  **無線は切ってください。** 有線と無線の両方が繋がっていると、")
    print("  どちらを使うか迷って繋がらないことがあります。")
    print()
    print("  入れ終わったら、相手のPCから届くか確かめます:")
    print(f"      ping {other}")
    print("  これが通れば、あとはファイアウォールだけです。")


def menu_firewall() -> None:
    """LAN対戦の準備を、この画面だけで済ませる。

    PowerShellのコマンドを打たなくても、ファイアウォールの許可は
    ここから入れられます（管理者の確認に「はい」と答えるだけ）。
    固定IPはWindowsの設定でしか変えられないので、手順を出します。
    """

    import platform

    while True:
        print()
        print("== LAN対戦の準備 ==")
        print("  くわしい手順は docs/LAN.md にあります（上から順に")
        print("  たどれば終わるように書いてあります）。")
        print()
        print("  **ケーブルで詰まったら、同じWi-Fiにつなぐほうが簡単です。**")
        print("  2台を同じWi-Fi（スマホのテザリングでも可）につないで、")
        print("  サーバー役で14番 → 出たURLを選手役の20番に入れるだけ。")
        print("  設定は1つも要りません（docs/LAN.md の最初の節）。")
        print("  1. このPCのアドレスを見る")
        print("  2. 固定IPの入れ方を見る")
        print("  3. ファイアウォールで受信を許可する")
        print("  4. ファイアウォールの許可を消す")
        print("  5. 相手のPCに届くか確かめる")
        print("  6. **いまの状態をまとめて見る**（これが一番速い）")
        print("  7. 手順書（docs/LAN.md）をそのまま表示する")
        print("  0. 戻る")
        choice = ask("番号", "1")
        if choice == "0":
            return
        if choice == "1":
            from hexa.localserver import local_ips

            print()
            print("  このPCのアドレス（別PCから繋ぎやすい順）:")
            for found in local_ips():
                note = ""
                if found.startswith("127."):
                    note = "  ← 自分からしか見えません"
                elif found.startswith("169.254."):
                    note = "  ← DHCPなし。固定IPを入れてください（2番）"
                print(f"    {found}{note}")
            continue
        if choice == "2":
            role = ask_choice("このPCはどちらですか",
                              ["サーバー", "クライアント"], "サーバー")
            _print_static_ip_steps(role)
            continue
        if choice in ("3", "4"):
            if platform.system() != "Windows":
                print("  このPCはWindowsではないので、設定は要りません。")
                continue
            port = ask("ポート番号", "8080")
            try:
                number = int(port)
            except ValueError:
                print("  数字を入れてください。")
                continue
            if choice == "3":
                done, message = open_firewall(number)
            else:
                done, message = close_firewall(number)
            print()
            print(f"  {'成功' if done else '失敗'}: {message}")
            if done and choice == "3":
                print("  入っている規則は、対話で次を打てば確認できます:")
                print(f'    netsh advfirewall firewall show rule '
                      f'name="{_firewall_rule_name(number)}"')
            continue
        if choice == "5":
            target = ask("相手のURL（例 http://192.168.50.1:8080）", "")
            if not target:
                print("  URLが空です。")
                continue
            from hexa.client import ProconClient

            token = ask("トークン（分からなければ空のまま）", "")
            if token:
                ok, message = ProconClient(target, token).check_connection()
                print()
                print(f"  {'成功' if ok else '失敗'}: {message}")
                if not ok:
                    _explain_connection_failure(target)
            else:
                _explain_connection_failure(target)
            continue
        if choice == "6":
            _lan_overview()
            continue
        if choice == "7":
            path = BASE_DIR / "docs" / "LAN.md"
            if path.exists():
                print()
                print(path.read_text(encoding="utf-8"))
            else:
                print(f"  {path} が見つかりません。")
            continue
        print("  その番号はありません。")


def _lan_overview() -> None:
    """**いまの状態を1画面で出す。** LANで詰まったとき最初に見る所。

    LANでつながらない原因は、ほぼ次の4つに収まります。

      1. アドレスが別の網（DHCPが無いのに固定IPを入れていない）
      2. ファイアウォールが受信を止めている（とくに「パブリック」）
      3. サーバーが 127.0.0.1 でしか待っていない
      4. 無線と有線が両方生きていて、経路を取り違えている

    どれなのかは、この4つを並べて見れば分かります。
    """

    import platform
    import subprocess

    from hexa.localserver import local_ips

    print()
    print("  == いまの状態 ==")
    found = list(local_ips())
    others = [item for item in found if not item.startswith("127.")]
    print("  (1) このPCのアドレス")
    for item in found:
        note = ""
        if item.startswith("127."):
            note = "  ← 自分からしか見えません"
        elif item.startswith("169.254."):
            note = "  ← DHCPなし。相手も169.254なら通じます"
        print(f"        {item}{note}")
    if not others:
        print("        ※ LANのアドレスがありません。ケーブルを差して")
        print("           1分待ってから、もう一度6番を押してください。")
    if len(others) > 1:
        print("        ※ **2つ以上あります。** 無線と有線の両方が生きて")
        print("           いる可能性があります。試合中は無線を切るのが")
        print("           確実です（Windows + A → Wi-Fi を押す）。")
    if platform.system() != "Windows":
        print("  (2)(3)(4) このPCはWindowsではないので省きます。")
        return
    def run(command: str) -> str:
        try:
            done = subprocess.run(["powershell", "-NoProfile", "-Command",
                                   command], capture_output=True, text=True,
                                  timeout=20)
            return (done.stdout or "") + (done.stderr or "")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""
    print("  (2) ファイアウォールの許可（この道具が入れた規則）")
    text = run('Get-NetFirewallRule -DisplayName "procon*"'
               ' | Select-Object -ExpandProperty DisplayName')
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if names:
        for name in names:
            print(f"        {name} … 入っています")
    else:
        print("        ありません。サーバー役なら3番で入れてください。")
    print("  (3) いま待ち受けているポート")
    text = run("Get-NetTCPConnection -State Listen"
               " | Where-Object {$_.LocalPort -ge 8000"
               " -and $_.LocalPort -lt 9000}"
               " | ForEach-Object {\"$($_.LocalAddress):$($_.LocalPort)\"}")
    rows = sorted({line.strip() for line in text.splitlines() if line.strip()})
    if rows:
        for row in rows:
            note = ""
            if row.startswith("0.0.0.0") or row.startswith("::"):
                note = "  ← 外から受けられます（これが正解）"
            elif row.startswith("127.") :
                note = "  ← 自分専用。14番で立て直してください"
            print(f"        {row}{note}")
    else:
        print("        8000〜8999で待っているものはありません。")
        print("        サーバー役なら14番でサーバーを立ててください。")
    print("  (4) ネットワークの種類（パブリックだと受信が止まりがち）")
    text = run("Get-NetConnectionProfile"
               " | ForEach-Object {\"$($_.InterfaceAlias) ="
               " $($_.NetworkCategory)\"}")
    for line in text.splitlines():
        if line.strip():
            print(f"        {line.strip()}")
    print()
    print("  困ったら docs/LAN.md（7番で表示）を上から順に。")


MENU = [
    ("1", "問題を作る", menu_generate),
    ("2", "AIを学習させる", menu_train),
    ("2a", "おまかせ学習（放置用）", menu_auto_train),
    ("3", "問題をAI全員で解く", menu_solve),
    ("4", "ビューアを開く", menu_viewer),
    ("5", "学習の状況を見る", menu_status),
    ("6", "動作テストを実行する", menu_tests),
    ("7", "公式サーバーと照合する", menu_verify_server),
    ("8", "本番モードで戦う（サーバーに接続）", menu_online),
    ("9", "本番モードの通し練習（ローカル）", menu_production_rehearsal),
    ("10", "学習をやり直す", menu_reset),
    ("11", "AIを大きくする（学習を引き継ぐ）", menu_grow),
    ("12", "このPCの性能を登録する", menu_machine),
    ("13", "通信だけ確認する", menu_connection_test),
    ("14", "練習用サーバーを立てる（別PC用）", menu_serve),
    ("15", "練習場で回し続ける（非公式アリーナ）", menu_arena),
    ("16", "複数セット同時学習（高速・多数のAI）", menu_multi_train),
    ("17", "セットを1つにまとめる", menu_merge_sets),
    ("18", "リアルタイム監視を開く", menu_live_viewer),
    ("19", "トークンを登録する", menu_token),
    ("20", "本番（大会当日）", menu_tournament),
    ("21", "LAN対戦の準備（固定IP・ファイアウォール）", menu_firewall),
]


def main() -> None:
    print("ヘキサうどん AI（第37回 高専プロコン 競技部門）")
    print(build_line())
    print(f"作業フォルダ: {BASE_DIR}")
    print(f"Python {sys.version.split()[0]}")
    while True:
        print()
        print("―" * 46)
        for key, label, _handler in MENU:
            print(f"  {key}. {label}")
        print("  0. 終了")
        choice = ask("番号", "3")
        if choice == "0":
            print("終了します。")
            return
        for key, _label, handler in MENU:
            if key == choice:
                try:
                    handler()
                except KeyboardInterrupt:
                    print("\n  中断しました。")
                break
        else:
            print("  その番号はありません。")


if __name__ == "__main__":
    main()
