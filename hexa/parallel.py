"""探索を複数のコアで走らせる。

**プールは1つだけ作り、プログラムが終わるまで使い回します。**

Windows には fork がなく spawn なので、子プロセスを作るたびに Python を
起動し直します。実測で11プロセスの起動に **10.6秒** かかりました。
以前は事前試走と毎日の探索でそれぞれ作り直していたため、1日60秒のうち
10秒が起動に消えていました。タスクマネージャーで「1日が終わるとCPUが
落ち、しばらくして上がる」ように見えたのはこれです。

いまは最初に1つ作り、全部の探索で使い回します。試合が終わっても
閉じません（次の試合でも使うため）。プログラムが終わるときに
`shutdown()` で閉じます。

子プロセスは Ctrl+C を無視します。親を止めたときに11個が同時に
例外を吐いて画面を埋める、ということが起きません。
"""

from __future__ import annotations

import atexit
import os
import time

# 子プロセスごとに覚えておく読み込み済みのもの。
#
# **4件ごとに作り直すと重すぎます。**
# 実戦の32x32（175歩）で1件1.1秒かかっていました。問題の読み込みと
# 経路表の下ごしらえが、32x32では無視できない重さだからです。
# 1回読んだら、その子プロセスが生きているあいだ使い回します。
_CACHE: dict = {}


def _load_once(problem_path: str, population_path: str):
    """問題・重み・経路表を1回だけ作る。"""

    key = (problem_path, population_path)
    got = _CACHE.get(key)
    if got is not None:
        return got
    from hexa.ai import load_population
    from hexa.pathfind import PathFinder
    from hexa.planner import DayPlanner
    from hexa.problem import load_problem

    problem = load_problem(problem_path)
    brains = load_population(population_path) if population_path else []
    finder = PathFinder(problem)
    planner = DayPlanner(problem, finder)
    _CACHE.clear()
    _CACHE[key] = (problem, brains, finder, planner)
    return _CACHE[key]


# 全体で1つだけ持つプール。
_POOL = None
_POOL_FILES: tuple[str, str] = ("", "")


def _ignore_interrupt() -> None:
    """子プロセスの初期化。Ctrl+C を無視します。"""

    import signal

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def start_pool(problem_path: str, population_path: str, workers: int):
    """プールを1つ作る。すでにあればそれを返す。

    問題が変わった（次の試合になった）ときだけ作り直します。
    """

    global _POOL, _POOL_FILES
    if workers <= 1:
        return None
    if _POOL is not None and _POOL_FILES == (problem_path, population_path):
        return _POOL
    shutdown()
    try:
        import multiprocessing as mp

        pool = mp.Pool(processes=workers, initializer=_ignore_interrupt)
        # 空の仕事を人数分投げて、起動だけ先に済ませる。
        tasks = [
            pool.apply_async(_worker, ((problem_path, population_path, [], 0.0),))
            for _ in range(workers)
        ]
        for task in tasks:
            try:
                task.get(timeout=60)
            except Exception:
                pass
        _POOL = pool
        _POOL_FILES = (problem_path, population_path)
        return _POOL
    except Exception:
        _POOL = None
        return None


def shutdown() -> None:
    """プールを閉じる。プログラムの終わりと、問題が変わったときに呼びます。"""

    global _POOL, _POOL_FILES
    if _POOL is None:
        return
    try:
        _POOL.terminate()
        _POOL.join()
    except Exception:
        pass
    _POOL = None
    _POOL_FILES = ("", "")


atexit.register(shutdown)


# 子プロセス1つが受け持つ組み合わせの数。
# 細かく渡すと起動と通信の回数が増えるので、まとめて渡します。
CHUNK = 4


def _worker(payload):
    """子プロセスで動く。組み合わせをいくつか試して結果を返す。"""

    import random

    from hexa.ai import Policy, load_population
    from hexa.pathfind import PathFinder
    from hexa.planner import DayPlanner
    from hexa.problem import load_problem
    from hexa.production import (TRIAL_ROUTE_BUDGET, GreedyPolicy,
                                 _rollout_match)
    from hexa.rules import AgentKind

    problem_path, population_path, jobs, deadline = payload
    if not jobs:
        return []
    try:
        problem = load_problem(problem_path)
        brains = load_population(population_path) if population_path else []
    except Exception as error:
        # 子プロセス側の失敗は親に伝わらないので、印を返します。
        return [("error", f"{type(error).__name__}: {error}", 0, ())]

    finder = PathFinder(problem)
    planner = DayPlanner(problem, finder)
    results = []
    from hexa.production import kinds_for_seats

    for job in jobs:
        # **席も受け取ります。** 種別は車ごとに提出するので、どの出発点を
        # 補給車にするかは自由です。以前は必ず後ろ詰めにしていたので、
        # 本番PC（複数コア）では席の探索がまったく効いていませんでした。
        if len(job) >= 6:
            label, brain_index, supply_count, noise, seed, seats = job[:6]
        else:
            label, brain_index, supply_count, noise, seed = job[:5]
            seats = tuple(range(problem.agent_count - supply_count,
                                problem.agent_count))
        if time.monotonic() > deadline:
            break
        if supply_count >= problem.agent_count:
            continue
        kinds = kinds_for_seats(problem.agent_count, seats)
        if brain_index < 0:
            policy = GreedyPolicy(problem)
        else:
            if brain_index >= len(brains):
                continue
            # 事前試走でも親と同じ作戦の数にします（渡し忘れていました）。
            policy = Policy(
                brains[brain_index], problem, random.Random(seed), noise,
                route_budget=TRIAL_ROUTE_BUDGET,
            )
        try:
            score = _rollout_match(problem, planner, kinds, policy)
        except Exception:
            score = None
        if score is not None:
            results.append((list(score), label, supply_count,
                            tuple(seats)))
    return results


def _drain(pool, worker, payloads, deadline, workers):
    """仕事を流し続けて、終わったものから集める。

    手の空いた子が遊ばないよう、終わった子にすぐ次を渡します。
    """

    collected = []
    errors: list[str] = []
    pending = []
    index = 0
    try:
        while index < len(payloads) and len(pending) < workers * 2:
            pending.append(pool.apply_async(worker, (payloads[index],)))
            index += 1
        while pending and time.monotonic() < deadline:
            done_at = None
            for position, task in enumerate(pending):
                if task.ready():
                    done_at = position
                    break
            if done_at is None:
                time.sleep(0.005)
                continue
            task = pending.pop(done_at)
            try:
                got = task.get(timeout=0.5)
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                got = []
            for item in got:
                if item and item[0] == "error":
                    errors.append(str(item[1]))
                else:
                    collected.append(item)
            if index < len(payloads) and time.monotonic() < deadline:
                pending.append(pool.apply_async(worker, (payloads[index],)))
                index += 1
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    return collected, errors


def run_trials(problem_path, population_path, jobs, deadline, workers):
    """事前試走を並列で走らせる。プールは使い回します。"""

    if workers <= 1 or not jobs:
        return []
    pool = start_pool(problem_path, population_path, workers)
    if pool is None:
        return []
    chunks = [jobs[i:i + CHUNK] for i in range(0, len(jobs), CHUNK)]
    payloads = [
        (problem_path, population_path, chunk, deadline) for chunk in chunks
    ]
    collected, errors = _drain(pool, _worker, payloads, deadline, workers)
    if errors and not collected:
        raise RuntimeError(errors[0])
    return collected


# 「順路を磨く仕事」の印。brain_index のところに入れます。
POLISH_JOB = -3


def _polish_job(problem, planner, kinds, state, road, acquired, routes,
                skip: int, budget: int, deadline, supply_order=None,
                phase: int = 0):
    """子プロセスで順路を磨く。良くなった案を1つ返す（なければ None）。"""

    from hexa.improve import RoutePolisher
    from hexa.production import PlannedRoutePolicy, _build_candidate

    def evaluate(trial, order):
        try:
            # 子の案は親が `_candidate_from_plan` で採点し直すので、
            # ここでは公式シミュレータに通しません（2割速くなります）。
            return _build_candidate(
                "磨き", problem, planner, kinds,
                list(state["positions"]), list(state["fuels"]), road,
                state["day_steps"], PlannedRoutePolicy(problem, trial, order),
                acquired, state["day_index"], None, None,
                state["steps_ahead"], state["next_steps"], verify=False,
            )
        except Exception:
            return None

    base = evaluate(routes, supply_order or [])
    if base is None:
        return None
    from hexa.production import make_prefilter

    polisher = RoutePolisher(
        problem, kinds, routes, evaluate, key=base.key,
        supply_order=supply_order,
        prefilter=make_prefilter(problem, planner.finder, kinds,
                                 list(state["positions"]),
                                 list(state["fuels"]), road,
                                 state["day_steps"]),
    )
    # 子は同じ土台から始めるので、崩し方も子ごとにずらします。
    # 全員が同じ順で崩すと、同じ谷を何度も登り直すことになります。
    polisher.shake = phase
    # **子ごとに違う種類の手から始めます。**
    # 同じ種類ばかり試すと、受け持ちの大きな入れ替えに誰も
    # たどり着きません。
    polisher.phase = phase
    polisher.moves = polisher._new_moves()
    # **子ごとに試す手の開始位置をずらします。**
    #
    # 盤面が小さいと手の総数が skip より少ないことがあります。
    # そのまま進めると、1手も試さないまま「出し切った」になって
    # 仕事が丸ごと無駄になります（16x16では手が数千しかありません）。
    # 行き止まったら先頭に戻してやり直します。
    for _ in range(skip):
        if polisher.exhausted:
            break
        try:
            next(polisher.moves)
        except StopIteration:
            polisher.moves = polisher._new_moves()
            polisher.exhausted = False
            break
    best = None
    for _ in range(budget):
        if time.monotonic() > deadline or polisher.exhausted:
            break
        found = polisher.step()
        if found is not None:
            best = found
    if best is None:
        return None
    return (list(best.key), best.label,
            [list(row) for row in best.action_plan])


def _day_worker(payload):
    """子プロセスで1日分の計画をいくつか作る。

    事前試走と違い、その日の途中の状態（位置・燃料・道路）を渡します。
    戻すのは「スコアと行動計画」だけにして、重い中身は送りません。
    """

    import random

    from hexa.ai import Policy, load_population
    from hexa.pathfind import PathFinder
    from hexa.planner import DayPlanner
    from hexa.problem import load_problem
    from hexa.production import (DAY_ROUTE_BUDGET, BrandFirstPolicy,
                                 GreedyPolicy, NoisyGreedyPolicy,
                                 _build_candidate)
    from hexa.rules import AgentKind, RoadStatus

    (problem_path, population_path, state, jobs, deadline) = payload
    if not jobs:
        return []
    try:
        problem, brains, _finder, planner = _load_once(
            problem_path, population_path
        )
    except Exception as error:
        return [("error", f"{type(error).__name__}: {error}", None)]

    kinds = tuple(AgentKind(value) for value in state["kinds"])
    road = {int(k): RoadStatus(v) for k, v in state["roads"].items()}
    acquired = set(state["acquired"])
    results = []
    for job in jobs:
        label, brain_index, noise, seed = job[:4]
        if time.monotonic() > deadline:
            break
        if brain_index == POLISH_JOB:
            # **順路を磨く仕事。**
            #
            # 親から「いまの最良案の順路」を受け取り、手を1つずつ
            # 試して実際に1日ぶんを再現します。親は1コアしか使えない
            # ので、ここで11コア分に増やします。
            #
            # 子ごとに試す手の開始位置をずらすので、同じ手を
            # 重ねて試しません。
            routes, skip, budget = job[4], job[5], job[6]
            order = list(job[7]) if len(job) > 7 else []
            phase = int(job[8]) if len(job) > 8 else 0
            found = _polish_job(
                problem, planner, kinds, state, road, acquired,
                {int(k): list(v) for k, v in routes.items()}, skip, budget,
                deadline, order, phase,
            )
            if found is not None:
                results.append(found)
            continue
        if brain_index == -2:
            policy = BrandFirstPolicy(problem)
        elif brain_index == -1:
            policy = (
                NoisyGreedyPolicy(problem, random.Random(seed), noise)
                if noise > 0 else GreedyPolicy(problem)
            )
        else:
            if brain_index >= len(brains):
                continue
            # **親と同じ作戦の数にします。**
            #
            # 渡し忘れていたため、子は既定の12個で判断していました。
            # 親は6個です。多ければ良いわけではなく、実測では
            # 親72玉に対し子50玉と、**子のほうが悪い**結果でした。
            # 1回の判断が重くなり、道すじを伸ばす回数が減るためです。
            policy = Policy(brains[brain_index], problem,
                            random.Random(seed), noise,
                            route_budget=DAY_ROUTE_BUDGET)
        try:
            candidate = _build_candidate(
                label, problem, planner, kinds,
                list(state["positions"]), list(state["fuels"]), road,
                state["day_steps"], policy, acquired, state["day_index"],
                None, None, state["steps_ahead"], state["next_steps"],
            )
        except Exception:
            candidate = None
        if candidate is not None:
            results.append((list(candidate.key), candidate.label,
                            [list(row) for row in candidate.action_plan]))
    return results


class DaySearch:
    """1日分の探索を子プロセスに投げたまま、少しずつ回収する入れ物。

    **親を待たせないための仕組みです。**

    以前は「並列が全部終わってから親が探し始める」形でした。
    締切の7割で並列を切り上げていたため、残り3割は親1つだけが動き、
    CPUが「100%が20秒 → 10%が20秒」と波打っていました。

    いまは投げたらすぐ戻り、親は自分の探索を続けます。
    その合間に `harvest()` を呼ぶと、終わったぶんだけ受け取れます。
    """

    def __init__(self, pool, problem_path, population_path, state,
                 jobs, deadline, workers):
        self.pool = pool
        self.deadline = deadline
        self.workers = workers
        self.chunks = [jobs[i:i + CHUNK] for i in range(0, len(jobs), CHUNK)]
        self.args = (problem_path, population_path, state)
        self.index = 0
        self.pending = []
        self.collected = 0
        # これまでに積んだ「磨き」の仕事の数（開始位置をずらすのに使う）
        self.polish_queued = 0
        # まだ子に渡していない「磨き」の数。`_fill` が渡すたびに減ります。
        self.polish_left = 0
        self._fill()

    def _fill(self) -> None:
        """手の空いた子に次の仕事を渡す。"""

        # **多めに投げておきます。**
        #
        # 以前は「子の数の2倍」でした。32x32の長い日は1案250ミリ秒
        # かかるので、22チャンクでは1秒ちょっとで尽きます。
        # 回収は0.2秒ごとなので、そのたびに数個しか補充できず、
        # 子が手持ち無沙汰になっていました（理論2200案に対し573案）。
        #
        # プールが順番に配ってくれるので、多く積んでも害はありません。
        #
        # **ただし積みすぎると「磨き」が待たされます。**
        # 親は最良案が変わるたびに磨く仕事を割り込ませますが、
        # すでに投げた仕事の後ろに並ぶため、積みすぎると順番が
        # 回ってきません。6倍なら、32x32（1案250ミリ秒）でも
        # 11コアで約6秒ぶんの手持ちになり、子は遊ばず、磨きの
        # 待ちも1秒ほどで済みます。
        while (self.index < len(self.chunks)
               and len(self.pending) < self.workers * 6
               and time.monotonic() < self.deadline):
            problem_path, population_path, state = self.args
            chunk = self.chunks[self.index]
            payload = (problem_path, population_path, state,
                       chunk, self.deadline)
            try:
                self.pending.append(self.pool.apply_async(_day_worker, (payload,)))
            except Exception:
                break
            if chunk and chunk[0][1] == POLISH_JOB:
                self.polish_left -= 1
            self.index += 1

    def add_polish_jobs(self, routes, budget: int, count: int,
                        supply_order=None, offset: int = 0,
                        gap: int = 0) -> None:
        """「いまの最良案の順路を磨く」仕事を、次の順番に割り込ませる。

        子ごとに試す手の開始位置をずらすので、同じ手を重ねません。
        `offset` は、同じ順路に仕事を**継ぎ足す**ときに使います
        （前に出したぶんと違う崩し方から始めさせます）。
        """

        if not routes:
            return
        plain = {int(index): list(spots) for index, spots in routes.items()}
        order = [int(car) for car in (supply_order or [])]
        from hexa.improve import _MOVE_PHASES

        phases = len(_MOVE_PHASES)
        jobs = [
            (f"磨き{offset + slot}", POLISH_JOB, 0.0, 0, plain,
             # 空読みする量は4通りで打ち切ります。継ぎ足すたびに
             # `offset` が増えるので、そのまま掛けると何十万手も
             # 空読みすることになり、仕事が空振りします。
             (((offset + slot) // phases) % 4) * budget, budget, order,
             (offset + slot) % phases)
            for slot in range(count)
        ]
        chunks = [jobs[i:i + 1] for i in range(len(jobs))]
        # **いま投げる列の先頭に入れます。**
        #
        # `gap` を渡すと、磨き1本ごとに既存の仕事（ゆらぎ探索）を
        # その数だけ挟みます。**仕事の長さがまるで違う**ので、数で
        # 混ぜても時間では混ざりません。磨き1本は2500手（24x24で
        # 約3.4秒）、ゆらぎ1かたまりは4案（約0.03秒）です。1本ずつ
        # 交互にすると、時間の99%が磨きになってしまいます。
        at = self.index
        for chunk in chunks:
            self.chunks[at:at] = [chunk]
            at += 1 + max(0, gap)
        self.polish_queued += len(chunks)
        self.polish_left += len(chunks)
        self._fill()

    def polish_backlog(self) -> int:
        """まだ子に渡していない「磨き」の仕事の数。

        本番の列は、1日のはじめに積んだ「ゆらぎ探索」でいっぱいです。
        磨きはその後ろに並ぶので、順番が回ってきません。親は
        これを見て、磨きが尽きる前に継ぎ足します。
        """

        return self.polish_left

    def harvest(self):
        """終わったぶんだけ受け取る。待ちません。"""

        out = []
        still = []
        for task in self.pending:
            if not task.ready():
                still.append(task)
                continue
            try:
                got = task.get(timeout=0.1)
            except Exception:
                got = []
            for item in got:
                if item and item[0] != "error":
                    out.append(item)
        self.pending = still
        self.collected += len(out)
        self._fill()
        return out

    def stop(self) -> None:
        """もう受け取らない。残りの仕事は締切で自然に止まります。"""

        self.pending = []
        self.index = len(self.chunks)
        self.polish_left = 0


def start_day_search(problem_path, population_path, state, jobs,
                     deadline, workers):
    """1日分の探索を投げて、入れ物を返す。すぐ戻ります。"""

    if workers <= 1 or not jobs:
        return None
    pool = start_pool(problem_path, population_path, workers)
    if pool is None:
        return None
    return DaySearch(pool, problem_path, population_path, state,
                     jobs, deadline, workers)


def run_day_search(problem_path, population_path, state, jobs, deadline, workers):
    """1日分の計画を並列で作る。プールは使い回します。

    **毎日プールを作り直さないこと**が肝心です。Windows では起動に
    10秒かかるので、1日60秒のうち6分の1が消えていました。
    """

    if workers <= 1 or not jobs:
        return []
    pool = start_pool(problem_path, population_path, workers)
    if pool is None:
        return []
    chunks = [jobs[i:i + CHUNK] for i in range(0, len(jobs), CHUNK)]
    payloads = [
        (problem_path, population_path, state, chunk, deadline)
        for chunk in chunks
    ]
    collected, _errors = _drain(pool, _day_worker, payloads, deadline, workers)
    return collected


def warm_up(problem_path, population_path, workers):
    """子プロセスを先に起こしておく。start_pool と同じです。"""

    return start_pool(problem_path, population_path, workers)


def usable_workers(limit: int = 0) -> int:
    """使ってよいプロセス数。1 なら並列にしません。

    **論理プロセッサ数から1を引いた数**を使います。1つは本体
    （探索を続ける親プロセスと、監視の書き出し）のために空けます。

    以前は「コア数の半分」にしていましたが、本番用PCは試合中これ以外に
    することがないので、使い切るほうが候補を多く作れます。
    """

    try:
        count = os.cpu_count() or 1
    except Exception:
        return 1
    if count <= 2:
        return 1
    allowed = max(2, count - 1)
    if limit > 0:
        allowed = min(allowed, limit)
    return allowed
