"""ルール・生成・計画・学習の回帰テスト。"""

import random
import unittest
from pathlib import Path

from hexa.ai import Brain, new_population
from hexa.generator import GeneratorSettings, PRESETS, generate_problem
from hexa.hexgrid import neighbor
from hexa.match import run_match
from hexa.pathfind import PathFinder
from hexa.problem import validate_problem
from hexa.rules import Terrain
from hexa.rules import (
    MOUNTAIN_FUEL_COST,
    MOUNTAIN_MOVE_STEPS,
    PLAIN_FUEL_COST,
    PLAIN_MOVE_STEPS,
    ROAD_FUEL_COST,
    ROAD_MOVE_STEPS,
    AgentKind,
    RoadStatus,
    Terrain,
    fuel_cost,
    move_steps,
)
from hexa.scoring import Score, rank_scores
from hexa.simulator import InvalidAnswer, road_status_for_day, simulate_day


class RulesTest(unittest.TestCase):
    def test_move_steps_match_table_1(self):
        # 募集要項 p.9 表1
        self.assertEqual(move_steps(Terrain.PLAIN, None), 2)
        self.assertEqual(move_steps(Terrain.MOUNTAIN, None), 3)
        self.assertEqual(move_steps(Terrain.ROAD, RoadStatus.SMOOTH), 1)
        self.assertEqual(move_steps(Terrain.ROAD, RoadStatus.BUSY), 2)
        self.assertEqual(move_steps(Terrain.ROAD, RoadStatus.JAMMED), 4)
        self.assertEqual(PLAIN_MOVE_STEPS, 2)
        self.assertEqual(MOUNTAIN_MOVE_STEPS, 3)
        self.assertEqual(ROAD_MOVE_STEPS[RoadStatus.JAMMED], 4)

    def test_fuel_costs_match_table_1(self):
        # 道路は状態によらず 2（質問回答集その2 Q28 で「燃量」は誤記と確定）
        self.assertEqual(fuel_cost(Terrain.PLAIN), PLAIN_FUEL_COST)
        self.assertEqual(fuel_cost(Terrain.MOUNTAIN), MOUNTAIN_FUEL_COST)
        self.assertEqual(fuel_cost(Terrain.ROAD), ROAD_FUEL_COST)
        self.assertEqual((PLAIN_FUEL_COST, MOUNTAIN_FUEL_COST, ROAD_FUEL_COST), (1, 2, 2))

    def test_pond_cannot_be_left(self):
        with self.assertRaises(ValueError):
            move_steps(Terrain.POND, None)


class HexGridTest(unittest.TestCase):
    def test_even_rows_are_shifted_right(self):
        # 質問回答集その1 Q1 と 募集要項 図1（10x10の座標例）
        width, height = 10, 10
        # 図1の座標例: 0行目(0～9)は右にずれ、1行目(10～19)はずれない。
        # よってセル11の左上はセル0、右上はセル1になる。
        self.assertEqual(neighbor(11, 0, width, height), 0)
        self.assertEqual(neighbor(11, 1, width, height), 1)
        # 0行目のセル1の左下は11、右下は12
        self.assertEqual(neighbor(1, 4, width, height), 11)
        self.assertEqual(neighbor(1, 3, width, height), 12)

    def test_directions_follow_q47(self):
        # 質問回答集その2 Q47: 2=右, 5=左
        self.assertEqual(neighbor(11, 2, 10, 10), 12)
        self.assertEqual(neighbor(11, 5, 10, 10), 10)

    def test_map_is_not_a_torus(self):
        # 質問回答集その1 Q2
        self.assertIsNone(neighbor(0, 5, 10, 10))
        self.assertIsNone(neighbor(0, 0, 10, 10))


class GeneratorTest(unittest.TestCase):
    def test_every_preset_passes_validation(self):
        for name, settings in PRESETS.items():
            for seed in (1, 2, 3):
                problem = generate_problem(seed, settings)
                self.assertEqual(validate_problem(problem), [], f"{name} seed={seed}")

    def test_day_steps_stay_inside_the_official_range(self):
        # 質問回答集その1 Q20
        problem = generate_problem(11, PRESETS["large"])
        span = problem.width + problem.height
        for steps in problem.day_steps:
            self.assertGreaterEqual(steps, span)
            self.assertLessEqual(steps, span * 4)

    def test_fuel_limit_follows_q62(self):
        # 質問回答集その3 Q62: 1日目のステップ数の1～3倍
        problem = generate_problem(12, PRESETS["medium"])
        self.assertGreaterEqual(problem.fuel_limit, problem.day_steps[0])
        self.assertLessEqual(problem.fuel_limit, problem.day_steps[0] * 3)

    def test_stocks_never_exceed_the_agent_count(self):
        # 募集要項 p.9
        problem = generate_problem(13, PRESETS["large"])
        for spot in problem.spots:
            self.assertGreaterEqual(spot.stocks, 1)
            self.assertLessEqual(spot.stocks, problem.agent_count)


class TrafficTest(unittest.TestCase):
    def setUp(self):
        self.problem = generate_problem(21, PRESETS["small"])

    def test_first_day_is_all_smooth(self):
        # 募集要項 p.9: 1日目の道路の状態は全て順調
        status = road_status_for_day(self.problem, [])
        self.assertTrue(all(value == RoadStatus.SMOOTH for value in status.values()))

    def test_second_day_uses_only_the_first_day(self):
        road = next(
            position
            for position in range(self.problem.cell_count)
            if self.problem.terrain_at(position) == Terrain.ROAD
        )
        players = self.problem.players
        need = self.problem.jammed_threshold * players
        status = road_status_for_day(self.problem, [{road: need}])
        self.assertEqual(status[road], RoadStatus.JAMMED)

    def test_threshold_boundaries_are_inclusive(self):
        # 交通量 < 混雑 → 順調 / 混雑 <= 交通量 < 渋滞 → 混雑 / 渋滞 <= 交通量 → 渋滞
        problem = self.problem
        road = next(
            position
            for position in range(problem.cell_count)
            if problem.terrain_at(position) == Terrain.ROAD
        )
        players = problem.players
        just_below = problem.busy_threshold * players - 1
        exactly_busy = problem.busy_threshold * players
        exactly_jammed = problem.jammed_threshold * players
        self.assertEqual(
            road_status_for_day(problem, [{road: just_below}])[road], RoadStatus.SMOOTH
        )
        self.assertEqual(
            road_status_for_day(problem, [{road: exactly_busy}])[road], RoadStatus.BUSY
        )
        self.assertEqual(
            road_status_for_day(problem, [{road: exactly_jammed}])[road], RoadStatus.JAMMED
        )

    def test_two_days_are_summed(self):
        problem = self.problem
        road = next(
            position
            for position in range(problem.cell_count)
            if problem.terrain_at(position) == Terrain.ROAD
        )
        half = problem.jammed_threshold * problem.players // 2 + 1
        status = road_status_for_day(problem, [{road: half}, {road: half}])
        self.assertEqual(status[road], RoadStatus.JAMMED)

    def test_only_the_last_two_days_count(self):
        problem = self.problem
        road = next(
            position
            for position in range(problem.cell_count)
            if problem.terrain_at(position) == Terrain.ROAD
        )
        huge = problem.jammed_threshold * problem.players * 10
        status = road_status_for_day(problem, [{road: huge}, {}, {}])
        self.assertEqual(status[road], RoadStatus.SMOOTH)


class InvalidAnswerTest(unittest.TestCase):
    def setUp(self):
        self.problem = generate_problem(31, PRESETS["small"])
        self.kinds = tuple(AgentKind.TOURING for _ in self.problem.agent_positions)
        self.fuels = tuple(self.problem.fuel_limit for _ in self.problem.agent_positions)

    def _run(self, plan):
        return simulate_day(
            problem=self.problem,
            agent_kinds=self.kinds,
            start_positions=self.problem.agent_positions,
            start_fuels=self.fuels,
            action_plan=plan,
            day_steps=self.problem.day_steps[0],
            road_status={},
        )

    def test_plan_shorter_than_the_day_is_rejected(self):
        plan = [[-1] for _ in self.kinds]
        with self.assertRaises(InvalidAnswer):
            self._run(plan)

    def test_out_of_range_action_is_rejected(self):
        plan = [[6] + [-1] for _ in self.kinds]
        with self.assertRaises(InvalidAnswer):
            self._run(plan)

    def test_waiting_the_whole_day_is_valid(self):
        steps = self.problem.day_steps[0]
        plan = [[-steps] for _ in self.kinds]
        result = self._run(plan)
        self.assertEqual(list(result.positions), list(self.problem.agent_positions))
        # 待機中は燃料を消費しない（質問回答集その1 Q16）
        self.assertEqual(list(result.fuels), list(self.fuels))

    def test_wrong_agent_count_is_rejected(self):
        with self.assertRaises(InvalidAnswer):
            self._run([[-1]])


class PathFinderTest(unittest.TestCase):
    def test_cache_avoids_recomputation(self):
        problem = generate_problem(41, PRESETS["medium"])
        finder = PathFinder(problem)
        for _ in range(50):
            finder.table(problem.agent_positions[0], {})
        self.assertEqual(finder.compute_count, 1)
        self.assertEqual(finder.query_count, 50)

    def test_path_directions_lead_to_the_target(self):
        problem = generate_problem(42, PRESETS["small"])
        finder = PathFinder(problem)
        origin = problem.agent_positions[0]
        table = finder.table(origin, {})
        target = problem.spots[0].position
        self.assertTrue(table.reachable(target))
        position = origin
        for direction in table.directions_to(target):
            found = None
            for candidate_direction, candidate in finder.neighbors[position]:
                if candidate_direction == direction:
                    found = candidate
            self.assertIsNotNone(found)
            position = found
        self.assertEqual(position, target)


class ScoringTest(unittest.TestCase):
    def test_priority_is_brand_then_daily_then_noodles(self):
        # 募集要項 p.12 の優先順位
        scores = [Score(3, 5, 100), Score(4, 1, 1), Score(3, 6, 2)]
        self.assertEqual(rank_scores(scores), [3, 1, 2])

    def test_ties_share_a_rank(self):
        scores = [Score(2, 2, 2), Score(2, 2, 2)]
        self.assertEqual(rank_scores(scores), [1, 1])


class MatchTest(unittest.TestCase):
    def test_match_produces_valid_answers_for_every_preset(self):
        brains = new_population()
        for name in ("small", "medium"):
            problem = generate_problem(51, PRESETS[name])
            result = run_match(problem, brains, seed=7, epsilon=0.3, kind_explore=0.5)
            self.assertEqual(len(result.teams), 4)
            for team in result.teams:
                self.assertEqual(len(team.action_plans_by_day), problem.day_count)
                for day_index, plan in enumerate(team.action_plans_by_day):
                    self.assertEqual(len(plan), problem.agent_count)
                    for actions in plan:
                        total = sum(-a if a < 0 else 0 for a in actions)
                        self.assertGreaterEqual(total, 0)

    def test_supply_cars_actually_refuel(self):
        settings = GeneratorSettings(
            height=12, width=12, agent_count=6, spot_count=10,
            brand_count=5, day_count=6, players=4,
        )
        problem = generate_problem(61, settings)
        brains = new_population()
        refuels = 0
        for seed in range(6):
            result = run_match(problem, brains, seed=seed, epsilon=0.4, kind_explore=1.0)
            for team in result.teams:
                for day_result in team.day_results:
                    refuels += len(day_result.refuels)
        self.assertGreater(refuels, 0, "補給がまったく起きていません")

    def test_four_ai_do_not_move_identically(self):
        from hexa.viz import route_overlap_rate

        problem = generate_problem(71, PRESETS["medium"])
        brains = new_population()
        result = run_match(problem, brains, seed=3, epsilon=0.25, kind_explore=0.6,
                           record_trace=True)
        self.assertLess(route_overlap_rate(result), 0.95)


class LearningTest(unittest.TestCase):
    def test_updates_move_the_prediction_towards_the_reward(self):
        brain = Brain.new("test", 1)
        features = [1.0, 0.5, 0.0, 0.2, 0.8, 0.1, 0.9, 0.7, 1.0, 1.0, 0.5, 0.0, 0.3, 0.2]
        rewards = [1.0, 0.5, 0.4]
        before = brain.predict_touring(features)
        for _ in range(200):
            brain.update_touring(features, rewards, 0.05)
        after = brain.predict_touring(features)
        for index in range(3):
            self.assertLess(
                abs(after[index] - rewards[index]),
                abs(before[index] - rewards[index]),
            )

    def test_supply_count_bandit_tries_every_option_first(self):
        brain = Brain.new("test", 2)
        rng = random.Random(0)
        seen = set()
        for _ in range(200):
            supply_count = brain.best_supply_count(4, rng)
            seen.add(supply_count)
            brain.record_kind_result(4, supply_count, (1, 1, 1))
        self.assertEqual(seen, {0, 1, 2, 3})

    def test_bandit_prefers_the_better_configuration(self):
        brain = Brain.new("test", 3)
        rng = random.Random(0)
        for supply_count in range(4):
            brain.record_kind_result(4, supply_count, (1, 1, 1))
        for _ in range(20):
            brain.record_kind_result(4, 2, (5, 5, 5))
        self.assertEqual(brain.best_supply_count(4, rng), 2)


if __name__ == "__main__":
    unittest.main()


class VisualizationTest(unittest.TestCase):
    def test_writing_twice_keeps_the_index_valid(self):
        """index.json 自身を読み込もうとして落ちないこと。

        以前、data フォルダの *.json を全部読む実装になっており、
        一覧ファイル index.json（中身は配列）まで読んで
        AttributeError: 'list' object has no attribute 'get' で落ちていた。
        """

        import json
        import tempfile
        from pathlib import Path

        from hexa.viz import build_visualization, write_visualization

        problem = generate_problem(81, PRESETS["small"])
        brains = new_population()
        result = run_match(problem, brains, seed=11, record_trace=True)

        with tempfile.TemporaryDirectory() as directory:
            viewer_dir = Path(directory)
            for slug in ("first", "second", "third"):
                payload = build_visualization(result, title=slug)
                write_visualization(payload, viewer_dir, slug)

            index_path = viewer_dir / "data" / "index.json"
            with open(index_path, encoding="utf-8") as handle:
                entries = json.load(handle)
            slugs = {entry["slug"] for entry in entries}
            self.assertEqual(slugs, {"first", "second", "third"})

            index_js = (viewer_dir / "data" / "index.js").read_text(encoding="utf-8")
            self.assertTrue(index_js.startswith("window.HEXA_INDEX = "))
            self.assertTrue((viewer_dir / "data" / "first.js").exists())

    def test_index_survives_a_stray_file(self):
        import tempfile
        from pathlib import Path

        from hexa.viz import build_visualization, write_visualization

        problem = generate_problem(82, PRESETS["small"])
        brains = new_population()
        result = run_match(problem, brains, seed=12, record_trace=True)

        with tempfile.TemporaryDirectory() as directory:
            viewer_dir = Path(directory)
            data_dir = viewer_dir / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "broken.json").write_text("{ not json", encoding="utf-8")
            (data_dir / "array.json").write_text("[1, 2, 3]", encoding="utf-8")

            payload = build_visualization(result, title="ok")
            write_visualization(payload, viewer_dir, "ok")

            import json

            with open(data_dir / "index.json", encoding="utf-8") as handle:
                entries = json.load(handle)
            self.assertEqual([entry["slug"] for entry in entries], ["ok"])


class ProductionTest(unittest.TestCase):
    def test_agents_do_not_stall_when_spots_are_available(self):
        """待機を選び続けて一歩も動かなくなる不具合の回帰テスト。

        原因は、獲得済みの系列しか残っていない場面で第1優先（種類数）の
        予測がわずかに負になり、待機（ほぼ0）に第1優先で負けて、
        第2・第3優先が一度も見られなくなることだった。
        """

        from hexa.ai import Brain, Policy

        problem = generate_problem(91, PRESETS["medium"])
        brain = Brain.new("stall", 5)
        # 「行く」の第1優先だけをわずかに負にする重みを作る。
        width = len(brain.touring_heads[0])
        brain.touring_heads[0] = [0.0] * width
        brain.touring_heads[0][2] = -0.2      # 作戦を実行する選択にだけ効く
        brain.touring_heads[1] = [0.0] * width
        brain.touring_heads[1][2] = 0.9       # 第2優先では「行く」が圧勝
        brain.touring_heads[2] = [0.0] * width
        brain.touring_heads[2][2] = 0.9

        policy = Policy(brain, problem, random.Random(0), epsilon=0.0)
        finder = PathFinder(problem)
        decision = policy.choose_touring_target(
            agent_index=0,
            step=0,
            remaining=problem.day_steps[0],
            position=problem.agent_positions[0],
            fuel=problem.fuel_limit,
            stock={spot.position: spot.stocks for spot in problem.spots},
            reserved={},
            acquired_today=set(),
            acquired_brands=set(range(10)),   # 全系列すでに獲得済み
            brands_today=set(),
            road_status={},
            finder=finder,
        )
        self.assertIsNotNone(
            decision.target,
            "行ける店があるのに待機を選んでいます",
        )

    def test_greedy_baseline_never_idles_when_a_spot_is_reachable(self):
        from hexa.production import GreedyPolicy

        problem = generate_problem(92, PRESETS["medium"])
        policy = GreedyPolicy(problem)
        finder = PathFinder(problem)
        decision = policy.choose_touring_target(
            agent_index=0,
            step=0,
            remaining=problem.day_steps[0],
            position=problem.agent_positions[0],
            fuel=problem.fuel_limit,
            stock={spot.position: spot.stocks for spot in problem.spots},
            reserved={},
            acquired_today=set(),
            acquired_brands=set(),
            brands_today=set(),
            road_status={},
            finder=finder,
        )
        self.assertIsNotNone(decision.target)

    def test_agents_keep_collecting_on_later_days(self):
        """後半の日でも獲得が続くこと（初日だけ動いて止まる不具合の回帰テスト）。"""

        brains = new_population()
        problem = generate_problem(93, PRESETS["medium"])
        result = run_match(problem, brains, seed=5)
        for team in result.teams:
            last_day = team.day_results[-1]
            moved = any(
                len(day.acquisitions) > 0 for day in team.day_results[len(team.day_results) // 2:]
            )
            self.assertTrue(
                moved,
                f"{team.brain_name} が試合後半に1玉も獲得していません",
            )
            self.assertGreater(len(last_day.stay_counts), 1)


class ViewerDataTest(unittest.TestCase):
    def setUp(self):
        from hexa.viz import build_visualization

        self.problem = generate_problem(101, PRESETS["medium"])
        self.result = run_match(
            self.problem, new_population(), seed=7, record_trace=True
        )
        self.payload = build_visualization(self.result, "viewer-test")

    def test_every_day_has_one_frame_per_step_plus_the_start(self):
        """ビューアのステップ数が公式のステップ数と合っていること。

        0ステップ目の状態が1フレーム、そのあと1〜Nステップ目の反映後が
        Nフレームなので、合計は daySteps + 1 になる。
        """

        for team in self.payload["teams"]:
            for day_index, day in enumerate(team["days"]):
                self.assertEqual(
                    len(day["steps"]),
                    self.problem.day_steps[day_index] + 1,
                    f"{team['name']} の{day_index + 1}日目のフレーム数が合いません",
                )

    def test_every_frame_carries_position_and_fuel_for_all_agents(self):
        for team in self.payload["teams"]:
            for day in team["days"]:
                for frame in day["steps"]:
                    self.assertEqual(len(frame["positions"]), self.problem.agent_count)
                    self.assertEqual(len(frame["fuels"]), self.problem.agent_count)

    def test_viewer_stock_formula_matches_the_simulator(self):
        """ビューアは「最大在庫 − そこまでの獲得数」で残り在庫を出している。

        シミュレータが持っている残り在庫と一致することを確かめる。
        """

        for team_index, team in enumerate(self.payload["teams"]):
            for day_index, day in enumerate(team["days"]):
                last_step = self.problem.day_steps[day_index]
                stocks = {spot.position: spot.stocks for spot in self.problem.spots}
                for event in day["acquisitions"]:
                    if event["step"] <= last_step:
                        stocks[event["pos"]] -= 1
                self.assertEqual(
                    stocks,
                    self.result.teams[team_index].day_results[day_index].remaining_stocks,
                )

    def test_stock_never_goes_negative(self):
        for team in self.payload["teams"]:
            for day_index, day in enumerate(team["days"]):
                counts = {}
                for event in day["acquisitions"]:
                    counts[event["pos"]] = counts.get(event["pos"], 0) + 1
                for spot in self.problem.spots:
                    self.assertLessEqual(counts.get(spot.position, 0), spot.stocks)


class SupplyCarTest(unittest.TestCase):
    def test_two_supply_cars_pick_different_touring_cars(self):
        """補給車2台が同じ巡回車に張り付かないこと。

        2台とも同じマスに集まると、片方の働きがまるごと無駄になる。
        味方がまだ担当していない巡回車があるうちは、そちらを選ぶ。
        """

        from hexa.ai import Policy
        from hexa.rules import AgentKind

        problem = generate_problem(103, PRESETS["medium"])
        brain = new_population()[0]
        policy = Policy(brain, problem, random.Random(0), epsilon=0.0)
        finder = PathFinder(problem)
        kinds = tuple(
            AgentKind.SUPPLY if index >= problem.agent_count - 2 else AgentKind.TOURING
            for index in range(problem.agent_count)
        )
        fuels = [problem.fuel_limit // 2] * problem.agent_count

        first = policy.choose_supply_target(
            agent_index=problem.agent_count - 2, step=0,
            remaining=problem.day_steps[0], position=problem.agent_positions[-2],
            kinds=kinds, positions=list(problem.agent_positions), fuels=fuels,
            plan_targets=[None] * problem.agent_count, supply_claims={},
            road_status={}, finder=finder,
        )
        self.assertIsNotNone(first.target_agent)
        second = policy.choose_supply_target(
            agent_index=problem.agent_count - 1, step=0,
            remaining=problem.day_steps[0], position=problem.agent_positions[-1],
            kinds=kinds, positions=list(problem.agent_positions), fuels=fuels,
            plan_targets=[None] * problem.agent_count,
            supply_claims={first.target_agent: 1},
            road_status={}, finder=finder,
        )
        self.assertNotEqual(
            second.target_agent, first.target_agent,
            "2台目の補給車が1台目と同じ巡回車を選んでいます",
        )


    def test_supply_cars_move_instead_of_idling_all_day(self):
        """補給車が一日じゅう止まったままにならないこと。

        以前は「いま満タンだから用がない」と切っていたため、相手が空になるまで
        補給車が動かなかった。合流地点に着いた時点の燃料で判断するようにしてある。
        """

        from hexa.ai import Policy
        from hexa.rules import AgentKind

        problem = generate_problem(102, PRESETS["medium"])
        brain = new_population()[0]
        policy = Policy(brain, problem, random.Random(0), epsilon=0.0)
        finder = PathFinder(problem)

        kinds = tuple(
            AgentKind.SUPPLY if index >= problem.agent_count - 1 else AgentKind.TOURING
            for index in range(problem.agent_count)
        )
        # 全員が満タンの状態、つまり以前は候補が1つも出なかった場面。
        fuels = [problem.fuel_limit] * problem.agent_count
        decision = policy.choose_supply_target(
            agent_index=problem.agent_count - 1,
            step=0,
            remaining=problem.day_steps[0],
            position=problem.agent_positions[-1],
            kinds=kinds,
            positions=list(problem.agent_positions),
            fuels=fuels,
            plan_targets=[None] * problem.agent_count,
            supply_claims={},
            road_status={},
            finder=finder,
        )
        self.assertGreater(
            len(decision.candidate_features), 1,
            "全員満タンだと合流先の候補が1つも出ていません",
        )
        self.assertIsNotNone(
            decision.target,
            "候補があるのに補給車が待機を選んでいます",
        )


class OpponentModelTest(unittest.TestCase):
    """敵の道路の使い方を読む部分のテスト。"""

    def setUp(self):
        from hexa.opponents import OpponentModel
        from hexa.rules import Terrain

        self.problem = generate_problem(201, PRESETS["medium"])
        self.model = OpponentModel(self.problem)
        self.road = next(
            position
            for position in range(self.problem.cell_count)
            if self.problem.terrain_at(position) == Terrain.ROAD
        )

    def test_jammed_road_we_did_not_use_is_blamed_on_opponents(self):
        """自分が一切踏んでいない道路が渋滞していたら、敵が踏んでいる。"""

        from hexa.rules import RoadStatus

        self.model.observe_status({self.road: RoadStatus.JAMMED}, [{}])
        self.assertGreater(self.model.estimated_daily(self.road), 0.0)

    def test_our_own_traffic_is_subtracted(self):
        """同じ渋滞でも、自分がたくさん踏んでいたなら敵の寄与は小さい。"""

        from hexa.opponents import OpponentModel
        from hexa.rules import RoadStatus

        heavy = OpponentModel(self.problem)
        heavy.observe_status(
            {self.road: RoadStatus.JAMMED},
            [{self.road: self.problem.jammed_threshold * self.problem.players}],
        )
        light = OpponentModel(self.problem)
        light.observe_status({self.road: RoadStatus.JAMMED}, [{}])
        self.assertLess(
            heavy.estimated_daily(self.road),
            light.estimated_daily(self.road),
        )

    def test_sightings_give_a_lower_bound(self):
        """敵が道路の上にいるのを見たら、その分は必ず踏んでいる。"""

        for _ in range(3):
            self.model.observe_positions([self.road])
            self.model.observed_days += 1
        self.assertGreaterEqual(self.model.estimated_daily(self.road), 1.0)

    def test_hot_roads_are_sorted_by_estimate(self):
        from hexa.rules import RoadStatus, Terrain

        roads = [
            position
            for position in range(self.problem.cell_count)
            if self.problem.terrain_at(position) == Terrain.ROAD
        ][:3]
        status = {roads[0]: RoadStatus.JAMMED, roads[1]: RoadStatus.BUSY,
                  roads[2]: RoadStatus.SMOOTH}
        self.model.observe_status(status, [{}])
        hot = self.model.hot_roads()
        self.assertGreater(len(hot), 0)
        values = [value for _position, value in hot]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(hot[0][0], roads[0])

    def test_prediction_counts_our_planned_traffic(self):
        """これから自分が踏む分も、翌日の予測に入ること。"""

        from hexa.rules import RoadStatus

        quiet = self.model.predict_status([{}])
        self.assertEqual(quiet[self.road], RoadStatus.SMOOTH)
        busy = self.model.predict_status(
            [{}],
            extra_own={self.road: self.problem.jammed_threshold * self.problem.players},
        )
        self.assertEqual(busy[self.road], RoadStatus.JAMMED)


class BlockingTest(unittest.TestCase):
    def test_idle_supply_car_parks_on_a_road_the_opponents_use(self):
        """合流先がない補給車は、敵がよく踏む道路の上で待つこと。

        補給車は燃料を使わず、うどんも獲得できないので、待つ場所を変えるだけなら
        自分の損はゼロ。そこに居座ると翌日その道を使う敵だけが遅くなる。
        """

        from hexa.ai import Policy
        from hexa.rules import AgentKind, Terrain

        problem = generate_problem(202, PRESETS["medium"])
        brain = new_population()[0]
        policy = Policy(brain, problem, random.Random(0), epsilon=0.0)
        finder = PathFinder(problem)

        # 巡回車を1体も置かない構成にして、合流先が存在しない状況を作る。
        kinds = tuple(AgentKind.SUPPLY for _ in range(problem.agent_count))
        roads = [
            position
            for position in range(problem.cell_count)
            if problem.terrain_at(position) == Terrain.ROAD
        ]
        table = finder.table(problem.agent_positions[0], {})
        reachable_roads = [
            position for position in roads
            if table.reachable(position) and table.steps[position] <= problem.day_steps[0]
        ]
        self.assertTrue(reachable_roads, "到達できる道路がない問題でした")
        target = reachable_roads[0]
        heat = {target: 9.0}

        decision = policy.choose_supply_target(
            agent_index=0, step=0, remaining=problem.day_steps[0],
            position=problem.agent_positions[0], kinds=kinds,
            positions=list(problem.agent_positions),
            fuels=[0] * problem.agent_count,
            plan_targets=[None] * problem.agent_count, supply_claims={},
            road_status={}, finder=finder, road_heat=heat,
        )
        self.assertEqual(decision.target, target)
        self.assertIsNone(decision.target_agent)

    def test_blocking_never_overrides_a_real_rendezvous(self):
        """補給できる相手がいるなら、妨害より補給を優先すること。"""

        from hexa.ai import Policy
        from hexa.rules import AgentKind, Terrain

        problem = generate_problem(203, PRESETS["medium"])
        brain = new_population()[0]
        policy = Policy(brain, problem, random.Random(0), epsilon=0.0)
        finder = PathFinder(problem)
        kinds = tuple(
            AgentKind.SUPPLY if index == problem.agent_count - 1 else AgentKind.TOURING
            for index in range(problem.agent_count)
        )
        roads = [
            position
            for position in range(problem.cell_count)
            if problem.terrain_at(position) == Terrain.ROAD
        ]
        heat = {roads[0]: 99.0}
        fuels = [problem.fuel_limit // 3] * problem.agent_count

        decision = policy.choose_supply_target(
            agent_index=problem.agent_count - 1, step=0,
            remaining=problem.day_steps[0],
            position=problem.agent_positions[-1], kinds=kinds,
            positions=list(problem.agent_positions), fuels=fuels,
            plan_targets=[None] * problem.agent_count, supply_claims={},
            road_status={}, finder=finder, road_heat=heat,
        )
        self.assertIsNotNone(decision.target_agent, "妨害を優先してしまっています")


class CapacityGrowthTest(unittest.TestCase):
    """AIを大きくしても判断が変わらないことの証明テスト。

    新しく増える重みはすべて0から始まる。掛け算の項に0を掛ければ必ず0なので、
    広げた直後の予測値は数学的にまったく同じになる。
    「大きくしたせいで弱くなる」が原理的に起きないことを、
    予測値と実際の行動計画の両方で確かめる。
    """

    def _trained_brain(self):
        from hexa.ai import Brain

        rng = random.Random(1)
        brain = Brain.new("grow", 5, capacity="simple")
        for head in brain.touring_heads:
            for index in range(len(head)):
                head[index] = rng.uniform(-1.0, 1.0)
        for index in range(len(brain.supply_weights)):
            brain.supply_weights[index] = rng.uniform(-1.0, 1.0)
        return brain

    def test_predictions_are_identical_after_growing(self):
        from hexa.ai import SUPPLY_BASE_COUNT, TOURING_BASE_COUNT

        rng = random.Random(2)
        brain = self._trained_brain()
        touring = [[rng.uniform(0, 1) for _ in range(TOURING_BASE_COUNT)] for _ in range(300)]
        supply = [[rng.uniform(0, 1) for _ in range(SUPPLY_BASE_COUNT)] for _ in range(300)]
        before_t = [brain.predict_touring(x) for x in touring]
        before_s = [brain.predict_supply(x) for x in supply]

        brain.grow("cross")
        brain.grow("large")

        for original, grown in zip(before_t, (brain.predict_touring(x) for x in touring)):
            for head in range(3):
                self.assertEqual(original[head], grown[head])
        for original, grown in zip(before_s, (brain.predict_supply(x) for x in supply)):
            self.assertEqual(original, grown)

    def test_new_weights_start_at_zero(self):
        from hexa.ai import SUPPLY_BASE_COUNT, TOURING_BASE_COUNT

        brain = self._trained_brain()
        brain.grow("large")
        for head in brain.touring_heads:
            self.assertTrue(all(value == 0.0 for value in head[TOURING_BASE_COUNT:]))
        self.assertTrue(
            all(value == 0.0 for value in brain.supply_weights[SUPPLY_BASE_COUNT:])
        )

    def test_base_weights_survive_growing(self):
        from hexa.ai import TOURING_BASE_COUNT

        brain = self._trained_brain()
        original = [list(head[:TOURING_BASE_COUNT]) for head in brain.touring_heads]
        brain.grow("large")
        for head_index, head in enumerate(brain.touring_heads):
            self.assertEqual(head[:TOURING_BASE_COUNT], original[head_index])

    def test_action_plans_are_identical_after_growing(self):
        import copy

        from hexa.ai import new_population

        brains = new_population(count=4, capacity="simple")
        warmup = generate_problem(5, PRESETS["small"])
        for seed in range(8):
            result = run_match(warmup, brains, seed=seed, epsilon=0.3, kind_explore=0.5)
            for index, team in enumerate(result.teams):
                for features, rewards in team.touring_samples:
                    brains[index].update_touring(features, rewards, 0.05)
                for features, reward in team.supply_samples:
                    brains[index].update_supply(features, reward, 0.05)

        grown = copy.deepcopy(brains)
        for brain in grown:
            brain.grow("large")

        problem = generate_problem(42, PRESETS["medium"])
        before = run_match(problem, brains, seed=99)
        after = run_match(problem, grown, seed=99)
        for team_before, team_after in zip(before.teams, after.teams):
            self.assertEqual(
                team_before.action_plans_by_day, team_after.action_plans_by_day
            )
            self.assertEqual(team_before.score.key, team_after.score.key)

    def test_cannot_shrink(self):
        from hexa.ai import Brain

        brain = Brain.new("big", 1, capacity="large")
        with self.assertRaises(ValueError):
            brain.grow("simple")

    def test_growing_survives_save_and_load(self):
        import tempfile
        from pathlib import Path as P

        from hexa.ai import TOURING_BASE_COUNT, load_population, save_population

        brain = self._trained_brain()
        rng = random.Random(3)
        sample = [rng.uniform(0, 1) for _ in range(TOURING_BASE_COUNT)]
        before = brain.predict_touring(sample)
        brain.grow("large")
        with tempfile.TemporaryDirectory() as directory:
            path = P(directory) / "p.json"
            save_population([brain], path)
            loaded = load_population(path)[0]
        self.assertEqual(loaded.touring_cross, brain.touring_cross)
        self.assertEqual(loaded.predict_touring(sample), before)


class MapVarietyTest(unittest.TestCase):
    def test_every_archetype_produces_valid_problems(self):
        from hexa.generator import ARCHETYPES, random_settings

        for name in ARCHETYPES:
            for seed in range(4):
                settings = random_settings(random.Random(seed), 9, archetype=name)
                problem = generate_problem(seed * 13 + 1, settings)
                self.assertEqual(validate_problem(problem), [], f"{name} seed={seed}")

    def test_archetypes_really_differ(self):
        """型ごとに地形の割合がはっきり変わること。"""

        from hexa.generator import random_settings
        from hexa.rules import Terrain

        def ratio(name, terrain):
            settings = random_settings(random.Random(7), 9, archetype=name)
            problem = generate_problem(99, settings)
            return sum(1 for c in problem.cells if c == int(terrain)) / problem.cell_count

        self.assertGreater(ratio("street", Terrain.ROAD), ratio("plain", Terrain.ROAD) * 2)
        self.assertGreater(
            ratio("mountain", Terrain.MOUNTAIN), ratio("plain", Terrain.MOUNTAIN) * 2
        )
        self.assertGreater(ratio("islands", Terrain.POND), ratio("street", Terrain.POND) * 2)

    def test_random_settings_cover_the_whole_official_range(self):
        from hexa.generator import random_settings

        rng = random.Random(5)
        heights = set()
        for _ in range(200):
            settings = random_settings(rng, 9)
            heights.add(settings.height)
        self.assertLessEqual(min(heights), 10)
        self.assertGreaterEqual(max(heights), 30)

    def test_generation_never_fails_across_the_range(self):
        from hexa.generator import random_settings

        rng = random.Random(9)
        for _ in range(60):
            settings = random_settings(rng, 9)
            problem = generate_problem(rng.randrange(1, 10 ** 9), settings)
            self.assertEqual(validate_problem(problem), [])


class AutoTrainTest(unittest.TestCase):
    """おまかせ学習の部品のテスト。"""

    def test_archetype_validation_covers_every_type(self):
        from hexa.generator import ARCHETYPES
        from hexa.training import build_archetype_validation

        problems = build_archetype_validation(2, 4)
        self.assertEqual(set(problems), set(ARCHETYPES))
        for name, items in problems.items():
            self.assertEqual(len(items), 2)
            for problem in items:
                self.assertEqual(validate_problem(problem), [], name)

    def test_bias_favours_the_weakest_archetype(self):
        from hexa.training import _bias_from_scores

        scores = {
            "street": (5.0, 30.0, 100.0),
            "mountain": (5.0, 30.0, 50.0),
            "plain": (5.0, 30.0, 100.0),
        }
        bias = _bias_from_scores(scores)
        self.assertGreater(bias["mountain"], bias["street"])
        self.assertAlmostEqual(bias["street"], bias["plain"])
        self.assertGreaterEqual(min(bias.values()), 1.0)

    def test_bias_is_flat_when_all_types_are_equal(self):
        from hexa.training import _bias_from_scores

        scores = {name: (5.0, 30.0, 100.0) for name in ("a", "b", "c")}
        bias = _bias_from_scores(scores)
        self.assertEqual(set(round(v, 6) for v in bias.values()), {1.0})

    def test_weighted_pick_follows_the_bias(self):
        from hexa.training import _pick_archetype

        rng = random.Random(0)
        counts = {}
        bias = {"mountain": 9.0, "street": 1.0}
        for _ in range(500):
            name = _pick_archetype(rng, bias)
            counts[name] = counts.get(name, 0) + 1
        self.assertGreater(counts.get("mountain", 0), counts.get("street", 0) * 3)

    def test_capacity_is_reported_correctly(self):
        from hexa.ai import Brain
        from hexa.training import current_capacity

        for name in ("simple", "cross", "large"):
            self.assertEqual(current_capacity(Brain.new("x", 1, capacity=name)), name)


class LongRunTest(unittest.TestCase):
    """何十時間も回すときにだけ出る問題の回帰テスト。"""

    def test_rounds_do_not_repeat_the_same_problems(self):
        """ラウンドごとに種を変えないと、同じ問題列を延々と解き直すことになる。"""

        from hexa.training import _curriculum_settings

        def sequence(seed):
            rng = random.Random(seed)
            return [
                _curriculum_settings(
                    index / 9, ("small", "medium", "large"), 9, rng, 0.6
                ).describe()
                for index in range(12)
            ]

        first = sequence(20260910 + 1 * 1_000_003)
        second = sequence(20260910 + 2 * 1_000_003)
        self.assertNotEqual(first, second)

    def test_progress_spans_the_whole_run_not_one_round(self):
        """小分けに回しても、探索率と学習率が全体を通して進むこと。"""

        from hexa.training import TrainingSettings, _global_progress

        early = TrainingSettings(episodes=2000, progress_offset=0, progress_total=100000)
        late = TrainingSettings(episodes=2000, progress_offset=98000, progress_total=100000)
        self.assertLess(_global_progress(early, 1000), 0.05)
        self.assertGreater(_global_progress(late, 1000), 0.95)

    def test_progress_falls_back_to_the_round_when_total_is_unknown(self):
        from hexa.training import TrainingSettings, _global_progress

        settings = TrainingSettings(episodes=100)
        self.assertAlmostEqual(_global_progress(settings, 99), 1.0)

    def test_broken_weights_are_detected(self):
        from hexa.ai import Brain
        from hexa.training import _weights_are_finite

        brain = Brain.new("x", 1)
        self.assertTrue(_weights_are_finite(brain))
        brain.touring_heads[0][0] = float("nan")
        self.assertFalse(_weights_are_finite(brain))
        brain.touring_heads[0][0] = float("inf")
        self.assertFalse(_weights_are_finite(brain))
        brain.touring_heads[0][0] = 1e9
        self.assertFalse(_weights_are_finite(brain))

    def test_history_grows_so_menu_five_can_show_progress(self):
        from hexa.ai import Brain

        brain = Brain.new("h", 1)
        self.assertEqual(brain.history, [])
        brain.history.append({"episodes": 1, "avgBrand": 1.0,
                              "avgDailyBrand": 1.0, "avgNoodles": 1.0})
        payload = brain.to_json_dict()
        restored = Brain.from_json_dict(payload)
        self.assertEqual(len(restored.history), 1)

    def test_kind_stats_stay_bounded(self):
        """何百万回回しても、統計のキーが増え続けないこと。"""

        from hexa.ai import Brain

        brain = Brain.new("k", 1)
        rng = random.Random(0)
        for _ in range(20000):
            agents = rng.randint(3, 8)
            brain.record_kind_result(agents, rng.randint(0, agents - 1), (1, 1, 1))
        self.assertLessEqual(len(brain.kind_stats), 8 * 8)


class ValidationCoverageTest(unittest.TestCase):
    """検証が本選の3条件すべてを見ていることの確認。

    以前は自作プリセット（8x8〜32x32）で測っていたが、
    系列数が本選の1/2〜1/3しかなく（1系列3店 対 本選1.1〜1.6店）、
    「取り逃してもリカバリが効く」別のゲームで判断していた。
    いまは本選の3条件（1回戦・準決勝・決勝）をそのまま使う。
    """

    def test_default_covers_every_round(self):
        from hexa.generator import HONSEN_ROUNDS
        from hexa.training import build_archetype_validation, validation_sizes

        self.assertEqual(set(validation_sizes(3)), set(HONSEN_ROUNDS))
        problems = build_archetype_validation(3, 10)
        sizes = {(p.height, p.agent_count) for items in problems.values() for p in items}
        self.assertIn((16, 4), sizes)
        self.assertIn((24, 5), sizes)
        self.assertIn((32, 7), sizes)

    def test_every_archetype_gets_every_round(self):
        from hexa.generator import ARCHETYPES, HONSEN_ROUNDS

        from hexa.training import build_archetype_validation

        problems = build_archetype_validation(3, 10)
        self.assertEqual(set(problems), set(ARCHETYPES))
        for name, items in problems.items():
            sizes = {(p.height, p.agent_count) for p in items}
            self.assertEqual(len(sizes), len(HONSEN_ROUNDS), f"{name} の条件が揃っていません")

    def test_brand_counts_match_the_official_ranges(self):
        """系列数が本選の指定どおりであること。ここが本選との一番の違いだった。"""

        from hexa.generator import HONSEN_ROUNDS
        from hexa.training import build_archetype_validation

        problems = build_archetype_validation(3, 10)
        by_side = {spec["side"]: spec["brands"] for spec in HONSEN_ROUNDS.values()}
        for items in problems.values():
            for problem in items:
                low, high = by_side[problem.height]
                brands = len({spot.brand for spot in problem.spots})
                self.assertGreaterEqual(brands, low, f"{problem.height}x{problem.height}")
                self.assertLessEqual(brands, high, f"{problem.height}x{problem.height}")

    def test_small_setting_is_flagged_as_incomplete(self):
        from hexa.training import validation_sizes

        self.assertNotIn("決勝", validation_sizes(2))

    def test_validation_problems_are_all_legal(self):
        from hexa.training import build_archetype_validation

        problems = build_archetype_validation(3, 10)
        for name, items in problems.items():
            for problem in items:
                self.assertEqual(validate_problem(problem), [], name)


class MapAwarenessTest(unittest.TestCase):
    """AIがマップの大きさと地形を区別できることの確認。

    以前は材料がすべて「マップ基準の割合」に正規化されていたため、
    8x8 と 32x32 の区別がつかず、「大きいマップではこう戦う」という
    場面に応じた戦い方を学習できなかった。
    """

    def _map_features(self, problem):
        from hexa.ai import TOURING_BASE_COUNT, Policy, new_population

        brain = new_population(count=2)[0]
        policy = Policy(brain, problem, random.Random(0), 0.0)
        finder = PathFinder(problem)
        decision = policy.choose_touring_target(
            agent_index=0, step=0, remaining=problem.day_steps[0],
            position=problem.agent_positions[0], fuel=problem.fuel_limit,
            stock={spot.position: spot.stocks for spot in problem.spots},
            reserved={}, acquired_today=set(), acquired_brands=set(),
            brands_today=set(), road_status={}, finder=finder,
            kinds=(), positions=list(problem.agent_positions),
            day_index=0, day_count=problem.day_count,
        )
        return decision.candidate_features[0][31:TOURING_BASE_COUNT]

    def test_small_and_large_maps_look_different(self):
        from hexa.generator import GeneratorSettings

        def build(side):
            return generate_problem(2026, GeneratorSettings(
                height=side, width=side, agent_count=6, spot_count=8,
                brand_count=4, day_count=6, players=9,
            ))

        small = self._map_features(build(8))
        big = self._map_features(build(32))
        self.assertNotEqual(small, big)
        # 広さの材料は 8x8 で0、32x32 で1になる
        self.assertAlmostEqual(small[0], 0.0, places=6)
        self.assertAlmostEqual(big[0], 1.0, places=6)

    def test_terrain_mix_is_visible(self):
        from hexa.generator import random_settings

        features = {}
        for name in ("street", "mountain", "islands", "plain"):
            settings = random_settings(random.Random(1), 9, archetype=name)
            features[name] = self._map_features(generate_problem(2026, settings))
        # 池・道路・山地はそれぞれ 25・26・27 番目（0起点で 2・3・4）
        self.assertGreater(features["street"][3], features["plain"][3] * 2)
        self.assertGreater(features["mountain"][4], features["plain"][4] * 2)
        self.assertGreater(features["islands"][2], features["street"][2] * 2)

    def test_fuel_tightness_is_visible(self):
        import dataclasses

        from hexa.generator import GeneratorSettings

        base = generate_problem(2026, GeneratorSettings(
            height=12, width=12, agent_count=6, spot_count=8,
            brand_count=4, day_count=6, players=9,
        ))
        tight = dataclasses.replace(base, fuel_limit=base.day_steps[0])
        loose = dataclasses.replace(base, fuel_limit=base.day_steps[0] * 3)
        self.assertLess(self._map_features(tight)[5], self._map_features(loose)[5])

    def test_jam_tendency_is_visible(self):
        import dataclasses

        from hexa.generator import GeneratorSettings

        base = generate_problem(2026, GeneratorSettings(
            height=12, width=12, agent_count=6, spot_count=8,
            brand_count=4, day_count=6, players=9,
        ))
        easy = dataclasses.replace(base, busy_threshold=1, jammed_threshold=2)
        hard = dataclasses.replace(base, busy_threshold=5, jammed_threshold=10)
        self.assertGreater(self._map_features(easy)[6], self._map_features(hard)[6])

    def test_map_features_are_bounded(self):
        from hexa.generator import random_settings

        rng = random.Random(3)
        for _ in range(30):
            settings = random_settings(rng, rng.randint(2, 10))
            problem = generate_problem(rng.randrange(1, 10 ** 9), settings)
            for value in self._map_features(problem):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class MemoryGuardTest(unittest.TestCase):
    """メモリ不足でプロセスごと殺された事故の回帰テスト。

    経路表のキャッシュを「表の個数」で制限していたため、32x32 のマップでは
    1プロセスで610MB、64並列で38GBに達し、OOM Killer に殺されていた。
    マップのセル数に応じて貯める数を減らすようにしてある。
    """

    def test_cache_limit_shrinks_on_large_maps(self):
        from hexa.generator import GeneratorSettings

        def limit(side):
            problem = generate_problem(2026, GeneratorSettings(
                height=side, width=side, agent_count=8,
                spot_count=min(32, side), brand_count=4, day_count=6, players=10,
            ))
            return PathFinder(problem)._cache_limit

        small, big = limit(8), limit(32)
        self.assertGreater(small, big * 4)

    def test_cache_budget_is_bounded_by_cells(self):
        from hexa.generator import GeneratorSettings
        from hexa.pathfind import CACHE_CELL_BUDGET

        for side in (8, 12, 20, 32):
            problem = generate_problem(2026, GeneratorSettings(
                height=side, width=side, agent_count=8,
                spot_count=min(32, side), brand_count=4, day_count=6, players=10,
            ))
            finder = PathFinder(problem)
            cells = finder._cache_limit * problem.cell_count
            self.assertLessEqual(cells, CACHE_CELL_BUDGET * 1.05, f"{side}x{side}")

    def test_cache_never_goes_below_a_usable_size(self):
        from hexa.generator import GeneratorSettings

        problem = generate_problem(2026, GeneratorSettings(
            height=32, width=32, agent_count=8, spot_count=32,
            brand_count=4, day_count=6, players=10,
        ))
        self.assertGreaterEqual(PathFinder(problem)._cache_limit, 64)

    def test_cache_still_avoids_recomputation(self):
        problem = generate_problem(41, PRESETS["medium"])
        finder = PathFinder(problem)
        for _ in range(50):
            finder.table(problem.agent_positions[0], {})
        self.assertEqual(finder.compute_count, 1)


class HonsenConditionTest(unittest.TestCase):
    """本選の公開条件どおりの問題が作れることの確認。

    出典: 本選マップに関するお知らせ（参考マップ sample_16 / 24 / 32 を含む）
      1回戦・敗者復活戦 16x16 / 4体 / 16店 / 系列10-14 / タイプ選択60秒
      準決勝           24x24 / 5体 / 24店 / 系列14-20 / タイプ選択90秒
      決勝             32x32 / 7体 / 32店 / 系列20-28 / タイプ選択120秒
    """

    def test_every_round_matches_the_announcement(self):
        from hexa.generator import HONSEN_ROUNDS, generate_problem, honsen_settings

        for name, spec in HONSEN_ROUNDS.items():
            for seed in range(4):
                settings = honsen_settings(name, random.Random(seed), players=10)
                problem = generate_problem(seed * 31 + 1, settings)
                self.assertEqual(validate_problem(problem), [], name)
                self.assertEqual(problem.height, spec["side"], name)
                self.assertEqual(problem.width, spec["side"], name)
                self.assertEqual(problem.agent_count, spec["agents"], name)
                self.assertEqual(len(problem.spots), spec["spots"], name)
                brands = len({spot.brand for spot in problem.spots})
                self.assertGreaterEqual(brands, spec["brands"][0], name)
                self.assertLessEqual(brands, spec["brands"][1], name)

    def test_archetype_ranges_actually_apply(self):
        """型ごとの燃料と基準値が本当に効いていること。

        以前は生成側が型の指定を無視しており、どの型でも
        燃料1〜3倍・混雑1〜5・渋滞2〜10 のままだった。
        """

        from hexa.generator import ARCHETYPES, generate_problem, random_settings

        rng = random.Random(4)
        for name in ("plain", "honsen"):
            shape = ARCHETYPES[name]
            for _ in range(8):
                settings = random_settings(rng, 10, archetype=name)
                problem = generate_problem(rng.randrange(1, 10 ** 9), settings)
                ratio = problem.fuel_limit / problem.day_steps[0]
                self.assertGreaterEqual(ratio, shape["fuel"][0] - 0.06, name)
                self.assertLessEqual(ratio, shape["fuel"][1] + 0.06, name)
                self.assertGreaterEqual(problem.busy_threshold, shape["busy"][0], name)
                self.assertLessEqual(problem.busy_threshold, shape["busy"][1], name)
                self.assertLessEqual(problem.jammed_threshold, shape["jammed"][1], name)

    def test_day_steps_stay_inside_the_official_range(self):
        """各日のステップ数が公式の範囲（W+H の1〜4倍）に収まること。

        参考マップは1.0〜1.5倍と下限寄りだったが、実際に配られた例では
        1.09〜3.91倍と大きく振れていた（125, 108, 106, 108, 35, 82, 37, 78, 107）。
        どちらも出せるようにしてある。
        """

        from hexa.generator import HONSEN_ROUNDS, generate_problem, honsen_settings

        for name in HONSEN_ROUNDS:
            settings = honsen_settings(name, random.Random(7), players=10)
            problem = generate_problem(99, settings)
            span = problem.width + problem.height
            for steps in problem.day_steps:
                self.assertGreaterEqual(steps, span, name)
                self.assertLessEqual(steps, span * 4, name)

    def test_type_selection_fits_in_the_official_time(self):
        """タイプ選択が制限時間に必ず収まること。

        以前は時間制限がなく、32x32 では実測124秒かかって120秒を超えていた。
        超えると種別を提出できず、全エージェントが巡回車になる。
        """

        import time

        from hexa.ai import new_population
        from hexa.generator import HONSEN_ROUNDS, generate_problem, honsen_settings
        from hexa.production import choose_production_kinds, kind_time_budget

        brains = new_population(count=10, capacity="simple")
        for name, spec in HONSEN_ROUNDS.items():
            settings = honsen_settings(name, random.Random(3), players=10, day_count=10)
            problem = generate_problem(2026, settings)
            budget = kind_time_budget(problem)
            self.assertLess(budget, spec["kind_seconds"], name)
            started = time.monotonic()
            kinds = choose_production_kinds(
                problem, brains, budget_seconds=budget, log=lambda message: None
            )
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, spec["kind_seconds"], f"{name} が制限時間を超えました")
            self.assertEqual(len(kinds), problem.agent_count, name)

    def test_at_least_one_touring_car_always_remains(self):
        from hexa.ai import new_population
        from hexa.generator import generate_problem, honsen_settings
        from hexa.production import choose_production_kinds
        from hexa.rules import AgentKind

        brains = new_population(count=4, capacity="simple")
        settings = honsen_settings("1回戦", random.Random(5), players=10)
        problem = generate_problem(11, settings)
        kinds = choose_production_kinds(problem, brains, log=lambda message: None)
        self.assertGreaterEqual(sum(1 for k in kinds if k == AgentKind.TOURING), 1)


class SampleMapTest(unittest.TestCase):
    """公式が公開した参考マップが読めることの確認。

    server/ と同じく配布物なので同梱していない。problems/sample/ に
    置かれているときだけ走る。
    """

    SAMPLE_DIR = Path(__file__).resolve().parent.parent / "problems" / "sample"

    @unittest.skipIf(
        not (Path(__file__).resolve().parent.parent / "problems" / "sample").exists(),
        "problems/sample/ に参考マップが置かれていません",
    )
    def test_sample_maps_pass_validation(self):
        from hexa.problem import load_problem

        found = sorted(self.SAMPLE_DIR.glob("sample_*.json"))
        self.assertTrue(found, "sample_*.json が見つかりません")
        for path in found:
            problem = load_problem(path)
            self.assertEqual(validate_problem(problem), [], path.name)


class ConnectionTest(unittest.TestCase):
    """本番の通信まわりのテスト。

    公式の進行では「試合開始と同時に問題データが取得可能になります」。
    つまり試合開始前にアクセスするとエラーが返ります。以前は
    play_production_match がいきなり get_setting を呼んでいたので、
    早めに接続すると例外で落ちていました。
    """

    def _serve(self, handler_factory):
        import threading
        from http.server import HTTPServer

        server = HTTPServer(("127.0.0.1", 0), handler_factory)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def _handler(self, state):
        import json
        from http.server import BaseHTTPRequestHandler

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if state["ready"]:
                    body = json.dumps(state["setting"]).encode()
                    self.send_response(200)
                else:
                    body = json.dumps(
                        {"code": "AccessTimeError", "message": "まだ試合前です"}
                    ).encode()
                    self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def _setting(self):
        problem = generate_problem(5, PRESETS["small"])
        raw = problem.to_json_dict()
        raw["startsAt"] = 1_778_227_200
        return raw

    def test_connection_check_succeeds_before_the_match_starts(self):
        """試合前でも「届いている」と判定すること。事前確認に使うため。"""

        from hexa.client import ProconClient

        state = {"ready": False, "setting": self._setting()}
        url = self._serve(self._handler(state))
        ok, message = ProconClient(url, "token").check_connection()
        self.assertTrue(ok, message)
        self.assertIn("試合前", message)

    def test_connection_check_reports_unreachable(self):
        from hexa.client import ProconClient

        ok, message = ProconClient("http://127.0.0.1:1", "token").check_connection()
        self.assertFalse(ok)
        self.assertIn("届きませんでした", message)

    def test_waiting_returns_once_the_match_opens(self):
        """試合開始前に呼んでも落ちず、開いた瞬間に取れること。"""

        import threading

        from hexa.client import ProconClient

        state = {"ready": False, "setting": self._setting()}
        url = self._serve(self._handler(state))
        threading.Timer(0.6, lambda: state.__setitem__("ready", True)).start()

        client = ProconClient(url, "token")
        problem, raw = client.wait_for_setting(poll_interval=0.2, timeout=20.0)
        self.assertEqual(problem.height, generate_problem(5, PRESETS["small"]).height)
        self.assertNotEqual(int(raw["startsAt"]), 0)

    def test_waiting_gives_up_eventually(self):
        from hexa.client import ProconClient

        state = {"ready": False, "setting": self._setting()}
        url = self._serve(self._handler(state))
        client = ProconClient(url, "token")
        with self.assertRaises(TimeoutError):
            client.wait_for_setting(poll_interval=0.2, timeout=1.0)


class LanServerTest(unittest.TestCase):
    def test_local_ip_is_usable(self):
        from hexa.localserver import local_ip

        address = local_ip()
        self.assertTrue(address)
        self.assertEqual(address.count("."), 3)


class TerrainStructureTest(unittest.TestCase):
    """地形が「かたまり」になっていることの確認。

    公開された参考マップは、道路も山地も明らかに固まっている
    （sample_32 は道路の出現率24%に対し、隣が道路の割合が36%）。
    以前の生成器は完全なランダム散布だったので、道路が1マスずつ孤立し、
    道路をつないで速く走ることができなかった。本選とは別のゲームになる。
    """

    def _excess(self, problem):
        """出現率に対して「隣が同じ地形」がどれだけ多いか。0なら完全にばらばら。"""

        from hexa.hexgrid import neighbor_table
        from hexa.rules import Terrain

        table = neighbor_table(problem.width, problem.height)
        same: dict[int, int] = {}
        total: dict[int, int] = {}
        for position in range(problem.cell_count):
            terrain = problem.cells[position]
            for _direction, neighbour in table[position]:
                total[terrain] = total.get(terrain, 0) + 1
                if problem.cells[neighbour] == terrain:
                    same[terrain] = same.get(terrain, 0) + 1
        result = {}
        for terrain in total:
            share = sum(1 for c in problem.cells if c == terrain) / problem.cell_count
            result[Terrain(terrain).name] = same.get(terrain, 0) / total[terrain] - share
        return result

    def test_terrain_clusters_like_the_official_maps(self):
        from hexa.generator import generate_problem, honsen_settings

        for name in ("1回戦", "準決勝", "決勝"):
            settings = honsen_settings(name, random.Random(5), players=10, day_count=5)
            problem = generate_problem(2026, settings)
            excess = self._excess(problem)
            self.assertGreater(excess["ROAD"], 0.02, f"{name}: 道路が固まっていません")
            self.assertGreater(excess["MOUNTAIN"], 0.02, f"{name}: 山地が固まっていません")

    def test_clump_zero_means_scattered(self):
        import dataclasses

        from hexa.generator import generate_problem, honsen_settings

        settings = honsen_settings("決勝", random.Random(5), players=10, day_count=5)
        scattered = dataclasses.replace(settings, clump=0.0)
        clustered = dataclasses.replace(settings, clump=0.25)
        low = self._excess(generate_problem(7, scattered))["ROAD"]
        high = self._excess(generate_problem(7, clustered))["ROAD"]
        self.assertLess(low, high)

    def test_clustered_maps_stay_legal(self):
        from hexa.generator import ARCHETYPES, generate_problem, random_settings

        rng = random.Random(11)
        for name in ARCHETYPES:
            for _ in range(3):
                settings = random_settings(rng, 10, archetype=name)
                problem = generate_problem(rng.randrange(1, 10 ** 9), settings)
                self.assertEqual(validate_problem(problem), [], name)

    def test_roads_form_usable_runs(self):
        """道路が2マス以上つながっていること。1マスずつ孤立していると意味がない。"""

        from hexa.generator import generate_problem, honsen_settings
        from hexa.hexgrid import neighbor_table
        from hexa.rules import Terrain

        settings = honsen_settings("決勝", random.Random(5), players=10, day_count=5)
        problem = generate_problem(2026, settings)
        table = neighbor_table(problem.width, problem.height)
        roads = [p for p in range(problem.cell_count) if problem.cells[p] == int(Terrain.ROAD)]
        connected = sum(
            1 for position in roads
            if any(problem.cells[nb] == int(Terrain.ROAD) for _d, nb in table[position])
        )
        self.assertGreater(connected / max(1, len(roads)), 0.5)


class ArenaMapTest(unittest.TestCase):
    """実際に配られた例のマップが扱えることの確認。

    参考マップより地形の塊が大きく（山地+0.56 / 池+0.60）、
    各日のステップ数も W+H の1.09〜3.91倍と大きく振れていた。
    """

    PATH = Path(__file__).resolve().parent.parent / "problems" / "sample" / "arena_example_16.json"

    @unittest.skipIf(
        not (Path(__file__).resolve().parent.parent / "problems" / "sample"
             / "arena_example_16.json").exists(),
        "実例マップが置かれていません",
    )
    def test_example_map_is_legal_and_solvable(self):
        from hexa.ai import new_population
        from hexa.problem import load_problem

        problem = load_problem(self.PATH)
        self.assertEqual(validate_problem(problem), [])
        brains = new_population(count=10, capacity="simple")
        result = run_match(problem, brains[:problem.players], seed=1)
        self.assertEqual(len(result.teams), problem.players)
        best = max(result.teams, key=lambda team: team.score.key)
        self.assertGreater(best.score.brand_count, 0)

    def test_generator_can_reach_that_level_of_clustering(self):
        """一枚岩の山や湖を作れること。以前は +0.19 が上限だった。"""

        import dataclasses

        from hexa.generator import generate_problem, honsen_settings
        from hexa.hexgrid import neighbor_table

        settings = honsen_settings("1回戦", random.Random(3), players=10)
        settings = dataclasses.replace(settings, clump=0.55)
        problem = generate_problem(2026, settings)
        table = neighbor_table(problem.width, problem.height)
        same = {}
        total = {}
        for position in range(problem.cell_count):
            terrain = problem.cells[position]
            for _direction, neighbour in table[position]:
                total[terrain] = total.get(terrain, 0) + 1
                if problem.cells[neighbour] == terrain:
                    same[terrain] = same.get(terrain, 0) + 1
        best = max(
            same.get(t, 0) / total[t]
            - sum(1 for c in problem.cells if c == t) / problem.cell_count
            for t in total
        )
        self.assertGreater(best, 0.3)

    def test_day_steps_span_the_whole_official_range(self):
        """極端に短い日も長い日も出ること。"""

        from hexa.generator import generate_problem, honsen_settings

        rng = random.Random(9)
        ratios = []
        for _ in range(12):
            settings = honsen_settings("1回戦", rng, players=10)
            problem = generate_problem(rng.randrange(1, 10 ** 9), settings)
            span = problem.width + problem.height
            ratios.extend(steps / span for steps in problem.day_steps)
        self.assertLess(min(ratios), 1.4)
        self.assertGreater(max(ratios), 3.0)


class MultiTrainTest(unittest.TestCase):
    def test_sets_merge_even_with_mixed_capacities(self):
        """セットごとに大きさが違っても1つにまとめられること。"""

        import tempfile

        from hexa.ai import load_population, new_population, save_population
        from hexa.multitrain import merge_sets, set_dir

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, capacity in enumerate(("simple", "cross", "large")):
                folder = set_dir(root, index)
                folder.mkdir(parents=True)
                save_population(
                    new_population(seed=index * 977, count=4, capacity=capacity),
                    folder / "population.json",
                )
            count = merge_sets(root, log=lambda message: None)
            self.assertEqual(count, 12)
            merged = load_population(root / "population.json")
            self.assertEqual(len(merged), 12)
            self.assertEqual(len({brain.name for brain in merged}), 12)
            widths = {len(brain.touring_heads[0]) for brain in merged}
            self.assertEqual(len(widths), 3, "大きさが混ざっていません")

    def test_merged_population_can_play(self):
        import tempfile

        from hexa.ai import load_population, new_population, save_population
        from hexa.multitrain import merge_sets, set_dir

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, capacity in enumerate(("simple", "large")):
                folder = set_dir(root, index)
                folder.mkdir(parents=True)
                save_population(
                    new_population(seed=index * 31, count=5, capacity=capacity),
                    folder / "population.json",
                )
            merge_sets(root, log=lambda message: None)
            brains = load_population(root / "population.json")
            problem = generate_problem(42, PRESETS["small"])
            result = run_match(problem, brains[:4], seed=1)
            self.assertEqual(len(result.teams), 4)


class ChokepointTest(unittest.TestCase):
    """道路の幅が1マスしかない場所（急所）を扱えることの確認。

    実戦のマップで、道路を1セット渋滞させるだけで全スポットへの
    合計ステップが19伸びる地形があった（1日43ステップの試合で）。
    全道路が渋滞すると272→441（+62%）。
    こういう場所に頼った計画は、翌日ひっくり返される。
    """

    def _problem(self):
        from hexa.generator import generate_problem, honsen_settings

        settings = honsen_settings("1回戦", random.Random(11), players=10, day_count=4)
        return generate_problem(4242, settings)

    def test_jammed_roads_are_routed_around_correctly(self):
        """渋滞した道路は、実際の状態どおりに避けて通ること。"""

        from hexa.rules import RoadStatus, Terrain

        problem = self._problem()
        finder = PathFinder(problem)
        roads = [
            cell for cell in range(problem.cell_count)
            if problem.terrain_at(cell) == Terrain.ROAD
        ]
        start = problem.agent_positions[0]
        clear = finder.table(start, {})
        jammed = finder.table(start, {cell: RoadStatus.JAMMED for cell in roads})
        # 渋滞していれば、どこかの目的地は必ず遠くなる（同じか長くなる）
        worse = sum(
            1 for spot in problem.spots
            if clear.reachable(spot.position) and jammed.reachable(spot.position)
            and jammed.steps[spot.position] > clear.steps[spot.position]
        )
        self.assertGreater(worse, 0, "渋滞しても経路が1つも変わっていません")
        for spot in problem.spots:
            if clear.reachable(spot.position) and jammed.reachable(spot.position):
                self.assertGreaterEqual(
                    jammed.steps[spot.position], clear.steps[spot.position]
                )

    def test_route_reports_its_fragility(self):
        """急所に頼った道すじは、遠回りの大きさとして表に出ること。"""

        from hexa.routes import build_routes, default_recipes

        problem = self._problem()
        finder = PathFinder(problem)
        available = {spot.position: spot.stocks for spot in problem.spots}
        routes = build_routes(
            problem, finder, {}, problem.agent_positions[0],
            problem.fuel_limit, problem.day_steps[0],
            available, set(), set(), default_recipes(),
        )
        self.assertTrue(routes)
        for route in routes:
            self.assertGreaterEqual(route.detour, 0)
        # 作戦によって脆さが違うこと（全部同じなら材料として働かない）
        self.assertGreater(len({route.detour for route in routes}), 1)

    def test_fragility_reaches_the_features(self):
        from hexa.ai import TOURING_BASE_COUNT, Policy, new_population

        problem = self._problem()
        brain = new_population(count=2)[0]
        policy = Policy(brain, problem, random.Random(0), 0.0)
        finder = PathFinder(problem)
        decision = policy.choose_touring_target(
            agent_index=0, step=0, remaining=problem.day_steps[0],
            position=problem.agent_positions[0], fuel=problem.fuel_limit,
            stock={spot.position: spot.stocks for spot in problem.spots},
            reserved={}, acquired_today=set(), acquired_brands=set(),
            brands_today=set(), road_status={}, finder=finder,
            kinds=(), positions=list(problem.agent_positions),
            day_index=0, day_count=problem.day_count,
        )
        # 28番が「渋滞したときの遠回り」。候補ごとに値が違うはず。
        values = {features[28] for features in decision.candidate_features}
        self.assertGreater(len(values), 1)
        for value in values:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class StrandedAgentTest(unittest.TestCase):
    """燃料切れで止まった巡回車が、その日ずっと動けなくなる不具合の回帰テスト。

    実戦で「7体中2体が5日間ずっと止まったまま、在庫は60以上残っている」
    という状態を確認した。原因は2つ。

      1. 行ける店が1つもないとき、その日の残り全部を待機に決めていた。
         途中で補給されても再検討しないので、動き出せなかった。
      2. 補給車が立ち往生した巡回車を最優先にしていなかった。
    """

    def test_no_options_means_a_short_wait_not_the_whole_day(self):
        from hexa.planner import _AgentPlanState, _wait_length
        from hexa.rules import AgentKind

        kinds = (AgentKind.TOURING,)
        state = _AgentPlanState()
        state.had_options = False
        waited = _wait_length(0, kinds, 120, [state])
        self.assertLess(waited, 120, "行ける店がないと一日じゅう固定されています")
        self.assertGreaterEqual(waited, 1)

    def test_supply_cars_prioritise_stranded_touring_cars(self):
        """燃料が尽きた巡回車がいるなら、そこへ行くことを最優先にすること。"""

        from hexa.ai import Policy, new_population
        from hexa.rules import AgentKind

        problem = generate_problem(207, PRESETS["medium"])
        brain = new_population()[0]
        policy = Policy(brain, problem, random.Random(0), epsilon=0.0)
        finder = PathFinder(problem)
        kinds = tuple(
            AgentKind.SUPPLY if index == problem.agent_count - 1 else AgentKind.TOURING
            for index in range(problem.agent_count)
        )
        # 0番だけ燃料0（立ち往生）、他は満タン
        fuels = [problem.fuel_limit] * problem.agent_count
        fuels[0] = 0
        decision = policy.choose_supply_target(
            agent_index=problem.agent_count - 1, step=0,
            remaining=problem.day_steps[0], position=problem.agent_positions[-1],
            kinds=kinds, positions=list(problem.agent_positions), fuels=fuels,
            plan_targets=[None] * problem.agent_count, supply_claims={},
            road_status={}, finder=finder,
        )
        self.assertEqual(decision.target_agent, 0, "立ち往生した巡回車を選んでいません")

    def test_touring_cars_do_not_idle_a_whole_match(self):
        """1試合を通して一度も動かない巡回車が出ないこと。"""

        from hexa.ai import new_population
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        brains = new_population(count=5, capacity="simple")
        settings = honsen_settings("決勝", random.Random(3), players=5, day_count=6)
        problem = gen(2026, settings)
        result = run_match(problem, brains, seed=1, record_trace=True)
        for team in result.teams:
            for index, kind in enumerate(team.agent_kinds):
                if kind != AgentKind.TOURING:
                    continue
                moved = any(
                    any(action >= 0 for action in plan[index])
                    for plan in team.action_plans_by_day
                )
                self.assertTrue(
                    moved, f"{team.brain_name} の巡回車{index}が一度も動いていません"
                )


class TacticPoolTest(unittest.TestCase):
    """作戦が増えても1回の判断が重くならないことの確認。

    手持ちが48個まで増えると1判断20ミリ秒（8個なら5ミリ秒）かかり、
    実戦で1日の候補が数千件から20件まで落ちていた。
    """

    def test_working_set_is_capped(self):
        from hexa.ai import new_population
        from hexa.routes import SEED_RECIPES, mutate_recipe

        brain = new_population(count=2)[0]
        rng = random.Random(0)
        while len(brain.recipes) < 48:
            brain.recipes.append(mutate_recipe(brain.recipes[0], rng))
        working = brain.working_recipes()
        self.assertEqual(len(working), 12)
        self.assertEqual(len(brain.recipes), 48, "手持ちが減っています")
        # 出発点の8つは必ず入る
        for index, (_name, weights) in enumerate(SEED_RECIPES):
            self.assertEqual([round(v, 6) for v in working[index]], list(weights))

    def test_best_performers_are_preferred(self):
        from hexa.ai import new_population
        from hexa.routes import SEED_RECIPES, mutate_recipe

        brain = new_population(count=2)[0]
        rng = random.Random(1)
        for _ in range(10):
            brain.recipes.append(mutate_recipe(brain.recipes[0], rng))
        brain.recipe_stats = [[1.0, 0.0, 0.0] for _ in brain.recipes]
        star = len(SEED_RECIPES) + 3
        brain.recipe_stats[star] = [1.0, 99.0, 99.0]
        working = brain.working_recipes()
        self.assertIn(brain.recipes[star], working)


class PathSpeedTest(unittest.TestCase):
    """経路探索を作り直したときの、結果が変わっていないことの確認。

    処理時間の9割を占めていたダイクストラ法を、辞書からリストに
    置き換えた（1試走 2.40秒 → 0.60秒）。速くなっても答えが
    変わっていないことを確かめる。
    """

    def test_paths_are_still_shortest_in_steps(self):
        """重み0のときは、従来どおりステップ数が最短であること。"""

        from hexa.hexgrid import neighbor
        from hexa.rules import Terrain, move_steps

        problem = generate_problem(51, PRESETS["medium"])
        finder = PathFinder(problem)
        table = finder.table(problem.agent_positions[0], {}, 0)
        for spot in problem.spots:
            if not table.reachable(spot.position):
                continue
            # たどった道すじのステップ数が、表の値と一致すること
            position = table.origin
            total = 0
            for direction in table.directions_to(spot.position):
                total += move_steps(problem.terrain_at(position), None)
                nxt = neighbor(position, direction, problem.width, problem.height)
                self.assertIsNotNone(nxt)
                position = nxt
            self.assertEqual(position, spot.position)
            self.assertEqual(total, table.steps[spot.position])

    def test_fuel_weight_changes_the_route(self):
        """燃料を重く見ると、道路を避けた道すじになること。"""

        import dataclasses

        from hexa.generator import generate_problem as gen, honsen_settings

        settings = honsen_settings("決勝", random.Random(5), players=10, day_count=5)
        settings = dataclasses.replace(settings, road_ratio=0.35, clump=0.3)
        problem = gen(77, settings)
        finder = PathFinder(problem)
        start = problem.agent_positions[0]
        cheap = finder.table(start, {}, 4)
        fast = finder.table(start, {}, 0)
        shared = [
            spot.position for spot in problem.spots
            if cheap.reachable(spot.position) and fast.reachable(spot.position)
        ]
        self.assertTrue(shared)
        cheaper = sum(
            1 for position in shared if cheap.fuel[position] < fast.fuel[position]
        )
        self.assertGreater(cheaper, 0, "燃料を重く見ても経路が変わっていません")
        for position in shared:
            # 燃料重視のほうが燃料を多く使うことはない
            self.assertLessEqual(cheap.fuel[position], fast.fuel[position])

    def test_unreachable_stays_unreachable(self):
        problem = generate_problem(52, PRESETS["small"])
        finder = PathFinder(problem)
        for weight in (0, 2, 4):
            table = finder.table(problem.agent_positions[0], {}, weight)
            for position in range(problem.cell_count):
                if problem.terrain_at(position) == Terrain.POND:
                    self.assertFalse(
                        table.reachable(position) and position != table.origin
                    )

    def test_cached_helpers_are_reused(self):
        """道路セルと渋滞状態を作り直していないこと。"""

        from hexa.routes import _jammed_status, road_cells_of

        problem = generate_problem(53, PRESETS["medium"])
        finder = PathFinder(problem)
        first = road_cells_of(finder, problem)
        self.assertIs(first, road_cells_of(finder, problem))
        jam = _jammed_status(finder, problem)
        self.assertIs(jam, _jammed_status(finder, problem))


class AssignmentSwapTest(unittest.TestCase):
    """巡回車どうしの担当の入れ替えが働くことの確認。

    エージェントは番号の若い順に目標を決めるので、
    「0番が店Aを取ったが、実は1番のほうがAに近かった」という
    先着優先の無駄が残っていた。全ペアを試して合計が減るなら交換する。

    実測（未見6問の玉数）:
      入れ替えなし 220.5 → 入れ替えあり 228.2（+3.5%）
      日別累計も 78.83 → 79.33 と改善
    """

    def test_swap_reduces_total_travel(self):
        from hexa.planner import _AgentPlanState, _swap_assignments
        from hexa.rules import AgentKind

        # 入れ替えで得をする組み合わせがある問題を探す
        found = None
        for seed in range(40):
            problem = generate_problem(61 + seed, PRESETS["medium"])
            finder = PathFinder(problem)
            positions = [problem.agent_positions[0], problem.agent_positions[1]]

            def cost(agent, target, finder=finder, positions=positions):
                return finder.table(positions[agent], {}, 0).steps.get(target)

            spots = [spot.position for spot in problem.spots]
            pairs = [
                (a, b)
                for a in spots for b in spots
                if a != b
                and None not in (cost(0, a), cost(1, b), cost(0, b), cost(1, a))
                and cost(0, b) + cost(1, a) < cost(0, a) + cost(1, b)
            ]
            if pairs:
                found = (problem, finder, positions, pairs[0])
                break
        self.assertIsNotNone(found, "入れ替えで得をする問題が見つかりません")
        problem, finder, positions, (target_a, target_b) = found

        kinds = (AgentKind.TOURING, AgentKind.TOURING)
        fuels = [problem.fuel_limit, problem.fuel_limit]
        remaining = problem.day_steps[0]

        states = [_AgentPlanState(), _AgentPlanState()]
        states[0].target = target_a
        states[1].target = target_b
        calls = []

        def start_route(index, target, left=0):
            calls.append((index, target))
            return True

        _swap_assignments(
            kinds, positions, fuels, states, {}, {}, [set(), set()],
            finder, {}, remaining, start_route,
        )
        self.assertEqual(states[0].target, target_b, "入れ替わっていません")
        self.assertEqual(states[1].target, target_a)

    def test_swap_keeps_targets_when_nothing_improves(self):
        from hexa.planner import _AgentPlanState, _swap_assignments
        from hexa.rules import AgentKind

        problem = generate_problem(62, PRESETS["medium"])
        finder = PathFinder(problem)
        spots = [spot.position for spot in problem.spots][:2]
        kinds = (AgentKind.TOURING, AgentKind.TOURING)
        positions = [problem.agent_positions[0], problem.agent_positions[1]]
        fuels = [problem.fuel_limit, problem.fuel_limit]

        def cost(agent, target):
            return finder.table(positions[agent], {}, 0).steps.get(target)

        best = min(
            [(spots[0], spots[1]), (spots[1], spots[0])],
            key=lambda pair: (cost(0, pair[0]) or 10 ** 9) + (cost(1, pair[1]) or 10 ** 9),
        )
        states = [_AgentPlanState(), _AgentPlanState()]
        states[0].target = best[0]
        states[1].target = best[1]
        _swap_assignments(
            kinds, positions, fuels, states, {}, {}, [set(), set()],
            finder, {}, problem.day_steps[0], lambda *a, **k: True,
        )
        # すでに最良なので変わらない
        self.assertEqual(states[0].target, best[0])
        self.assertEqual(states[1].target, best[1])

    def test_supply_cars_are_not_swapped(self):
        from hexa.planner import _AgentPlanState, _swap_assignments
        from hexa.rules import AgentKind

        problem = generate_problem(63, PRESETS["medium"])
        finder = PathFinder(problem)
        kinds = (AgentKind.SUPPLY, AgentKind.SUPPLY)
        states = [_AgentPlanState(), _AgentPlanState()]
        states[0].target = problem.spots[0].position
        states[1].target = problem.spots[1].position
        before = [state.target for state in states]
        _swap_assignments(
            kinds, list(problem.agent_positions[:2]), [0, 0], states, {}, {},
            [set(), set()], finder, {}, problem.day_steps[0], lambda *a, **k: True,
        )
        self.assertEqual([state.target for state in states], before)


class DayLengthFeatureTest(unittest.TestCase):
    """「今日はどれだけ長い日か」が材料に入っていることの確認。

    これまでは「残りに対する割合」しか見ておらず、
    32歩の日と246歩の日を絶対値で区別できなかった。
    短い日は近場を確実に、長い日は遠征、という使い分けに必要。
    """

    def _policy_and_problem(self):
        from hexa.ai import Policy, new_population
        from hexa.generator import generate_problem as gen, honsen_settings

        settings = honsen_settings("1回戦", random.Random(3), players=10, day_count=6)
        problem = gen(2026, settings)
        brain = new_population(count=2)[0]
        return Policy(brain, problem, random.Random(0), 0.0), problem

    def _ask(self, policy, problem, day_index):
        finder = PathFinder(problem)
        return policy.choose_touring_target(
            agent_index=0, step=0, remaining=problem.day_steps[day_index],
            position=problem.agent_positions[0], fuel=problem.fuel_limit,
            stock={spot.position: spot.stocks for spot in problem.spots},
            reserved={}, acquired_today=set(), acquired_brands=set(),
            brands_today=set(), road_status={}, finder=finder,
            kinds=(), positions=list(problem.agent_positions),
            day_index=day_index, day_count=problem.day_count,
        )

    def test_day_length_tracks_the_actual_steps(self):
        policy, problem = self._policy_and_problem()
        seen = []
        for day in range(problem.day_count):
            self._ask(policy, problem, day)
            seen.append((problem.day_steps[day], policy.day_length))
        # 長い日ほど値が大きい
        ordered = sorted(seen)
        self.assertLessEqual(ordered[0][1], ordered[-1][1])
        for _steps, value in seen:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertGreater(len({round(v, 3) for _s, v in seen}), 1, "日ごとに変わっていません")

    def test_feature_reaches_the_candidates(self):
        policy, problem = self._policy_and_problem()
        short = min(range(problem.day_count), key=lambda d: problem.day_steps[d])
        long = max(range(problem.day_count), key=lambda d: problem.day_steps[d])
        a = self._ask(policy, problem, short).candidate_features[0][29]
        b = self._ask(policy, problem, long).candidate_features[0][29]
        self.assertLess(a, b, "短い日と長い日で材料29番が同じです")


class SupplySwapTest(unittest.TestCase):
    """補給車どうしの担当の入れ替え。

    補給車が2台以上の試合でしか働かない（多くの試合は1台）ので
    効果は限定的だが、決勝のような大きい試合では2台選ばれることがある。
    """

    def test_swap_shortens_total_arrival(self):
        from hexa.planner import _AgentPlanState, _swap_supply_assignments
        from hexa.rules import AgentKind

        problem = generate_problem(71, PRESETS["large"])
        finder = PathFinder(problem)
        if problem.agent_count < 4:
            self.skipTest("エージェントが足りません")
        kinds = tuple(
            AgentKind.SUPPLY if index >= problem.agent_count - 2 else AgentKind.TOURING
            for index in range(problem.agent_count)
        )
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit] * problem.agent_count
        states = [_AgentPlanState() for _ in range(problem.agent_count)]
        first_supply = problem.agent_count - 2
        states[first_supply].target_agent = 1
        states[first_supply + 1].target_agent = 0

        def arrival(agent, target):
            return finder.table(positions[agent], {}, 0).steps.get(positions[target])

        before = (arrival(first_supply, 1) or 0) + (arrival(first_supply + 1, 0) or 0)
        _swap_supply_assignments(
            kinds, positions, fuels, states, finder, {},
            problem.day_steps[0], problem.fuel_limit,
        )
        after = (
            (arrival(first_supply, states[first_supply].target_agent) or 0)
            + (arrival(first_supply + 1, states[first_supply + 1].target_agent) or 0)
        )
        self.assertLessEqual(after, before, "入れ替えで合計が悪化しています")

    def test_stranded_cars_are_not_delayed(self):
        """燃料が尽きた巡回車の到着が遅くなる交換はしないこと。"""

        from hexa.planner import _AgentPlanState, _swap_supply_assignments
        from hexa.rules import AgentKind

        problem = generate_problem(72, PRESETS["large"])
        finder = PathFinder(problem)
        if problem.agent_count < 4:
            self.skipTest("エージェントが足りません")
        kinds = tuple(
            AgentKind.SUPPLY if index >= problem.agent_count - 2 else AgentKind.TOURING
            for index in range(problem.agent_count)
        )
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit] * problem.agent_count
        fuels[0] = 0   # 0番が立ち往生
        states = [_AgentPlanState() for _ in range(problem.agent_count)]
        supply = problem.agent_count - 2

        def arrival(agent, target):
            return finder.table(positions[agent], {}, 0).steps.get(positions[target])

        # 0番に近いほうの補給車を0番に割り当てておく
        near = supply if (arrival(supply, 0) or 10 ** 9) <= (arrival(supply + 1, 0) or 10 ** 9) else supply + 1
        far = supply + 1 if near == supply else supply
        states[near].target_agent = 0
        states[far].target_agent = 1
        best = arrival(near, 0)

        _swap_supply_assignments(
            kinds, positions, fuels, states, finder, {},
            problem.day_steps[0], problem.fuel_limit,
        )
        who = next(
            index for index in (supply, supply + 1)
            if states[index].target_agent == 0
        )
        self.assertLessEqual(
            arrival(who, 0) or 10 ** 9, best,
            "立ち往生した巡回車への到着が遅くなりました",
        )


class WaitBudgetTest(unittest.TestCase):
    """待機を学習に任せる仕組みが、既定では無効であることの確認。

    実測（未見12問）:
      待機なし        日別 76.50 / 玉 234.8  ← 採用
      待機あり2割まで 日別 75.83 / 玉 239.8
      待機あり4割まで 日別 75.42 / 玉 238.8

    日別累計は公式の第2優先で玉数より上。実戦で1位を取った2試合は
    どちらも日別累計で勝っているので、玉数が増えても採用しない。
    """

    def test_waiting_is_disabled_by_default(self):
        from hexa.planner import TOURING_WAIT_BUDGET

        self.assertEqual(TOURING_WAIT_BUDGET, 0.0)

    def test_touring_cars_still_move_when_waiting_is_allowed(self):
        """待機を許しても、一日じゅう動かない巡回車は出ないこと。"""

        import hexa.planner as planner_module
        from hexa.ai import new_population
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        original = planner_module.TOURING_WAIT_BUDGET
        planner_module.TOURING_WAIT_BUDGET = 0.4
        try:
            brains = new_population(count=4, capacity="simple")
            settings = honsen_settings("準決勝", random.Random(5), players=4, day_count=6)
            problem = gen(99, settings)
            result = run_match(problem, brains, seed=2, record_trace=True)
            for team in result.teams:
                for index, kind in enumerate(team.agent_kinds):
                    if kind != AgentKind.TOURING:
                        continue
                    moved = any(
                        any(action >= 0 for action in plan[index])
                        for plan in team.action_plans_by_day
                    )
                    self.assertTrue(moved, f"巡回車{index}が一度も動いていません")
        finally:
            planner_module.TOURING_WAIT_BUDGET = original

    def test_brand_guard_blocks_waiting(self):
        """今日まだ取っていない系列が残っているうちは待たせないこと。"""

        import hexa.planner as planner_module
        from hexa.ai import Policy, new_population
        from hexa.generator import generate_problem as gen, honsen_settings

        original = planner_module.TOURING_WAIT_BUDGET
        planner_module.TOURING_WAIT_BUDGET = 0.4
        try:
            settings = honsen_settings("1回戦", random.Random(7), players=10, day_count=5)
            problem = gen(123, settings)
            brain = new_population(count=2)[0]
            # 待機にだけ強く効く重みを入れて、それでも動くことを見る
            width = len(brain.touring_heads[0])
            for head in range(3):
                brain.touring_heads[head] = [0.0] * width
                brain.touring_heads[head][1] = 5.0    # 「待機か」に大きな重み
            policy = Policy(brain, problem, random.Random(0), 0.0)
            decision = policy.choose_touring_target(
                agent_index=0, step=0, remaining=problem.day_steps[0],
                position=problem.agent_positions[0], fuel=problem.fuel_limit,
                stock={spot.position: spot.stocks for spot in problem.spots},
                reserved={}, acquired_today=set(), acquired_brands=set(),
                brands_today=set(), road_status={}, finder=PathFinder(problem),
                kinds=(), positions=list(problem.agent_positions),
                day_index=0, day_count=problem.day_count, allow_wait=True,
            )
            self.assertIsNotNone(
                decision.target, "系列が残っているのに待機を選んでいます"
            )
        finally:
            planner_module.TOURING_WAIT_BUDGET = original


class WeightStabilityTest(unittest.TestCase):
    """重みが発散しないことの確認。

    以前は `重み += 学習率 × 誤差 × 材料` とそのまま足していた。
    この形が安定するのは `学習率 < 2 / |材料|^2` のときだけで、
    材料が増えると同じ学習率で必ず破綻する。

        simple  材料  40個 → |材料|^2 = 15.2  → 上限 0.132（0.08 は安全）
        cross   材料 211個 → |材料|^2 = 30.2  → 上限 0.066（0.08 は危険）
        large   材料 781個 → |材料|^2 = 121.5 → 上限 0.017（0.08 は危険）

    実測では large で**3エピソード**で壊れ、重みが 3.85e+117 まで飛んだ。
    60万エピソード回したログでは、2万回目以降ほぼ毎ラウンド
    「重みが壊れたAIがいます」が出続け、61時間かけて一歩も進まなかった。
    """

    def _train(self, capacity, episodes=30, alpha=0.08):
        from hexa.ai import new_population
        from hexa.generator import generate_problem as gen, honsen_settings

        brains = new_population(count=3, capacity=capacity)
        rng = random.Random(1)
        for episode in range(episodes):
            settings = honsen_settings("1回戦", rng, players=3, day_count=4)
            problem = gen(rng.randrange(1, 10 ** 9), settings)
            result = run_match(problem, brains, seed=episode, epsilon=0.3, kind_explore=0.5)
            for index, team in enumerate(result.teams):
                for features, rewards in team.touring_samples:
                    brains[index].update_touring(features, rewards, alpha)
                for features, reward in team.supply_samples:
                    brains[index].update_supply(features, reward, alpha)
        return brains

    def _finite(self, brains):
        for brain in brains:
            for head in brain.touring_heads:
                for value in head:
                    if value != value or abs(value) > 1e6:
                        return False
            for value in brain.supply_weights:
                if value != value or abs(value) > 1e6:
                    return False
        return True

    def test_large_does_not_diverge(self):
        """一番大きい large でも重みが飛ばないこと。"""

        brains = self._train("large")
        self.assertTrue(self._finite(brains), "large で重みが壊れました")

    def test_every_capacity_survives(self):
        for capacity in ("simple", "cross", "large"):
            brains = self._train(capacity, episodes=20)
            self.assertTrue(self._finite(brains), f"{capacity} で重みが壊れました")

    def test_step_is_scaled_by_feature_size(self):
        """一歩の大きさが材料の数で揃っていること。"""

        from hexa.ai import _step_scale

        small = [1.0] * 10
        big = [1.0] * 1000
        self.assertGreater(_step_scale(small), _step_scale(big))
        # 材料がすべて0でも割り算が壊れない
        self.assertGreater(_step_scale([0.0] * 50), 0.0)
        self.assertLessEqual(_step_scale([0.0] * 50), 1.0)

    def test_learning_still_reduces_error(self):
        """安定させても、ちゃんと誤差が減ること。"""

        from hexa.ai import new_population

        brain = new_population(count=2, capacity="cross")[0]
        base = [0.3] * brain.touring_heads[0].__len__()
        base = [0.3] * 40
        while len(base) < 40:
            base.append(0.0)
        target = [5.0, 30.0, 100.0]
        first = None
        for step in range(200):
            before = brain.predict_touring(base)
            error = sum(abs(t - b) for t, b in zip(target, before))
            if first is None:
                first = error
            brain.update_touring(base, target, 0.5)
        after = brain.predict_touring(base)
        last = sum(abs(t - b) for t, b in zip(target, after))
        self.assertLess(last, first * 0.2, "誤差が十分に減っていません")


class SubmissionTrackingTest(unittest.TestCase):
    """提出に失敗した案を予測に使わないことの確認。

    実戦で毎試合3〜5件の「ずれを検出」が出ていた。原因は、
    送れなかった案でも `best` を進めて、それを予測に使っていたこと。
    サーバーが持っているのは**最後に受理された回答**なので、
    送れなかった案を採用したことにすると予測と実際がずれる。
    """

    def test_failed_submit_returns_minus_one(self):
        from hexa.client import ProconApiError
        from hexa.production import _submit

        class Failing:
            def post_actions(self, plan):
                raise ProconApiError(400, "TestError", "わざと失敗")

        self.assertEqual(_submit(Failing(), [[0]], lambda message: None), -1)

    def test_successful_submit_returns_revision(self):
        from hexa.production import _submit

        class Working:
            def post_actions(self, plan):
                return 7

        self.assertEqual(_submit(Working(), [[0]], lambda message: None), 7)

    def test_production_uses_accepted_plan_for_prediction(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("accepted = best if revision >= 0 else None", source)
        self.assertIn("final = accepted if accepted is not None else best", source)
        self.assertIn("if sent < 0:", source)


class CrashSafetyTest(unittest.TestCase):
    """本番モードが落ちないことの確認。

    本選で止まったら0点。メモリ不足でも想定外の失敗でも、
    提出だけは続ける形にしてある。
    """

    def test_production_guards_every_failure(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("except MemoryError", source)
        self.assertIn("except Exception as error", source)
        # 状態が取れなくても全員待機を送る
        self.assertIn("_fallback_plan(problem.agent_count, day_steps)", source)

    def test_fallback_plan_is_valid(self):
        from hexa.production import _fallback_plan

        plan = _fallback_plan(5, 40)
        self.assertEqual(len(plan), 5)
        for actions in plan:
            self.assertTrue(actions)
            for action in actions:
                self.assertLessEqual(action, -1)


class NineTeamTest(unittest.TestCase):
    """9チーム戦に対応していることの確認。

    本選の初戦は津山高専＋8チームで9チーム戦。
    交通量は「全チームの滞在数 ÷ players」なので、players が変われば
    道路の混み方も変わる。9でも10でも戦えなければならない。
    """

    def test_nine_team_match_runs(self):
        from hexa.ai import new_population
        from hexa.generator import generate_problem as gen, honsen_settings

        brains = new_population(count=9, capacity="simple")
        settings = honsen_settings("1回戦", random.Random(3), players=9, day_count=5)
        problem = gen(2026, settings)
        self.assertEqual(problem.players, 9)
        result = run_match(problem, brains, seed=1)
        self.assertEqual(len(result.teams), 9)

    def test_training_mixes_nine_and_ten(self):
        from collections import Counter

        from hexa.training import _curriculum_settings

        rng = random.Random(2)
        seen = Counter()
        for _ in range(200):
            settings = _curriculum_settings(
                0.5, ("small", "medium", "large"), 10, rng, 0.6
            )
            seen[settings.players] += 1
        self.assertIn(9, seen)
        self.assertIn(10, seen)


class PondClusterTest(unittest.TestCase):
    """池と山地が実マップ並みに固まることの確認。

    練習場の実マップを測ると 池 +0.63 / 山地 +0.52 / 道路 +0.28。
    以前の生成器は池 +0.26 どまりで、ばらけすぎていた。
    """

    def _excess(self, problem):
        from hexa.hexgrid import neighbor_table
        from hexa.rules import Terrain

        table = neighbor_table(problem.width, problem.height)
        same: dict[int, int] = {}
        total: dict[int, int] = {}
        for position in range(problem.cell_count):
            terrain = problem.cells[position]
            for _direction, neighbour in table[position]:
                total[terrain] = total.get(terrain, 0) + 1
                if problem.cells[neighbour] == terrain:
                    same[terrain] = same.get(terrain, 0) + 1
        result = {}
        for terrain in total:
            share = sum(1 for c in problem.cells if c == terrain) / problem.cell_count
            result[Terrain(terrain).name] = same.get(terrain, 0) / total[terrain] - share
        return result

    def test_ponds_can_form_large_lakes(self):
        from hexa.generator import ARCHETYPES, generate_problem as gen, honsen_settings

        rng = random.Random(5)
        best = 0.0
        for name in ARCHETYPES:
            for _ in range(3):
                settings = honsen_settings(
                    rng.choice(["1回戦", "準決勝", "決勝"]), rng,
                    players=10, archetype=name,
                )
                problem = gen(rng.randrange(1, 10 ** 9), settings)
                self.assertEqual(validate_problem(problem), [], name)
                best = max(best, self._excess(problem).get("POND", 0.0))
        self.assertGreater(best, 0.45, "池が実マップ並みに固まりません")

    def test_mountains_can_form_ranges(self):
        from hexa.generator import ARCHETYPES, generate_problem as gen, honsen_settings

        rng = random.Random(9)
        best = 0.0
        for name in ARCHETYPES:
            for _ in range(3):
                settings = honsen_settings("決勝", rng, players=10, archetype=name)
                problem = gen(rng.randrange(1, 10 ** 9), settings)
                best = max(best, self._excess(problem).get("MOUNTAIN", 0.0))
        self.assertGreater(best, 0.40, "山地が実マップ並みに固まりません")


class RouteReorderTest(unittest.TestCase):
    """訪問順の並べ替えが働くことの確認。

    作戦は「次に一番良い店」を継ぎ足して道すじを作るので、
    1店ずつ見ると最善でも並び全体では遠回りになることがある。

    実戦の9日ぶんを調べると、合計2574歩のうち249歩（9%）が
    並べ替えだけで削れた。1日目の車1は110歩が64歩まで縮む。

    実測（未見10問）:
      並べ替えなし 玉 246.3
      並べ替えあり 玉 251.0（+1.9%）
      種類と日別累計はどちらも変わらない
    """

    def _setup(self, seed=3):
        from hexa.generator import generate_problem as gen, honsen_settings

        settings = honsen_settings("準決勝", random.Random(seed), players=10, day_count=9)
        problem = gen(2026, settings)
        return problem, PathFinder(problem)

    def test_reorder_never_makes_it_longer(self):
        from hexa.routes import _walk_cost, reorder_route

        problem, finder = self._setup()
        spots = [spot.position for spot in problem.spots][:3]
        start = problem.agent_positions[0]
        before, _fuel = _walk_cost(finder, {}, start, spots, 10 ** 6, 10 ** 6)
        order, after, _f = reorder_route(finder, {}, start, spots, 10 ** 6, 10 ** 6)
        self.assertGreaterEqual(before, after, "並べ替えで遠回りになりました")
        self.assertEqual(sorted(order), sorted(spots), "訪問する店が変わりました")

    def test_reorder_keeps_the_same_spots(self):
        """順番を変えるだけで、取れる店は変わらないこと。"""

        from hexa.routes import reorder_route

        problem, finder = self._setup(seed=7)
        for size in (2, 3):
            spots = [spot.position for spot in problem.spots][:size]
            order, steps, fuel = reorder_route(
                finder, {}, problem.agent_positions[0], spots, 10 ** 6, 10 ** 6
            )
            self.assertEqual(sorted(order), sorted(spots))
            self.assertGreaterEqual(steps, 0)
            self.assertGreaterEqual(fuel, 0)

    def test_reorder_respects_the_budget(self):
        """歩数や燃料が足りない並びは採用しないこと。"""

        from hexa.routes import reorder_route

        problem, finder = self._setup(seed=11)
        spots = [spot.position for spot in problem.spots][:3]
        order, steps, fuel = reorder_route(
            finder, {}, problem.agent_positions[0], spots, 1, 1
        )
        self.assertEqual(steps, -1, "予算を超える並びを採用しています")

    def test_built_routes_are_reordered(self):
        from hexa.routes import build_routes, default_recipes

        problem, finder = self._setup()
        available = {spot.position: spot.stocks for spot in problem.spots}
        routes = build_routes(
            problem, finder, {}, problem.agent_positions[0],
            problem.fuel_limit, problem.day_steps[0],
            available, set(), set(), default_recipes(),
        )
        self.assertTrue(routes)
        for route in routes:
            self.assertGreaterEqual(route.steps, 0)
            self.assertLessEqual(route.steps, problem.day_steps[0])
            self.assertLessEqual(route.fuel, problem.fuel_limit)

    def test_lookahead_stays_at_three(self):
        from hexa.routes import MAX_ROUTE_LENGTH

        self.assertEqual(MAX_ROUTE_LENGTH, 3)


class BrandFirstPolicyTest(unittest.TestCase):
    """系列優先の方策が候補に入っていることの確認。

    日別累計は公式の第2優先で、玉数より上。実戦で1位を取った試合は
    どちらも日別累計で勝っていた。

    以前**すべての判断で**新しい系列を優先させたら、日別累計まで
    下がった（77.90→73.00）。遠い店へ無理に向かうと、近場を回れば
    取れたはずの別の系列を落とすため。

    そこで**候補の1つとして加えるだけ**にした。最終的な選択は
    公式の優先順位で行うので、勝つのは本当に日別累計が増えるときだけ。

    実測（未見10問）:
      系列優先なし 日別 67.10 / 玉 145.5
      系列優先あり 日別 68.10 / 玉 139.9  ← 採用（日別が上位の指標）
    """

    def _problem(self, seed=3):
        from hexa.generator import generate_problem as gen, honsen_settings

        settings = honsen_settings("準決勝", random.Random(seed), players=10, day_count=6)
        return gen(2026, settings)

    def test_policy_prefers_uncollected_brands(self):
        from hexa.production import BrandFirstPolicy

        problem = self._problem()
        policy = BrandFirstPolicy(problem)
        finder = PathFinder(problem)
        stock = {spot.position: spot.stocks for spot in problem.spots}
        brands = sorted({spot.brand for spot in problem.spots})
        # 半分の系列は今日すでに取った状態にする
        taken = set(brands[: len(brands) // 2])
        decision = policy.choose_touring_target(
            agent_index=0, step=0, remaining=problem.day_steps[0],
            position=problem.agent_positions[0], fuel=problem.fuel_limit,
            stock=stock, reserved={}, acquired_today=set(),
            acquired_brands=set(), brands_today=taken,
            road_status={}, finder=finder, kinds=(),
            positions=list(problem.agent_positions),
            day_index=0, day_count=problem.day_count,
        )
        self.assertIsNotNone(decision.target)
        brand = {spot.position: spot.brand for spot in problem.spots}[decision.target]
        self.assertNotIn(brand, taken, "今日すでに取った系列を選んでいます")

    def test_policy_avoids_brands_mates_already_target(self):
        """味方が向かっている系列は避けること。"""

        from hexa.production import BrandFirstPolicy

        problem = self._problem(seed=11)
        policy = BrandFirstPolicy(problem)
        finder = PathFinder(problem)
        stock = {spot.position: spot.stocks for spot in problem.spots}
        brand_of = {spot.position: spot.brand for spot in problem.spots}
        first = policy.choose_touring_target(
            agent_index=0, step=0, remaining=problem.day_steps[0],
            position=problem.agent_positions[0], fuel=problem.fuel_limit,
            stock=stock, reserved={}, acquired_today=set(),
            acquired_brands=set(), brands_today=set(),
            road_status={}, finder=finder, kinds=(),
            positions=list(problem.agent_positions),
            day_index=0, day_count=problem.day_count,
        )
        mates = [None] * problem.agent_count
        mates[1] = first.target
        second = policy.choose_touring_target(
            agent_index=0, step=0, remaining=problem.day_steps[0],
            position=problem.agent_positions[0], fuel=problem.fuel_limit,
            stock=stock, reserved={}, acquired_today=set(),
            acquired_brands=set(), brands_today=set(),
            road_status={}, finder=finder, kinds=(),
            positions=list(problem.agent_positions),
            day_index=0, day_count=problem.day_count,
            teammate_targets=mates,
        )
        if second.target is not None and first.target is not None:
            self.assertNotEqual(
                brand_of[second.target], brand_of[first.target],
                "味方と同じ系列を狙っています",
            )

    def test_production_includes_the_policy(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("BrandFirstPolicy(problem)", source)
        self.assertIn("系列優先", source)

    def test_adding_a_candidate_cannot_lower_the_key(self):
        """候補を増やしても、選ばれる計画が悪くなることはないこと。"""

        from hexa.production import _candidate_key

        class Fake:
            def __init__(self, brands, noodles, fuel):
                self.acquisitions = [
                    type("E", (), {"brand": b})() for b in brands
                ]
                self.fuels = [fuel]

        a = _candidate_key(Fake([0, 1], 2, 5), set(), 100)
        b = _candidate_key(Fake([0, 1, 2], 3, 5), set(), 100)
        self.assertGreater(b, a, "系列が多い計画が勝たなければおかしい")


class LiveMonitorTest(unittest.TestCase):
    """リアルタイム監視が本番の処理を邪魔しないことの確認。

    本番は1日60秒しかない。監視のために待たされてはいけないので、
    やるのは軽いJSONを1つ書くだけ（実測0.4ミリ秒）。
    書き込みに失敗しても試合は止めない。
    """

    def _problem(self):
        from hexa.generator import generate_problem as gen, honsen_settings

        return gen(2026, honsen_settings("1回戦", random.Random(3), players=10, day_count=5))

    def test_writing_is_fast(self):
        import tempfile
        import time

        from hexa.livemonitor import LiveMonitor
        from hexa.rules import AgentKind

        problem = self._problem()
        kinds = tuple(AgentKind.TOURING for _ in range(problem.agent_count))
        with tempfile.TemporaryDirectory() as directory:
            monitor = LiveMonitor(Path(directory))
            monitor.start_match(problem, kinds)
            started = time.monotonic()
            for _ in range(20):
                monitor.searching(0, 500, "貪欲", 12.5)
            elapsed = (time.monotonic() - started) / 20
            self.assertLess(elapsed, 0.05, "書き出しが遅すぎます")

    def test_state_file_is_valid_json(self):
        import json
        import tempfile

        from hexa.livemonitor import LIVE_NAME, LiveMonitor
        from hexa.rules import AgentKind

        problem = self._problem()
        kinds = tuple(AgentKind.TOURING for _ in range(problem.agent_count))
        with tempfile.TemporaryDirectory() as directory:
            monitor = LiveMonitor(Path(directory))
            monitor.start_match(problem, kinds)
            data = json.loads((Path(directory) / LIVE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(data["header"]["height"], problem.height)
            self.assertEqual(len(data["header"]["spots"]), len(problem.spots))
            self.assertIn("status", data)

    def test_failure_does_not_raise(self):
        """書き込めない場所を指定しても例外を出さないこと。"""

        from hexa.livemonitor import LiveMonitor
        from hexa.rules import AgentKind

        problem = self._problem()
        kinds = tuple(AgentKind.TOURING for _ in range(problem.agent_count))
        monitor = LiveMonitor(Path("/proc/絶対に書けない場所"))
        monitor.start_match(problem, kinds)      # 例外が出なければ合格
        monitor.searching(0, 1, "x", 1.0)
        monitor.finish_match("種類1", [])

    def test_production_reports_to_the_monitor(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("monitor.start_match(problem, kinds)", source)
        self.assertIn("monitor.finish_day(", source)
        self.assertIn("monitor.finish_match(", source)


class TrackTest(unittest.TestCase):
    """ビューアに渡す「1歩ごとの位置」の確認。

    以前はビューア側が行動値から方向を解釈し直していたが、
    その方向表が hexgrid.neighbor と別物で、**軌跡とエージェントの
    動きがずれていた。** 公式シミュレータの結果をそのまま渡す形にした。
    """

    def _plan(self, seed=3):
        from hexa.ai import Policy, new_population
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.planner import DayPlanner
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(seed),
                                            players=10, day_count=4))
        brain = new_population(count=2, capacity="simple")[0]
        finder = PathFinder(problem)
        planner = DayPlanner(problem, finder)
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit if k == AgentKind.TOURING else 0 for k in kinds]
        plan = planner.plan_day(
            kinds, positions, fuels, {}, problem.day_steps[0],
            Policy(brain, problem, random.Random(0), 0.0), set(), 0, problem.day_count,
        )
        return problem, kinds, positions, fuels, plan

    def test_track_starts_and_ends_correctly(self):
        from hexa.production import _tracks_of

        problem, kinds, positions, fuels, plan = self._plan()
        tracks = _tracks_of(problem, kinds, positions, fuels, {},
                            plan.action_plan, problem.day_steps[0])
        self.assertEqual(len(tracks), problem.agent_count)
        for index in range(problem.agent_count):
            self.assertEqual(tracks[index][0], positions[index], "開始位置が違います")
            self.assertEqual(tracks[index][-1], plan.end_positions[index],
                             "終了位置が違います")

    def test_track_only_moves_to_neighbours(self):
        """1歩で隣以外へ飛ばないこと。方向の解釈違いがあれば必ず落ちます。"""

        from hexa.hexgrid import neighbor
        from hexa.production import _tracks_of

        problem, kinds, positions, fuels, plan = self._plan(seed=9)
        tracks = _tracks_of(problem, kinds, positions, fuels, {},
                            plan.action_plan, problem.day_steps[0])
        for row in tracks:
            for before, after in zip(row, row[1:]):
                if before == after:
                    continue
                neighbours = {
                    neighbor(before, d, problem.width, problem.height)
                    for d in range(6)
                }
                self.assertIn(after, neighbours,
                              f"{before} から {after} へ飛んでいます")

    def test_track_length_matches_day_steps(self):
        from hexa.production import _tracks_of

        problem, kinds, positions, fuels, plan = self._plan(seed=11)
        tracks = _tracks_of(problem, kinds, positions, fuels, {},
                            plan.action_plan, problem.day_steps[0])
        for row in tracks:
            self.assertEqual(len(row), problem.day_steps[0] + 1)

    def test_failure_returns_empty(self):
        """壊れた計画を渡しても例外を出さないこと。"""

        from hexa.production import _tracks_of

        problem, kinds, positions, fuels, _plan = self._plan(seed=13)
        broken = [[99] * 5 for _ in range(problem.agent_count)]
        self.assertEqual(
            _tracks_of(problem, kinds, positions, fuels, {}, broken,
                       problem.day_steps[0]),
            [],
        )


class ViewerServerTest(unittest.TestCase):
    """監視ビューアを配るサーバーの確認。

    以前は functools.partial を継承しようとして TypeError で落ちていた
    （partial はクラスではない）。先に継承してから partial を作る。
    """

    def test_server_starts_and_serves(self):
        import urllib.request

        import main

        server, url = main.start_viewer_server(open_browser=False)
        self.assertIsNotNone(server, "ビューアのサーバーが起動しません")
        try:
            body = urllib.request.urlopen(url, timeout=5).read()
            self.assertIn(b"<canvas", body)
        finally:
            server.shutdown()

    def test_menus_offer_the_viewer(self):
        import inspect

        import main

        for name in ("menu_online", "menu_arena"):
            source = inspect.getsource(getattr(main, name))
            self.assertIn("start_viewer_server", source, f"{name} が開きません")


class MonitorPayloadTest(unittest.TestCase):
    """ビューアが読む JSON に、必要な項目がすべて入っていることの確認。

    実戦で「ビューアの右側が壊れる」「マップが壊れる」という不具合が出た。
    原因は、本番モードが refuels を**件数だけ**渡していて
    refuelEvents が作られず、daySteps・startPositions・startFuels も
    送っていなかったこと。ビューアはそれらを前提に描くので崩れていた。
    """

    NEEDED = ("daySteps", "startPositions", "startFuels", "positions", "fuels",
              "acquired", "refuelEvents", "actions", "tracks", "roads",
              "brandTotal", "dailyTotal", "noodleTotal")

    def _run_days(self, directory, days=3):
        import random as rnd

        from hexa.ai import Policy, new_population
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.livemonitor import LiveMonitor
        from hexa.planner import DayPlanner
        from hexa.production import _tracks_of
        from hexa.rules import AgentKind
        from hexa.simulator import road_status_for_day, simulate_day

        problem = gen(2026, honsen_settings("1回戦", rnd.Random(5),
                                            players=10, day_count=days))
        brains = new_population(count=3, capacity="simple")
        finder = PathFinder(problem)
        planner = DayPlanner(problem, finder)
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit if k == AgentKind.TOURING else 0 for k in kinds]
        monitor = LiveMonitor(directory)
        monitor.start_match(problem, kinds)
        stays, brands, daily, noodles = [], set(), 0, 0
        for index in range(problem.day_count):
            road = road_status_for_day(problem, stays)
            plan = planner.plan_day(
                kinds, positions, fuels, road, problem.day_steps[index],
                Policy(brains[0], problem, rnd.Random(index), 0.0),
                brands, index, problem.day_count,
            )
            result = simulate_day(
                problem, kinds, tuple(positions), tuple(fuels),
                plan.action_plan, problem.day_steps[index], road, record_trace=True,
            )
            today = {e.brand for e in result.acquisitions}
            brands |= today
            daily += len(today)
            noodles += len(result.acquisitions)
            monitor.finish_day(
                day_index=index, label="test", candidates=10, resubmissions=0,
                revision=1, day_steps=problem.day_steps[index],
                start_positions=positions, start_fuels=fuels,
                positions=list(result.positions), fuels=list(result.fuels),
                acquisitions=list(result.acquisitions), refuels=list(result.refuels),
                action_plan=plan.action_plan,
                tracks=_tracks_of(problem, kinds, positions, fuels, road,
                                  plan.action_plan, problem.day_steps[index]),
                road_status=road, enemy_positions=[], brand_total=len(brands),
                daily_total=daily, noodle_total=noodles,
            )
            stays.append(dict(result.stay_counts))
            positions = list(result.positions)
            fuels = list(result.fuels)
        return problem

    def test_every_field_is_present(self):
        import json
        import tempfile

        from hexa.livemonitor import LIVE_NAME

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem = self._run_days(root)
            data = json.loads((root / LIVE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(len(data["days"]), problem.day_count)
            for day in data["days"]:
                for key in self.NEEDED:
                    self.assertIn(key, day, f"{day['day']}日目に {key} がありません")

    def test_tracks_cover_every_step(self):
        import json
        import tempfile

        from hexa.livemonitor import LIVE_NAME

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem = self._run_days(root)
            data = json.loads((root / LIVE_NAME).read_text(encoding="utf-8"))
            for day in data["days"]:
                self.assertEqual(len(day["tracks"]), problem.agent_count)
                for row in day["tracks"]:
                    self.assertEqual(len(row), day["daySteps"] + 1)

    def test_refuel_events_have_detail(self):
        """補給が件数ではなく、1件ずつの記録になっていること。"""

        import json
        import tempfile

        from hexa.livemonitor import LIVE_NAME

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run_days(root)
            data = json.loads((root / LIVE_NAME).read_text(encoding="utf-8"))
            events = [e for day in data["days"] for e in day["refuelEvents"]]
            self.assertTrue(events, "補給の記録がありません")
            for event in events:
                for key in ("step", "agent", "pos", "before", "after"):
                    self.assertIn(key, event)

    def test_production_passes_everything(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        for key in ("day_steps=day_steps", "start_positions=list(positions)",
                    "start_fuels=list(fuels)", "refuels=list(final.day_result.refuels)"):
            self.assertIn(key, source, f"{key} を渡していません")

    def test_preview_has_the_same_shape(self):
        """探索中に見せる暫定案も、同じ項目を持つこと。"""

        import random as rnd

        from hexa.ai import Policy, new_population
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.livemonitor import LiveMonitor
        from hexa.planner import DayPlanner
        from hexa.production import _build_candidate, _preview_day
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", rnd.Random(5),
                                            players=10, day_count=3))
        brains = new_population(count=2, capacity="simple")
        finder = PathFinder(problem)
        planner = DayPlanner(problem, finder)
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit if k == AgentKind.TOURING else 0 for k in kinds]
        best = _build_candidate(
            "x", problem, planner, kinds, positions, fuels, {},
            problem.day_steps[0], Policy(brains[0], problem, rnd.Random(0), 0.0),
            set(), 0,
        )
        self.assertIsNotNone(best)

        class FakeState:
            road_status: dict = {}

        monitor = LiveMonitor(Path("/tmp"), enabled=False)
        preview = _preview_day(
            monitor, problem, kinds, positions, fuels, FakeState(), best,
            0, problem.day_steps[0], 5, 0, 1, set(), 0, 0,
        )
        for key in self.NEEDED:
            self.assertIn(key, preview, f"暫定案に {key} がありません")
        self.assertEqual(len(preview["tracks"]), problem.agent_count)


class CandidateSpeedTest(unittest.TestCase):
    """1日に作れる候補の数を増やす最適化の確認。

    実戦の24x24・10日で、候補が115〜432件しか作れていなかった
    （16x16では1500件前後）。1案に80ミリ秒かかっていたため。

    プロファイルで3つの無駄を見つけて直した。

      1. _score が tuple/zip/sum を使い、1案あたり18万回呼ばれていた
      2. 在庫のない店も毎回調べていた
      3. road_key が毎回ソートし直していた（1案あたり8万回）

    結果: 1案 80ms → 52〜61ms。55秒で 654案 → 902〜1055案。
    **答えは1つも変わっていない**（未見10問で完全一致）。
    """

    def test_score_matches_the_plain_formula(self):
        """書き下した採点が、元の式と同じ値を返すこと。"""

        from hexa.routes import _score

        class Spot:
            def __init__(self, brand, stocks):
                self.brand = brand
                self.stocks = stocks

        def reference(recipe, spot, steps, need, acquired, seen, share, enemy, jam):
            values = (
                steps / 30.0, need / 30.0, -spot.stocks / 7.0,
                0.0 if spot.brand not in acquired else 1.0,
                0.0 if spot.brand not in seen else 1.0,
                share, -enemy, jam,
            )
            return sum(w * v for w, v in zip(recipe, values))

        rng = random.Random(5)
        for _ in range(60):
            recipe = [rng.uniform(-1.5, 1.5) for _ in range(8)]
            spot = Spot(rng.randrange(0, 20), rng.randint(1, 7))
            steps, need = rng.randint(1, 90), rng.randint(0, 90)
            acquired = set(rng.sample(range(20), 7))
            seen = set(rng.sample(range(20), 5))
            share, enemy, jam = rng.random(), rng.random() * 50, rng.random()
            got = _score(recipe, spot, steps, need, acquired, seen, share, enemy, jam)
            want = reference(recipe, spot, steps, need, acquired, seen, share, enemy, jam)
            self.assertAlmostEqual(got, want, places=9)

    def test_road_key_cache_is_consistent(self):
        """道路状態のキャッシュが、違う状態を取り違えないこと。"""

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import RoadStatus, Terrain

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=5))
        finder = PathFinder(problem)
        roads = [
            cell for cell in range(problem.cell_count)
            if problem.terrain_at(cell) == Terrain.ROAD
        ]
        start = problem.agent_positions[0]
        clear = finder.table(start, {}, 0)
        jammed = finder.table(start, {c: RoadStatus.JAMMED for c in roads}, 0)
        again = finder.table(start, {}, 0)
        # 状態を切り替えても、最初の結果と同じものが返ること
        self.assertEqual(again.steps, clear.steps)
        self.assertNotEqual(jammed.steps, clear.steps)

    def test_empty_spots_do_not_break_routes(self):
        """在庫が1つもないときに落ちないこと。"""

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.routes import build_routes, default_recipes

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        finder = PathFinder(problem)
        routes = build_routes(
            problem, finder, {}, problem.agent_positions[0],
            problem.fuel_limit, problem.day_steps[0],
            {}, set(), set(), default_recipes(),
        )
        self.assertEqual(routes, [])


class HexLayoutTest(unittest.TestCase):
    """ビューアの六角形の並べ方が、ルールの幾何と一致することの確認。

    実戦で「軌跡とエージェントの動きが合わない」という不具合が出た。
    データ（1歩ごとの位置）は8015回検査して違反0件だったので、
    原因は**描画位置**だった。

    一番上を1行目とすると、奇数行目が右・偶数行目が左へずれる。
    プログラム上の row=0 が1行目なので、**偶数インデックスの行が右**。
    ビューアはこれを逆にしていた。
    """

    def _spread(self, shift_even: bool) -> tuple[float, float]:
        """その並べ方で、隣り合うマスの距離がどれだけばらつくか。"""

        import math

        from hexa.hexgrid import neighbor

        width = height = 12

        def x_of(position: int) -> float:
            row = position // width
            offset = 0.5 if ((row % 2 == 0) == shift_even) else 0.0
            return (position % width + offset + 0.5) * 1.732

        def y_of(position: int) -> float:
            return (position // width + 0.5) * 1.5

        distances = []
        for position in range(width * height):
            for direction in range(6):
                other = neighbor(position, direction, width, height)
                if other is None:
                    continue
                distances.append(
                    math.dist((x_of(position), y_of(position)),
                              (x_of(other), y_of(other)))
                )
        return (min(distances), max(distances))

    def test_even_rows_shift_right(self):
        low, high = self._spread(True)
        self.assertLess(high - low, 0.01, "偶数インデックス行を右にずらすのが正しい")

    def test_odd_rows_do_not_fit(self):
        low, high = self._spread(False)
        self.assertGreater(high - low, 0.5, "逆向きならばらつくはず")

    def test_viewer_uses_the_right_offset(self):
        html = (Path(__file__).resolve().parent.parent
                / "viewer" / "live.html").read_text(encoding="utf-8")
        self.assertIn("(1-((((p/h.width)|0)%2)))*0.5", html,
                      "ビューアの行のずれが逆です")


class EnvFileTest(unittest.TestCase):
    """トークンを .env に保存する仕組みの確認。

    当日の入力ミスを減らすため。push されないよう .gitignore にも入れる。
    """

    def test_gitignore_hides_env(self):
        text = (Path(__file__).resolve().parent.parent
                / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", text, "トークンが push されてしまいます")

    def test_round_trip(self):
        import tempfile

        import main

        original = main.BASE_DIR
        with tempfile.TemporaryDirectory() as directory:
            main.BASE_DIR = Path(directory)
            try:
                self.assertEqual(main.load_env(), {})
                main.save_env({"PROCON_TOKEN": "abc"})
                self.assertEqual(main.load_env()["PROCON_TOKEN"], "abc")
                # 既存を書き換え、新しい項目を足す
                main.save_env({"PROCON_TOKEN": "xyz", "PROCON_URL": "https://x.test"})
                values = main.load_env()
                self.assertEqual(values["PROCON_TOKEN"], "xyz")
                self.assertEqual(values["PROCON_URL"], "https://x.test")
            finally:
                main.BASE_DIR = original

    def test_comments_and_quotes_are_handled(self):
        import tempfile

        import main

        original = main.BASE_DIR
        with tempfile.TemporaryDirectory() as directory:
            main.BASE_DIR = Path(directory)
            try:
                (Path(directory) / ".env").write_text(
                    '# コメント\nPROCON_TOKEN="quoted"\n\nBAD LINE\n',
                    encoding="utf-8",
                )
                self.assertEqual(main.load_env()["PROCON_TOKEN"], "quoted")
            finally:
                main.BASE_DIR = original

    def test_tournament_menu_exists(self):
        import main

        keys = [key for key, _name, _fn in main.MENU]
        self.assertIn("19", keys)
        self.assertIn("20", keys)


class ViewerRobustnessTest(unittest.TestCase):
    """監視ビューアのサーバーが、切断で落ちないことの確認。

    Windows でブラウザが読み込みを途中でやめると
    ConnectionAbortedError が飛び、標準エラーが例外で埋まっていた。
    """

    def test_server_survives_aborted_requests(self):
        import socket
        import time
        import urllib.request

        import main

        server, url = main.start_viewer_server(open_browser=False)
        self.assertIsNotNone(server)
        try:
            for _ in range(5):
                connection = socket.create_connection(
                    ("127.0.0.1", server.server_port), timeout=3
                )
                connection.sendall(b"GET /live.html HTTP/1.1\r\nHost: x\r\n\r\n")
                connection.recv(8)
                connection.close()
            time.sleep(0.3)
            body = urllib.request.urlopen(url, timeout=5).read()
            self.assertIn(b"<canvas", body, "切断のあとサーバーが死んでいます")
        finally:
            server.shutdown()


class FuelLookaheadTest(unittest.TestCase):
    """燃料を明日に残す判断の確認。

    実戦の試合839で、3日目の開始時に燃料が [96, 11, 6, 0] まで枯れ、
    そこから3304案を試しても13系列中12しか取れなかった。
    **当日の探索では解決できない**状態で、原因は前日までの使い切り。

    1日ずつ「種類→日別→玉」で最良を選ぶので、明日の燃料が残らない。
    評価の3番目に「明日動けない巡回車の少なさ」を入れた。
    日別累計より下、玉数より上の順位にしてある。

    ただし**燃料に余裕がある試合でだけ**効かせる。実戦9試合で測ると、
    余裕のない試合では玉を落とすだけだった（試合836で782→647）。
    """

    def test_margin_reflects_the_fuel_budget(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.production import _fuel_margin

        settings = honsen_settings("1回戦", random.Random(3), players=10, day_count=5)
        problem = gen(2026, settings)
        margin = _fuel_margin(problem)
        expected = problem.fuel_limit / (sum(problem.day_steps) / 2.0)
        self.assertAlmostEqual(margin, expected, places=6)
        self.assertGreater(margin, 0.0)

    def test_shortfall_counts_only_touring_cars(self):
        from hexa.production import _fuel_shortfall
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        kinds = (AgentKind.TOURING, AgentKind.TOURING, AgentKind.SUPPLY)
        result = DayResult(
            positions=(0, 0, 0), fuels=(0, 100, 0), acquisitions=[], refuels=[],
            stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        # 1台目だけが足りない。補給車は数えない。
        self.assertEqual(_fuel_shortfall(result, kinds, 100, 80), 1)
        # 先読みしないときは常に0
        self.assertEqual(_fuel_shortfall(result, kinds, 100, 0), 0)

    def test_key_puts_fuel_between_brands_and_noodles(self):
        """燃料は日別累計より下、玉数より上であること。"""

        from hexa.production import _candidate_key
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        kinds = (AgentKind.TOURING,)

        def make(brands, noodles, fuel):
            events = [type("E", (), {"brand": b})() for b in brands]
            events += [type("E", (), {"brand": brands[0]})() for _ in range(noodles - len(brands))]
            return DayResult(
                positions=(0,), fuels=(fuel,), acquisitions=events, refuels=[],
                stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
                step_positions=[], step_fuels=[],
            )

        # 日別が多いほうが勝つ（燃料が少なくても）
        many_brands = _candidate_key(make([0, 1, 2], 3, 0), set(), 100, kinds, 80)
        few_brands = _candidate_key(make([0], 3, 100), set(), 100, kinds, 80)
        self.assertGreater(many_brands, few_brands)

        # 日別が同じなら、燃料が残るほうが勝つ（玉が少なくても）
        keep_fuel = _candidate_key(make([0, 1], 2, 100), set(), 100, kinds, 80)
        more_balls = _candidate_key(make([0, 1], 9, 0), set(), 100, kinds, 80)
        self.assertGreater(keep_fuel, more_balls)

    def test_lookahead_is_off_when_fuel_is_tight(self):
        """燃料に余裕がない試合では先読みを使わないこと。"""

        import dataclasses

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.production import FUEL_LOOKAHEAD_MARGIN, _fuel_margin

        settings = honsen_settings("決勝", random.Random(5), players=10, day_count=10)
        # 歩数を伸ばして燃料を相対的に細らせる
        settings = dataclasses.replace(
            settings, day_steps_low_ratio=3.0, day_steps_high_ratio=4.0,
            fuel_ratio_low=1.0, fuel_ratio_high=1.2,
        )
        problem = gen(2026, settings)
        self.assertLess(_fuel_margin(problem), FUEL_LOOKAHEAD_MARGIN)


class SharedSpotTest(unittest.TestCase):
    """在庫の多い店に複数台が向かえることの確認。

    実戦の32x32で、1日目に32店すべてを訪問しているのに
    **取り切ったのは6店だけ**だった。取り逃し65玉のうち53玉が
    在庫5以上の店。「1台でも予約していれば他は行けない」という
    作りだったため、在庫7の店に1台しか行かず6玉を捨てていた。

    ただし**今日まだ誰も取っていない系列の店は1台だけ**にする。
    2台目を向けると別の系列を取る車が減り、日別累計が落ちるため。

    実戦9試合での合計:
      修正前          日別1099 / 玉2958
      無条件に重ねる  日別1095 / 玉2986  ← 日別が落ちる
      系列を守って重ねる 日別1099 / 玉2986  ← 採用
    """

    def _policy(self, seed=3):
        from hexa.ai import Policy, new_population
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("決勝", random.Random(seed),
                                            players=10, day_count=6))
        brain = new_population(count=2, capacity="simple")[0]
        return problem, Policy(brain, problem, random.Random(0), 0.0)

    def test_rich_spot_accepts_more_than_one_car(self):
        problem, policy = self._policy()
        finder = PathFinder(problem)
        rich = max(problem.spots, key=lambda s: s.stocks)
        if rich.stocks < 2:
            self.skipTest("在庫2以上の店がありません")
        stock = {spot.position: spot.stocks for spot in problem.spots}
        # その系列は今日すでに取ってある、という状態にする
        decision = policy.choose_touring_target(
            agent_index=0, step=0, remaining=problem.day_steps[0],
            position=problem.agent_positions[0], fuel=problem.fuel_limit,
            stock=stock, reserved={rich.position: 1},
            acquired_today=set(), acquired_brands=set(),
            brands_today={rich.brand}, road_status={}, finder=finder,
            kinds=(), positions=list(problem.agent_positions),
            day_index=0, day_count=problem.day_count,
        )
        # 1台が予約済みでも、在庫が残っていれば候補から外れない
        features = decision.candidate_features
        self.assertTrue(features, "候補が1つもありません")

    def test_fresh_brand_spot_takes_only_one_car(self):
        """今日まだ取っていない系列の店には、2台目を向かわせないこと。"""

        from hexa.ai import Policy, new_population
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("決勝", random.Random(5),
                                            players=10, day_count=6))
        brain = new_population(count=2, capacity="simple")[0]
        policy = Policy(brain, problem, random.Random(0), 0.0)
        finder = PathFinder(problem)
        target = max(problem.spots, key=lambda s: s.stocks)
        stock = {spot.position: spot.stocks for spot in problem.spots}

        decision = policy.choose_touring_target(
            agent_index=0, step=0, remaining=problem.day_steps[0],
            position=problem.agent_positions[0], fuel=problem.fuel_limit,
            stock=stock, reserved={target.position: 1},
            acquired_today=set(), acquired_brands=set(),
            brands_today=set(),            # まだ何も取っていない
            road_status={}, finder=finder, kinds=(),
            positions=list(problem.agent_positions),
            day_index=0, day_count=problem.day_count,
        )
        self.assertNotEqual(decision.target, target.position,
                            "未取得の系列の店に2台目を向けています")

    def test_greedy_follows_the_same_rule(self):
        import inspect

        from hexa.production import GreedyPolicy

        source = inspect.getsource(GreedyPolicy.choose_touring_target)
        self.assertIn("limit = 1 if brand not in brands_today", source)


class FuelMarginTest(unittest.TestCase):
    """燃料の余裕の見積もりの確認。

    実戦9試合を調べ、2つの誤りを見つけた。

    **補給車を数えていなかった** ── 補給車の燃料は無限で、合流すれば
    満タンにできる。それを無視していたので「余裕なし」と判定され、
    燃料を残す判断が働かなかった。試合833は見かけ0.28倍が、
    補給車1台を数えると2.82倍になる。

    **残す量の基準が甘すぎた** ── 「明日の歩数の4分の1」としていたが、
    平地は2歩で燃料1なので、全部動くには歩数の半分が要る。
    試合833の7日目（32歩）は開始燃料15で系列を落としていた。

    この2つを直して、9試合中8試合で日別累計が1位と並ぶようになった
    （作業前は4試合）。
    """

    def _problem(self, seed=3, days=6):
        from hexa.generator import generate_problem as gen, honsen_settings

        return gen(2026, honsen_settings("1回戦", random.Random(seed),
                                         players=10, day_count=days))

    def test_supply_cars_raise_the_margin(self):
        from hexa.production import _fuel_margin

        problem = self._problem()
        alone = _fuel_margin(problem, 0)
        with_supply = _fuel_margin(problem, 1)
        self.assertGreater(with_supply, alone, "補給車を数えていません")
        # 補給車2台なら、さらに増える
        self.assertGreater(_fuel_margin(problem, 2), with_supply)

    def test_margin_matches_the_formula(self):
        from hexa.production import _fuel_margin

        problem = self._problem(seed=7)
        need = sum(problem.day_steps) / 2.0
        supplied = problem.fuel_limit * 1 * max(0, problem.day_count - 1)
        expected = (problem.fuel_limit + supplied) / need
        self.assertAlmostEqual(_fuel_margin(problem, 1), expected, places=6)

    def test_reserve_is_half_the_steps(self):
        """残す燃料の目安が『明日の歩数の半分』であること。"""

        from hexa.production import _fuel_shortfall
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        kinds = (AgentKind.TOURING, AgentKind.TOURING)
        # 明日が40歩なら、20あれば足りる。19なら足りない。
        enough = DayResult(
            positions=(0, 0), fuels=(20, 20), acquisitions=[], refuels=[],
            stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        short = DayResult(
            positions=(0, 0), fuels=(19, 19), acquisitions=[], refuels=[],
            stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        self.assertEqual(_fuel_shortfall(enough, kinds, 100, 40), 0)
        self.assertEqual(_fuel_shortfall(short, kinds, 100, 40), 2)

    def test_production_passes_the_supply_count(self):
        import inspect

        from hexa.production import play_production_match

        source = " ".join(inspect.getsource(play_production_match).split())
        self.assertIn("_fuel_margin( problem, sum(1 for k in kinds", source,
                      "補給車の数を渡していません")


class MonitorRetryTest(unittest.TestCase):
    """監視の書き出しが、たまの失敗で止まらないことの確認。

    実戦で「途中からビューアが更新されなくなる」という報告があった。
    原因は、書き込みに1回でも失敗すると enabled を False にして
    **二度と再開しなかった**こと。Windows では、ブラウザやウイルス対策が
    ファイルを開いている一瞬に os.replace が失敗することがある。

    3回まで即座に再試行し、20回続けて失敗したときだけ諦める。
    """

    def _problem(self):
        from hexa.generator import generate_problem as gen, honsen_settings

        return gen(2026, honsen_settings("1回戦", random.Random(3),
                                         players=10, day_count=4))

    def test_survives_intermittent_failures(self):
        import json
        import os
        import tempfile

        from hexa.livemonitor import LIVE_NAME, LiveMonitor
        from hexa.rules import AgentKind

        problem = self._problem()
        kinds = tuple(AgentKind.TOURING for _ in range(problem.agent_count))
        with tempfile.TemporaryDirectory() as directory:
            monitor = LiveMonitor(Path(directory))
            monitor.start_match(problem, kinds)
            real = os.replace
            calls = [0]

            def flaky(source, target):
                calls[0] += 1
                if calls[0] % 2 == 1:
                    raise OSError("わざと失敗")
                return real(source, target)

            os.replace = flaky
            try:
                for index in range(10):
                    monitor.searching(0, index, "x", 10.0)
                    monitor._last_search = 0
            finally:
                os.replace = real

            self.assertTrue(monitor.enabled, "1回の失敗で止まっています")
            data = json.loads((Path(directory) / LIVE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(data["status"]["candidates"], 9,
                             "最後の書き出しが届いていません")

    def test_no_temporary_files_are_left(self):
        import os
        import tempfile

        from hexa.livemonitor import LiveMonitor
        from hexa.rules import AgentKind

        problem = self._problem()
        kinds = tuple(AgentKind.TOURING for _ in range(problem.agent_count))
        with tempfile.TemporaryDirectory() as directory:
            monitor = LiveMonitor(Path(directory))
            monitor.start_match(problem, kinds)
            real = os.replace
            os.replace = lambda s, t: (_ for _ in ()).throw(OSError("常に失敗"))
            try:
                for _ in range(5):
                    monitor.searching(0, 1, "x", 1.0)
                    monitor._last_search = 0
            finally:
                os.replace = real
            left = [f for f in os.listdir(directory) if f.startswith(".live")]
            self.assertEqual(left, [], "一時ファイルが残っています")

    def test_gives_up_only_after_many_failures(self):
        import os
        import tempfile

        from hexa.livemonitor import LiveMonitor
        from hexa.rules import AgentKind

        problem = self._problem()
        kinds = tuple(AgentKind.TOURING for _ in range(problem.agent_count))
        with tempfile.TemporaryDirectory() as directory:
            monitor = LiveMonitor(Path(directory))
            monitor.start_match(problem, kinds)
            real = os.replace
            os.replace = lambda s, t: (_ for _ in ()).throw(OSError("常に失敗"))
            try:
                for _ in range(5):
                    monitor.searching(0, 1, "x", 1.0)
                    monitor._last_search = 0
                self.assertTrue(monitor.enabled, "5回で諦めるのは早すぎます")
                for _ in range(25):
                    monitor.searching(0, 1, "x", 1.0)
                    monitor._last_search = 0
                self.assertFalse(monitor.enabled, "いつまでも諦めないのも困ります")
            finally:
                os.replace = real


class PositionLookaheadTest(unittest.TestCase):
    """位置の先読みが既定で無効であることの確認。

    試合838の3日目（34歩）が14系列に届かないのは「前日の終わり方」が
    原因だと見立てて実装したが、**9試合すべてで結果が同一**だった。

    理由は、計画がすでに店の近くで1日を終えているため。最後に訪れるのが
    店なので、終了位置は自然と店の上か隣になる。計算を増やすだけなので
    無効にしてある。
    """

    def test_disabled_by_default(self):
        from hexa.production import POSITION_REACH_RATIO

        self.assertEqual(POSITION_REACH_RATIO, 0.0)

    def test_penalty_is_zero_when_disabled(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.production import _position_penalty
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING,)
        result = DayResult(
            positions=(0,), fuels=(10,), acquisitions=[], refuels=[],
            stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        self.assertEqual(
            _position_penalty(result, kinds, problem, finder, 100), 0
        )
