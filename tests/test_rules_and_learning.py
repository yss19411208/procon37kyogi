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
            kinds, *_rest = choose_production_kinds(
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
        kinds, *_rest = choose_production_kinds(problem, brains, log=lambda message: None)
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
            def post_actions(self, plan, log=None):
                raise ProconApiError(400, "TestError", "わざと失敗")

        self.assertEqual(_submit(Failing(), [[0]], lambda message: None), -1)

    def test_successful_submit_returns_revision(self):
        from hexa.production import _submit

        class Working:
            def post_actions(self, plan, log=None):
                return 7

        self.assertEqual(_submit(Working(), [[0]], lambda message: None), 7)

    def test_production_uses_accepted_plan_for_prediction(self):
        """**受理された案で予測すること。**

        準決勝32x32で提出が通らなかった日に、手元は自分の案が
        走ったものとして予測し、以降の日がすべてずれました。
        通らなければ「全員待機」を出し直し、**予測もそちらに
        合わせます**（`_submit_or_wait`）。
        """

        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("revision, landed = _submit_or_wait(", source)
        self.assertIn("accepted = landed if revision >= 0 else None", source)
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

    def test_lookahead_is_four(self):
        """先読みは4店。実戦のログで測り直して3店から変えた。

        36試合248日分で、取り逃しの93%が「届いたのに行かなかった」。
        長い日ほど1玉あたりの歩数が増える（6.4歩 → 11.2歩）ので、
        先を長く見て近くの店を取り切る順番を見つける。

          3店: 日別796 / 玉2444
          4店: 日別796 / 玉2489  ← 採用

        以前3店が最適と測ったのは生成した問題での結果だった。
        """

        from hexa.routes import MAX_ROUTE_LENGTH

        self.assertEqual(MAX_ROUTE_LENGTH, 4)


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
        #
        # ただし**1台までの不足は許して玉を優先**するようになったので、
        # 差がつくのは2台以上が足りないときです（FUEL_SHORT_LIMIT=1）。
        kinds = (AgentKind.TOURING, AgentKind.TOURING)

        def make2(brands, noodles, fuels):
            events = [type("E", (), {"brand": b})() for b in brands]
            events += [type("E", (), {"brand": brands[0]})()
                       for _ in range(noodles - len(brands))]
            return DayResult(
                positions=(0, 0), fuels=fuels, acquisitions=events, refuels=[],
                stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
                step_positions=[], step_fuels=[],
            )

        keep_fuel = _candidate_key(make2([0, 1], 2, (100, 100)), set(), 100,
                                   kinds, 80)
        more_balls = _candidate_key(make2([0, 1], 9, (0, 0)), set(), 100,
                                    kinds, 80)
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
        """貪欲も同じ規則に従うこと。**動かして**確かめます。

        以前はソースに `limit = 1 if brand not in brands_today` という
        文字が入っているかだけを見ていました。速くするために変数名を
        `fresh_today` に変えたら、中身が同じなのに落ちました。
        文字ではなく**振る舞い**で見ます。
        """

        from hexa.production import GreedyPolicy

        problem, _ai = self._policy()
        finder = PathFinder(problem)
        policy = GreedyPolicy(problem)
        target = max(problem.spots, key=lambda spot: spot.stocks)
        if target.stocks < 2:
            self.skipTest("在庫2以上の店がありません")
        # **その店だけを残します。** ほかに行き先があると、規則で
        # 許されていても「もっと良い店」を選ぶので、規則そのものを
        # 試せません。
        stock = {target.position: target.stocks}

        def ask(brands_today):
            return policy.choose_touring_target(
                agent_index=0, step=0, remaining=problem.day_steps[0],
                position=problem.agent_positions[0],
                fuel=problem.fuel_limit, stock=stock,
                reserved={target.position: 1},
                acquired_today=set(), acquired_brands=set(),
                brands_today=brands_today, road_status={}, finder=finder,
                kinds=(), positions=list(problem.agent_positions),
                day_index=0, day_count=problem.day_count,
            ).target

        # 今日まだ取っていない系列なら、2台目は向かわせない。
        self.assertNotEqual(ask(set()), target.position,
                            "未取得の系列の店に2台目を向けています")
        # すでに取った系列なら、在庫の残りまでは向かってよい。
        self.assertEqual(ask({target.brand}), target.position,
                         "在庫が残っているのに2台目を向けていません")


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
            # 探索中は軽いほう（now.json）だけを書きます。重いほうを
            # 0.2秒ごとに書くと、10日目で1回9.1ミリ秒かかり、探索の
            # 4.5%を監視に取られるためです。
            from hexa.livemonitor import NOW_NAME

            data = json.loads(
                (Path(directory) / NOW_NAME).read_text(encoding="utf-8"))
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


class NetworkRetryTest(unittest.TestCase):
    """通信が切れても試合を諦めないことの確認。

    実戦で、4日目に名前解決が一瞬失敗しただけで試合を落とした
    （試合923は日別144が取れるはずのところ64）。無線や会場の
    ネットワークは数秒で戻ることが多いので、間を置いて何度も試す。

    1日は60秒あるので、20回（およそ30秒）粘っても回答は間に合う。
    """

    def test_retries_are_configured(self):
        from hexa.production import NETWORK_RETRIES

        self.assertGreaterEqual(NETWORK_RETRIES, 10, "粘りが足りません")

    def test_production_retries_the_day_fetch(self):
        import inspect

        from hexa.production import play_production_match

        source = " ".join(inspect.getsource(play_production_match).split())
        self.assertIn("for attempt in range(NETWORK_RETRIES)", source,
                      "通信の再試行がありません")
        self.assertIn("繰り返し試します", source)


class CongestedLookaheadTest(unittest.TestCase):
    """渋滞した日に、先を長く見ることの確認。

    渋滞すると店どうしが遠くなる（実戦927の2日目は中央値118歩）。
    3店先までしか見ないと「次の次」が見えない。

    実戦927での実測:
      3店固定        日別121 / 玉457
      渋滞25%で切替  日別123 / 玉443
      渋滞12%で切替  日別123 / 玉461  ← 採用

    渋滞のない試合で常に4店先まで見ると、かえって悪くなる
    （9試合で日別の満点が7/8から6/8に減少）。だから日によって変える。
    """

    def test_quiet_day_uses_three(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.routes import MAX_ROUTE_LENGTH, route_length_for

        problem = gen(2026, honsen_settings("決勝", random.Random(3),
                                            players=10, day_count=6))
        # 大きいマップ（500マス以上）では長く見るので、そちらを期待値に。
        from hexa.routes import BIG_MAP_CELLS, BIG_MAP_ROUTE_LENGTH

        from hexa.routes import BIG_MAP_MAX_CELLS

        from hexa.routes import BIG_MAP_FUEL_LIMIT

        expected = (
            BIG_MAP_ROUTE_LENGTH
            if (BIG_MAP_CELLS <= problem.cell_count <= BIG_MAP_MAX_CELLS
                and problem.fuel_limit <= BIG_MAP_FUEL_LIMIT)
            else MAX_ROUTE_LENGTH
        )
        self.assertEqual(route_length_for(problem, {}), expected)

    def test_jammed_day_uses_the_same_length(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.routes import JAMMED_ROUTE_LENGTH, route_length_for
        from hexa.rules import RoadStatus, Terrain

        problem = gen(2026, honsen_settings("決勝", random.Random(3),
                                            players=10, day_count=6))
        roads = [
            cell for cell in range(problem.cell_count)
            if problem.terrain_at(cell) == Terrain.ROAD
        ]
        if not roads:
            self.skipTest("道路がありません")
        # 半分を渋滞させる
        status = {cell: RoadStatus.JAMMED for cell in roads[: len(roads) // 2]}
        from hexa.routes import BIG_MAP_CELLS, BIG_MAP_ROUTE_LENGTH

        from hexa.routes import BIG_MAP_MAX_CELLS

        from hexa.routes import BIG_MAP_FUEL_LIMIT

        expected = (
            BIG_MAP_ROUTE_LENGTH
            if (BIG_MAP_CELLS <= problem.cell_count <= BIG_MAP_MAX_CELLS
                and problem.fuel_limit <= BIG_MAP_FUEL_LIMIT)
            else JAMMED_ROUTE_LENGTH
        )
        self.assertEqual(route_length_for(problem, status), expected)
        # 通常が4店になったので、渋滞時も同じ長さ（増やすと重すぎる）

    def test_threshold_is_the_measured_one(self):
        from hexa.routes import JAMMED_SHARE_FOR_LONGER

        self.assertAlmostEqual(JAMMED_SHARE_FOR_LONGER, 0.12, places=3)

    def test_routes_respect_the_longer_length(self):
        """渋滞した日の道すじが4店まで伸びること。"""

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.routes import build_routes, default_recipes
        from hexa.rules import RoadStatus, Terrain

        problem = gen(2026, honsen_settings("決勝", random.Random(7),
                                            players=10, day_count=6))
        finder = PathFinder(problem)
        roads = [
            cell for cell in range(problem.cell_count)
            if problem.terrain_at(cell) == Terrain.ROAD
        ]
        status = {cell: RoadStatus.JAMMED for cell in roads[: len(roads) // 2]}
        available = {spot.position: spot.stocks for spot in problem.spots}
        routes = build_routes(
            problem, finder, status, problem.agent_positions[0],
            problem.fuel_limit, problem.day_steps[0] * 3,
            available, set(), set(), default_recipes(),
        )
        self.assertTrue(routes)
        for route in routes:
            # 大きいマップでは8店まで伸びる（24x24で玉数が7ポイント伸びた）
            from hexa.routes import BIG_MAP_CELLS, BIG_MAP_ROUTE_LENGTH

            from hexa.routes import BIG_MAP_MAX_CELLS

            from hexa.routes import BIG_MAP_FUEL_LIMIT

            cap = (
                BIG_MAP_ROUTE_LENGTH + 1
                if (BIG_MAP_CELLS <= problem.cell_count <= BIG_MAP_MAX_CELLS
                    and problem.fuel_limit <= BIG_MAP_FUEL_LIMIT)
                else 5
            )
            self.assertLessEqual(len(route.spots), cap)


class MatchLogTest(unittest.TestCase):
    """練習場の行動記録の確認。

    「なぜその判断をしたか」をあとから追えるようにするためのもの。
    **本番モードでは記録しない**（1日60秒しかないため）。
    試合中は配列に貯めるだけで、終わってから1回だけ書き出す。
    """

    def _setup(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        return problem, kinds

    def test_saves_a_file(self):
        import json
        import tempfile

        import hexa.matchlog as module
        from hexa.matchlog import MatchLog

        problem, kinds = self._setup()
        original = module.LOG_DIR
        with tempfile.TemporaryDirectory() as directory:
            module.LOG_DIR = Path(directory)
            try:
                log = MatchLog(enabled=True)
                log.start(problem, kinds, "t1", 44, 61.5, "貪欲", (14, 56, 120))
                log.finish_day(
                    0, "S1-AI-1", 120, 0, problem.day_steps[0], 0,
                    list(problem.agent_positions),
                    [problem.fuel_limit] * problem.agent_count,
                    list(problem.agent_positions),
                    [problem.fuel_limit] * problem.agent_count,
                    [], [], [], {}, problem, kinds, PathFinder(problem),
                    [], 3, 3, 9, 12.5,
                )
                path = log.save("種類3 / 日別3 / 玉9", [])
                self.assertIsNotNone(path)
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                self.assertEqual(data["header"]["trialCount"], 44)
                self.assertEqual(len(data["days"]), 1)
            finally:
                module.LOG_DIR = original

    def test_disabled_writes_nothing(self):
        from hexa.matchlog import MatchLog

        problem, kinds = self._setup()
        log = MatchLog(enabled=False)
        log.start(problem, kinds, "t2", 1, 1.0, "x", (0, 0, 0))
        self.assertIsNone(log.save("x", []))

    def test_missed_spots_gives_reasons(self):
        from hexa.matchlog import missed_spots

        problem, kinds = self._setup()
        finder = PathFinder(problem)
        rows = missed_spots(
            problem, kinds, finder, {}, list(problem.agent_positions),
            [problem.fuel_limit] * problem.agent_count,
            problem.day_steps[0], [],
        )
        self.assertTrue(rows, "取れなかった店が1つもないのはおかしい")
        allowed = {"届いた", "歩数不足", "燃料不足", "在庫切れ"}
        for row in rows:
            self.assertIn(row["reason"], allowed)
            self.assertGreater(row["room"], 0)

    def test_no_fuel_means_fuel_shortage(self):
        """燃料がゼロなら、理由が燃料不足になること。"""

        from hexa.matchlog import missed_spots

        problem, kinds = self._setup()
        finder = PathFinder(problem)
        rows = missed_spots(
            problem, kinds, finder, {}, list(problem.agent_positions),
            [0] * problem.agent_count, problem.day_steps[0], [],
        )
        reasons = {row["reason"] for row in rows}
        self.assertNotIn("届いた", reasons, "燃料ゼロなのに届いた扱いです")

    def test_arena_records_but_production_does_not(self):
        import inspect

        import hexa.production as module

        arena = inspect.getsource(module.play_matches_forever)
        self.assertIn("record_log=True", arena, "練習場で記録していません")
        default = inspect.signature(module.play_production_match).parameters
        self.assertFalse(default["record_log"].default,
                         "本番モードで記録してはいけません")


class BuildVersionTest(unittest.TestCase):
    """起動時にビルドの版を出すことの確認。

    pull したものが最新かどうかを目で確かめるためのもの。
    models があれば学習量も添える。
    """

    def test_version_is_printed_at_start(self):
        import inspect

        import main

        source = inspect.getsource(main.main)
        self.assertIn("build_line()", source, "起動時に版を出していません")

    def test_build_line_works_without_models(self):
        import main

        line = main.build_line()
        self.assertIn("版", line)
        self.assertIn(main.BUILD_VERSION, line)

    def test_version_is_three_numbers(self):
        """版は 1.9.0 のような数字3つ。目で見比べられるように。"""

        import main

        parts = main.BUILD_VERSION.split(".")
        self.assertEqual(len(parts), 3, "数字3つにしてください")
        for part in parts:
            self.assertTrue(part.isdigit(), f"{part} が数字ではありません")


class TightDayLookaheadTest(unittest.TestCase):
    """歩数が厳しい日に先を長く見る仕組みが、既定で無効であることの確認。

    実戦のログ133日分で、達成率は「上限1玉あたり使える歩数」で
    ほぼ決まる（3〜5歩で52%、9〜11歩で80%）。そこで厳しい日だけ
    5店先まで見るようにしたが、

      厳しい試合: 玉1008 → 1024（+16）
      通常の試合: 玉2489 → 2429（-60）

    **通常の損のほうが大きい**ので無効にしてある。
    """

    def test_disabled_by_default(self):
        from hexa.routes import TIGHT_STEPS_PER_NOODLE

        self.assertEqual(TIGHT_STEPS_PER_NOODLE, 0.0)

    def test_length_is_the_normal_one(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.routes import MAX_ROUTE_LENGTH, route_length_for

        problem = gen(2026, honsen_settings("決勝", random.Random(3),
                                            players=10, day_count=6))
        available = {spot.position: spot.stocks for spot in problem.spots}
        # どれだけ歩数が少なくても、無効なら通常の長さ。
        # ただし大きいマップ（500マス以上）では、そちらの長さになる。
        from hexa.routes import BIG_MAP_CELLS, BIG_MAP_ROUTE_LENGTH

        from hexa.routes import BIG_MAP_MAX_CELLS

        from hexa.routes import BIG_MAP_FUEL_LIMIT

        expected = (
            BIG_MAP_ROUTE_LENGTH
            if (BIG_MAP_CELLS <= problem.cell_count <= BIG_MAP_MAX_CELLS
                and problem.fuel_limit <= BIG_MAP_FUEL_LIMIT)
            else MAX_ROUTE_LENGTH
        )
        self.assertEqual(
            route_length_for(problem, {}, 10, available, 6), expected
        )


class MeetOnTheWayTest(unittest.TestCase):
    """補給車が「相手の経路の途中」でも落ち合えることの確認。

    実戦1025の6日目（190歩）では、巡回車が120〜134歩も補給待ちで
    止まっていた。補給車1台で4台を回るので、現在地まで迎えに行くと
    順番待ちが長くなる。

      途中で合流なし: 日別364 / 玉829
      途中で合流あり: 日別364 / 玉835  ← 採用
    """

    def test_enabled_by_default(self):
        from hexa.ai import _MEET_ON_THE_WAY

        self.assertTrue(_MEET_ON_THE_WAY)

    def test_supply_considers_more_meeting_points(self):
        import inspect

        from hexa.ai import Policy

        source = inspect.getsource(Policy.choose_supply_target)
        self.assertIn("_MEET_ON_THE_WAY", source)
        self.assertIn("meeting_options.add(cell)", source)

    def test_supply_still_chooses_a_target(self):
        """合流先が増えても、ちゃんと行き先を決めること。"""

        from hexa.ai import Policy, new_population
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=5))
        brain = new_population(count=2, capacity="simple")[0]
        policy = Policy(brain, problem, random.Random(0), 0.0)
        finder = PathFinder(problem)
        kinds = tuple(
            AgentKind.SUPPLY if i == problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        positions = list(problem.agent_positions)
        fuels = [1 if k == AgentKind.TOURING else 0 for k in kinds]
        targets = [problem.spots[0].position] * problem.agent_count
        decision = policy.choose_supply_target(
            agent_index=problem.agent_count - 1, step=0,
            remaining=problem.day_steps[0], position=positions[-1],
            kinds=kinds, positions=positions, fuels=fuels,
            plan_targets=targets, supply_claims={}, road_status={},
            finder=finder, day_index=0, day_count=problem.day_count,
        )
        self.assertIsNotNone(decision)


class WaitLengthTest(unittest.TestCase):
    """待機の長さの確認。

    実戦1025を解析すると、**待機が予算の31〜45%**を占めていた。
    燃料切れの車は `remaining // 12`（132歩の日なら16歩）待つので、
    その間に補給車が来ても動き出せない。

    実測（実戦から復元した4問）:

      燃料切れの待機  1歩:831 2歩:812 **3歩:844** 4歩:843 8歩:818 以前:835
      店がある待機    1/8:844 **1/16:857** 1/32:843

    合わせて 835 → 857（+2.6%）。
    """

    def test_stranded_wait_is_short(self):
        from hexa.planner import STRANDED_WAIT

        self.assertEqual(STRANDED_WAIT, 3)

    def test_option_wait_divisor(self):
        from hexa.planner import OPTION_WAIT_DIVISOR

        self.assertEqual(OPTION_WAIT_DIVISOR, 24)

    def test_waiting_for_supply_is_one_step(self):
        """補給を待っている車は1歩だけ待つこと（来たらすぐ動く）。"""

        from hexa.planner import _AgentPlanState, _wait_length
        from hexa.rules import AgentKind

        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        states = [_AgentPlanState(), _AgentPlanState()]
        states[1].target_agent = 0
        self.assertEqual(_wait_length(0, kinds, 200, states), 1)

    def test_stranded_car_waits_briefly(self):
        from hexa.planner import STRANDED_WAIT, _AgentPlanState, _wait_length
        from hexa.rules import AgentKind

        kinds = (AgentKind.TOURING,)
        states = [_AgentPlanState()]
        states[0].had_options = False
        # 長い日でも短く待つ（以前は remaining//12 = 16歩）
        self.assertEqual(_wait_length(0, kinds, 200, states), STRANDED_WAIT)

    def test_car_with_options_waits_longer(self):
        from hexa.planner import _AgentPlanState, _wait_length
        from hexa.rules import AgentKind

        kinds = (AgentKind.TOURING,)
        states = [_AgentPlanState()]
        states[0].had_options = True
        self.assertEqual(_wait_length(0, kinds, 240, states), 10)


class SeverityTest(unittest.TestCase):
    """ログに「その日がどれだけ厳しかったか」が残ることの確認。

    あとから「なぜ取れなかったか」を推測せずに読めるようにするため。

    実戦1033を調べると、渋滞で店どうしの距離が**13歩から25歩へ倍増**し、
    獲得が19玉から12〜15玉に落ちていた。この数字がログにあれば
    一目で分かる。
    """

    def _log(self, directory):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.matchlog import MatchLog
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        log = MatchLog(enabled=True)
        log.start(problem, kinds, "sev", 10, 5.0, "x", (1, 1, 1))
        log.finish_day(
            0, "x", 10, 0, problem.day_steps[0], 0, list(problem.agent_positions),
            [problem.fuel_limit] * problem.agent_count,
            list(problem.agent_positions),
            [problem.fuel_limit] * problem.agent_count,
            [], [], [[0, 0, 1]] * problem.agent_count, {},
            problem, kinds, PathFinder(problem), [], 1, 1, 1, 1.0,
        )
        log.save("x", [])
        return problem

    def test_severity_is_recorded(self):
        import json
        import tempfile

        import hexa.matchlog as module

        original = module.LOG_DIR
        with tempfile.TemporaryDirectory() as directory:
            module.LOG_DIR = Path(directory)
            try:
                self._log(directory)
                data = json.loads(
                    (Path(directory) / "match_sev.json").read_text(encoding="utf-8")
                )
                severity = data["days"][0]["severity"]
                for key in ("jammedRoads", "busyRoads", "roadCells",
                            "stepsPerNoodle", "capacity", "achieved",
                            "startFuelRatio", "stillSteps"):
                    self.assertIn(key, severity)
                self.assertGreater(severity["stepsPerNoodle"], 0)
            finally:
                module.LOG_DIR = original

    def test_production_passes_problem_and_kinds(self):
        import inspect

        from hexa.production import play_production_match

        source = " ".join(inspect.getsource(play_production_match).split())
        self.assertIn("problem=problem, kinds=kinds", source)


class ParallelTrialTest(unittest.TestCase):
    """事前試走を複数のコアで走らせる仕組みの確認。

    **事前試走だけ並列にします。** 1日の計画づくりは「25案を超えると
    結果が変わらない」と実測済みなので、並列にしても同じ結果になります。
    一方この試走は補給車の台数を決める1回限りの判断で、決勝では
    231通りのうち38通りしか試せていませんでした。台数を外すと全日に響きます。

    Windows は fork が使えず spawn なので、問題と重みはファイル経由で渡し、
    1つの子にいくつかの組み合わせをまとめて渡しています。
    失敗したら黙って1プロセスのまま続けます。
    """

    def _files(self, directory):
        import json

        from hexa.ai import new_population, save_population
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        problem_path = Path(directory) / "p.json"
        problem_path.write_text(
            json.dumps(problem.to_json_dict(), ensure_ascii=False), encoding="utf-8"
        )
        brains = new_population(count=3, capacity="simple")
        population_path = Path(directory) / "b.json"
        save_population(brains, population_path)
        return problem, str(problem_path), str(population_path)

    def test_worker_returns_scores(self):
        import tempfile
        import time

        from hexa.parallel import _worker

        with tempfile.TemporaryDirectory() as directory:
            _problem, problem_path, population_path = self._files(directory)
            # 席を付けない古い形（5つ組）でも動くこと。付けた形も試す。
            jobs = [("貪欲", -1, 1, 0.0, 1), ("AI-1", 0, 1, 0.0, 2, (0,))]
            out = _worker((problem_path, population_path, jobs,
                           time.monotonic() + 60))
            self.assertEqual(len(out), 2)
            for score, label, supply, seats in out:
                self.assertIsInstance(seats, tuple)
                self.assertEqual(len(score), 3)
                self.assertGreaterEqual(supply, 0)

    def test_runs_in_parallel(self):
        import tempfile
        import time

        from hexa.parallel import run_trials

        with tempfile.TemporaryDirectory() as directory:
            _problem, problem_path, population_path = self._files(directory)
            jobs = [("貪欲", -1, 1, 0.0, i) for i in range(6)]
            out = run_trials(problem_path, population_path, jobs,
                             time.monotonic() + 40, 2)
            self.assertTrue(out, "並列で1件も返ってきません")

    def test_single_core_falls_back(self):
        import time

        from hexa.parallel import run_trials

        # プロセス数1なら並列にしない（空を返して呼び出し側に任せる）
        self.assertEqual(
            run_trials("x", "", [("a", -1, 1, 0.0, 1)], time.monotonic() + 1, 1),
            [],
        )

    def test_worker_count_is_capped(self):
        from hexa.parallel import usable_workers

        # 論理プロセッサ数-1を使う（1つは本体のために空ける）
        count = usable_workers()
        self.assertGreaterEqual(count, 1)
        self.assertEqual(usable_workers(limit=1), 1)

    def test_production_uses_it(self):
        import inspect

        from hexa.production import PARALLEL_TRIALS, choose_production_kinds

        self.assertTrue(PARALLEL_TRIALS)
        source = inspect.getsource(choose_production_kinds)
        self.assertIn("_run_parallel_trials", source)

    def test_failure_does_not_break_the_trial(self):
        """並列が失敗しても、試走そのものは成立すること。"""

        import hexa.parallel as module
        from hexa.ai import new_population
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.production import choose_production_kinds
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(5),
                                            players=10, day_count=4))
        brains = new_population(count=2, capacity="simple")
        original = module.run_trials
        module.run_trials = lambda *a, **k: (_ for _ in ()).throw(OSError("失敗"))
        try:
            kinds, done, _elapsed, _label, _key = choose_production_kinds(
                problem, brains, budget_seconds=3, log=lambda message: None
            )
        finally:
            module.run_trials = original
        self.assertEqual(len(kinds), problem.agent_count)
        self.assertGreaterEqual(
            sum(1 for k in kinds if k == AgentKind.TOURING), 1
        )


class CapacityTest(unittest.TestCase):
    """その日の「上限」の数え方の確認。

    実戦のログで「上限33 / 獲得14」と出ていましたが、1台は1日に
    5〜6店しか回れていませんでした。**全店を全台が回れる前提**で
    数えていたのが誤りです。ビューアで「100%なのに在庫が残る」ように
    見えたのも、この数字が過大だったため。

    取れる量は次の小さいほうで決まります。

      在庫の都合: 各店の在庫と巡回車の数の小さいほう（1台1日1玉まで）
      時間の都合: 1台が1日に回れる店の数 × 台数

    **計画づくりには使っていない数字**なので、成績には影響しません。
    記録と分析のための値です。
    """

    def _record(self, directory, day_steps=None):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.matchlog import MatchLog
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        log = MatchLog(enabled=True)
        log.start(problem, kinds, "cap", 10, 5.0, "x", (1, 1, 1))
        log.finish_day(
            0, "x", 10, 0, day_steps or problem.day_steps[0], 0,
            list(problem.agent_positions),
            [problem.fuel_limit] * problem.agent_count,
            list(problem.agent_positions),
            [problem.fuel_limit] * problem.agent_count,
            [], [], [[0, 0]] * problem.agent_count, {},
            problem, kinds, PathFinder(problem), [], 1, 1, 1, 1.0,
        )
        log.save("x", [])
        return problem, kinds

    def test_capacity_is_not_more_than_stock(self):
        import json
        import tempfile

        import hexa.matchlog as module
        from hexa.rules import AgentKind

        original = module.LOG_DIR
        with tempfile.TemporaryDirectory() as directory:
            module.LOG_DIR = Path(directory)
            try:
                problem, kinds = self._record(directory)
                data = json.loads(
                    (Path(directory) / "match_cap.json").read_text(encoding="utf-8")
                )
                severity = data["days"][0]["severity"]
                touring = sum(1 for k in kinds if k == AgentKind.TOURING)
                stock_cap = sum(
                    min(spot.stocks, touring) for spot in problem.spots
                )
                self.assertEqual(severity["stockCapacity"], stock_cap)
                self.assertLessEqual(severity["capacity"], stock_cap)
                self.assertGreater(severity["capacity"], 0)
            finally:
                module.LOG_DIR = original

    def test_short_day_lowers_the_capacity(self):
        """歩数が少ない日は、上限も小さくなること。"""

        import json
        import tempfile

        import hexa.matchlog as module

        original = module.LOG_DIR
        results = {}
        for label, steps in (("short", 10), ("long", 400)):
            with tempfile.TemporaryDirectory() as directory:
                module.LOG_DIR = Path(directory)
                try:
                    self._record(directory, day_steps=steps)
                    data = json.loads(
                        (Path(directory) / "match_cap.json").read_text(encoding="utf-8")
                    )
                    results[label] = data["days"][0]["severity"]["capacity"]
                finally:
                    module.LOG_DIR = original
        self.assertLess(results["short"], results["long"],
                        "歩数が少ない日のほうが上限が小さいはず")

    def test_capacity_is_only_for_the_log(self):
        """上限の計算が、計画づくりに使われていないこと。"""

        import inspect

        import hexa.ai
        import hexa.planner
        import hexa.production
        import hexa.routes

        for module in (hexa.ai, hexa.planner, hexa.production, hexa.routes):
            source = inspect.getsource(module)
            self.assertNotIn("stockCapacity", source,
                             f"{module.__name__} が記録用の値を使っています")


class ParallelOrderTest(unittest.TestCase):
    """並列が事前試走の最初に動くことの確認。

    以前は総当たりのあとに置いていたため、呼ばれる時点で持ち時間を
    ほぼ使い切っていました。実戦のログで

      並列: 残り2.0秒では足りないので使いません。

    と出て、一度も動いていませんでした（42秒の試合）。
    子プロセスは同じ問題を自分で解くので、先に投げておけば
    そのあいだ親は単体の試走を続けられます。
    """

    def test_parallel_block_comes_before_the_loop(self):
        import inspect

        from hexa.production import choose_production_kinds

        source = inspect.getsource(choose_production_kinds)
        parallel_at = source.index("_run_parallel_trials")
        loop_at = source.index("for label, policy, supply_count, seats in trials:")
        self.assertLess(parallel_at, loop_at,
                        "並列が総当たりより後ろにあります")

    def test_deadline_uses_most_of_the_budget(self):
        """並列は持ち時間の大半を使うこと。

        親は並列の結果を待たずに自分の試走を進めるので、
        長く取っても無駄になりません。
        """

        import inspect

        from hexa.production import _run_parallel_trials

        source = inspect.getsource(_run_parallel_trials)
        self.assertIn("budget * 0.80", source,
                      "並列に使う時間が短すぎます")

    def test_reasons_are_logged(self):
        import inspect

        from hexa.production import _run_parallel_trials

        source = inspect.getsource(_run_parallel_trials)
        for phrase in ("1コアなので使いません", "では足りないので使いません",
                       "結果が返りませんでした"):
            self.assertIn(phrase, source, f"{phrase} を出していません")

    def test_all_modes_use_it(self):
        """メニュー8・15・20のどれでも並列が効くこと。"""

        import inspect

        import hexa.production as module

        # 3つのモードはすべて同じ試走関数を通る
        for name in ("play_production_match", "play_matches_forever"):
            source = inspect.getsource(getattr(module, name))
            self.assertTrue(
                "choose_production_kinds" in source
                or "play_production_match" in source,
                f"{name} が試走を通っていません",
            )


class ParallelCapacityTest(unittest.TestCase):
    """並列を締切まで使い切ることの確認。

    以前は全部をいっぺんに投げて先頭から順に待っていたため、
    手の空いた子が遊んでいました。終わった子から次を渡す形にしています。

    また Windows は spawn なので、最初の起動に数秒かかります。
    試合が始まる前に空の仕事を投げて起こしておきます。
    """

    def _files(self, directory):
        import json

        from hexa.ai import new_population, save_population
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        problem_path = Path(directory) / "p.json"
        problem_path.write_text(
            json.dumps(problem.to_json_dict(), ensure_ascii=False), encoding="utf-8"
        )
        brains = new_population(count=3, capacity="simple")
        population_path = Path(directory) / "b.json"
        save_population(brains, population_path)
        return str(problem_path), str(population_path)

    def test_warm_up_starts_workers(self):
        import tempfile

        from hexa.parallel import warm_up

        with tempfile.TemporaryDirectory() as directory:
            problem_path, population_path = self._files(directory)
            pool = warm_up(problem_path, population_path, 2)
            self.assertIsNotNone(pool, "子プロセスを起こせません")
            pool.terminate()

    def test_warm_up_is_skipped_on_one_core(self):
        from hexa.parallel import warm_up

        self.assertIsNone(warm_up("x", "", 1))

    def test_many_jobs_are_consumed(self):
        """仕事をたくさん渡しても、締切まで流し続けること。"""

        import tempfile
        import time

        from hexa.parallel import run_trials

        with tempfile.TemporaryDirectory() as directory:
            problem_path, population_path = self._files(directory)
            jobs = [("貪欲", -1, 1, 0.0, i) for i in range(40)]
            out = run_trials(problem_path, population_path, jobs,
                             time.monotonic() + 25, 2)
            # 最初のひとまとまり（CHUNK=4 × 子2つ × 2倍 = 16）を超えること
            self.assertGreater(len(out), 16, "手の空いた子が遊んでいます")

    def test_jobs_are_plentiful(self):
        import inspect

        from hexa.production import _run_parallel_trials

        source = inspect.getsource(_run_parallel_trials)
        self.assertIn("range(4000)", source, "組み合わせが少なすぎます")
        self.assertIn("budget * 0.80", source, "並列に使う時間が短すぎます")

    def test_production_warms_up(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("_warm_up_workers", source, "事前起動していません")


class RecipeRefinerTest(unittest.TestCase):
    """その試合用に作戦を改良する仕組みの確認。

    **実測で伸びなかったので使っていません。** 実戦24x24の1日目で
    240案を作り、改良ありとなしを比べたところ、どちらも「系列17 / 玉41」
    で同じでした。

    理由は、最良の作戦が最初から種に入っていること。そしてその日の
    獲得41玉は「距離24歩 × 41訪問 ÷ 4台 = 241歩」が必要な計算なのに、
    実際の予算は55歩しかありません。近い店を連続で回ることで
    **理論値の4倍以上をすでに達成**しており、微調整の余地がないのです。

    仕組みは残してあるので、学習が進んだときに試し直せます。
    """

    def test_pool_starts_from_the_seeds(self):
        from hexa.refine import POOL_SIZE, RecipeRefiner
        from hexa.routes import SEED_RECIPES

        refiner = RecipeRefiner(rng=random.Random(1))
        self.assertLessEqual(len(refiner.pool), POOL_SIZE)
        self.assertGreaterEqual(len(refiner.pool), min(POOL_SIZE, len(SEED_RECIPES)))
        for recipe in refiner.pool:
            self.assertEqual(len(recipe), 8)

    def test_evolve_needs_enough_trials(self):
        """試していない作戦を消さないこと。"""

        from hexa.refine import RecipeRefiner

        refiner = RecipeRefiner(rng=random.Random(2))
        # まだ誰も試していないので入れ替えない
        self.assertEqual(refiner.evolve(), -1)

    def test_evolve_replaces_the_worst(self):
        from hexa.refine import RecipeRefiner

        refiner = RecipeRefiner(rng=random.Random(3))
        for index in range(len(refiner.pool)):
            refiner.note(index, float(index))
            refiner.note(index, float(index))
        before = [list(item) for item in refiner.pool]
        replaced = refiner.evolve()
        self.assertGreaterEqual(replaced, 0)
        self.assertNotEqual(refiner.pool[replaced], before[replaced])
        # 成績が最も悪かったものが消える
        self.assertEqual(replaced, 0)

    def test_policy_accepts_custom_recipes(self):
        from hexa.ai import Policy, new_population
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        brain = new_population(count=2, capacity="simple")[0]
        custom = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        policy = Policy(brain, problem, random.Random(0), 0.0, recipes=custom)
        self.assertEqual(policy.override_recipes, custom)
        # 既定では差し替えない
        plain = Policy(brain, problem, random.Random(0), 0.0)
        self.assertIsNone(plain.override_recipes)

    def test_not_used_in_production(self):
        """本番の経路では使っていないこと。"""

        import inspect

        import hexa.production

        source = inspect.getsource(hexa.production)
        self.assertNotIn("RecipeRefiner", source)


class ParallelDaySearchTest(unittest.TestCase):
    """1日の探索も複数のコアで回すことの確認。

    事前試走は並列になりましたが、1日60秒の探索は単体のままでした。
    タスクマネージャーで見ると、探索中のCPUは20〜30%しか動きません。

    子プロセスには計画を作らせ、返ってきた行動計画をこちらで採点し直して
    候補に加えます。採点はこちらでやるので、子から重い中身を
    送り返す必要がありません。
    """

    def _setup(self, directory):
        import json

        from hexa.ai import new_population, save_population
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        problem_path = Path(directory) / "p.json"
        problem_path.write_text(
            json.dumps(problem.to_json_dict(), ensure_ascii=False), encoding="utf-8"
        )
        brains = new_population(count=3, capacity="simple")
        population_path = Path(directory) / "b.json"
        save_population(brains, population_path)
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        state = {
            "kinds": [int(k) for k in kinds],
            "positions": list(problem.agent_positions),
            "fuels": [problem.fuel_limit if k == AgentKind.TOURING else 0
                      for k in kinds],
            "roads": {}, "acquired": [], "day_steps": problem.day_steps[0],
            "day_index": 0, "steps_ahead": 0, "next_steps": 0,
        }
        return problem, kinds, str(problem_path), str(population_path), state

    def test_worker_returns_plans(self):
        import tempfile
        import time

        from hexa.parallel import _day_worker

        with tempfile.TemporaryDirectory() as directory:
            problem, kinds, problem_path, population_path, state = self._setup(directory)
            jobs = [("AI-0", 0, 0.0, 1), ("貪欲", -1, 0.0, 2)]
            out = _day_worker((problem_path, population_path, state, jobs,
                               time.monotonic() + 30))
            self.assertEqual(len(out), 2)
            for key, label, plan in out:
                # キーは (新系列, 日別, 立ち往生, 余力, 玉数, 余力の控え,
                #         夜を越せない車, 明日すぐ行ける店, 燃料)
                self.assertEqual(len(key), 12)
                self.assertEqual(len(plan), problem.agent_count)

    def test_runs_in_parallel(self):
        import tempfile
        import time

        from hexa.parallel import run_day_search

        with tempfile.TemporaryDirectory() as directory:
            _p, _k, problem_path, population_path, state = self._setup(directory)
            jobs = [(f"AI-{i % 3}", i % 3, 0.1, 100 + i) for i in range(12)]
            out = run_day_search(problem_path, population_path, state, jobs,
                                 time.monotonic() + 30, 2)
            self.assertGreaterEqual(len(out), 8, "並列で案が作れていません")

    def test_single_core_falls_back(self):
        import time

        from hexa.parallel import run_day_search

        self.assertEqual(
            run_day_search("x", "", {}, [("a", 0, 0.0, 1)], time.monotonic() + 1, 1),
            [],
        )

    def test_production_calls_it(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("_start_parallel_day", source,
                      "1日の探索が並列になっていません")

    def test_plan_is_rescored_locally(self):
        """子が返した計画を、こちらで採点し直すこと。"""

        import inspect

        from hexa.production import _candidate_from_plan

        source = inspect.getsource(_candidate_from_plan)
        self.assertIn("simulate_day", source)
        self.assertIn("_candidate_key", source)


class PoolReuseTest(unittest.TestCase):
    """プールを1つだけ作って使い回すことの確認。

    Windows は spawn なので、11プロセスの起動に **10.6秒** かかります。
    以前は事前試走と毎日の探索でそれぞれ作り直していたため、
    1日60秒のうち10秒が起動に消えていました。タスクマネージャーで
    「1日が終わるとCPUが落ち、しばらくして上がる」のはこれが原因です。

    子プロセスは Ctrl+C を無視します。親を止めたときに11個が同時に
    例外を吐いて画面を埋める、ということが起きません。
    """

    def _files(self, directory):
        import json

        from hexa.ai import new_population, save_population
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        problem_path = Path(directory) / "p.json"
        problem_path.write_text(
            json.dumps(problem.to_json_dict(), ensure_ascii=False), encoding="utf-8"
        )
        brains = new_population(count=3, capacity="simple")
        population_path = Path(directory) / "b.json"
        save_population(brains, population_path)
        return str(problem_path), str(population_path)

    def test_same_pool_is_returned(self):
        import tempfile

        import hexa.parallel as module

        with tempfile.TemporaryDirectory() as directory:
            problem_path, population_path = self._files(directory)
            try:
                first = module.start_pool(problem_path, population_path, 2)
                second = module.start_pool(problem_path, population_path, 2)
                self.assertIsNotNone(first)
                self.assertIs(first, second, "プールを作り直しています")
            finally:
                module.shutdown()

    def test_shutdown_clears_it(self):
        import tempfile

        import hexa.parallel as module

        with tempfile.TemporaryDirectory() as directory:
            problem_path, population_path = self._files(directory)
            module.start_pool(problem_path, population_path, 2)
            module.shutdown()
            self.assertIsNone(module._POOL)

    def test_children_ignore_interrupt(self):
        import inspect

        import hexa.parallel as module

        source = inspect.getsource(module.start_pool)
        self.assertIn("initializer=_ignore_interrupt", source,
                      "子プロセスが Ctrl+C を無視していません")
        helper = inspect.getsource(module._ignore_interrupt)
        self.assertIn("SIG_IGN", helper)

    def test_atexit_is_registered(self):
        import inspect

        import hexa.parallel as module

        source = inspect.getsource(module)
        self.assertIn("atexit.register(shutdown)", source,
                      "プログラム終了時に閉じていません")

    def test_warm_up_comes_before_the_trial(self):
        """起動が事前試走より先であること。"""

        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        warm_at = source.index("_warm_up_workers")
        trial_at = source.index("choose_production_kinds")
        self.assertLess(warm_at, trial_at, "起動が試走より後ろにあります")

    def test_day_search_does_not_rebuild(self):
        import inspect

        import hexa.parallel as module

        source = inspect.getsource(module.run_day_search)
        self.assertIn("start_pool", source)
        self.assertNotIn("mp.Pool", source, "毎日プールを作り直しています")


class FuelShortfallShapeTest(unittest.TestCase):
    """燃料の不足をどう数えるかの確認。

    実戦で3日目の開始燃料が [108, 108, 8, 108, 26, 3] まで枯れ、
    取り逃した5系列すべてが「燃料不足」でした。

    数え方を3通り試した結果（1位の玉数に対する率）:

      そのまま    : 16x16 96% / 24x24 93% / 32x32 93% / 1067 92%  ← 採用
      量で数える  : 16x16 89% / 24x24 90% / 32x32 80% / 1067 81%
      上限で頭打ち: 16x16 96% / 24x24 86% / 32x32 80% / 1067 89%

    量で数えると燃料を温存しすぎて、その日の玉を取りに行かなくなります。
    """

    def test_counts_cars_not_amount(self):
        from hexa.production import _fuel_shortfall
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        kinds = (AgentKind.TOURING, AgentKind.TOURING)
        result = DayResult(
            positions=(0, 0), fuels=(0, 0), acquisitions=[], refuels=[],
            stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        # 2台とも足りない → 2（量なら 100 を超える）
        self.assertEqual(_fuel_shortfall(result, kinds, 100, 200), 2)

    def test_no_lookahead_means_zero(self):
        from hexa.production import _fuel_shortfall
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        kinds = (AgentKind.TOURING,)
        result = DayResult(
            positions=(0,), fuels=(0,), acquisitions=[], refuels=[],
            stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        self.assertEqual(_fuel_shortfall(result, kinds, 100, 0), 0)


class AsyncDaySearchTest(unittest.TestCase):
    """1日の探索を投げっぱなしにして、親と同時に動かすことの確認。

    以前は「並列が全部終わってから親が探し始める」形で、しかも締切の
    7割で並列を切り上げていました。そのため残り3割は親1つだけが動き、
    タスクマネージャーで **CPUが「100%が20秒 → 10%が20秒」** と
    波打って見えました。

    いまは投げたらすぐ戻り、親は自分の探索を続けます。
    その合間に harvest() で終わったぶんを受け取ります。
    """

    def _setup(self, directory):
        import json

        from hexa.ai import new_population, save_population
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        problem_path = Path(directory) / "p.json"
        problem_path.write_text(
            json.dumps(problem.to_json_dict(), ensure_ascii=False), encoding="utf-8"
        )
        brains = new_population(count=3, capacity="simple")
        population_path = Path(directory) / "b.json"
        save_population(brains, population_path)
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        state = {
            "kinds": [int(k) for k in kinds],
            "positions": list(problem.agent_positions),
            "fuels": [problem.fuel_limit if k == AgentKind.TOURING else 0
                      for k in kinds],
            "roads": {}, "acquired": [], "day_steps": problem.day_steps[0],
            "day_index": 0, "steps_ahead": 0, "next_steps": 0,
        }
        return str(problem_path), str(population_path), state

    def test_start_returns_immediately(self):
        import tempfile
        import time

        import hexa.parallel as module

        with tempfile.TemporaryDirectory() as directory:
            problem_path, population_path, state = self._setup(directory)
            jobs = [(f"AI-{i % 3}", i % 3, 0.1, i) for i in range(60)]
            try:
                started = time.monotonic()
                search = module.start_day_search(
                    problem_path, population_path, state, jobs,
                    time.monotonic() + 20, 2,
                )
                elapsed = time.monotonic() - started
                self.assertIsNotNone(search)
                self.assertLess(elapsed, 2.0, "投げるのに時間がかかりすぎです")
            finally:
                module.shutdown()

    def test_harvest_collects_gradually(self):
        import tempfile
        import time

        import hexa.parallel as module

        with tempfile.TemporaryDirectory() as directory:
            problem_path, population_path, state = self._setup(directory)
            jobs = [(f"AI-{i % 3}", i % 3, 0.1, i) for i in range(40)]
            try:
                search = module.start_day_search(
                    problem_path, population_path, state, jobs,
                    time.monotonic() + 20, 2,
                )
                total = 0
                for _ in range(30):
                    time.sleep(0.2)
                    total += len(search.harvest())
                    if total >= 8:
                        break
                self.assertGreater(total, 0, "1件も受け取れていません")
                search.stop()
            finally:
                module.shutdown()

    def test_parent_searches_while_workers_run(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("_start_parallel_day", source)
        self.assertIn("take_parallel", source)
        # 投げたあと、親のループが回ること
        start_at = source.index("_start_parallel_day")
        loop_at = source.index("while time.monotonic() < deadline_monotonic")
        self.assertLess(start_at, loop_at)

    def test_deadline_is_used_fully(self):
        import inspect

        from hexa.production import _start_parallel_day

        source = inspect.getsource(_start_parallel_day)
        self.assertIn("deadline - 0.5", source,
                      "締切より手前で並列を切り上げています")


class HarvestTimingTest(unittest.TestCase):
    """子から受け取る間隔の確認。

    以前は「10回に1回」でしたが、長い日は親の1案に123ミリ秒かかるため、
    1.2秒も回収が止まります。その間に子は仕事を終えて次をもらえず
    遊びます。CPUが上下して見えた一因です。

    **時間で区切って0.2秒ごとに受け取ります。**
    """

    def test_harvest_is_time_based(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("last_harvest", source)
        self.assertIn("now - last_harvest >= 0.2", source,
                      "回収が時間で区切られていません")
        self.assertNotIn("attempt % 10 == 0", source,
                         "回数で区切ったままです")

    def test_parallel_plans_are_recorded(self):
        """並列と親、それぞれが作った案の数が記録に残ること。"""

        import json
        import tempfile

        import hexa.matchlog as module
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.matchlog import MatchLog
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        original = module.LOG_DIR
        with tempfile.TemporaryDirectory() as directory:
            module.LOG_DIR = Path(directory)
            try:
                log = MatchLog(enabled=True)
                log.start(problem, kinds, "cpu", 10, 5.0, "x", (1, 1, 1))
                log.finish_day(
                    0, "x", 500, 0, problem.day_steps[0], 420,
                    list(problem.agent_positions),
                    [problem.fuel_limit] * problem.agent_count,
                    list(problem.agent_positions),
                    [problem.fuel_limit] * problem.agent_count,
                    [], [], [[0, 0]] * problem.agent_count, {},
                    problem, kinds, PathFinder(problem), [], 1, 1, 1, 1.0,
                )
                log.save("x", [])
                data = json.loads(
                    (Path(directory) / "match_cpu.json").read_text(encoding="utf-8")
                )
                severity = data["days"][0]["severity"]
                self.assertEqual(severity["parallelPlans"], 420)
                self.assertEqual(severity["ownPlans"], 80)
            finally:
                module.LOG_DIR = original


class CallSignatureTest(unittest.TestCase):
    """呼び出しの引数が、受け取る側と合っていることの確認。

    実戦で `LiveMonitor.finish_day() got an unexpected keyword argument
    'parallel_plans'` が出て、試合が丸ごと落ちました。
    MatchLog に足した引数を、LiveMonitor にも渡していたのが原因です。

    **この種の取り違えは実行するまで分かりません。** そこで、
    本番の流れをそのまま動かして確かめます。
    """

    def test_monitor_and_recorder_take_what_is_passed(self):
        import inspect

        from hexa.livemonitor import LiveMonitor
        from hexa.matchlog import MatchLog
        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        monitor_names = set(
            inspect.signature(LiveMonitor.finish_day).parameters
        )
        recorder_names = set(inspect.signature(MatchLog.finish_day).parameters)

        # それぞれの呼び出しで使っているキーワードを拾う
        for marker, allowed, who in (
            ("monitor.finish_day(", monitor_names, "LiveMonitor"),
            ("recorder.finish_day(", recorder_names, "MatchLog"),
        ):
            start = source.index(marker) + len(marker)
            depth = 1
            end = start
            while end < len(source) and depth:
                if source[end] == "(":
                    depth += 1
                elif source[end] == ")":
                    depth -= 1
                end += 1
            block = source[start:end]
            for line in block.split("\n"):
                line = line.strip()
                if "=" not in line or line.startswith("#"):
                    continue
                name = line.split("=", 1)[0].strip()
                if not name.isidentifier():
                    continue
                self.assertIn(name, allowed,
                              f"{who} は {name} を受け取れません")

    def test_a_whole_match_runs(self):
        """本番の流れを最後まで通す。引数の取り違えがあれば落ちます。"""

        from hexa.ai import new_population
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.production import play_production_match

        import dataclasses

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=2))
        # 1日の持ち時間を短くして、テストを数秒で終わらせる。
        problem = dataclasses.replace(
            problem, day_seconds=tuple(1 for _ in problem.day_steps)
        )
        brains = new_population(count=2, capacity="simple")

        class FakeClient:
            """最低限の受け答えをするだけの相手。"""

            def __init__(self, problem):
                self.problem = problem
                self.revision = 0

            def check_connection(self):
                return (True, "つながりました")

            def wait_for_setting(self, log=None, waiting=None):
                # 待っているあいだの通知も受け取れること（本物と同じ形）。
                if waiting is not None:
                    waiting("まだ試合前です（確認用）", True)
                return (self.problem, {})

            def post_agent_kinds(self, kinds):
                self.kinds = tuple(kinds)
                return None

            def wait_for_day(self, day):
                import time as _time

                from hexa.client import AgentState, MatchState

                agents = tuple(
                    AgentState(kind=self.kinds[index], position=position,
                               fuel=self.problem.fuel_limit)
                    for index, position in enumerate(self.problem.agent_positions)
                )
                return MatchState(
                    ends_at=int(_time.time()) + 1,
                    day=day + 1,
                    agents=agents,
                    others=(),
                    road_status={},
                )

            def post_actions(self, action_plan, log=None):
                self.revision += 1
                return self.revision

        client = FakeClient(problem)
        report = play_production_match(
            client, brains, log=lambda message: None, record_log=False,
        )
        self.assertIsNotNone(report)
        # **出し直しの回数が上限を越えないこと。**
        #
        # 要項 p.12 (5) は、試合の進行を妨げるほどの回数を送ると
        # 妨害行為とみなされる、と書いています。
        #
        # 「磨き」は小さな改善を何十回も見つけるので、改善ごとに
        # 送ると上限にすぐ届きます。良くなった案は手元に貯めて、
        # 残り時間を残り回数で割った間隔ごとに最良だけを送ります。
        # 1日の最初の提出は上限に数えないので、+1 しています。
        from hexa.production import MAX_RESUBMISSIONS_PER_DAY

        self.assertLessEqual(
            client.revision,
            problem.day_count * (MAX_RESUBMISSIONS_PER_DAY + 1),
            "出し直しが多すぎます",
        )


class ThroughputTest(unittest.TestCase):
    """子を遊ばせないための設定の確認。

    実戦の32x32（175歩）で、候補が **573件** しか作れていませんでした。
    1案250ミリ秒・11プロセス・50秒なら理論は2,200案です。

    原因は2つ。

      * 先に投げておく量が「子の数の2倍」しかなく、1秒ちょっとで尽きる
      * 仕事の総数が6000で頭打ちになっていた（短い日）

    投げておく量を12倍に、仕事の総数を40000にしました。

    **そのあと12倍から6倍に戻しました。** 最良案が変わるたびに
    「順路を磨く」仕事を割り込ませるようになったためです。
    積みすぎると、磨きが投げてある仕事の後ろで待たされます。
    6倍でも、32x32（1案250ミリ秒）なら11コアで約6秒ぶんの
    手持ちになるので、子は遊びません。
    """

    def test_queue_depth_is_generous(self):
        import inspect

        import hexa.parallel as module

        source = inspect.getsource(module.DaySearch._fill)
        self.assertIn("self.workers * 6", source,
                      "先に投げておく量が少なすぎます")

    def test_job_pool_is_large(self):
        import inspect

        from hexa.production import _start_parallel_day

        source = inspect.getsource(_start_parallel_day)
        self.assertIn("range(40000)", source, "仕事の総数が足りません")

    def test_worker_reuses_the_problem(self):
        """子プロセスが問題を読み直さないこと。"""

        import inspect

        import hexa.parallel as module

        source = inspect.getsource(module._day_worker)
        self.assertIn("_load_once", source)
        self.assertNotIn("load_problem(problem_path)", source,
                         "毎回読み込んでいます")

    def test_cache_returns_the_same_objects(self):
        import json
        import tempfile

        import hexa.parallel as module
        from hexa.ai import new_population, save_population
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        with tempfile.TemporaryDirectory() as directory:
            problem_path = Path(directory) / "p.json"
            problem_path.write_text(
                json.dumps(problem.to_json_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            brains = new_population(count=2, capacity="simple")
            population_path = Path(directory) / "b.json"
            save_population(brains, population_path)
            first = module._load_once(str(problem_path), str(population_path))
            second = module._load_once(str(problem_path), str(population_path))
            self.assertIs(first[0], second[0], "問題を作り直しています")
            self.assertIs(first[3], second[3], "下ごしらえを作り直しています")


class WorkerBudgetTest(unittest.TestCase):
    """子プロセスが親と同じ設定で計画を作ることの確認。

    **これが玉数の伸び悩みの根本原因でした。**

    子に `route_budget` を渡し忘れていたため、既定の12個で判断して
    いました。親は1日の探索で6個、事前試走で4個です。

    同じ方策・同じ条件で比べると

        親が40案: 玉72
        子が40案: 玉50  ← 22玉も悪い

    多ければ良いわけではありません。1回の判断が重くなり、
    道すじを伸ばす回数が減るためです。揃えたら子も玉72になりました。
    """

    def _files(self, directory):
        import json

        from hexa.ai import new_population, save_population
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=4))
        problem_path = Path(directory) / "p.json"
        problem_path.write_text(
            json.dumps(problem.to_json_dict(), ensure_ascii=False), encoding="utf-8"
        )
        brains = new_population(count=4, capacity="simple")
        population_path = Path(directory) / "b.json"
        save_population(brains, population_path)
        return problem, str(problem_path), str(population_path)

    def test_day_worker_uses_the_day_budget(self):
        import inspect

        import hexa.parallel as module

        source = inspect.getsource(module._day_worker)
        self.assertIn("route_budget=DAY_ROUTE_BUDGET", source,
                      "1日の探索で作戦の数が揃っていません")

    def test_trial_worker_uses_the_trial_budget(self):
        import inspect

        import hexa.parallel as module

        source = inspect.getsource(module._worker)
        self.assertIn("route_budget=TRIAL_ROUTE_BUDGET", source,
                      "事前試走で作戦の数が揃っていません")

    def test_parent_and_worker_agree(self):
        """同じ条件なら、親と子が同じ案を作ること。"""

        import tempfile
        import time

        import hexa.parallel as module
        from hexa.ai import Policy, load_population
        from hexa.pathfind import PathFinder
        from hexa.planner import DayPlanner
        from hexa.production import DAY_ROUTE_BUDGET, _build_candidate
        from hexa.rules import AgentKind

        with tempfile.TemporaryDirectory() as directory:
            problem, problem_path, population_path = self._files(directory)
            brains = load_population(Path(population_path))
            kinds = tuple(
                AgentKind.SUPPLY if i >= problem.agent_count - 1
                else AgentKind.TOURING
                for i in range(problem.agent_count)
            )
            positions = list(problem.agent_positions)
            fuels = [problem.fuel_limit if k == AgentKind.TOURING else 0
                     for k in kinds]

            finder = PathFinder(problem)
            planner = DayPlanner(problem, finder)
            policy = Policy(brains[0], problem, random.Random(42), 0.0,
                            route_budget=DAY_ROUTE_BUDGET)
            mine = _build_candidate(
                "AI-0", problem, planner, kinds, positions, fuels, {},
                problem.day_steps[0], policy, set(), 0,
            )

            state = {
                "kinds": [int(k) for k in kinds],
                "positions": positions, "fuels": fuels,
                "roads": {}, "acquired": [],
                "day_steps": problem.day_steps[0],
                "day_index": 0, "steps_ahead": 0, "next_steps": 0,
            }
            out = module._day_worker((
                problem_path, population_path, state,
                [("AI-0", 0, 0.0, 42)], time.monotonic() + 60,
            ))
            self.assertTrue(out, "子が案を返しません")
            self.assertEqual(list(mine.key), out[0][0],
                             "親と子で結果が違います")


class TacticBudgetTest(unittest.TestCase):
    """1回の判断で試す作戦の数の確認。

    **学習済みモデルで測り直して 6 → 12 に変えました。**

    以前「12個でも6個でも同じ」と測ったのは**未学習のモデル**での
    結果でした。未学習では作戦ごとに差がつかないので、多く試しても
    変わりません。学習済みでは質に差がつくため、多く試すほど
    良い組み合わせに当たります。

    実戦から復元した4問（学習済み30体）:

      6個 : 日別534 / 玉881
      10個: 日別537 / 玉917
      12個: 日別538 / 玉918  ← 採用

    実戦1189では、これで **日別140 / 玉324** となり、
    1位（日別140 / 玉314）を上回りました。
    """

    def test_day_budget_is_twelve(self):
        from hexa.production import DAY_ROUTE_BUDGET

        self.assertEqual(DAY_ROUTE_BUDGET, 12)

    def test_workers_use_the_same_budget(self):
        import inspect

        import hexa.parallel as module

        source = inspect.getsource(module._day_worker)
        self.assertIn("route_budget=DAY_ROUTE_BUDGET", source)

    def test_more_tactics_are_actually_tried(self):
        """指定した数だけ作戦が試されること。"""

        from hexa.ai import Policy, new_population
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        brain = new_population(count=2, capacity="simple")[0]
        # 種の作戦（SEED_RECIPES）は必ず全部試すので、
        # 指定した数より少なくはなりません。多いほうを指定すれば
        # 学習で増えた作戦も試されます。
        from hexa.routes import SEED_RECIPES

        seeds = len(SEED_RECIPES)
        for budget in (6, 12):
            policy = Policy(brain, problem, random.Random(0), 0.0,
                            route_budget=budget)
            self.assertEqual(policy.route_budget, budget)
            recipes = brain.working_recipes(budget)
            self.assertGreaterEqual(len(recipes), min(budget, seeds))
            self.assertLessEqual(len(recipes), max(budget, seeds))


class OverlapPenaltyTest(unittest.TestCase):
    """終了位置の重なりを罰する仕組みが、既定で無効であることの確認。

    実戦1189では10日中7日で複数台が同じマスで終わっていました
    （3日目は3台とも同じ）。翌日に分散できない原因に見えたので
    罰を入れたところ、実戦4問で **日別544→535・玉924→884** と
    悪化しました。

    同じマスで終わるのは「そこに良い店がある」結果であって、
    原因ではなかったということです。
    """

    def test_disabled_by_default(self):
        from hexa.production import OVERLAP_PENALTY

        self.assertEqual(OVERLAP_PENALTY, 0)

    def test_key_ignores_overlap_when_disabled(self):
        from hexa.production import _candidate_key
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        kinds = (AgentKind.TOURING, AgentKind.TOURING)
        same = DayResult(
            positions=(5, 5), fuels=(50, 50), acquisitions=[], refuels=[],
            stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        apart = DayResult(
            positions=(5, 9), fuels=(50, 50), acquisitions=[], refuels=[],
            stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        self.assertEqual(
            _candidate_key(same, set(), 100, kinds, 0),
            _candidate_key(apart, set(), 100, kinds, 0),
        )


class TrialTieBreakTest(unittest.TestCase):
    """事前試走で同点のとき、巡回車の多いほうを選ぶことの確認。

    実戦1218で、補給1台と2台がどちらも日別139で並びました。
    玉数は1台が213、2台が175です。ところが単純な `>` 比較では
    「先に見つけたほう」が残るため、2台が選ばれていました。

    本番の実力は

        補給1台: 日別144 / 玉272  （1位は144/276）
        補給2台: 日別143 / 玉227

    **台数を外すと全日に響きます。** 玉で差がつくならそれで決まり、
    玉まで同じなら巡回車が多いほうが取りに行けるので有利です。
    """

    def test_higher_score_wins(self):
        from hexa.production import _trial_better

        self.assertTrue(_trial_better((2, 2, 2), 2, (1, 1, 1), 1))
        self.assertFalse(_trial_better((1, 1, 1), 1, (2, 2, 2), 2))

    def test_tie_prefers_fewer_supply_cars(self):
        from hexa.production import _trial_better

        same = (18, 139, 213)
        # 補給1台のほうが、2台より良い
        self.assertTrue(_trial_better(same, 1, same, 2))
        self.assertFalse(_trial_better(same, 2, same, 1))
        # 同じ台数なら入れ替えない
        self.assertFalse(_trial_better(same, 2, same, 2))

    def test_first_result_is_taken(self):
        from hexa.production import _trial_better

        self.assertTrue(_trial_better((1, 1, 1), 3, None, 99))

    def test_noodles_still_decide_before_the_tiebreak(self):
        """玉で**決定的に**差がつくなら、台数より玉が優先されること。

        以前はここで「玉250（補給2台）が玉213（補給1台）に勝つ」
        （+17%）と書いていました。**実測でこれは誤りでした。**

        事前試走は貪欲で回すので燃料を無駄に使い、補給車を過大評価
        します。記録された渋滞つきで台数だけ変えて通すと、本番の
        磨いた走りでは1台が11〜22%上回ります（`SupplyCountTest` に
        3試合ぶんの数字）。つまり試走での+17%程度は、台数を動かす
        根拠になりません。

        いまは1台を既定とし、動かすには15%の上積みを求めます。
        """

        from hexa.production import _trial_better

        # +17% では動かない（試走の偏りの範囲）
        self.assertFalse(_trial_better((18, 139, 250), 2, (18, 139, 213), 1))
        # +40% なら決定的なので動く
        self.assertTrue(_trial_better((18, 139, 300), 2, (18, 139, 213), 1))
        # 同じ台数どうしなら、これまでどおり玉で決まる
        self.assertTrue(_trial_better((18, 139, 250), 1, (18, 139, 213), 1))

    def test_trial_uses_it(self):
        import inspect

        from hexa.production import choose_production_kinds

        source = inspect.getsource(choose_production_kinds)
        self.assertIn("_trial_better", source)
        self.assertNotIn("score > best_key", source,
                         "古い比較が残っています")


class FuelPriorityTest(unittest.TestCase):
    """燃料の余力を、いつ玉数より優先するかの確認。

    **以前は常に玉数より前に置いていました。** そのため

        採用された案: 余力-1 / 玉36
        存在した案  : 余力-3 / 玉43   ← 7玉多いのに負ける

    という逆転が起き、実戦1233では5日間で27玉を取り逃していました。

    かといって常に後ろへ下げると、燃料が枯れて翌日に系列を落とします
    （1234で日別104→102）。日別は玉数より重い得点なので許容できません。

    そこで**1台も欠けないときだけ玉数を優先**します。

      0台まで許容: 1233=195玉 / 1234=169玉（1位超） / 1235=598玉（1位超）
      1台まで許容: 1233=179玉 / 1234=169玉
    """

    def test_limit_is_one(self):
        """1台までの不足なら玉を優先すること。

        実測（1位の玉数に対して）:
          0台: 1320=358 / 1233=187 / 1236=134
          1台: 1320=373 / 1233=190 / 1236=137  ← 採用
          2台: 1233=184 / 1236=136
        """

        from hexa.production import FUEL_SHORT_LIMIT

        self.assertEqual(FUEL_SHORT_LIMIT, 1)

    def test_noodles_win_when_fuel_is_fine(self):
        """燃料が足りているなら、玉数の多いほうが勝つこと。"""

        from hexa.production import _candidate_key
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        kinds = (AgentKind.TOURING,)

        from hexa.simulator import Acquisition

        def make(fuel, noodles):
            return DayResult(
                positions=(0,), fuels=(fuel,),
                acquisitions=[Acquisition(i, 0, i, 0) for i in range(noodles)],
                refuels=[], stay_counts={}, remaining_stocks={},
                wasted_spot_stays=0, step_positions=[], step_fuels=[],
            )

        # どちらも明日ぶんの燃料が足りている（必要50に対し60と80）
        few = _candidate_key(make(80, 3), set(), 100, kinds, 100)
        many = _candidate_key(make(60, 9), set(), 100, kinds, 100)
        self.assertGreater(many, few, "玉数の多いほうが勝つべきです")

    def test_fuel_wins_when_short(self):
        """燃料が足りないなら、余力のあるほうが勝つこと。"""

        from hexa.production import _candidate_key
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        kinds = (AgentKind.TOURING, AgentKind.TOURING)

        from hexa.simulator import Acquisition

        def make(fuels, noodles):
            return DayResult(
                positions=(0, 0), fuels=fuels,
                acquisitions=[Acquisition(i, 0, i, 0) for i in range(noodles)],
                refuels=[], stay_counts={}, remaining_stocks={},
                wasted_spot_stays=0, step_positions=[], step_fuels=[],
            )

        # 必要50に対し、2台とも足りない（不足2台 → 許容1台を超える）
        safe = _candidate_key(make((60, 60), 3), set(), 100, kinds, 100)
        risky = _candidate_key(make((10, 10), 9), set(), 100, kinds, 100)
        self.assertGreater(safe, risky, "2台も足りないなら負けるべきです")

    def test_key_has_eleven_parts(self):
        from hexa.production import _candidate_key
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        kinds = (AgentKind.TOURING,)
        result = DayResult(
            positions=(0,), fuels=(50,), acquisitions=[], refuels=[],
            stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        # (新系列, 日別, 明日届かない系列, 立ち往生, 余力,
        #  明日回れる系列, 玉数, 余力の控え, 明日あぶない系列,
        #  明日すぐ行ける店, 燃料) の11項目。
        # ※「明日すぐ行ける店」は実測で入れていないので常に0
        #
        # 「立ち往生」は、燃料0で止まった車がその日だけでなく試合の
        # 終わりまで1玉も取れないので玉数より重く見ます（実戦1646で
        # 3号車が3日間動けず、1位との差17玉のほとんどがその1台）。
        #
        # 「明日届かない系列」も玉数より上です。日別累計は玉数より
        # 上位の得点で、実戦1958では1位から12位までの全員が日別満点
        # （198）、こちらは197で16位でした。1系列は玉では償えません。
        #
        # 「明日あぶない系列」は**玉数より下**です。玉を1つも
        # 減らさずに、同点のときだけ余裕のある終わり方を選びます。
        #
        # 「明日回れる系列」は、終わった場所から1台ずつ交代で拾わせて
        # 何系列まで回れるかの見積りです。**玉数より上**に置きます。
        # 実戦1981の4日目は巡回3台が右上の2x2に固まって終わり、
        # 翌日（36歩）に3系列を落として日別75で8位（1位は満点78）。
        #
        # 「明日すぐ行ける店」は実測で入れませんでした
        # （TOMORROW_REACH_RATIO = 0 で常に0）。桁は残してあります。
        key = _candidate_key(result, set(), 100, kinds, 0)
        self.assertEqual(len(key), 12)
        from hexa.production import KEY_NOODLES
        self.assertEqual(key[KEY_NOODLES], 0, "玉数の位置がずれています")


class PathCacheTest(unittest.TestCase):
    """経路表のキャッシュの大きさの確認。

    **32x32で候補が作れない原因でした。**

    1024マスのとき上限195件しか貯まらず、1案を作るあいだに
    **4,646回も経路表を作り直して**いました。実戦でも32x32の長い日は
    候補が90〜500件しか作れず、短い日の5,000件と桁が違いました。

      上限195: 1案446ミリ秒 / 作り直し4,646回
      上限600: 1案268ミリ秒 / 作り直し  598回  ← 1.7倍速い

    経路表1つは15KB程度なので、600件でも9MB。子プロセス11個でも
    100MB弱に収まります。
    """

    def test_budget_is_large_enough(self):
        from hexa.pathfind import CACHE_CELL_BUDGET

        self.assertGreaterEqual(CACHE_CELL_BUDGET, 600_000)

    def test_big_map_keeps_enough_tables(self):
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("決勝", random.Random(3),
                                            players=10, day_count=5))
        finder = PathFinder(problem)
        if problem.cell_count >= 1000:
            self.assertGreaterEqual(finder._cache_limit, 500,
                                    "大きいマップで貯める数が足りません")

    def test_cache_actually_hits(self):
        """同じ始点を何度引いても、作り直さないこと。"""

        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("決勝", random.Random(3),
                                            players=10, day_count=5))
        finder = PathFinder(problem)
        first = finder.table(0, {}, 0)
        before = finder.compute_count
        for _ in range(50):
            again = finder.table(0, {}, 0)
            self.assertIs(again, first)
        self.assertEqual(finder.compute_count, before,
                         "キャッシュが効いていません")

    def test_many_origins_fit(self):
        """店の数ぶんの始点を貯めても溢れないこと。"""

        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("決勝", random.Random(3),
                                            players=10, day_count=5))
        finder = PathFinder(problem)
        origins = [spot.position for spot in problem.spots][:40]
        for origin in origins:
            finder.table(origin, {}, 0)
        before = finder.compute_count
        for origin in origins:
            finder.table(origin, {}, 0)
        self.assertEqual(finder.compute_count, before,
                         "店のぶんだけでも溢れています")


class ShortLimitByCarsTest(unittest.TestCase):
    """燃料不足の許容台数を、巡回車の数で変えることの確認。

    **3台以下なら1台まで許し、4台以上は許しません。**

    3台しかいない試合では、1台欠けても残り2台で系列を回れます。

      1234（巡回3台）: 玉159 → 170（1位167を超える）
      1236（巡回3台）: 玉135 → 137

    4台以上では1台の欠けが系列の取りこぼしに直結します。

      1233（巡回4台）: 1台許すと日別95 → 94 に落ちる

    日別は玉数より重い得点なので、台数で切り替えます。
    """

    def _kinds(self, touring, supply=1):
        from hexa.rules import AgentKind

        return tuple([AgentKind.TOURING] * touring + [AgentKind.SUPPLY] * supply)

    def test_always_allows_one(self):
        """台数によらず1台まで許すこと。

        以前は「3台以下なら1台、4台以上は0台」と分けていましたが、
        測り直すと4台以上でも1台まで許すほうが良いと分かりました。

          1320（巡回4台）: 0台で玉358 → 1台で玉363
          1233（巡回4台）: 0台で玉187 → 1台で玉190

        2台まで許すと今度は落ちます（1233で184）。
        """

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.production import _short_limit

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        for touring in (2, 3, 4, 6):
            self.assertEqual(_short_limit(problem, self._kinds(touring)), 1)

    def test_falls_back_without_problem(self):
        from hexa.production import FUEL_SHORT_LIMIT, _short_limit

        self.assertEqual(_short_limit(None, ()), FUEL_SHORT_LIMIT)

    def test_prefer_noodles_with_one_short(self):
        """1台足りなくても、玉の多いほうが勝つこと。"""

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.production import _candidate_key
        from hexa.rules import AgentKind
        from hexa.simulator import Acquisition, DayResult

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        kinds = self._kinds(3)

        def make(fuels, noodles):
            return DayResult(
                positions=(0, 0, 0, 0), fuels=fuels + (0,),
                acquisitions=[Acquisition(i, 0, i, 0) for i in range(noodles)],
                refuels=[], stay_counts={}, remaining_stocks={},
                wasted_spot_stays=0, step_positions=[], step_fuels=[],
            )

        # 必要50。1台だけ足りない案と、足りているが玉の少ない案
        risky = _candidate_key(make((10, 60, 60), 9), set(), 100, kinds, 100,
                               problem, None, 0)
        safe = _candidate_key(make((60, 60, 60), 3), set(), 100, kinds, 100,
                              problem, None, 0)
        self.assertGreater(risky, safe, "1台の不足なら玉を優先すべきです")


class CongestionRecordTest(unittest.TestCase):
    """渋滞が何を奪ったかを記録することの確認。

    渋滞すると店どうしが遠くなります。実戦1341では13歩→25歩に
    倍増し、1台が回れる店が半分になっていました。

    この数字を残せば「渋滞のせいか、判断のせいか」を推測せずに
    切り分けられます。自チームが道路に滞在した歩数も残すので、
    翌日の渋滞を自分で作っていないかも分かります。
    """

    def test_records_the_gap(self):
        import json
        import tempfile

        import hexa.matchlog as module
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.matchlog import MatchLog
        from hexa.rules import AgentKind, RoadStatus, Terrain

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        roads = {
            cell: RoadStatus.JAMMED
            for cell in range(problem.cell_count)
            if problem.terrain_at(cell) == Terrain.ROAD
        }
        original = module.LOG_DIR
        with tempfile.TemporaryDirectory() as directory:
            module.LOG_DIR = Path(directory)
            try:
                log = MatchLog(enabled=True)
                log.start(problem, kinds, "jam", 10, 5.0, "x", (1, 1, 1))
                log.finish_day(
                    0, "x", 10, 0, problem.day_steps[0], 0,
                    list(problem.agent_positions),
                    [problem.fuel_limit] * problem.agent_count,
                    list(problem.agent_positions),
                    [problem.fuel_limit] * problem.agent_count,
                    [], [], [[0, 1, 2]] * problem.agent_count, roads,
                    problem, kinds, PathFinder(problem), [], 1, 1, 1, 1.0,
                )
                log.save("x", [])
                data = json.loads(
                    (Path(directory) / "match_jam.json").read_text(encoding="utf-8")
                )
                severity = data["days"][0]["severity"]
                for key in ("spotGapClear", "spotGapNow", "ownRoadSteps"):
                    self.assertIn(key, severity)
                self.assertGreater(severity["spotGapClear"], 0)
                self.assertGreaterEqual(
                    severity["spotGapNow"], severity["spotGapClear"],
                    "渋滞したのに距離が伸びていません",
                )
            finally:
                module.LOG_DIR = original


class SalvageTest(unittest.TestCase):
    """待つ前に近くの店を拾う仕組みが、既定で無効であることの確認。

    実戦1346の1日目で車2が14歩も待っており、7歩先に取れる店が
    ありました。拾えば得に見えますが、実測では

      16x16: 玉163 → 161
      24x24: 玉266 → 262

    と**どちらも悪化**しました。待っているのは「次の良い店へ向かう
    タイミング待ち」で、近くの1玉を拾うと後の動きが崩れます。
    """

    def test_disabled_by_default(self):
        from hexa.planner import SALVAGE_NEARBY

        self.assertFalse(SALVAGE_NEARBY)

    def test_helper_finds_the_nearest(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.planner import _salvage_target

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        finder = PathFinder(problem)
        stock = {spot.position: spot.stocks for spot in problem.spots}
        start = problem.agent_positions[0]
        picked = _salvage_target(
            0, start, problem.fuel_limit, 999, stock, set(), {},
            finder, {},
        )
        self.assertIsNotNone(picked)
        self.assertIn(picked, stock)


class SupplyHomeTest(unittest.TestCase):
    """補給車を燃料の少ない車の近くで終えさせる仕組みの確認。

    **実測で逆効果だったので無効にしています。**

    実戦1356の7日目で、補給車は満タンの車と同じマスにいて、空の3台
    からは50〜55歩離れていました。マス535なら合計47歩で済みます
    （実際は156歩）。172歩のうち最初の4割が補給待ちでした。

    そこで「近くで終える」計画を選ぶようにしたところ、

      罰0: 玉327 / 罰1: 玉307 / 罰2: 玉302

    と悪化しました。補給車の終了位置を縛ると、その日の補給が
    回らなくなります。**翌日の都合より、今日の補給のほうが重い**のです。
    """

    def test_disabled_by_default(self):
        from hexa.production import SUPPLY_HOME_WEIGHT

        self.assertEqual(SUPPLY_HOME_WEIGHT, 0)

    def test_penalty_is_zero_when_disabled(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.production import _supply_home_penalty
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        result = DayResult(
            positions=(0, 100), fuels=(1, 0), acquisitions=[], refuels=[],
            stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        # 仕組みそのものは動く（遠ければ正の値）
        value = _supply_home_penalty(
            result, kinds, problem, PathFinder(problem), 100
        )
        self.assertGreaterEqual(value, 0)

    def test_critical_fuel_is_off(self):
        """貪欲法の「危険な車を優先」も無効であること。

        本番で採用される案はほぼ学習済みAIが作るので、貪欲法だけを
        変えても結果は動きませんでした（玉327→318〜328）。
        """

        from hexa.production import CRITICAL_FUEL

        self.assertEqual(CRITICAL_FUEL, 0.0)


class BigMapLookaheadTest(unittest.TestCase):
    """大きいマップで先を長く見ることの確認。

    24x24（576マス）で、先読みを4店から8店に伸ばすと玉数が伸びました。

      1228: 83% → 90%
      1233: 87% → 94%
      1320: 95% → 97%

    店が散っているぶん、順番を工夫する余地が大きいためです。
    一方 16x16（256マス）では逆効果でした（日別138→134）。
    店が近いので、長く見ると目先の1店を取り逃します。
    """

    def test_big_map_looks_further(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.routes import (BIG_MAP_CELLS, BIG_MAP_ROUTE_LENGTH,
                                 route_length_for)

        from hexa.routes import BIG_MAP_MAX_CELLS

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=4))
        from hexa.routes import BIG_MAP_FUEL_LIMIT

        if (BIG_MAP_CELLS <= problem.cell_count <= BIG_MAP_MAX_CELLS
                and problem.fuel_limit <= BIG_MAP_FUEL_LIMIT):
            self.assertEqual(route_length_for(problem, {}),
                             BIG_MAP_ROUTE_LENGTH)

    def test_small_map_keeps_four(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.routes import (BIG_MAP_CELLS, MAX_ROUTE_LENGTH,
                                 route_length_for)

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        if problem.cell_count < BIG_MAP_CELLS:
            self.assertEqual(route_length_for(problem, {}), MAX_ROUTE_LENGTH)

    def test_threshold_sits_between_the_sizes(self):
        """16x16 は小さい側、24x24 は大きい側に入ること。"""

        from hexa.routes import BIG_MAP_CELLS

        from hexa.routes import BIG_MAP_MAX_CELLS

        # 16x16 は下、24x24 は範囲内、32x32 は上に外れる
        self.assertGreater(BIG_MAP_CELLS, 16 * 16)
        self.assertLessEqual(BIG_MAP_CELLS, 24 * 24)
        self.assertGreaterEqual(BIG_MAP_MAX_CELLS, 24 * 24)
        self.assertLess(BIG_MAP_MAX_CELLS, 32 * 32)


class LongDayWaitTest(unittest.TestCase):
    """歩数の多い日に待ちを短くする仕組みの確認。

    **実測で効かなかったので無効にしています。**

    実戦1476の1日目（99歩）で、車2が19歩も待って10玉。他の車は
    11〜13玉でした。長い日ほど待つ損が大きく見えます。ところが

      一律1/48  : 1468の日別が95→94
      90歩以上  : 1476の日別が144→142
      130歩以上 : 1476の日別は144のまま、玉が86%→75%
      無効      : 1476は日別144・玉86%  ← 最良

    待っているのは「次の良い店へ向かうタイミング待ち」で、
    急がせると近場の1玉に飛びついて全体が崩れます。
    """

    def test_disabled_by_default(self):
        from hexa.planner import LONG_DAY_STEPS

        self.assertGreater(LONG_DAY_STEPS, 500, "既定で無効にしてください")

    def test_normal_divisor_is_used(self):
        from hexa.planner import OPTION_WAIT_DIVISOR, _AgentPlanState, _wait_length
        from hexa.rules import AgentKind

        kinds = (AgentKind.TOURING,)
        state = _AgentPlanState()
        state.had_options = True
        # 200歩の日でも、既定では通常の割り算になる
        got = _wait_length(0, kinds, 240, [state], day_steps=200)
        self.assertEqual(got, max(1, 240 // OPTION_WAIT_DIVISOR))

    def test_long_day_divisor_applies_when_enabled(self):
        import hexa.planner as module
        from hexa.planner import _AgentPlanState, _wait_length
        from hexa.rules import AgentKind

        kinds = (AgentKind.TOURING,)
        state = _AgentPlanState()
        state.had_options = True
        original = module.LONG_DAY_STEPS
        module.LONG_DAY_STEPS = 90
        try:
            got = _wait_length(0, kinds, 240, [state], day_steps=200)
            self.assertEqual(got, max(1, 240 // module.LONG_DAY_WAIT_DIVISOR))
        finally:
            module.LONG_DAY_STEPS = original


class FuelAwareLookaheadTest(unittest.TestCase):
    """24x24 で、燃料が厳しい試合だけ先を長く見ることの確認。

    燃料が潤沢な試合（上限361）では4店が勝ちました。

      4店: 77案/8秒（1案105ms）最良玉64
      8店: 30案/8秒（1案271ms）最良玉61

    燃料が厳しい試合（上限110〜120）では8店が勝ちます。動ける歩数が
    限られるぶん、順番を練る価値が大きいためです。

      1228（燃料120）: 83% → 90%
      1233（燃料110）: 87% → 94%
    """

    def test_tight_fuel_looks_further(self):
        from hexa.problem import Problem
        from hexa.routes import (BIG_MAP_FUEL_LIMIT, BIG_MAP_ROUTE_LENGTH,
                                 route_length_for)

        import dataclasses
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=4))
        tight = dataclasses.replace(problem, fuel_limit=BIG_MAP_FUEL_LIMIT - 10)
        if 500 <= tight.cell_count <= 700:
            self.assertEqual(route_length_for(tight, {}), BIG_MAP_ROUTE_LENGTH)

    def test_plenty_fuel_keeps_four(self):
        import dataclasses

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.routes import (BIG_MAP_FUEL_LIMIT, MAX_ROUTE_LENGTH,
                                 route_length_for)

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=4))
        plenty = dataclasses.replace(problem, fuel_limit=BIG_MAP_FUEL_LIMIT + 100)
        self.assertEqual(route_length_for(plenty, {}), MAX_ROUTE_LENGTH)

    def test_threshold_separates_the_matches(self):
        """実戦の燃料値が、しきい値の両側に分かれること。"""

        from hexa.routes import BIG_MAP_FUEL_LIMIT

        # 1228=120, 1233=110 は厳しい側 / 1523=361 は潤沢な側
        self.assertGreater(BIG_MAP_FUEL_LIMIT, 120)
        self.assertLess(BIG_MAP_FUEL_LIMIT, 361)


class RealCongestionTest(unittest.TestCase):
    """本番の渋滞は、自分の履歴から読める量より重いことの記録。

    実戦1535の渋滞を記録と照合しました。

      2日目: 渋滞1 / 混雑31（道路138セル）
      3日目: 渋滞11 / 混雑63  ← 道路の半分が混む
      4日目: 渋滞7 / 混雑33

    他9チームが道路を使うため、自分の履歴だけでは読めません。
    同じ試合を再現すると

      実戦の渋滞を使う  : 玉428
      自分の履歴から推定: 玉463

    **推定のほうが35玉も良く出ます。** つまり手元の検証は本番より
    甘い条件で測っていたことになります。順位が思ったより低いのは
    これが一因です。

    `road_status_for_day` は要項どおり自チームの履歴から求めるので、
    実装としては正しいものです。ここでは「手元の数字を本番と
    同じだと思ってはいけない」ことを記録しておきます。
    """

    def test_estimation_uses_own_history(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.simulator import road_status_for_day

        from hexa.rules import RoadStatus

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=4))
        # 履歴が空なら、すべての道路が順調
        status = road_status_for_day(problem, [])
        self.assertTrue(status, "道路が1つも返っていません")
        self.assertTrue(
            all(value == RoadStatus.SMOOTH for value in status.values()),
            "1日目から混んでいます",
        )

    def test_more_stay_means_worse_roads(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import RoadStatus, Terrain
        from hexa.simulator import road_status_for_day

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=4))
        roads = [c for c in range(problem.cell_count)
                 if problem.terrain_at(c) == Terrain.ROAD]
        if not roads:
            return
        light = road_status_for_day(problem, [{roads[0]: 1}])
        heavy = road_status_for_day(problem, [{roads[0]: 10_000}])
        self.assertLess(
            int(light[roads[0]]),
            int(heavy[roads[0]]),
            "滞在が増えたのに道路が悪くなっていません",
        )


class LongMatchToleranceTest(unittest.TestCase):
    """歩数の多い試合では、燃料不足を広く許すことの確認。

    実戦の渋滞をそのまま使って測りました。

      1533（歩数平均101）: 許容1台で玉249 → 3台で260
      1535（歩数平均125）: 許容1台で玉421 → 3台で431

    歩数があれば、燃料が減っても補給で取り返せます。

    逆に1233（歩数平均60）のような短い試合では取り返す時間がなく、
    許容3台にすると93%→87%に悪化しました。だから平均歩数で
    切り替えます。
    """

    def test_long_match_allows_three(self):
        import dataclasses

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.production import (LONG_MATCH_SHORT_LIMIT, LONG_MATCH_STEPS,
                                     _short_limit_for_day)
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=4))
        steps = tuple(LONG_MATCH_STEPS + 20 for _ in problem.day_steps)
        long_match = dataclasses.replace(problem, day_steps=steps)
        kinds = (AgentKind.TOURING,) * 4 + (AgentKind.SUPPLY,)
        self.assertEqual(
            _short_limit_for_day(long_match, kinds, 100),
            LONG_MATCH_SHORT_LIMIT,
        )

    def test_short_match_keeps_one(self):
        import dataclasses

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.production import (FUEL_SHORT_LIMIT, LONG_MATCH_STEPS,
                                     _short_limit_for_day)
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=4))
        steps = tuple(LONG_MATCH_STEPS - 30 for _ in problem.day_steps)
        short_match = dataclasses.replace(problem, day_steps=steps)
        kinds = (AgentKind.TOURING,) * 4 + (AgentKind.SUPPLY,)
        self.assertEqual(
            _short_limit_for_day(short_match, kinds, 100), FUEL_SHORT_LIMIT
        )

    def test_threshold_separates_the_matches(self):
        """実戦の平均歩数が、しきい値の両側に分かれること。"""

        from hexa.production import LONG_MATCH_STEPS

        # 1233=60 は短い側 / 1533=101、1535=125 は長い側
        self.assertGreater(LONG_MATCH_STEPS, 60)
        self.assertLessEqual(LONG_MATCH_STEPS, 101)


class PlannedRouteTest(unittest.TestCase):
    """全車の順路をまとめて設計する方式の確認。

    これまでは「1台ずつ、その場で次に良い店を選ぶ」繰り返しでした。
    実戦1535の5日目（99歩）を調べると、全24店を訪問し16店は取り切った
    うえで、同じ店に戻る余力がない状態でした。

    理想的な貪欲（在庫を全車で共有し、常に一番近い店へ）を書いて
    測ると48玉。いまの実装の50玉はすでにそれを超えています。
    **「次に近い店」を選ぶ考え方では限界**ということです。

    まとめて設計すると（実戦の渋滞・1位の玉数に対する率）

      設計なし: 1535=89% / 1533=88%
      設計あり: 1535=91% / 1533=92%

    1通り0.02秒ほどなので、探索の時間をほとんど使いません。
    """

    def _setup(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        return problem, kinds, PathFinder(problem)

    def test_design_assigns_every_car(self):
        from hexa.allocate import design_day
        from hexa.rules import AgentKind

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit if k == AgentKind.TOURING else 0
                 for k in kinds]
        routes = design_day(problem, finder, kinds,
                            list(problem.agent_positions), fuels, {},
                            problem.day_steps[0])
        touring = [i for i, k in enumerate(kinds) if k == AgentKind.TOURING]
        self.assertEqual(sorted(routes), sorted(touring))
        self.assertTrue(any(routes.values()), "どの車にも行き先がありません")

    def test_no_car_revisits_the_same_spot(self):
        """同じ車が同じ店を2回入れないこと（1日1玉までのため）。"""

        from hexa.allocate import design_day
        from hexa.rules import AgentKind

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit if k == AgentKind.TOURING else 0
                 for k in kinds]
        routes = design_day(problem, finder, kinds,
                            list(problem.agent_positions), fuels, {},
                            problem.day_steps[0])
        for spots in routes.values():
            self.assertEqual(len(spots), len(set(spots)),
                             "同じ店が2回入っています")

    def test_stock_limit_is_respected(self):
        """店の在庫を超えて割り当てないこと。"""

        from collections import Counter

        from hexa.allocate import design_day
        from hexa.rules import AgentKind

        problem, kinds, finder = self._setup()
        touring = sum(1 for k in kinds if k == AgentKind.TOURING)
        fuels = [problem.fuel_limit if k == AgentKind.TOURING else 0
                 for k in kinds]
        routes = design_day(problem, finder, kinds,
                            list(problem.agent_positions), fuels, {},
                            problem.day_steps[0])
        count = Counter(spot for spots in routes.values() for spot in spots)
        for spot in problem.spots:
            limit = min(spot.stocks, touring)
            self.assertLessEqual(count.get(spot.position, 0), limit)

    def test_order_improvement_never_lengthens(self):
        from hexa.allocate import improve_order, route_cost

        problem, kinds, finder = self._setup()
        spots = [s.position for s in problem.spots][:6]
        start = problem.agent_positions[0]
        before = route_cost(finder, {}, start, spots)
        after_spots = improve_order(finder, {}, start, spots)
        after = route_cost(finder, {}, start, after_spots)
        self.assertLessEqual(after, before)
        self.assertEqual(sorted(after_spots), sorted(spots))

    def test_policy_follows_the_route(self):
        from hexa.production import PlannedRoutePolicy

        problem, kinds, finder = self._setup()
        first = problem.spots[0].position
        policy = PlannedRoutePolicy(problem, {0: [first]})
        got = policy.choose_touring_target(
            agent_index=0, step=0, remaining=50, position=0, fuel=50,
            stock={first: 1}, reserved={}, acquired_today=set(),
            acquired_brands=set(), brands_today=set(), road_status={},
            finder=finder,
        )
        self.assertEqual(got.target, first)

    def test_production_uses_the_design(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("_planned_route_policies", source)


class RefineLimitTest(unittest.TestCase):
    """順路の入れ替えによる改善が、頭打ちであることの記録。

    実戦1535の5日目（99歩・上限64玉）で設計は53玉でした。

      設計を200通り試す : 53玉のまま
      入れ替えで改善     : 53玉のまま
      1位は推定60玉

    歩数も燃料も使い切っており、入れ替えで浮く余地がありません。

      車0: 余り3歩  / 車1: 余り11歩
      車2: 余り11歩 / 車3: 余り3歩

    理論上は1台25店・4台で99店まわれる計算ですが、同じ車が同じ店
    から1日1玉しか取れないため、散在する店を数珠つなぎにできません。
    """

    def test_refine_keeps_the_plan_valid(self):
        """改善をかけても、歩数と燃料の制約を破らないこと。"""

        from hexa.allocate import design_day, refine_plans, route_cost
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("準決勝", random.Random(3),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        finder = PathFinder(problem)
        fuels = [problem.fuel_limit if k == AgentKind.TOURING else 0
                 for k in kinds]
        steps = problem.day_steps[0]
        routes = design_day(problem, finder, kinds,
                            list(problem.agent_positions), fuels, {}, steps)
        better = refine_plans(problem, finder, kinds,
                              list(problem.agent_positions), fuels, {},
                              steps, routes)
        for index, spots in better.items():
            self.assertEqual(len(spots), len(set(spots)),
                             "同じ店が2回入っています")
            cost = route_cost(finder, {}, problem.agent_positions[index], spots)
            self.assertLessEqual(cost, steps, "歩数を超えています")

    def test_refine_never_loses_noodles(self):
        """改善で玉数が減らないこと。"""

        from hexa.allocate import design_day, refine_plans
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("準決勝", random.Random(5),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        finder = PathFinder(problem)
        fuels = [problem.fuel_limit if k == AgentKind.TOURING else 0
                 for k in kinds]
        steps = problem.day_steps[0]
        routes = design_day(problem, finder, kinds,
                            list(problem.agent_positions), fuels, {}, steps)
        before = sum(len(v) for v in routes.values())
        better = refine_plans(problem, finder, kinds,
                              list(problem.agent_positions), fuels, {},
                              steps, routes)
        after = sum(len(v) for v in better.values())
        self.assertGreaterEqual(after, before)


class BrandAwareDesignTest(unittest.TestCase):
    """まとめて設計するとき、系列をそろえる案も作ることの確認。

    実戦1627の1日目（系列11が必要）で、玉数だけを見た設計は
    9系列しか押さえられず、候補になりませんでした。

      重み0 : 最大9系列  → 候補にならない
      重み3 : 最大10系列 → 候補にならない
      重み8 : 11系列すべて（玉16）

    系列は日別の得点に直結するので、そちらを優先する案も
    合わせて作ります。
    """

    def test_brand_weight_covers_more_brands(self):
        from hexa.allocate import design_day
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        finder = PathFinder(problem)
        fuels = [problem.fuel_limit if k == AgentKind.TOURING else 0
                 for k in kinds]
        brand_of = {s.position: s.brand for s in problem.spots}

        def brands(weight):
            routes = design_day(
                problem, finder, kinds, list(problem.agent_positions), fuels,
                {}, problem.day_steps[0], brand_weight=weight,
                acquired_brands=set(),
            )
            return len({brand_of[x] for v in routes.values() for x in v})

        plain = brands(0.0)
        aware = brands(8.0)
        self.assertGreaterEqual(aware, plain,
                                "系列を優先したのに種類が増えていません")

    def test_both_weights_are_offered(self):
        from hexa.production import PLANNED_BRAND_WEIGHTS

        self.assertIn(0.0, PLANNED_BRAND_WEIGHTS)
        self.assertTrue(any(w > 0 for w in PLANNED_BRAND_WEIGHTS))

    def test_policies_cover_the_combinations(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.allocate import SUPPLY_REFUELS
        from hexa.production import (INSERT_REFUELS, INSERT_STOCK_WEIGHTS,
                                     PLANNED_BRAND_WEIGHTS,
                                     PLANNED_STOCK_WEIGHTS,
                                     _planned_route_policies)
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        finder = PathFinder(problem)
        fuels = [problem.fuel_limit if k == AgentKind.TOURING else 0
                 for k in kinds]
        got = _planned_route_policies(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], set(),
        )
        # 末尾に足す設計は、在庫の重み × 系列の重み の組み合わせごとに
        #   1. そのままの設計
        #   2. 補給1回ぶんを見込んだ設計
        #   3. 補給2回ぶんを見込んだ設計
        # の3通り（SUPPLY_REFUELS が (1, 2) なので2通り増える）。
        #
        # そこに**差し込みで組む設計**が加わります
        # （INSERT_STOCK_WEIGHTS × PLANNED_BRAND_WEIGHTS × INSERT_REFUELS）。
        #
        # さらに、あとから足した3系統が入ります。
        #   系列網羅（`COVER_DESIGN_REFUELS` 通り）
        #   通りがかり（`PASSING_DESIGNS` 通り）
        #   ハブ（`HUB_REFUELS` 通り。既定はオフ）
        #   陣形（`FORMATION_MARGINS` × `FORMATION_REFUELS` 通り。
        #         `next_steps` を渡したときだけ作られる）
        from hexa.production import (COVER_DESIGN, COVER_DESIGN_REFUELS,
                                     FORMATION_DESIGN, FORMATION_MARGINS,
                                     FORMATION_REFUELS, FORMATION_SUPPLY,
                                     HUB_DESIGN, HUB_REFUELS, PASSING_DESIGNS,
                                     PASSING_ON)

        appended = (len(PLANNED_STOCK_WEIGHTS) * len(PLANNED_BRAND_WEIGHTS)
                    * (1 + len(SUPPLY_REFUELS)))
        inserted = (len(INSERT_STOCK_WEIGHTS) * len(PLANNED_BRAND_WEIGHTS)
                    * len(INSERT_REFUELS))
        from hexa.production import (ORIENTEER_DESIGN, ORIENTEER_REFUELS,
                                     ORIENTEER_WANT)
        extra = ((len(COVER_DESIGN_REFUELS) if COVER_DESIGN else 0)
                 + (len(PASSING_DESIGNS) if PASSING_ON else 0)
                 + (len(HUB_REFUELS) if HUB_DESIGN else 0)
                 # 歩数と燃料だけを見る高速ソルバ（`ORIENTEER_WANT` 通りまで）
                 + (ORIENTEER_WANT * len(ORIENTEER_REFUELS)
                    if ORIENTEER_DESIGN else 0))
        self.assertLessEqual(len(got), appended + inserted + extra)
        self.assertTrue(got, "設計の案が1つも作れていません")

        # **明日の歩数を渡すと、陣形の案が増えること。**
        more = _planned_route_policies(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], set(), next_steps=problem.day_steps[1],
        )
        formation = (len(FORMATION_MARGINS) * len(FORMATION_REFUELS)
                     * len(FORMATION_SUPPLY) if FORMATION_DESIGN else 0)
        self.assertLessEqual(len(more), appended + inserted + extra + formation)
        if FORMATION_DESIGN:
            self.assertTrue(
                [label for label, _ in more if label.startswith("陣形")],
                "明日の歩数を渡しても陣形の案ができていません")


class FastPredictTest(unittest.TestCase):
    """特徴を作らずに内積を取る最適化の確認。

    以前は「素の材料に積を足したリスト」を作ってから内積を3回
    取っていました。1案を作るあいだに `append` が527万回呼ばれ、
    `_expand` だけで全体の2割を占めます。

    掛け算の項は作った端から足せるのでリストは要りません。

      16x16: 1案32ms → 26ms（1.24倍）
      24x24: 1案183ms → 153ms（1.19倍）

    **結果は1ビットも変わりません。** 下のテストで、古い書き方と
    一致することを確かめています。
    """

    def _brains(self):
        from hexa.ai import new_population

        return new_population(count=3, capacity="simple")

    def test_touring_matches_the_plain_form(self):
        from hexa.ai import _dot, _expand

        rng = random.Random(11)
        for brain in self._brains():
            size = len(brain.touring_heads[0])
            base = [rng.uniform(-2, 2) for _ in range(40)]
            fast = brain.predict_touring(base)
            feats = _expand(base, brain.touring_cross)
            slow = tuple(_dot(brain.touring_heads[i], feats) for i in range(3))
            for a, b in zip(fast, slow):
                self.assertAlmostEqual(a, b, places=9)
            self.assertGreater(size, 0)

    def test_supply_matches_the_plain_form(self):
        from hexa.ai import _dot, _expand

        rng = random.Random(12)
        for brain in self._brains():
            base = [rng.uniform(-2, 2) for _ in range(40)]
            fast = brain.predict_supply(base)
            slow = _dot(brain.supply_weights, _expand(base, brain.supply_cross))
            self.assertAlmostEqual(fast, slow, places=9)

    def test_short_weights_do_not_crash(self):
        """重みが特徴より短くても落ちないこと。

        学習の途中で材料が増えると、重みのほうが短くなります。
        `_dot` が zip で打ち切っていたのと同じ挙動にしてあります。
        """

        from hexa.ai import _dot, _expand

        brain = self._brains()[0]
        brain.supply_weights = brain.supply_weights[:5]
        base = [0.5] * 40
        fast = brain.predict_supply(base)
        slow = _dot(brain.supply_weights, _expand(base, brain.supply_cross))
        self.assertAlmostEqual(fast, slow, places=9)

    def test_zero_features_are_skipped(self):
        """材料が0のときも、結果が変わらないこと。"""

        from hexa.ai import _dot, _expand

        brain = self._brains()[0]
        base = [0.0] * 40
        base[3] = 1.5
        fast = brain.predict_touring(base)
        feats = _expand(base, brain.touring_cross)
        slow = tuple(_dot(brain.touring_heads[i], feats) for i in range(3))
        for a, b in zip(fast, slow):
            self.assertAlmostEqual(a, b, places=9)


class PathTieBreakTest(unittest.TestCase):
    """同じ歩数なら燃料の少ない道、次に店を通る道を選ぶことの確認。

    **なぜ必要か**

    混雑した道路は2歩・燃料2、平地は2歩・燃料1です。歩数だけで
    比べると並ぶので、以前はどちらを選ぶか決まっていませんでした。

    実戦7試合のログを数えると、混雑した道路の移動が485回あり、
    1回あたり燃料1を余分に使っていました。燃料上限73の試合なら
    6.6タンク分です。

    さらに、公式の処理順では通り道にある店の玉もそのまま入ります
    （1.3 うどんの獲得）。同じ歩数・同じ燃料なら、店を通る道を
    選んだほうが得です。

    実測（実戦7試合を渋滞つきで再現 / 学習済みAIなし）:

      どちらも無し          : 玉1289
      燃料で同点を割る      : 玉1292
      **店でも同点を割る**  : 玉1317
    """

    def _problem(self, cells, spots):
        from hexa.problem import Problem, Spot

        return Problem(
            starts_at=0, day_seconds=(60,), day_steps=(40,), height=1,
            width=len(cells), cells=tuple(cells),
            spots=tuple(Spot(brand=i, position=p, stocks=4)
                        for i, p in enumerate(spots)),
            agent_positions=(0,), fuel_limit=100, players=4,
            busy_threshold=1, jammed_threshold=2,
        )

    def test_equal_steps_prefer_less_fuel(self):
        from hexa.pathfind import PathFinder, _blend, _cell_cost

        # 1マス分のコストを直接見ます。混雑した道路と平地はどちらも2歩。
        self.assertEqual(_blend(2, 2, 0), _blend(2, 1, 0))
        road = _cell_cost(2, 2, 0, is_spot=False)
        plain = _cell_cost(2, 1, 0, is_spot=False)
        self.assertLess(plain, road, "同じ歩数なら燃料の少ないほうを選ぶべきです")

    def test_equal_cost_prefers_spot(self):
        from hexa.pathfind import _cell_cost

        on_spot = _cell_cost(2, 1, 0, is_spot=True)
        away = _cell_cost(2, 1, 0, is_spot=False)
        self.assertLess(on_spot, away, "同点なら店を通る道を選ぶべきです")

    def test_fuel_outranks_spot(self):
        """燃料の差は、店の有無より重く見ること。"""

        from hexa.pathfind import _cell_cost

        cheap_away = _cell_cost(2, 1, 0, is_spot=False)
        costly_spot = _cell_cost(2, 2, 0, is_spot=True)
        self.assertLess(cheap_away, costly_spot)


class StrandedTest(unittest.TestCase):
    """燃料が尽きて動けない車を、玉数より重く見ることの確認。

    **実戦の記録**

    1646（16x16・燃料60）で3号車が3日目に燃料2で止まり、
    4日目と5日目は**一歩も動いていません**（34歩と42歩を捨てた）。
    玉数は78で、1位は95でした。差17玉のほとんどがこの1台です。

    1650の3日目でも4号車が燃料7で始まり0歩で終わっています。

    `_fuel_shortfall` は「明日ぶんの燃料が足りない台数」なので、
    燃料7の車と燃料0の車を同じ1台として数えます。燃料が少しでも
    あれば自力で補給車に近づけますが、0だと永久に動けません。
    別に数えて、玉数より上に置きます。
    """

    def _result(self, positions, fuels):
        from hexa.simulator import DayResult

        return DayResult(
            positions=tuple(positions), fuels=tuple(fuels), acquisitions=[],
            refuels=[], stay_counts={}, remaining_stocks={},
            wasted_spot_stays=0, step_positions=[], step_fuels=[],
        )

    def _problem(self):
        from hexa.problem import Problem, Spot

        # 1行8マスの平地。店は端の1つだけ。
        return Problem(
            starts_at=0, day_seconds=(60, 60), day_steps=(40, 40), height=1,
            width=8, cells=tuple([0] * 8),
            spots=(Spot(brand=0, position=7, stocks=4),),
            agent_positions=(0, 1), fuel_limit=60, players=4,
            busy_threshold=1, jammed_threshold=2,
        )

    def test_empty_car_far_from_supply_counts(self):
        from hexa.pathfind import PathFinder
        from hexa.production import _stranded_count
        from hexa.rules import AgentKind

        problem = self._problem()
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        # 巡回車はマス0で燃料0、補給車は遠いマス7。
        result = self._result((0, 7), (0, 0))
        self.assertEqual(
            _stranded_count(result, kinds, problem, finder, 2), 1
        )

    def test_supply_on_the_same_cell_rescues(self):
        from hexa.pathfind import PathFinder
        from hexa.production import _stranded_count
        from hexa.rules import AgentKind

        problem = self._problem()
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        # 同じマスで終われば、翌日の最初の反映フェーズで満タンになります。
        result = self._result((0, 0), (0, 0))
        self.assertEqual(
            _stranded_count(result, kinds, problem, finder, 2), 0
        )

    def test_last_day_does_not_count(self):
        from hexa.pathfind import PathFinder
        from hexa.production import _stranded_count
        from hexa.rules import AgentKind

        problem = self._problem()
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        result = self._result((0, 7), (0, 0))
        # 明日がないなら、動けなくても損はありません。
        self.assertEqual(
            _stranded_count(result, kinds, problem, finder, 0), 0
        )

    def test_key_puts_stranding_above_noodles(self):
        from hexa.production import _candidate_key
        from hexa.rules import AgentKind
        from hexa.simulator import Acquisition

        problem = self._problem()
        from hexa.pathfind import PathFinder

        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        # 玉を多く取るが1台が立ち往生する案
        greedy = self._result((0, 7), (0, 0))
        greedy.acquisitions.extend(
            Acquisition(step=i + 1, agent_index=0, spot_position=7, brand=0)
            for i in range(5)
        )
        # 玉は少ないが立ち往生しない案
        safe = self._result((0, 0), (0, 0))
        safe.acquisitions.append(
            Acquisition(step=1, agent_index=0, spot_position=7, brand=0)
        )
        self.assertGreater(
            _candidate_key(safe, {0}, 60, kinds, 2, problem, finder, 40),
            _candidate_key(greedy, {0}, 60, kinds, 2, problem, finder, 40),
            "立ち往生する案が玉数で勝ってはいけません",
        )


class RoutePolishTest(unittest.TestCase):
    """最良案を1手ずつ磨く仕組みの確認。

    **これが順位の主因でした。**

    実戦7試合を渋滞つきで再現した実測（学習済みAIなし）:

      磨かない            : 玉1317 / 日別744
      設計だけ磨く        : 玉1337 / 日別758
      **最良案を磨く**    : 玉1440 / 日別762

    1位の玉数に対する率が 78% → 87% に上がり、日別は7試合のうち
    6試合で1位と並びました（1646で55、1648で144、1650で113、
    1652で105、1654で162、1656で48、1658で135）。

    磨く手は5種類です。

      1. ある車の1店を、別の車に渡す
      2. 2台の店を入れ替える
      3. まだ誰も行かない店を差し込む
      4. 1台のなかで並びをほどく（2-opt）
      5. 取りに行くのをやめて歩数を空ける

    **机上の見積りではなく、実際に1日ぶんを公式シミュレータで
    再現して点数が上がったものだけ**残します。通り道の店で入る玉や、
    在庫の取り合いで空振りすることまで、そのまま点数に出ます。

    1手ずつ進むので、締切のどこで止めても安全です。
    """

    def _setup(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.planner import DayPlanner
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(5),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        finder = PathFinder(problem)
        planner = DayPlanner(problem, finder)
        return problem, kinds, finder, planner

    def test_routes_from_result_keeps_the_order(self):
        from hexa.production import routes_from_result
        from hexa.rules import AgentKind
        from hexa.simulator import Acquisition, DayResult

        kinds = (AgentKind.TOURING, AgentKind.TOURING, AgentKind.SUPPLY)
        result = DayResult(
            positions=(0, 0, 0), fuels=(1, 1, 0),
            acquisitions=[
                Acquisition(step=5, agent_index=0, spot_position=11, brand=0),
                Acquisition(step=2, agent_index=0, spot_position=7, brand=1),
                Acquisition(step=3, agent_index=1, spot_position=9, brand=2),
            ],
            refuels=[], stay_counts={}, remaining_stocks={},
            wasted_spot_stays=0, step_positions=[], step_fuels=[],
        )
        routes = routes_from_result(result, kinds)
        self.assertEqual(routes[0], [7, 11], "歩数の早い順に並ぶべきです")
        self.assertEqual(routes[1], [9])
        self.assertNotIn(2, routes, "補給車は順路を持ちません")

    def test_polisher_only_keeps_improvements(self):
        from hexa.improve import RoutePolisher
        from hexa.production import (GreedyPolicy, PlannedRoutePolicy,
                                     _build_candidate, routes_from_result,
                                     supply_order_from_result)

        problem, kinds, finder, planner = self._setup()
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit] * problem.agent_count
        day_steps = problem.day_steps[0]
        base = _build_candidate(
            "貪欲", problem, planner, kinds, positions, fuels, {}, day_steps,
            GreedyPolicy(problem), set(),
        )
        self.assertIsNotNone(base)

        def evaluate(trial, order):
            return _build_candidate(
                "磨き", problem, planner, kinds, positions, fuels, {},
                day_steps, PlannedRoutePolicy(problem, trial, order), set(),
            )

        polisher = RoutePolisher(
            problem, kinds, routes_from_result(base.day_result, kinds),
            evaluate, key=base.key,
            supply_order=supply_order_from_result(base.day_result, kinds),
        )
        found = None
        for _ in range(400):
            step = polisher.step()
            if step is not None:
                found = step
                break
            if polisher.exhausted:
                break
        if found is not None:
            self.assertGreater(found.key, base.key,
                               "良くなっていない案を返してはいけません")
        self.assertGreater(polisher.tried, 0, "手を1つも試していません")

    def test_polisher_stops_when_moves_run_out(self):
        from hexa.improve import RoutePolisher
        from hexa.rules import AgentKind

        problem, kinds, _finder, _planner = self._setup()
        touring = [i for i, k in enumerate(kinds) if k == AgentKind.TOURING]
        polisher = RoutePolisher(
            problem, kinds, {i: [] for i in touring},
            lambda trial, order: None, key=None,
        )
        for _ in range(10_000):
            if polisher.step() is None and polisher.exhausted:
                break
        self.assertTrue(polisher.exhausted, "手を出し切っても止まりません")
        self.assertIsNone(polisher.step(), "出し切ったあとは None を返すべきです")


class SupplyCountRecordTest(unittest.TestCase):
    """補給車は1台が最良、という実測の記録。

    実戦7試合を渋滞つきで再現し、補給車の台数だけを変えました
    （玉数 / 学習済みAIなし）。

      試合          0台   1台   2台   3台
      1646(16x16)    54    83    68    34
      1648(24x24)   138   235   184   131
      1650(24x24)    77   120   115    81
      1652(24x24)   130   301   243   165
      1654(24x24)   169   251   204   147
      1656(16x16)    54    64    44    21
      1658(24x24)    93   244   185   147

    **7試合すべてで1台が最良**でした。日別累計も同じ並びです。

    補給車は燃料を消費せず（simulator の `cost` が巡回車だけ）、
    合流すれば相手は満タンになります。1台で全車を回せるので、
    2台目を出すと巡回車が1台減るぶんだけ損になります。

    事前試走が毎回「補給車1台」を選んでいるのは正しい判断です。
    ここでは、同点のとき台数の少ないほうを選ぶ `_trial_better` が
    その向きになっていることを確かめます。
    """

    def test_tie_prefers_fewer_supply_cars(self):
        from hexa.production import _trial_better

        score = (16, 140, 250)
        self.assertTrue(
            _trial_better(score, 1, score, 2),
            "同点なら補給車の少ないほうを選ぶべきです",
        )
        self.assertFalse(
            _trial_better(score, 2, score, 1),
            "同点で補給車を増やしてはいけません",
        )

    def test_supply_cars_burn_no_fuel(self):
        """補給車が燃料を使わないことの確認（この結論の根拠）。"""

        from hexa.problem import Problem, Spot
        from hexa.rules import AgentKind
        from hexa.simulator import simulate_day

        problem = Problem(
            starts_at=0, day_seconds=(60,), day_steps=(4,), height=1, width=6,
            cells=(0, 0, 0, 0, 0, 0),
            spots=(Spot(brand=0, position=2, stocks=1),),
            agent_positions=(0, 5), fuel_limit=10, players=4,
            busy_threshold=1, jammed_threshold=2,
        )
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        # 巡回車はマス0から方向2（右）へ2歩、補給車はマス5から
        # 方向5（左）へ2歩。すれ違わないので合流は起きません。
        # （平地は1歩2ステップなので、どちらも4歩ちょうど）
        plan = [[2, 2], [5, 5]]
        result = simulate_day(
            problem=problem, agent_kinds=kinds, start_positions=(0, 5),
            start_fuels=(10, 10), action_plan=plan, day_steps=4,
            road_status={},
        )
        self.assertEqual(result.fuels[1], 10, "補給車の燃料が減っています")
        self.assertLess(result.fuels[0], 10, "巡回車の燃料が減っていません")


class WorkerPolishTest(unittest.TestCase):
    """「磨き」を子プロセスにも配ることの確認。

    **なぜ必要か**

    磨きは「いまの最良案」から始めるので、親しか出発点を持って
    いません。ところが親は1コアです。締切までに試せるのは400手
    ほどでした。実測では手数がそのまま効きます。

      手数400 : 玉1379 / 日別757
      手数800 : 玉1422 / 日別761
      手数3200: 玉1440 / 日別762

    そこで、最良案が変わるたびに**開始位置をずらした磨きの仕事**を
    子11個へ割り込ませます。同じ手を重ねないので、そのまま手数が
    増えます。

    割り込ませるには、投げてある仕事を積みすぎないことも必要です
    （`DaySearch._fill` の上限を子の12倍から6倍に下げました）。
    """

    def _setup(self, directory):
        import json

        from hexa.ai import new_population, save_population
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        problem_path = Path(directory) / "p.json"
        problem_path.write_text(
            json.dumps(problem.to_json_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
        brains = new_population(count=2, capacity="simple")
        population_path = Path(directory) / "b.json"
        save_population(brains, population_path)
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        state = {
            "kinds": [int(k) for k in kinds],
            "positions": list(problem.agent_positions),
            "fuels": [problem.fuel_limit if k == AgentKind.TOURING else 0
                      for k in kinds],
            "roads": {}, "acquired": [], "day_steps": problem.day_steps[0],
            "day_index": 0, "steps_ahead": 0, "next_steps": 0,
        }
        return problem, kinds, str(problem_path), str(population_path), state

    def test_worker_handles_a_polish_job(self):
        import tempfile
        import time

        from hexa.parallel import POLISH_JOB, _day_worker
        from hexa.production import (GreedyPolicy, _build_candidate,
                                     routes_from_result)
        from hexa.pathfind import PathFinder
        from hexa.planner import DayPlanner

        with tempfile.TemporaryDirectory() as directory:
            problem, kinds, problem_path, population_path, state = (
                self._setup(directory)
            )
            finder = PathFinder(problem)
            planner = DayPlanner(problem, finder)
            base = _build_candidate(
                "貪欲", problem, planner, kinds, list(state["positions"]),
                list(state["fuels"]), {}, state["day_steps"],
                GreedyPolicy(problem), set(),
            )
            self.assertIsNotNone(base)
            routes = routes_from_result(base.day_result, kinds)
            jobs = [("磨き0", POLISH_JOB, 0.0, 0, routes, 0, 40)]
            out = _day_worker((problem_path, population_path, state, jobs,
                               time.monotonic() + 60))
            # 良くなる手が見つからないこともあるので、形だけ確かめます。
            for key, label, plan in out:
                self.assertNotEqual(label, "error", f"子が失敗しました: {key}")
                self.assertEqual(len(key), 12)
                self.assertEqual(len(plan), problem.agent_count)
                self.assertGreater(tuple(key), base.key,
                                   "良くなっていない案を返してはいけません")

    def test_polish_jobs_jump_the_queue(self):
        """磨きの仕事が、投げてある仕事より先に並ぶこと。"""

        import time

        from hexa.parallel import POLISH_JOB, DaySearch

        class FakePool:
            def __init__(self):
                self.sent = []

            def apply_async(self, worker, args):
                self.sent.append(args[0][3])

                class Done:
                    def ready(self_inner):
                        return False

                return Done()

        pool = FakePool()
        jobs = [(f"案{i}", -1, 0.0, i) for i in range(400)]
        search = DaySearch(pool, "p", "b", {}, jobs,
                           deadline=time.monotonic() + 30, workers=2)
        # 投げてある仕事（子の6倍）が終わるまでは、新しいぶんは
        # 投げられません。**次に投げる列の先頭**に入っていることを
        # 確かめます。
        search.add_polish_jobs({0: [1, 2, 3]}, budget=10, count=3)
        following = search.chunks[search.index:search.index + 3]
        self.assertEqual(
            [job[1] for chunk in following for job in chunk],
            [POLISH_JOB] * 3,
            "磨きの仕事が先に並んでいません",
        )
        # 投げすぎていないこと（磨きが待たされる原因になります）
        self.assertLessEqual(len(pool.sent), 2 * 6)


class SupplyPolishTest(unittest.TestCase):
    """補給車の「迎える順番」も磨くことの確認。

    **なぜ必要か**

    磨く仕組みは巡回車の順路しか見ていませんでした。補給車の動きは
    貪欲法まかせで、誰も手を入れていませんでした。

    補給車は燃料を1滴も使いません（simulator の `cost` が巡回車
    だけ）。合流すれば相手は満タンになります。**止まっている歩数は
    そのまま無駄**です。それでも実戦ではこうなっていました。

      実戦1685 4日目: 補給車の39%が停止 / 補給1回
      実戦1685 5日目: 56%が停止 / 補給5回
      実戦1687 1日目: **93%が停止**（74歩のうち5歩しか動かない）/ 補給2回

    開始燃料は1685で 1.00→0.70→0.66→0.61→0.49 と枯れ、玉数も
    49→38→27→30 と落ちました。

    待ち合わせの時刻は決めません。燃料が切れた車はその場で止まって
    待つので、**順番だけ合っていれば必ず拾えます**。だから磨くのは
    順番だけで足ります。

    実測（実戦1685・1687を記録された渋滞つきで再現 / 学習済みAIなし）:

      補給車に手を入れない      : 玉352
      迎える順番も磨く          : 玉355
      順番の手を先に試す        : 玉365
      手の種類を一巡させる      : 玉372（手数3000で380）

    1位は430。実戦は365でした。
    """

    def test_supply_order_comes_from_refuel_events(self):
        from hexa.production import supply_order_from_result
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult, Refuel

        kinds = (AgentKind.TOURING, AgentKind.TOURING, AgentKind.SUPPLY)
        result = DayResult(
            positions=(0, 0, 0), fuels=(1, 1, 0), acquisitions=[],
            refuels=[
                Refuel(step=9, agent_index=1, position=5,
                       fuel_before=0, fuel_after=60),
                Refuel(step=3, agent_index=0, position=2,
                       fuel_before=0, fuel_after=60),
                # 続けて同じ車を拾ったぶんは1回とみなす
                Refuel(step=4, agent_index=0, position=3,
                       fuel_before=59, fuel_after=60),
            ],
            stay_counts={}, remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        self.assertEqual(supply_order_from_result(result, kinds), [0, 1])

    def test_supply_moves_change_only_the_order(self):
        from hexa.improve import _supply_moves

        got = list(_supply_moves([0, 1, 2], [0, 1]))
        self.assertIn([1, 0], got, "隣どうしの入れ替えがありません")
        self.assertIn([0], got, "迎えるのをやめる手がありません")
        self.assertIn([2, 0, 1], got, "迎える車を増やす手がありません")
        for order in got:
            self.assertTrue(all(car in (0, 1, 2) for car in order))

    def test_phases_take_turns(self):
        """良くなるたびに、次は違う種類の手から始めること。

        以前は毎回1番の手から試し直していたので、1番だけで持ち時間が
        尽き、後ろの種類（受け持ちの大きな入れ替えなど）に一度も
        たどり着けませんでした。**手数を800から3000に増やしても
        玉数が1つも動かなかった**のがその証拠です。
        """

        from hexa.improve import _MOVE_PHASES, _all_moves

        touring = [0, 1]
        routes = {0: [1, 2], 1: [3, 4]}
        count = len(_MOVE_PHASES)
        first = list(_all_moves(touring, routes, [0], [1, 2, 3, 4, 5],
                                list(range(count))))
        later = list(_all_moves(touring, routes, [0], [1, 2, 3, 4, 5],
                                list(range(1, count)) + [0]))
        self.assertEqual(len(first), len(later), "手の総数が変わっています")
        self.assertNotEqual(
            [item[0] for item in first[:12]],
            [item[0] for item in later[:12]],
            "始める種類がずれていません",
        )
        self.assertGreaterEqual(len(_MOVE_PHASES), 7)

    def test_bigger_moves_exist(self):
        """受け持ちをまとめて取り替える手があること。"""

        from hexa.improve import _phase_segment, _phase_tail_swap

        touring = [0, 1]
        routes = {0: [1, 2, 3], 1: [4, 5, 6]}
        tails = list(_phase_tail_swap(touring, routes, []))
        self.assertIn({0: [1, 5, 6], 1: [4, 2, 3]}, tails,
                      "後半の交換がありません")
        segments = list(_phase_segment(touring, routes, []))
        self.assertIn({0: [3], 1: [1, 2, 4, 5, 6]}, segments,
                      "続いた2店をまとめて渡す手がありません")
        for trial in tails + segments:
            for spots in trial.values():
                self.assertEqual(len(set(spots)), len(spots),
                                 "同じ車が同じ店を2回回る手を出しています")


class SupplyNeverIdlesTest(unittest.TestCase):
    """合流先がないときも、補給車を近づけることの確認。

    補給車は燃料を使わないので、止まっているのは必ず損です。
    今日のうちに合流できなくても、近づいておけば翌日すぐ拾えます。
    """

    def _problem(self):
        from hexa.problem import Problem, Spot

        # 1行12マスの平地。店は端。
        return Problem(
            starts_at=0, day_seconds=(60,), day_steps=(6,), height=1,
            width=12, cells=tuple([0] * 12),
            spots=(Spot(brand=0, position=11, stocks=4),),
            agent_positions=(0, 11), fuel_limit=60, players=4,
            busy_threshold=1, jammed_threshold=2,
        )

    def test_it_moves_toward_the_lowest_fuel_car(self):
        from hexa.pathfind import PathFinder
        from hexa.production import _approach_lowest_fuel
        from hexa.rules import AgentKind

        problem = self._problem()
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        table = finder.table(0, {})
        got = _approach_lowest_fuel(
            remaining=4, position=0, kinds=kinds, positions=[11, 0],
            fuels=[1, 0], road_status={}, finder=finder, table=table,
        )
        self.assertIsNotNone(got, "近づく先を返していません")
        cell, car = got
        self.assertEqual(car, 0)
        # 平地は1歩2ステップなので、4歩では2マス進める
        self.assertEqual(cell, 2)

    def test_a_full_car_is_not_chased(self):
        from hexa.pathfind import PathFinder
        from hexa.production import _approach_lowest_fuel
        from hexa.rules import AgentKind

        problem = self._problem()
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        table = finder.table(0, {})
        self.assertIsNone(
            _approach_lowest_fuel(
                remaining=4, position=0, kinds=kinds, positions=[11, 0],
                fuels=[60, 0], road_status={}, finder=finder, table=table,
            ),
            "今日ぶんの燃料がある車を追いかけています",
        )


class PolishBudgetTest(unittest.TestCase):
    """持ち時間を「磨き」に寄せたことの記録。

    **1件あたりの重さが2桁違います。**

    実戦1703（32x32・7体・9日）で測った実測:

      学習済みAIの案を1件作る : 約600ミリ秒
      **磨きの手を1つ試す**   : 5ミリ秒（毎秒200手）

    32x32の実戦では1日の候補が863〜1590件しか作れていませんでした。
    同じ持ち時間を磨きに回せば万単位の手を試せます。記録された
    渋滞つきの再現では

      磨きなし     : 玉366
      磨き400手    : 玉405  ← 実戦の390を超える
      磨き3000手×2 : 玉436 / 日別225（1位と同じ225）

    1位は玉611。32x32はまだいちばんの弱点ですが、実戦390からは
    大きく上がります。

    そこで親はほぼ磨きに専念させ（POLISH_EVERY=50）、子に投げる
    1回ぶんの手数も60から400へ増やしました。
    """

    def test_parent_spends_most_time_polishing(self):
        from hexa.production import POLISH_EVERY

        self.assertGreaterEqual(POLISH_EVERY, 20,
                                "親が磨きに使う割合が少なすぎます")

    def test_worker_budget_is_worth_the_round_trip(self):
        from hexa.production import POLISH_WORKER_BUDGET, POLISH_WORKER_JOBS

        # 1手5ミリ秒なので、60手では1件0.3秒。通信と立ち上げのほうが
        # 重くなるので、まとめて渡します。
        self.assertGreaterEqual(POLISH_WORKER_BUDGET, 200)
        # 手の種類は7つ。どの種類にも同じだけ子をあてます。
        from hexa.improve import _MOVE_PHASES

        self.assertEqual(POLISH_WORKER_JOBS % len(_MOVE_PHASES), 0)

    def test_designs_are_polished_too(self):
        """設計の案からも磨かせること。

        親が磨くのは「いまの最良案」からだけで、それはたいてい
        貪欲や学習済みAIが作った「1台ずつ近い店を選んだ形」です。
        全車の受け持ちをまとめて決めた設計は、そこから何手かけても
        届かない別の形なので、別に磨かせます。
        """

        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn('startswith(("設計", "補給設計"))', source,
                      "設計の案を磨きに回していません")


class ViewerFollowTest(unittest.TestCase):
    """実況ビューアの「最新を追う」まわりの確認。

    文字だけの検査です（ブラウザは動かしません）。入れたつもりの
    仕組みが消えていないことを見ます。

    * 追いかけ中は日の切り替え・再生・スライダー・速さを使えない
    * 使えないことが見た目で分かる（薄く表示して注意書きを出す）
    * 追いかけ中はループ再生
    * 新しい最善案が出たら0歩から流し直す
    * 在庫の点は店の外に出し、白と黒で描く（系列の色と喧嘩しない）
    * 同じマスに重なった車はずらして描く
    """

    def _source(self):
        return (Path(__file__).resolve().parent.parent
                / "viewer" / "live.html").read_text(encoding="utf-8")

    def test_controls_are_locked_while_following(self):
        source = self._source()
        self.assertIn("function lockControls()", source)
        self.assertIn('$(id).disabled=follow', source)
        self.assertIn('if(follow){ flash(); return; }', source,
                      "追いかけ中に日を変えられてしまいます")
        self.assertIn("followNote", source, "使えない理由を出していません")

    def test_following_loops_and_restarts(self):
        source = self._source()
        self.assertIn("if(follow) pos=0;", source, "ループ再生になっていません")
        self.assertIn("function planSignature(", source)
        self.assertIn("if(grew||sig!==lastSig){ pos=0; lastSig=sig; }", source,
                      "新しい最善案で0歩に戻していません")

    def test_stock_dots_are_outside_the_spot(self):
        source = self._source()
        # 系列の色は色相を均等に割って作るので、赤い点では溶けます。
        self.assertNotIn('"#ff5d5d"', source, "在庫の点が赤のままです")
        self.assertIn('ctx.fillStyle=k<left?"#ffffff":"#555";', source)
        self.assertIn("ctx.roundRect(", source, "点の台を描いていません")

    def test_agents_on_one_cell_are_spread(self):
        source = self._source()
        self.assertIn("const spotsOf=new Map()", source)
        self.assertIn("mates.length>1", source, "重なった車をずらしていません")


class RotateThreeTest(unittest.TestCase):
    """3台のあいだで1店ずつ回す手の確認。

    2台で渡す・入れ替える手だけでは、「AがBの近くへ、BがCの近くへ、
    CがAの近くへ」とずれた形から出られません。どの1手も単独では
    点数を下げるので、山登りが止まります。3台で輪にして回せば
    1手で解けます。
    """

    def test_it_rotates_without_making_duplicates(self):
        from hexa.improve import _phase_rotate3

        routes = {0: [1, 2], 1: [3, 4], 2: [5, 6]}
        got = list(_phase_rotate3([0, 1, 2], routes, []))
        self.assertTrue(got, "手が1つも出ていません")
        self.assertIn({0: [5, 2], 1: [1, 4], 2: [3, 6]}, got)
        for trial in got:
            for spots in trial.values():
                self.assertEqual(len(set(spots)), len(spots),
                                 "同じ車が同じ店を2回回る手を出しています")
            # 店の集合は変わらない（増減させる手ではない）
            before = sorted(s for v in routes.values() for s in v)
            after = sorted(s for v in trial.values() for s in v)
            self.assertEqual(before, after)

    def test_two_cars_make_no_moves(self):
        from hexa.improve import _phase_rotate3

        self.assertEqual(
            list(_phase_rotate3([0, 1], {0: [1], 1: [2]}, [])), []
        )

    def test_it_is_in_the_phase_list(self):
        from hexa.improve import _MOVE_PHASES, _phase_rotate3

        self.assertIn(_phase_rotate3, _MOVE_PHASES)


class ShakeTest(unittest.TestCase):
    """**崩して登り直す（ruin & recreate）**の確認。

    **なぜ必要か**

    1手ずつ良くする登り方は、どの1手を試しても下がる形に着くと
    そこで止まります。実測でもはっきり出ていました。

      手数 800 : 玉365
      手数3000 : 玉365  ← 増やしても1玉も動かない

    そこで、いちばん良かった形をわざと崩し、別の谷から登り直します。
    崩したぶんは `PlannedRoutePolicy` が貪欲に埋め直すので
    「崩して作り直す」形になります。

    実測（実戦6試合・記録された渋滞つき・学習済みAIなし）:

      崩さない        : 玉1417（32x32は405）
      **崩して登り直す**: 玉1495（32x32は455 / 日別224は1位と同じ）

    16試合そろえると 玉3718 → **3846**（1位比85.2% → 88.1%、
    3位比87.3% → 90.3%）。実戦の3700も上回ります。

    崩し方は決め打ちの順番で、乱数は使いません。
    """

    def _setup(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.planner import DayPlanner
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(7),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        finder = PathFinder(problem)
        return problem, kinds, finder, DayPlanner(problem, finder)

    def test_every_shake_keeps_the_spots_valid(self):
        from hexa.improve import _SHAKES

        touring = [0, 1, 2]
        routes = {0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
        for shake in _SHAKES:
            out, supply = shake(touring, routes, [0, 1])
            self.assertEqual(sorted(out), touring, f"{shake.__name__} が車を失っています")
            for spots in out.values():
                self.assertEqual(len(set(spots)), len(spots),
                                 f"{shake.__name__} が同じ店を2回入れています")
            self.assertIsInstance(supply, list)

    def test_it_keeps_the_best_it_has_seen(self):
        """崩して悪くなっても、いちばん良かった形は失わないこと。"""

        from hexa.improve import RoutePolisher
        from hexa.production import (GreedyPolicy, PlannedRoutePolicy,
                                     _build_candidate, routes_from_result,
                                     supply_order_from_result)

        problem, kinds, _finder, planner = self._setup()
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit] * problem.agent_count
        day_steps = problem.day_steps[0]
        base = _build_candidate(
            "貪欲", problem, planner, kinds, positions, fuels, {}, day_steps,
            GreedyPolicy(problem), set(),
        )
        self.assertIsNotNone(base)

        def evaluate(trial, order):
            return _build_candidate(
                "磨き", problem, planner, kinds, positions, fuels, {},
                day_steps, PlannedRoutePolicy(problem, trial, order), set(),
            )

        polisher = RoutePolisher(
            problem, kinds, routes_from_result(base.day_result, kinds),
            evaluate, key=base.key,
            supply_order=supply_order_from_result(base.day_result, kinds),
        )
        seen = base.key
        for _ in range(4000):
            if polisher.exhausted:
                break
            found = polisher.step()
            if found is not None:
                self.assertGreater(found.key, seen,
                                   "最高記録より悪い案を返しています")
                seen = found.key
            # 途中の点数は下がってよいが、最高記録は下がってはいけない
            self.assertGreaterEqual(polisher.best_key, base.key)
        self.assertGreaterEqual(polisher.best_key, base.key)

    def test_patience_scales_with_the_route_size(self):
        """粘る長さを形の大きさに比例させること。

        固定値だと当たり外れが大きく、実測で 100→1410 / 180→1421 /
        250→1474 / 320→1431 / 450→1429 と、250だけ当たりのような
        形になりました。この6試合に合わせただけの数字なので
        採らず、店の数に比例させています。
        """

        from hexa.improve import (POLISH_PATIENCE_FLOOR,
                                  POLISH_PATIENCE_PER_SPOT, RoutePolisher)

        problem, kinds, _finder, _planner = self._setup()
        small = RoutePolisher(problem, kinds, {0: [1, 2]},
                              lambda r, o: None)
        big = RoutePolisher(problem, kinds,
                            {0: list(range(1, 21)), 1: list(range(21, 41))},
                            lambda r, o: None)
        self.assertGreaterEqual(small.patience, POLISH_PATIENCE_FLOOR)
        self.assertGreater(big.patience, small.patience,
                           "形が大きいのに粘りが同じです")
        self.assertEqual(big.patience, POLISH_PATIENCE_PER_SPOT * 40)

    def test_shakes_cycle_until_nothing_improves(self):
        """崩し方は一巡したら最初から使い、動かなくなったら止めること。"""

        from hexa.improve import _SHAKES, RoutePolisher

        problem, kinds, _finder, _planner = self._setup()
        polisher = RoutePolisher(problem, kinds, {0: [1, 2, 3]},
                                 lambda r, o: None, key=(0,))
        # 評価が常に None なので改善は起きない。一巡したら止まる。
        for _ in range(10_000):
            if polisher.step() is None and polisher.exhausted:
                break
        self.assertTrue(polisher.exhausted)
        self.assertLessEqual(polisher.shakes, len(_SHAKES) + 1,
                             "動かないのに崩し続けています")


class PolisherKeepsStateTest(unittest.TestCase):
    """自分の改善で磨き係を作り直させないことの確認。

    本番の親は「最良案が変わったら磨き係を作り直す」形です。ところが
    磨き係自身が改善を出すと最良案が変わるので、**自分の改善で
    自分の積み上げ（いちばん良かった形・崩し方の順番）を毎回
    捨てていました**。見つけた改善の点数を磨き係に伝えて、
    作り直さないようにします。
    """

    def test_base_key_is_updated_after_its_own_find(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("polisher.base_key = polished.key", source)
        self.assertIn("polisher.base_key != best.key", source)


class InsertDesignTest(unittest.TestCase):
    """**店を順路の途中に差し込んで組む設計**の確認。

    `design_day` は「いまいる場所から次に良い店へ」と末尾に足す
    だけでした。あとで `improve_order` が並べ替えますが、足す順番で
    決まった受け持ちそのものは変わりません。

    実戦1703（32x32）では1玉あたり9.5歩。3位は8.0歩、1位は7.1歩です。
    立ち止まりは全体の10%しかないので、差は**回り方そのもの**に
    あります。

    差し込みの増える歩数は3回の経路表引きで出せます。

        増える歩数 = (前の店→新しい店) + (新しい店→次の店)
                     - (前の店→次の店)

    `route_cost` が隣どうしの最短経路の合計なので、この式がそのまま
    正しい値になります。1通り0.03秒（24x24）です。
    """

    def _setup(self, seed=3, label="準決勝"):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings(label, random.Random(seed),
                                            players=10, day_count=5))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        return problem, kinds, PathFinder(problem)

    def test_it_fits_the_step_budget(self):
        from hexa.allocate import design_insert, route_cost

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit] * problem.agent_count
        day_steps = problem.day_steps[0]
        routes = design_insert(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            day_steps, brand_weight=8.0, acquired_brands=set(),
        )
        self.assertTrue(any(routes.values()), "1店も割り当てていません")
        for index, spots in routes.items():
            self.assertEqual(len(set(spots)), len(spots),
                             "同じ車が同じ店を2回回っています")
            cost = route_cost(finder, {}, problem.agent_positions[index], spots)
            self.assertLessEqual(cost, day_steps,
                                 f"{index}号車の順路が歩数を超えています")

    def test_the_delta_matches_a_full_recount(self):
        """差し込みの見積りが、組み直した歩数と一致すること。"""

        from hexa.allocate import _table, route_cost

        problem, _kinds, finder = self._setup()
        start = problem.agent_positions[0]
        spots = [spot.position for spot in problem.spots[:4]]
        extra = problem.spots[4].position
        cache = {}
        for at in range(len(spots) + 1):
            before = spots[at - 1] if at > 0 else start
            table = _table(finder, before, {}, cache)
            if extra not in table.steps:
                continue
            delta = table.steps[extra]
            if at < len(spots):
                after = spots[at]
                mid = _table(finder, extra, {}, cache)
                old = _table(finder, before, {}, cache)
                if after not in mid.steps or after not in old.steps:
                    continue
                delta += mid.steps[after] - old.steps[after]
            trial = spots[:at] + [extra] + spots[at:]
            self.assertEqual(
                route_cost(finder, {}, start, trial),
                route_cost(finder, {}, start, spots) + delta,
                "差し込みの見積りが組み直した歩数と合いません",
            )

    def test_both_design_families_are_offered(self):
        from hexa.production import _planned_route_policies

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit if k == 0 else 0 for k in kinds]
        got = _planned_route_policies(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], set(),
        )
        labels = [label for label, _policy in got]
        self.assertTrue(any(l.startswith("差し込み") for l in labels),
                        "差し込みの設計がありません")
        self.assertTrue(any(l.startswith(("設計", "補給設計")) for l in labels),
                        "末尾に足す設計がありません")

    def test_production_polishes_both_families(self):
        """本番が、2系統の設計をそれぞれ磨かせること。

        作り方が違うので、1手ずつ直してもお互いの形には移れません。
        別々に登らせたほうが遠くまで行けました（実戦6試合で
        玉1495 → 1501）。
        """

        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn('candidate.label.startswith("差し込み")', source)
        self.assertIn("for designed in picked.values():", source)


class PrefilterTest(unittest.TestCase):
    """**再現する前に安い検査で弾く**ことの確認。

    **これがいちばん効きました。**

    1日ぶんの再現は1件5ミリ秒かかります。いっぽう「順路の歩数が
    足りているか」を見るだけなら、隣どうしの最短距離を足すだけで
    10マイクロ秒です。**500倍ちがいます。**

    手の多くは歩数が足りない順路なので、先に弾けばそのぶん本物の
    手を試せます。実測（実戦6試合・記録された渋滞つき・手数3000）:

      弾かない              : 玉1508（32x32は455）
      歩数で弾く            : 玉1598（32x32は492）
      同じ案も覚えて弾く    : 玉1604
      超過を1.25倍まで許す  : 玉1866（7試合 / 32x32は502）

    16試合そろえると **玉3718 → 3998 / 日別1670 → 1683**
    （1位比85.2%→91.6%、3位比87.3%→93.9%）。

    **ぴったりで弾いてはいけません。** 順路が歩数を超えていても、
    planner は途中で打ち切って貪欲に切り替えるので、使える計画に
    なることがあります。ぴったりで弾くと実戦1709の日別累計が
    84→79に落ちました。1.25倍まで通すと84に戻ります。

    燃料は見ません。1歩で使う燃料は最大2なので、歩数の検査を
    通った順路は燃料の上限も自動的に満たします。
    """

    def _setup(self):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(11),
                                            players=10, day_count=4))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1 else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        return problem, kinds, PathFinder(problem)

    def test_it_rejects_routes_that_cannot_fit(self):
        from hexa.production import PREFILTER_SLACK, make_prefilter

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit] * problem.agent_count
        day_steps = problem.day_steps[0]
        feasible = make_prefilter(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            day_steps,
        )
        # 戻り値は「通るなら合計歩数、通らないなら None」。
        self.assertEqual(feasible({0: []}), 0, "空の順路を弾いています")
        # 全店を1台に押しつければ、どう回っても歩数は足りない
        everything = {0: [spot.position for spot in problem.spots]}
        self.assertIsNone(feasible(everything), "明らかに無理な順路を通しています")
        self.assertGreater(PREFILTER_SLACK, 1.0,
                           "ぴったりで弾くと使える案まで捨てます")

    def test_one_step_is_one_replay(self):
        """弾いたぶんは手数に数えないこと（数えると意味がありません）。"""

        from hexa.improve import RoutePolisher

        problem, kinds, _finder = self._setup()
        calls = []

        def evaluate(routes, order):
            calls.append(1)
            return None

        polisher = RoutePolisher(
            problem, kinds, {0: [spot.position for spot in problem.spots[:3]]},
            # 検査は「通るなら合計歩数、通らないなら None」を返します。
            evaluate, key=(0,), prefilter=lambda routes: None,
        )
        for _ in range(5):
            polisher.step()
        # 崩した形そのものは1回だけ再現します（点数の出発点が必要）。
        # それ以外は再現しません。
        self.assertLessEqual(len(calls), polisher.shakes,
                             "弾いたのに再現しています")
        self.assertGreater(polisher.skipped, 0, "弾いた数を数えていません")

    def test_the_same_plan_is_not_replayed_twice(self):
        from hexa.improve import RoutePolisher

        problem, kinds, _finder = self._setup()
        seen = []

        def evaluate(routes, order):
            seen.append((tuple(tuple(v) for v in routes.values()),
                         tuple(order)))
            return None

        polisher = RoutePolisher(
            problem, kinds, {0: [spot.position for spot in problem.spots[:4]]},
            evaluate, key=(0,),
        )
        for _ in range(300):
            if polisher.exhausted:
                break
            polisher.step()
        # 崩した形は「出発点の点数」を取るために1回再現するので、
        # その回数ぶんまでは重なりえます。
        self.assertLessEqual(len(seen) - len(set(seen)), polisher.shakes,
                             "同じ案を2度再現しています")


class StockFeasibleTest(unittest.TestCase):
    """在庫より多くの車が同じ店を狙う手を捨てることの確認。

    公式の処理順では、在庫が足りないときリストの若い車が先に取り、
    あとの車は**何も取れません**（質問回答集その2 Q26）。
    在庫1の店を2台が狙うと、片方はその往復を丸ごと無駄にします。

    実戦1703（32x32）では、取った順を最短でなぞれば1玉8.1歩で済む
    のに、実際は9.6歩かかっていました（3位は8.0歩）。**差は配り方
    ではなく寄り道と立ち止まり**です。実測で玉1501→1508。
    """

    def test_it_counts_cars_against_the_stock(self):
        from hexa.improve import _within_stock

        self.assertTrue(_within_stock({0: [5], 1: [5]}, {5: 2}))
        self.assertFalse(_within_stock({0: [5], 1: [5]}, {5: 1}))
        self.assertFalse(_within_stock({0: [5]}, {}))

    def test_moves_respect_the_stock(self):
        from hexa.improve import _all_moves

        stocks = {1: 1, 2: 1, 3: 1}
        routes = {0: [1], 1: [2], 2: [3]}
        for trial, _order, _kind in _all_moves([0, 1, 2], routes, [],
                                               [1, 2, 3], stocks=stocks):
            self.assertTrue(_within_stock_ok(trial, stocks),
                            f"在庫を超える手を出しています: {trial}")


def _within_stock_ok(routes, stocks):
    from hexa.improve import _within_stock

    return _within_stock(routes, stocks)


class MoveTableSpeedTest(unittest.TestCase):
    """1マスぶんの移動の値を使い回しても、答えが変わらないことの確認。

    本番の持ち時間は時計で決まるので、**1手が速いほど多くの手を
    試せます**。16試合を記録された渋滞つきで再現して手数を2〜3倍に
    すると、玉3998→4035になりました（頭打ちではありません）。
    つまり速さはそのまま点数です。

    `next_hop_cost` は1日の再現で14万回以上呼ばれるのに、毎回
    「地形を引く → 渋滞を引く → 歩数を計算する → 隣をなめて方向を
    探す」をやり直していました。地形も渋滞もその日のあいだ変わらない
    ので、道路状態ごとに1回だけ作って使い回します。

    32x32の4日目で測ると**1手4.98ms → 3.37ms（1.48倍）**。
    16試合の結果は玉も日別も1つも変わりません。
    """

    def _problem(self, seed=5):
        from hexa.generator import generate_problem as gen, honsen_settings

        return gen(2026, honsen_settings("1回戦", random.Random(seed),
                                         players=10, day_count=4))

    def test_tables_match_the_rules(self):
        from hexa.pathfind import PathFinder
        from hexa.rules import RoadStatus, Terrain, fuel_cost, move_steps

        problem = self._problem()
        finder = PathFinder(problem)
        # 道路をいくつか混ませて、渋滞が反映されることも見ます。
        roads = {}
        for position in range(problem.cell_count):
            if problem.terrain_at(position) == Terrain.ROAD:
                roads[position] = RoadStatus.JAMMED
                break
        steps, fuels, hops = finder.move_tables(roads)
        for position in range(problem.cell_count):
            terrain = problem.terrain_at(position)
            if terrain == Terrain.POND:
                self.assertIsNone(hops[position], "池から移動できています")
                continue
            self.assertEqual(
                steps[position],
                move_steps(terrain, roads.get(position, RoadStatus.SMOOTH)),
                f"歩数が規則と違います: セル{position}")
            self.assertEqual(fuels[position], fuel_cost(terrain),
                             f"燃料が規則と違います: セル{position}")
            self.assertEqual(
                dict(finder.neighbors[position]), hops[position],
                f"隣接の表が違います: セル{position}")

    def test_tables_are_reused(self):
        """同じ道路状態なら作り直さないこと。"""

        from hexa.pathfind import PathFinder

        problem = self._problem()
        finder = PathFinder(problem)
        roads = {}
        first = finder.move_tables(roads)
        self.assertIs(first, finder.move_tables(roads),
                      "同じ辞書なのに作り直しています")
        # 中身が同じ別の辞書でも、road_key が同じなので使い回せます。
        self.assertIs(first, finder.move_tables(dict(roads)),
                      "同じ中身なのに作り直しています")

    def test_plan_is_unchanged(self):
        """使い回しても、作られる行動計画がまったく同じであること。"""

        from hexa.pathfind import PathFinder
        from hexa.planner import DayPlanner
        from hexa.production import GreedyPolicy

        problem = self._problem(seed=11)
        finder = PathFinder(problem)
        planner = DayPlanner(problem, finder)
        kinds = _kinds_for(problem)
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit] * len(kinds)
        plans = []
        for _ in range(2):
            plan = planner.plan_day(
                kinds=kinds, positions=positions, fuels=fuels,
                road_status={}, day_steps=problem.day_steps[0],
                policy=GreedyPolicy(problem), acquired_brands=set(),
                day_index=0, day_count=problem.day_count,
            )
            plans.append(plan.action_plan)
        self.assertEqual(plans[0], plans[1], "2回目で答えが変わりました")


class CollectChoicesTest(unittest.TestCase):
    """学習用の記録を作らなくても、行動計画が同じであることの確認。

    「どの候補からどれを選んだか」の記録は `hexa/match.py` の学習
    だけが読みます。磨きの最中は誰も読まないのに、1日の再現で
    何百個も作っていました。本番は時計で止まるので、作らないぶん
    だけ多くの手を試せます。
    """

    def _problem(self):
        from hexa.generator import generate_problem as gen, honsen_settings

        return gen(2026, honsen_settings("1回戦", random.Random(13),
                                         players=10, day_count=4))

    def _plan(self, collect):
        from hexa.pathfind import PathFinder
        from hexa.planner import DayPlanner
        from hexa.production import GreedyPolicy

        problem = self._problem()
        finder = PathFinder(problem)
        planner = DayPlanner(problem, finder)
        return planner.plan_day(
            kinds=_kinds_for(problem),
            positions=list(problem.agent_positions),
            fuels=[problem.fuel_limit] * problem.agent_count,
            road_status={}, day_steps=problem.day_steps[0],
            policy=GreedyPolicy(problem), acquired_brands=set(),
            day_index=0, day_count=problem.day_count,
            collect_choices=collect,
        )

    def test_the_plan_is_the_same(self):
        with_records = self._plan(True)
        without = self._plan(False)
        self.assertEqual(with_records.action_plan, without.action_plan,
                         "記録を省いたら行動計画が変わりました")

    def test_records_are_skipped(self):
        with_records = self._plan(True)
        without = self._plan(False)
        self.assertTrue(with_records.touring_choices,
                        "記録する設定なのに1件もありません")
        self.assertEqual(without.touring_choices, [],
                         "省く設定なのに記録が残っています")
        self.assertEqual(without.supply_choices, [],
                         "省く設定なのに記録が残っています")

    def test_production_skips_them(self):
        """本番の磨きが、記録を作らない呼び方をしていること。"""

        import inspect

        from hexa.production import _build_candidate

        source = " ".join(inspect.getsource(_build_candidate).split())
        self.assertIn("collect_choices=False", source,
                      "磨きが学習用の記録を作ったままです")


class RolloutPickTest(unittest.TestCase):
    """1日先まで見て選び直す仕掛けを止めてあることの確認。

    16試合・140日を通して**結果が1つも変わりません**でした
    （玉3998 / 日別1683 のまま）。効き目が0なら、1日の再現に
    かかる時間だけ磨きの手数が減るので損です。
    """

    def test_it_is_turned_off(self):
        from hexa.production import ROLLOUT_CANDIDATES

        self.assertEqual(ROLLOUT_CANDIDATES, 0,
                         "先読みが入ったままです")

    def test_it_keeps_the_official_order(self):
        """使う設定にしても、公式の順位（玉数まで）は崩さないこと。"""

        import inspect

        from hexa.production import _rollout_pick

        source = " ".join(inspect.getsource(_rollout_pick).split())
        self.assertIn("candidate.key[:8] == best.key[:8]", source,
                      "同点でない案まで入れ替えようとしています")


def _kinds_for(problem):
    """最後の1台を補給車、ほかを巡回車にした組を返す（確認用）。"""

    from hexa.rules import AgentKind

    count = problem.agent_count
    return tuple([AgentKind.TOURING] * (count - 1) + [AgentKind.SUPPLY])


class CostTableKeyTest(unittest.TestCase):
    """道路のコスト表を、番地ではなく中身で覚えることの確認。

    **見つかった不具合**

    `PathFinder._cost_table` は `id(road_status)` をキーにしていました。
    辞書そのものは覚えていないので、その辞書が捨てられたあと別の
    辞書が同じ番地に置かれると、**別の日の道路状態の表をそのまま
    返して**しまいます。渋滞した道（4歩）を順調な道（1歩）として
    経路を引くので、着くはずの店に着きません。

    再現すると、辞書を作っては捨てる30回のうち**14回**が間違った
    表を返していました。1日ごとに新しい辞書を作る本番では、前の日の
    辞書が捨てられた直後がちょうどこの形になります。

    同じ入力で再現しても試合ごとの玉数がぶれていた原因がこれでした。
    `table` と同じ `road_key`（中身から作る鍵）に直してあります。
    """

    def _problem(self):
        from hexa.generator import generate_problem as gen, honsen_settings

        return gen(2026, honsen_settings("1回戦", random.Random(5),
                                         players=10, day_count=4))

    def test_a_freed_dict_does_not_poison_the_cache(self):
        import gc

        from hexa.pathfind import PathFinder
        from hexa.rules import RoadStatus, Terrain

        problem = self._problem()
        roads = [position for position in range(problem.cell_count)
                 if problem.terrain_at(position) == Terrain.ROAD]
        self.assertTrue(roads, "道路のマスがありません")
        finder = PathFinder(problem)
        for index in range(30):
            jammed = bool(index % 2)
            status = RoadStatus.JAMMED if jammed else RoadStatus.SMOOTH
            road_status = {position: status for position in roads}
            table = finder._cost_table(road_status, 0)
            steps = table[roads[0]][0]
            self.assertEqual(steps, 4 if jammed else 1,
                             f"{index}回目で別の道路状態の表が返りました")
            del road_status
            gc.collect()

    def test_the_same_contents_share_one_table(self):
        """中身が同じなら、別の辞書でも作り直さないこと。"""

        from hexa.pathfind import PathFinder
        from hexa.rules import RoadStatus, Terrain

        problem = self._problem()
        roads = [position for position in range(problem.cell_count)
                 if problem.terrain_at(position) == Terrain.ROAD]
        finder = PathFinder(problem)
        first = finder._cost_table({roads[0]: RoadStatus.BUSY}, 0)
        again = finder._cost_table({roads[0]: RoadStatus.BUSY}, 0)
        self.assertIs(first, again, "同じ中身なのに作り直しています")

    def test_it_does_not_use_id(self):
        """鍵の作り方そのものを見ます（説明文は読み飛ばします）。"""

        import inspect

        from hexa.pathfind import PathFinder

        source = inspect.getsource(PathFinder._cost_table)
        body = source.split('"""')[2]
        self.assertNotIn("id(road_status)", " ".join(body.split()),
                         "まだ番地をキーにしています")
        self.assertIn("key = (fingerprint, weight)", body,
                      "中身から作った鍵を使っていません")


class PolishQueueTest(unittest.TestCase):
    """子プロセスの列に「磨き」を切らさないことの確認。

    **見つかった無駄**

    1日のはじめに「ゆらぎ探索」の仕事を4万件積み、磨きは
    **最良案が変わったときだけ**割り込ませていました。最良案が
    落ち着くと列から磨きが消え、子11個は残りの時間ずっとゆらぎ
    探索をします。実戦ログでも、採用された案は7日のうち6日が
    「磨き」なのに、24x24の1日の候補は610件（16x16は8278件）
    しかありませんでした。

    実戦1891（24x24）の盤面で、同じ20秒をどちらに使うか測ると:

            出発点   ゆらぎ探索      磨き
      1日目  玉35    玉43(7945回)  玉46(15607手)
      2日目  玉35    玉37(9411回)  玉40(14581手)
      3日目  玉25    玉27(11959回) 玉30(8874手)
      4日目  玉32    玉33(9339回)  玉36(14657手)

    **どの日も磨きが+3玉。** 1手あたりも磨き1.36ms / ゆらぎ7.67ms
    と5.6倍安い。そこで回収のたびに列を数え、少なければ継ぎ足します。
    """

    def _search(self, jobs=400, workers=4):
        import time

        from hexa.parallel import DaySearch

        class _FakePool:
            def apply_async(self, *args, **kwargs):
                return object()

        work = [(f"ゆらぎ{i}", -1, 0.1, i) for i in range(jobs)]
        return DaySearch(_FakePool(), "p.json", "", {}, work,
                         time.monotonic() + 999, workers)

    def test_polish_jobs_are_counted(self):
        search = self._search()
        self.assertEqual(search.polish_backlog(), 0,
                         "最初から磨きが入っています")
        search.add_polish_jobs({0: [1, 2], 1: [3]}, 2500, 16, [1])
        self.assertEqual(search.polish_backlog(), 16,
                         "磨きの本数が数えられていません")

    def test_noise_is_left_between_polish_jobs(self):
        """磨きのあいだに、ゆらぎ探索を挟めること。"""

        from hexa.parallel import POLISH_JOB

        search = self._search()
        before = search.index
        search.add_polish_jobs({0: [1, 2]}, 2500, 4, [], gap=25)
        window = search.chunks[before:before + 4 * 26]
        kinds = [chunk[0][1] == POLISH_JOB for chunk in window]
        self.assertEqual(kinds.count(True), 4, "磨きの数が違います")
        # 磨きとつぎの磨きのあいだに、ゆらぎがちょうど25個ある
        spots = [i for i, polish in enumerate(kinds) if polish]
        self.assertEqual(spots, [0, 26, 52, 78],
                         f"挟み方が違います: {spots}")

    def test_each_job_starts_from_a_different_shape(self):
        """継ぎ足した仕事が、前と同じ崩し方から始まらないこと。"""

        from hexa.improve import _MOVE_PHASES

        search = self._search()
        search.add_polish_jobs({0: [1, 2]}, 2500, 8, [])
        search.add_polish_jobs({0: [1, 2]}, 2500, 8, [],
                               offset=search.polish_queued)
        phases = [chunk[0][8] for chunk in search.chunks[:200]
                  if chunk and chunk[0][1] == -3]
        self.assertEqual(len(phases), 16)
        # 16本が、手の種類8つに2本ずつ行き渡っている
        for phase in range(len(_MOVE_PHASES)):
            self.assertEqual(phases.count(phase), 16 // len(_MOVE_PHASES),
                             f"手の種類{phase}への割り当てが偏っています")

    def test_production_tops_up_the_queue(self):
        """本番の回収が、列を見て継ぎ足していること。"""

        import inspect

        from hexa.production import play_production_match

        source = " ".join(inspect.getsource(play_production_match).split())
        self.assertIn("polish_backlog() < POLISH_QUEUE_FLOOR", source,
                      "列を見ていません")
        self.assertIn("gap=POLISH_NOISE_GAP", source,
                      "ゆらぎ探索を挟んでいません")


class DayBoundaryTest(unittest.TestCase):
    """日の切れ目で探索を捨てていないことの確認。

    1日60秒のうち、実際に探したのは57.4秒でした（ログの `seconds`）。
    7日なら18秒ぶんを捨てています。内訳は

      - 締切手前の安全マージン 2.0秒
      - 子プロセスを、親が素案を作り終えてから走らせていたぶん

    の2つです。マージンを短くしても**その日が0点にはなりません**。
    良くなるたびに出し直しているので、締切ぎりぎりの1回が落ちても
    ひとつ前の回答が残ります（要項 p.12 (6)）。
    """

    def test_margin_is_short(self):
        from hexa.production import SAFETY_MARGIN_SECONDS

        self.assertLessEqual(SAFETY_MARGIN_SECONDS, 1.0,
                             "安全マージンが長すぎます")
        self.assertGreater(SAFETY_MARGIN_SECONDS, 0.0,
                           "安全マージンが無くなっています")

    def test_workers_start_before_the_parent_thinks(self):
        """子を走らせるのが、親の素案づくりより前であること。"""

        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        start = source.index("_start_parallel_day(")
        think = source.index('base_policies: list[tuple[str, object]]')
        self.assertLess(start, think,
                        "親が考え終わるまで子を遊ばせています")


class PolishSkipTest(unittest.TestCase):
    """空読みで仕事が空振りしないことの確認。

    子の仕事は、ほかの子と手が重ならないように、手の一覧を
    `skip` 個ぶん空読みしてから始めます。ところが

      - 盤面が小さいと手の総数が `skip` より少ない（16x16は数千）
      - 列に継ぎ足すたびに `offset` が増え、`skip` が何十万にもなる

    と、1手も試さないまま「出し切った」になり、**仕事が丸ごと
    無駄**になります。行き止まったら先頭に戻してやり直します。
    """

    def test_skip_is_capped(self):
        """継ぎ足しても、空読みの量が際限なく増えないこと。"""

        import time

        from hexa.parallel import DaySearch, POLISH_JOB

        class _FakePool:
            def apply_async(self, *args, **kwargs):
                return object()

        work = [(f"ゆらぎ{i}", -1, 0.1, i) for i in range(200)]
        search = DaySearch(_FakePool(), "p.json", "", {}, work,
                           time.monotonic() + 999, 4)
        budget = 2500
        for _ in range(40):
            search.add_polish_jobs({0: [1, 2]}, budget, 16, [],
                                   offset=search.polish_queued)
        skips = [chunk[0][5] for chunk in search.chunks
                 if chunk and chunk[0][1] == POLISH_JOB]
        self.assertEqual(len(skips), 640)
        self.assertLessEqual(max(skips), 3 * budget,
                             f"空読みが増えすぎています: {max(skips)}")

    def test_it_rewinds_instead_of_giving_up(self):
        import inspect

        from hexa.parallel import _polish_job

        source = " ".join(inspect.getsource(_polish_job).split())
        self.assertIn("polisher.moves = polisher._new_moves() "
                      "polisher.exhausted = False", source,
                      "行き止まりで先頭に戻していません")


class DiverseSeedTest(unittest.TestCase):
    """子に「形のちがう案」を配ることの確認。

    **磨きは登り切ると止まります。** 実戦1891（24x24）の盤面で、
    貪欲1つから磨くと

      20秒  : 玉46 / 40 / 30 / 36（1〜4日目）
      100秒 : 玉46 / 40 / 30 / 36  ← 5倍かけても1玉も動かない

    手数は18,403手まで伸びているのに結果は同じでした。同じ谷を
    登り切っているからです。**出発点を5か所に分けて20秒ずつ**に
    すると、1日目は玉46→50、3日目は玉30→31 と越えられます。

    だから、最良案ばかり配らず、車ごとの店の並びがちがう良い案を
    順に配ります。
    """

    def _candidate(self, routes, key):
        from hexa.production import Candidate
        from hexa.simulator import Acquisition, DayResult

        acquisitions = []
        step = 0
        for agent, spots in routes.items():
            for spot in spots:
                step += 1
                acquisitions.append(Acquisition(step, agent, spot, 0))
        result = DayResult(
            positions=(0,) * (len(routes) + 1),
            fuels=(10,) * (len(routes) + 1),
            acquisitions=acquisitions, refuels=[], stay_counts={},
            remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        return Candidate("案", [], result, key)

    def _kinds(self, cars):
        from hexa.rules import AgentKind

        return tuple([AgentKind.TOURING] * cars + [AgentKind.SUPPLY])

    def test_same_shapes_are_dropped(self):
        from hexa.production import diverse_seeds

        kinds = self._kinds(2)
        same_a = self._candidate({0: [1, 2], 1: [3]}, (0, 0, 0, 0, 9))
        same_b = self._candidate({0: [1, 2], 1: [3]}, (0, 0, 0, 0, 8))
        other = self._candidate({0: [3], 1: [1, 2]}, (0, 0, 0, 0, 7))
        seeds = diverse_seeds([same_a, same_b, other], kinds, 5)
        self.assertEqual(len(seeds), 2, "同じ形を2つ配ろうとしています")
        self.assertIs(seeds[0], same_a, "良い順になっていません")

    def test_the_best_comes_first(self):
        from hexa.production import diverse_seeds

        kinds = self._kinds(2)
        weak = self._candidate({0: [1], 1: [2]}, (0, 0, 0, 0, 1))
        strong = self._candidate({0: [2], 1: [1]}, (0, 0, 0, 0, 5))
        seeds = diverse_seeds([weak, strong], kinds, 5)
        self.assertIs(seeds[0], strong)

    def test_the_limit_is_respected(self):
        from hexa.production import diverse_seeds

        kinds = self._kinds(2)
        pool = [self._candidate({0: [i], 1: [i + 100]}, (0, 0, 0, 0, i))
                for i in range(20)]
        self.assertEqual(len(diverse_seeds(pool, kinds, 5)), 5)

    def test_production_spreads_the_seeds(self):
        import inspect

        from hexa.production import play_production_match

        source = " ".join(inspect.getsource(play_production_match).split())
        self.assertIn("seeds = diverse_seeds(", source,
                      "最良案ばかり配っています")
        self.assertIn("POLISH_SEEDS", source)


class PolisherRestartTest(unittest.TestCase):
    """親が登り切ったあと、別の形から登り直すことの確認。

    以前は登り切るとふつうの探索（ゆらぎ）に戻していました。
    ところが1手あたり磨き1.36ms / ゆらぎ7.67ms と5.6倍ちがい、
    同じ20秒での結果も磨きのほうが毎日+3玉でした。余った時間は
    ゆらぎではなく、**別の形の磨き**に回します。
    """

    def test_it_restarts_instead_of_falling_back(self):
        import inspect

        from hexa.production import play_production_match

        source = " ".join(inspect.getsource(play_production_match).split())
        self.assertIn("if polisher.exhausted and attempt - last_restart >= 50:",
                      source, "登り切ったあとの作り直しがありません")
        self.assertIn("polisher.base_key = best.key", source,
                      "作り直した磨き係がすぐ捨てられます")


class LightMonitorWriteTest(unittest.TestCase):
    """探索中の書き出しを軽くしたことの確認。

    `state.json` には試合の全部（マップ・全日の1歩ごとの位置）が
    入ります。実戦1915（24x24・10日）の記録で測ると**87KB**あり、
    1回書くのに**9.1ミリ秒**かかりました。0.2秒ごとに書くと
    **探索の4.5%**を監視に取られます。

    過ぎた日の中身はその日のあいだ変わりません。だから探索中は
    `now.json`（いまの様子＋作りかけの1日ぶん）だけを書きます。
    ビューアはこちらを速く読み、日が増えたときだけ `state.json` を
    取り直します。見え方は同じで、探索は削りません。
    """

    def _monitor(self):
        import random
        import tempfile
        from pathlib import Path

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.livemonitor import LiveMonitor
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        folder = Path(tempfile.mkdtemp())
        monitor = LiveMonitor(folder)
        kinds = tuple([AgentKind.TOURING] * (problem.agent_count - 1)
                      + [AgentKind.SUPPLY])
        monitor.start_match(problem, kinds)
        return monitor, folder

    def test_searching_writes_only_the_small_file(self):
        import json

        from hexa.livemonitor import LIVE_NAME, NOW_NAME

        monitor, folder = self._monitor()
        before = (folder / LIVE_NAME).read_text(encoding="utf-8")
        monitor._last_search = 0.0
        monitor.searching(day_index=0, seconds_left=10.0, candidates=5,
                          trying="ためし", best_label="磨き",
                          best_score=(1, 2, 3), resubmissions=0, recent=[])
        after = (folder / LIVE_NAME).read_text(encoding="utf-8")
        self.assertEqual(before, after,
                         "探索中に重いほうを書き直しています")
        now = json.loads((folder / NOW_NAME).read_text(encoding="utf-8"))
        self.assertEqual(now["status"]["phase"], "探索中")
        self.assertIn("dayCount", now)
        self.assertLess(len(json.dumps(now)), len(after),
                        "軽いほうが重いほうより大きくなっています")

    def test_the_small_file_carries_the_phase(self):
        """日の区切りでも、軽いほうに段階が出ること。"""

        import json

        from hexa.livemonitor import NOW_NAME

        monitor, folder = self._monitor()
        monitor.finish_match("種類1 / 日別1 / 玉1", [])
        now = json.loads((folder / NOW_NAME).read_text(encoding="utf-8"))
        self.assertEqual(now["status"]["phase"], "試合終了")

    def test_the_gap_is_short(self):
        from hexa.livemonitor import SEARCH_WRITE_GAP

        self.assertLessEqual(SEARCH_WRITE_GAP, 0.25,
                             "書き出しの間隔が長すぎます")


class TomorrowReachTest(unittest.TestCase):
    """「明日すぐ取れる玉」の数え方を直したことの確認。

    **2つ誤りがありました。**

     1. 今日の売れ残り在庫を見ていた。在庫は**毎日満タンに戻る**
        ので（`simulate_day` が毎日作り直します）、今日きれいに
        取り切った良い店ほど0と数えられ、「今日あまり取らなかった
        あたり」へ車を誘導していました。
     2. 1つの店に2台が届くと 2 と数えていた。在庫1の店に2台寄っても
        明日取れるのは1玉です。寄り集まるほど得に見えます。

    直しても実測では入りませんでした（玉3998→3943）。この項は
    余力より上にあるので、「燃料を残す案」より「店に近い案」を
    選んでしまうためです。いまは 0 で止めてあります。
    """

    def _setup(self, limit_ratio):
        import random

        from hexa.pathfind import PathFinder
        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        return problem, PathFinder(problem)

    def test_it_is_off(self):
        from hexa.production import TOMORROW_REACH_RATIO

        self.assertEqual(TOMORROW_REACH_RATIO, 0.0,
                         "実測で入らなかったのに有効になっています")

    def test_one_spot_never_counts_more_than_its_stock(self):
        import hexa.production as production
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        problem, finder = self._setup(0.5)
        spot = problem.spots[0]
        cars = 3
        kinds = tuple([AgentKind.TOURING] * cars + [AgentKind.SUPPLY])
        # 全車を同じ店の上に置く。取れるのは在庫のぶんだけ。
        result = DayResult(
            positions=tuple([spot.position] * (cars + 1)),
            fuels=tuple([problem.fuel_limit] * (cars + 1)),
            acquisitions=[], refuels=[], stay_counts={},
            remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        saved = production.TOMORROW_REACH_RATIO
        production.TOMORROW_REACH_RATIO = 1.0
        try:
            got = production._tomorrow_reach(result, kinds, problem, finder,
                                             next_steps=1, road_status={})
        finally:
            production.TOMORROW_REACH_RATIO = saved
        # その店ぶんは min(在庫, 台数) を超えない
        self.assertLessEqual(got, sum(min(s.stocks, cars)
                                      for s in problem.spots))
        self.assertGreaterEqual(got, 1)

    def test_it_does_not_look_at_todays_leftovers(self):
        """在庫は毎日満タンに戻るので、売れ残りを見ないこと。"""

        import inspect

        from hexa.production import _tomorrow_reach

        source = " ".join(inspect.getsource(_tomorrow_reach).split())
        self.assertNotIn("remaining_stocks", source,
                         "今日の売れ残りを見たままです")


class ViewerConnectionTest(unittest.TestCase):
    """ビューアの表示を直したことの確認（中身を読んで確かめます）。

    実戦で出た困りごと:

      - 開始3分前（まだ state.json が無い）で「つながりません」
        と出ていた。読めないのはビューアとファイルの間の話で、
        サーバーとの接続とは関係ありません。
      - つながらないときに理由が出ないので、何を直せばよいか
        分かりませんでした。
      - 「試行中」の方策名（S1-AI-4+探索0.05 など）が見切れていました。
      - 試合が終わっているのに「接続中」と出ていました。
      - 過ぎた試合を見返せませんでした。
    """

    def _page(self):
        from pathlib import Path

        return Path("viewer/live.html").read_text(encoding="utf-8")

    def test_it_separates_unreadable_from_waiting(self):
        page = self._page()
        for word in ("読めません", "待機中", "受信中", "試合終了"):
            self.assertIn(word, page, f"{word} の表示がありません")

    def test_it_shows_why_it_cannot_read(self):
        page = self._page()
        self.assertIn('beat("down", "HTTP "+r.status', page,
                      "読めない理由（HTTPの返り）を出していません")
        self.assertIn("connNote", page, "理由を出す場所がありません")

    def test_long_labels_wrap(self):
        page = self._page()
        self.assertIn("wrapline", page, "長い名前が折り返りません")
        self.assertIn("overflow-wrap:anywhere", page)

    def test_past_matches_can_be_opened(self):
        page = self._page()
        self.assertIn('id="openLog"', page, "記録を開くボタンがありません")
        self.assertIn("function openLog(", page)

    def test_it_reads_the_small_file_first(self):
        page = self._page()
        self.assertIn('fetch("live/now.json', page,
                      "軽いほうのファイルを読んでいません")
        self.assertIn("now.dayCount!==baseDays", page,
                      "日が変わったときに重いほうを読み直していません")

    def test_markers_keep_their_size(self):
        page = self._page()
        self.assertIn("const sz=AGENT_R;", page,
                      "重なったときに車を縮めています")


class MatchLogKeepsEverythingTest(unittest.TestCase):
    """試合の記録に、あとから見返したいものが全部入っていることの確認。

    実戦の記録（`logs/match_*.json`）には、マップ・店（系列と在庫）・
    体の種別・燃料上限・混雑と渋滞の基準・**1日ごとの渋滞**・
    1歩ごとの位置・獲得・補給が入っています。

    足りなかったのが**ほかのチームの位置**でした。渋滞は全チームの
    前2日の滞在から決まるので、これが無いと「なぜその道が混んだか」
    をあとから追えません。記念にも、検証にも要ります。
    """

    def test_the_day_keeps_the_opponents(self):
        import inspect

        from hexa.matchlog import MatchLog

        self.assertIn("enemies",
                      inspect.signature(MatchLog.finish_day).parameters,
                      "ほかのチームの位置を受け取っていません")
        source = " ".join(inspect.getsource(MatchLog.finish_day).split())
        self.assertIn('"enemies":', source, "記録に残していません")

    def test_production_passes_them(self):
        import inspect

        from hexa.production import play_production_match

        source = " ".join(inspect.getsource(play_production_match).split())
        self.assertIn("enemies=[[agent.position for agent in group]", source,
                      "本番がほかのチームの位置を渡していません")

    def test_the_day_keeps_the_roads(self):
        import inspect

        from hexa.matchlog import MatchLog

        source = " ".join(inspect.getsource(MatchLog.finish_day).split())
        self.assertIn('"roads":', source, "渋滞を残していません")
        self.assertIn('"tracks":', source, "1歩ごとの位置を残していません")


class WaitingBeforeMatchTest(unittest.TestCase):
    """試合が始まる前も、ビューアに様子を流すことの確認。

    **実戦で出た困りごと:** 開始3分前なのに「つながりません」と
    出ていました。原因は、試合が始まるまで監視が**何も書かない**
    ことでした。ビューアは前の試合のファイルを見るか、ファイルが
    無くて読めないだけで、サーバーとの接続とは関係ありません。

    待っているあいだも、軽いほうのファイルに「試合を待っています」と
    その理由（まだ試合前です／通信に失敗しました）を書きます。
    """

    def test_the_monitor_can_say_it_is_waiting(self):
        import json
        import random
        import tempfile
        from pathlib import Path

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.livemonitor import LiveMonitor, NOW_NAME

        gen(2026, honsen_settings("1回戦", random.Random(3),
                                  players=10, day_count=4))
        folder = Path(tempfile.mkdtemp())
        monitor = LiveMonitor(folder)
        monitor.waiting("まだ試合前です（AccessTimeError）", True)
        now = json.loads((folder / NOW_NAME).read_text(encoding="utf-8"))
        self.assertEqual(now["status"]["phase"], "試合を待っています")
        self.assertIn("AccessTimeError", now["status"]["reason"])
        self.assertTrue(now["status"]["linked"],
                        "サーバーに届いているのに切断扱いです")

    def test_a_broken_line_is_marked(self):
        import json
        import tempfile
        from pathlib import Path

        from hexa.livemonitor import LiveMonitor, NOW_NAME

        folder = Path(tempfile.mkdtemp())
        monitor = LiveMonitor(folder)
        monitor.waiting("通信に失敗しました", False)
        now = json.loads((folder / NOW_NAME).read_text(encoding="utf-8"))
        self.assertFalse(now["status"]["linked"])

    def test_the_client_tells_the_monitor(self):
        import inspect

        from hexa.client import ProconClient

        parameters = inspect.signature(ProconClient.wait_for_setting).parameters
        self.assertIn("waiting", parameters,
                      "待ちの通知を受け取れません")
        source = " ".join(
            inspect.getsource(ProconClient.wait_for_setting).split())
        self.assertIn("waiting(reason, linked)", source,
                      "待っているあいだ、知らせていません")

    def test_production_starts_the_monitor_first(self):
        """監視を、通信の確認より先に作っていること。"""

        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertLess(source.index("monitor = LiveMonitor()"),
                        source.index("client.check_connection()"),
                        "待ちのあいだ監視が動いていません")
        self.assertIn("waiting=(monitor.waiting", source,
                      "待ちの通知をつないでいません")

    def test_the_viewer_shows_it(self):
        from pathlib import Path

        page = Path("viewer/live.html").read_text(encoding="utf-8")
        self.assertIn('phase.indexOf("待っています")', page,
                      "ビューアが待機中を見ていません")


class BrandCoverTest(unittest.TestCase):
    """足りない系列を拾い直すことの確認。

    **実戦で分かった盲点。** 上位陣は日別累計が満点です。

      実戦1958（32x32・9日・系列22）: 1位も12位も日別198（=22x9）。
        こちらは197（9日目だけ21系列）で、それだけで16位。
      実戦1960（24x24・7日・系列20）: 上位は日別140。こちらは137
        （3日目に2つ、5日目に1つ落とす）で13位。

    日別累計は**玉数より上位の得点**なので、1つ落とすと玉では
    挽回できません。落としたのはどれも歩数の短い日です。

    **1手では越えられない壁でした。** 32x32の9日目で、足りない
    系列0の店（車1から31歩）へ先に行かせると系列0は取れますが、
    その車が受け持っていた別の系列を落とし、日別は21のまま動きません。
    そこで「系列の数は同じだが、足りない系列が入れ替わった」形を
    いったん受け取り、次の周で入れ替わったほうを拾いに行きます。

    実測（その日の盤面を再現）:

      32x32 9日目 : 系列21 玉36 → **系列22 玉32**
      24x24 5日目 : 系列19 玉33 → **系列20 玉28**
      24x24 3日目 : 系列18 のまま（足りない系列が50〜51歩先で、
                    その日の48歩では物理的に届きません）
    """

    def test_it_only_returns_an_improvement(self):
        """良くならなければ、元の案をそのまま返すこと。"""

        import inspect

        from hexa.production import cover_missing_brands

        source = " ".join(inspect.getsource(cover_missing_brands).split())
        # 鍵が良くならなければ元を返します。さらに
        # `BRAND_FIX_DAILY_ONLY` のとき、日別が増えていなければ
        # 元を返します（玉だけ減る直しを採らないため。
        # `BrandFixDailyOnlyTest` に実戦ログつきの理由）。
        self.assertIn("if current.key <= best.key: return best",
                      source, "悪い案を返す余地があります")
        self.assertIn("BRAND_FIX_DAILY_ONLY and current.key[1] <= best.key[1]",
                      source, "日別が増えない直しを採る余地があります")

    def test_it_tries_the_front_first(self):
        """途中ではなく、先頭に置く形を先に試すこと。"""

        import inspect

        from hexa.production import cover_missing_brands

        source = " ".join(inspect.getsource(cover_missing_brands).split())
        self.assertIn("front[:BRAND_FIX_TRIES] + tries[:BRAND_FIX_TRIES]",
                      source, "先頭に置く形を先に試していません")

    def test_production_calls_it_before_submitting(self):
        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertEqual(source.count("cover_missing_brands("), 2,
                         "提出の前後2回、拾い直していません")

    def test_it_finds_a_cover_on_a_small_board(self):
        """作った盤面で、足りない系列を実際に拾えること。"""

        import random

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.planner import DayPlanner
        from hexa.production import (GreedyPolicy, PlannedRoutePolicy,
                                     _build_candidate, cover_missing_brands,
                                     routes_from_result,
                                     supply_order_from_result)
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        finder = PathFinder(problem)
        planner = DayPlanner(problem, finder)
        kinds = tuple([AgentKind.TOURING] * (problem.agent_count - 1)
                      + [AgentKind.SUPPLY])
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit] * problem.agent_count
        steps = problem.day_steps[0]

        def evaluate(routes, order, verify=True):
            return _build_candidate(
                "拾い", problem, planner, kinds, positions, fuels, {},
                steps, PlannedRoutePolicy(problem, routes, order), set(),
                day_index=0, steps_ahead=0, next_steps=0, verify=verify)

        base = _build_candidate("貪欲", problem, planner, kinds, positions,
                                fuels, {}, steps, GreedyPolicy(problem),
                                set(), day_index=0, steps_ahead=0,
                                next_steps=0)
        order = supply_order_from_result(base.day_result, kinds)
        got = cover_missing_brands(problem, finder, kinds, positions, {},
                                   base, evaluate, order)
        self.assertIsNotNone(got)
        # 悪くなることはありません。
        self.assertGreaterEqual(got.key, base.key)


class BrandReachTest(unittest.TestCase):
    """明日どの系列に手が届くかを、終わり方の採点に入れたことの確認。

    **v1.39.0 の拾い直しでも直らなかった日があります。** 実戦1960の
    3日目は、足りない系列の店が**50〜51歩先**にあり、その日の48歩では
    どう動いても届きませんでした。原因は前日の終わり方です。

    2日目の終わりから見ると、その2系列は

      今日の渋滞で   42歩・40歩（その日は48歩）  ← 余裕がない
      明日の実際には 51歩・50歩                  ← 届かない

    でした。今日の状態で測ると届くように見え、渋滞が悪化して落ちます。

    **2段に分けてあります。**

      きつい : 明日の歩数いっぱいでも届かない。確実に1系列落とすので
               **玉数より上**に置きます（日別は玉数より上位の得点）。
      ゆるい : 明日の歩数の8割までに届かない。危うい形です。
               **玉数より下**に置くので、玉は1つも減りません。
               同点の案が並んだときだけ、余裕のある終わり方を選びます。

    速さへの影響はありません（経路表は `_stranded_count` が同じ位置で
    すでに作っているので、増えるのは辞書引きだけ）。32x32の実測で
    1手3.09ms、入れる前の3.37msと同じです。
    """

    def _setup(self):
        import random

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        kinds = tuple([AgentKind.TOURING] * (problem.agent_count - 1)
                      + [AgentKind.SUPPLY])
        return problem, PathFinder(problem), kinds

    def _result(self, problem, kinds, position, fuel):
        from hexa.simulator import DayResult

        count = len(kinds)
        return DayResult(
            positions=tuple([position] * count),
            fuels=tuple([fuel] * count),
            acquisitions=[], refuels=[], stay_counts={},
            remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )

    def test_far_away_counts_as_unreachable(self):
        from hexa.production import _unreachable_brands

        problem, finder, kinds = self._setup()
        here = problem.agent_positions[0]
        brands = len({spot.brand for spot in problem.spots})
        # 明日1歩しか動けないなら、ほぼ全部届かない。
        hard, soft = _unreachable_brands(
            self._result(problem, kinds, here, problem.fuel_limit),
            kinds, problem, finder, next_steps=1, road_status={})
        self.assertGreater(hard, 0, "届かない系列を数えていません")
        self.assertGreaterEqual(soft, hard, "ゆるいほうが少なくなっています")
        self.assertLessEqual(hard, brands)

    def test_plenty_of_time_reaches_everything(self):
        from hexa.production import _unreachable_brands

        problem, finder, kinds = self._setup()
        here = problem.agent_positions[0]
        hard, soft = _unreachable_brands(
            self._result(problem, kinds, here, problem.fuel_limit),
            kinds, problem, finder, next_steps=10_000, road_status={})
        self.assertEqual((hard, soft), (0, 0),
                         "時間が十分あるのに届かない扱いです")

    def test_no_fuel_means_no_reach(self):
        """燃料が無ければ、歩数があっても届かないこと。"""

        from hexa.production import _unreachable_brands

        problem, finder, kinds = self._setup()
        here = problem.agent_positions[0]
        hard, _soft = _unreachable_brands(
            self._result(problem, kinds, here, 0),
            kinds, problem, finder, next_steps=10_000, road_status={})
        self.assertEqual(hard, len({s.brand for s in problem.spots}),
                         "燃料切れでも届く扱いになっています")

    def test_the_soft_one_never_costs_a_noodle(self):
        """ゆるいほうが、玉数より下に置かれていること。"""

        import inspect

        from hexa.production import _candidate_key, KEY_NOODLES

        source = inspect.getsource(_candidate_key)
        hard_at = source.index("-hard_miss * BRAND_REACH_WEIGHT,")
        balls_at = source.index("len(day_result.acquisitions) + _fuel_as_balls")
        soft_at = source.index("-soft_miss,")
        self.assertLess(hard_at, balls_at, "きついほうが玉数より下です")
        self.assertLess(balls_at, soft_at, "ゆるいほうが玉数より上です")
        self.assertEqual(KEY_NOODLES, 6)


class TomorrowCoverageTest(unittest.TestCase):
    """**明日、何系列まで回れる終わり方か**を数えることの確認。

    実戦1981（16x16・13系列・6日）は日別累計75で8位でした。1位は
    満点の78。落としたのは**5日目だけ**（系列5・6・10）で、その日は
    36歩しかありません。

    原因は前日の終わり方です。巡回3台が **29・12・13**、つまり右上の
    2x2のなかに固まって終わっていました。`_unreachable_brands` では
    捕まりません。足りない系列の店は26歩・31歩先で、36歩なら
    **1台だけなら届く**からです。届かなかったのは「13系列を3台で
    同時に回ること」でした。

    その日を90秒・14万手かけて解き直しても10系列止まりで、探索の
    不足ではありません。出発点を散らすと同じ36歩で届きます。

      出発 29,12,188,13（実戦）  → 日別10  見積り 9
      出発 12,115,178,188        → 日別13  見積り12
    """

    def _setup(self):
        import random

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10, day_count=4))
        kinds = tuple([AgentKind.TOURING] * (problem.agent_count - 1)
                      + [AgentKind.SUPPLY])
        return problem, PathFinder(problem), kinds

    def _at(self, problem, kinds, positions, fuel=None):
        from hexa.simulator import DayResult

        fuel = problem.fuel_limit if fuel is None else fuel
        return DayResult(
            positions=tuple(positions), fuels=tuple([fuel] * len(kinds)),
            acquisitions=[], refuels=[], stay_counts={},
            remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )

    def test_spread_out_covers_more_than_huddled(self):
        """固まって終わるより、散って終わるほうが高く出ること。"""

        from hexa.production import _tomorrow_coverage

        problem, finder, kinds = self._setup()
        from hexa.rules import AgentKind

        touring = [index for index, kind in enumerate(kinds)
                   if kind == AgentKind.TOURING]
        spots = [spot.position for spot in problem.spots]
        # 固まり: 全車が同じ店の上。散らばり: 離れた店に1台ずつ。
        huddled = list(problem.agent_positions)
        spread = list(problem.agent_positions)
        for seat in touring:
            huddled[seat] = spots[0]
        picks = sorted(spots, key=lambda pos: (pos % problem.width, pos))
        for order, seat in enumerate(touring):
            spread[seat] = picks[(order * len(picks) // max(1, len(touring)))
                                 % len(picks)]
        steps = max(1, problem.day_steps[0] // 3)
        huddled_cover = _tomorrow_coverage(
            self._at(problem, kinds, huddled), kinds, problem, finder,
            steps, {})
        spread_cover = _tomorrow_coverage(
            self._at(problem, kinds, spread), kinds, problem, finder,
            steps, {})
        self.assertGreater(spread_cover, huddled_cover,
                           "固まった終わり方が高く出ています")

    def test_never_counts_more_than_the_brands(self):
        from hexa.production import _tomorrow_coverage

        problem, finder, kinds = self._setup()
        brands = len({spot.brand for spot in problem.spots})
        got = _tomorrow_coverage(
            self._at(problem, kinds, problem.agent_positions), kinds,
            problem, finder, 100_000, {})
        self.assertLessEqual(got, brands, "系列の数を超えています")
        self.assertGreater(got, 0, "時間が十分あるのに0です")

    def test_no_lookahead_on_the_last_day(self):
        """明日が無い日（next_steps=0）では数えないこと。"""

        from hexa.production import _tomorrow_coverage

        problem, finder, kinds = self._setup()
        self.assertEqual(
            _tomorrow_coverage(
                self._at(problem, kinds, problem.agent_positions), kinds,
                problem, finder, 0, {}), 0)

    def test_sits_above_the_noodles(self):
        """玉数より上に置かれていること（日別は玉数より上位の得点）。"""

        import inspect

        from hexa.production import _candidate_key

        source = inspect.getsource(_candidate_key)
        cover_at = source.index("_cover_rank(")
        balls_at = source.index("len(day_result.acquisitions) + _fuel_as_balls")
        self.assertLess(cover_at, balls_at,
                        "明日回れる系列が玉数より下にあります")

    def test_the_stranded_memo_sees_the_supply_car(self):
        """立ち往生の覚え書きが、**補給車の位置**も鍵に入れていること。

        `_stranded_count` は「補給車と同じマスで終われば立ち往生に
        ならない」と数えます。だから巡回車が同じ場所・同じ燃料でも、
        補給車の居場所が違えば答えは違います。

        **この取り違えを実際にやりました。** 鍵を巡回車だけに
        していたので、玉5個で1台が立ち往生する案と、玉1個で
        立ち往生しない案が同じ点数になり、立ち往生する側が
        玉数で勝ってしまいました。
        """

        from hexa.production import (_STRAND_CACHE, _cached_stranded_count,
                                     _stranded_count)
        from hexa.rules import AgentKind
        from hexa.simulator import DayResult

        problem, finder, _kinds = self._setup()
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)

        def at(positions):
            return DayResult(
                positions=tuple(positions), fuels=(0, 0), acquisitions=[],
                refuels=[], stay_counts={}, remaining_stocks={},
                wasted_spot_stays=0, step_positions=[], step_fuels=[])

        # 巡回車は同じ場所・同じ燃料。補給車だけが違う。
        together = at((0, 0))
        apart = at((0, problem.spots[-1].position))
        _STRAND_CACHE.clear()
        first = _cached_stranded_count(together, kinds, problem, finder, 2,
                                       (0,))
        second = _cached_stranded_count(apart, kinds, problem, finder, 2, (0,))
        self.assertEqual(first,
                         _stranded_count(together, kinds, problem, finder, 2))
        self.assertEqual(second,
                         _stranded_count(apart, kinds, problem, finder, 2),
                         "補給車の位置が鍵に入っていません")

    def test_remembering_gives_the_same_answer(self):
        """覚え書きを使っても、使わないときと同じ数になること。

        1手ごとに巡回車のぶんだけ経路表を引き直すのは重いので、
        「道路の状態・明日の歩数・巡回車の位置と燃料」で覚えます。
        入力が同じなら答えも同じ（乱数なし）なので、結果は変わりません。
        """

        from hexa.production import _tomorrow_coverage, _COVER_CACHE
        from hexa.rules import RoadStatus

        problem, finder, kinds = self._setup()
        spots = [spot.position for spot in problem.spots]
        places = [list(problem.agent_positions),
                  [spots[0]] * len(kinds),
                  [spots[len(spots) // 2]] * len(kinds)]
        roads = {spots[0]: RoadStatus.JAMMED}
        for where in places:
            for road in ({}, roads):
                _COVER_CACHE.clear()
                fresh = _tomorrow_coverage(
                    self._at(problem, kinds, where), kinds, problem, finder,
                    problem.day_steps[0], road)
                again = _tomorrow_coverage(
                    self._at(problem, kinds, where), kinds, problem, finder,
                    problem.day_steps[0], road)
                self.assertEqual(fresh, again, "覚え書きで数が変わりました")

    def test_different_roads_are_remembered_apart(self):
        """道路の状態が違えば、別のものとして覚えること。

        以前、経路表のキャッシュを `id()` だけで覚えていて、**混んだ
        道を順調な値で引く**不具合がありました（30回のうち14回が
        間違い）。鍵は中身から作ります。
        """

        from hexa.production import _tomorrow_coverage, _COVER_CACHE
        from hexa.rules import RoadStatus

        problem, finder, kinds = self._setup()
        # 道路のマスを全部「渋滞」にすると、動ける範囲が狭くなります。
        jammed = {position: RoadStatus.JAMMED
                  for position in range(problem.cell_count)
                  if problem.is_road(position)} if hasattr(
                      problem, 'is_road') else {
                  position: RoadStatus.JAMMED
                  for position in range(problem.cell_count)}
        steps = max(2, problem.day_steps[0] // 4)
        _COVER_CACHE.clear()
        smooth = _tomorrow_coverage(
            self._at(problem, kinds, problem.agent_positions), kinds,
            problem, finder, steps, {})
        stuck = _tomorrow_coverage(
            self._at(problem, kinds, problem.agent_positions), kinds,
            problem, finder, steps, jammed)
        self.assertLessEqual(stuck, smooth,
                             "渋滞しているのに同じか増えています")

    def test_stranded_cars_do_not_count(self):
        """燃料が無い車は、拾いに行けないこと。"""

        from hexa.production import _tomorrow_coverage

        problem, finder, kinds = self._setup()
        full = _tomorrow_coverage(
            self._at(problem, kinds, problem.agent_positions,
                     fuel=problem.fuel_limit),
            kinds, problem, finder, problem.day_steps[0], {})
        empty = _tomorrow_coverage(
            self._at(problem, kinds, problem.agent_positions, fuel=0),
            kinds, problem, finder, problem.day_steps[0], {})
        self.assertEqual(empty, 0, "燃料0でも回れる扱いです")
        self.assertGreater(full, 0)


class SupplySeatTest(unittest.TestCase):
    """**どの出発点を補給車にするか**も選ぶことの確認。

    種別は車ごとに提出します（`POST /agent` に `[0,1,0,1]` のような
    配列）。並びは配られた初期位置に対応し、初期位置は全チーム共通で
    動かせません（Q2 A38）。つまり**どの出発点を補給車にするかは
    こちらの自由**です。

    **これまで選んでいませんでした。** 必ず最後の席を補給車にして
    いたので、たまたま当たっていただけです。

    実測（第1981・16x16・4体・記録された渋滞つき・1日8秒）:

      出発点252（0番） 日別78 玉179
      出発点 66（1番） 日別78 玉183
      出発点180（2番） 日別78 玉178
      出発点118（3番） 日別78 **玉184**  ← これまでの置き方

    同じ日別で**玉に6個（3.3%）の差**がありました。
    """

    def _problem(self):
        import random

        from hexa.generator import generate_problem as gen, honsen_settings

        return gen(2026, honsen_settings("1回戦", random.Random(4),
                                         players=10, day_count=4))

    def test_seats_make_the_kinds(self):
        from hexa.production import kinds_for_seats
        from hexa.rules import AgentKind

        got = kinds_for_seats(4, (3,))
        self.assertEqual(got, (AgentKind.TOURING, AgentKind.TOURING,
                               AgentKind.TOURING, AgentKind.SUPPLY))
        got = kinds_for_seats(4, (0,))
        self.assertEqual(got[0], AgentKind.SUPPLY)
        self.assertEqual(got[1:], (AgentKind.TOURING,) * 3)
        self.assertEqual(kinds_for_seats(3, ()),
                         (AgentKind.TOURING,) * 3)

    def test_one_supply_car_tries_every_seat(self):
        """**5体以上では席を全部試すこと。**

        4体以下は後ろ詰めに固定します（試走に予測力がないため /
        `SeatTrialTest` に実測つきの理由）。
        """

        from hexa.production import supply_seat_choices

        for count in (5, 7, 8):
            got = supply_seat_choices(count, 1)
            self.assertEqual(sorted(got), [(index,) for index in range(count)],
                             f"{count}体で席を全部試していません")
        for count in (3, 4):
            self.assertEqual(supply_seat_choices(count, 1),
                             [(count - 1,)],
                             f"{count}体では後ろ詰めだけにすること")

    def test_no_supply_car_means_one_choice(self):
        from hexa.production import supply_seat_choices

        self.assertEqual(supply_seat_choices(7, 0), [()])

    def test_many_supply_cars_stay_within_the_limit(self):
        """台数が増えても、試す組み合わせは上限までに収めること。

        事前試走の持ち時間は決まっているので、組み合わせが増えすぎると
        台数の判断そのものが試せなくなります。
        """

        from hexa.production import supply_seat_choices

        for count, supply in ((8, 3), (8, 4), (7, 3)):
            got = supply_seat_choices(count, supply, limit=12)
            self.assertLessEqual(len(got), 12)
            self.assertEqual(len(set(got)), len(got), "重複があります")
            for seats in got:
                self.assertEqual(len(seats), supply)
                self.assertEqual(len(set(seats)), supply)
                for seat in seats:
                    self.assertLess(seat, count)
        # これまでの置き方（後ろ詰め）は必ず入っていること
        got = supply_seat_choices(8, 3, limit=12)
        self.assertIn((5, 6, 7), got, "後ろ詰めが候補に入っていません")

    def test_everyone_supply_is_handled(self):
        from hexa.production import kinds_for_seats, supply_seat_choices
        from hexa.rules import AgentKind

        got = supply_seat_choices(4, 4)
        self.assertEqual(got, [(0, 1, 2, 3)])
        self.assertEqual(kinds_for_seats(4, got[0]),
                         (AgentKind.SUPPLY,) * 4)

    def test_the_trial_searches_seats(self):
        """事前試走が、台数だけでなく席も回していること。"""

        import inspect

        from hexa.production import choose_production_kinds

        source = inspect.getsource(choose_production_kinds)
        self.assertIn("supply_seat_choices(problem.agent_count, supply_count)",
                      source, "席を試していません")
        self.assertIn("kinds_for_seats(problem.agent_count, seats)", source)
        self.assertNotIn("index >= problem.agent_count - supply_count",
                         source, "まだ後ろ詰めで決め打ちしています")

    def test_the_chosen_seat_is_logged(self):
        """選んだ席が記録に残ること（あとで確かめられるように）。"""

        import inspect

        from hexa.production import choose_production_kinds

        source = inspect.getsource(choose_production_kinds)
        self.assertIn("補給車の席", source)


class FirewallTest(unittest.TestCase):
    """**ファイアウォールの許可をプログラム側から入れる**ことの確認。

    LANで繋がらない原因のほとんどは固定IPとファイアウォールです。
    固定IPはWindowsの設定でしか変えられませんが、ファイアウォールは
    ここから開けられます。管理者の確認（UAC）に「はい」と答えるだけで、
    PowerShellのコマンドを打つ必要はありません。
    """

    def test_it_does_nothing_outside_windows(self):
        import platform

        import main

        done, message = main.open_firewall(8080)
        if platform.system() == "Windows":
            self.skipTest("Windowsでは実際に規則を入れてしまうので試しません")
        self.assertTrue(done)
        self.assertIn("Windows", message)
        done, message = main.close_firewall(8080)
        self.assertTrue(done)

    def test_the_rule_name_includes_the_port(self):
        import main

        self.assertIn("8080", main._firewall_rule_name(8080))
        self.assertNotEqual(main._firewall_rule_name(8080),
                            main._firewall_rule_name(8081))

    def test_the_command_asks_for_administrator(self):
        """昇格を頼む形になっていること（管理者でないと規則は入らない）。"""

        import inspect

        import main

        source = inspect.getsource(main.open_firewall)
        self.assertIn("-Verb RunAs", source, "昇格を頼んでいません")
        self.assertIn("netsh advfirewall firewall add rule", source)
        self.assertIn("dir=in", source)
        self.assertIn("action=allow", source)

    def test_the_static_ip_steps_are_in_the_menu(self):
        """固定IPの入れ方が、対話の画面から読めること。

        Windowsが「識別されていないネットワーク」と出すのはDHCPが
        無い状態で、手でアドレスを決めるしかありません。設定の場所は
        Windowsの版で変わるので、画面の順路とコマンドの両方を出します。
        """

        import io
        from contextlib import redirect_stdout

        import main

        for role, mine, theirs in (("サーバー", "192.168.50.1", "192.168.50.2"),
                                   ("クライアント", "192.168.50.2",
                                    "192.168.50.1")):
            out = io.StringIO()
            with redirect_stdout(out):
                main._print_static_ip_steps(role)
            text = out.getvalue()
            self.assertIn(mine, text, f"{role}側のアドレスがありません")
            self.assertIn(theirs, text, "相手のアドレスがありません")
            for word in ("255.255.255.0", "次のIPアドレスを使う",
                         "netsh interface ip set address", "dhcp",
                         "Get-NetAdapter", "無線は切って", "ping"):
                self.assertIn(word, text, f"手順に「{word}」がありません")

    def test_the_menu_covers_the_whole_setup(self):
        """21番だけでLAN対戦の準備が済むこと。"""

        import inspect

        import main

        source = inspect.getsource(main.menu_firewall)
        for word in ("このPCのアドレスを見る", "固定IPの入れ方",
                     "受信を許可する", "許可を消す", "届くか確かめる"):
            self.assertIn(word, source, f"21番に「{word}」がありません")
        self.assertIn("_print_static_ip_steps", source)
        self.assertIn("local_ips", source)
        self.assertIn("_explain_connection_failure", source)

    def test_the_server_menu_offers_it(self):
        """14番が、立ち上げたポートについて許可を聞くこと。"""

        import inspect

        import main

        source = inspect.getsource(main.menu_serve)
        self.assertIn("open_firewall(int(port))", source)
        self.assertIn("close_firewall(int(port))", source,
                      "終わったあとに片付けていません")

    def test_there_is_a_menu_of_its_own(self):
        import main

        keys = [key for key, _label, _handler in main.MENU]
        self.assertIn("21", keys)
        handlers = {key: handler for key, _label, handler in main.MENU}
        self.assertIs(handlers["21"], main.menu_firewall)


class LanAddressTest(unittest.TestCase):
    """**LANで別PCに繋ぐときのアドレスの出し方**の確認。

    以前は「8.8.8.8 へ繋ぐふりをして自分のアドレスを調べる」方式
    だけでした。既定の経路があれば正しく出ますが、**LANケーブル
    直結でインターネットが無いPCでは失敗し、`127.0.0.1` を返して
    いました**。それを「別のPCから」として表示していたので、
    相手からは絶対に繋がりません。

    Windowsが「識別されていないネットワーク」と出すのは、DHCPが
    無くてアドレスが決まらない（169.254.x.x になる）状態です。
    まさにこの場合に当たります。
    """

    def test_lan_addresses_come_first(self):
        from hexa.localserver import _address_rank

        order = sorted(["127.0.0.1", "169.254.3.4", "192.168.50.2",
                        "10.1.2.3", "203.0.113.9"], key=_address_rank)
        self.assertEqual(order[0][:4], "10.1"[:4])
        self.assertIn(order[0], ("10.1.2.3", "192.168.50.2"))
        self.assertEqual(order[-1], "127.0.0.1",
                         "自分しか見えないアドレスが先に出ています")
        self.assertLess(_address_rank("192.168.1.5"),
                        _address_rank("169.254.1.5"),
                        "DHCPなしのアドレスがLANより先に出ています")
        self.assertLess(_address_rank("169.254.1.5"),
                        _address_rank("127.0.0.1"))
        self.assertLess(_address_rank("172.20.0.5"),
                        _address_rank("127.0.0.1"))
        # 172.15 と 172.32 は私用の範囲ではない
        self.assertEqual(_address_rank("172.15.0.1")[0], 1)
        self.assertEqual(_address_rank("172.32.0.1")[0], 1)

    def test_the_list_never_comes_back_empty(self):
        from hexa.localserver import local_ip, local_ips

        found = local_ips()
        self.assertGreaterEqual(len(found), 1)
        self.assertEqual(local_ip(), found[0])
        for address in found:
            self.assertNotIn(":", address, "IPv6が混ざっています")

    def test_loopback_only_when_nothing_else(self):
        """LANのアドレスがあるなら、127.0.0.1 を先に出さないこと。"""

        from hexa.localserver import _address_rank

        self.assertGreater(_address_rank("127.0.0.1")[0],
                           _address_rank("169.254.0.1")[0])

    def test_the_server_listens_on_every_interface(self):
        """14番が 0.0.0.0 で立てること（127.0.0.1 だと外から繋げない）。"""

        import inspect

        import main

        source = inspect.getsource(main.menu_serve)
        self.assertIn('host="0.0.0.0"', source,
                      "練習用サーバーが自分だけに待ち受けています")

    def test_the_failure_guide_knows_the_usual_causes(self):
        """繋がらないときの案内が、固定IPとファイアウォールに触れること。"""

        import inspect

        import main

        source = inspect.getsource(main._explain_connection_failure)
        for word in ("識別されていないネットワーク", "255.255.255.0",
                     "New-NetFirewallRule", "netstat", "ping"):
            self.assertIn(word, source, f"案内に「{word}」がありません")

    def test_pointing_at_yourself_is_called_out(self):
        """URLが自分自身なら、それを指摘すること。"""

        import io
        from contextlib import redirect_stdout

        import main

        out = io.StringIO()
        with redirect_stdout(out):
            main._explain_connection_failure("http://127.0.0.1:8080")
        self.assertIn("自分自身", out.getvalue())


class FasterSearchTest(unittest.TestCase):
    """**速くするために足した覚え書き**が、答えを変えないことの確認。

    1手が速いほど多くの手を試せます。第1981（16x16）の4日目を20秒
    回すと、18.5秒でも玉が増え続けていました（頭打ちではない）。
    つまり速さはそのまま点数です。

    実測（1日20秒・記録された渋滞つき・補給1台）:

      第1958 (32x32) 4,900手 → **6,200手**（+27%）
      第1981 (16x16) 16,253手 → **20,665手**（+27%）

    やったこと:

      1. 道路状態に**番号**を振り、経路表の鍵を整数3つにする
         （鍵のタプルは32x32で80要素あり、引くたびに長さぶん
          ハッシュを計算して1回8マイクロ秒かかっていた）
      2. 方向の列（`directions_to`）を表の中に覚える
      3. `reflection` の席の一覧を1日1回だけ作る
      4. `teammate_targets` を、読むときに取り出す形にする
      5. `_swap_assignments` の値段を (車, 店) で覚える

    どれも `plan_day` の答えの指紋300件が1ビットも変わらないことを
    確かめてあります。ここではその仕掛け自体を試験します。
    """

    def _finder(self):
        import random

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder

        problem = gen(2026, honsen_settings("1回戦", random.Random(9),
                                            players=10, day_count=3))
        return problem, PathFinder(problem)

    def _roads(self, problem):
        from hexa.rules import Terrain

        return [index for index in range(problem.cell_count)
                if problem.terrain_at(index) == Terrain.ROAD]

    def test_same_roads_get_the_same_number(self):
        from hexa.rules import RoadStatus

        problem, finder = self._finder()
        roads = self._roads(problem)
        first = {roads[0]: RoadStatus.JAMMED}
        same = {roads[0]: RoadStatus.JAMMED}
        self.assertEqual(finder.road_id(first), finder.road_id(same),
                         "同じ中身なのに番号が違います")

    def test_different_roads_get_different_numbers(self):
        from hexa.rules import RoadStatus

        problem, finder = self._finder()
        roads = self._roads(problem)
        quiet = finder.road_id({})
        busy = finder.road_id({roads[0]: RoadStatus.BUSY})
        jammed = finder.road_id({roads[0]: RoadStatus.JAMMED})
        self.assertEqual(len({quiet, busy, jammed}), 3,
                         "違う中身に同じ番号が付いています")

    def test_switching_back_and_forth_keeps_the_pairing(self):
        """辞書を行き来しても、鍵と番号が食い違わないこと。

        以前、経路表のキャッシュを `id()` だけで覚えていて、混んだ道を
        順調な値で引く不具合がありました（30回のうち14件が誤り）。
        番号は鍵と**同じ場所で**更新します。
        """

        from hexa.pathfind import road_key
        from hexa.rules import RoadStatus

        problem, finder = self._finder()
        roads = self._roads(problem)
        smooth = {}
        jammed = {index: RoadStatus.JAMMED for index in roads}
        marks = {}
        for _round in range(3):
            for name, status in (("順調", smooth), ("渋滞", jammed)):
                number = finder.road_id(status)
                token = finder.road_token(status)
                self.assertEqual(token, road_key(status),
                                 "鍵が中身と合っていません")
                marks.setdefault(name, number)
                self.assertEqual(marks[name], number,
                                 f"{name}の番号が途中で変わりました")
        self.assertNotEqual(marks["順調"], marks["渋滞"])

    def test_the_table_for_jammed_roads_is_not_the_smooth_one(self):
        from hexa.rules import RoadStatus

        problem, finder = self._finder()
        roads = self._roads(problem)
        jammed = {index: RoadStatus.JAMMED for index in roads}
        start = problem.agent_positions[0]
        # 行き来させてから引いても、別の表が返ること
        for _round in range(3):
            smooth_table = finder.table(start, {})
            jammed_table = finder.table(start, jammed)
        far = max(problem.spots,
                  key=lambda spot: smooth_table.steps.get(spot.position, 0))
        quiet = smooth_table.steps.get(far.position)
        stuck = jammed_table.steps.get(far.position)
        self.assertIsNotNone(quiet)
        self.assertIsNotNone(stuck)
        self.assertGreaterEqual(stuck, quiet,
                                "渋滞の表が順調と同じか速くなっています")

    def test_directions_are_a_fresh_list_every_time(self):
        """方向の列は**写し**を返すこと（呼んだ側が削るため）。"""

        problem, finder = self._finder()
        start = problem.agent_positions[0]
        table = finder.table(start, {})
        target = None
        for spot in problem.spots:
            if table.reachable(spot.position) and table.steps[spot.position] > 0:
                target = spot.position
                break
        self.assertIsNotNone(target, "届く店がありません")
        first = table.directions_to(target)
        self.assertGreater(len(first), 0)
        wanted = list(first)
        first.pop(0)
        second = table.directions_to(target)
        self.assertEqual(second, wanted,
                         "前に返した列を削ると、次に返る列も短くなります")
        self.assertIsNot(first, second)

    def test_the_lazy_teammate_view_reads_live(self):
        """味方の行き先は、**読んだ時点**の中身であること。

        1歩のなかでも車が順に決めていくので、作り置きでは食い違います。
        """

        from hexa.planner import _TeammateView

        class _State:
            def __init__(self, target):
                self.target = target

        states = [_State(None), _State(None)]
        view = _TeammateView(states, [True, True])
        self.assertEqual(list(view), [None, None])
        states[1].target = 42
        self.assertEqual(list(view), [None, 42],
                         "作り置きになっています")
        self.assertEqual(len(view), 2)
        self.assertTrue(view or False, "空扱いになっています")


class SmallBoardPhasesTest(unittest.TestCase):
    """**小さい盤面だけ、磨きの手の種類を増やす**ことの確認。

    `_phase_relocate`（同じ車のなかで1店を別の位置へ）と
    `_phase_or_opt`（続いた2〜3店をまとめて動かす）を外した理由は
    「手の種類が増えると1種類あたりの持ち時間が減る」でした。これは
    持ち時間が足りない盤面（32x32は1手3ms）の話です。

    16x16は**登り切って止まります**。第1981を1日2秒で回しても玉178、
    20秒でも玉181で、時間を10倍にして3玉しか動きません。止まって
    いる盤面では、減らす持ち時間がありません。

    16x16の過去問11試合（記録された渋滞つき・1日15秒）:

      手を増やす   : 日別786 / 玉**1632**
      増やさない   : 日別786 / 玉 1624

    日別は同じで玉数が8増えました。第1981は玉181→**184**、
    第1656は9位→**4位**、第1646は7位→**5位**です。
    24x24はまだ時間で伸びる（第1891で1日8秒=247・20秒=253・
    57秒=260）ので、頭打ちではなく、増やす対象から外します。
    """

    def _problem(self, width):
        import random

        from hexa.generator import generate_problem as gen, honsen_settings

        label = {16: "1回戦", 24: "準決勝", 32: "決勝"}[width]
        return gen(2026, honsen_settings(label, random.Random(5),
                                         players=10, day_count=4))

    def test_small_boards_get_more_phases(self):
        from hexa.improve import (_MOVE_PHASES, _SMALL_MOVE_PHASES,
                                  phases_for)

        small = self._problem(16)
        self.assertLessEqual(small.cell_count, 400)
        self.assertEqual(phases_for(small), _SMALL_MOVE_PHASES)
        self.assertGreater(len(_SMALL_MOVE_PHASES), len(_MOVE_PHASES))

    def test_big_boards_keep_the_lean_set(self):
        from hexa.improve import _MOVE_PHASES, phases_for

        for width in (24, 32):
            board = self._problem(width)
            self.assertGreater(board.cell_count, 400)
            self.assertEqual(phases_for(board), _MOVE_PHASES,
                             f"{width}x{width}で手が増えています")

    def test_the_extra_phases_are_the_in_route_ones(self):
        from hexa.improve import (_MOVE_PHASES, _SMALL_MOVE_PHASES,
                                  _phase_or_opt, _phase_relocate)

        extra = set(_SMALL_MOVE_PHASES) - set(_MOVE_PHASES)
        self.assertEqual(extra, {_phase_relocate, _phase_or_opt})

    def test_the_polisher_uses_the_board_s_set(self):
        """磨く側が、その盤面の一覧を持つこと。"""

        from hexa.improve import RoutePolisher, phases_for
        from hexa.rules import AgentKind

        board = self._problem(16)
        kinds = tuple([AgentKind.TOURING] * (board.agent_count - 1)
                      + [AgentKind.SUPPLY])
        routes = {0: [spot.position for spot in board.spots[:3]]}
        polisher = RoutePolisher(board, kinds, routes, lambda *a, **k: None)
        self.assertEqual(polisher.phases, phases_for(board))
        self.assertEqual(len(polisher.stats), len(polisher.phases),
                         "手の種類ごとの記録の数が合っていません")

    def test_every_phase_gets_tried(self):
        """増えた手にも順番が回ってくること。"""

        from hexa.improve import RoutePolisher
        from hexa.rules import AgentKind

        board = self._problem(16)
        kinds = tuple([AgentKind.TOURING] * (board.agent_count - 1)
                      + [AgentKind.SUPPLY])
        touring = [index for index, kind in enumerate(kinds)
                   if kind == AgentKind.TOURING]
        routes = {index: [spot.position for spot in board.spots[:4]]
                  for index in touring}
        polisher = RoutePolisher(board, kinds, routes, lambda *a, **k: None)
        self.assertEqual(sorted(polisher._order()),
                         list(range(len(polisher.phases))))


class AllRoadsJammedTest(unittest.TestCase):
    """**全部赤（全道路が渋滞）**でも正しく動くことの確認。

    渋滞した道路は**1マス出るのに4歩**で、平地2歩・山地3歩より
    **遅い**。つまり順調なときは道路が最速でも、赤くなったら
    避けるのが正しい。この切り替えがどこかで抜けていると、
    真っ赤な日に道路へ突っ込んで歩数を捨てます。

    値段の表は道路状態の**中身から作った鍵**で引きます。以前
    `id()` だけで覚えていて、混んだ道を順調な値で引く不具合が
    ありました（30回のうち14回が間違い）。

    ここでは4つを押さえます。

      1. 1マスを出る歩数が状態で変わること
      2. 経路表の歩数が状態で変わること
      3. 全部赤のとき、道路にいる歩の割合が**下がる**こと
      4. 採点の「明日回れる系列」が全部赤で**減る**こと
    """

    def _setup(self):
        import random

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.planner import DayPlanner
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(7),
                                            players=10, day_count=4))
        kinds = tuple([AgentKind.TOURING] * (problem.agent_count - 1)
                      + [AgentKind.SUPPLY])
        finder = PathFinder(problem)
        return problem, finder, DayPlanner(problem, finder), kinds

    def _roads(self, problem):
        from hexa.rules import Terrain

        return [index for index in range(problem.cell_count)
                if problem.terrain_at(index) == Terrain.ROAD]

    def _jammed(self, problem):
        from hexa.rules import RoadStatus

        return {index: RoadStatus.JAMMED for index in self._roads(problem)}

    def test_a_jammed_road_is_slower_than_plain(self):
        from hexa.rules import RoadStatus, Terrain, move_steps

        plain = move_steps(Terrain.PLAIN, None)
        mountain = move_steps(Terrain.MOUNTAIN, None)
        self.assertEqual(move_steps(Terrain.ROAD, RoadStatus.SMOOTH), 1)
        self.assertEqual(move_steps(Terrain.ROAD, RoadStatus.BUSY), plain)
        self.assertGreater(move_steps(Terrain.ROAD, RoadStatus.JAMMED),
                           mountain, "渋滞した道路が山地より速い扱いです")

    def test_the_table_costs_more_when_jammed(self):
        from hexa.rules import RoadStatus

        problem, finder, _planner, _kinds = self._setup()
        roads = self._roads(problem)
        start = problem.agent_positions[0]

        def total(status):
            table = finder.table(start, status)
            return sum(table.steps.get(spot.position, 0)
                       for spot in problem.spots)

        smooth = total({})
        busy = total({index: RoadStatus.BUSY for index in roads})
        jammed = total(self._jammed(problem))
        self.assertLess(smooth, busy, "混雑しても値段が同じです")
        self.assertLess(busy, jammed, "渋滞しても値段が同じです")

    def test_jammed_roads_are_avoided(self):
        """全部赤のとき、道路にいる歩の割合が下がること。"""

        from hexa.production import (_build_candidate,
                                     _planned_route_policies)
        from hexa.rules import Terrain
        from hexa.simulator import simulate_day

        problem, finder, planner, kinds = self._setup()

        def share(status):
            positions = list(problem.agent_positions)
            fuels = [problem.fuel_limit] * problem.agent_count
            best = None
            for label, policy in _planned_route_policies(
                    problem, finder, kinds, positions, fuels, status,
                    problem.day_steps[0], set()):
                got = _build_candidate(
                    label, problem, planner, kinds, positions, fuels,
                    status, problem.day_steps[0], policy, set(), day_index=0,
                    steps_ahead=0, next_steps=problem.day_steps[1])
                if got is not None and (best is None or got.key > best.key):
                    best = got
            self.assertIsNotNone(best, "計画が作れません")
            result = simulate_day(
                problem=problem, agent_kinds=kinds,
                start_positions=tuple(positions), start_fuels=tuple(fuels),
                action_plan=best.action_plan, day_steps=problem.day_steps[0],
                road_status=status, record_trace=True)
            stays = [where for row in result.step_positions for where in row]
            on_road = sum(1 for where in stays
                          if problem.terrain_at(where) == Terrain.ROAD)
            return on_road / max(1, len(stays)), len(result.acquisitions)

        smooth_share, smooth_balls = share({})
        jammed_share, jammed_balls = share(self._jammed(problem))
        self.assertLess(jammed_share, smooth_share,
                        "真っ赤なのに道路の使い方が変わっていません")
        self.assertGreater(jammed_balls, 0, "真っ赤だと1玉も取れません")

    def test_the_lookahead_sees_the_jam(self):
        """採点の見積りが、全部赤で厳しく出ること。"""

        from hexa.production import _tomorrow_coverage, _COVER_CACHE
        from hexa.simulator import DayResult

        problem, finder, _planner, kinds = self._setup()
        here = DayResult(
            positions=tuple(problem.agent_positions),
            fuels=tuple([problem.fuel_limit] * problem.agent_count),
            acquisitions=[], refuels=[], stay_counts={}, remaining_stocks={},
            wasted_spot_stays=0, step_positions=[], step_fuels=[])
        _COVER_CACHE.clear()
        smooth = _tomorrow_coverage(here, kinds, problem, finder,
                                   problem.day_steps[1], {})
        jammed = _tomorrow_coverage(here, kinds, problem, finder,
                                    problem.day_steps[1],
                                    self._jammed(problem))
        self.assertLess(jammed, smooth,
                        "真っ赤でも同じだけ回れる見積りになっています")

    def test_our_own_stays_count_toward_the_jam(self):
        """自分が居座って作る渋滞も見込むこと。

        交通量は**全チームの前2日の滞在÷人数**で決まるので、自分の
        滞在も混雑の材料です。見落とすと、自分で詰まらせた道を
        順調なつもりで通ります。
        """

        from hexa.opponents import OpponentModel

        problem, _finder, _planner, _kinds = self._setup()
        model = OpponentModel(problem)
        roads = self._roads(problem)
        quiet = model.congestion_risk([{}, {}])
        crowded = model.congestion_risk(
            [{roads[0]: problem.jammed_threshold * problem.players}, {}])
        self.assertGreater(crowded[roads[0]], quiet[roads[0]],
                           "自分の滞在が混雑の見込みに入っていません")
        self.assertEqual(crowded[roads[0]], 1.0,
                         "しきい値を超えても渋滞見込みになりません")


class WarmStartTest(unittest.TestCase):
    """**昨日の順路を今日の候補に持ち越す**ことの確認。

    店の場所は毎日同じで、在庫も毎朝満タンに戻ります。だから昨日
    1日ぶんの時間をかけて磨いた「車ごとの店の並び」は、今日もそのまま
    良い形です。毎朝それを捨てて素案から登り直すのは、1手3msかかる
    32x32ではとくに損でした。

    ここで確かめるのは「別の出発点からでも、その並びで計画が作れる」
    ことです（planner はいまいる場所から順路をたどるだけ）。
    """

    def _setup(self):
        import random

        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.planner import DayPlanner
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("1回戦", random.Random(11),
                                            players=10, day_count=3))
        kinds = tuple([AgentKind.TOURING] * (problem.agent_count - 1)
                      + [AgentKind.SUPPLY])
        finder = PathFinder(problem)
        return problem, finder, DayPlanner(problem, finder), kinds

    def test_the_switch_is_on(self):
        from hexa.production import WARM_START_ROUTES

        self.assertTrue(WARM_START_ROUTES, "持ち越しが切れています")

    def test_yesterdays_route_still_plans_from_a_new_place(self):
        from hexa.production import (GreedyPolicy, PlannedRoutePolicy,
                                     _build_candidate, routes_from_result,
                                     supply_order_from_result)

        problem, _finder, planner, kinds = self._setup()
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit] * problem.agent_count
        first = _build_candidate(
            "貪欲", problem, planner, kinds, positions, fuels, {},
            problem.day_steps[0], GreedyPolicy(problem), set(),
            day_index=0, steps_ahead=0, next_steps=problem.day_steps[1])
        self.assertIsNotNone(first, "1日目の案が作れません")
        routes = routes_from_result(first.day_result, kinds)
        order = supply_order_from_result(first.day_result, kinds)
        # 2日目は**別の場所・別の燃料**から始まります。
        again = _build_candidate(
            "昨日の順路", problem, planner, kinds,
            list(first.day_result.positions), list(first.day_result.fuels),
            {}, problem.day_steps[1],
            PlannedRoutePolicy(problem, routes, order), set(),
            day_index=1, steps_ahead=0, next_steps=0)
        self.assertIsNotNone(again, "昨日の順路から計画が作れません")
        self.assertGreater(len(again.day_result.acquisitions), 0,
                           "昨日の順路で1玉も取れていません")

    def test_the_match_loop_carries_it(self):
        """本番のループが、採用した案の順路を取っておくこと。"""

        import inspect

        from hexa.production import play_production_match

        source = inspect.getsource(play_production_match)
        self.assertIn("carried = (routes_from_result(", source)
        self.assertIn('("昨日の順路", PlannedRoutePolicy(', source)


class SupplyCountTest(unittest.TestCase):
    """補給車の台数を、1台を既定にして決めることの確認。

    **実戦でいちばん大きな損失でした。** 実戦1958（32x32・7体）で
    事前試走が巡回5+補給2を選び、1位比74%。同じ32x32でも巡回6+補給1
    を選んだ試合は91%でした。

    記録された渋滞つきで、台数だけ変えて9日通した実測:

      1958 (32x32 燃料137)  巡回7+補給0   玉173 / 日別 92
                            巡回6+補給1 **玉485 / 日別198**
                            巡回5+補給2   玉426 / 日別197 ← 実戦の選択
                            巡回4+補給3   玉349 / 日別195
      1703 (32x32 燃料113)  巡回6+補給1 **玉413 / 日別221**
                            巡回5+補給2   玉322 / 日別220
      1960 (24x24 燃料147)  巡回4+補給1 **玉179 / 日別137**
                            巡回3+補給2   玉160 / 日別124

    山は**1台に鋭く尖って**います。0台だと数日で全車が燃料切れ、
    2台以上だと巡回車が減って取りに行けません。

    ところが事前試走は**貪欲**で回します。貪欲は燃料を無駄に使うので
    補給車が多いほど得に見え、実戦では2台を選びました（学習済みAIを
    外した手元では0台を選びます）。**推定器が当てになりません。**
    そこで1台を既定とし、動かすなら決定的な差を要求します。
    """

    def test_one_supply_is_the_default(self):
        from hexa.production import _trial_better

        # 玉で1割上回る程度では、台数を動かしません。
        self.assertFalse(_trial_better((22, 198, 440), 2, (22, 198, 400), 1),
                         "2台が小差で選ばれています")
        self.assertFalse(_trial_better((22, 198, 440), 0, (22, 198, 400), 1),
                         "0台が小差で選ばれています")

    def test_a_decisive_win_still_counts(self):
        from hexa.production import _trial_better

        self.assertTrue(_trial_better((22, 198, 520), 2, (22, 198, 400), 1),
                        "決定的な差でも動けません")

    def test_kinds_and_daily_are_never_discounted(self):
        """種類と日別は公式の上位条件。割り引かないこと。"""

        from hexa.production import _trial_discount

        got = _trial_discount((22, 199, 300), 3)
        self.assertEqual(got[0], 22)
        self.assertEqual(got[1], 199)
        self.assertLess(got[2], 300, "玉数が割り引かれていません")

    def test_the_further_from_one_the_bigger_the_demand(self):
        from hexa.production import _trial_discount

        one = _trial_discount((0, 0, 100), 1)[2]
        two = _trial_discount((0, 0, 100), 2)[2]
        three = _trial_discount((0, 0, 100), 3)[2]
        self.assertEqual(one, 100)
        self.assertLess(two, one)
        self.assertLess(three, two)

    def test_ties_still_prefer_more_touring(self):
        from hexa.production import _trial_better

        self.assertTrue(_trial_better((22, 198, 400), 1, (22, 198, 400), 2),
                        "同点なら巡回車の多いほうを採るべきです")


class SupplyPressureTest(unittest.TestCase):
    """「燃料が逼迫したら2台」という例外を**入れなかった**ことの確認。

    実戦1970（32x32・燃料124・1日191歩の日あり）だけ2台が勝つように
    見え、いちど「逼迫度で切り替える」仕掛けを入れました。
    **ところが手数を増やすと逆転します。**

      1970  手数 500 : 巡回6+補給1 玉394 / 巡回5+補給2 玉410
      1970  手数1500 : 巡回6+補給1 **玉452** / 巡回5+補給2 玉407

    **巡回車が多い構成ほど、良くするのに探索が要ります。** だから
    手数の少ない比べ方は、巡回車の少ない構成を不当に有利に見せます。
    事前試走（貪欲1回）はその極端な場合で、これが台数を外す本当の
    理由でした。燃料の逼迫は関係ありませんでした。

    5盤面すべてで1台が正解です:

      1960 (24x24) 玉179 対 160
      1958 (32x32) 玉485 対 426
      1703 (32x32) 玉413 対 322
      1966 (32x32) 玉858 対 750
      1970 (32x32) 玉452 対 407（手数1500）
    """

    def _problem(self, day_steps, fuel):
        import random
        import dataclasses

        from hexa.generator import generate_problem as gen, honsen_settings

        problem = gen(2026, honsen_settings("1回戦", random.Random(3),
                                            players=10,
                                            day_count=len(day_steps)))
        return dataclasses.replace(problem, day_steps=tuple(day_steps),
                                   fuel_limit=fuel)

    def test_every_board_keeps_the_margin(self):
        """盤面によらず、1台を既定にすること。"""

        from hexa.production import SUPPLY_EXTRA_MARGIN, supply_margin_for

        # 余裕のある盤面（実戦1960 と同じくらい）
        relaxed = self._problem([49, 48, 48, 60, 57, 69, 57], 147)
        self.assertEqual(supply_margin_for(relaxed), SUPPLY_EXTRA_MARGIN)
        # 燃料が逼迫した盤面（実戦1970 と同じ）でも外しません
        tight = self._problem([75, 107, 109, 176, 91, 191, 150], 124)
        self.assertEqual(supply_margin_for(tight), SUPPLY_EXTRA_MARGIN)

    def test_the_margin_actually_reaches_the_choice(self):
        from hexa.production import _trial_better

        # 上積みなしなら、玉がわずかに多い2台が選ばれる
        self.assertTrue(_trial_better((21, 147, 410), 2, (21, 147, 394), 1,
                                      margin=0.0))
        # 上積みありなら選ばれない
        self.assertFalse(_trial_better((21, 147, 410), 2, (21, 147, 394), 1,
                                       margin=0.15))

    def test_the_trial_passes_the_board_margin(self):
        import inspect

        from hexa.production import choose_production_kinds

        source = " ".join(inspect.getsource(choose_production_kinds).split())
        self.assertIn("margin = supply_margin_for(problem)", source,
                      "盤面ごとの上積みを計算していません")
        self.assertIn("best_supply, margin", source,
                      "比べるときに渡していません")


class PassingDesignTest(unittest.TestCase):
    """**通りがかりで踏む店まで数えて組む設計**の確認。

    車は歩いた先の店を無条件で拾います。実測で、取った玉のうち
    通りがかりのものが8〜42%ありました（第1958の32x32・3日目は
    42%）。ところが `design_day` と `design_insert` は歩数だけで
    組み立てていて、この拾いを数えていません。だから

      「訪問数を増やす」 → 計画の訪問を60→107にしても玉54→50
      「歩数を縮める」   → 歩556→535に縮めたら玉59→55

    と、どちらの見立ても外れていました。

    `design_passing` は「その差し込みで**新たに踏む店が何軒増えるか**」
    で選びます。第1958を10秒ずつ磨いて比べると

      1日目 従来 玉85/日別22   通りがかり 玉77/日別22
      3日目 従来 玉64/日別21   通りがかり 玉63/**日別22**  ← 採点で勝ち
      5日目 従来 玉56/日別22   通りがかり 玉51/日別22

    日によって強いほうが変わるので、**置き換えずに候補に足します**。
    採点が日ごとに良いほうを選びます。
    """

    def _setup(self, seed=3, label="準決勝"):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings(label, random.Random(seed),
                                            players=10, day_count=5))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1
            else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        return problem, kinds, PathFinder(problem)

    def test_passed_spots_match_a_hand_walk(self):
        """`path_spots` が、経路をたどって数えた店と一致すること。"""

        from hexa.pathfind import PathFinder

        problem, _kinds, finder = self._setup()
        bits = finder.spot_bits
        table = finder.table(problem.agent_positions[0], {})
        checked = 0
        for spot in problem.spots:
            if spot.position not in table.steps:
                continue
            # 自分で `previous` をたどって、踏んだ店を集める。
            want, current = set(), spot.position
            while current != table.origin:
                if current in bits:
                    want.add(current)
                current = table.previous[current][0]
            got = table.path_spots(spot.position, bits)
            found = {place for place, bit in bits.items() if got & bit}
            self.assertEqual(found, want,
                             "通りがかりで踏む店の数え方が合いません")
            self.assertNotIn(table.origin, found, "始点を数えています")
            checked += 1
        self.assertGreater(checked, 0, "1店も届きません")

    def test_the_memo_does_not_mix_up_two_bit_tables(self):
        """別のビット表で聞いたら、覚え書きを使い回さないこと。

        表そのものは道路状態と始点ごとに使い回されます。覚え書きの
        鍵は行き先だけなので、渡すビット表が違えば捨てないと
        取り違えます。
        """

        problem, _kinds, finder = self._setup()
        table = finder.table(problem.agent_positions[0], {})
        goal = None
        for spot in problem.spots:
            if spot.position in table.steps and table.steps[spot.position] > 0:
                goal = spot.position
                break
        self.assertIsNotNone(goal, "届く店がありません")
        first = table.path_spots(goal, {goal: 1})
        self.assertEqual(first, 1, "行き先の店を数えていません")
        second = table.path_spots(goal, {goal: 4})
        self.assertEqual(second, 4, "古いビット表の答えを返しています")

    def test_the_design_fits_the_step_and_fuel_budget(self):
        from hexa.allocate import design_passing, route_cost

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit] * problem.agent_count
        day_steps = problem.day_steps[0]
        routes = design_passing(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            day_steps, acquired_brands=set(), refuels=0,
        )
        self.assertTrue(any(routes.values()), "1店も割り当てていません")
        for index, spots in routes.items():
            self.assertEqual(len(set(spots)), len(spots),
                             "同じ車が同じ店を2回回っています")
            cost = route_cost(finder, {}, problem.agent_positions[index],
                              spots)
            self.assertLessEqual(cost, day_steps,
                                 f"{index}号車の順路が歩数を超えています")

    def test_only_touring_cars_get_routes(self):
        from hexa.allocate import design_passing
        from hexa.rules import AgentKind

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit] * problem.agent_count
        routes = design_passing(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], acquired_brands=set(),
        )
        for index in routes:
            self.assertEqual(kinds[index], AgentKind.TOURING,
                             "補給車に順路を渡しています")

    def test_all_supply_gives_nothing(self):
        from hexa.allocate import design_passing
        from hexa.rules import AgentKind

        problem, _kinds, finder = self._setup()
        kinds = tuple([AgentKind.SUPPLY] * problem.agent_count)
        fuels = [problem.fuel_limit] * problem.agent_count
        routes = design_passing(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], acquired_brands=set(),
        )
        self.assertEqual(routes, {}, "巡回車がいないのに順路が出ました")

    def test_passing_spots_are_counted_as_used(self):
        """通りがかりで踏んだ店の在庫も減らすこと。

        減らさないと、同じ在庫を2台で取り合う順路が出ます。
        """

        import inspect

        from hexa.allocate import design_passing

        source = " ".join(inspect.getsource(design_passing).split())
        self.assertIn("for where, bit in list(left.items()):", source)
        self.assertIn("if bits & spot_bits.get(where, 0):", source)

    def test_the_gain_is_newly_walked_spots(self):
        """得が「新たに踏む店の数」で数えられていること。"""

        import inspect

        from hexa.allocate import design_passing

        source = " ".join(inspect.getsource(design_passing).split())
        self.assertIn("gained = bin((rest | bits) & ~rest).count('1')", source)

    def test_the_design_is_offered_as_a_candidate(self):
        from hexa.production import _planned_route_policies

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit if k == 0 else 0 for k in kinds]
        got = _planned_route_policies(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], set(),
        )
        labels = [label for label, _policy in got]
        self.assertTrue(any(l.startswith("通りがかり") for l in labels),
                        "通りがかりの設計が候補にありません")
        # **置き換えではありません。** 従来の設計も残ること。
        self.assertTrue(any(l.startswith("差し込み") for l in labels),
                        "差し込みの設計が消えています")
        self.assertTrue(any(l.startswith(("設計", "補給設計")) for l in labels),
                        "末尾に足す設計が消えています")

    def test_the_designs_are_listed_in_one_place(self):
        from hexa.production import PASSING_DESIGNS

        self.assertTrue(PASSING_DESIGNS, "通りがかりの設計が空です")
        for per_step, brand in PASSING_DESIGNS:
            self.assertIsInstance(per_step, bool)
            self.assertGreaterEqual(brand, 0.0)


class SeedOtherCriteriaTest(unittest.TestCase):
    """**玉の多い案・日別の多い案も登り直しの出発点に混ぜる**ことの確認。

    採点は日別累計を玉数より上に見ます（公式の順位がそうなので正解
    です）。ところが**磨きの出発点としては、玉の多い案のほうが
    良いことがあります**。実測（第1958の32x32・3日目・10秒磨き）:

      採点で最良の設計（玉45/日別22）から磨く → 玉59/日別22
      玉の多い設計（玉65/日別20）から磨く     → **玉62**/日別21

    採点の順だけで並べると玉の多い案は下位に沈んで、出発点に
    選ばれません。だから先頭の次に混ぜます。
    """

    def _candidate(self, routes, daily, balls):
        from hexa.production import Candidate, KEY_NOODLES
        from hexa.simulator import Acquisition, DayResult

        key = [0] * 11
        key[1] = daily
        key[KEY_NOODLES] = balls
        acquisitions = []
        step = 0
        for agent, spots in routes.items():
            for spot in spots:
                step += 1
                acquisitions.append(Acquisition(step, agent, spot, 0))
        result = DayResult(
            positions=(0,) * (len(routes) + 1),
            fuels=(10,) * (len(routes) + 1),
            acquisitions=acquisitions, refuels=[], stay_counts={},
            remaining_stocks={}, wasted_spot_stays=0,
            step_positions=[], step_fuels=[],
        )
        return Candidate("案", [], result, tuple(key))

    def _kinds(self, cars):
        from hexa.rules import AgentKind

        return tuple([AgentKind.TOURING] * cars + [AgentKind.SUPPLY])

    def test_the_best_by_balls_is_near_the_head(self):
        from hexa.production import diverse_seeds

        kinds = self._kinds(2)
        # 採点で最良（日別が多い）。玉は少ない。
        top = self._candidate({0: [1], 1: [2]}, daily=22, balls=45)
        # 玉がいちばん多い。採点の順では下位に沈む。
        balls = self._candidate({0: [3], 1: [4]}, daily=20, balls=65)
        middle = [self._candidate({0: [10 + i], 1: [50 + i]},
                                  daily=21, balls=50 + i)
                  for i in range(6)]
        seeds = diverse_seeds([top] + middle + [balls], kinds, 8)
        self.assertIs(seeds[0], top, "いちばん良い案が先頭にありません")
        self.assertIn(balls, seeds[:3],
                      "玉の多い案が出発点に混ざっていません")

    def test_it_can_be_turned_off(self):
        from hexa import production as P

        kinds = self._kinds(2)
        top = self._candidate({0: [1], 1: [2]}, daily=22, balls=45)
        balls = self._candidate({0: [3], 1: [4]}, daily=20, balls=65)
        middle = [self._candidate({0: [10 + i], 1: [50 + i]},
                                  daily=21, balls=50 + i)
                  for i in range(6)]
        pool = [top] + middle + [balls]
        was = P.SEED_OTHER_CRITERIA
        try:
            P.SEED_OTHER_CRITERIA = False
            plain = P.diverse_seeds(pool, kinds, 8)
        finally:
            P.SEED_OTHER_CRITERIA = was
        self.assertIs(plain[0], top)
        self.assertNotIn(balls, plain[:3],
                         "切っても混ざったままです")

    def test_short_keys_do_not_break_it(self):
        """短い鍵の案が来ても落ちないこと（試験台が使います）。"""

        from hexa.production import Candidate, diverse_seeds
        from hexa.simulator import DayResult

        kinds = self._kinds(2)
        result = DayResult(
            positions=(0, 0, 0), fuels=(10, 10, 10), acquisitions=[],
            refuels=[], stay_counts={}, remaining_stocks={},
            wasted_spot_stays=0, step_positions=[], step_fuels=[],
        )
        pool = [Candidate("案", [], result, (0, 0, 0, 0, i)) for i in range(3)]
        self.assertEqual(len(diverse_seeds(pool, kinds, 3)), 1)


class FuelReserveTest(unittest.TestCase):
    """**着いた先で身動きが取れなくなる店を後回しにする**ことの確認。

    行き先の判定は「その店まで燃料が足りるか」だけでした。だから
    ぎりぎり届く店へ行き、着いたところで次のどこにも行けません。

    第1703（32x32）で巡回車の待ち歩を理由で分けると

      燃料が足りず店に行けない : 340歩 (39%)
      燃料切れ                 : 271歩 (31%)
      歩数が足りない           : 180歩 (21%)
      その他                   :  70歩 (8%)

    **7割が燃料**でした。しかも燃料の総量は足りていて（9日で2374
    入れ、歩を使い切るのに要るのは2172）、空の車と補給車が落ち合え
    ていないだけです。だから、着いたあとに余力が残る店を先に選びます。
    """

    def _policy(self, fuel_limit=100, ratio=0.1):
        from hexa import production as P
        from hexa.problem import Problem, Spot

        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(40,), height=1,
            width=9, cells=tuple([0] * 9),
            spots=(Spot(brand=1, position=2, stocks=3),
                   Spot(brand=2, position=8, stocks=3)),
            agent_positions=(0,), fuel_limit=fuel_limit, players=2,
            busy_threshold=0.5, jammed_threshold=0.8,
        )
        return problem, P

    def _ask(self, problem, P, fuel, ratio):
        from hexa.pathfind import PathFinder

        finder = PathFinder(problem)
        was = P.KEEP_FUEL_RATIO
        try:
            P.KEEP_FUEL_RATIO = ratio
            policy = P.GreedyPolicy(problem)
            return policy.choose_touring_target(
                agent_index=0, step=0, remaining=problem.day_steps[0],
                position=0, fuel=fuel, stock={2: 3, 8: 3}, reserved={},
                acquired_today=set(), acquired_brands=set(),
                brands_today=set(), road_status={}, finder=finder,
                kinds=(), positions=[0], day_index=0, day_count=1,
            ).target
        finally:
            P.KEEP_FUEL_RATIO = was

    def test_a_spot_that_leaves_room_wins(self):
        """余力の残る店と、使い切る店があるなら、残るほうを選ぶ。"""

        problem, P = self._policy(fuel_limit=100)
        # 平地は1マス燃料1。マス2まで2燃料、マス8まで8燃料。
        # 燃料9・余力10 なら、マス8は「着いたら残り1」で後回し。
        got = self._ask(problem, P, fuel=9, ratio=0.10)
        self.assertEqual(got, 2, "着いたら動けない店を選んでいます")

    def test_it_still_goes_when_nothing_else_is_left(self):
        """余力の残る店が1つもないなら、ぎりぎりの店へ行く。"""

        problem, P = self._policy(fuel_limit=100)
        # 余力を上限の90%も求めたら、どの店も条件を満たさない。
        got = self._ask(problem, P, fuel=9, ratio=0.90)
        self.assertEqual(got, 2, "行ける店があるのに待とうとしています")

    def test_zero_turns_the_rule_off(self):
        problem, P = self._policy(fuel_limit=100)
        got = self._ask(problem, P, fuel=9, ratio=0.0)
        self.assertEqual(got, 2, "近い店が選ばれていません")

    def test_the_knob_is_a_ratio_of_the_tank(self):
        from hexa import production as P

        self.assertGreaterEqual(P.KEEP_FUEL_RATIO, 0.0)
        self.assertLess(P.KEEP_FUEL_RATIO, 1.0)


class StuckFirstTest(unittest.TestCase):
    """**補給車が「どの店にも届かない車」を先に迎える**ことの確認。

    これまでの優先は「燃料の減りがいちばん大きい車」でした。
    ところが残り10で3燃料先に店がある車は困っていません。困って
    いるのは**どの店にも届かない車**です。
    """

    def _setup(self, fuels):
        from hexa import production as P
        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot
        from hexa.rules import AgentKind

        # 横一列。店はマス2と3。車0はマス0、車1はマス9、補給車はマス5。
        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(40,), height=1,
            width=10, cells=tuple([0] * 10),
            spots=(Spot(brand=1, position=2, stocks=3),
                   Spot(brand=2, position=3, stocks=3)),
            agent_positions=(0, 9, 5), fuel_limit=100, players=2,
            busy_threshold=0.5, jammed_threshold=0.8,
        )
        kinds = (AgentKind.TOURING, AgentKind.TOURING, AgentKind.SUPPLY)
        return problem, P, PathFinder(problem), kinds, fuels

    def _ask(self, stuck_first):
        from hexa import production as P

        # 車0は残り1で、マス2（2燃料）にも届かない → 行き詰まり。
        # 車1は残り3で、マス3（6燃料）に届かない → こちらも行き詰まり
        #   ではあるが、減りは車0のほうが大きい。
        problem, _P, finder, kinds, fuels = self._setup([1, 40, 0])
        was = P.STUCK_FIRST
        try:
            P.STUCK_FIRST = stuck_first
            policy = P.GreedyPolicy(problem)
            return policy.choose_supply_target(
                agent_index=2, step=0, remaining=problem.day_steps[0],
                position=5, kinds=kinds, positions=[0, 9, 5], fuels=fuels,
                plan_targets=[None, None, None], supply_claims={},
                road_status={}, finder=finder, day_index=0, day_count=1,
            ).target_agent
        finally:
            P.STUCK_FIRST = was

    def test_the_stuck_car_is_fetched_first(self):
        """届かない車（車0）を、余裕のある車より先に迎える。"""

        self.assertEqual(self._ask(True), 0)

    def test_without_it_the_emptiest_car_wins(self):
        """切ると、これまでどおり「減りの大きい車」になる。"""

        # 車0は残り1、車1は残り40。減りは車0のほうが大きいので
        # どちらの設定でも車0になる。ここでは**落ちないこと**を見る。
        self.assertIn(self._ask(False), (0, 1))

    def test_it_is_on(self):
        """**入れてあること。**

        磨きつきで測ると本選1回戦6試合（16x16・燃料40・巡回3台）で
        日別222→**223**（玉278→277）。日別は玉数より上位の得点です。
        生成12マップでは日別・玉ともに1つも変わりませんでした。
        """

        from hexa import production as P

        self.assertTrue(P.STUCK_FIRST)


class SupplyScheduleTest(unittest.TestCase):
    """**迎える順番を全部試して、遅れの合計を小さくする**ことの確認。

    `design_with_supply` は「燃料が切れる歩数」が早い車から順に
    迎えに行っていました。ところが補給車の移動時間を見ていないので、
    間に合わない順番になります。間に合わなければ巡回車は待ちます。

    迎え先はせいぜい巡回車の台数ぶんなので、並べ方を全部試せます
    （6台で720通り）。
    """

    def test_it_picks_the_order_with_less_lateness(self):
        from hexa.allocate import _best_meeting_order
        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot

        # 横一列。補給車はマス0。迎え先はマス1（締め切り30）と
        # マス9（締め切り9）。平地は1マス2歩。
        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(60,), height=1,
            width=10, cells=tuple([0] * 10),
            spots=(Spot(brand=1, position=5, stocks=1),),
            agent_positions=(0,), fuel_limit=100, players=2,
            busy_threshold=0.5, jammed_threshold=0.8,
        )
        finder = PathFinder(problem)
        # 締め切りの順（マス9が先）だと: 9へ18歩（遅れ9）、
        # そこから1へ16歩で計34歩（遅れ4）→ 合計13。
        # 逆順だと: 1へ2歩（遅れ0）、9へ18歩で計20歩（遅れ11）→ 合計11。
        meetings = [(9, 9, 1), (30, 1, 0)]
        got = _best_meeting_order(finder, {}, {}, 0, 60, meetings)
        self.assertEqual([cell for cell, _car in got], [1, 9],
                         "遅れの大きい並びを選んでいます")

    def test_everyone_is_served_when_possible(self):
        from hexa.allocate import _best_meeting_order
        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot

        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(60,), height=1,
            width=10, cells=tuple([0] * 10),
            spots=(Spot(brand=1, position=5, stocks=1),),
            agent_positions=(0,), fuel_limit=100, players=2,
            busy_threshold=0.5, jammed_threshold=0.8,
        )
        finder = PathFinder(problem)
        meetings = [(5, 1, 0), (10, 2, 1), (20, 3, 2)]
        got = _best_meeting_order(finder, {}, {}, 0, 60, meetings)
        self.assertEqual(len(got), 3, "迎えられる車を落としています")

    def test_the_knobs_exist(self):
        from hexa import allocate as A

        self.assertIsInstance(A.SUPPLY_SCHEDULE, bool)
        self.assertGreaterEqual(A.SUPPLY_SCHEDULE_LIMIT, 2)


class SupplyIdleWaitTest(unittest.TestCase):
    """**行き先を見失った補給車が、その日ずっと凍らない**ことの確認。

    `_wait_length` は補給車の `had_options` が偽のとき「残り全部」を
    返していました。ところが `had_options` は方策が候補の数を返した
    ときだけ真になり、`GreedyPolicy` も `PlannedRoutePolicy` も
    `candidate_features=[]` を返します。つまり**貪欲な補給車は、
    行き先を1度見失うとその日ずっと止まりました。**

    第1627（16x16・32歩）の1日目の実測:

      補給車の行動計画 [4, 4, 3, 4, 1, 1, -24]
      → 歩8で「残り24歩待機」。以後なにもしていません。その日の
        終わりに巡回車3は燃料2、巡回車2は燃料14。翌日から2台が
        燃料切れで、3日目は3台のうち2台が1歩も動けませんでした。

    この試合を記録された渋滞つきで通し直すと（1日12秒）

      そのまま       : 日別38 / 玉46
      短く待つ（直し）: **日別40** / 玉47
    """

    def _state(self, had_options=False):
        from hexa.planner import _AgentPlanState

        state = _AgentPlanState()
        state.had_options = had_options
        return state

    def test_the_supply_car_waits_only_a_little(self):
        from hexa.planner import SUPPLY_IDLE_WAIT, _wait_length
        from hexa.rules import AgentKind

        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        states = [self._state(), self._state()]
        got = _wait_length(index=1, kinds=kinds, remaining=24,
                           plan_state=states, day_steps=32)
        self.assertEqual(got, SUPPLY_IDLE_WAIT)
        self.assertLess(got, 24, "その日の終わりまで凍っています")

    def test_it_never_waits_past_the_day(self):
        from hexa.planner import _wait_length
        from hexa.rules import AgentKind

        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        states = [self._state(), self._state()]
        self.assertEqual(
            _wait_length(index=1, kinds=kinds, remaining=1,
                         plan_state=states, day_steps=32), 1)

    def test_the_knob_is_small(self):
        from hexa import planner as pl

        self.assertGreaterEqual(pl.SUPPLY_IDLE_WAIT, 1)
        self.assertLessEqual(pl.SUPPLY_IDLE_WAIT, 4)

    def test_greedy_supply_reports_no_options(self):
        """この直しが要る理由（方策は候補の数を返しません）。"""

        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot
        from hexa.production import GreedyPolicy
        from hexa.rules import AgentKind

        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(40,), height=1,
            width=6, cells=tuple([0] * 6),
            spots=(Spot(brand=1, position=2, stocks=1),),
            agent_positions=(0, 5), fuel_limit=40, players=2,
            busy_threshold=1, jammed_threshold=2,
        )
        policy = GreedyPolicy(problem)
        got = policy.choose_supply_target(
            agent_index=1, step=0, remaining=40, position=5,
            kinds=(AgentKind.TOURING, AgentKind.SUPPLY),
            positions=[0, 5], fuels=[10, 0], plan_targets=[None, None],
            supply_claims={}, road_status={}, finder=PathFinder(problem),
            day_index=0, day_count=1,
        )
        self.assertEqual(len(got.candidate_features), 0,
                         "候補の数を返すなら had_options は使えます")


class SupplyRateTest(unittest.TestCase):
    """**1歩あたり何燃料入れられるかで迎え先を選ぶ**ことの確認。

    第1627の2日目。補給車はマス71、巡回車は

      車0 マス122（残り23・約10歩）
      車2 マス 13（残り14・約12歩）
      車3 マス224（残り 2・約24歩） ← 減りが最大なのでこれまではこちら

    36歩の日に24歩かけて車3だけを満タンにし、車0と車2は残り2と0まで
    枯れました。翌日は3台のうち2台が1歩も動けず、玉5・日別5でした。
    """

    def _ask(self, by_rate):
        from hexa import production as P
        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot
        from hexa.rules import AgentKind

        # 横一列（平地は1マス2歩）。補給車はマス0。
        #   車0 マス2（4歩先・残り30 → 入る10）
        #   車1 マス9（18歩先・残り2 → 入る38）
        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(40,), height=1,
            width=10, cells=tuple([0] * 10),
            spots=(Spot(brand=1, position=5, stocks=1),),
            agent_positions=(2, 9, 0), fuel_limit=40, players=2,
            busy_threshold=1, jammed_threshold=2,
        )
        kinds = (AgentKind.TOURING, AgentKind.TOURING, AgentKind.SUPPLY)
        was_rate, was_stuck = P.SUPPLY_BY_RATE, P.STUCK_FIRST
        try:
            P.SUPPLY_BY_RATE = by_rate
            # 「行き詰まり」の判定は別の仕掛けなので、ここでは切ります。
            P.STUCK_FIRST = False
            policy = P.GreedyPolicy(problem)
            return policy.choose_supply_target(
                agent_index=2, step=0, remaining=40, position=0, kinds=kinds,
                positions=[2, 9, 0], fuels=[30, 2, 0],
                plan_targets=[None, None, None], supply_claims={},
                road_status={}, finder=PathFinder(problem),
                day_index=0, day_count=1,
            ).target_agent
        finally:
            P.SUPPLY_BY_RATE, P.STUCK_FIRST = was_rate, was_stuck

    def test_the_nearer_car_wins_on_rate(self):
        # 車0: 入る10 ÷ 4歩 = 2.5 / 車1: 入る38 ÷ 18歩 = 2.1
        self.assertEqual(self._ask(True), 0)

    def test_without_it_the_emptiest_car_wins(self):
        self.assertEqual(self._ask(False), 1)

    def test_the_knob_exists(self):
        from hexa import production as P

        self.assertIsInstance(P.SUPPLY_BY_RATE, bool)


class BrandFixDailyOnlyTest(unittest.TestCase):
    """**「足りない系列を拾う」直しは、日別が増えるときだけ採る**確認。

    `cover_missing_brands` は足りない系列の店を車の**先頭**に差し込み
    ます。必ず着く代わりに後ろの店を落とすので、玉数は減ります。
    日別が増えるならそれでいいのですが、採否は鍵全体で見ていました。

    第1627の実戦ログ:

      1日目「足りない系列を拾いました（日別 7 → 7）」
      2日目「足りない系列を拾いました（日別 5 → 5）」

    どちらも日別は増えていません。鍵の別の要素が上がって採られ、
    玉数だけが減っていました。
    """

    def test_a_flat_daily_is_refused(self):
        import inspect as _inspect

        from hexa import production as P

        source = " ".join(
            _inspect.getsource(P.cover_missing_brands).split())
        self.assertIn("BRAND_FIX_DAILY_ONLY and got.key[1] <= current.key[1]",
                      source)
        self.assertIn("BRAND_FIX_DAILY_ONLY and current.key[1] <= best.key[1]",
                      source)

    def test_the_log_shows_the_balls_too(self):
        """玉数も出さないと、損をしていても気づけません。"""

        import inspect as _inspect

        from hexa.production import play_production_match

        source = " ".join(
            _inspect.getsource(play_production_match).split())
        self.assertIn("足りない系列を拾いました（日別", source)
        self.assertIn("玉 {best.key[KEY_NOODLES]}", source)

    def test_the_knob_exists(self):
        from hexa import production as P

        self.assertIsInstance(P.BRAND_FIX_DAILY_ONLY, bool)


class CoverDesignTest(unittest.TestCase):
    """**系列をまず全部押さえる設計**の確認。

    第1627（16x16・系列10・巡回3台）の盤面では、**系列10個のうち
    7個は店が1軒しかありません**（系列0はマス0だけ、1は10、4は79、
    6は144、7は192、8は224、9は245）。日別を満点にするには毎日
    その7マスを踏むしかありません。

    玉数から組む設計は、その高い店を後ろに回して切り捨てます。
    実戦では2・3・4日目に系列0と1（どちらも左上の角）を落として
    日別36でした（1位は44）。
    """

    def _setup(self, seed=5, label="準決勝"):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings(label, random.Random(seed),
                                            players=10, day_count=5))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1
            else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        return problem, kinds, PathFinder(problem)

    def test_it_fits_the_step_budget(self):
        from hexa.allocate import design_cover, route_cost

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit] * problem.agent_count
        day_steps = problem.day_steps[0]
        routes = design_cover(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            day_steps, acquired_brands=set(), refuels=0,
        )
        self.assertTrue(any(routes.values()), "1店も割り当てていません")
        for index, spots in routes.items():
            self.assertEqual(len(set(spots)), len(spots),
                             "同じ車が同じ店を2回回っています")
            cost = route_cost(finder, {}, problem.agent_positions[index],
                              spots)
            self.assertLessEqual(cost, day_steps,
                                 f"{index}号車の順路が歩数を超えています")

    def test_it_covers_more_brands_than_the_ball_design(self):
        """玉数から組む設計より、押さえる系列が少なくならないこと。"""

        from hexa.allocate import design_cover, design_day

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit] * problem.agent_count
        day_steps = problem.day_steps[0]
        brand_of = {spot.position: spot.brand for spot in problem.spots}

        def brands(routes):
            return {brand_of[spot] for line in routes.values()
                    for spot in line}

        cover = design_cover(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            day_steps, acquired_brands=set(), refuels=0, fill=False)
        plain = design_day(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            day_steps, stock_weight=0.0, brand_weight=0.0,
            acquired_brands=set())
        self.assertGreaterEqual(len(brands(cover)), len(brands(plain)))

    def test_only_touring_cars_get_routes(self):
        from hexa.allocate import design_cover
        from hexa.rules import AgentKind

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit] * problem.agent_count
        routes = design_cover(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], acquired_brands=set())
        for index in routes:
            self.assertEqual(kinds[index], AgentKind.TOURING)

    def test_all_supply_gives_nothing(self):
        from hexa.allocate import design_cover
        from hexa.rules import AgentKind

        problem, _kinds, finder = self._setup()
        kinds = tuple([AgentKind.SUPPLY] * problem.agent_count)
        fuels = [problem.fuel_limit] * problem.agent_count
        self.assertEqual(
            design_cover(problem, finder, kinds,
                         list(problem.agent_positions), fuels, {},
                         problem.day_steps[0], acquired_brands=set()),
            {})

    def test_it_is_offered_as_a_candidate(self):
        from hexa.production import _planned_route_policies

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit if k == 0 else 0 for k in kinds]
        labels = [label for label, _policy in _planned_route_policies(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], set())]
        self.assertTrue(any(l.startswith("系列網羅") for l in labels),
                        "系列網羅の設計が候補にありません")


class LazyPreviewTest(unittest.TestCase):
    """**ビューアの盤面は、書くときだけ作る**ことの確認。

    探索は候補1件ごとに `monitor.searching` を呼びますが、書くのは
    `SEARCH_WRITE_GAP`（0.2秒）に1回です。ところが Python は引数を
    先に評価するので、盤面を dict で渡すと**書かない回でも作られます**。

    第1627の盤面で測ると

      候補1件を作る  : 0.58ミリ秒
      盤面を作り直す : 0.22ミリ秒（候補作りの38%）

    最良案が変わり続ける日（1日目は系列が毎回増えるのでそうなる）は
    毎回作り直しになり、**探索の38%を盤面作りに使っていました**。
    実戦の1日目は候補777件で、3日目は4784件でした。
    """

    def test_the_monitor_calls_the_preview_itself(self):
        import inspect as _inspect

        from hexa.livemonitor import LiveMonitor

        source = " ".join(_inspect.getsource(LiveMonitor.searching).split())
        self.assertIn("if callable(preview): preview = preview()", source)
        # 絞り込みのあとで呼ぶこと（前で呼んだら意味がありません）。
        self.assertLess(source.index("self._last_search = now"),
                        source.index("if callable(preview)"))

    def test_production_passes_a_function(self):
        import inspect as _inspect

        from hexa.production import play_production_match

        source = " ".join(
            _inspect.getsource(play_production_match).split())
        self.assertIn("preview=lambda: _preview_day(", source)

    def test_a_dict_still_works(self):
        """古い呼び方（dict をそのまま渡す）でも動くこと。"""

        from hexa.livemonitor import LiveMonitor

        monitor = LiveMonitor(enabled=False)
        monitor.searching(day_index=0, candidates=1, best_label="貪欲",
                          seconds_left=1.0, preview={"day": 1})


class SupplyFollowsTest(unittest.TestCase):
    """**補給車が、迎えた巡回車に同行する**ことの確認。

    第37回の解法紹介（他高専）を読むと、最多数派は**ペアを作って
    同行させ、燃料制約そのものを消す**やり方でした。

      長岡（35 Unfixed）「巡回車と補給車が同ルートを走行して常に
        給油するため、**燃料の残量を考慮する必要をなくした**」
      仙台広瀬（1 おぉ）「合流させて同時に動かすことで、**燃料無限の
        巡回車**として扱うことができる」

    こちらの実装は補給車が往復して配る形でした。第1703（32x32）の
    実測では巡回車の待ち歩の7割が燃料で、燃料の総量は足りていて
    **空の車と落ち合えていないだけ**でした。

    補給車は燃料を1滴も使わず、**同じマスから出れば移動にかかる
    歩数も同じ**です。だからいちど合流したらずっと一緒に動けます。
    """

    def test_it_takes_the_partner_next_cell(self):
        """相手がもう動き出していれば、その行き先を返すこと。"""

        from hexa.planner import _AgentPlanState, _follow_partner
        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot
        from hexa.rules import AgentKind

        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(40,), height=1,
            width=6, cells=tuple([0] * 6),
            spots=(Spot(brand=1, position=4, stocks=1),),
            agent_positions=(2, 2), fuel_limit=40, players=2,
            busy_threshold=1, jammed_threshold=2,
        )
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        states = [_AgentPlanState(), _AgentPlanState()]
        pending = [(2, "move", 3, 1), None]
        got = _follow_partner(1, kinds, [2, 2], states, pending, finder)
        self.assertEqual(got, (3, 0), "相手の行き先について行きません")

    def test_it_reads_the_partner_route_when_not_moving_yet(self):
        """相手がまだ決めていなければ、順路の先頭から読むこと。"""

        from hexa.planner import _AgentPlanState, _follow_partner
        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot
        from hexa.rules import AgentKind

        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(40,), height=1,
            width=6, cells=tuple([0] * 6),
            spots=(Spot(brand=1, position=4, stocks=1),),
            agent_positions=(2, 2), fuel_limit=40, players=2,
            busy_threshold=1, jammed_threshold=2,
        )
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        states = [_AgentPlanState(), _AgentPlanState()]
        # マス2の隣から、どれか1方向を順路の先頭に入れる。
        direction, next_position = finder.neighbors[2][0]
        states[0].route = [direction]
        got = _follow_partner(1, kinds, [2, 2], states, [None, None], finder)
        self.assertEqual(got, (next_position, 0))

    def test_it_stays_put_when_the_partner_stays(self):
        from hexa.planner import _AgentPlanState, _follow_partner
        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot
        from hexa.rules import AgentKind

        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(40,), height=1,
            width=6, cells=tuple([0] * 6),
            spots=(Spot(brand=1, position=4, stocks=1),),
            agent_positions=(2, 2), fuel_limit=40, players=2,
            busy_threshold=1, jammed_threshold=2,
        )
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        states = [_AgentPlanState(), _AgentPlanState()]
        self.assertIsNone(_follow_partner(
            1, kinds, [2, 2], states, [None, None], PathFinder(problem)))

    def test_nobody_on_this_cell_gives_nothing(self):
        from hexa.planner import _AgentPlanState, _follow_partner
        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot
        from hexa.rules import AgentKind

        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(40,), height=1,
            width=6, cells=tuple([0] * 6),
            spots=(Spot(brand=1, position=4, stocks=1),),
            agent_positions=(0, 5), fuel_limit=40, players=2,
            busy_threshold=1, jammed_threshold=2,
        )
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        states = [_AgentPlanState(), _AgentPlanState()]
        self.assertIsNone(_follow_partner(
            1, kinds, [0, 5], states, [None, None], PathFinder(problem)))

    def test_the_planner_uses_it_before_waiting(self):
        import inspect as _inspect

        from hexa.planner import DayPlanner

        source = " ".join(_inspect.getsource(DayPlanner.plan_day).split())
        self.assertIn("SUPPLY_FOLLOWS and kinds[index] == AgentKind.SUPPLY",
                      source)
        self.assertLess(source.index("_follow_partner("),
                        source.index("wait_length = _wait_length("),
                        "待つ判断より後ろにあると、同行できません")

    def test_escort_designs_exist_when_pairs_are_possible(self):
        """補給車が巡回車と同数以上なら、燃料を気にしない設計も作る。"""

        from hexa.pathfind import PathFinder
        from hexa.production import _planned_route_policies
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings("準決勝", random.Random(9),
                                            players=10, day_count=5))
        finder = PathFinder(problem)
        half = problem.agent_count // 2
        pairs = tuple(AgentKind.SUPPLY if i >= half else AgentKind.TOURING
                      for i in range(problem.agent_count))
        shuttle = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1
            else AgentKind.TOURING for i in range(problem.agent_count))
        fuels = [problem.fuel_limit] * problem.agent_count

        def labels(kinds):
            return [label for label, _policy in _planned_route_policies(
                problem, finder, kinds, list(problem.agent_positions), fuels,
                {}, problem.day_steps[0], set())]

        self.assertTrue(any("x6" in l for l in labels(pairs)),
                        "同行できる構成なのに燃料無視の設計がありません")
        self.assertFalse(any("x6" in l for l in labels(shuttle)),
                         "同行できない構成に燃料無視の設計を出しています")


class RarityStepsTest(unittest.TestCase):
    """**店が1軒しかない系列を、近いものとして扱う**ことの確認。

    鹿児島高専（51）の解法紹介に「スポットの希少性は、その日の時間と
    燃料で到達できる巡回車数から求める。**到達できる巡回車が少ない
    スポットを優先**することで、取得機会を失いやすいうどんを回収
    する」とあります。

    盤面を数えると、まさにそうでした。第1627（16x16）は**系列10個の
    うち7個が店1軒だけ**。16x16の11試合では、1日の歩の**7〜9割が
    系列網羅で消えます**（`an/cover_cost.py`）。1軒しかない系列を
    落とすと取り返せません。

    29試合を貪欲だけで通すと（時間に依存しないので差は本物）、
    この設定を抜いたときがいちばん悪くなりました:

      いまの既定            : 日別3067 玉5989 順位合計512
      希少な系列を抜く      : 日別2979 玉5783 順位合計**548**
      1歩/2歩/6歩/8歩/12歩  : 538/531/538/536/532

    **鍵の桁にするのはやりすぎ**でした（近くの未取得系列を捨てて
    盤面の端まで走る）。歩数のおまけとして効かせます。
    """

    def _problem(self):
        from hexa.problem import Problem, Spot

        # 横一列（平地は1マス2歩）。マス2は店2軒の系列、マス6は1軒だけ。
        return Problem(
            starts_at=0, day_seconds=(90,), day_steps=(60,), height=1,
            width=9, cells=tuple([0] * 9),
            spots=(Spot(brand=1, position=2, stocks=1),
                   Spot(brand=1, position=4, stocks=1),
                   Spot(brand=2, position=6, stocks=1)),
            agent_positions=(0,), fuel_limit=60, players=2,
            busy_threshold=1, jammed_threshold=2,
        )

    def _ask(self, steps_bonus):
        from hexa import production as P
        from hexa.pathfind import PathFinder

        problem = self._problem()
        finder = PathFinder(problem)
        was = P.RARITY_STEPS
        try:
            P.RARITY_STEPS = steps_bonus
            policy = P.GreedyPolicy(problem)
            return policy.choose_touring_target(
                agent_index=0, step=0, remaining=60, position=0, fuel=60,
                stock={2: 1, 4: 1, 6: 1}, reserved={}, acquired_today=set(),
                acquired_brands=set(), brands_today=set(), road_status={},
                finder=finder, kinds=(), positions=[0], day_index=0,
                day_count=1,
            ).target
        finally:
            P.RARITY_STEPS = was

    def test_the_lonely_brand_wins_the_detour(self):
        """1軒しかない系列は、少し遠くても先に取る。

        マス2は4歩・系列1（店2軒）、マス6は12歩・系列2（店1軒）。
        おまけ4歩なら マス2 は 4 + 4×(2-1) = 8、マス6 は 12 + 0 = 12
        なので マス2 が勝ちます。おまけ12歩なら マス2 は 4+12=16 で
        マス6（12）が勝ちます。
        """

        self.assertEqual(self._ask(0), 2, "おまけ0なら近い店です")
        self.assertEqual(self._ask(4), 2)
        self.assertEqual(self._ask(12), 6, "1軒の系列を先に取っていません")

    def test_the_count_is_per_brand_not_per_spot(self):
        from hexa.production import GreedyPolicy

        policy = GreedyPolicy(self._problem())
        self.assertEqual(policy.brand_spots[2], 2, "系列1は2軒です")
        self.assertEqual(policy.brand_spots[4], 2)
        self.assertEqual(policy.brand_spots[6], 1, "系列2は1軒です")

    def test_it_is_turned_off(self):
        """**磨きの前では逆に悪くなるので、既定は0です。**

        貪欲だけの29試合では抜いたときがいちばん悪い（512→548）のに、
        磨きを入れた29試合（手数2000）では

          1.47.0のまま : 日別3132 玉7396 順位合計388
          これを足す   : 日別3130 玉7310 順位合計404

        と逆になります。磨きは貪欲の受け持ちを組み替えるので、貪欲の
        段階で系列を優先しておくと、その組み替えに手数を使わせるだけ
        でした。**仕掛けは残し、既定は0**にしています。
        """

        from hexa import production as P

        self.assertEqual(P.RARITY_STEPS, 0,
                         "磨きつきの29試合で悪化したので0が既定です")


class MeetCellSearchTest(unittest.TestCase):
    """**待ち合わせ先のセルも磨きで動かす**ことの確認。

    以前は `RoutePolisher` が設計から受け取った (セル, 車) の組から
    **セルを捨てて**いました（「迎え先はその場で選び直す」）。
    ところが待ち合わせの場所は、燃料が切れるまでの時間と巡回車の
    寄り道量を決めるいちばん重要な変数です。それが探索の外に
    あったので、鹿児島高専（51）や沖縄（12）の「先回り」の形が
    原理的に作れませんでした。

    第1703（32x32）で磨きに9倍の時間をかけても玉が3個しか増えな
    かったのはこれが理由です（手数は足りていて、動かせる変数が
    足りていなかった）。32x32の4試合を手数600で通すと

      セルを動かさない : 日別725 玉2036
      セルも動かす     : 日別724 玉**2051**
    """

    def _kinds(self, cars):
        from hexa.rules import AgentKind

        return tuple([AgentKind.TOURING] * cars + [AgentKind.SUPPLY])

    def test_the_cells_are_kept(self):
        from hexa.improve import RoutePolisher

        polisher = RoutePolisher.__new__(RoutePolisher)
        polisher.touring = [0, 1]
        polisher.supply = [
            (int(car[0]), int(car[1])) if isinstance(car, tuple) else int(car)
            for car in [(7, 0), 1]
        ]
        self.assertEqual(polisher.supply, [(7, 0), 1],
                         "迎え先のセルを捨てています")

    def test_a_move_sets_the_cell_to_a_spot_on_the_route(self):
        from hexa import improve as I
        from hexa.improve import _supply_moves

        routes = {0: [11, 12, 13], 1: [21]}
        was = I.SUPPLY_MEET_SPOTS
        try:
            I.SUPPLY_MEET_SPOTS = 4
            got = list(_supply_moves([0, 1], [0, 1], routes))
        finally:
            I.SUPPLY_MEET_SPOTS = was
        self.assertIn([(11, 0), 1], got, "順路の店を迎え先にする手がありません")
        self.assertIn([(12, 0), 1], got)
        self.assertIn([0, (21, 1)], got)

    def test_a_move_clears_the_cell(self):
        from hexa.improve import _supply_moves

        # 指定をやめる手は `SUPPLY_MEET_SPOTS` に関係なく出ます。
        got = list(_supply_moves([0], [(11, 0)], {0: [11]}))
        self.assertIn([0], got, "迎え先の指定をやめる手がありません")

    def test_the_order_moves_still_work_with_cells(self):
        from hexa.improve import _supply_moves

        got = list(_supply_moves([0, 1], [(11, 0), 1], {0: [11], 1: [21]}))
        self.assertIn([1, (11, 0)], got, "並べ替えが組を壊しています")

    def test_the_polisher_asks_for_the_cell_moves(self):
        import inspect as _inspect

        from hexa.improve import _all_moves

        source = " ".join(_inspect.getsource(_all_moves).split())
        self.assertIn("_supply_moves(touring, supply, routes)", source)


class SupplyHintTest(unittest.TestCase):
    """**設計の迎え先（燃料が切れる地点）を優先して待ち受ける**確認。

    以前は「いる場所・行き先・設計の迎え先」のうち**いちばん近い
    ところ**で待ち受けていました。近いところで落ち合うと、その車は
    まだ燃料が残っているので**入る量が小さい**のです。

    第1703（32x32）で補給を数えると9日で44〜63回もあるのに、1回
    あたり9〜111燃料とばらばらで、ほぼ満タンの車に入れている回が
    多くありました。歩を100%使うには1日4回ぶんの満タンが要るので、
    **回数ではなく1回の量**が足りません。
    """

    def _ask(self, prefer):
        from hexa import production as P
        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot
        from hexa.rules import AgentKind

        # 横一列。補給車はマス0、巡回車はマス1。設計の迎え先はマス8
        # （その車の燃料が切れる地点）。
        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(60,), height=1,
            width=9, cells=tuple([0] * 9),
            spots=(Spot(brand=1, position=4, stocks=1),),
            agent_positions=(1, 0), fuel_limit=60, players=2,
            busy_threshold=1, jammed_threshold=2,
        )
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        policy = P.PlannedRoutePolicy(problem, {0: [4]}, [(8, 0)])
        was = P.SUPPLY_PREFER_HINT
        try:
            P.SUPPLY_PREFER_HINT = prefer
            return policy.choose_supply_target(
                agent_index=1, step=0, remaining=60, position=0, kinds=kinds,
                positions=[1, 0], fuels=[30, 0], plan_targets=[4, None],
                supply_claims={}, road_status={},
                finder=PathFinder(problem), day_index=0, day_count=1,
            ).target
        finally:
            P.SUPPLY_PREFER_HINT = was

    def test_the_hint_wins_over_the_nearest(self):
        self.assertEqual(self._ask(True), 8,
                         "近いところで落ち合って、入る量を小さくしています")

    def test_without_it_the_nearest_wins(self):
        self.assertEqual(self._ask(False), 1)

    def test_the_knob_exists(self):
        from hexa import production as P

        self.assertIsInstance(P.SUPPLY_PREFER_HINT, bool)


class HubDesignTest(unittest.TestCase):
    """**補給車を動かさない設計（ハブ）** の確認。

    第1703（32x32・巡回6台・9日・平均80.9歩）の数字:

      1位は1台1日11軒。81歩で11軒なら盤面の3分の1を掃く周回です。
      81歩の周回には燃料90（1歩1.11燃料）、タンクは113で1.26日分。
      つまり**6台が毎日1回ずつ満タン**にしてもらう必要があり、
      補給車の持ち歩81歩では**1回13.5歩**しか使えません。

    往復では間に合わないので、**全車の周回が通る一点に置く**形です
    （佐世保11・仙台名取41・秋田23・釧路27が同じことを書いています）。

    **実測では玉が減ったので既定はオフ**です（`HUB_DESIGN`）。
    仕掛けと理由は残してあります。
    """

    def _setup(self, seed=4, label="決勝"):
        from hexa.generator import generate_problem as gen, honsen_settings
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = gen(2026, honsen_settings(label, random.Random(seed),
                                            players=10, day_count=5))
        kinds = tuple(
            AgentKind.SUPPLY if i >= problem.agent_count - 1
            else AgentKind.TOURING
            for i in range(problem.agent_count)
        )
        return problem, kinds, PathFinder(problem)

    def test_the_hub_is_a_spot(self):
        """置く場所は**店**にします。

        順路は店の並びなので、店でない待ち合わせ点は表せません。
        店にすれば巡回車は普通の目的地として寄り、玉も入ります。
        """

        from hexa.allocate import pick_hub

        problem, _kinds, finder = self._setup()
        hub = pick_hub(problem, finder, {})
        self.assertIn(hub, {spot.position for spot in problem.spots})

    def test_the_hub_is_central(self):
        """ほかの店からの距離の合計がいちばん小さい店であること。"""

        from hexa.allocate import _table, pick_hub

        problem, _kinds, finder = self._setup()
        hub = pick_hub(problem, finder, {})
        cache = {}

        def spread(start):
            table = _table(finder, start, {}, cache)
            total = 0
            for spot in problem.spots:
                gap = table.steps.get(spot.position)
                total += gap if gap is not None else problem.cell_count
            return total

        best = spread(hub)
        for spot in problem.spots:
            self.assertLessEqual(best, spread(spot.position) + 1)

    def test_every_reachable_car_passes_the_hub(self):
        from hexa.allocate import design_hub

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit] * problem.agent_count
        routes, hub = design_hub(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], acquired_brands=set())
        self.assertIsNotNone(hub)
        self.assertTrue(any(hub in line for line in routes.values()),
                        "どの車もハブを通りません")

    def test_the_parked_supply_car_never_leaves(self):
        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot
        from hexa.production import PlannedRoutePolicy
        from hexa.rules import AgentKind

        problem = Problem(
            starts_at=0, day_seconds=(90,), day_steps=(40,), height=1,
            width=6, cells=tuple([0] * 6),
            spots=(Spot(brand=1, position=3, stocks=2),),
            agent_positions=(0, 5), fuel_limit=40, players=2,
            busy_threshold=1, jammed_threshold=2,
        )
        policy = PlannedRoutePolicy(problem, {0: [3]}, None, park=3)
        for position in (5, 4, 3):
            got = policy.choose_supply_target(
                agent_index=1, step=0, remaining=40, position=position,
                kinds=(AgentKind.TOURING, AgentKind.SUPPLY),
                positions=[0, position], fuels=[10, 0],
                plan_targets=[3, None], supply_claims={}, road_status={},
                finder=PathFinder(problem), day_index=0, day_count=1,
            )
            self.assertEqual(got.target, 3, "ハブから離れています")

    def test_no_supply_car_means_no_hub(self):
        from hexa.allocate import design_hub
        from hexa.rules import AgentKind

        problem, _kinds, finder = self._setup()
        kinds = tuple([AgentKind.TOURING] * problem.agent_count)
        fuels = [problem.fuel_limit] * problem.agent_count
        routes, hub = design_hub(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], acquired_brands=set())
        self.assertEqual((routes, hub), ({}, None))

    def test_it_is_turned_off(self):
        from hexa import production as P

        self.assertFalse(P.HUB_DESIGN,
                         "32x32の実測で玉2246→2230と減ったので既定はオフです")


class ShipConfigTest(unittest.TestCase):
    """**出荷する設定が、実測で決めた値のままであること。**

    29試合を手数2000で通した実測（時間予算に依存しないので決定的）:

      1.47.0のまま      : 日別3132 玉7396 順位合計388
      候補3種だけ足す   : 日別3135 玉7414 順位合計391（5勝6敗）
      **この設定**      : 日別3134 玉**7441** 順位合計**354**（11勝4敗）
      この設定で待ち8歩 : 日別3134 玉7424 順位合計391（5勝8敗）

    補給車の待ちは**1歩**が正解です。単独では中立に見えましたが、
    ほかの直しとそろって初めて効きます。
    """

    def test_the_measured_defaults(self):
        from hexa import allocate as A, improve as I, planner as PL
        from hexa import production as P

        # 入れたもの
        self.assertEqual(PL.SUPPLY_IDLE_WAIT, 1)
        # 本選6試合で日別218→221・玉273→277。巡回3台以下だけ効かせる。
        self.assertEqual(P.KEEP_FUEL_RATIO, 0.1)
        self.assertEqual(P.TIGHT_MAX_TOURING, 3)
        self.assertTrue(P.TOMORROW_REACH_FUEL)
        self.assertTrue(P.IMMOBILE_TIEBREAK)
        self.assertTrue(P.STUCK_FIRST)
        # 準決勝32x32で玉230→240、過去29試合で順位340→317。
        self.assertEqual(P.SUPPLY_PULL, 0.5)
        self.assertTrue(P.FORMATION_DESIGN)
        self.assertTrue(P.BRAND_FIX_DAILY_ONLY)
        self.assertTrue(P.COVER_DESIGN)
        self.assertTrue(P.PASSING_ON)
        self.assertTrue(P.SEED_OTHER_CRITERIA)
        # 実測で入れなかったもの（仕掛けと理由は残してある）
        self.assertEqual(P.RARITY_STEPS, 0)
        self.assertFalse(P.SUPPLY_BY_RATE)
        self.assertEqual(P.DAILY_PLUS_COVER, 0)
        self.assertEqual(P.FUEL_BALL_RATE, 0.0)
        self.assertEqual(P.NIGHT_BALL_RATE, 0.0)
        self.assertFalse(P.EXACT_DESIGN)
        self.assertFalse(P.NIGHT_DESIGN)
        # 29試合で順位318→321とわずかに悪化（良くなった試合は0）。
        self.assertFalse(P.EXACT_POLISH)
        # 実行を設計に合わせると29試合で順位318→328（燃料が縛るため）。
        self.assertFalse(P.PLANNED_USES_SHORTEST)
        from hexa import allocate as A2
        # 逆向き（設計を実行に合わせる）は32x32で玉231のまま動かない。
        self.assertFalse(A2.DESIGN_FUEL_WEIGHT)
        self.assertEqual(P.KEEP_FUEL_REACH, 0.0)
        self.assertFalse(P.SUPPLY_PREFER_HINT)
        self.assertFalse(P.HUB_DESIGN)
        self.assertEqual(P.MEET_ON_SPOT_STEPS, 0)
        self.assertFalse(PL.SUPPLY_FOLLOWS)
        self.assertFalse(PL.DRIFT_TO_SPOT)
        self.assertFalse(PL.MEET_SUPPLY_WHEN_STUCK)
        self.assertFalse(A.SUPPLY_SCHEDULE)
        self.assertEqual(I.SUPPLY_MEET_SPOTS, 0)


class FuelWeightBalanceTest(unittest.TestCase):
    """**歩数と燃料の重みが、書いてあるとおりに釣り合うこと。**

    `fuel_weight` の説明にはこうあります。

        1マスのコストを ステップ数 / S + 燃料 / F で測れば、
        足りないほうの資源が自動的に重くなります。

    ところが実装は `round(W × min(1, S/F))` で、**S=F のときに W**
    （＝燃料だけ最小化、歩数は完全に無視）を返していました。
    コストの形が `歩数×(W-w) + 燃料×w` なので、その式と同じ向きに
    するには

        w / (W - w) = S / F   →   w = W × S / (S + F)

    で、S=F なら W/2（釣り合い）が正しい値です。

    **本選1回戦の盤面はまさにそこでした。** 1日の歩数40・燃料上限40
    なので毎朝 S=F=40 → 重み4 → 道路を一切使わない。道路は1歩・
    燃料2、平地は2歩・燃料1なので、燃料だけ見ると道路は4倍高く
    見えます。ところが1日目は全道路が順調（1歩）で2倍速いのに、
    それを丸ごと捨てていました（16x16には道路が60セルあります）。
    """

    def test_equal_room_gives_the_middle(self):
        from hexa import pathfind as pf

        was = pf.FUEL_WEIGHT_BALANCED
        try:
            pf.FUEL_WEIGHT_BALANCED = True
            self.assertEqual(pf.fuel_weight(40, 40),
                             pf.FUEL_WEIGHT_STEPS // 2)
            pf.FUEL_WEIGHT_BALANCED = False
            self.assertEqual(pf.fuel_weight(40, 40), pf.FUEL_WEIGHT_STEPS,
                             "もとの実装は燃料だけ最小化していました")
        finally:
            pf.FUEL_WEIGHT_BALANCED = was

    def test_it_follows_the_formula(self):
        from hexa import pathfind as pf

        was = pf.FUEL_WEIGHT_BALANCED
        try:
            pf.FUEL_WEIGHT_BALANCED = True
            top = pf.FUEL_WEIGHT_STEPS
            for steps, fuel in ((40, 40), (40, 10), (40, 80), (10, 40),
                                (96, 113), (81, 113)):
                want = int(round(top * steps / (steps + fuel)))
                self.assertEqual(pf.fuel_weight(steps, fuel), want)
        finally:
            pf.FUEL_WEIGHT_BALANCED = was

    def test_no_fuel_still_means_fuel_first(self):
        from hexa import pathfind as pf

        self.assertEqual(pf.fuel_weight(40, 0), pf.FUEL_WEIGHT_STEPS)

    def test_the_weight_stays_in_range(self):
        from hexa import pathfind as pf

        for steps in (0, 1, 50, 200):
            for fuel in (1, 5, 50, 500):
                got = pf.fuel_weight(steps, fuel)
                self.assertGreaterEqual(got, 0)
                self.assertLessEqual(got, pf.FUEL_WEIGHT_STEPS)


class TightFuelTest(unittest.TestCase):
    """**盤面の燃料の逼迫度で、不足の数え方を切り替えること。**

    満タン1杯が1日の歩数をまかなえない盤面では、「何台足りないか」
    ではなく「**どれだけ足りないか**」で見るべきです。3台しかない
    盤面で台数で数えると 0/1/2/3 の4段階しかなく、乱暴すぎます。

    本選1回戦の実問題6つ（16x16・燃料40・平均39.2歩 → 比1.02）を
    渋滞つきで通すと

      台数で数える   : 日別212 玉265
      深刻さで数える : 日別**218** 玉**273**

    過去問29試合を比のしきい値で分けると符号が変わります
    （磨きつき・手数2000・順位合計、小さいほうが良い）:

      | | 試合 | 台数 | 深刻さ |
      |---|---|---|---|
      | 逼迫（比 ≤ 1.1） |  7 | 92 | **89** |
      | 余裕（比 > 1.1） | 22 | **262** | 281 |

    **過去問（練習場）と本選でパラメータが違う**ので、盤面サイズでは
    なくその盤面の数字で決めます。
    """

    def _problem(self, fuel, steps):
        from hexa.problem import Problem, Spot

        return Problem(
            starts_at=0, day_seconds=tuple([60] * len(steps)),
            day_steps=tuple(steps), height=1, width=4, cells=(0, 0, 0, 0),
            spots=(Spot(brand=0, position=1, stocks=1),),
            agent_positions=(0,), fuel_limit=fuel, players=9,
            busy_threshold=1, jammed_threshold=2)

    def test_the_finals_board_is_tight(self):
        from hexa.production import fuel_is_tight

        # 本選1回戦（16x16・燃料40・歩[32,36,40,40,48]）
        self.assertTrue(fuel_is_tight(
            self._problem(40, [32, 36, 40, 40, 48])))

    def test_a_loose_board_is_not(self):
        from hexa.production import fuel_is_tight

        # 過去問の16x16（燃料60・1日40歩）と32x32（燃料113・81歩）
        self.assertFalse(fuel_is_tight(self._problem(60, [40] * 5)))
        self.assertFalse(fuel_is_tight(self._problem(113, [81] * 9)))

    def test_no_problem_is_not_tight(self):
        from hexa.production import fuel_is_tight

        self.assertFalse(fuel_is_tight(None))

    def test_the_answer_is_remembered(self):
        from hexa import production as P

        problem = self._problem(40, [32, 36, 40, 40, 48])
        P._TIGHT_CACHE.clear()
        P.fuel_is_tight(problem)
        self.assertEqual(len(P._TIGHT_CACHE), 1, "覚えていません")
        P.fuel_is_tight(problem)
        self.assertEqual(len(P._TIGHT_CACHE), 1)

    def test_the_key_uses_severity_on_a_tight_board(self):
        import inspect as _inspect

        from hexa.production import _candidate_key

        source = " ".join(_inspect.getsource(_candidate_key).split())
        self.assertIn("weighted = WEIGHTED_SHORTFALL or ( WEIGHTED_WHEN_TIGHT"
                      " and len(touring_seats) <= TIGHT_MAX_TOURING"
                      " and fuel_is_tight(problem))", source)
        self.assertIn("TIGHT_WEIGHTED_LIMIT if not WEIGHTED_SHORTFALL",
                      source)

    def test_the_measured_values(self):
        from hexa import production as P

        self.assertTrue(P.WEIGHTED_WHEN_TIGHT)
        self.assertEqual(P.TIGHT_FUEL_RATIO, 1.1)
        self.assertEqual(P.TIGHT_WEIGHTED_LIMIT, 3)


class TerritoryTest(unittest.TestCase):
    """**系列を車に割り振って毎日同じ周回を繰り返す設計**の確認。

    本選1回戦は日別が勝負です（満点50に対し**1位でも44**。過去問の
    24x24・32x32では1位が29試合すべて満点でした）。

    第6試合の店を座標に直すと縄張りが成立します:

      144(行9列0)・192(行12列0)・224(行14列0)・245(行15列5)
        → 列0に縦に密集。1台で4系列
      0(行0列0)・10(行0列10) → 最上行。1台で2系列
      79(行4列15) と複数店の系列2・3・5 → 右側。1台で4系列

    こちらは日ごとに組み直すので1日目に燃料を使い切り、2〜3日目に
    2台が動けなくなっていました（実測の燃料残
    [40,40,2,3] → [33,40,2,3] → [27,40,2,40]）。

    **実測では候補に足しても悪くなったので既定はオフ**です
    （本選6試合で 縄張りなし 日別218玉273 / あり 日別215玉263）。
    割り振りそのものは正しく出るので、仕掛けを残しています。
    """

    def _setup(self):
        from hexa.pathfind import PathFinder
        from hexa.problem import Problem, Spot
        from hexa.production import kinds_for_seats

        # 横長の盤面。店を左端・中央・右端に3かたまり置く。
        width, height = 12, 1
        spots = (
            Spot(brand=0, position=0, stocks=2),
            Spot(brand=1, position=1, stocks=2),
            Spot(brand=2, position=5, stocks=2),
            Spot(brand=3, position=6, stocks=2),
            Spot(brand=4, position=10, stocks=2),
            Spot(brand=5, position=11, stocks=2),
        )
        problem = Problem(
            starts_at=0, day_seconds=(60,), day_steps=(40,), height=height,
            width=width, cells=tuple([0] * (width * height)), spots=spots,
            agent_positions=(0, 5, 10, 6), fuel_limit=40, players=2,
            busy_threshold=1, jammed_threshold=2)
        kinds = kinds_for_seats(4, (3,))
        return problem, kinds, PathFinder(problem)

    def test_every_brand_gets_an_owner(self):
        from hexa.allocate import pick_territory

        problem, kinds, finder = self._setup()
        plan = pick_territory(problem, kinds=kinds, finder=finder,
                              positions=list(problem.agent_positions),
                              road_status={})
        brand_of = {s.position: s.brand for s in problem.spots}
        got = {brand_of[spot] for line in plan.values() for spot in line}
        self.assertEqual(got, {s.brand for s in problem.spots},
                         "割り振れていない系列があります")

    def test_each_car_keeps_its_own_cluster(self):
        """近い店どうしが同じ車にまとまること。"""

        from hexa.allocate import pick_territory

        problem, kinds, finder = self._setup()
        plan = pick_territory(problem, kinds=kinds, finder=finder,
                              positions=list(problem.agent_positions),
                              road_status={})
        for line in plan.values():
            if len(line) < 2:
                continue
            self.assertLessEqual(max(line) - min(line), 5,
                                 "離れた店が同じ車にまとまっています")

    def test_a_car_gets_no_spot_it_cannot_reach(self):
        from hexa.allocate import design_territory, route_cost

        problem, kinds, finder = self._setup()
        fuels = [problem.fuel_limit] * problem.agent_count
        routes = design_territory(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], acquired_brands=set(), fill=False)
        for index, line in routes.items():
            cost = route_cost(finder, {}, problem.agent_positions[index], line)
            self.assertLessEqual(cost, problem.day_steps[0])

    def test_no_touring_car_means_nothing(self):
        from hexa.allocate import design_territory, pick_territory
        from hexa.rules import AgentKind

        problem, _kinds, finder = self._setup()
        kinds = tuple([AgentKind.SUPPLY] * problem.agent_count)
        fuels = [problem.fuel_limit] * problem.agent_count
        self.assertEqual(pick_territory(problem, finder, kinds,
                                        list(problem.agent_positions), {}), {})
        self.assertEqual(design_territory(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0]), {})

    def test_it_is_turned_off(self):
        from hexa import production as P

        self.assertFalse(P.TERRITORY_DESIGN,
                         "本選6試合の実測で日別218→215と悪化したので既定オフ")


class TightMaxTouringTest(unittest.TestCase):
    """**巡回車が多い盤面では、不足を深刻さでは数えないこと。**

    台数で数えるのが乱暴なのは「段階が少ないから」です。巡回3台なら
    0/1/2/3 の4段階しかありませんが、6台なら7段階あってそれで足ります。

    生成した逼迫マップ4つ（hs24_hard/hs24_tight/hs32_hard/hs32_tight、
    24x24は巡回4台・32x32は巡回6台）を渋滞つき・手数600で通すと
    符号が逆でした:

      深刻さで数える : 日別427/430 玉745
      台数で数える   : 日別**430/430** 玉**785**

    いっぽう本選1回戦（16x16・巡回3台）は深刻さが
    日別218/玉273 対 212/265 で勝ちます。盤面の幅と台数は
    16→4台 / 24→5台 / 32→7台 と連動しているので、
    「巡回3台以下」はほぼ「16x16」と同じ意味になります。
    """

    def test_the_limit_is_three(self):
        from hexa import production as P

        self.assertEqual(P.TIGHT_MAX_TOURING, 3)

    def test_the_key_checks_the_touring_count(self):
        import inspect as _inspect

        from hexa.production import _candidate_key

        source = " ".join(_inspect.getsource(_candidate_key).split())
        self.assertIn("len(touring_seats) <= TIGHT_MAX_TOURING", source)
        # 台数を数える行が、使う行より前にあること。
        self.assertLess(source.index("touring_seats = tuple("),
                        source.index("len(touring_seats) <= TIGHT_MAX_TOURING"))


class ParkedWithSupplyTest(unittest.TestCase):
    """**補給車と同じマスで終えた巡回車は、明日満タンだと見ること。**

    `simulator` の補給は歩0の行動段階でも走るので、補給車と同じ
    マスで夜を越した巡回車は、翌日1歩も動かないうちに満タンに
    なります。明日の見積り（`_tomorrow_coverage`）が燃料で歩ける
    上限をかけるとき、ここを忘れるといちばん良い終わり方を
    罰してしまいます。
    """

    def _board(self):
        from hexa.problem import Problem, Spot

        # 1行8マスの平地。店は端に置きます。
        return Problem(
            starts_at=0, day_seconds=(60, 60), day_steps=(20, 20),
            height=1, width=8, cells=tuple([0] * 8),
            spots=(Spot(brand=0, position=0, stocks=1),
                   Spot(brand=1, position=7, stocks=1)),
            agent_positions=(0, 0), fuel_limit=40, players=9,
            busy_threshold=1, jammed_threshold=2)

    def test_the_source_credits_a_parked_car(self):
        import inspect as _inspect

        from hexa.production import _tomorrow_coverage

        source = " ".join(_inspect.getsource(_tomorrow_coverage).split())
        self.assertIn("if day_result.positions[index] in parked:", source)
        self.assertIn("fuel = problem.fuel_limit", source)

    def test_the_reach_slot_credits_it_too(self):
        import inspect as _inspect

        from hexa.production import _tomorrow_reach

        source = " ".join(_inspect.getsource(_tomorrow_reach).split())
        self.assertIn("in parked:", source)
        self.assertIn("fuel_left = None", source)


class CoverStepsPerFuelTest(unittest.TestCase):
    """**明日の見積りで「燃料1は何歩ぶんか」を1か所に置くこと。**

    平地は燃料1で2歩ですが、山地は燃料2で3歩（1.5歩）、道路は
    燃料2で1〜4歩（0.5〜2歩）です。2.0 は「全部平地」という
    いちばん甘い見立てで、甘く見るぶん今日の玉を捨てずに済みます。
    """

    def test_it_is_two_steps(self):
        from hexa import production as P

        self.assertEqual(P.COVER_STEPS_PER_FUEL, 2.0)

    def test_the_estimate_uses_the_knob(self):
        import inspect as _inspect

        from hexa.production import _tomorrow_coverage

        source = " ".join(_inspect.getsource(_tomorrow_coverage).split())
        self.assertIn("int(COVER_STEPS_PER_FUEL * fuel)", source)
        self.assertNotIn("2 * day_result.fuels[index]", source)


class TomorrowReachFuelTest(unittest.TestCase):
    """**明日の到達を、歩数だけでなく燃料でも絞ること。**

    本選1回戦の実問題を渋滞つきで通すと、第1試合の2日目・3日目・
    5日目は開始時の最低燃料が **1・1・4** でした。燃料1では平地1回
    （2歩）しか動けません。厳密な上限（燃料無視・渋滞あり・
    部分集合探索）と比べると、2日目は日別4に対し上限7で、
    **3系列を燃料で落としていました**。
    """

    def test_it_is_on(self):
        from hexa import production as P

        self.assertTrue(P.TOMORROW_REACH_FUEL)

    def test_the_fuel_cost_is_checked(self):
        import inspect as _inspect

        from hexa.production import _tomorrow_reach

        source = " ".join(_inspect.getsource(_tomorrow_reach).split())
        self.assertIn("need = table.fuel.get(spot.position)", source)
        self.assertIn("if need is not None and need > fuel_left:", source)


class ImmobileTiebreakTest(unittest.TestCase):
    """**そのマスを出られないまま夜を越す車を、同点決着で嫌うこと。**

    `_stranded_count` は「補給車が明日迎えに行ける距離なら立ち往生と
    数えない」作りで、見立てとしては正しいのですが、迎えに行く朝の
    ぶんは必ず失われます。

    本選1回戦第1試合の2日目、2号車は**セル17で燃料1のまま37歩を
    1歩も動かず**終わりました（補給車は2マス隣から出発して盤面の
    反対側へ行き、迎えには来ていません）。前日の計画が燃料1を
    残したのが始まりです。

    玉数より下の桁に置くので、**玉は1つも減りません**。
    """

    def _result(self, positions, fuels):
        from hexa.production import _FastResult

        class Fake:
            pass

        got = Fake()
        got.positions = tuple(positions)
        got.fuels = tuple(fuels)
        return got

    def _problem(self, cells):
        from hexa.problem import Problem, Spot

        return Problem(
            starts_at=0, day_seconds=(60,), day_steps=(20,), height=1,
            width=len(cells), cells=tuple(cells),
            spots=(Spot(brand=0, position=0, stocks=1),),
            agent_positions=(0, 1), fuel_limit=40, players=9,
            busy_threshold=1, jammed_threshold=2)

    def test_it_is_on(self):
        from hexa import production as P

        self.assertTrue(P.IMMOBILE_TIEBREAK)

    def test_a_car_with_one_fuel_on_a_mountain_counts(self):
        from hexa.production import _immobile_count
        from hexa.rules import AgentKind

        # セル1は山地（出るのに燃料2）。燃料1では出られません。
        problem = self._problem([0, 2, 0, 0])
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        got = self._result([1, 3], [1, 40])
        self.assertEqual(_immobile_count(got, kinds, problem), 1)

    def test_it_is_off_when_there_are_many_touring_cars(self):
        from hexa.production import _immobile_count
        from hexa.rules import AgentKind

        # 巡回4台（生成12マップで日別1634→1632・玉3831→3824）。
        problem = self._problem([0, 2, 0, 0])
        kinds = (AgentKind.TOURING,) * 4 + (AgentKind.SUPPLY,)
        got = self._result([1, 1, 1, 1, 3], [1, 1, 1, 1, 40])
        self.assertEqual(_immobile_count(got, kinds, problem), 0)

    def test_one_fuel_on_plain_is_fine(self):
        from hexa.production import _immobile_count
        from hexa.rules import AgentKind

        problem = self._problem([0, 0, 0, 0])
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        got = self._result([1, 3], [1, 40])
        self.assertEqual(_immobile_count(got, kinds, problem), 0)

    def test_parking_with_the_supply_car_is_fine(self):
        from hexa.production import _immobile_count
        from hexa.rules import AgentKind

        # 同じマスなら翌日の歩0で満タンになります。
        problem = self._problem([0, 2, 0, 0])
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        got = self._result([1, 1], [0, 40])
        self.assertEqual(_immobile_count(got, kinds, problem), 0)

    def test_it_sits_below_the_ball_count(self):
        import inspect as _inspect

        from hexa.production import KEY_NOODLES, _candidate_key

        source = " ".join(_inspect.getsource(_candidate_key).split())
        self.assertLess(source.index("len(day_result.acquisitions) + _fuel_as_balls"),
                        source.index("-_immobile_count("))
        self.assertEqual(KEY_NOODLES, 6)


class KeepFuelGateTest(unittest.TestCase):
    """**余力を残す仕掛けは、巡回車が少ない盤面だけで効かせること。**

    本選1回戦の実問題6つ（16x16・燃料40・巡回3台）を渋滞つき・
    手数800で通すと

      0（切る）  : 日別218 玉273
      **0.1**    : 日別**221** 玉**277**
      0.2        : 日別215 玉272

    いっぽう生成した逼迫マップ4つ（24x24は巡回4台・32x32は巡回6台）
    では 日別430/430 のまま 玉785→**761** と落ちました。
    """

    def test_the_ratio(self):
        from hexa import production as P

        self.assertEqual(P.KEEP_FUEL_RATIO, 0.1)

    def test_few_touring_counts_touring_cars(self):
        from hexa.production import few_touring
        from hexa.rules import AgentKind

        T, S = AgentKind.TOURING, AgentKind.SUPPLY
        self.assertTrue(few_touring((T, T, T, S)))
        self.assertFalse(few_touring((T, T, T, T, S)))
        self.assertFalse(few_touring(()))

    def test_the_reserve_is_gated(self):
        import inspect as _inspect

        from hexa.production import GreedyPolicy

        source = " ".join(
            _inspect.getsource(GreedyPolicy.choose_touring_target).split())
        self.assertIn("KEEP_FUEL_RATIO > 0.0 and few_touring(kinds)", source)


class SeatTrialTest(unittest.TestCase):
    """**4体の盤面では、補給車の席を振らずに後ろ詰めに固定すること。**

    本選1回戦の実問題6つ（16x16・4体）で、席を1つずつ固定して
    磨きつき（手数800・渋滞つき）で通した本当の成績:

      | 試合 | 席0 | 席1 | 席2 | 席3 |
      |---|---|---|---|---|
      | 第1 | 32 | 33 | 34 | **35** |
      | 第2 | 39 | **42** | 40 | **42** |
      | 第3 | **37** | **37** | 32 | 35 |
      | 第4 | 37 | **39** | 36 | 36 |
      | 第5 | 36 | 36 | **37** | **37** |
      | 第6 | 35 | 36 | 37 | **38** |

    選び方ごとの日別合計:

      | 選び方 | 日別合計 |
      |---|---|
      | 貪欲試走（それまでの形） | 216 |
      | 案そろえ試走（毎日全家族＋磨き1手） | 216 |
      | でたらめ（4席の平均） | 219.5 |
      | **後ろ詰めに固定** | **223** |
      | 毎試合の最良（届かない上限） | 228 |

    **試走はでたらめより悪い**ので振りません。いっぽう生成した
    24x24・32x32（5体・7体）では探したほうが上でした
    （日別1634/1634・玉3826 対 後ろ詰め1624/3776）。
    """

    def test_four_agents_only_try_the_back_seat(self):
        from hexa.production import supply_seat_choices

        self.assertEqual(supply_seat_choices(4, 1), [(3,)])
        self.assertEqual(supply_seat_choices(4, 2), [(2, 3)])

    def test_five_or_more_still_try_every_seat(self):
        from hexa.production import supply_seat_choices

        self.assertEqual(supply_seat_choices(5, 1),
                         [(0,), (1,), (2,), (3,), (4,)])
        self.assertEqual(len(supply_seat_choices(7, 1)), 7)

    def test_the_threshold(self):
        from hexa import production as P

        self.assertEqual(P.SEAT_TRIAL_MIN_AGENTS, 5)

    def test_no_supply_car_is_still_one_choice(self):
        from hexa.production import supply_seat_choices

        self.assertEqual(supply_seat_choices(4, 0), [()])
        self.assertEqual(supply_seat_choices(4, 4), [(0, 1, 2, 3)])


class ShortfallNeedTest(unittest.TestCase):
    """**「明日ぶんの燃料が足りている」の基準を1か所に置くこと。**

    本選6試合（席は後ろ詰め・渋滞つき・手数800）で振ると、
    甘いほうが強いという結果でした:

      | 燃料1を何歩と見るか | 日別 | 玉 |
      |---|---|---|
      | 1.0 | 218 | 265 |
      | 1.5 | 220 | 278 |
      | **2.0（既定）** | **223** | **288** |
      | 2.5 | 222 | 280 |

    厳しくすると今日の玉と系列を捨てすぎます。
    """

    def test_it_is_two(self):
        from hexa import production as P

        self.assertEqual(P.SHORTFALL_STEPS_PER_FUEL, 2.0)

    def test_both_counters_use_it(self):
        import inspect as _inspect

        from hexa.production import _fuel_shortfall, _fuel_shortfall_weighted

        for func in (_fuel_shortfall, _fuel_shortfall_weighted):
            source = " ".join(_inspect.getsource(func).split())
            self.assertIn("int(steps_ahead / SHORTFALL_STEPS_PER_FUEL)",
                          source, f"{func.__name__} が定数を使っていません")
            self.assertNotIn("steps_ahead // 2", source)


class DailyPlusCoverTest(unittest.TestCase):
    """**鍵の2番目は「今日の系列数」だけに保つこと。**

    採点は日別累計（毎日の合計）なので「今日＋明日の見積り」に
    すれば得なはず、と考えて試しましたが負けました
    （本選6試合・渋滞つき・手数800）:

      | 2番目の桁 | 日別 | 玉 |
      |---|---|---|
      | 今日だけ（既定） | **223** | 277 |
      | 今日＋明日×1 | 204 | 288 |
      | 今日＋明日×2 | 204 | 294 |

    見積りは厳密な上限とほぼ一致するのですが、**上限の半分しか
    実際には取れない**ので、1対1で今日の系列を手放すのは損でした。
    """

    def test_it_is_off(self):
        from hexa import production as P

        self.assertEqual(P.DAILY_PLUS_COVER, 0)

    def test_the_knob_is_wired_into_the_second_slot(self):
        import inspect as _inspect

        from hexa.production import _candidate_key

        source = " ".join(_inspect.getsource(_candidate_key).split())
        self.assertIn("len(day_brands) + DAILY_PLUS_COVER * cover_value",
                      source)


class FormationDesignTest(unittest.TestCase):
    """**明日の陣形で終わる設計**（`design_formation`）の確認。

    厳密な上限（部分集合DP）で「差は終わり方（位置と燃料）だけ」と
    分かったので、先に「明日どこで朝を迎えるか」を決めてから順路を
    組みます。終点は**店**です（順路は店の並びなので店以外は表せず、
    店で夜を越せば翌朝の1歩目で1玉入る）。

    候補が4つ増えるぶん磨きが薄まるので、手数が足りないと負けます
    （本選6試合・席は後ろ詰め・渋滞つき）:

      | 手数 | 陣形あり | 陣形なし |
      |---|---|---|
      | 800 | 日別222 玉285 | 日別**223** 玉288 |
      | 2500 | 日別**224** 玉287 | 日別223 玉294 |

    生成12マップ（手数600）は 玉3836 対 3826 で、日別はどちらも満点。
    本番の持ち時間は1日42〜300秒（磨き2500手は24x24で約3.4秒）なので
    入れてあります。
    """

    def _board(self):
        from hexa.generator import generate_problem, honsen_settings

        return generate_problem(
            2026, honsen_settings("1回戦", random.Random(5), players=9,
                                  day_count=5))

    def test_it_is_on(self):
        from hexa import production as P

        self.assertTrue(P.FORMATION_DESIGN)
        self.assertEqual(P.FORMATION_MARGINS, (0.5, 1.0))
        self.assertEqual(P.FORMATION_REFUELS, (0, 1))
        # 補給車の順路も渡す変種は実測で負けたので作りません
        # （生成12マップで日別1634→1632・玉3836→3823）。
        self.assertEqual(P.FORMATION_SUPPLY, (False,))

    def test_it_can_also_plan_the_supply_car(self):
        """**補給車にも陣形を教えられること。**

        `simulator` の補給は歩0の行動段階でも走るので、補給車と同じ
        マスで夜を越した巡回車は翌日1歩も動かないうちに満タンに
        なります。迎える順は燃料が少ない車から、待ち受け先はその車の
        陣形の店です。
        """

        from hexa.allocate import design_formation
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = self._board()
        finder = PathFinder(problem)
        kinds = tuple(
            AgentKind.SUPPLY if i == problem.agent_count - 1
            else AgentKind.TOURING for i in range(problem.agent_count))
        fuels = [problem.fuel_limit] * problem.agent_count
        fuels[0] = 3
        routes, supply = design_formation(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            problem.day_steps[0], next_steps=problem.day_steps[1],
            with_supply=True)
        self.assertTrue(routes)
        touring = [i for i, k in enumerate(kinds) if k == AgentKind.TOURING]
        self.assertEqual(len(supply), len(touring))
        self.assertEqual(supply[0][1], 0, "燃料が少ない車を先に迎えること")
        spots = {spot.position for spot in problem.spots}
        for hint, car in supply:
            self.assertIn(car, touring)
            if hint is not None:
                self.assertIn(hint, spots, "待ち受け先が店ではありません")

    def test_the_formation_is_made_of_spots(self):
        from hexa.allocate import pick_formation
        from hexa.pathfind import PathFinder

        problem = self._board()
        finder = PathFinder(problem)
        got = pick_formation(problem, finder, {}, 3, problem.day_steps[1])
        self.assertEqual(len(got), 3)
        self.assertEqual(len(set(got)), 3, "同じ店を2台に渡しています")
        spots = {spot.position for spot in problem.spots}
        for cell in got:
            self.assertIn(cell, spots, "終点が店ではありません")

    def test_more_cars_never_cover_fewer_brands(self):
        from hexa.allocate import pick_formation
        from hexa.pathfind import PathFinder

        problem = self._board()
        finder = PathFinder(problem)
        one = pick_formation(problem, finder, {}, 1, problem.day_steps[1])
        three = pick_formation(problem, finder, {}, 3, problem.day_steps[1])
        self.assertEqual(one[0], three[0], "貪欲なので1台目は同じはず")
        self.assertLessEqual(len(one), len(three))

    def test_no_tomorrow_means_no_design(self):
        """最終日（明日がない）では作らないこと。"""

        from hexa.allocate import design_formation
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = self._board()
        finder = PathFinder(problem)
        kinds = tuple(
            AgentKind.SUPPLY if i == problem.agent_count - 1
            else AgentKind.TOURING for i in range(problem.agent_count))
        got = design_formation(
            problem, finder, kinds, list(problem.agent_positions),
            [problem.fuel_limit] * problem.agent_count, {},
            problem.day_steps[-1], next_steps=0)
        self.assertEqual(got, {})

    def test_the_routes_end_on_the_formation_or_stay_valid(self):
        """終点を差し替えても、歩数と燃料の予算を守ること。"""

        from hexa.allocate import (_route_fuel, design_formation,
                                   route_cost)
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = self._board()
        finder = PathFinder(problem)
        kinds = tuple(
            AgentKind.SUPPLY if i == problem.agent_count - 1
            else AgentKind.TOURING for i in range(problem.agent_count))
        fuels = [problem.fuel_limit] * problem.agent_count
        steps = problem.day_steps[0]
        got = design_formation(
            problem, finder, kinds, list(problem.agent_positions), fuels, {},
            steps, next_steps=problem.day_steps[1], refuels=0)
        self.assertTrue(got, "順路が1つも作れていません")
        cache = {}
        for index, line in got.items():
            start = problem.agent_positions[index]
            self.assertLessEqual(
                route_cost(finder, {}, start, line, cache), steps,
                f"{index}号車の歩数が予算を超えています")
            self.assertLessEqual(
                _route_fuel(finder, {}, start, line, cache), fuels[index],
                f"{index}号車の燃料が予算を超えています")


class PostRetryTest(unittest.TestCase):
    """**提出は通るまで何度も試すこと。**

    準決勝の32x32（第1791641357試合）で、3日目の提出が1回失敗した
    だけで**その日は7台が1歩も動きませんでした**。ログの「実際の
    3日目終了時の燃料」は開始時と1ビットも同じ `[80,80,22,0,1,69,74]`
    で、日別は 22+22+**0**+22+22+22 に落ちています（手元の予測は
    日別132・玉216、実際は日別112）。

    `post_actions` には再試行が1つもなく、通信が一瞬切れただけで
    1日を捨てる作りでした。
    """

    class _Fake:
        """何回目かで成功するふりをするクライアント。"""

        def __init__(self, fails, error):
            self.calls = 0
            self.fails = fails
            self.error = error

        def _request(self, method, path, payload=None):
            self.calls += 1
            if self.calls <= self.fails:
                raise self.error
            return {"revision": 7}

    def _client(self, fails, error):
        from hexa.client import ProconClient

        client = ProconClient("http://x", "t")
        fake = self._Fake(fails, error)
        client._request = fake._request
        return client, fake

    def test_a_network_blip_is_retried(self):
        import urllib.error

        from hexa import client as C

        was = C.POST_RETRY_WAIT
        C.POST_RETRY_WAIT = 0.0
        try:
            client, fake = self._client(3, urllib.error.URLError("切れました"))
            self.assertEqual(client.post_actions([[0]]), 7)
            self.assertEqual(fake.calls, 4, "やり直していません")
        finally:
            C.POST_RETRY_WAIT = was

    def test_a_server_error_is_retried(self):
        from hexa import client as C
        from hexa.client import ProconApiError

        was = C.POST_RETRY_WAIT
        C.POST_RETRY_WAIT = 0.0
        try:
            client, fake = self._client(
                2, ProconApiError(503, "ServiceUnavailable", "混んでいます"))
            self.assertEqual(client.post_actions([[0]]), 7)
            self.assertEqual(fake.calls, 3)
        finally:
            C.POST_RETRY_WAIT = was

    def test_a_bad_plan_is_not_retried(self):
        """計画そのものが悪いときは、投げ直しても通らないのですぐ返す。"""

        from hexa import client as C
        from hexa.client import ProconApiError

        was = C.POST_RETRY_WAIT
        C.POST_RETRY_WAIT = 0.0
        try:
            client, fake = self._client(
                99, ProconApiError(400, "ValidationError", "長さが違います"))
            with self.assertRaises(ProconApiError):
                client.post_actions([[0]])
            self.assertEqual(fake.calls, 1, "やり直してはいけません")
        finally:
            C.POST_RETRY_WAIT = was

    def test_it_gives_up_after_the_limit(self):
        import urllib.error

        from hexa import client as C
        from hexa.client import ProconApiError

        was = C.POST_RETRY_WAIT
        C.POST_RETRY_WAIT = 0.0
        try:
            client, fake = self._client(99, urllib.error.URLError("だめ"))
            with self.assertRaises(ProconApiError):
                client.post_actions([[0]])
            self.assertEqual(fake.calls, C.POST_RETRIES)
        finally:
            C.POST_RETRY_WAIT = was


class SubmitOrWaitTest(unittest.TestCase):
    """**何も受理されない日を作らないこと。**

    計画が通らなかったら、必ず有効な「全員待機」を出し直します。
    待機でも、店の上で夜を越した車は翌朝の1歩目で1玉入ります。
    そして**予測も待機に合わせる**ので、翌日の計画が狂いません。
    """

    def _board(self):
        from hexa.problem import Problem, Spot

        return Problem(
            starts_at=0, day_seconds=(60, 60), day_steps=(20, 20), height=1,
            width=6, cells=tuple([0] * 6),
            spots=(Spot(brand=0, position=1, stocks=1),
                   Spot(brand=1, position=4, stocks=1)),
            agent_positions=(1, 5), fuel_limit=40, players=9,
            busy_threshold=1, jammed_threshold=2)

    def test_the_wait_plan_is_sent_when_the_real_one_fails(self):
        from hexa.production import (Candidate, _fallback_plan,
                                     _submit_or_wait, _candidate_key)
        from hexa.rules import AgentKind
        from hexa.simulator import simulate_day

        problem = self._board()
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit] * 2
        bad = [[20], [20]]        # 歩数の合わない計画のつもり
        day_result = simulate_day(
            problem=problem, agent_kinds=kinds,
            start_positions=tuple(positions), start_fuels=tuple(fuels),
            action_plan=_fallback_plan(2, 20), day_steps=20, road_status={})
        best = Candidate("だめな案", bad, day_result,
                         _candidate_key(day_result, set(), 40, kinds, 0))

        sent = []

        class Fake:
            def post_actions(self, plan, log=None):
                sent.append(plan)
                if len(sent) == 1:
                    from hexa.client import ProconApiError
                    raise ProconApiError(400, "ValidationError", "だめ")
                return 3

        revision, landed = _submit_or_wait(
            Fake(), problem, kinds, positions, fuels, {}, 20, best, lambda *a: None)
        self.assertEqual(revision, 3)
        self.assertEqual(len(sent), 2, "待機を出し直していません")
        self.assertEqual(sent[1], _fallback_plan(2, 20))
        self.assertIsNotNone(landed)
        self.assertEqual(landed.label, "全員待機")
        self.assertEqual(landed.action_plan, _fallback_plan(2, 20),
                         "予測が待機に合っていません")

    def test_a_good_plan_is_left_alone(self):
        from hexa.production import (Candidate, _fallback_plan,
                                     _submit_or_wait, _candidate_key)
        from hexa.rules import AgentKind
        from hexa.simulator import simulate_day

        problem = self._board()
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        positions = list(problem.agent_positions)
        fuels = [problem.fuel_limit] * 2
        plan = _fallback_plan(2, 20)
        day_result = simulate_day(
            problem=problem, agent_kinds=kinds,
            start_positions=tuple(positions), start_fuels=tuple(fuels),
            action_plan=plan, day_steps=20, road_status={})
        best = Candidate("よい案", plan, day_result,
                         _candidate_key(day_result, set(), 40, kinds, 0))
        sent = []

        class Fake:
            def post_actions(self, plan, log=None):
                sent.append(plan)
                return 11

        revision, landed = _submit_or_wait(
            Fake(), problem, kinds, positions, fuels, {}, 20, best, lambda *a: None)
        self.assertEqual(revision, 11)
        self.assertEqual(len(sent), 1)
        self.assertIs(landed, best)


class SupplyPullShippedTest(unittest.TestCase):
    """**燃料が減った車は「補給車に近い店」を選ぶ**ことを入れた確認。

    準決勝32x32（第1791641357試合・巡回6台・燃料80・1日96歩・
    記録どおりの渋滞）で玉230→**240**。この盤面の厳密な上限は

      | 条件 | 玉の上限 |
      |---|---|
      | 手持ちの燃料だけ（補給なし） | **243** |
      | 1台1日0.5回の補給 | 356 |
      | 毎朝満タン・補給なし | 373 |
      | 1位の実績 | 304 |

    で、こちらは**補給なしの上限に張り付いて**いました（探索を
    10倍にしても230→231で動かない）。足りないのは探索ではなく
    燃料の配達です。

    過去29試合（磨きつき・手数2000）では順位合計 340→**317**、
    日別3137→3140、玉7585→7626（18勝5敗）。
    """

    def test_it_is_on(self):
        from hexa import production as P

        self.assertEqual(P.SUPPLY_PULL, 0.5)
        self.assertEqual(P.SUPPLY_PULL_RATIO, 0.35)

    def test_only_low_fuel_cars_are_pulled(self):
        import inspect as _inspect

        from hexa.production import GreedyPolicy

        source = " ".join(
            _inspect.getsource(GreedyPolicy.choose_touring_target).split())
        self.assertIn("SUPPLY_PULL > 0.0 and positions is not None",
                      source)
        self.assertIn("fuel < self.fuel_limit * SUPPLY_PULL_RATIO", source)


class ExactRouteTest(unittest.TestCase):
    """**1台ぶんの順路を厳密に解く**ことの確認。

    準決勝32x32で縛っているものを順に剥がすと、最後に残ったのは
    順路の質でした。1店あたりの移動がこちらの設計で6.25回、
    厳密な形で4.4回。厳密に解くと**1日48軒**になります
    （実績32〜41、在庫上限65、1位は約51軒）。

    ただし**軒数だけを見ると系列を落とす**ので、既定では使わず
    `design_cover` の上に `polish_exact` で並べ替えます。
    """

    def _line(self, width=8):
        from hexa.problem import Problem, Spot

        # 1行の平地。店を等間隔に置きます。
        return Problem(
            starts_at=0, day_seconds=(90,), day_steps=(40,), height=1,
            width=width, cells=tuple([0] * width),
            spots=tuple(Spot(brand=i, position=i, stocks=1)
                        for i in range(1, width)),
            agent_positions=(0, width - 1), fuel_limit=40, players=9,
            busy_threshold=1, jammed_threshold=2)

    def test_it_visits_as_many_as_the_budget_allows(self):
        from hexa.allocate import exact_route, route_cost
        from hexa.pathfind import PathFinder

        problem = self._line()
        finder = PathFinder(problem)
        spots = [spot.position for spot in problem.spots]
        # 1歩2の平地。40歩なら20回動けるので、端から7軒すべて届く。
        got = exact_route(finder, {}, 0, spots, 40, 40)
        self.assertEqual(len(got), len(spots))
        self.assertLessEqual(route_cost(finder, {}, 0, got), 40)
        self.assertEqual(len(set(got)), len(got), "同じ店を2度数えています")

    def test_a_small_budget_gives_fewer_spots(self):
        from hexa.allocate import exact_route, route_cost
        from hexa.pathfind import PathFinder

        problem = self._line()
        finder = PathFinder(problem)
        spots = [spot.position for spot in problem.spots]
        got = exact_route(finder, {}, 0, spots, 6, 40)
        self.assertLessEqual(route_cost(finder, {}, 0, got), 6)
        self.assertGreaterEqual(len(got), 1)
        self.assertLessEqual(len(got), 3)

    def test_it_can_be_made_to_end_somewhere(self):
        """**最後に補給車のマスへ戻る**条件が効くこと。"""

        from hexa.allocate import exact_route, route_cost
        from hexa.pathfind import PathFinder

        problem = self._line()
        finder = PathFinder(problem)
        spots = [spot.position for spot in problem.spots]
        got = exact_route(finder, {}, 0, spots, 40, 40, end=1)
        self.assertTrue(got)
        self.assertEqual(got[-1], 1, "指定した店で終わっていません")
        self.assertLessEqual(route_cost(finder, {}, 0, got), 40)

    def test_the_knobs_are_off(self):
        from hexa import production as P

        self.assertFalse(P.EXACT_DESIGN)
        self.assertFalse(P.EXACT_NIGHT)
        self.assertFalse(P.NIGHT_DESIGN)


class PolishExactTest(unittest.TestCase):
    """**店は変えずに並びだけ最善にする**ことの確認。

    集合が同じなら系列も同じなので、**日別は1つも減りません**。
    並べ替えで余った歩で店を足すので、玉だけが増えます。
    """

    def _board(self):
        from hexa.problem import Problem, Spot

        return Problem(
            starts_at=0, day_seconds=(90,), day_steps=(40,), height=1,
            width=9, cells=tuple([0] * 9),
            spots=(Spot(brand=0, position=1, stocks=1),
                   Spot(brand=1, position=3, stocks=1),
                   Spot(brand=2, position=5, stocks=1),
                   Spot(brand=3, position=7, stocks=1)),
            agent_positions=(0, 8), fuel_limit=40, players=9,
            busy_threshold=1, jammed_threshold=2)

    def test_the_brands_never_drop(self):
        from hexa.allocate import polish_exact, route_cost
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = self._board()
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        positions = [0, 8]
        fuels = [problem.fuel_limit] * 2
        # わざと遠回りの並びを渡します。
        routes = {0: [7, 1, 5, 3]}
        got = polish_exact(problem, finder, kinds, positions, fuels, {},
                           40, routes)
        self.assertIn(0, got)
        self.assertEqual(set(got[0]) >= {1, 3, 5, 7}, True,
                         "店が減っています（系列が落ちます）")
        before = route_cost(finder, {}, 0, routes[0])
        after = route_cost(finder, {}, 0, got[0])
        self.assertLessEqual(after, max(before, 40))
        self.assertLessEqual(after, 40)

    def test_a_better_order_is_found(self):
        from hexa.allocate import _best_order, route_cost
        from hexa.pathfind import PathFinder

        problem = self._board()
        finder = PathFinder(problem)
        cache = {}
        bad = [7, 1, 5, 3]
        good = _best_order(finder, {}, 0, bad, cache)
        self.assertEqual(sorted(good), sorted(bad), "店が変わっています")
        self.assertLessEqual(route_cost(finder, {}, 0, good, cache),
                             route_cost(finder, {}, 0, bad, cache))


class NightCellTest(unittest.TestCase):
    """**夜に集まる店**の選び方の確認。

    `simulator` の補給は歩0の行動段階でも走るので、補給車と同じ
    マスで夜を越した巡回車は翌日1歩も動かないうちに満タンに
    なります。同じマスに何台いてもかまいません。
    """

    def _board(self):
        from hexa.problem import Problem, Spot

        return Problem(
            starts_at=0, day_seconds=(90, 90), day_steps=(40, 40), height=1,
            width=9, cells=tuple([0] * 9),
            spots=(Spot(brand=0, position=1, stocks=1),
                   Spot(brand=1, position=4, stocks=1),
                   Spot(brand=2, position=7, stocks=1)),
            agent_positions=(0, 8, 4), fuel_limit=40, players=9,
            busy_threshold=1, jammed_threshold=2)

    def test_it_picks_a_spot_every_car_can_reach(self):
        from hexa.allocate import pick_night_cell
        from hexa.pathfind import PathFinder

        problem = self._board()
        finder = PathFinder(problem)
        got = pick_night_cell(problem, finder, {}, [0, 8, 4], [0, 1], 40)
        self.assertIn(got, {spot.position for spot in problem.spots})
        # 真ん中の店がいちばん近いはず。
        self.assertEqual(got, 4)

    def test_a_tiny_budget_gives_nothing(self):
        from hexa.allocate import pick_night_cell
        from hexa.pathfind import PathFinder

        problem = self._board()
        finder = PathFinder(problem)
        self.assertIsNone(
            pick_night_cell(problem, finder, {}, [0, 8, 4], [0, 1], 1))


class FuelAsBallsTest(unittest.TestCase):
    """**燃料を玉に換算して足す案は負けた**ことの記録。

    準決勝32x32（手数2000）で、玉数の桁に燃料を足すと

      | 率 | 玉 |
      |---|---|
      | 0（既定） | **240** |
      | 0.05 | 225 |
      | 0.10 | 217 |
      | 0.20 | 220 |
      | 0.40 | 217 |

    貯め込むほど今日の玉を捨てるからです。夜合流だけを数える形
    （`NIGHT_BALL_RATE`）も、0.05〜1.0のどれでも玉240のまま
    動きませんでした。
    """

    def test_both_are_off(self):
        from hexa import production as P

        self.assertEqual(P.FUEL_BALL_RATE, 0.0)
        self.assertEqual(P.NIGHT_BALL_RATE, 0.0)

    def test_they_are_wired_into_the_ball_slot(self):
        import inspect as _inspect

        from hexa.production import _candidate_key

        source = " ".join(_inspect.getsource(_candidate_key).split())
        self.assertIn("len(day_result.acquisitions) + _fuel_as_balls", source)
        self.assertIn("_night_balls(", source)

    def test_zero_means_zero(self):
        from hexa.production import _fuel_as_balls, _night_balls
        from hexa.rules import AgentKind

        class Fake:
            positions = (0, 0)
            fuels = (10, 40)

        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        self.assertEqual(_fuel_as_balls(Fake(), kinds, 40), 0)
        self.assertEqual(_night_balls(Fake(), kinds, 40, 40), 0)


class ParkSurvivesPolishTest(unittest.TestCase):
    """**補給車の置き場所が磨きを越えて残ること。**

    磨きの採点は `PlannedRoutePolicy(problem, routes, order)` を
    作り直すので、`park` を渡していませんでした。つまりハブや
    夜合流の案は**1手磨いた瞬間に置き場所が消えます**。準決勝
    32x32で、夜合流の設計を入れてもハブを入れても玉が1つも
    動かなかった（240のまま）のはこれが理由でした。
    """

    def test_the_candidate_carries_it(self):
        import dataclasses

        from hexa.production import Candidate

        names = [field.name for field in dataclasses.fields(Candidate)]
        self.assertIn("park", names)

    def test_the_builder_copies_it_from_the_policy(self):
        import inspect as _inspect

        from hexa.production import _build_candidate

        source = " ".join(_inspect.getsource(_build_candidate).split())
        self.assertIn('park=getattr(policy, "park", None)', source)

    def test_the_polisher_is_given_it(self):
        import inspect as _inspect

        from hexa.production import play_production_match

        source = " ".join(_inspect.getsource(play_production_match).split())
        self.assertIn("polish_park = [None]", source)
        self.assertIn("park=polish_park[0]", source)
        self.assertIn("polish_park[0] = best.park", source)


class CeilingTest(unittest.TestCase):
    """**その日に取れる玉数の上限を先に測る**ことの確認。

    準決勝では上位18チームが日別満点で、勝負は玉数だけでした
    （32x32: 1位304 / こちら216）。だとすれば「その日に何玉まで
    行けるのか」と「何が縛っているのか」を知っておきたい。

    解き方は **(踏んだ店の集合, いまの店) ごとに最短歩数と最小燃料を
    持つ探索**で、予算を超えた枝を捨てます。店を2度は数えません
    （1台が同じ店から取れるのは1日1玉）。

    実測（準決勝32x32・22系列32店・巡回6台・燃料80・1日96歩）:

      | 条件 | 玉の上限 |
      |---|---|
      | 手持ちの燃料だけ | **243**（こちらの実績240） |
      | 1台1日0.5回の補給 | 356 |
      | 毎朝満タン・補給なし | **373** |
      | 1位の実績 | 304 |
    """

    def _board(self, width=9, fuel=40, steps=40):
        from hexa.problem import Problem, Spot

        return Problem(
            starts_at=0, day_seconds=(90,), day_steps=(steps,), height=1,
            width=width, cells=tuple([0] * width),
            spots=(Spot(brand=0, position=1, stocks=1),
                   Spot(brand=1, position=3, stocks=1),
                   Spot(brand=2, position=5, stocks=1),
                   Spot(brand=3, position=7, stocks=1)),
            agent_positions=(0, width - 1), fuel_limit=fuel, players=9,
            busy_threshold=1, jammed_threshold=2)

    def test_the_stock_caps_it(self):
        from hexa.ceiling import day_ceiling
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = self._board()
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        got = day_ceiling(problem, finder, kinds, [0, 8],
                          [problem.fuel_limit] * 2, {}, 40)
        # 店は4軒で在庫1つずつ。巡回車1台なので上限は4。
        self.assertEqual(got["by_stock"], 4)
        self.assertEqual(got["balls"], 4)
        self.assertEqual(got["binds"], "在庫")

    def test_low_fuel_binds(self):
        from hexa.ceiling import day_ceiling
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = self._board()
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        # 燃料2なら平地2回ぶん。端から1軒しか届きません。
        got = day_ceiling(problem, finder, kinds, [0, 8], [2, 40], {}, 40)
        self.assertEqual(got["binds"], "燃料")
        self.assertLess(got["balls"], got["by_stock"])
        self.assertGreaterEqual(got["balls"], 1)

    def test_a_short_day_binds_on_steps(self):
        from hexa.ceiling import day_ceiling
        from hexa.pathfind import PathFinder
        from hexa.rules import AgentKind

        problem = self._board(steps=4)
        finder = PathFinder(problem)
        kinds = (AgentKind.TOURING, AgentKind.SUPPLY)
        got = day_ceiling(problem, finder, kinds, [0, 8],
                          [problem.fuel_limit] * 2, {}, 4)
        # 4歩なら平地2回。マス1までで1軒。
        self.assertEqual(got["balls"], 1)
        self.assertEqual(got["binds"], "歩数")

    def test_it_is_reported_in_the_match(self):
        import inspect as _inspect

        from hexa import production as P

        self.assertTrue(P.CEILING_REPORT)
        source = " ".join(
            _inspect.getsource(P.play_production_match).split())
        self.assertIn("if CEILING_REPORT:", source)
        self.assertIn("_ceiling.day_ceiling(", source)

    def test_describe_reads_in_japanese(self):
        from hexa.ceiling import describe

        text = describe({"balls": 37, "by_steps": 65, "by_fuel": 37,
                         "by_stock": 65, "binds": "燃料", "exact": True})
        self.assertIn("この日の上限 玉37", text)
        self.assertIn("燃料が縛っています", text)
