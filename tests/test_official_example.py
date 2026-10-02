"""公式「質問回答集 補足資料(行動詳細)」p.3-p.4 の実例をそのまま再現するテスト。

この資料には、8体のエージェントについて6ステップ分の座標・燃料・うどん・滞在数が
1マスずつ表になっています。ここを完全に一致させることが、シミュレータが公式ルール
どおりであることの最も強い根拠になります。

資料の設定:
  セル0・1 は道路（セル0=順調、セル1=混雑）、セル2・3 は平地。
  セル2 に最大在庫4のスポット。巡回車の最大燃料積載量は3。1日は6ステップ。
  巡回車A～D はセル0、巡回車E と 補給車A はセル2、巡回車F・G はセル3。
"""

import unittest

from hexa.problem import Problem, Spot
from hexa.rules import AgentKind, RoadStatus
from hexa.simulator import simulate_day


def build_example_problem() -> Problem:
    # セルが横一列に 0-1-2-3 と並ぶ最小のマップ。
    # 0行目は偶数行なので、方向2(右)が col+1、方向5(左)が col-1 になる。
    return Problem(
        starts_at=0,
        day_seconds=(60,),
        day_steps=(6,),
        height=1,
        width=4,
        cells=(1, 1, 0, 0),          # 道路, 道路, 平地, 平地
        spots=(Spot(brand=0, position=2, stocks=4),),
        agent_positions=(0, 0, 0, 0, 2, 3, 3, 2),
        fuel_limit=3,
        players=1,
        busy_threshold=1,
        jammed_threshold=2,
    )


# 資料の行動計画。表の並び順が 0～7 のエージェント番号に対応する。
EXAMPLE_PLAN = [
    [2, 2, 2, -1],    # 0 巡回車A  RT, RT, RT, WT(1)
    [-1, 2, 2, 2],    # 1 巡回車B  WT(1), RT, RT, RT
    [-2, 2, 2, -1],   # 2 巡回車C  WT(2), RT, RT, WT(1)
    [-3, 2, -2],      # 3 巡回車D  WT(3), RT, WT(2)
    [5, 5, -2],       # 4 巡回車E  LF, LF, WT(2)
    [5, 5, -2],       # 5 巡回車F  LF, LF, WT(2)
    [-2, 5, 5],       # 6 巡回車G  WT(2), LF, LF
    [5, 5, -2],       # 7 補給車A  LF, LF, WT(2)
]

EXAMPLE_KINDS = (
    AgentKind.TOURING,
    AgentKind.TOURING,
    AgentKind.TOURING,
    AgentKind.TOURING,
    AgentKind.TOURING,
    AgentKind.TOURING,
    AgentKind.TOURING,
    AgentKind.SUPPLY,
)


class OfficialExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        problem = build_example_problem()
        self.problem = problem
        self.result = simulate_day(
            problem=problem,
            agent_kinds=EXAMPLE_KINDS,
            start_positions=problem.agent_positions,
            start_fuels=tuple(
                problem.fuel_limit if kind == AgentKind.TOURING else 0
                for kind in EXAMPLE_KINDS
            ),
            action_plan=[list(actions) for actions in EXAMPLE_PLAN],
            day_steps=6,
            road_status={0: RoadStatus.SMOOTH, 1: RoadStatus.BUSY},
            record_trace=True,
        )

    def test_final_positions_match_the_official_table(self) -> None:
        # 表の「最終」列の座標
        self.assertEqual(self.result.positions, [3, 3, 2, 1, 0, 1, 1, 0])

    def test_final_fuel_matches_the_official_table(self) -> None:
        # 表の「最終」列の燃料。補給車は燃料を消費しないので初期値のまま。
        self.assertEqual(self.result.fuels[:7], [0, 0, 1, 1, 3, 1, 1])

    def test_acquisition_order_matches_the_official_notes(self) -> None:
        # 巡回車E(step1) → F(step2) → A(step3) → B(step4) の順で在庫4を使い切る。
        events = [(event.step, event.agent_index) for event in self.result.acquisitions]
        self.assertEqual(events, [(1, 4), (2, 5), (3, 0), (4, 1)])

    def test_g_loses_the_tie_break_to_b(self) -> None:
        # 資料の注記: 巡回車Bと巡回車Gが同時到着したが、在庫1なのでID順でBのみ獲得。
        winners = {event.agent_index for event in self.result.acquisitions}
        self.assertIn(1, winners)
        self.assertNotIn(6, winners)

    def test_c_arrives_after_the_stock_is_gone(self) -> None:
        # 資料の注記: 巡回車Cはスポットに到達するが在庫がないので獲得できない。
        self.assertEqual(self.result.positions[2], 2)
        self.assertNotIn(2, {event.agent_index for event in self.result.acquisitions})

    def test_e_is_always_refuelled_by_the_supply_car(self) -> None:
        # 資料の注記: 巡回車Eは補給車と同時に移動するので燃料は常に最大積載量。
        self.assertEqual(self.result.fuels[4], self.problem.fuel_limit)
        refuelled = {event.agent_index for event in self.result.refuels}
        self.assertIn(4, refuelled)

    def test_f_is_never_refuelled(self) -> None:
        # 資料の注記: 巡回車Fは常に補給車の1セル後ろなので補給されない。
        refuelled = {event.agent_index for event in self.result.refuels}
        self.assertNotIn(5, refuelled)

    def test_stay_counts_match_the_official_table(self) -> None:
        # 表の「滞在数」行の最終値。合計は 8体 × 6ステップ = 48 になる。
        self.assertEqual(self.result.stay_counts, {0: 12, 1: 17, 2: 12, 3: 7})
        self.assertEqual(sum(self.result.stay_counts.values()), 48)


if __name__ == "__main__":
    unittest.main()
