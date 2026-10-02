"""公式の簡易版サーバーとの照合テスト。

`server/` に公式配布のバイナリが置かれているときだけ走ります。
置かれていなければスキップします（バイナリは配布物なので同梱していません）。

**このテストは実時間で1試合ぶん待ちます。** 公式サーバーは各日の回答時間を
実際に経過させてから次の日を配るので、日数×回答時間ぶんはどうしてもかかります。
止まっているように見えないよう、日ごとに進み具合を出しています。
"""

import dataclasses
import sys
import time
import unittest
from pathlib import Path

from hexa.ai import new_population
from hexa.generator import GeneratorSettings, generate_problem
from hexa.localserver import find_server_binary, start_server
from hexa.verify import verify_against_server

BASE_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = BASE_DIR / "server"


def _note(message: str) -> None:
    """テスト実行中でも見える進捗表示。止まったように見えないようにする。"""

    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _binary():
    binary = find_server_binary(SERVER_DIR)
    if binary is None:
        return None
    try:
        binary.chmod(binary.stat().st_mode | 0o111)
    except OSError:
        pass
    return binary


SKIP_REASON = (
    "公式サーバーのバイナリが server/ にないのでスキップしました。"
    "https://www.procon.gr.jp/news/2026/08/20/117 から「簡易サーバー.zip」を落として "
    "中身を server/ に置くと、このテストが動きます（配布物なので同梱していません）。"
)


@unittest.skipIf(_binary() is None, SKIP_REASON)
class OfficialServerTest(unittest.TestCase):
    def _run(self, problem, brains, label):
        tokens = [f"token-p{index}" for index in range(len(brains))]
        expected = (
            problem.day_count * problem.day_seconds[0] + 8
        )
        _note(f"{label}: {problem.day_count}日 × 回答時間{problem.day_seconds[0]}秒 "
              f"→ およそ{expected}秒かかります")
        started = time.monotonic()
        server = start_server(
            _binary(), problem, tokens, BASE_DIR / "tmp_server_test",
            names=[brain.name for brain in brains],
            kind_deadline="3s", match_start_delay="2s",
        )
        try:
            report = verify_against_server(
                server.base_url, tokens, brains, problem,
                log=lambda message: _note("  " + message),
            )
        finally:
            server.stop()
        _note(f"  完了（{time.monotonic() - started:.0f}秒）")
        return report

    def test_positions_fuel_and_traffic_match_the_official_server(self):
        brains = new_population()
        settings = GeneratorSettings(
            height=10, width=10, agent_count=4, spot_count=8, brand_count=4,
            day_count=4, players=4, day_seconds=4,
        )
        problem = generate_problem(2026, settings)
        report = self._run(problem, brains, "位置・燃料・道路状態の照合")
        self.assertGreaterEqual(report.state_checks, 1)
        self.assertGreaterEqual(report.traffic_checks, 4)
        self.assertEqual(report.mismatches, [])

    def test_congested_roads_also_match(self):
        # 混雑と渋滞が実際に発生する設定にして、交通量の判定まで確かめる。
        brains = new_population()
        settings = GeneratorSettings(
            height=8, width=8, agent_count=6, spot_count=8, brand_count=4,
            day_count=5, players=4, day_seconds=4, road_ratio=0.35,
        )
        problem = generate_problem(31337, settings)
        problem = dataclasses.replace(problem, busy_threshold=1, jammed_threshold=2)
        report = self._run(problem, brains, "混雑・渋滞まで含めた照合")
        self.assertEqual(report.mismatches, [])


if __name__ == "__main__":
    unittest.main()
