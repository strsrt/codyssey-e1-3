import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import main


class MatrixAndMacTests(unittest.TestCase):
    def test_matrix_stores_and_reads_indexed_values(self):
        matrix = main.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        matrix.set_value(1, 2, 10.5)
        self.assertEqual(matrix.get_value(1, 2), 10.5)
        self.assertEqual(matrix._values, [1, 2, 3, 4, 5, 10.5, 7, 8, 9])
        self.assertEqual(matrix.to_2d()[1][2], 10.5)

    def test_matrix_expands_compact_fill_and_cell_overrides(self):
        matrix = main.Matrix({
            "size": 3,
            "fill": 0.0,
            "overrides": [[1, 2, 5e-10]],
        })
        self.assertEqual(matrix.get_value(0, 0), 0.0)
        self.assertEqual(matrix.get_value(1, 2), 5e-10)
        self.assertEqual(matrix.to_2d()[1][2], 5e-10)

    def test_matrix_rejects_invalid_compact_override(self):
        with self.assertRaisesRegex(ValueError, "범위를 벗어났습니다"):
            main.Matrix({
                "size": 2,
                "fill": 0.0,
                "overrides": [[2, 0, 1.0]],
            })

    def test_matrix_supports_required_sizes(self):
        for size in (3, 5, 13, 25):
            matrix = main.Matrix([[0] * size for _ in range(size)])
            self.assertEqual(matrix.size, size)

    def test_matrix_rejects_non_square_or_non_numeric_values(self):
        with self.assertRaisesRegex(ValueError, "정사각형"):
            main.Matrix([[1, 2], [3]])
        with self.assertRaisesRegex(ValueError, "유한한 숫자"):
            main.Matrix([[float("inf")]])

    def test_mac_calculates_sum_with_loops(self):
        pattern = main.Matrix([[1, 2], [3, 4]])
        filter_matrix = main.Matrix([[4, 3], [2, 1]])
        score = main.mac(pattern, filter_matrix)
        self.assertEqual(score, 20.0)
        self.assertIsInstance(score, float)
        self.assertEqual(
            main.mac_2d(pattern.to_2d(), filter_matrix.to_2d()),
            main.mac(pattern, filter_matrix),
        )

    def test_mac_rejects_different_dimensions(self):
        with self.assertRaisesRegex(ValueError, "크기"):
            main.mac(main.Matrix([[1]]), main.Matrix([[1, 2], [3, 4]]))

    def test_decide_uses_strict_epsilon_boundary(self):
        self.assertEqual(main.decide(1.0, 1.0 + 0.5e-9), "UNDECIDED")
        self.assertEqual(main.decide(1.0, 1.0 + 2e-9), "B")

    def test_modes_format_epsilon_ties_according_to_their_policies(self):
        self.assertEqual(main.decide_user_mode(1.0, 1.0), "판정 불가")
        self.assertEqual(main.decide_json_mode(1.0, 1.0), "UNDECIDED")
        self.assertEqual(main.decide_user_mode(2.0, 1.0), "A")
        self.assertEqual(main.decide_json_mode(2.0, 1.0), "Cross")

    def test_normalize_label_centralizes_alias_case_and_whitespace(self):
        self.assertEqual(main.normalize_label(" + ", "expected"), "Cross")
        self.assertEqual(main.normalize_label(" cRoSs ", "filter"), "Cross")
        self.assertEqual(main.normalize_label(" X ", "expected"), "X")
        with self.assertRaisesRegex(ValueError, "정규화할 수 없습니다"):
            main.normalize_label("circle", "expected")

    def test_generator_builds_cross_and_x_for_multiple_sizes(self):
        for size in (3, 5, 13, 25):
            patterns = main.generate_patterns(size)
            center = size // 2
            self.assertEqual(patterns["Cross"].get_value(center, 0), 1)
            self.assertEqual(patterns["Cross"].get_value(0, center), 1)
            self.assertEqual(patterns["Cross"].get_value(0, 0), 0)
            self.assertEqual(patterns["X"].get_value(0, 0), 1)
            self.assertEqual(patterns["X"].get_value(0, size - 1), 1)
            self.assertEqual(patterns["X"].get_value(0, center), 0)

    def test_generator_rejects_non_positive_size(self):
        with self.assertRaisesRegex(ValueError, "양의 정수"):
            main.generate_patterns(0)


class JsonAnalysisTests(unittest.TestCase):
    def test_repository_data_uses_explicit_float_rows(self):
        data = json.loads(main.DATA_FILE.read_text(encoding="utf-8"))
        for filters in data["filters"].values():
            for matrix in filters.values():
                self.assertIsInstance(matrix, list)
                self.assertIsInstance(matrix[0], list)
                self.assertIsInstance(matrix[0][0], float)
        epsilon_input = data["patterns"]["size_5_2"]["input"]
        self.assertIsInstance(epsilon_input, list)
        self.assertEqual(epsilon_input[2][2], 5e-10)

    def test_repository_data_has_eight_cases_and_three_intentional_failures(self):
        report = main.analyze_json_data(main.DATA_FILE)
        self.assertEqual(report["total"], 8)
        self.assertEqual(report["passed"], 5)
        self.assertEqual(
            [case["id"] for case in report["failed"]],
            ["size_5_2", "size_13_2", "size_25_2"],
        )
        self.assertTrue(all("reason" not in case for case in report["failed"]))
        epsilon_case = next(
            case for case in report["failed"] if case["id"] == "size_5_2"
        )
        self.assertEqual(epsilon_case["actual"], "UNDECIDED")
        self.assertLess(
            abs(epsilon_case["cross_score"] - epsilon_case["x_score"]),
            main.TIE_EPSILON,
        )

    def make_data(self):
        return {
            "filters": {
                "size_3": {
                    "cross": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "x": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                }
            },
            "patterns": {
                "size_3_1": {
                    "input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "expected": "+",
                },
                "size_3_2": {
                    "input": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                    "expected": "x",
                },
            },
        }

    def write_data(self, directory, data):
        path = Path(directory) / "data.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_normalizes_filter_and_expected_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_data(directory, self.make_data())
            report = main.analyze_json_data(path)
        self.assertEqual((report["total"], report["passed"]), (2, 2))
        self.assertEqual(
            [result["actual"] for result in report["results"]],
            ["Cross", "X"],
        )

    def test_dimension_mismatch_fails_only_the_related_case(self):
        data = self.make_data()
        data["patterns"]["size_3_1"]["input"] = [[1, 2], [3, 4]]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_data(directory, data)
            report = main.analyze_json_data(path)
        self.assertEqual((report["total"], report["passed"]), (2, 1))
        self.assertIn("크기", report["failed"][0]["reason"])

    def test_filter_pattern_size_mismatch_is_a_case_failure(self):
        data = self.make_data()
        data["filters"]["size_3"]["cross"] = [[1, 0], [0, 1]]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_data(directory, data)
            report = main.analyze_json_data(path)
        self.assertEqual((report["total"], report["passed"]), (2, 0))
        self.assertIn("필터 크기", report["failed"][0]["reason"])

    def test_filter_label_and_missing_size_are_case_failures(self):
        data = self.make_data()
        del data["filters"]["size_3"]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_data(directory, data)
            report = main.analyze_json_data(path)
        self.assertEqual((report["total"], report["passed"]), (2, 0))
        self.assertTrue(all("filters에 size_3" in item["reason"] for item in report["failed"]))

    def test_json_mode_prints_failure_reason_and_summary(self):
        data = self.make_data()
        data["patterns"]["size_3_1"]["input"] = [[1]]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_data(directory, data)
            output = StringIO()
            with patch.object(main, "DATA_FILE", path), \
                    patch.object(main, "run_performance_analysis"), \
                    redirect_stdout(output):
                main.run_json_mode()
        self.assertIn("FAIL -", output.getvalue())
        self.assertIn("전체 테스트: 2 / 통과: 1 / 실패: 1", output.getvalue())


class InputAndPerformanceTests(unittest.TestCase):
    def test_application_handles_keyboard_interrupt(self):
        output = StringIO()
        with patch.object(main, "main", side_effect=KeyboardInterrupt), \
                redirect_stdout(output):
            main.run_application()
        self.assertIn("Ctrl+C", output.getvalue())

    def test_application_handles_end_of_input(self):
        output = StringIO()
        with patch.object(main, "main", side_effect=EOFError), \
                redirect_stdout(output):
            main.run_application()
        self.assertIn("입력이 종료", output.getvalue())

    def test_application_reports_unexpected_errors(self):
        output = StringIO()
        with patch.object(main, "main", side_effect=RuntimeError("test failure")), \
                redirect_stdout(output):
            main.run_application()
        self.assertIn("RuntimeError: test failure", output.getvalue())

    def test_read_matrix_retries_incorrect_row_length_and_bad_number(self):
        inputs = iter(["1 2", "a 2 3", "1 2 3", "4 5 6", "7 8 9"])
        output = StringIO()
        with patch("builtins.input", side_effect=lambda _: next(inputs)), \
                redirect_stdout(output):
            matrix = main.read_matrix(3, "패턴")
        self.assertEqual(matrix.get_value(2, 2), 9.0)
        self.assertEqual(output.getvalue().count(main.INPUT_FORMAT_ERROR), 2)

    def test_performance_compares_same_data_with_equal_repetitions(self):
        self.assertEqual(main.PERFORMANCE_REPETITIONS, 10)
        original_mac = main.mac
        original_mac_2d = main.mac_2d
        flat_calls = []
        two_d_calls = []

        def counting_mac(pattern, filter_matrix):
            flat_calls.append((pattern.to_2d(), filter_matrix.to_2d()))
            return original_mac(pattern, filter_matrix)

        def counting_mac_2d(pattern, filter_matrix):
            two_d_calls.append((pattern, filter_matrix))
            return original_mac_2d(pattern, filter_matrix)

        with patch.object(main, "mac", side_effect=counting_mac), \
                patch.object(main, "mac_2d", side_effect=counting_mac_2d):
            main.run_performance_analysis((3, 5))
        self.assertEqual(len(flat_calls), 2 * main.PERFORMANCE_REPETITIONS)
        self.assertEqual(len(two_d_calls), 2 * main.PERFORMANCE_REPETITIONS)
        for flat_input, two_d_input in zip(flat_calls, two_d_calls):
            self.assertEqual(flat_input, two_d_input)


if __name__ == "__main__":
    unittest.main()
