import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


DATA_FILE = Path(__file__).with_name("data.json")
SIZES = (3, 5, 13, 25)
PERFORMANCE_REPETITIONS = 1000
TIE_EPSILON = 1e-9
CASE_KEY = re.compile(r"^size_(\d+)_(.+)$")
INPUT_FORMAT_ERROR = (
    "입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요."
)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


class Matrix:
    """Square numeric matrix stored in row-major one-dimensional form."""

    def __init__(self, values: Any):
        if not isinstance(values, list) or not values:
            raise ValueError("행렬은 비어 있지 않은 2차원 배열이어야 합니다.")
        if any(not isinstance(row, list) for row in values):
            raise ValueError("행렬의 각 행은 배열이어야 합니다.")
        size = len(values)
        if any(len(row) != size for row in values):
            raise ValueError("행렬은 행과 열의 크기가 같은 정사각형이어야 합니다.")
        if any(not _is_number(value) for row in values for value in row):
            raise ValueError("행렬 값은 유한한 숫자여야 합니다.")
        self.size = size
        self._values = [value for row in values for value in row]

    def _check_index(self, row: int, column: int) -> None:
        if not (0 <= row < self.size and 0 <= column < self.size):
            raise IndexError("행렬 인덱스가 범위를 벗어났습니다.")

    def get_value(self, row: int, column: int) -> Union[int, float]:
        self._check_index(row, column)
        return self._values[row * self.size + column]

    def set_value(
        self, row: int, column: int, value: Union[int, float]
    ) -> None:
        self._check_index(row, column)
        if not _is_number(value):
            raise ValueError("행렬 값은 유한한 숫자여야 합니다.")
        self._values[row * self.size + column] = value

    def to_2d(self) -> List[List[Union[int, float]]]:
        return [
            self._values[row * self.size:(row + 1) * self.size]
            for row in range(self.size)
        ]


def generate_patterns(size: int) -> Dict[str, Matrix]:
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("패턴 크기는 양의 정수여야 합니다.")
    center = size // 2
    cross = []
    x_pattern = []
    for row in range(size):
        cross_row = []
        x_row = []
        for column in range(size):
            cross_row.append(1 if row == center or column == center else 0)
            x_row.append(1 if row == column or row + column == size - 1 else 0)
        cross.append(cross_row)
        x_pattern.append(x_row)
    return {"Cross": Matrix(cross), "X": Matrix(x_pattern)}


def mac(pattern: Matrix, filter_matrix: Matrix) -> float:
    if not isinstance(pattern, Matrix) or not isinstance(filter_matrix, Matrix):
        raise TypeError("MAC 입력은 Matrix 객체여야 합니다.")
    if pattern.size != filter_matrix.size:
        raise ValueError(
            f"패턴 크기({pattern.size})와 필터 크기({filter_matrix.size})가 다릅니다."
        )

    score = 0.0
    pattern_values = pattern._values
    filter_values = filter_matrix._values
    for pattern_value, filter_value in zip(pattern_values, filter_values):
        score += pattern_value * filter_value
    return score


def mac_2d(
    pattern: List[List[Union[int, float]]],
    filter_matrix: List[List[Union[int, float]]],
) -> float:
    if len(pattern) != len(filter_matrix):
        raise ValueError("패턴과 필터의 크기가 다릅니다.")
    score = 0.0
    for row in range(len(pattern)):
        for column in range(len(pattern)):
            score += pattern[row][column] * filter_matrix[row][column]
    return score


def normalize_label(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} 라벨은 문자열이어야 합니다.")
    label = value.strip().lower()
    if label in ("cross", "+"):
        return "Cross"
    if label == "x":
        return "X"
    raise ValueError(f"{field}의 라벨 '{value}'을(를) Cross 또는 X로 정규화할 수 없습니다.")


def decide(score_a: float, score_b: float) -> str:
    if abs(score_a - score_b) < TIE_EPSILON:
        return "UNDECIDED"
    return "A" if score_a > score_b else "B"


def read_matrix(size: int, label: str) -> Matrix:
    print(f"{label} 행렬을 한 줄씩 입력하세요. (각 행 {size}개 숫자, 공백 구분)")
    values = []
    while len(values) < size:
        raw = input(f"  {len(values) + 1}/{size}행: ").strip()
        try:
            row = [float(token) for token in raw.split()]
        except ValueError:
            print(INPUT_FORMAT_ERROR)
            continue
        if len(row) != size or any(not math.isfinite(value) for value in row):
            print(INPUT_FORMAT_ERROR)
            continue
        values.append(row)
    return Matrix(values)


def run_user_mode() -> None:
    print("\n[사용자 입력 모드]")
    print("필터 입력 방식: 1. 직접 입력  2. Cross/X 자동 생성")
    filter_choice = read_choice("선택 (1/2): ", ("1", "2"))
    if filter_choice == "1":
        filter_a = read_matrix(3, "필터 A")
        filter_b = read_matrix(3, "필터 B")
    else:
        generated = generate_patterns(3)
        filter_a = generated["Cross"]
        filter_b = generated["X"]
        print("필터 A=Cross, 필터 B=X 패턴을 생성해 저장했습니다.")
    print("필터 A와 B를 3x3 행렬로 저장했습니다.")
    print("패턴 입력 방식: 1. 직접 입력  2. Cross/X 자동 생성")
    pattern_choice = read_choice("선택 (1/2): ", ("1", "2"))
    if pattern_choice == "1":
        pattern = read_matrix(3, "패턴")
    else:
        pattern_label = read_choice("생성할 패턴 (Cross/X): ", ("cross", "x"))
        generated_pattern = generate_patterns(3)
        pattern = generated_pattern["Cross" if pattern_label == "cross" else "X"]
        print(f"{'Cross' if pattern_label == 'cross' else 'X'} 패턴을 생성했습니다.")

    score_a, time_a = timed_mac(pattern, filter_a)
    score_b, time_b = timed_mac(pattern, filter_b)
    decision = decide(score_a, score_b)
    display = {"A": "A", "B": "B", "UNDECIDED": "판정 불가"}[decision]
    print("\n[3x3 판정]")
    print(f"필터 A 점수: {score_a:g}, 연산 시간: {time_a:.6f} ms")
    print(f"필터 B 점수: {score_b:g}, 연산 시간: {time_b:.6f} ms")
    print(f"판정: {display}")
    run_performance_analysis((3,))


def read_choice(prompt: str, choices: Tuple[str, ...]) -> str:
    while True:
        choice = input(prompt).strip().lower()
        if choice in choices:
            return choice
        print(f"입력 형식 오류: {', '.join(choices)} 중 하나를 입력하세요.")


def timed_mac(pattern: Matrix, filter_matrix: Matrix) -> Tuple[float, float]:
    start = time.perf_counter_ns()
    score = mac(pattern, filter_matrix)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    return score, elapsed_ms


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"data.json 파일을 찾을 수 없습니다: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON 형식 오류 (줄 {error.lineno}, 열 {error.colno}): {error.msg}"
        ) from error


def _normalize_filter_map(raw_filters: Any, size: int) -> Dict[str, Matrix]:
    if not isinstance(raw_filters, dict):
        raise ValueError(f"filters.size_{size}는 라벨-행렬 객체여야 합니다.")
    normalized = {}
    for raw_label, raw_matrix in raw_filters.items():
        label = normalize_label(raw_label, f"size_{size} 필터 키")
        if label in normalized:
            raise ValueError(f"size_{size}에 중복된 {label} 필터 키가 있습니다.")
        normalized[label] = Matrix(raw_matrix)
    for required in ("Cross", "X"):
        if required not in normalized:
            raise ValueError(f"size_{size} 필터에 {required} 키가 없습니다.")
    return normalized


def _analyze_case(
    case_id: str, raw_case: Any, raw_filters: Any
) -> Dict[str, Any]:
    key_match = CASE_KEY.fullmatch(case_id)
    if not key_match:
        raise ValueError("케이스 키는 size_{N}_{idx} 형식이어야 합니다.")
    size = int(key_match.group(1))
    if not isinstance(raw_case, dict):
        raise ValueError("패턴 항목은 input과 expected를 포함한 객체여야 합니다.")
    if "input" not in raw_case or "expected" not in raw_case:
        raise ValueError("패턴 항목에 input 또는 expected가 없습니다.")

    expected = normalize_label(raw_case["expected"], f"{case_id} expected")
    pattern = Matrix(raw_case["input"])
    if pattern.size != size:
        raise ValueError(
            f"키의 크기({size})와 패턴 크기({pattern.size})가 다릅니다."
        )
    if not isinstance(raw_filters, dict):
        raise ValueError("최상위 filters는 객체여야 합니다.")
    filter_key = f"size_{size}"
    if filter_key not in raw_filters:
        raise ValueError(f"filters에 {filter_key} 항목이 없습니다.")

    filters = _normalize_filter_map(raw_filters[filter_key], size)
    for label, filter_matrix in filters.items():
        if filter_matrix.size != pattern.size:
            raise ValueError(
                f"{label} 필터 크기({filter_matrix.size})와 "
                f"패턴 크기({pattern.size})가 다릅니다."
            )

    cross_score = mac(pattern, filters["Cross"])
    x_score = mac(pattern, filters["X"])
    winner = decide(cross_score, x_score)
    actual = "Cross" if winner == "A" else "X" if winner == "B" else winner
    return {
        "id": case_id,
        "cross_score": cross_score,
        "x_score": x_score,
        "actual": actual,
        "expected": expected,
        "passed": actual == expected,
    }


def analyze_json_data(path: Path = DATA_FILE) -> Dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 값은 객체여야 합니다.")
    raw_filters = data.get("filters")
    raw_patterns = data.get("patterns")
    if not isinstance(raw_filters, dict):
        raise ValueError("JSON에 filters 객체가 없습니다.")
    if not isinstance(raw_patterns, dict):
        raise ValueError("JSON에 patterns 객체가 없습니다.")
    if not raw_patterns:
        raise ValueError("patterns에 분석할 케이스가 없습니다.")

    results = []
    for case_id, raw_case in raw_patterns.items():
        try:
            results.append(_analyze_case(case_id, raw_case, raw_filters))
        except (TypeError, ValueError) as error:
            results.append({
                "id": str(case_id),
                "passed": False,
                "reason": str(error),
            })
    return {
        "total": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": [result for result in results if not result["passed"]],
        "results": results,
    }


def run_json_mode() -> None:
    print("\n[JSON 분석 모드]")
    try:
        report = analyze_json_data(DATA_FILE)
    except ValueError as error:
        report = {
            "total": 1,
            "passed": 0,
            "failed": [{"id": "data.json", "reason": str(error)}],
            "results": [],
        }

    for result in report["results"]:
        if "reason" in result:
            print(f"{result['id']}: FAIL - {result['reason']}")
        else:
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"{result['id']}: Cross 점수={result['cross_score']:g}, "
                f"X 점수={result['x_score']:g}, 판정={result['actual']}, "
                f"expected={result['expected']} -> {status}"
            )

    run_performance_analysis()
    print("\n[결과 요약]")
    print(
        f"전체 테스트: {report['total']} / 통과: {report['passed']} / "
        f"실패: {len(report['failed'])}"
    )
    if not report["failed"]:
        print("실패 케이스: 없음")
    else:
        print("실패 케이스:")
        for failure in report["failed"]:
            reason = failure["reason"] if "reason" in failure else (
                f"expected={failure['expected']}, actual={failure['actual']}"
            )
            print(f"- {failure['id']}: {reason}")


def run_performance_analysis(sizes: Tuple[int, ...] = SIZES) -> None:
    print("\n[성능 분석]")
    print(
        f"크기(NxN) | 2D 평균(ms) | 1D 평균(ms) | 속도비(2D/1D) | "
        f"연산 횟수(N²), 각 {PERFORMANCE_REPETITIONS}회"
    )
    for size in sizes:
        generated = generate_patterns(size)
        pattern = generated["Cross"]
        filter_matrix = generated["X"]
        pattern_2d = pattern.to_2d()
        filter_2d = filter_matrix.to_2d()

        baseline_ns = 0
        optimized_ns = 0
        for _ in range(PERFORMANCE_REPETITIONS):
            start = time.perf_counter_ns()
            mac_2d(pattern_2d, filter_2d)
            baseline_ns += time.perf_counter_ns() - start

            start = time.perf_counter_ns()
            mac(pattern, filter_matrix)
            optimized_ns += time.perf_counter_ns() - start

        baseline_ms = baseline_ns / PERFORMANCE_REPETITIONS / 1_000_000
        optimized_ms = optimized_ns / PERFORMANCE_REPETITIONS / 1_000_000
        speedup = baseline_ms / optimized_ms if optimized_ms else float("inf")
        print(
            f"{size:>5}x{size:<2}   {baseline_ms:>10.6f}   "
            f"{optimized_ms:>10.6f}   {speedup:>13.3f}x   "
            f"{size * size:>8}"
        )


def main() -> None:
    print("Mini NPU MAC 시뮬레이터")
    while True:
        print("\n입력 방식 선택")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        choice = input("선택 (1/2): ").strip()
        if choice == "1":
            run_user_mode()
            return
        if choice == "2":
            run_json_mode()
            return
        print("입력 형식 오류: 1 또는 2를 입력하세요.")


def run_application() -> None:
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자 요청(Ctrl+C)으로 프로그램을 안전하게 종료합니다.")
    except EOFError:
        print("\n입력이 종료되어 프로그램을 안전하게 종료합니다.")
    except Exception as error:
        print(
            "\n예기치 않은 오류가 발생해 프로그램을 종료합니다: "
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    run_application()
