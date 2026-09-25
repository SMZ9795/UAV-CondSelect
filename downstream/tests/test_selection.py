"""Check paper selection traces, objective semantics, and the standalone CLI."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.selection import objective, select


class SelectionTests(unittest.TestCase):
    def test_paper_traces(self):
        for name in ("rt_context", "yolo_context"):
            matrix = json.loads((ROOT / f"results/recomputed/{name}_selection.json").read_text())
            traces = json.loads((ROOT / f"results/recomputed/{name}_selection_traces.json").read_text())
            for expected in traces:
                with self.subTest(detector=name, weight=expected["lambda"]):
                    self.assertEqual(select(matrix["matrix"], expected["lambda"], matrix["order"]), expected)

    def test_coverage_uses_full_pool_and_overlap_keeps_sign(self):
        matrix = [[1, -0.5, 0.2], [-0.5, 1, 0.8], [0.2, 0.8, 1]]
        self.assertEqual(objective(matrix, [0, 1], 2), {"F": 2.8, "R": -0.5, "J": 3.8})

    def test_exact_ties_follow_input_order(self):
        result = select([[1, 1, 1]] * 3, 1, ["snow", "rain", "fog"])
        self.assertEqual(result["deleted"], ["snow", "rain"])
        self.assertEqual(result["retained"], ["fog"])
        self.assertEqual(result["stop"], "singleton")

    def test_zero_gain_stops(self):
        result = select([[1, 1], [1, 1]], 0, ["rain", "snow"])
        self.assertEqual(result["deleted"], [])
        self.assertEqual(result["retained"], ["rain", "snow"])
        self.assertIsNone(result["steps"][0]["deleted"])

    def test_negative_overlap_does_not_become_absolute(self):
        result = select([[1, -0.5], [-0.5, 1]], 4, ["rain", "snow"])
        self.assertEqual(result["retained"], ["rain", "snow"])
        self.assertEqual(result["final"]["J"], 4.0)

    def test_singleton(self):
        result = select([[1]], 3, ["rain"])
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["final"], {"F": 1.0, "R": 0.0, "J": 1.0})

    def test_invalid_inputs(self):
        cases = [
            ([], 0, []),
            ([[1]], 0, "rain"),
            ([[1]], 0, {"rain": 0}),
            ([[1]], -1, ["rain"]),
            ([[1]], float("nan"), ["rain"]),
            ([[1]], float("inf"), ["rain"]),
            ([[1]], 0, ["rain", "snow"]),
            ([[1, 0], [0]], 0, ["rain", "snow"]),
            ([[1, 0], [0, 1]], 0, ["rain", "rain"]),
            ([[1]], 0, [""]),
            ([[1, 0.3], [0.2, 1]], 0, ["rain", "snow"]),
            ([[1, float("nan")], [float("nan"), 1]], 0, ["rain", "snow"]),
            ([[1, 2], [2, 1]], 0, ["rain", "snow"]),
            ([[0.9]], 0, ["rain"]),
        ]
        for matrix, weight, order in cases:
            with self.subTest(matrix=matrix, weight=weight, order=order):
                with self.assertRaises(ValueError):
                    select(matrix, weight, order)

    def test_cli_without_site_packages_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "selection.json"
            result = subprocess.run([
                sys.executable, "-S", str(ROOT / "tools/select_conditions.py"),
                "--matrix", str(ROOT / "results/recomputed/rt_context_selection.json"),
                "--lambda", "0.6", "--output", str(output),
            ], cwd=directory, text=True, capture_output=True, check=True)
            self.assertIn("Selected conditions: snow, dust, low_light", result.stdout)
            self.assertEqual(json.loads(output.read_text())["retained"], ["snow", "dust", "low_light"])

    def test_cli_rejects_invalid_json_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text('{"matrix": [[1]]}')
            result = subprocess.run([
                sys.executable, "-S", str(ROOT / "tools/select_conditions.py"),
                "--matrix", str(path), "--lambda", "0.6",
            ], text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("input must contain matrix and order", result.stderr)


if __name__ == "__main__":
    unittest.main()
