"""CPU-only checks for candidate selection and prompt preservation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from rerank import select_candidates
from generate_evalplus_tips import TIP_TEMPLATE, inject_prompt, clean_tip


class InferenceTests(unittest.TestCase):
    def test_tie_keeps_base(self):
        rows = [
            {"task_id": "a", "solution": "base"},
            {"task_id": "a", "solution": "guided"},
        ]
        self.assertEqual(select_candidates(rows, [1.0, 1.0])[0]["solution"], "base")

    def test_strict_max_per_task(self):
        rows = [
            {"task_id": "a", "solution": "base"},
            {"task_id": "b", "solution": "base"},
            {"task_id": "a", "solution": "sft"},
            {"task_id": "a", "solution": "rl"},
        ]
        result = {
            r["task_id"]: r["solution"] for r in select_candidates(rows, [0, 3, 2, 1])
        }
        self.assertEqual(result, {"a": "sft", "b": "base"})

    def test_incomplete_scores_fail(self):
        with self.assertRaises(ValueError):
            select_candidates([{"task_id": "a"}], [])

    def test_nonfinite_score_fails(self):
        with self.assertRaises(ValueError):
            select_candidates([{"task_id": "a"}], [float("nan")])

    def test_template_matches_launcher_file(self):
        path = Path(__file__).resolve().parents[1] / "inference/proposer_template.txt"
        self.assertEqual(path.read_text(), TIP_TEMPLATE)

    def test_prompt_retains_contract(self):
        prompt = (
            'def example(x):\n    """Return x.\n    >>> example(2)\n    2\n    """\n'
        )
        enhanced = inject_prompt("humaneval", prompt, "Keep the input unchanged.")
        self.assertIn(">>> example(2)", enhanced)
        self.assertIn("Keep the input unchanged.", enhanced)
        self.assertTrue(enhanced.endswith('"""\n'))

    def test_tip_separator(self):
        self.assertEqual(
            clean_tip("analysis\n###TIPS###\nPreserve ordering."), "Preserve ordering."
        )


if __name__ == "__main__":
    unittest.main()
