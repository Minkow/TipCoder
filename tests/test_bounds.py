import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from bounds import summarize


class BoundsTests(unittest.TestCase):
    def test_three_way_union_intersection(self):
        result = summarize(
            {"a": True, "b": False},
            [{"a": False, "b": True}, {"a": True, "b": True}],
            {"a": False, "b": True},
        )
        self.assertEqual(result["lower"], 0)
        self.assertEqual(result["upper"], 100)
        self.assertEqual(result["base"], 50)
        self.assertEqual(result["rm"], 50)
        self.assertEqual(result["raw_by_branch"], [50, 100])

    def test_no_drop_samples(self):
        with self.assertRaises(ValueError):
            summarize({"a": True, "b": False}, [{"a": True}])

    def test_selected_must_cover_all_tasks(self):
        with self.assertRaises(ValueError):
            summarize({"a": True}, [{"a": False}], {})
