import unittest
from pathlib import Path

from utils.dashboard import load_metric_rows_from_csv


class DashboardDataTests(unittest.TestCase):
    def test_load_metric_rows_from_csv_contains_expected_models(self):
        root = Path(__file__).resolve().parents[1]
        rows = load_metric_rows_from_csv(root / "metrics.csv")
        self.assertIn("Logistic Regression", rows)
        self.assertIn("KNN", rows)
        self.assertIn("Random Forest", rows)
        self.assertIn("Neural Network", rows)


if __name__ == "__main__":
    unittest.main()
