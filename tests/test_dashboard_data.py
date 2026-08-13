import unittest
from pathlib import Path

from utils.dashboard import get_dashboard_payload, load_metric_rows_from_csv


class DashboardDataTests(unittest.TestCase):
    def test_load_metric_rows_from_csv_contains_expected_models(self):
        root = Path(__file__).resolve().parents[1]
        rows = load_metric_rows_from_csv(root / "metrics.csv")
        self.assertIn("Logistic Regression", rows)
        self.assertIn("KNN", rows)
        self.assertIn("Random Forest", rows)
        self.assertIn("Neural Network", rows)

    def test_dashboard_warns_when_dataset_is_too_small(self):
        payload = get_dashboard_payload()
        self.assertIn("dataset_warning", payload)
        self.assertIn("too small", payload["dataset_warning"].lower())
        self.assertIn("prediction_confidence", payload)


if __name__ == "__main__":
    unittest.main()
