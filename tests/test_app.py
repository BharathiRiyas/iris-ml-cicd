import unittest
from app import app

class TestFlaskAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_valid(self):
        response = self.client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.get_json()["class"], ["setosa", "versicolor", "virginica"])

    def test_predict_invalid(self):
        response = self.client.post("/predict", json={"features": [5.1, 3.5]})
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
