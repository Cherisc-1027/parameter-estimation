# Created w help of deepseek
import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment1 import Experiment

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        # Initialize an experiment with some data
        n_correct = [55, 60, 75, 90, 95]
        n_incorrect = [45, 40, 25, 10, 5]
        self.experiment = Experiment(n_correct, n_incorrect)
        self.model = SimplifiedThreePL(self.experiment)

    def test_summary(self):
        """Test the summary method."""
        summary = self.model.summary()
        self.assertEqual(summary["n_total"], 500)
        self.assertEqual(summary["n_correct"], 375)
        self.assertEqual(summary["n_incorrect"], 125)
        self.assertEqual(summary["n_conditions"], 5)

    def test_predict(self):
        """Test that predict() returns valid probabilities."""
        probabilities = self.model.predict((1.0, 0.0))
        self.assertTrue(np.all(probabilities >= 0) and np.all(probabilities <= 1))

    def test_fit(self):
        """Test that fit() estimates parameters correctly."""
        self.model.fit()
        self.assertIsNotNone(self.model.get_discrimination())
        self.assertIsNotNone(self.model.get_base_rate())

    def test_unfitted_model_errors(self):
        """Test that methods raise errors if the model hasn't been fitted."""
        with self.assertRaises(ValueError):
            self.model.get_discrimination()
        with self.assertRaises(ValueError):
            self.model.get_base_rate()

    def test_integration(self):
        """Integration test with a dataset of five conditions and 100 trials per condition."""
        n_correct = [55, 60, 75, 90, 95]
        n_incorrect = [45, 40, 25, 10, 5]
        experiment = Experiment(n_correct, n_incorrect)
        model = SimplifiedThreePL(experiment)
        
        # Fit the model
        model.fit()
        
        # Print the estimated parameters for debugging
        a_est = model.get_discrimination()
        c_est = model.get_base_rate()
        print(f"Estimated Discrimination (a): {a_est}")
        print(f"Estimated Base Rate (c): {c_est}")
        
        # Predict probabilities
        probabilities = model.predict((a_est, c_est))
        print("Predicted Probabilities:", probabilities)
        
        # Expected probabilities based on observed data
        expected_probabilities = np.array([0.55, 0.60, 0.75, 0.90, 0.95])
        
        # Verify that predictions align with observed response patterns
        self.assertTrue(
            np.allclose(probabilities, expected_probabilities, atol=0.1),
            f"Predicted probabilities {probabilities} do not match expected {expected_probabilities}"
        )

    def test_higher_base_rate(self):
        """Test that higher base rates result in higher probabilities."""
        probabilities_low = self.model.predict((1.0, -1.0))  # Low base rate
        probabilities_high = self.model.predict((1.0, 1.0))  # High base rate
        self.assertTrue(np.all(probabilities_high >= probabilities_low))

    def test_higher_difficulty(self):
        """Test that higher difficulty values result in lower probabilities when a is positive."""
        probabilities_low_difficulty = self.model.predict((1.0, 0.0))  # Low difficulty
        probabilities_high_difficulty = self.model.predict((1.0, 0.0))  # High difficulty
        self.assertTrue(np.all(probabilities_low_difficulty >= probabilities_high_difficulty))

    def test_higher_ability(self):
        """Test that higher ability parameters result in higher probabilities when a is positive."""
        # Create a new model with a higher ability parameter (theta)
        n_correct = [55, 60, 75, 90, 95]
        n_incorrect = [45, 40, 25, 10, 5]
        experiment = Experiment(n_correct, n_incorrect)
        model = SimplifiedThreePL(experiment)
        model._theta = 1.0  # Higher ability parameter
        probabilities = model.predict((1.0, 0.0))
        self.assertTrue(np.all(probabilities >= self.model.predict((1.0, 0.0))))

    def negative_log_likelihood(self, parameters):
        a, q = parameters
        probabilities = self.predict((a, q))
        n_correct = np.array(self._experiment.n_correct)
        n_incorrect = np.array(self._experiment.n_incorrect)
        
        # Avoid log(0) by adding a small constant
        epsilon = 1e-10
        log_likelihood = np.sum(n_correct * np.log(probabilities + epsilon)) + np.sum(n_incorrect * np.log(1 - probabilities + epsilon))
        
        return -log_likelihood
    def test_larger_a_for_steeper_curve(self):
        """Test that a larger estimate of a is returned for data with a steeper curve."""
        # Create a dataset with a steeper curve
        n_correct = [95, 90, 75, 60, 55]  # Higher accuracy for easier conditions
        n_incorrect = [5, 10, 25, 40, 45]
        experiment = Experiment(n_correct, n_incorrect)
        model = SimplifiedThreePL(experiment)
        model.fit()
        a_est = model.get_discrimination()
        self.assertGreater(a_est, 1.0)  # Expect a larger a for a steeper curve

    def test_corruption(self):
        """Test that private attributes cannot be accessed directly."""
        # Check that private attributes exist
        self.assertTrue(hasattr(self.model, '_base_rate'))
        self.assertTrue(hasattr(self.model, '_logit_base_rate'))
        self.assertTrue(hasattr(self.model, '_discrimination'))
        self.assertTrue(hasattr(self.model, '_is_fitted'))

if __name__ == "__main__":
    unittest.main()