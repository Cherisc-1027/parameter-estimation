# Created w help of deepseek
import numpy as np
from scipy.optimize import minimize

class SimplifiedThreePL:
    def __init__(self, experiment):
        """
        Initialize the SimplifiedThreePL model.
        """
        self._experiment = experiment
        self._difficulties = np.array([2.0, 1.0, 0.0, -1.0, -2.0])  # Fixed difficulty parameters
        self._theta = 0.0  # Fixed participant ability parameter
        self._logit_base_rate = None  # Logit of the base rate parameter (q)
        self._discrimination = None  # Discrimination parameter (a)
        self._is_fitted = False  # Flag to check if the model has been fitted
        self._base_rate = None  # Base rate parameter (c)
    def summary(self):
        """
        Return a summary of the experiment data.
        """
        n_total = sum(self._experiment.n_correct) + sum(self._experiment.n_incorrect)
        return {
            "n_total": n_total,
            "n_correct": sum(self._experiment.n_correct),
            "n_incorrect": sum(self._experiment.n_incorrect),
            "n_conditions": len(self._difficulties)
        }

    def predict(self, parameters):
        """
        Predict the probability of a correct response in each condition.
        """
        a, q = parameters
        c = 1 / (1 + np.exp(-q))  # Inverse logit transform
        
        # Calculate probabilities using the 3PL formula
        exponent = -a * (self._theta - self._difficulties)
        probabilities = c + (1 - c) / (1 + np.exp(exponent))
        
        return probabilities
    def negative_log_likelihood(self, parameters):
        a, q = parameters
        probabilities = self.predict((a, q))
        n_correct = np.array(self._experiment.n_correct)
        n_incorrect = np.array(self._experiment.n_incorrect)
        
        # Avoid log(0) by adding a small constant
        epsilon = 1e-10
        log_likelihood = np.sum(n_correct * np.log(probabilities + epsilon)) + np.sum(n_incorrect * np.log(1 - probabilities + epsilon))
        
        return -log_likelihood

    def fit(self):
        """
        Fit the model using maximum likelihood estimation.
        """
        # Initial guess for parameters (a, q)
        initial_guess = [2.0, 0.0]  # Start with a larger initial guess for a

        # Define bounds for parameters (a, q)
        bounds = [(0.1, 10), (-10, 10)]  # a > 0, q can be any real value

        # Minimize the negative log-likelihood
        result = minimize(self.negative_log_likelihood, initial_guess, method="L-BFGS-B", bounds=bounds)

        # Update parameters
        self._discrimination = result.x[0]
        self._logit_base_rate = result.x[1]
        self._is_fitted = True
    def get_discrimination(self):
        """
        Get the estimated discrimination parameter (a).
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self._discrimination

    def get_base_rate(self):
        """
        Get the estimated base rate parameter (c).
        """
        if not self._is_fitted:
            raise ValueError("Model hasn't been fitted yet.")
        return 1 / (1 + np.exp(-self._logit_base_rate))