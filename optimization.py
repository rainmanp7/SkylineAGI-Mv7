# Beginning of optimization.py
# Last updated: Nov 23, 2024
# Purpose: Handles hyperparameter optimization using Bayesian methods with Optuna.
# Todo: implement early stopping and fix models.

import os
import json
import warnings
import numpy as np
from typing import Dict, Tuple
import logging
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.metrics import mean_squared_error
from models import create_model  # Ensure this is implemented and matches your architecture.
from utils import load_data, compute_complexity_factor  # Ensure these utilities exist.

# Suppress specific warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class HyperparameterOptimization:
    """
    Class for handling hyperparameter optimization tasks with Bayesian methods using Optuna.
    """

    def __init__(self, config_file: str = 'config.json', max_iterations=10, tolerance=0.05):
        """
        Initialize the optimization class and load configuration from the specified file.
        """
        self.cache = {}
        self.config = self._load_config(config_file)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_file: str) -> Dict:
        """
        Load configuration settings from a JSON file.
        """
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON in '{config_file}'.")

    def objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Objective function to be optimized by Optuna.
        """
        # Suggest hyperparameters for the Bayesian optimization process
        kernel_type = trial.suggest_categorical('kernel_type', ['matern', 'rbf'])
        length_scale = trial.suggest_loguniform('length_scale', 1e-5, 1e5)
        kernel_constant = trial.suggest_uniform('kernel_constant', 0.1, 10)
        
        # Configure the kernel based on suggested hyperparameters
        if kernel_type == 'matern':
            kernel = C(kernel_constant) * Matern(length_scale=length_scale, nu=2.5)
        else:
            kernel = C(kernel_constant) * RBF(length_scale=length_scale)
        
        # Perform Bayesian optimization with the configured kernel
        gp_model = GaussianProcessRegressor(kernel=kernel)
        gp_model.fit(X_train, y_train)
        predictions = gp_model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        
        # Return the negative MSE as the objective value (since Optuna minimizes)
        return -mse

    def perform_optimization(self, X_train, y_train, X_val, y_val):
        """
        Manages the overall optimization process using Optuna.
        """
        # Initialize Optuna study
        study = optuna.create_study(direction='minimize')
        
        # Perform optimization
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=self.max_iterations)
        
        # Extract the best trial and its hyperparameters
        best_trial = study.best_trial
        best_hyperparameters = best_trial.params
        
        # Create and save the best model based on the optimal hyperparameters
        best_model = create_model(self.config['model_type'], best_hyperparameters)
        model_save_path = os.path.join(self.config['model_save_dir'], 'best_model.pkl')
        best_model.save(model_save_path)
        
        return best_hyperparameters, -best_trial.value  # Return the negative value to get the original MSE

# Example usage
def main():
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 5)
    y_val = np.random.rand(20)

    optimizer = HyperparameterOptimization(max_iterations=10, tolerance=0.01)
    best_result = optimizer.perform_optimization(X_train, y_train, X_val, y_val)

    print("Best Hyperparameters:", best_result[0])
    print("Best MSE:", best_result[1])


if __name__ == "__main__":
    main()
