# cache_utils.py
# updated to use optuna Dec 12 2024
import functools
import hashlib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from logging_config import setup_logging

# Set up logging
logger = setup_logging(script_name="cache_utils")

# Caching utilities
cache_conditions = {
    'X_train_hash': None,
    'y_train_hash': None,
    'hyperparameters_hash': None,
}

def compute_hash(data):
    """ Computes a SHA256 hash of the given data. """
    return hashlib.sha256(str(data).encode()).hexdigest()

def cached_bayesian_fit(func):
    """ Decorator for caching Bayesian optimization results. """
    cache = {}

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        input_hash = compute_hash(args)
        hyperparameters_hash = compute_hash(kwargs.get("hyperparameters", {}))

        if input_hash in cache and cache_conditions['hyperparameters_hash'] == hyperparameters_hash:
            logger.info("Using cached results...")
            return cache[input_hash]

        result = func(self, *args, **kwargs)
        cache[input_hash] = result
        cache_conditions['hyperparameters_hash'] = hyperparameters_hash
        return result

    wrapper.cache_clear = lambda: cache.clear()
    return wrapper

class BayesianHyperparameterTuning:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def _objective(self, trial):
        """ Objective function for Optuna. """
        kernel_constant = trial.suggest_float("kernel_constant", 0.1, 10.0)
        kernel_length_scale = trial.suggest_float("kernel_length_scale", 0.1, 10.0)
        n_restarts_optimizer = trial.suggest_int("n_restarts_optimizer", 1, 100)

        kernel = C(kernel_constant) * RBF(kernel_length_scale)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
        gpr.fit(self.X_train, self.y_train)
        y_pred = gpr.predict(self.X_val)
        score = r2_score(self.y_val, y_pred)
        
        return -score  # Minimize the negative R² score

    @cached_bayesian_fit
    def tune_hyperparameters(self, n_trials=50):
        """ Perform Bayesian hyperparameter tuning with caching and early stopping. """
        study = optuna.create_study(direction="minimize")
        best_value = float('inf')

        for trial in range(n_trials):
            study.optimize(self._objective, n_trials=1)  # Optimize one trial at a time

            current_best_value = study.best_value
            logger.info(f"Trial {trial + 1}/{n_trials}, Best Value: {current_best_value}")

            # Check for improvement
            if current_best_value < best_value:
                best_value = current_best_value  # Update the best value
                logger.info("Improvement found!")
            else:
                logger.info("No improvement found, stopping optimization.")
                break  # Stop if no improvement in this trial

        best_hyperparameters = study.best_params
        return best_hyperparameters, -best_value  # Return negative because we minimized negative R² score

    def evaluate_best_model(self, best_hyperparameters):
        """ Evaluate the best model with the tuned hyperparameters. """
        kernel = C(best_hyperparameters["kernel_constant"]) * RBF(best_hyperparameters["kernel_length_scale"])
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=best_hyperparameters["n_restarts_optimizer"])
        gpr.fit(self.X_train, self.y_train)
        y_pred = gpr.predict(self.X_val)
        mse = mean_squared_error(self.y_val, y_pred)
        r2 = r2_score(self.y_val, y_pred)
        
        return {"MSE": mse, "R2": r2}

# Example usage
if __name__ == "__main__":
    # Sample data generation
    X_train = np.random.rand(100, 5)  # 100 samples with 5 features
    y_train = np.random.rand(100)      # Corresponding target values
    X_val = np.random.rand(20, 5)      # Validation set (20 samples)
    y_val = np.random.rand(20)          # Corresponding validation target values

    # Initialize tuner
    tuner = BayesianHyperparameterTuning(X_train, y_train, X_val, y_val)

    # Perform tuning with early stopping
    best_hyperparameters, best_score = tuner.tune_hyperparameters(n_trials=50)
    
    logger.info(f"Best Hyperparameters: {best_hyperparameters}")
    logger.info(f"Best R^2 Score: {best_score}")

    # Evaluate model with the best hyperparameters found
    evaluation_metrics = tuner.evaluate_best_model(best_hyperparameters)
    
    logger.info(f"Evaluation Metrics (MSE and R^2): {evaluation_metrics}")
