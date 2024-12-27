# optimization.py
# Updated Dec 27, 2024

import numpy as np
from sklearn.metrics import mean_squared_error
import optuna
from models import create_model
from utils import load_data, compute_complexity_factor

class HyperparameterOptimization:
    def __init__(self):
        """Initialize the hyperparameter optimization class."""
        self.best_hyperparams = None
        self.best_value = None

    def optimize(self, X_train, y_train, X_val, y_val, max_iterations=10):
        """
        Optimize hyperparameters using Optuna.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation data.
            y_val (np.ndarray): Validation labels.
            max_iterations (int): Maximum number of optimization iterations.

        Returns:
            dict: Best hyperparameters.
            float: Best validation loss.
        """
        def objective(trial):
            # Define hyperparameters to optimize
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            epochs = trial.suggest_int("epochs", 10, 100)

            # Create and train the model
            input_shape = X_train.shape[1]
            output_shape = y_train.shape[1]
            model = create_model(input_shape, output_shape)

            # Placeholder for training logic (replace with actual training)
            predictions = model.predict(X_val)
            mse = mean_squared_error(y_val, predictions)
            return mse

        # Create an Optuna study and optimize
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=max_iterations)

        # Store the best hyperparameters and value
        self.best_hyperparams = study.best_params
        self.best_value = study.best_value

        return self.best_hyperparams, self.best_value

def main():
    # Load the data
    X_train, X_val, y_train, y_val = load_data()

    # Compute the complexity factor
    complexity_factor = compute_complexity_factor(X_train)
    print(f"Complexity Factor: {complexity_factor}")

    # Initialize the hyperparameter optimizer
    optimizer = HyperparameterOptimization()

    # Optimize the hyperparameters
    best_hyperparams, best_value = optimizer.optimize(X_train, y_train, X_val, y_val)

    print("Best Hyperparameters:", best_hyperparams)
    print("Best MSE:", best_value)

if __name__ == "__main__":
    main()