# models.py
# Updated Dec 27, 2024

import numpy as np
from scipy.special import softmax as scipy_softmax
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import optuna

class SkylineAGI32:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def predict(self, inputs):
        """Predict output for given inputs."""
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        weighted_sum = np.dot(inputs, self.weights) + self.biases
        output = scipy_softmax(weighted_sum)  # Corrected softmax function
        return output

    def train(self, X_train, y_train_hard, y_train_soft, X_val, y_val, 
              epochs=100, batch_size=32, patience=10, temperature=3.0, alpha=0.5):
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            # Simulate a training loop
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train_soft = X_train[indices], y_train_soft[indices]
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train_soft[i:i + batch_size]
                # Placeholder for weight updates
                predictions = self.predict(X_batch)
                loss = np.mean((predictions - y_batch) ** 2)  # MSE loss
            
            # Validation loss
            val_predictions = self.predict(X_val)
            val_loss = np.mean((val_predictions - y_val) ** 2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, max_iterations=10):
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)  # Updated
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            epochs = trial.suggest_int("epochs", 10, 100)

            self.train(X_train, y_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
            predictions = self.predict(X_val)
            mse = mean_squared_error(y_val, predictions)
            return mse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=max_iterations)

        best_hyperparams = study.best_params
        best_value = study.best_value
        return best_hyperparams, best_value

def create_model(input_shape: int, output_shape: int) -> SkylineAGI32:
    """
    Create a new SkylineAGI32 model with random weights and biases.

    Args:
        input_shape (int): Number of input features.
        output_shape (int): Number of output classes.

    Returns:
        SkylineAGI32: A new instance of the SkylineAGI32 model.
    """
    # Initialize random weights and biases
    weights = np.random.randn(input_shape, output_shape)
    biases = np.zeros(output_shape)

    # Create and return the model
    return SkylineAGI32(weights, biases)

def main():
    np.random.seed(42)
    
    # Example with a multi-class dataset (load_digits) for clarity
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # One-hot encode y for multi-class classification
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))
    
    X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # Initialize the model
    input_shape = X_train.shape[1]  # Number of input features
    output_shape = y_train.shape[1]  # Number of output classes
    skyline_model = create_model(input_shape, output_shape)

    # Optimize the hyperparameters
    best_result = skyline_model.optimize_hyperparameters(X_train, y_train, X_val, y_val)

    print("Best Hyperparameters:", best_result[0])
    print("Best MSE:", best_result[1])

    # Continue training the model with the optimized hyperparameters
    skyline_model.train(X_train, y_train, y_train, X_val, y_val, epochs=200, batch_size=32, patience=20)

if __name__ == "__main__":
    main()
