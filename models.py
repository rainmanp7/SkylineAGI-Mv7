import numpy as np
from scipy.special import softmax as scipy_softmax
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os

def softmax(x):
    """Softmax activation function using scipy."""
    return scipy_softmax(x, axis=1)

def one_hot_encode(labels, num_classes):
    """One-hot encode the labels."""
    return np.eye(num_classes)[labels]

class SkylineAGI32:
    """Custom neural network model."""
    def __init__(self, weights, biases, learning_rate=0.002, regularization=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.weights = weights
        self.biases = biases
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = np.zeros_like(weights)
        self.v_weights = np.zeros_like(weights)
        self.m_biases = np.zeros_like(biases)
        self.v_biases = np.zeros_like(biases)
        self.t = 0  # Time step

    def predict(self, inputs):
        """Predict output for given inputs."""
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        weighted_sum = np.dot(inputs, self.weights) + self.biases
        output = softmax(weighted_sum)
        return output

    def train(self, X_train, y_train_hard, y_train_soft, epochs=100, batch_size=32, patience=10, temperature=3.0, alpha=0.5):
        """Train the model using mini-batch gradient descent with regularization, early stopping, and Nadam optimizer."""
        num_samples, num_features = X_train.shape
        num_classes = y_train_hard.shape[1]
        
        best_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            self.t += 1  # Increment time step
            # Shuffle the data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_hard_shuffled = y_train_hard[indices]
            y_train_soft_shuffled = y_train_soft[indices]
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch_hard = y_train_hard_shuffled[i:i + batch_size]
                y_batch_soft = y_train_soft_shuffled[i:i + batch_size]
                
                # Forward pass
                weighted_sum = np.dot(X_batch, self.weights) + self.biases
                output = softmax(weighted_sum)
                
                # Compute loss (cross-entropy with regularization and knowledge distillation)
                loss_hard = -np.mean(y_batch_hard * np.log(output + 1e-8))
                loss_soft = -np.mean(y_batch_soft * np.log(output + 1e-8))
                loss = alpha * loss_hard + (1 - alpha) * loss_soft + self.regularization * np.sum(self.weights ** 2)
                
                # Backward pass
                error_hard = output - y_batch_hard
                error_soft = output - y_batch_soft
                error = alpha * error_hard + (1 - alpha) * error_soft
                d_weights = (np.dot(X_batch.T, error) / batch_size) + 2 * self.regularization * self.weights
                d_biases = np.mean(error, axis=0)
                
                # Update first and second moment estimates for weights
                self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * d_weights
                self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (d_weights ** 2)
                
                # Update first and second moment estimates for biases
                self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * d_biases
                self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (d_biases ** 2)
                
                # Compute bias-corrected first and second moment estimates
                m_weights_hat = self.m_weights / (1 - self.beta1 ** self.t)
                v_weights_hat = self.v_weights / (1 - self.beta2 ** self.t)
                m_biases_hat = self.m_biases / (1 - self.beta1 ** self.t)
                v_biases_hat = self.v_biases / (1 - self.beta2 ** self.t)
                
                # Compute Nesterov updates
                m_weights_hat_nadam = self.beta1 * m_weights_hat + (1 - self.beta1) * d_weights / (1 - self.beta1 ** self.t)
                m_biases_hat_nadam = self.beta1 * m_biases_hat + (1 - self.beta1) * d_biases / (1 - self.beta1 ** self.t)
                
                # Update weights and biases
                self.weights -= self.learning_rate * m_weights_hat_nadam / (np.sqrt(v_weights_hat) + self.epsilon)
                self.biases -= self.learning_rate * m_biases_hat_nadam / (np.sqrt(v_biases_hat) + self.epsilon)
            
            if epoch % 10 == 0:  # Print loss every 10 epochs
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

# Define cache file paths
preprocessed_data_cache = 'preprocessed_data.joblib'
hyperparameters_cache = 'hyperparameters.joblib'

# Load hyperparameters from cache if they exist, otherwise define them
if os.path.exists(hyperparameters_cache):
    hyperparameters = load(hyperparameters_cache)
    # Ensure all required keys are present
    required_keys = [
        'learning_rate', 'regularization', 'epochs', 'batch_size', 'patience',
        'max_iter_logistic', 'multi_class_logistic', 'beta1', 'beta2', 'epsilon',
        'temperature', 'alpha'
    ]
    for key in required_keys:
        if key not in hyperparameters:
            print(f"Key '{key}' not found in hyperparameters. Updating cache.")
            os.remove(hyperparameters_cache)
            break
else:
    hyperparameters = {}

if not hyperparameters:
    hyperparameters = {
        'learning_rate': 0.002,  # Adjusted learning rate for Nadam
        'regularization': 0.01,
        'epochs': 200,
        'batch_size': 32,
        'patience': 20,
        'max_iter_logistic': 1000,
        'multi_class_logistic': 'ovr',
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'temperature': 3.0,  # Temperature for soft targets
        'alpha': 0.5  # Weight for hard labels in the loss function
    }
    dump(hyperparameters, hyperparameters_cache)

# Load preprocessed data from cache if it exists, otherwise preprocess and save it
if os.path.exists(preprocessed_data_cache):
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = load(preprocessed_data_cache)
else:
    # Load dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # One-hot encode the target labels
    num_classes = len(np.unique(y))
    y_train_encoded = one_hot_encode(y_train, num_classes=num_classes)
    y_test_encoded = one_hot_encode(y_test, num_classes=num_classes)
    # Cache the preprocessed data
    dump((X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded), preprocessed_data_cache)

# Initialize weights, biases
num_features = X_train.shape[1]
num_classes = len(np.unique(y_train))
weights = np.random.randn(num_features, num_classes) * np.sqrt(2 / num_features)  # Xavier initialization
biases = np.zeros(num_classes)  # Initialize biases to zero

# Initialize models
skyline_model = SkylineAGI32(weights, biases, 
                            learning_rate=hyperparameters['learning_rate'], 
                            regularization=hyperparameters['regularization'], 
                            beta1=hyperparameters['beta1'],
                            beta2=hyperparameters['beta2'],
                            epsilon=hyperparameters['epsilon'])
logistic_model = LogisticRegression(max_iter=hyperparameters['max_iter_logistic'], 
                                    multi_class=hyperparameters['multi_class_logistic'])

# Training loop for logistic regression (teacher model)
logistic_model.fit(X_train, y_train)

# Generate soft targets from the teacher model
y_train_soft = logistic_model.predict_proba(X_train)
y_test_soft = logistic_model.predict_proba(X_test)

# Train the student model with knowledge distillation
skyline_model.train(X_train, y_train_encoded, y_train_soft, 
                    epochs=hyperparameters['epochs'], 
                    batch_size=hyperparameters['batch_size'], 
                    patience=hyperparameters['patience'],
                    temperature=hyperparameters['temperature'],
                    alpha=hyperparameters['alpha'])

# Evaluation
y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

y_pred_skyline_probs = skyline_model.predict(X_test)
y_pred_skyline_labels = np.argmax(y_pred_skyline_probs, axis=1)
accuracy_skyline = accuracy_score(y_test, y_pred_skyline_labels)

print(f"Logistic Regression Accuracy: {accuracy_logistic:.4f}")
print(f"Skyline AGI 3.2 Accuracy: {accuracy_skyline:.4f}")
print(f"Best Model: {'Logistic Regression' if accuracy_logistic > accuracy_skyline else 'Skyline AGI 3.2'}")
