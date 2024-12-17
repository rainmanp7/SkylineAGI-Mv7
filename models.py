# simplified_models.py
# Created Nov 14th 2024

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import memory_profiler
from collections import defaultdict
from attention_mechanism import MultiHeadAttention, ContextAwareAttention

# Setting up logging
logging.basicConfig(level=logging.INFO)

class SkylineModel(nn.Module):
    def __init__(self, input_size, num_heads, context_size):
        super(SkylineModel, self).__init__()
        self.multi_head_attention = MultiHeadAttention(input_size=input_size, num_heads=num_heads)
        self.context_aware_attention = ContextAwareAttention(input_size=input_size, context_size=context_size)
        self.fc = nn.Linear(input_size, 1)  # Example feedforward layer

    def forward(self, x, context):
        # Apply multi-head attention
        x = self.multi_head_attention(x)

        # Apply context-aware attention
        x = self.context_aware_attention(x, context)

        # Apply feedforward layer
        x = self.fc(x)

        return x

class BaseModel:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.criterion = None

    def fit(self, X, y, epochs=10):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

@dataclass
class ModelMetrics:
    mae: float
    mse: float
    r2: float
    training_time: float
    memory_usage: float
    prediction_latency: float

class SimpleModelValidator:
    def __init__(self):
        self.metrics_history = defaultdict(list)

    def validate_model(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_key: str
    ) -> ModelMetrics:
        start_time = time.time()
        memory_usage = memory_profiler.memory_usage()

        # Make predictions
        y_pred = model.predict(X_val)

        # Calculate metrics
        metrics = ModelMetrics(
            mae=mean_absolute_error(y_val, y_pred),
            mse=mean_squared_error(y_val, y_pred),
            r2=r2_score(y_val, y_pred),
            training_time=time.time() - start_time,
            memory_usage=max(memory_usage) - min(memory_usage),
            prediction_latency=self._measure_prediction_latency(model, X_val)
        )

        # Store metrics
        self.metrics_history[model_key].append(metrics)

        return metrics

    def _measure_prediction_latency(
        self,
        model: Any,
        X: np.ndarray,
        n_iterations: int = 100
    ) -> float:
        latencies = []
        for _ in range(n_iterations):
            start_time = time.time()
            model.predict(X[:100])  # Use small batch for latency test
            latencies.append(time.time() - start_time)
        return np.mean(latencies)

class SkylineModelWrapper(BaseModel):
    def __init__(self, input_size, num_heads, context_size, learning_rate=0.001):
        super(SkylineModelWrapper, self).__init__()
        self.model = SkylineModel(input_size, num_heads, context_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def fit(self, X, y, epochs=10):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor, X_tensor[:, 0, :])  # Using the first sequence element as context
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor, X_tensor[:, 0, :])  # Using the first sequence element as context
        return outputs.numpy().flatten()

    def evaluate(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            outputs = self.model(X_tensor, X_tensor[:, 0, :])  # Using the first sequence element as context
            loss = self.criterion(outputs, y_tensor)
        accuracy = r2_score(y, outputs.numpy().flatten())
        return loss.item(), accuracy

    @property
    def feature_importances_(self):
        # Placeholder for feature importance
        return {"feature1": 0.5, "feature2": 0.5}  # Example feature importance

def main():
    # Example usage of SkylineModelWrapper
    input_size = 128
    num_heads = 8
    context_size = 128  # Adjusted to match the context tensor size
    batch_size = 4
    sequence_length = 10

    # Generate random data
    X_train = np.random.rand(batch_size, sequence_length, input_size)
    y_train = np.random.rand(batch_size)
    X_val = np.random.rand(batch_size, sequence_length, input_size)
    y_val = np.random.rand(batch_size)

    # Initialize the model
    model = SkylineModelWrapper(input_size, num_heads, context_size)

    # Train the model
    model.fit(X_train, y_train, epochs=5)

    # Validate the model
    validator = SimpleModelValidator()
    metrics = validator.validate_model(model, X_val, y_val, "skyline_model")
    print(metrics)

if __name__ == "__main__":
    main()
