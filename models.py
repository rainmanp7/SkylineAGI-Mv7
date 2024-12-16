# 9 tier reworked
# removed the simple medium complex
# models.py
# memory added fitted Nov14

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import time
import memory_profiler
import logging
from collections import defaultdict
import torch.nn as nn
from memory_manager import MemoryManager
from attention_mechanism import MultiHeadAttention, ContextAwareAttention


class SkylineModel(nn.Module):
    def __init__(self, input_size, num_heads, context_size):
        super(SkylineModel, self).__init__()
        self.multi_head_attention = MultiHeadAttention(input_size=input_size, num_heads=num_heads)
        self.context_aware_attention = ContextAwareAttention(input_size=input_size, context_size=context_size)
        # Add other layers, such as feedforward, residual connections, etc.

    def forward(self, x, context):
        # Apply multi-head attention
        x = self.multi_head_attention(x)

        # Apply context-aware attention
        x = self.context_aware_attention(x, context)

        # Apply other layers
        # ...

        return x

class BaseModel:
    def __init__(self):  # Fixed double underscores
        self.model = None
        
    def fit(self, X, y):
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

class ModelValidator:
    def __init__(self):  # Fixed double underscores
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
        
    def _measure_prediction_latency(  # Fixed method name and added underscore prefix
        self,
        model: Any,
        X: np.ndarray,
        n_iterations: int = 100
    ) -> float:
        latencies = []
        for _ in range(n_iterations):  # Fixed iteration variable
            start_time = time.time()
            model.predict(X[:100])  # Use small batch for latency test
            latencies.append(time.time() - start_time)
        return np.mean(latencies)

@dataclass
class ExpandedModelMetrics(ModelMetrics):
    loss: float
    accuracy: float
    feature_importance: Dict[str, float]

class ExpandedModelValidator(ModelValidator):
    def validate_model(self, model, X_val, y_val, model_key):
        # Call the original validate_model method
        metrics = super().validate_model(model, X_val, y_val, model_key)
        
        try:
            # Compute additional metrics
            loss, accuracy = model.evaluate(X_val, y_val)
            feature_importance = model.feature_importances_
            
            # Create the expanded metrics object
            expanded_metrics = ExpandedModelMetrics(
                mae=metrics.mae,
                mse=metrics.mse,
                r2=metrics.r2,
                training_time=metrics.training_time,
                memory_usage=metrics.memory_usage,
                prediction_latency=metrics.prediction_latency,
                loss=loss,
                accuracy=accuracy,
                feature_importance=feature_importance
            )
            
            # Store the expanded metrics
            self.metrics_history[model_key].append(expanded_metrics)
            return expanded_metrics
            
        except AttributeError as e:
            logging.warning(f"Could not compute expanded metrics: {str(e)}")
            return metrics


### End Model Validation and Monitoring
