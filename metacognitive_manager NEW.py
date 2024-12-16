# metacognitive_manager.py
# Updated Sun, Dec 15th 2024

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
import time

@dataclass
class SystemState:
    capabilities: Dict[str, float]
    confidence_levels: Dict[str, float]
    performance_metrics: Dict[str, Tuple[float, float, float]]
    active_components: List[str]

class MetaCognitiveManager:
    def __init__(self):
        self.system_state = SystemState(
            capabilities={},
            confidence_levels={},
            performance_metrics={},
            active_components=[]
        )
        self.anomaly_threshold = 0.95
        self.performance_history = []
        self.adaptation_range = (0.1, 0.5)  # Example adaptation range for MSE
        self.adaptation_rate = 1.1  # Rate to adjust parameters

    def monitor_performance(self, model_key: str, performance: Tuple[float, float, float]) -> None:
        """
        Monitor the performance of a model and store it in the performance history.
        Args:
            model_key: The identifier of the model.
            performance: A tuple of (mae, mse, r2) metrics.
        """
        self.performance_history.append((model_key, performance))
        logging.info(f"Model {model_key} performance: MAE={performance[0]}, MSE={performance[1]}, R2={performance[2]}")

    def update_system_parameters(self) -> None:
        """
        Analyze the performance history and update system parameters accordingly.
        """
        if len(self.performance_history) < 10:
            logging.info("Not enough performance data to update system parameters.")
            return

        # Calculate the average performance across all models
        total_mae, total_mse, total_r2 = 0, 0, 0
        for _, (mae, mse, r2) in self.performance_history:
            total_mae += mae
            total_mse += mse
            total_r2 += r2
        
        avg_mae = total_mae / len(self.performance_history)
        avg_mse = total_mse / len(self.performance_history)
        avg_r2 = total_r2 / len(self.performance_history)

        logging.info(f"Average Performance: MAE={avg_mae}, MSE={avg_mse}, R2={avg_r2}")

        # Adjust system parameters based on the average performance
        if avg_mse > self.adaptation_range[1]:
            logging.info("Decreasing system parameters due to poor average MSE.")
        elif avg_mse < self.adaptation_range[0]:
            logging.info("Increasing system parameters due to good average MSE.")
        else:
            logging.info("System parameters remain unchanged.")

    def run_metacognitive_tasks(self) -> None:
        """
        Periodically run metacognitive tasks, such as monitoring performance and updating system parameters.
        """
        while True:
            for component in self.system_state.active_components:
                metrics = self.system_state.performance_metrics.get(component)
                if metrics:
                    self.monitor_performance(component, metrics)
            
            self.update_system_parameters()
            time.sleep(60)  # Run metacognitive tasks every minute

    def update_self_model(self, component_name: str, metrics: Dict[str, float]) -> None:
        """Updates the internal self-model with new performance data"""
        self.system_state.performance_metrics[component_name] = metrics
        self._detect_anomalies(component_name, metrics)
        self._update_confidence_levels(component_name)

    def _detect_anomalies(self, component_name: str, metrics: Dict[str, float]) -> None:
        """Detects performance anomalies using statistical analysis"""
        for metric_name, value in metrics.items():
            history = [metric[1] for metric in self.performance_history if metric[0] == component_name]
            if history:
                mean = np.mean(history)
                std = np.std(history)
                z_score = abs((value - mean) / std) if std != 0 else 0
                
                if z_score > self.anomaly_threshold:
                    logging.warning(
                        f"Anomaly detected in {component_name}: {metric_name} = {value}"
                    )

    def _update_confidence_levels(self, component_name: str) -> None:
        """Updates confidence levels based on performance metrics"""
        metrics = self.system_state.performance_metrics[component_name]
        avg_performance = np.mean(list(metrics.values()))
        self.system_state.confidence_levels[component_name] = avg_performance

    def get_component_state(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves the current state of a specific component"""
        if component_name not in self.system_state.active_components:
            return None
            
        return {
            'confidence': self.system_state.confidence_levels.get(component_name),
            'performance': self.system_state.performance_metrics.get(component_name),
            'capabilities': self.system_state.capabilities.get(component_name)
        }

# Standalone Execution
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s"
    )

    logging.info("Initializing MetaCognitiveManager for standalone execution...")
    manager = MetaCognitiveManager()

    # Simulate performance data for 10 models
    simulated_data = [
        ("Model_A", (0.2, 0.15, 0.85)),
        ("Model_B", (0.3, 0.25, 0.75)),
        ("Model_C", (0.1, 0.05, 0.95)),
        ("Model_D", (0.25, 0.2, 0.8)),
        ("Model_E", (0.35, 0.3, 0.7)),
        ("Model_F", (0.4, 0.35, 0.65)),
        ("Model_G", (0.2, 0.18, 0.88)),
        ("Model_H", (0.28, 0.22, 0.78)),
        ("Model_I", (0.3, 0.27, 0.73)),
        ("Model_J", (0.15, 0.1, 0.9))
    ]

    for model, performance in simulated_data:
        manager.monitor_performance(model, performance)

    manager.update_system_parameters()

    logging.info(f"Final system state: {manager.system_state}")
