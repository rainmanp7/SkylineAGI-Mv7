# internal_process_monitor.py
import psutil
import subprocess
import time
from collections import deque
from typing import Dict, Optional
from logging_config import setup_logging

# Set up logging
logger = setup_logging(script_name="internal_process_monitor")

class InternalProcessMonitor:
    """
    Monitors internal process metrics, including CPU, memory, task queue, 
    knowledge base updates, model training, and inference times.
    """

    def __init__(self, max_history_size: int = 100):
        """
        Initializes the monitor with a specified history size.

        :param max_history_size: Maximum number of historical data points to store
        """
        self.max_history_size = max_history_size

        # Historical data
        self.cpu_usage_history = deque(maxlen=max_history_size)
        self.memory_usage_history = deque(maxlen=max_history_size)
        self.task_queue_length_history = deque(maxlen=max_history_size)
        self.knowledge_base_updates_history = deque(maxlen=max_history_size)
        self.model_training_time_history = deque(maxlen=max_history_size)
        self.model_inference_time_history = deque(maxlen=max_history_size)
        self.timestamps = deque(maxlen=max_history_size)

        # Task management
        self.current_task: Optional[str] = None
        self.task_metrics: Dict[str, Dict] = {}

    def start_task_monitoring(self, task_name: str) -> None:
        """
        Begins monitoring a new task.

        :param task_name: Name of the task to monitor
        """
        self.current_task = task_name
        if task_name not in self.task_metrics:
            self.task_metrics[task_name] = {
                "start_time": time.time(),
                "cpu_usage": [],
                "memory_usage": []
            }
        logger.info(f"Started monitoring task: {task_name}")

    def end_task_monitoring(self) -> None:
        """
        Ends the current task monitoring session.
        """
        if self.current_task:
            self.task_metrics[self.current_task]["end_time"] = time.time()
            logger.info(f"Ended monitoring task: {self.current_task}")
            self.current_task = None

    def _update_task_metrics(self, metric_name: str, value: float) -> None:
        """
        Updates task metrics for the current task.

        :param metric_name: Name of the metric to update (e.g., "cpu_usage")
        :param value: New value for the metric
        """
        if self.current_task:
            self.task_metrics[self.current_task][metric_name].append(value)

    def monitor_cpu_usage(self) -> None:
        """
        Records the current CPU usage.
        Falls back to subprocess if psutil fails.
        """
        try:
            current_cpu = psutil.cpu_percent()
        except (PermissionError, AttributeError, RuntimeError):
            current_cpu = self._get_cpu_usage_fallback()
        current_time = time.time()
        self.cpu_usage_history.append(current_cpu)
        self.timestamps.append(current_time)
        self._update_task_metrics("cpu_usage", current_cpu)
        logger.debug(f"CPU Usage: {current_cpu}%")

    def _get_cpu_usage_fallback(self) -> float:
        """
        Fallback method for getting CPU usage using subprocess.
        :return: Approximate CPU usage percentage
        """
        try:
            load_avg = subprocess.check_output("uptime", shell=True).decode().split("load average:")[1].split(",")[0].strip()
            return float(load_avg) * 100 / psutil.cpu_count()
        except Exception:
            logger.error("Failed to get CPU usage using subprocess. Returning 0.0.")
            return 0.0  # Return 0.0 if unable to fetch

    def monitor_memory_usage(self) -> None:
        """
        Records the current memory usage.
        """
        current_memory = psutil.virtual_memory().percent
        self.memory_usage_history.append(current_memory)
        self._update_task_metrics("memory_usage", current_memory)
        logger.debug(f"Memory Usage: {current_memory}%")

    def monitor_task_queue_length(self, queue_size: int) -> None:
        """
        Records the current task queue length.

        :param queue_size: Current size of the task queue
        """
        self.task_queue_length_history.append(queue_size)
        logger.debug(f"Task Queue Length: {queue_size}")

    def monitor_knowledge_base_updates(self, num_updates: int) -> None:
        """
        Records the number of knowledge base updates.

        :param num_updates: Number of updates
        """
        self.knowledge_base_updates_history.append(num_updates)
        logger.debug(f"Knowledge Base Updates: {num_updates}")

    def monitor_model_training_time(self, training_time: float) -> None:
        """
        Records the model training time.

        :param training_time: Time taken for model training
        """
        self.model_training_time_history.append(training_time)
        logger.debug(f"Model Training Time: {training_time}s")

    def monitor_model_inference_time(self, inference_time: float) -> None:
        """
        Records the model inference time.

        :param inference_time: Time taken for model inference
        """
        self.model_inference_time_history.append(inference_time)
        logger.debug(f"Model Inference Time: {inference_time}s")

    def get_historical_data(self) -> Dict:
        """
        Returns all historical data.
        """
        return {
            "cpu_usage": list(self.cpu_usage_history),
            "memory_usage": list(self.memory_usage_history),
            "task_queue_length": list(self.task_queue_length_history),
            "knowledge_base_updates": list(self.knowledge_base_updates_history),
            "model_training_time": list(self.model_training_time_history),
            "model_inference_time": list(self.model_inference_time_history),
            "timestamps": list(self.timestamps),
            "task_metrics": self.task_metrics
        }

    def generate_task_report(self, task_name: str) -> Dict:
        """
        Generates a report for the specified task.

        :param task_name: Name of the task to report on
        :return: Task report dictionary
        """
        if task_name not in self.task_metrics:
            logger.warning(f"No data for task: {task_name}")
            return {"error": f"No data for task: {task_name}"}

        task_data = self.task_metrics[task_name]
        report = {
            "task_name": task_name,
            "duration": task_data["end_time"] - task_data["start_time"],
            "avg_cpu": self._calculate_average(task_data["cpu_usage"]),
            "avg_memory": self._calculate_average(task_data["memory_usage"]),
            "peak_cpu": max(task_data["cpu_usage"]) if task_data["cpu_usage"] else 0,
            "peak_memory": max(task_data["memory_usage"]) if task_data["memory_usage"] else 0
        }
        logger.info(f"Generated task report for {task_name}: {report}")
        return report

    @staticmethod
    def _calculate_average(values: list) -> float:
        """
        Calculates the average of a list of values.

        :param values: List of values
        :return: Average value
        """
        return sum(values) / len(values) if values else 0


# Example usage
if __name__ == "__main__":
    monitor = InternalProcessMonitor(max_history_size=50)

    monitor.start_task_monitoring("Task 1")
    for _ in range(10):
        monitor.monitor_cpu_usage()
        monitor.monitor_memory_usage()
        time.sleep(0.1)  # Simulate work
    monitor.end_task_monitoring()

    task_report = monitor.generate_task_report("Task 1")
    logger.info(f"Task Report: {task_report}")
    historical_data = monitor.get_historical_data()
    logger.info(f"Historical Data: {historical_data}")

    # Print statements for solo run
    print(f"Task Report: {task_report}")
    print(f"Historical Data: {historical_data}")
