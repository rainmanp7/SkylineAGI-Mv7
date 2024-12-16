
# Begin main.py

import logging
import asyncio
from typing import List
import json

# Import necessary modules
from domain_knowledge_base import DomainKnowledgeBase
from agi_config import AGIConfiguration
from internal_process_monitor import InternalProcessMonitor
from cross_domain_generalization import CrossDomainGeneralization
from complexity import ComplexityAnalyzer
from optimization import adjust_search_space, parallel_bayesian_optimization
from knowledge_base import TieredKnowledgeBase, KnowledgeBase
from main_knowledge_base import MainKnowledgeBase
from metacognitive_manager import MetaCognitiveManager
from memory_manager import MemoryManager
from uncertainty_quantification import UncertaintyQuantification
from async_process_manager import AsyncProcessManager
from models import ProcessTask, model_validator, SkylineAGIModel
from parallel_bayesian_optimization_wrks import BayesianOptimizer, ParallelBayesianOptimization
from database import DatabaseManager  # Assuming a DatabaseManager class

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# **Added get_dataset_path function for centralized dataset management**
def get_dataset_path(domain: str, level: int) -> str:
    """
    Retrieves the dataset path from domain_dataset.json.
    
    Args:
    - domain (str): Domain of the dataset (e.g., 'Math', 'Science')
    - level (int): Level of the dataset (e.g., 1, 2, 3)
    
    Returns:
    - dataset_path (str): Path to the specified dataset
    """
    with open('domain_dataset.json') as f:
        datasets = json.load(f)
    # Construct the key to match the updated json structure
    key = f"level_{level}"
    return datasets.get(domain, {}).get(key, None)  # Returns None if not found

# Run Startup Diagnostics
def run_startup_diagnostics() -> bool:
    """Perform a series of startup diagnostics to ensure the system is operational."""
    print("Running startup diagnostics...")
    diagnostics_passed = True
    
    # Configuration
    try:
        config = AGIConfiguration()
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        diagnostics_passed = False

    # Knowledge Base
    try:
        knowledge_base = KnowledgeBase()
        print("Knowledge base initialized successfully.")
    except Exception as e:
        print(f"Error initializing knowledge base: {e}")
        diagnostics_passed = False

    # AGI Model
    try:
        model = SkylineAGIModel(config)
        print("AGI model created successfully.")
    except Exception as e:
        print(f"Error creating AGI model: {e}")
        diagnostics_passed = False

    # Process Monitor
    try:
        process_monitor = InternalProcessMonitor()
        print("Process monitor initialized successfully.")
    except Exception as e:
        print(f"Error initializing process monitor: {e}")
        diagnostics_passed = False

    # Bayesian Optimizer
    try:
        optimizer = BayesianOptimizer()  
        print("Bayesian optimizer initialized successfully.")
    except Exception as e:
        print(f"Error initializing Bayesian optimizer: {e}")
        diagnostics_passed = False

    # Cross-Domain Generalization
    try:
        cross_domain_generalization = CrossDomainGeneralization()
        print("Cross-domain generalization initialized successfully.")
    except Exception as e:
        print(f"Error initializing cross-domain generalization: {e}")
        diagnostics_passed = False

    # Metacognitive Manager
    try:
        metacognitive_manager = MetaCognitiveManager()
        print("Metacognitive manager initialized successfully.")
    except Exception as e:
        print(f"Error initializing metacognitive manager: {e}")
        diagnostics_passed = False

    # Database (Assuming DatabaseManager class)
    try:
        db_manager = DatabaseManager()
        print("Database connection established successfully.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        diagnostics_passed = False

    print("Startup diagnostics completed with result: SUCCESS" if diagnostics_passed else "FAILURE")
    return diagnostics_passed


class SkylineAGI:
    def __init__(self):
        self.config = AGIConfiguration()
        self.knowledge_base = DomainKnowledgeBase()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.internal_monitor = InternalProcessMonitor()
        self.cross_domain_generator = CrossDomainGeneralization(self.knowledge_base, self.config)
        self.metacognitive_manager = MetaCognitiveManager()
        self.memory_manager = MemoryManager()
        self.uncertainty_quantifier = UncertaintyQuantification()
        self.async_process_manager = AsyncProcessManager()
        self.database_manager = DatabaseManager()  # Database integration

    # **Updated process_domain to utilize get_dataset_path**
    async def process_domain(self, domain: str):
        """Asynchronously process a specific domain"""
        try:
            complexity_factor = self.get_complexity_factor(domain)
            # **Using get_dataset_path for centralized management**
            dataset_paths = [get_dataset_path(domain, level) for level in range(1, 4)]  # Assuming levels 1-3
            for i, dataset_path in enumerate(dataset_paths):
                await self.process_dataset(domain, dataset_path, complexity_factor, i)
        except Exception as e:
            logger.error(f"Error processing domain {domain}: {e}")



# Begin Updated Dec5 th 
    # **Updated process_dataset to reflect changesUpdated Code)**

async def process_dataset(self, domain: str, dataset_path: str, complexity: float, index: int):
    """
    Process individual datasets with complexity-aware optimization.

    Args:
    - domain (str): Domain of the dataset.
    - dataset_path (str): Path to the dataset.
    - complexity (float): Complexity factor for the dataset.
    - index (int): Index of the dataset (e.g., level).

    Returns:
    - None
    """
    try:
        optimizer = BayesianOptimizer()  
        optimized_params = optimizer.optimize(dataset_path, complexity)  

        # Load data with optimized parameters
        loaded_data = self.knowledge_base.load_domain_dataset(domain, index, optimized_params)
        self.internal_monitor.track_dataset_processing(dataset_path, complexity)
        self.cross_domain_generator.analyze_dataset(loaded_data)
        
        # Database Update (Example)
        try:
            self.database_manager.update_dataset_status(domain, dataset_path, "PROCESSED")
        except Exception as db_e:
            logger.error(f"Database update error (processed): {db_e}")
    except Exception as e:
        logger.error(f"Dataset processing error for {domain} at {dataset_path}: {e}", exc_info=True)
        # Database Update on Failure (with separate try-except)
        try:
            self.database_manager.update_dataset_status(domain, dataset_path, "FAILED")
        except Exception as db_e:
            logger.error(f"Database update error (failed): {db_e}")
        raise  # Re-raise the original error for further investigation
# end Dec5 update

#dec5 2nd part
    def get_complexity_factor(self, domain: str) -> float:
    """
    Determine complexity factor based on domain characteristics.

    Args:
    - domain (str): The domain for which to calculate the complexity factor.

    Returns:
    - float: The calculated complexity factor (base complexity * domain complexity).
    """
    try:
        base_complexity: float = self.config.get_dynamic_setting('complexity_factor', 10)
        domain_complexity: float = self.complexity_analyzer.analyze_domain(domain)
        return base_complexity * domain_complexity
    except Exception as e:
        logger.warning(f"Complexity calculation error for domain '{domain}': {e}", exc_info=True)
        return 10.0  # Default complexity factor on error

    async def run_metacognitive_evaluation(self):
        """Run metacognitive evaluation on processed datasets"""
        try:
            await self.metacognitive_manager.evaluate(self.knowledge_base.get_processed_datasets())
        except Exception as e:
            logger.error(f"Metacognitive evaluation error: {e}")

    async def run_uncertainty_quantification(self):
        """Quantify uncertainty for processed datasets"""
        try:
            await self.uncertainty_quantifier.quantify(self.knowledge_base.get_processed_datasets())
        except Exception as e:
            logger.error(f"Uncertainty quantification error: {e}")


async def main():
    """Main asynchronous execution entry point"""
    process_manager = AsyncProcessManager()
    agi = SkylineAGI()

    try:
        # Define domains to process
        domains = ['Math', 'Science']  

        # **Filter domains to skip ones without datasets (updated)**
        valid_domains = [domain for domain in domains if any(get_dataset_path(domain, level) for level in range(1, 4))]

        if not valid_domains:
            logger.warning("No valid domains with datasets found. Exiting...")
            return

        # Parallel optimization for valid domains
        tasks = [
            asyncio.create_task(
                agi.process_domain(domain)
            )
            for domain in valid_domains
        ]

        # Run metacognitive evaluation and uncertainty quantification in parallel
        tasks.append(asyncio.create_task(agi.run_metacognitive_evaluation()))
        tasks.append(asyncio.create_task(agi.run_uncertainty_quantification()))

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    except Exception as e:
        logger.error(f"Main execution error: {e}", exc_info=True)


async def run_monitoring(internal_monitor: InternalProcessMonitor, process_manager: AsyncProcessManager, knowledge_base: KnowledgeBase):
    """Background monitoring loop"""
    try:
        last_update_count = 0

        while True:
            internal_monitor.monitor_cpu_usage()
            internal_monitor.monitor_memory_usage()

            if not process_manager.task_queue.empty():
                internal_monitor.monitor_task_queue_length(process_manager.task_queue.qsize())

            current_update_count = len(knowledge_base.get_recent_updates())
            internal_monitor.monitor_knowledge_base_updates(current_update_count - last_update_count)
            last_update_count = current_update_count

            if hasattr(model_validator, 'metrics_history') and "model_key" in model_validator.metrics_history:
                metrics = model_validator.metrics_history["model_key"][-1]
                internal_monitor.monitor_model_training_time(metrics.training_time)
                internal_monitor.monitor_model_inference_time(metrics.prediction_latency)

            await asyncio.sleep(1)

    except asyncio.CancelledError:
        logger.info("Monitoring task canceled. Shutting down...")


if __name__ == "__main__":
    if not run_startup_diagnostics():
        print("Startup diagnostics failed. Exiting the application.")
        exit(1)

    print("Loading is complete.")  

    try:
        # Run monitoring in the background
        monitoring_task = asyncio.create_task(run_monitoring(InternalProcessMonitor(), AsyncProcessManager(), KnowledgeBase()))
        
        # Run the main application
        asyncio.run(main())
        
        # Cancel the monitoring task when the main application finishes
        monitoring_task.cancel()
        asyncio.run(asyncio.sleep(0.1))  # Allow for cleanup
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Exiting the application.")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")

# End main.py