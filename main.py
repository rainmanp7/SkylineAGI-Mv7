# main.py
# Updated to skip domain processing if no valid domains are found

import logging
import asyncio
import numpy as np

# Import necessary modules
from agi_config import AGIConfiguration
from internal_process_monitor import InternalProcessMonitor
from cross_domain_generalization import CrossDomainGeneralization
from complexity import AdvancedComplexityFactor
from optimization import HyperparameterOptimization
from metacognitive_manager import MetaCognitiveManager
from memory_manager import MemoryManager
from uncertainty_quantification import UncertaintyQuantification
from async_process_manager import AsyncProcessManager
from models import SkylineAGI32
from database_manager import DatabaseManager
from assimilation import AssimilationModule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    # AGI Model
    try:
        model = SkylineAGI32(weights=np.random.randn(64, 10), biases=np.zeros(10))
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
        optimizer = HyperparameterOptimization()
        print("Bayesian optimizer initialized successfully.")
    except Exception as e:
        print(f"Error initializing Bayesian optimizer: {e}")
        diagnostics_passed = False

    # Cross-Domain Generalization
    try:
        cross_domain_generalization = CrossDomainGeneralization(DatabaseManager())
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

    # Database
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
        self.complexity_analyzer = AdvancedComplexityFactor()
        self.internal_monitor = InternalProcessMonitor()
        self.cross_domain_generator = CrossDomainGeneralization(DatabaseManager())
        self.metacognitive_manager = MetaCognitiveManager()
        self.memory_manager = MemoryManager()
        self.uncertainty_quantifier = UncertaintyQuantification()
        self.async_process_manager = AsyncProcessManager()
        self.database_manager = DatabaseManager()
        self.assimilation_module = AssimilationModule(self.database_manager, self.memory_manager, self.metacognitive_manager)

    async def process_domain(self, domain: str):
        """Asynchronously process a specific domain"""
        try:
            complexity_factor = self.get_complexity_factor(domain)
            # Fetch dataset paths or data from the knowledge_base table, filtered by category
            query = "SELECT * FROM knowledge_base WHERE category = ?;"
            dataset_paths = self.database_manager.execute_query(query, (domain,))
            if not dataset_paths:
                logger.warning(f"No datasets found for domain: {domain}")
                return

            for i, dataset_path in enumerate(dataset_paths):
                await self.process_dataset(domain, dataset_path, complexity_factor, i)
        except Exception as e:
            logger.error(f"Error processing domain {domain}: {e}")

    async def process_dataset(self, domain: str, dataset_path: str, complexity: float, index: int):
        """Process individual datasets with complexity-aware optimization."""
        try:
            optimizer = HyperparameterOptimization()
            optimized_params = optimizer.perform_optimization(np.random.rand(100, 5), np.random.rand(100), np.random.rand(20, 5), np.random.rand(20))

            # Load data with optimized parameters (placeholder logic)
            loaded_data = np.random.rand(100, 5)  # Replace with actual data loading logic
            self.internal_monitor.track_dataset_processing(dataset_path, complexity)
            self.cross_domain_generator.analyze_dataset(loaded_data)

            # Database Update (Example)
            try:
                self.database_manager.update_domain_data(domain, (dataset_path, "PROCESSED"))
            except Exception as db_e:
                logger.error(f"Database update error (processed): {db_e}")
        except Exception as e:
            logger.error(f"Dataset processing error for {domain} at {dataset_path}: {e}", exc_info=True)
            try:
                self.database_manager.update_domain_data(domain, (dataset_path, "FAILED"))
            except Exception as db_e:
                logger.error(f"Database update error (failed): {db_e}")
            raise  # Re-raise the original error for further investigation

    def get_complexity_factor(self, domain: str) -> float:
        """Determine complexity factor based on domain characteristics."""
        try:
            base_complexity: float = self.config.get_dynamic_setting('complexity_factor', 10)
            domain_complexity: float = self.complexity_analyzer.calculate(np.random.rand(10), np.random.rand(10))
            return base_complexity * domain_complexity
        except Exception as e:
            logger.warning(f"Complexity calculation error for domain '{domain}': {e}", exc_info=True)
            return 10.0  # Default complexity factor on error

    async def run_metacognitive_evaluation(self):
        """Run metacognitive evaluation on processed datasets"""
        try:
            # Placeholder for metacognitive evaluation logic
            processed_datasets = []  # Replace with actual processed datasets
            if hasattr(self.metacognitive_manager, 'evaluate'):
                await self.metacognitive_manager.evaluate(processed_datasets)
            else:
                logger.error("MetacognitiveManager does not have an 'evaluate' method.")
        except Exception as e:
            logger.error(f"Metacognitive evaluation error: {e}")

    async def run_uncertainty_quantification(self):
        """Quantify uncertainty for processed datasets"""
        try:
            # Placeholder for uncertainty quantification logic
            processed_datasets = []  # Replace with actual processed datasets
            if hasattr(self.uncertainty_quantifier, 'quantify'):
                await self.uncertainty_quantifier.quantify(processed_datasets)
            else:
                logger.error("UncertaintyQuantification does not have a 'quantify' method.")
        except Exception as e:
            logger.error(f"Uncertainty quantification error: {e}")


async def main():
    """Main asynchronous execution entry point"""
    process_manager = AsyncProcessManager()
    agi = SkylineAGI()

    try:
        # Debug: Check database connection and table names
        db_manager = DatabaseManager()
        logger.info(f"Database Path: {db_manager.database_path}")
        logger.info(f"Database Connection: {'Connected' if db_manager.connection else 'Failed'}")
        table_names = db_manager.get_table_names()
        logger.info(f"Table Names: {table_names}")

        # Check if the knowledge_base table exists and has data
        if "knowledge_base" not in table_names:
            logger.warning("knowledge_base table does not exist. Skipping domain processing.")
        else:
            query = "SELECT COUNT(*) FROM knowledge_base;"
            result = db_manager.execute_query(query)
            if result and result[0][0] > 0:
                logger.info("knowledge_base table exists and contains data.")
            else:
                logger.warning("knowledge_base table is empty. Skipping domain processing.")

        # Define domains to process
        domains = ['Math', 'Science']

        # **Filter domains to skip ones without datasets (updated)**
        valid_domains = []
        for domain in domains:
            query = "SELECT * FROM knowledge_base WHERE category = ?;"
            dataset_paths = db_manager.execute_query(query, (domain,))
            if dataset_paths:
                valid_domains.append(domain)
        logger.info(f"Valid Domains: {valid_domains}")

        if not valid_domains:
            logger.warning("No valid domains with datasets found. Skipping domain processing.")
        else:
            # Parallel optimization for valid domains
            tasks = [
                asyncio.create_task(
                    agi.process_domain(domain)
                )
                for domain in valid_domains
            ]
            await asyncio.gather(*tasks)

        # Run metacognitive evaluation and uncertainty quantification
        await agi.run_metacognitive_evaluation()
        await agi.run_uncertainty_quantification()

    except Exception as e:
        logger.error(f"Main execution error: {e}", exc_info=True)


async def run_monitoring(internal_monitor: InternalProcessMonitor, process_manager: AsyncProcessManager):
    """Background monitoring loop"""
    try:
        last_update_count = 0

        while True:
            internal_monitor.monitor_cpu_usage()
            internal_monitor.monitor_memory_usage()

            if not process_manager.task_queue.empty():
                internal_monitor.monitor_task_queue_length(process_manager.task_queue.qsize())

            # Placeholder for knowledge base updates (replace with actual logic)
            current_update_count = 0
            internal_monitor.monitor_knowledge_base_updates(current_update_count - last_update_count)
            last_update_count = current_update_count

            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Monitoring error: {e}", exc_info=True)
    except asyncio.CancelledError:
        logger.info("Monitoring task canceled. Shutting down...")


if __name__ == "__main__":
    if not run_startup_diagnostics():
        print("Startup diagnostics failed. Exiting the application.")
        exit(1)

    print("Loading is complete.")  

    try:
        # Create a new event loop and run the main application
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run monitoring in the background
        monitoring_task = loop.create_task(run_monitoring(InternalProcessMonitor(), AsyncProcessManager()))
        
        # Run the main application
        loop.run_until_complete(main())
        
        # Cancel the monitoring task when the main application finishes
        monitoring_task.cancel()
        loop.run_until_complete(asyncio.sleep(0.1))  # Allow for cleanup
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Exiting the application.")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
    finally:
        loop.close()
        
