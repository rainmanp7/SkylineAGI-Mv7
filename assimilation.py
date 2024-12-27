# assimilation.py
# Updated Dec 27, 2024

from typing import Tuple, Any
from database_manager import DatabaseManager
from memory_manager import MemoryManager
from metacognitive_manager import MetaCognitiveManager
from complexity import AdvancedComplexityFactor
from logging_config import setup_logging
import numpy as np

# Set up logging
logger = setup_logging(script_name="assimilation")

class AssimilationModule:
    def __init__(self, db_manager: DatabaseManager, memory_manager: MemoryManager, metaCognitive_manager: MetaCognitiveManager):
        """
        Initialize the AssimilationModule with necessary components.

        Args:
            db_manager (DatabaseManager): Manages database operations.
            memory_manager (MemoryManager): Manages memory storage and retrieval.
            metaCognitive_manager (MetaCognitiveManager): Manages metacognitive processes.
        """
        self.db_manager = db_manager
        self.memory_manager = memory_manager
        self.metacognitive_manager = metaCognitive_manager
        self.complexity_factor = AdvancedComplexityFactor()

    def assimilate(self, data: Tuple[Any, Any]) -> None:
        """
        Assimilate incoming data based on its calculated complexity.

        Args:
            data (Tuple[Any, Any]): A tuple containing input data (X) and labels (y).
        """
        try:
            X, y = data

            # Flatten the input data if it's a list
            if isinstance(X, list):
                X_flattened = np.array(X).flatten().tolist()  # Convert to a flattened list
            else:
                X_flattened = X

            # Calculate the complexity of the data
            calculated_complexity = self.complexity_factor.calculate(X_flattened, y)

            # Determine the domain of the data (example logic)
            domain = self._determine_domain(X_flattened, y)

            # Prepare data for database insertion
            data_tuple = (
                domain,  # domain
                "sample_topic",  # topic (replace with actual topic extraction logic)
                "sample_description",  # description (replace with actual description extraction logic)
                calculated_complexity,  # complexity
                '{"source": "sample_source", "tags": ["sample_tag"]}',  # metadata
                "2024-12-27 22:44:28",  # timestamp (replace with actual timestamp)
                "medium"  # difficulty (replace with actual difficulty calculation)
            )

            # Store the data in the database
            self.db_manager.update_domain_data("knowledge_base", data_tuple)

            # Store the new knowledge in memory for efficient retrieval
            self._store_in_memory(domain, calculated_complexity, (X_flattened, y))

            logger.info(f"Data assimilated successfully for domain: {domain}")
        except Exception as e:
            logger.error(f"Error during assimilation: {e}", exc_info=True)

    def _determine_domain(self, X: Any, y: Any) -> str:
        """
        Determine the domain of the input data (X, y).

        Args:
            X: Input data.
            y: Labels.

        Returns:
            str: The domain of the data (e.g., "science", "finance").
        """
        # Example logic: Determine domain based on the content of X or y
        if "science" in str(X) or "science" in str(y):
            return "science"
        elif "finance" in str(X) or "finance" in str(y):
            return "finance"
        else:
            return "unknown"

    def _store_in_memory(self, domain: str, complexity: float, data: Tuple[Any, Any]) -> None:
        """
        Store data in the appropriate memory store.

        Args:
            domain (str): The domain of the data.
            complexity (float): The calculated complexity of the data.
            data (Tuple[Any, Any]): The input data and labels.
        """
        # Use the domain as the key for memory storage
        key = f"{domain}_{complexity}"

        # Store in working memory (example logic)
        self.memory_manager.store_working_memory(key, data)

        # Optionally, store in short-term or long-term memory as needed
        # self.memory_manager.store_short_term_memory(key, data)
        # self.memory_manager.store_long_term_memory(key, data)

# Solo run test
def test_assimilation_module():
    """
    Test the AssimilationModule to ensure it works as expected.
    """
    logger.info("Starting solo run test for AssimilationModule...")

    # Initialize components
    db_manager = DatabaseManager()
    memory_manager = MemoryManager()
    metacognitive_manager = MetaCognitiveManager()

    # Create AssimilationModule instance
    assimilation_module = AssimilationModule(db_manager, memory_manager, metacognitive_manager)

    # Example data
    X = [[1, 2], [3, 4]]
    y = [0, 1]

    # Assimilate data
    logger.info("Assimilating sample data...")
    assimilation_module.assimilate((X, y))

    # Verify data was stored in memory
    logger.info("Verifying data in memory...")
    memory_data = memory_manager.get_all_memories()
    logger.info(f"Memory data: {memory_data}")

    # Verify data was stored in the database
    logger.info("Verifying data in the database...")
    db_data = db_manager.load_domain_dataset("knowledge_base")  # Replace with the correct table name
    logger.info(f"Database data: {db_data}")

    logger.info("Solo run test completed successfully!")

# Run the test if this file is executed directly
if __name__ == "__main__":
    test_assimilation_module()