# memory_manager.py
# Created on Nov13 2024
####
# Use the memory management functions as needed throughout the codebase. For example:
##
# Modified on Dec11 2024
from typing import Any, Dict
from logging_config import setup_logging

# Set up logging
logger = setup_logging(script_name="memory_manager")

class MemoryManager:
    def __init__(self):
        """Initialize memory stores for working, short-term, and long-term memories."""
        self.working_memory: Dict[str, Any] = {}
        self.short_term_memory: Dict[str, Any] = {}
        self.long_term_memory: Dict[str, Any] = {}
        logger.info("MemoryManager initialized.")

    def store_working_memory(self, key: str, value: Any) -> str:
        """Store a value in working memory."""
        self.working_memory[key] = value
        logger.info(f"Stored '{value}' under key '{key}' in Working Memory.")
        return "Working Memory Initialized"

    def store_short_term_memory(self, key: str, value: Any) -> str:
        """Store a value in short-term memory."""
        self.short_term_memory[key] = value
        logger.info(f"Stored '{value}' under key '{key}' in Short-Term Memory.")
        return "Short-Term Memory Initialized"

    def store_long_term_memory(self, key: str, value: Any) -> str:
        """Store a value in long-term memory."""
        self.long_term_memory[key] = value
        logger.info(f"Stored '{value}' under key '{key}' in Long-Term Memory.")
        return "Long-Term Memory Initialized"

    def memory_consolidation(self) -> str:
        """Consolidate memories from working and short-term to long-term."""
        # Store all values from working and short-term memories to long-term memory
        for key, value in self.working_memory.items():
            self.long_term_memory[key] = value
            logger.debug(f"Consolidated '{value}' under key '{key}' from Working Memory to Long-Term Memory.")

        for key, value in self.short_term_memory.items():
            self.long_term_memory[key] = value
            logger.debug(f"Consolidated '{value}' under key '{key}' from Short-Term Memory to Long-Term Memory.")

        # Clear the working and short-term memories after consolidation
        self.working_memory.clear()
        self.short_term_memory.clear()
        logger.info("Memory Consolidation Activated.")
        return "Memory Consolidation Activated"

    def memory_retrieval(self, key: str, memory_type: str) -> Any:
        """Retrieve a value from specified memory type."""
        if memory_type == "working":
            value = self.working_memory.get(key, None)
            logger.info(f"Retrieved '{value}' under key '{key}' from Working Memory.")
            return value
        elif memory_type == "short_term":
            value = self.short_term_memory.get(key, None)
            logger.info(f"Retrieved '{value}' under key '{key}' from Short-Term Memory.")
            return value
        elif memory_type == "long_term":
            value = self.long_term_memory.get(key, None)
            logger.info(f"Retrieved '{value}' under key '{key}' from Long-Term Memory.")
            return value
        else:
            logger.warning(f"Invalid memory type: {memory_type}")
            return None

    def get_all_memories(self) -> Dict[str, Dict[str, Any]]:
        """Return a dictionary containing all types of memories."""
        memories = {
            'working_memory': dict(self.working_memory),
            'short_term_memory': dict(self.short_term_memory),
            'long_term_memory': dict(self.long_term_memory)
        }
        logger.debug(f"Retrieved all memories: {memories}")
        return memories

# Test Suite
if __name__ == "__main__":
    logger.info("### Running Memory Manager Test Suite ###")
    
    mm = MemoryManager()
    
    logger.info("\n1. Storing in Working Memory:")
    key, value = "test_key", "Hello, World!"
    logger.info(f"Storing '{value}' under key '{key}' in Working Memory...")
    print(mm.store_working_memory(key, value))
    logger.info(f"Working Memory After Store: {mm.get_all_memories()['working_memory']}")
    
    logger.info("\n2. Storing in Short-Term Memory:")
    key, value = "short_test", 12345
    logger.info(f"Storing '{value}' under key '{key}' in Short-Term Memory...")
    print(mm.store_short_term_memory(key, value))
    logger.info(f"Short-Term Memory After Store: {mm.get_all_memories()['short_term_memory']}")
    
    logger.info("\n3. Memory Consolidation:")
    print(mm.memory_consolidation())
    logger.info(f"Memories After Consolidation:\n {mm.get_all_memories()}")
    
    logger.info("\n4. Retrieving from Long-Term Memory (after consolidation):")
    retrieval_key = "test_key"
    logger.info(f"Retrieving value for key '{retrieval_key}' from Long-Term Memory...")
    print(mm.memory_retrieval(retrieval_key, "long_term"))
    
    logger.info("\n### Test Suite Completed ###")

# end of memory management.
