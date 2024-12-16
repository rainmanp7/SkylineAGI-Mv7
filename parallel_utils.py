# parallel_utils.py
import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import os
import logging
import time
import random

# Configure logging (adjustable when importing)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default level; can be changed when importing

# Create a handler for logging (e.g., to console or file)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)  # Default level for the handler

# Create a formatter and attach it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger (only if run standalone)
if __name__ == "__main__":
    logger.addHandler(handler)

@dataclass
class LearningTask:
    strategy_name: str
    data: Any
    parameters: Dict[str, Any]
    priority: int = 0

class AsyncParallelExecutor:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or os.cpu_count()
        self.executor = None  
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.results: Dict[str, Any] = {}
        self.total_execution_time = 0  # Initialize total execution time
        
    async def submit_task(self, task: LearningTask) -> None:
        await self.task_queue.put((task.priority, task))
        logger.info(f"Submitted task: {task.strategy_name} with priority {task.priority}")

    async def process_task(self, task: LearningTask) -> Any:
        loop = asyncio.get_event_loop()
        try:
            logger.info(f"Processing task: {task.strategy_name}")
            result = await loop.run_in_executor(
                self.executor,  
                self._execute_strategy,
                task
            )
            self.results[task.strategy_name] = result
            logger.info(f"Task {task.strategy_name} completed with result: {result}")
            self.total_execution_time += float(result['execution_time'].split(' ')[0])  # Update total execution time
            return result
        except Exception as e:
            logger.error(
                f"Error processing task {task.strategy_name}: {str(e)}",
                exc_info=True
            )
            return None
                
    @staticmethod
    def _execute_strategy(task: LearningTask) -> Any:
        try:
            duration = random.uniform(1, 5)  
            time.sleep(duration)
            return {
                'strategy': task.strategy_name,
                'status': 'completed',
                'parameters': task.parameters,
                'execution_time': f"{duration:.2f} seconds"  
            }
        except Exception as e:
            logger.error(f"Strategy execution error for {task.strategy_name}: {str(e)}", exc_info=True)
            return {"strategy": task.strategy_name, "status": "failed", "error": str(e)}
            
    async def run_parallel_tasks(self, tasks: List[LearningTask]) -> Dict[str, Any]:
        for task in tasks:
            await self.submit_task(task)
            
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        processors = []
        while not self.task_queue.empty():
            priority, task = await self.task_queue.get()
            processors.append((priority, self.process_task(task)))
        
        processors.sort(key=lambda x: x[0])
        await asyncio.gather(*[p[1] for p in processors])
        
        self.executor.shutdown()  
        return self.results
        
    async def cleanup(self) -> None:
        self.results.clear()

# Usage example (only runs when executed directly)
async def main():
    executor = AsyncParallelExecutor()
    
    tasks = [
        LearningTask("Strategy A", data=None, parameters={'param1': 1}, priority=1),
        LearningTask("Strategy B", data=None, parameters={'param2': 2}, priority=2),
        LearningTask("Strategy C", data=None, parameters={'param3': 3}, priority=3)
    ]
    
    try:
        results = await executor.run_parallel_tasks(tasks)
    finally:
        await executor.cleanup()
    
    # Enhanced Console Output
    logger.info("### Execution Summary ###")
    logger.info("-----------------------------------------")
    for strategy, result in results.items():
        if result is not None:  
            logger.info(f"**Strategy:** {strategy}")
            logger.info(f"  - **Status:** {result.get('status', 'N/A')}")
            logger.info(f"  - **Parameters:** {result.get('parameters', {})}")
            logger.info(f"  - **Execution Time:** {result.get('execution_time', 'N/A')}")
            logger.info("-----------------------------------------")
    logger.info(f"### Total Execution Time:** {executor.total_execution_time:.2f} seconds")
    logger.info("### Solo Run Completed ###")

# Run the executor (only when executed directly)
if __name__ == "__main__":
    asyncio.run(main())
