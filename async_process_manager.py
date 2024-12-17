# async_process_manager.py
# Updated Dec 14, 2024

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Any, Callable
from concurrent.futures import ProcessPoolExecutor
import resource
import psutil
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class ProcessTask:
    name: str
    priority: int
    function: Callable
    args: tuple
    kwargs: dict
    max_retries: int = 3
    current_retries: int = 0

class InternalProcessMonitor:
    def on_task_submitted(self, task: ProcessTask) -> None:
        logging.info(f"Task {task.name} submitted with priority {task.priority}")

    def on_task_completed(self, task: ProcessTask, result: Any) -> None:
        logging.info(f"Task {task.name} completed with result {result}")

    def on_task_failed(self, task: ProcessTask, error: Exception) -> None:
        logging.error(f"Task {task.name} failed with error {error}")

    def on_cleanup(self) -> None:
        logging.info("Cleanup completed")

class AsyncProcessManager:
    def __init__(self, max_workers: int = None, memory_limit: float = 0.8):
        self.max_workers = max_workers or os.cpu_count()
        self.memory_limit = memory_limit
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_queue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, Any] = {}
        self.process_monitor = InternalProcessMonitor()

    async def submit_task(self, task: ProcessTask) -> None:
        self.process_monitor.on_task_submitted(task)
        await self.task_queue.put((task.priority, task))

    async def _execute_task(self, task: ProcessTask) -> Any:
        try:
            if not await self._check_resources():
                await asyncio.sleep(1)
                return
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.process_pool, task.function, *task.args, **task.kwargs)
            self.process_monitor.on_task_completed(task, result)
            self.results[task.name] = result
            return result
        except Exception as e:
            self.process_monitor.on_task_failed(task, e)
            task.current_retries += 1
            logging.error(f"Task {task.name} failed (attempt {task.current_retries}): {str(e)}", exc_info=True)
            if task.current_retries >= task.max_retries:
                raise
            await asyncio.sleep(1)

    async def run_tasks(self) -> Dict[str, Any]:
        while not self.task_queue.empty():
            if not await self._check_resources():
                await asyncio.sleep(1)
                continue
            _, task = await self.task_queue.get()
            self.active_tasks[task.name] = asyncio.create_task(self._execute_task(task))
        await asyncio.gather(*self.active_tasks.values())
        return self.results

    async def cleanup(self) -> None:
        for task in self.active_tasks.values():
            task.cancel()
        self.process_pool.shutdown()
        self.results.clear()
        self.process_monitor.on_cleanup()

    async def _check_resources(self) -> bool:
        """Check if system has enough resources to start new task."""
        memory_percent = psutil.virtual_memory().percent / 100
        return memory_percent < self.memory_limit

# Solo mode for testing
if __name__ == "__main__":
    # Define a sample function to be executed by the task
    def sample_function(x: int, y: int) -> int:
        return x + y

    # Create an instance of AsyncProcessManager
    async_process_manager = AsyncProcessManager()

    # Create a sample task
    sample_task = ProcessTask(
        name="SampleTask",
        priority=1,
        function=sample_function,
        args=(3, 4),
        kwargs={}
    )

    # Run the async process manager in a solo mode
    async def main():
        # Submit the sample task
        await async_process_manager.submit_task(sample_task)

        # Run the tasks
        results = await async_process_manager.run_tasks()

        # Log the results
        logging.info(f"Results: {results}")

        # Cleanup
        await async_process_manager.cleanup()

    # Run the main function
    asyncio.run(main())
