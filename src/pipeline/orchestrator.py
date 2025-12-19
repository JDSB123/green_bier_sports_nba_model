"""Simple but robust pipeline orchestration system.

This module provides a lightweight orchestration framework for running
the NBA prediction pipeline with proper error handling, retries, and logging.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional, List, Dict, Any
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Status of a pipeline task."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of a pipeline task execution."""
    name: str
    status: TaskStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    output: Optional[Any] = None


@dataclass
class Task:
    """A task in the pipeline."""
    name: str
    func: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 2
    continue_on_failure: bool = False
    skip_if: Optional[Callable[[], bool]] = None


class Pipeline:
    """A pipeline orchestrator for managing task execution."""

    def __init__(self, name: str):
        """Initialize the pipeline.

        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self.results: Dict[str, TaskResult] = {}
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None

    def add_task(
        self,
        name: str,
        func: Callable,
        dependencies: Optional[List[str]] = None,
        max_retries: int = 2,
        continue_on_failure: bool = False,
        skip_if: Optional[Callable[[], bool]] = None,
    ) -> Task:
        """Add a task to the pipeline.

        Args:
            name: Unique name for the task
            func: Function to execute (can be sync or async)
            dependencies: List of task names that must complete first
            max_retries: Maximum number of retry attempts
            continue_on_failure: If True, pipeline continues even if task fails
            skip_if: Optional function that returns True if task should be skipped

        Returns:
            The created Task object
        """
        task = Task(
            name=name,
            func=func,
            dependencies=dependencies or [],
            max_retries=max_retries,
            continue_on_failure=continue_on_failure,
            skip_if=skip_if,
        )
        self.tasks[name] = task
        return task

    async def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task with retry logic.

        Args:
            task: The task to execute

        Returns:
            TaskResult with execution details
        """
        result = TaskResult(name=task.name, status=TaskStatus.PENDING)

        # Check if task should be skipped
        if task.skip_if and task.skip_if():
            logger.info(f"Skipping task: {task.name} (skip condition met)")
            result.status = TaskStatus.SKIPPED
            return result

        result.started_at = datetime.now(timezone.utc).isoformat()
        result.status = TaskStatus.RUNNING
        
        logger.info(f"Starting task: {task.name}")

        attempt = 0
        last_error = None

        while attempt <= task.max_retries:
            try:
                # Execute the task function
                if asyncio.iscoroutinefunction(task.func):
                    output = await task.func()
                else:
                    output = task.func()

                # Success
                result.status = TaskStatus.SUCCESS
                result.output = output
                result.completed_at = datetime.now(timezone.utc).isoformat()
                
                if result.started_at:
                    start_time = datetime.fromisoformat(result.started_at)
                    end_time = datetime.fromisoformat(result.completed_at)
                    result.duration_seconds = (end_time - start_time).total_seconds()

                logger.info(f"Task completed successfully: {task.name} ({result.duration_seconds:.2f}s)")
                return result

            except Exception as e:
                last_error = str(e)
                attempt += 1
                
                if attempt <= task.max_retries:
                    logger.warning(
                        f"Task {task.name} failed (attempt {attempt}/{task.max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Task {task.name} failed after {attempt} attempts: {e}")

        # All retries exhausted
        result.status = TaskStatus.FAILED
        result.error = last_error
        result.completed_at = datetime.now(timezone.utc).isoformat()
        
        if result.started_at:
            start_time = datetime.fromisoformat(result.started_at)
            end_time = datetime.fromisoformat(result.completed_at)
            result.duration_seconds = (end_time - start_time).total_seconds()

        return result

    def _check_dependencies(self, task: Task) -> bool:
        """Check if all dependencies for a task have completed successfully.

        Args:
            task: The task to check dependencies for

        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        for dep_name in task.dependencies:
            if dep_name not in self.results:
                return False
            
            result = self.results[dep_name]
            
            # Dependency must be successful or skipped
            if result.status not in [TaskStatus.SUCCESS, TaskStatus.SKIPPED]:
                return False

        return True

    async def run(self) -> Dict[str, TaskResult]:
        """Execute the pipeline.

        This will:
        1. Execute tasks in dependency order
        2. Retry failed tasks according to their configuration
        3. Stop execution if a critical task fails (continue_on_failure=False)

        Returns:
            Dictionary of task names to TaskResults
        """
        self.started_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Starting pipeline: {self.name}")
        logger.info(f"Tasks to execute: {', '.join(self.tasks.keys())}")

        executed = set()
        failed_critical = False

        while len(executed) < len(self.tasks) and not failed_critical:
            # Find tasks ready to execute
            ready_tasks = []
            
            for task_name, task in self.tasks.items():
                if task_name in executed:
                    continue
                
                if self._check_dependencies(task):
                    ready_tasks.append(task)

            if not ready_tasks:
                # No tasks ready - check if we're stuck
                if len(executed) < len(self.tasks):
                    remaining = set(self.tasks.keys()) - executed
                    # Find which dependencies are blocking
                    blocking_info = []
                    for task_name in remaining:
                        task = self.tasks[task_name]
                        failed_deps = [
                            dep for dep in task.dependencies
                            if dep in self.results and self.results[dep].status == TaskStatus.FAILED
                        ]
                        if failed_deps:
                            blocking_info.append(f"{task_name} (blocked by failed: {', '.join(failed_deps)})")
                        else:
                            missing_deps = [dep for dep in task.dependencies if dep not in self.results]
                            if missing_deps:
                                blocking_info.append(f"{task_name} (missing dependencies: {', '.join(missing_deps)})")
                    
                    logger.error(f"Pipeline stuck. Remaining tasks cannot execute: {remaining}")
                    if blocking_info:
                        logger.error(f"Blocking details: {'; '.join(blocking_info)}")
                    
                    # Mark remaining tasks as failed
                    for task_name in remaining:
                        task = self.tasks[task_name]
                        failed_deps = [dep for dep in task.dependencies if dep in self.results and self.results[dep].status == TaskStatus.FAILED]
                        error_msg = f"Dependencies not satisfied"
                        if failed_deps:
                            error_msg = f"Blocked by failed dependencies: {', '.join(failed_deps)}"
                        self.results[task_name] = TaskResult(
                            name=task_name,
                            status=TaskStatus.FAILED,
                            error=error_msg,
                        )
                        executed.add(task_name)
                break

            # Execute ready tasks
            for task in ready_tasks:
                result = await self._execute_task(task)
                self.results[task.name] = result
                executed.add(task.name)

                # Check if we should stop
                if result.status == TaskStatus.FAILED and not task.continue_on_failure:
                    logger.error(f"Critical task failed: {task.name}. Stopping pipeline.")
                    failed_critical = True
                    
                    # Mark remaining tasks as skipped
                    for task_name in self.tasks:
                        if task_name not in executed:
                            self.results[task_name] = TaskResult(
                                name=task_name,
                                status=TaskStatus.SKIPPED,
                                error="Pipeline stopped due to critical failure",
                            )
                            executed.add(task_name)
                    break

        self.completed_at = datetime.now(timezone.utc).isoformat()
        
        # Log summary
        self._log_summary()
        
        return self.results

    def _log_summary(self) -> None:
        """Log a summary of the pipeline execution."""
        if not self.started_at or not self.completed_at:
            return

        start_time = datetime.fromisoformat(self.started_at)
        end_time = datetime.fromisoformat(self.completed_at)
        total_duration = (end_time - start_time).total_seconds()

        success_count = sum(1 for r in self.results.values() if r.status == TaskStatus.SUCCESS)
        failed_count = sum(1 for r in self.results.values() if r.status == TaskStatus.FAILED)
        skipped_count = sum(1 for r in self.results.values() if r.status == TaskStatus.SKIPPED)

        logger.info("=" * 80)
        logger.info(f"Pipeline: {self.name}")
        logger.info(f"Total duration: {total_duration:.2f}s")
        logger.info(f"Tasks: {len(self.tasks)} total, {success_count} succeeded, {failed_count} failed, {skipped_count} skipped")
        
        # Log individual task results
        for task_name, result in self.results.items():
            status_symbol = "✓" if result.status == TaskStatus.SUCCESS else "✗" if result.status == TaskStatus.FAILED else "-"
            duration_str = f"{result.duration_seconds:.2f}s" if result.duration_seconds else "N/A"
            logger.info(f"  {status_symbol} {task_name}: {result.status.value} ({duration_str})")
            
            if result.error:
                logger.error(f"    Error: {result.error}")
        
        logger.info("=" * 80)


async def main():
    """Example usage of the pipeline orchestrator."""
    pipeline = Pipeline(name="Example Pipeline")

    # Add tasks
    pipeline.add_task(
        name="task1",
        func=lambda: print("Task 1 executed"),
    )

    pipeline.add_task(
        name="task2",
        func=lambda: print("Task 2 executed"),
        dependencies=["task1"],
    )

    pipeline.add_task(
        name="task3",
        func=lambda: print("Task 3 executed"),
        dependencies=["task1"],
    )

    pipeline.add_task(
        name="task4",
        func=lambda: print("Task 4 executed"),
        dependencies=["task2", "task3"],
    )

    # Run pipeline
    results = await pipeline.run()
    
    return results


if __name__ == "__main__":
    asyncio.run(main())

