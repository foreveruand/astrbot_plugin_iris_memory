"""
Scheduled Tasks Manager - Manages LLM-related scheduled tasks via AstrBot's CronJobManager

Tasks managed:
- memory_promotion: Memory upgrade evaluation (LLM)
- semantic_extraction: Semantic clustering and extraction (LLM)
- persona_batch_flush: Persona extraction batch (LLM)
- kg_auto_flush: Knowledge graph triple extraction batch (LLM)

All tasks are persistent (survive restarts) and use unique names prefixed with 'iris_memory_'.

Configuration Hot-Reload:
- Automatically reloads when 'scheduled_tasks' or 'rate_control' config changes
- Use reload() method for manual config reload
- Automatically updates or creates tasks as needed
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from iris_memory.config import get_store
from iris_memory.config.events import config_events
from iris_memory.core.constants import (
    RateControlDefaults,
    ScheduledTaskDefaults,
)
from iris_memory.utils.llm_rate_controller import LLMRateController
from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from astrbot.core.star.context import Context
    from iris_memory.services.memory_service import MemoryService

logger = get_logger("scheduled_tasks")


class ScheduledTaskManager:
    """Manages registration and execution of LLM-related scheduled tasks.

    Task Deduplication:
    - Uses task name as unique identifier (e.g., 'iris_memory_memory_promotion')
    - Checks existing jobs by name before registering
    - Updates existing jobs if cron expression changes

    Configuration Hot-Reload:
    - Subscribes to 'scheduled_tasks' and 'rate_control' config changes
    - Automatically updates task schedules when config changes
    - Call reload(config) for manual config reload
    """

    def __init__(self, context: Context, service: MemoryService) -> None:
        """Initialize the scheduled task manager.

        Args:
            context: AstrBot context (contains cron_manager)
            service: MemoryService instance for task handlers
        """
        self._context = context
        self._service = service
        self._task_name_to_job_id: dict[str, str] = {}
        self._task_key_to_name: dict[str, str] = {}
        self._initialized = False
        self._config: Any = None
        self._unsubscribe_fns: list[Callable[[], None]] = []

    async def initialize(self, config: Any) -> None:
        """Initialize rate controller and register all enabled tasks.

        Args:
            config: Plugin configuration object
        """
        if self._initialized:
            logger.warning(
                "ScheduledTaskManager already initialized, call reload() instead"
            )
            return

        self._config = config
        self._init_rate_controller(config)
        await self._register_all_tasks(config)
        self._subscribe_config_events()

        self._initialized = True
        logger.info(
            f"ScheduledTaskManager initialized with {len(self._task_name_to_job_id)} tasks"
        )

    def _subscribe_config_events(self) -> None:
        """Subscribe to config change events for hot-reload."""
        unsub1 = config_events.on_section(
            "scheduled_tasks", self._on_scheduled_tasks_change
        )
        unsub2 = config_events.on_section("rate_control", self._on_rate_control_change)

        self._unsubscribe_fns = [unsub1, unsub2]

    def _on_scheduled_tasks_change(
        self, key: str, old_value: Any, new_value: Any
    ) -> None:
        """Handle scheduled_tasks config change.

        Uses get_store() to get the latest config values since the event
        is triggered after config is updated.
        """
        logger.info(
            f"Config changed: {key} ({old_value} -> {new_value}), "
            "triggering scheduled tasks reload"
        )
        asyncio.create_task(self._async_reload_from_store())

    def _on_rate_control_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """Handle rate_control config change."""
        logger.info(
            f"Config changed: {key} ({old_value} -> {new_value}), "
            "triggering rate controller reload"
        )
        store = get_store()
        self._init_rate_controller(store)

    async def _async_reload_from_store(self) -> None:
        """Reload from global ConfigStore (for event handlers)."""
        store = get_store()
        await self.reload(store)

    async def reload(self, config: Any) -> None:
        """Reload task configuration.

        This method should be called when configuration changes.
        It will:
        - Update rate controller settings
        - Enable/disable tasks based on config
        - Update cron expressions for existing tasks
        - Create new tasks if needed

        Args:
            config: Plugin configuration object
        """
        logger.info("Reloading scheduled tasks configuration...")

        self._config = config
        self._init_rate_controller(config)

        await self._sync_tasks_with_config(config)

        logger.info(
            f"Scheduled tasks reloaded: {len(self._task_name_to_job_id)} active tasks"
        )

    def _init_rate_controller(self, config: Any) -> None:
        """Initialize LLM rate controller with config values."""
        max_concurrent = config.get(
            "rate_control.max_concurrent_calls",
            RateControlDefaults.MAX_CONCURRENT_CALLS,
        )
        min_interval_ms = config.get(
            "rate_control.min_call_interval_ms",
            RateControlDefaults.MIN_CALL_INTERVAL_MS,
        )

        LLMRateController.configure(
            max_concurrent=max_concurrent,
            min_interval_ms=min_interval_ms,
        )
        logger.info(
            f"Rate controller configured: max_concurrent={max_concurrent}, "
            f"min_interval={min_interval_ms}ms"
        )

    async def _sync_tasks_with_config(self, config: Any) -> None:
        """Sync all tasks with current configuration."""
        cron_manager = self._context.cron_manager
        if not cron_manager:
            logger.warning("CronJobManager not available")
            return

        task_definitions = self._get_task_definitions()

        for task_key, task_def in task_definitions.items():
            task_name = task_def["name"]
            enabled_key = f"scheduled_tasks.{task_key}.enabled"
            cron_key = f"scheduled_tasks.{task_key}.cron"

            enabled = config.get(enabled_key, True)
            cron_expr = config.get(cron_key, task_def["default_cron"])

            existing_job_id = self._task_name_to_job_id.get(task_name)

            logger.debug(
                f"Sync task '{task_key}': enabled={enabled} (key={enabled_key}), "
                f"existing_job_id={existing_job_id}"
            )

            if not enabled:
                if existing_job_id:
                    logger.info(
                        f"Disabling task '{task_name}' (job_id={existing_job_id})"
                    )
                    await self._disable_task(existing_job_id, task_name)
                else:
                    logger.debug(f"Task '{task_name}' already disabled, skipping")
                continue

            handler = self._get_handler(task_key)
            if handler is None:
                logger.warning(f"No handler for task '{task_key}', skipping")
                continue

            if existing_job_id:
                await self._update_task(
                    existing_job_id, task_name, cron_expr, cron_manager
                )
            else:
                await self._create_task(
                    task_key, task_name, cron_expr, task_def, handler, cron_manager
                )

    async def _disable_task(self, job_id: str, task_name: str) -> None:
        """Disable a task by deleting it."""
        cron_manager = self._context.cron_manager
        if not cron_manager:
            logger.warning(f"CronJobManager not available, cannot disable {task_name}")
            return

        try:
            logger.info(f"Deleting scheduled task: {task_name} (job_id={job_id})")
            await cron_manager.delete_job(job_id)
            if task_name in self._task_name_to_job_id:
                del self._task_name_to_job_id[task_name]
            logger.info(f"Disabled scheduled task: {task_name}")
        except Exception as e:
            logger.error(f"Failed to disable task '{task_name}': {e}", exc_info=True)

    async def _update_task(
        self, job_id: str, task_name: str, cron_expr: str, cron_manager: Any
    ) -> None:
        """Update an existing task's cron expression."""
        try:
            await cron_manager.update_job(
                job_id, cron_expression=cron_expr, enabled=True
            )
            logger.info(f"Updated scheduled task: {task_name} (cron: {cron_expr})")
        except Exception as e:
            logger.error(f"Failed to update task '{task_name}': {e}")

    async def _create_task(
        self,
        task_key: str,
        task_name: str,
        cron_expr: str,
        task_def: dict[str, Any],
        handler: Callable[[], Any],
        cron_manager: Any,
    ) -> None:
        """Create a new scheduled task."""
        try:
            job = await cron_manager.add_basic_job(
                name=task_name,
                cron_expression=cron_expr,
                handler=handler,
                description=task_def["description"],
                persistent=True,
                enabled=True,
            )
            self._task_name_to_job_id[task_name] = job.job_id
            self._task_key_to_name[task_key] = task_name
            logger.info(f"Registered scheduled task: {task_name} (cron: {cron_expr})")
        except Exception as e:
            logger.error(f"Failed to register task '{task_key}': {e}")

    async def _register_all_tasks(self, config: Any) -> None:
        """Register all enabled scheduled tasks.

        Checks for existing tasks by name before creating new ones.
        """
        cron_manager = self._context.cron_manager
        if not cron_manager:
            logger.warning("CronJobManager not available, scheduled tasks disabled")
            return

        existing_jobs = await self._get_existing_jobs_by_name(cron_manager)

        task_definitions = self._get_task_definitions()

        for task_key, task_def in task_definitions.items():
            task_name = task_def["name"]
            enabled = config.get(f"scheduled_tasks.{task_key}.enabled", True)

            if not enabled:
                logger.info(f"Scheduled task '{task_key}' is disabled by config")
                if task_name in existing_jobs:
                    await cron_manager.delete_job(existing_jobs[task_name])
                continue

            cron_expr = config.get(
                f"scheduled_tasks.{task_key}.cron",
                task_def["default_cron"],
            )

            handler = self._get_handler(task_key)
            if handler is None:
                logger.warning(f"No handler for task '{task_key}', skipping")
                continue

            if task_name in existing_jobs:
                job_id = existing_jobs[task_name]
                await self._update_or_reuse_task(
                    task_key,
                    task_name,
                    job_id,
                    cron_expr,
                    task_def,
                    handler,
                    cron_manager,
                )
            else:
                await self._create_task(
                    task_key, task_name, cron_expr, task_def, handler, cron_manager
                )

    async def _get_existing_jobs_by_name(self, cron_manager: Any) -> dict[str, str]:
        """Get existing jobs indexed by name.

        Returns:
            Dict mapping task name to job_id
        """
        existing = {}
        try:
            jobs = await cron_manager.list_jobs(job_type="basic")
            for job in jobs:
                if job.name and job.name.startswith("iris_memory_"):
                    existing[job.name] = job.job_id
        except Exception as e:
            logger.warning(f"Failed to list existing jobs: {e}")
        return existing

    async def _update_or_reuse_task(
        self,
        task_key: str,
        task_name: str,
        job_id: str,
        cron_expr: str,
        task_def: dict[str, Any],
        handler: Callable[[], Any],
        cron_manager: Any,
    ) -> None:
        """Update an existing task or reuse if cron matches."""
        try:
            cron_manager._basic_handlers[job_id] = handler
            await cron_manager.update_job(
                job_id, cron_expression=cron_expr, enabled=True
            )
            self._task_name_to_job_id[task_name] = job_id
            self._task_key_to_name[task_key] = task_name
            logger.info(f"Reused existing task: {task_name} (cron: {cron_expr})")
        except Exception as e:
            logger.warning(
                f"Failed to update existing task '{task_name}', recreating: {e}"
            )
            await cron_manager.delete_job(job_id)
            await self._create_task(
                task_key, task_name, cron_expr, task_def, handler, cron_manager
            )

    def _get_task_definitions(self) -> dict[str, dict[str, Any]]:
        """Get all task definitions."""
        return {
            "memory_promotion": ScheduledTaskDefaults.MEMORY_PROMOTION,
            "semantic_extraction": ScheduledTaskDefaults.SEMANTIC_EXTRACTION,
            "persona_batch_flush": ScheduledTaskDefaults.PERSONA_BATCH_FLUSH,
            "kg_auto_flush": ScheduledTaskDefaults.KG_AUTO_FLUSH,
        }

    def _get_handler(self, task_key: str) -> Callable[[], Any] | None:
        """Get the handler function for a task.

        Returns a sync function that creates an async task.
        This is required because APScheduler expects sync handlers.
        """
        handlers = {
            "memory_promotion": self._handle_memory_promotion,
            "semantic_extraction": self._handle_semantic_extraction,
            "persona_batch_flush": self._handle_persona_batch_flush,
            "kg_auto_flush": self._handle_kg_auto_flush,
        }
        return handlers.get(task_key)

    def _run_async(self, coro: Any) -> None:
        """Run an async coroutine in a new task.

        APScheduler calls handlers synchronously, so we need to
        schedule async work as a new task.
        """
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)

    def _handle_memory_promotion(self) -> None:
        """Handle memory promotion task (sync wrapper for async handler)."""
        self._run_async(self._async_memory_promotion())

    async def _async_memory_promotion(self) -> None:
        """Execute memory promotion: WORKING→EPISODIC, EPISODIC→SEMANTIC."""
        logger.debug("Executing scheduled task: memory_promotion")
        try:
            lifecycle = self._service.storage.lifecycle_manager
            if lifecycle and lifecycle.chroma_manager:
                await lifecycle._promote_memories()
                logger.debug("Memory promotion completed")
        except Exception as e:
            logger.error(f"Memory promotion task failed: {e}")

    def _handle_semantic_extraction(self) -> None:
        """Handle semantic extraction task (sync wrapper for async handler)."""
        self._run_async(self._async_semantic_extraction())

    async def _async_semantic_extraction(self) -> None:
        """Execute semantic extraction (Channel B)."""
        logger.debug("Executing scheduled task: semantic_extraction")
        try:
            lifecycle = self._service.storage.lifecycle_manager
            if lifecycle and lifecycle._semantic_extractor:
                await lifecycle._run_semantic_extraction()
                logger.debug("Semantic extraction completed")
        except Exception as e:
            logger.error(f"Semantic extraction task failed: {e}")

    def _handle_persona_batch_flush(self) -> None:
        """Handle persona batch flush task (sync wrapper for async handler)."""
        self._run_async(self._async_persona_batch_flush())

    async def _async_persona_batch_flush(self) -> None:
        """Execute persona batch processing flush."""
        logger.debug("Executing scheduled task: persona_batch_flush")
        try:
            persona_processor = self._service.analysis.persona_batch_processor
            if persona_processor and persona_processor.is_running:
                await persona_processor.flush_all_queues()
                logger.debug("Persona batch flush completed")
        except Exception as e:
            logger.error(f"Persona batch flush task failed: {e}")

    def _handle_kg_auto_flush(self) -> None:
        """Handle KG auto flush task (sync wrapper for async handler)."""
        self._run_async(self._async_kg_auto_flush())

    async def _async_kg_auto_flush(self) -> None:
        """Execute knowledge graph pending LLM flush."""
        logger.debug("Executing scheduled task: kg_auto_flush")
        try:
            kg_module = self._service.kg
            if kg_module and kg_module.enabled:
                await kg_module.flush_pending_llm()
                logger.debug("KG auto flush completed")
        except Exception as e:
            logger.error(f"KG auto flush task failed: {e}")

    async def shutdown(self) -> None:
        """Unregister all scheduled tasks and config event subscriptions."""
        for unsub in self._unsubscribe_fns:
            try:
                unsub()
            except Exception as e:
                logger.warning(f"Failed to unsubscribe config event: {e}")
        self._unsubscribe_fns.clear()

        cron_manager = self._context.cron_manager
        if not cron_manager:
            return

        for task_name, job_id in self._task_name_to_job_id.items():
            try:
                await cron_manager.delete_job(job_id)
                logger.debug(f"Unregistered scheduled task: {task_name}")
            except Exception as e:
                logger.warning(f"Failed to unregister task '{task_name}': {e}")

        self._task_name_to_job_id.clear()
        self._task_key_to_name.clear()
        logger.info("ScheduledTaskManager shutdown complete")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about scheduled tasks."""
        return {
            "initialized": self._initialized,
            "registered_tasks": list(self._task_name_to_job_id.keys()),
            "task_count": len(self._task_name_to_job_id),
        }
