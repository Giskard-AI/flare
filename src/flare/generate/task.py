import asyncio
import logging
from typing import Any

from flare.generate.generator import generate
from flare.schema import ModelConfig, Sample
from flare.stats import add_generation_stats


async def task_generate(
    run_name: str,
    model: ModelConfig,
    queue: asyncio.Queue[Sample],
):
    # Circular deps, so import is done inside the task (task pushing to another one)
    from flare.worker.registry import submit_to_scorer

    model_name = model.litellm_model
    logger = logging.getLogger(f"generate:{model_name}")
    while sample_data := await queue.get():
        # Get a "work item" out of the queue.
        try:
            logger.info("Starting on new sample %s", sample_data.id)

            result = await generate(run_name, model, sample_data, logger)
            logger.info("Generation done on sample %s", sample_data.id)

            if result is not None:
                add_generation_stats(model_name)
                submit_to_scorer(result)
            else:
                add_generation_stats(model_name, False)
        except Exception:
            # Handle retry
            add_generation_stats(model_name, False)
            logger.exception("Error while doing generation")
        finally:
            # Notify the queue that the "work item" has been processed.
            queue.task_done()
