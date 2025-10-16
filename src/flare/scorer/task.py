import asyncio
import logging
from pathlib import Path

from flare.schema import SampleWithOutputs
from flare.scorer.base import Scorer
from flare.stats import add_scoring_stats


async def task_scorer(
    run_name: str,
    scorer_name: str,
    queue: asyncio.Queue[SampleWithOutputs],
    scorer_instance: Scorer,
):
    logger = logging.getLogger(f"scorer:{scorer_name}")

    while sample_with_outputs := await queue.get():
        # Get a "work item" out of the queue.
        try:
            id_sample = str(sample_with_outputs.sample.id)
            model_name = sample_with_outputs.model_outputs.model

            logger.info("Starting on new sample %s", id_sample)
            target_path = (
                Path(run_name)
                / "result"
                / model_name.replace("/", "_").replace(" ", "_")
                / sample_with_outputs.sample.module
                / sample_with_outputs.sample.task
                / (id_sample + ".json")
            )

            logger.info("Checking path : %s", str(target_path))

            if target_path.is_file():
                logger.info("Skipping path : %s", str(target_path))
                add_scoring_stats(scorer_name, True)
                continue

            logger.info("Scoring sample %s", id_sample)
            sample_with_score = await scorer_instance.score(
                sample_with_outputs, logger=logger
            )
            logger.info("Scorer done on %s", sample_with_outputs.sample.id)
            add_scoring_stats(scorer_name, True)

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(sample_with_score.model_dump_json())
        except Exception:
            add_scoring_stats(scorer_name, False)
            logger.exception("Error while scoring")
        finally:
            # Notify the queue that the "work item" has been processed.
            queue.task_done()
