import asyncio
import logging
from copy import deepcopy

from flare.generate.task import task_generate
from flare.schema import ModelConfig, Sample, SampleWithOutputs, ScorerConfig
from flare.scorer.get_scorer import get_scorer
from flare.scorer.task import task_scorer

logger = logging.getLogger(__name__)

_queues_scorer: dict[str, asyncio.Queue[SampleWithOutputs]] = {}
_workers_scorer: dict[str, list[asyncio.Task]] = {}

_queues_generator: dict[str, asyncio.Queue[Sample]] = {}
_workers_generator: dict[str, list[asyncio.Task]] = {}


def register_scorer(run_name: str, scorer_name: str, conf: ScorerConfig, generators: list[ModelConfig]):
    # Create the scored tasks
    # We create a shared queue with all the workers for a same scorer
    queue = asyncio.Queue()
    parallelism = conf.parallelism
    scorer_instance = get_scorer(scorer_name, conf.models, generators=generators)
    _queues_scorer[scorer_name] = queue
    _workers_scorer[scorer_name] = []
    logger.info("Starting scorer %s with concurrency %s", scorer_name, parallelism)
    # In case parallelism is not defined, let's put 8
    for _ in range(parallelism):
        task = asyncio.create_task(
            task_scorer(run_name, scorer_name, queue, scorer_instance)
        )
        _workers_scorer[scorer_name].append(task)


def register_generator(run_name: str, config: ModelConfig):
    # We create a shared queue with all the workers for a same model
    queue = asyncio.Queue()
    model = config.litellm_model
    nb_try = config.nb_try
    parallelism = config.parallelism
    _queues_generator[model] = queue
    logger.info(
        "Starting generator on model %s with concurrency %s", model, parallelism
    )
    _workers_generator[model] = [
        asyncio.create_task(task_generate(run_name, config, queue))
        for _ in range(parallelism)
    ]


def submit_sample(sample: Sample):
    for queue in _queues_generator.values():
        queue.put_nowait(deepcopy(sample))


def submit_to_scorer(sample_with_outputs: SampleWithOutputs):
    _queues_scorer[sample_with_outputs.sample.evaluation.scorer].put_nowait(
        deepcopy(sample_with_outputs)
    )


def any_running_tasks():
    # Ugly hack => it's more precise than .empty(), and don't wait as with .join
    return any(
        q._unfinished_tasks > 0
        for q in list(_queues_generator.values()) + list(_queues_scorer.values())
    )


async def clean_stop():
    # Cancel our worker tasks.1
    tasks = [
        task
        for workers in [_workers_scorer, _workers_generator]
        for task_list in workers.values()
        for task in task_list
    ]
    # Cancel our worker tasks.
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
