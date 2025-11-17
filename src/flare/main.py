import argparse
import asyncio
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from shutil import rmtree

import dotenv
import uvloop

from flare.dashboard import live_dashboard, print_dashboard, update_display
from flare.schema import Config, Sample
from flare.scorer.get_scorer import SCORERS
from flare.stats import init_stats
from flare.worker.log import LOG_FILE, setup_log
from flare.worker.registry import (
    any_running_tasks,
    clean_stop,
    register_generator,
    register_scorer,
    submit_sample,
)

logger = logging.getLogger(__name__)


async def main(
    run_name: str,
    sample_path: str,
    config_path: str,
    run_folder: str,
    max_samples_per_task: int,
):
    samples_path = Path(sample_path).glob("**/*.jsonl")
    start_time = datetime.now()

    run_path = Path(run_folder) / run_name

    logger.info(f"Starting new run {run_name} in {run_folder}")
    # Cleaning up error folder
    rmtree(run_path / "error", ignore_errors=True)

    # Read configuration
    config_all = Config.model_validate_json(
        Path(config_path).read_text(encoding="utf-8")
    )
    config = config_all.models
    config_scorer = config_all.scorers
    # Loading all the sample
    samples = [
        Sample.model_validate_json(s)
        for file in samples_path
        for idx, s in enumerate(file.read_text().split("\n"))
        if s and idx < max_samples_per_task
    ]

    samples = [
        sample for sample in samples if sample.evaluation.scorer in config_scorer
    ]

    samples_count = Counter(sample.evaluation.scorer for sample in samples)

    init_stats(samples_count, len(config), start_time, config_all, run_name)

    # Validation of sample per scorer
    failures = {}
    for sample in samples:
        scorer_name = sample.evaluation.scorer
        if not SCORERS[scorer_name].validate_sample(sample):
            if scorer_name not in failures:
                failures[scorer_name] = []
            failures[scorer_name].append(str(sample.id))

    if len(failures) > 0:
        error_detail = "\n".join(
            [
                f"- {scorer_name}: {",".join(ids)}"
                for scorer_name, ids in failures.items()
            ]
        )
        raise ValueError("Invalid sample\n" + error_detail)

    # Create the scorer tasks
    for scorer_name, conf in config_scorer.items():
        register_scorer(run_path, scorer_name, conf, config)

    # Create all the workers according to the config
    for elt in config:
        register_generator(run_path, elt)

    # Push the sample in queues for generation
    for sample in samples:
        submit_sample(sample)

    with live_dashboard():
        while (
            any_running_tasks()
        ):  # Ugly hack => it's more precise than .empty(), and don't wait as with .join
            update_display(run_path / LOG_FILE)
            await asyncio.sleep(0.1)

    # Stopping tasks & waiting
    await clean_stop()

    end_time = datetime.now()

    # Give stats of the run
    logger.info("Ending run %s", run_name)
    logger.info("Ran in  %s", str(end_time - start_time))
    update_display(run_path / LOG_FILE)
    print_dashboard()


def main_cli():
    parser = argparse.ArgumentParser(description="Launch an evaluation")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to JSON configuration, with models and scorers",
    )
    parser.add_argument(
        "--sample-path",
        required=True,
        help="Path to JSON samples folder, like 'samples'",
    )
    parser.add_argument(
        "--run-path",
        required=False,
        default="runs",
        help="Path to save the runs",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the run to launch (will put the result in a name folder)",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=1e9,
        help="Maximum number of samples per task",
    )
    dotenv.load_dotenv()

    args = parser.parse_args()

    run_path = Path(args.run_path) / args.name
    if not run_path.exists():
        run_path.mkdir(parents=True, exist_ok=True)

    setup_log(run_path)
    uvloop.run(
        main(
            run_name=args.name,
            sample_path=args.sample_path,
            config_path=args.config_path,
            run_folder=args.run_path,
            max_samples_per_task=args.max_samples_per_task,
        )
    )


if __name__ == "__main__":
    main_cli()
