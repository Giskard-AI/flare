import json
import logging
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime

from flare.schema import Config


def logger():
    return logging.getLogger("stats")


STATS = {
    "nb_samples": 0,
    "total": 0,
    "current": 0,
    "models": defaultdict(lambda: {"success": 0, "error": 0}),
    "scorers": defaultdict(lambda: {"success": 0, "error": 0, "skipped": 0}),
}


def init_stats(
    samples_count: Counter,
    nb_models: int,
    start_time: datetime,
    config: Config,
    run_name: str,
):
    STATS["samples_per_model"] = samples_count
    STATS["nb_samples"] = samples_count.total()
    STATS["total"] = 2 * STATS["nb_samples"] * nb_models
    STATS["current"] = 0
    STATS["start"] = start_time
    STATS["run_name"] = run_name

    for elt in config.models:
        # Side effect of default dict
        STATS["models"][elt.litellm_model]
    for name in config.scorers.keys():
        # Side effect of default dict
        STATS["scorers"][name]


def add_generation_stats(model: str, success: bool = True):
    STATS["models"][model]["success" if success else "error"] += 1
    STATS["current"] += 1


def add_scoring_stats(name: str, success: bool = True):
    STATS["scorers"][name]["success" if success else "error"] += 1
    STATS["current"] += 1


def print_stats():
    stats = deepcopy(STATS)
    start_time = stats.pop("start")
    logger().info("#" * 50)
    logger().info("Elapsed time since start  %s", str(datetime.now() - start_time))
    logger().info(json.dumps(stats))
    logger().info("#" * 50)
