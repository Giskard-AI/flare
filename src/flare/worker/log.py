import logging
import logging.config
from datetime import datetime
from pathlib import Path

LOG_FILE = "run.log"


def setup_log(log_path: Path):
    # Initialisation of the logs, by making httpx a bit more silent
    logging.config.dictConfig(
        {"version": 1, "loggers": {"httpx": {"level": "WARNING"}}}
    )
    # First, we clean up the log file if needed (ie we rename with timestamp)
    log_file_path = log_path / LOG_FILE
    if log_file_path.is_file():
        log_file_path.rename(log_path / ("run" + "-" + str(datetime.now()) + ".log"))
    # Then we remove the file (but should not be here)
    log_file_path.unlink(missing_ok=True)
    # Now, we finish log config
    logging.basicConfig(
        level="INFO",
        filename=str(log_file_path),
        format="[blue]%(asctime)s[/blue] [bold green]%(levelname)-8s[/green bold] [bold cyan]%(name)s[/cyan bold] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
