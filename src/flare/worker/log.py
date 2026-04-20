import logging
import logging.config
from datetime import datetime
from pathlib import Path

LOG_FILE = "run.log"

_LITELLM_LOGGER_NAMES = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")


def _suppress_litellm_stream_handlers() -> None:
    """Remove LiteLLM's stderr StreamHandlers so logs only propagate to root (e.g. run.log).

    LiteLLM registers its own StreamHandler on these loggers (see litellm._logging); that
    duplicates file output and breaks Rich Live full-screen dashboards.
    """
    try:
        import litellm._logging  # noqa: F401 — registers StreamHandlers on import
    except ImportError:
        return

    for name in _LITELLM_LOGGER_NAMES:
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            if isinstance(h, logging.StreamHandler):
                lg.removeHandler(h)


def setup_log(log_path: Path, level: str = "INFO"):
    # Initialisation of the logs, by making httpx a bit more silent
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "loggers": {"httpx": {"level": "WARNING"}},
        }
    )
    # First, we clean up the log file if needed (ie we rename with timestamp)
    log_file_path = log_path / LOG_FILE
    if log_file_path.is_file():
        log_file_path.rename(log_path / ("run" + "-" + str(datetime.now()) + ".log"))
    # Then we remove the file (but should not be here)
    log_file_path.unlink(missing_ok=True)
    # Now, we finish log config
    logging.basicConfig(
        level=level,
        filename=str(log_file_path),
        format="[blue]%(asctime)s[/blue] [bold green]%(levelname)-8s[/green bold] [bold cyan]%(name)s[/cyan bold] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    _suppress_litellm_stream_handlers()
