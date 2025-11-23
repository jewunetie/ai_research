"""Logging configuration utilities for experiments."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    experiment_name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_logging: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for an experiment.

    Args:
        experiment_name: Name of the experiment (used for logger name and log file)
        log_dir: Directory for log files (default: logs/)
        level: Logging level (default: INFO)
        console: If True, log to console
        file_logging: If True, log to file
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )

    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_logging:
        # Create log directory if it doesn't exist
        if log_dir is None:
            log_dir = Path('logs')

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{experiment_name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    # Don't propagate to root logger
    logger.propagate = False

    return logger


def setup_experiment_logging(
    experiment_name: str,
    output_dir: Path,
    debug: bool = False,
) -> logging.Logger:
    """
    Set up logging for a full experiment run.

    This is a convenience function that sets up both console and file logging
    with appropriate defaults for experiments.

    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for results (logs will go in output_dir/logs)
        debug: If True, set level to DEBUG

    Returns:
        Configured logger instance
    """
    log_dir = output_dir / 'logs'
    level = logging.DEBUG if debug else logging.INFO

    logger = setup_logging(
        experiment_name=experiment_name,
        log_dir=log_dir,
        level=level,
        console=True,
        file_logging=True,
    )

    logger.info("=" * 80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerContext:
    """
    Context manager for temporary logging configuration.

    Useful for temporarily changing log levels or adding handlers.
    """

    def __init__(
        self,
        logger: logging.Logger,
        level: Optional[int] = None,
        handlers: Optional[list] = None,
    ):
        """
        Initialize context manager.

        Args:
            logger: Logger to modify
            level: Temporary log level (optional)
            handlers: Temporary handlers to add (optional)
        """
        self.logger = logger
        self.new_level = level
        self.new_handlers = handlers or []

        # Store original state
        self.original_level = logger.level
        self.original_handlers = logger.handlers.copy()

    def __enter__(self):
        """Enter context - apply temporary configuration."""
        if self.new_level is not None:
            self.logger.setLevel(self.new_level)

        for handler in self.new_handlers:
            self.logger.addHandler(handler)

        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original configuration."""
        # Restore level
        self.logger.setLevel(self.original_level)

        # Remove temporary handlers
        for handler in self.new_handlers:
            self.logger.removeHandler(handler)

        return False


def log_experiment_config(logger: logging.Logger, config: dict):
    """
    Log experiment configuration in a formatted way.

    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("Experiment Configuration:")
    logger.info("-" * 40)

    def log_dict(d: dict, indent: int = 0):
        """Recursively log nested dictionary."""
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info(f"{prefix}{key}: {value}")

    log_dict(config)
    logger.info("-" * 40)


def log_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    message: str = "Progress",
    log_every: int = 10,
):
    """
    Log progress at regular intervals.

    Args:
        logger: Logger instance
        current: Current iteration (1-indexed)
        total: Total iterations
        message: Progress message
        log_every: Log every N iterations
    """
    if current % log_every == 0 or current == total:
        percentage = (current / total) * 100
        logger.info(f"{message}: {current}/{total} ({percentage:.1f}%)")


def setup_file_handler(
    logger: logging.Logger,
    log_file: Path,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.FileHandler:
    """
    Add a file handler to an existing logger.

    Args:
        logger: Logger to add handler to
        log_file: Path to log file
        level: Logging level for this handler
        format_string: Custom format string (optional)

    Returns:
        Created file handler
    """
    # Create parent directory if needed
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Default format
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )

    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return file_handler
