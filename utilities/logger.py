# utilities/logger.py
import logging
import sys

# ANSI color codes
RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[36m",    # Cyan
    "INFO": "\033[32m",     # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",    # Red
    "CRITICAL": "\033[41m", # Red background
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, "")
        record.levelname_color = f"{color}{record.levelname}{RESET}"
        return super().format(record)

def get_logger(name: str, level=logging.INFO, logfile: str = None):
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Console handler with colored output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter(
            fmt="%(asctime)s | %(levelname_color)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%H:%M:%S"))
        logger.addHandler(stream_handler)

        if logfile:
            # Clear the log file
            with open(logfile, 'w'):
                pass

            # File handler with plain output
            file_handler = logging.FileHandler(logfile)
            file_handler.setFormatter(logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%H:%M:%S"
            ))
            logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger

def header(logger: logging.Logger, title: str, width: int = 80, char: str = "=", level: int = logging.INFO, stacklevel: int = 2):
    border = char * width
    centered_title = title.center(width)
    logger.log(level, border, stacklevel=stacklevel)
    logger.log(level, centered_title, stacklevel=stacklevel)
    logger.log(level, border, stacklevel=stacklevel)

def subheader(logger: logging.Logger, title: str, width: int = 80, left: str = "╭─", right: str = "─╮", level: int = logging.INFO, stacklevel: int = 2):
    padding = width - len(left) - len(title) - len(right)
    line = f"{left} {title} {right}{'-' * max(padding, 0)}"
    logger.log(level, line, stacklevel=stacklevel)

def subsubheader(logger: logging.Logger, title: str, width: int = 80, left: str = "---", right: str = "--------", level: int = logging.INFO, stacklevel: int = 2):
    padding = width - len(left) - len(title) - len(right)
    line = f"{left} {title} {right}{'-' * max(padding, 0)}"
    logger.log(level, line, stacklevel=stacklevel)
