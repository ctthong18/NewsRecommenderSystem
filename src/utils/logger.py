import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, ensure_ascii=False)


class StructuredLogger:
    """
    Enhanced logger with support for:
    - JSON and standard text formatting
    - Log rotation
    - Per-module log levels
    - Simultaneous file and console logging
    """
    
    def __init__(
        self,
        name: str = "news_recommendation",
        log_dir: Optional[str] = None,
        log_level: str = "INFO",
        use_json: bool = False,
        enable_rotation: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console_output: bool = True
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files. If None, only console logging
            log_level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            use_json: Use JSON format for logs
            enable_rotation: Enable log file rotation
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup log files to keep
            console_output: Enable console output
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.use_json = use_json
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()  # Clear any existing handlers
        
        # Create formatters
        if use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        
        # Add console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler with rotation
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / f"{name}.log"
            
            if enable_rotation:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
            else:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
            
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def set_module_level(self, module_name: str, level: str):
        """
        Set log level for a specific module.
        
        Args:
            module_name: Name of the module
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        module_logger = logging.getLogger(f"{self.name}.{module_name}")
        module_logger.setLevel(getattr(logging, level.upper()))
    
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """
        Get logger instance, optionally for a specific module.
        
        Args:
            module_name: Optional module name for module-specific logger
            
        Returns:
            Logger instance
        """
        if module_name:
            return logging.getLogger(f"{self.name}.{module_name}")
        return self.logger
    
    def log_with_context(
        self,
        level: str,
        message: str,
        extra_fields: Optional[Dict[str, Any]] = None
    ):
        """
        Log message with additional context fields.
        
        Args:
            level: Log level
            message: Log message
            extra_fields: Additional fields to include in log
        """
        log_func = getattr(self.logger, level.lower())
        
        if extra_fields and self.use_json:
            # Create a log record with extra fields
            record = self.logger.makeRecord(
                self.logger.name,
                getattr(logging, level.upper()),
                "(unknown file)",
                0,
                message,
                (),
                None
            )
            record.extra_fields = extra_fields
            self.logger.handle(record)
        else:
            log_func(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log_with_context("DEBUG", message, kwargs if kwargs else None)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log_with_context("INFO", message, kwargs if kwargs else None)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log_with_context("WARNING", message, kwargs if kwargs else None)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log_with_context("ERROR", message, kwargs if kwargs else None)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log_with_context("CRITICAL", message, kwargs if kwargs else None)


# Global logger instance
_global_logger: Optional[StructuredLogger] = None


def setup_logger(
    name: str = "news_recommendation",
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    use_json: bool = False,
    enable_rotation: bool = True,
    console_output: bool = True
) -> StructuredLogger:
    """
    Setup and configure the global logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Default log level
        use_json: Use JSON format for logs
        enable_rotation: Enable log file rotation
        console_output: Enable console output
        
    Returns:
        Configured StructuredLogger instance
    """
    global _global_logger
    _global_logger = StructuredLogger(
        name=name,
        log_dir=log_dir,
        log_level=log_level,
        use_json=use_json,
        enable_rotation=enable_rotation,
        console_output=console_output
    )
    return _global_logger


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance. If global logger not setup, creates a basic one.
    
    Args:
        module_name: Optional module name for module-specific logger
        
    Returns:
        Logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        # Setup basic logger if not configured
        setup_logger()
    
    return _global_logger.get_logger(module_name)


def setup_from_config(config: Dict[str, Any]) -> StructuredLogger:
    """
    Setup logger from configuration dictionary.
    
    Args:
        config: Configuration dictionary with logging settings
        
    Returns:
        Configured StructuredLogger instance
        
    Example config:
        {
            "logging": {
                "level": "INFO",
                "log_dir": "output/logs",
                "use_json": false,
                "enable_rotation": true,
                "console_output": true
            }
        }
    """
    logging_config = config.get("logging", {})
    paths_config = config.get("paths", {})
    
    return setup_logger(
        log_dir=logging_config.get("log_dir") or paths_config.get("log_dir"),
        log_level=logging_config.get("level", "INFO"),
        use_json=logging_config.get("use_json", False),
        enable_rotation=logging_config.get("enable_rotation", True),
        console_output=logging_config.get("console_output", True)
    )


# Backward compatibility: setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
