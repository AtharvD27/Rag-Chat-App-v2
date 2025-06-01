import yaml
import hashlib
import logging
import logging.handlers
import os
import sys
import functools
import time
import psutil
import mimetypes
import magic
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, List
from datetime import datetime
import json
from dataclasses import dataclass
import traceback


# ===================== LOGGING CONFIGURATION =====================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[34m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    COLORS = {
        logging.DEBUG: grey,
        logging.INFO: blue,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelno, self.grey)
        record.levelname = f"{log_color}{record.levelname}{self.reset}"
        return super().format(record)


def setup_logging(
    name: str = "rag_chat",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_level: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up dual logging (console + file) with proper formatting
    
    Args:
        name: Logger name
        log_level: File logging level
        log_file: Path to log file
        console_level: Console logging level (defaults to log_level)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler with rotation
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_level = console_level or log_level
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


# ===================== ERROR HANDLING FRAMEWORK =====================

class RAGException(Exception):
    """Base exception for RAG application"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class ConfigurationError(RAGException):
    """Configuration-related errors"""
    pass


class DocumentProcessingError(RAGException):
    """Document loading/processing errors"""
    pass


class VectorStoreError(RAGException):
    """Vectorstore operation errors"""
    pass


class ModelError(RAGException):
    """LLM-related errors"""
    pass


class SecurityError(RAGException):
    """Security validation errors"""
    pass


def retry_with_backoff(
    retries: int = 3,
    backoff_in_seconds: float = 1.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        retries: Number of retry attempts
        backoff_in_seconds: Initial backoff time
        exceptions: Tuple of exceptions to catch
        logger: Logger instance for retry logging
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            delay = backoff_in_seconds
            
            while retry_count < retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    if retry_count == retries:
                        if logger:
                            logger.error(f"Function {func.__name__} failed after {retries} retries: {str(e)}")
                        raise
                    
                    if logger:
                        logger.warning(
                            f"Function {func.__name__} failed (attempt {retry_count}/{retries}): {str(e)}. "
                            f"Retrying in {delay} seconds..."
                        )
                    
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            
            return wrapper
    return decorator


def handle_errors(logger: logging.Logger, raise_on_error: bool = True):
    """
    Decorator for consistent error handling and logging
    
    Args:
        logger: Logger instance
        raise_on_error: Whether to re-raise exceptions after logging
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RAGException as e:
                logger.error(
                    f"{func.__name__} failed with {e.__class__.__name__}: {str(e)}",
                    extra={
                        "error_code": e.error_code,
                        "details": e.details,
                        "function": func.__name__,
                        "traceback": traceback.format_exc()
                    }
                )
                if raise_on_error:
                    raise
                return None
            except Exception as e:
                logger.error(
                    f"{func.__name__} failed with unexpected error: {str(e)}",
                    extra={
                        "function": func.__name__,
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc()
                    }
                )
                if raise_on_error:
                    raise RAGException(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        error_code="UNEXPECTED_ERROR",
                        details={"original_error": str(e)}
                    )
                return None
        return wrapper
    return decorator


# ===================== UTILITY FUNCTIONS =====================

def compute_sha1(text: str) -> str:
    """Compute SHA1 hash of text"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get detailed file information"""
    path = Path(file_path)
    stat = path.stat()
    
    return {
        "name": path.name,
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "extension": path.suffix,
        "mime_type": mimetypes.guess_type(str(path))[0]
    }


def batch_iterator(items: List[Any], batch_size: int):
    """Yield batches of items"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# ===================== SECURITY VALIDATION =====================

class SecurityValidator:
    """Handles all security-related validations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.allowed_extensions = set(config.get("allowed_file_types", [".pdf", ".txt", ".json", ".html"]))
        self.max_file_size = config.get("max_file_size_mb", 100) * 1024 * 1024  # Convert to bytes
        self.allowed_dirs = [Path(d).resolve() for d in config.get("allowed_directories", ["./data"])]
        self.enable_content_scan = config.get("enable_content_scanning", True)
        
        # Initialize magic for file type detection
        self.mime_detector = magic.Magic(mime=True) if self.enable_content_scan else None
        
        # File type to extension mapping
        self.mime_to_ext = {
            "application/pdf": ".pdf",
            "text/plain": ".txt",
            "application/json": ".json",
            "text/html": ".html",
        }
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """
        Validate file path for security issues
        
        Args:
            file_path: Path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path validation fails
        """
        path = Path(file_path).resolve()
        
        # Check if path is within allowed directories
        if not any(self._is_subdirectory(path, allowed_dir) for allowed_dir in self.allowed_dirs):
            raise SecurityError(
                f"Access denied: {path} is outside allowed directories",
                error_code="PATH_TRAVERSAL",
                details={"path": str(path), "allowed_dirs": [str(d) for d in self.allowed_dirs]}
            )
        
        # Check if file exists
        if not path.exists():
            raise SecurityError(
                f"File not found: {path}",
                error_code="FILE_NOT_FOUND",
                details={"path": str(path)}
            )
        
        # Check file size
        if path.stat().st_size > self.max_file_size:
            raise SecurityError(
                f"File too large: {path} ({path.stat().st_size / 1024 / 1024:.2f}MB)",
                error_code="FILE_TOO_LARGE",
                details={
                    "path": str(path),
                    "size_mb": path.stat().st_size / 1024 / 1024,
                    "max_size_mb": self.max_file_size / 1024 / 1024
                }
            )
        
        # Check file extension
        if path.suffix.lower() not in self.allowed_extensions:
            raise SecurityError(
                f"File type not allowed: {path.suffix}",
                error_code="INVALID_FILE_TYPE",
                details={
                    "path": str(path),
                    "extension": path.suffix,
                    "allowed_extensions": list(self.allowed_extensions)
                }
            )
        
        # Verify file content matches extension
        if self.enable_content_scan and self.mime_detector:
            mime_type = self.mime_detector.from_file(str(path))
            expected_ext = self.mime_to_ext.get(mime_type)
            if expected_ext and expected_ext != path.suffix.lower():
                raise SecurityError(
                    f"File content doesn't match extension: {path}",
                    error_code="CONTENT_MISMATCH",
                    details={
                        "path": str(path),
                        "detected_type": mime_type,
                        "expected_extension": expected_ext,
                        "actual_extension": path.suffix
                    }
                )
        
        return path
    
    @staticmethod
    def _is_subdirectory(path: Path, directory: Path) -> bool:
        """Check if path is subdirectory of directory"""
        try:
            path.relative_to(directory)
            return True
        except ValueError:
            return False
        

# ===================== CONFIGURATION MANAGEMENT =====================

@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    required_fields: List[str]
    optional_fields: Dict[str, Any]  # field_name: default_value
    
    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill configuration with defaults"""
        validated = config.copy()
        
        # Check required fields
        for field in self.required_fields:
            if field not in validated:
                raise ConfigurationError(
                    f"Missing required configuration field: {field}",
                    error_code="MISSING_FIELD",
                    details={"field": field, "required_fields": self.required_fields}
                )
        
        # Fill optional fields with defaults
        for field, default in self.optional_fields.items():
            if field not in validated:
                validated[field] = default
        
        return validated


# Default configuration schema
DEFAULT_CONFIG_SCHEMA = ConfigSchema(
    required_fields=["data_path", "vector_db_path"],
    optional_fields={
        "snapshot_path": "./snapshots",
        "prompt_path": "./prompts.yaml",
        "chunk": {"size": 800, "overlap": 80},
        "embedding": {"model_name": "all-MiniLM-L6-v2"},
        "llm": {
            "temperature": 0.3,
            "max_tokens": 400,
            "n_ctx": 1536,
            "n_threads": 6
        },
        "security": {
            "allowed_file_types": [".pdf", ".txt", ".json", ".html"],
            "max_file_size_mb": 100,
            "allowed_directories": ["./data"],
            "enable_content_scanning": True
        },
        "logging": {
            "level": "INFO",
            "file_path": "./logs/rag_chat.log",
            "console_level": "INFO"
        },
        "performance": {
            "batch_size": 100,
            "max_concurrent_operations": 4,
            "enable_async": True
        }
    }
)


def load_config(
    config_path: str = "config.yaml",
    schema: Optional[ConfigSchema] = None,
    allow_env_override: bool = True
) -> Dict[str, Any]:
    """
    Load and validate configuration with environment variable override support
    
    Args:
        config_path: Path to configuration file
        schema: Configuration schema for validation
        allow_env_override: Whether to allow environment variable overrides
        
    Returns:
        Validated configuration dictionary
    """
    logger = logging.getLogger("config_loader")
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            error_code="CONFIG_NOT_FOUND",
            details={"path": config_path}
        )
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML in configuration file: {str(e)}",
            error_code="INVALID_YAML",
            details={"path": config_path, "error": str(e)}
        )
    
    # Apply schema validation
    if schema:
        config = schema.validate(config)
    else:
        config = DEFAULT_CONFIG_SCHEMA.validate(config)
    
    # Apply environment variable overrides
    if allow_env_override:
        config = apply_env_overrides(config)
    
    # Initialize security validator
    config["_security_validator"] = SecurityValidator(config.get("security", {}))
    
    logger.info(f"Configuration loaded successfully from {config_path}")
    return config


def apply_env_overrides(config: Dict[str, Any], prefix: str = "RAG_") -> Dict[str, Any]:
    """
    Override configuration values with environment variables
    
    Environment variables should be prefixed and use double underscores for nesting
    Example: RAG_LLM__TEMPERATURE=0.5
    """
    def set_nested(d: dict, keys: list, value: Any):
        """Set nested dictionary value"""
        for key in keys[:-1]:
            d = d.setdefault(key.lower(), {})
        d[keys[-1].lower()] = value
    
    for env_key, env_value in os.environ.items():
        if env_key.startswith(prefix):
            # Remove prefix and split by double underscore
            key_parts = env_key[len(prefix):].split("__")
            
            # Try to parse value as JSON, fallback to string
            try:
                value = json.loads(env_value)
            except json.JSONDecodeError:
                value = env_value
            
            set_nested(config, key_parts, value)
    
    return config


# ===================== PERFORMANCE UTILITIES =====================

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and record metric"""
        if operation not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
        
        return duration
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics"""
        result = {}
        for operation, times in self.metrics.items():
            if times:
                result[operation] = {
                    "count": len(times),
                    "total": sum(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times)
                }
        return result
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3)
            },
            "disk": {
                "percent": psutil.disk_usage('/').percent,
                "free_gb": psutil.disk_usage('/').free / (1024**3)
            }
        }


def time_operation(operation_name: Optional[str] = None, logger: Optional[logging.Logger] = None):
    """
    Decorator to time function execution
    
    Args:
        operation_name: Name for the operation (defaults to function name)
        logger: Logger for timing output
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if logger:
                    logger.info(f"{name} completed in {duration:.2f} seconds")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                if logger:
                    logger.error(f"{name} failed after {duration:.2f} seconds: {str(e)}")
                raise
        
        return wrapper
    return decorator
