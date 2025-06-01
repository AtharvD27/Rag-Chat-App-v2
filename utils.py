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