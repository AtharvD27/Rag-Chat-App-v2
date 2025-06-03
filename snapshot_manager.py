import os
import json
import uuid
import gzip
import shutil
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import asyncio

from langchain.schema import AIMessage, HumanMessage, BaseMessage
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.language_models import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel

# Import from enhanced utils
from utils import (
    setup_logging, handle_errors, time_operation, ensure_directory,
    get_file_info, performance_monitor
)
from monitoring import get_monitoring_instance


@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SessionMetadata:
    """Enhanced session metadata"""
    session_id: str
    alias: str
    created: datetime
    modified: datetime
    file_path: str
    total_turns: int = 0
    total_tokens: int = 0
    summary_available: bool = False
    compressed: bool = False
    size_bytes: int = 0
    model_used: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class SnapshotManager:
    """Enhanced snapshot manager with conversation summarization and compression"""
    
    def __init__(
        self, 
        snapshot_dir: str = "./snapshots",
        llm: Optional[BaseLLM] = None,
        config: Dict[str, Any] = None
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.session_dir = self.snapshot_dir / "sessions"
        self.metadata_dir = self.snapshot_dir / "metadata"
        self.summary_dir = self.snapshot_dir / "summaries"
        
        # Create directories
        for dir_path in [self.session_dir, self.metadata_dir, self.summary_dir]:
            ensure_directory(dir_path)
        
        # File paths
        self.alias_file = self.metadata_dir / "aliases.json"
        self.session_file = self.metadata_dir / "sessions.json"
        
        # Configuration
        self.config = config or {}
        memory_config = self.config.get("memory", {})
        self.max_turns_before_summary = memory_config.get("max_turns_before_summary", 20)
        self.summary_token_limit = memory_config.get("summary_token_limit", 500)
        self.compression_enabled = memory_config.get("enable_compression", True)
        self.auto_cleanup_days = memory_config.get("auto_cleanup_days", 30)
        
        # Setup logging
        log_config = self.config.get("logging", {})
        self.logger = setup_logging(
            name="rag_chat.SnapshotManager",
            log_level=log_config.get("level", "INFO"),
            log_file=log_config.get("file_path"),
            console_level=log_config.get("console_level")
        )
        
        # LLM for summarization
        self.llm = llm
        
        # Load metadata
        self.alias_map = self._load_json(self.alias_file)
        self.sessions_meta = self._load_json(self.session_file)
        
        # Initialize monitoring
        self.metrics = get_monitoring_instance(config)
        
        # Current session state
        self.session_id = None
        self.session_path = None
        self.history = []
        self.metadata = None
        self.memory = None
        
        # Perform cleanup on initialization
        self._cleanup_old_sessions()
        
        self.logger.info(f"SnapshotManager initialized with {len(self.sessions_meta)} sessions")
    
    def _load_json(self, path: Path) -> dict:
        """Load JSON file with error handling"""
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load {path}: {e}")
                return {}
        return {}
    
    def _save_json(self, path: Path, data: dict):
        """Save JSON file with error handling"""
        try:
            ensure_directory(path.parent)
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save {path}: {e}")
    
    @handle_errors(logger=None, raise_on_error=True)
    def start_new_session(
        self, 
        alias: Optional[str] = None,
        model_name: Optional[str] = None,
        tags: List[str] = None
    ) -> str:
        """Start a new conversation session"""
        self.session_id = str(uuid.uuid4())
        self.session_path = self.session_dir / f"{self.session_id}.json"
        self.history = []
        
        now = datetime.now(timezone.utc)
        alias = alias or f"session_{now.strftime('%Y%m%d_%H%M%S')}"
        
        self.metadata = SessionMetadata(
            session_id=self.session_id,
            alias=alias,
            created=now,
            modified=now,
            file_path=str(self.session_path),
            model_used=model_name,
            tags=tags or []
        )
        
        # Initialize memory (will be upgraded to summary memory if needed)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.logger.info(f"Started new session: {self.session_id} (alias: {alias})")
        
        # Record metrics
        self.metrics.record_metric("session_started", 1.0)
        
        return self.session_id
    
    @time_operation("session_resume")
    def resume_session(
        self, 
        identifier: str,
        load_summary: bool = True
    ) -> Optional[ConversationBufferMemory]:
        """Resume an existing session"""
        # Resolve alias to session ID
        session_id = self.alias_map.get(identifier, identifier)
        meta_dict = self.sessions_meta.get(session_id)
        
        if not meta_dict:
            self.logger.error(f"Session not found: {identifier}")
            return None
        
        # Convert dict to SessionMetadata
        meta = self._dict_to_metadata(meta_dict)
        
        self.session_id = session_id
        self.session_path = Path(meta.file_path)
        self.metadata = meta
        
        # Check if file exists
        file_path = self.session_path
        if meta.compressed:
            file_path = Path(str(self.session_path) + ".gz")
        
        if not file_path.exists():
            self.logger.error(f"Session file missing: {file_path}")
            return None
        
        try:
            # Load session data
            session_data = self._load_session_file(file_path)
            self.history = session_data.get("history", [])
            
            # Create memory
            if load_summary and meta.summary_available and self.llm:
                memory = self._load_with_summary(session_id, self.history)
            else:
                memory = self._load_without_summary(self.history)
            
            self.memory = memory
            
            self.logger.info(
                f"Resumed session {session_id} with {len(self.history)} turns"
                f"{' (with summary)' if load_summary and meta.summary_available else ''}"
            )
            
            # Record metrics
            self.metrics.record_metric("session_resumed", 1.0)
            
            return memory
            
        except Exception as e:
            self.logger.error(f"Failed to resume session: {e}")
            return None
    
    def _load_session_file(self, file_path: Path) -> Dict[str, Any]:
        """Load session file, handling compression"""
        if str(file_path).endswith(".gz"):
            with gzip.open(file_path, "rt") as f:
                return json.load(f)
        else:
            with open(file_path, "r") as f:
                return json.load(f)