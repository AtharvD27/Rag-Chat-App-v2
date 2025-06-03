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
    
    def _save_session_file(self, file_path: Path, data: Dict[str, Any], compress: bool = False):
        """Save session file with optional compression"""
        if compress:
            file_path = Path(str(file_path) + ".gz")
            with gzip.open(file_path, "wt") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
    
    def _load_with_summary(
        self, 
        session_id: str, 
        history: List[Dict]
    ) -> ConversationSummaryBufferMemory:
        """Load session with conversation summary"""
        # Load summary if available
        summary_path = self.summary_dir / f"{session_id}_summary.json"
        summary_data = {}
        
        if summary_path.exists():
            try:
                with open(summary_path, "r") as f:
                    summary_data = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load summary: {e}")
        
        # Create summary buffer memory
        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=self.summary_token_limit
        )
        
        # Load summary if available
        if summary_data.get("summary"):
            memory.moving_summary_buffer = summary_data["summary"]
            self.logger.debug(f"Loaded existing summary: {len(memory.moving_summary_buffer)} chars")
        
        # Load recent messages (keep last N turns in full)
        recent_turns = min(10, len(history))  # Keep last 10 turns in detail
        start_idx = max(0, len(history) - recent_turns)
        
        for item in history[start_idx:]:
            memory.chat_memory.add_user_message(HumanMessage(content=item["question"]))
            memory.chat_memory.add_ai_message(AIMessage(content=item["answer"]))
        
        return memory
    
    def _load_without_summary(self, history: List[Dict]) -> ConversationBufferMemory:
        """Load session without summary (traditional method)"""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        for item in history:
            memory.chat_memory.add_user_message(HumanMessage(content=item["question"]))
            memory.chat_memory.add_ai_message(AIMessage(content=item["answer"]))
        
        return memory
    
    def _dict_to_metadata(self, data: Dict[str, Any]) -> SessionMetadata:
        """Convert dictionary to SessionMetadata object"""
        # Handle datetime conversion
        for field in ["created", "modified"]:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return SessionMetadata(**data)
    
    def resume_latest(self) -> Optional[ConversationBufferMemory]:
        """Resume the most recent session"""
        sessions = self.list_sessions()
        if not sessions:
            self.logger.warning("No sessions found")
            return None
        
        latest = sessions[0]
        return self.resume_session(latest["id"])
    
    def list_sessions(
        self, 
        limit: int = 50,
        tag_filter: List[str] = None
    ) -> List[Dict[str, Any]]:
        """List available sessions with filtering"""
        sessions = []
        
        for sid, meta_dict in self.sessions_meta.items():
            try:
                meta = self._dict_to_metadata(meta_dict)
                
                # Apply tag filter if specified
                if tag_filter and not any(tag in meta.tags for tag in tag_filter):
                    continue
                
                # Load first message for preview
                first_msg = ""
                file_path = Path(meta.file_path)
                if meta.compressed:
                    file_path = Path(str(file_path) + ".gz")
                
                if file_path.exists():
                    try:
                        data = self._load_session_file(file_path)
                        history = data.get("history", [])
                        if history:
                            first_msg = history[0]["question"][:100]
                    except Exception:
                        first_msg = "(error loading preview)"
                else:
                    first_msg = "(missing session file)"
                
                sessions.append({
                    "id": sid,
                    "alias": meta.alias,
                    "created": meta.created,
                    "modified": meta.modified,
                    "turns": meta.total_turns,
                    "tokens": meta.total_tokens,
                    "size_mb": meta.size_bytes / (1024 * 1024),
                    "summary_available": meta.summary_available,
                    "compressed": meta.compressed,
                    "model": meta.model_used,
                    "tags": meta.tags,
                    "first_msg": first_msg
                })
                
            except Exception as e:
                self.logger.error(f"Error processing session {sid}: {e}")
                continue
        
        # Sort by modified date (most recent first)
        sessions.sort(key=lambda x: x["modified"], reverse=True)
        
        return sessions[:limit]
    
    @time_operation("record_turn")
    def record_turn(
        self, 
        question: str, 
        answer: str, 
        sources: List[Dict]
    ) -> None:
        """Record a conversation turn with automatic summarization check"""
        turn = ConversationTurn(
            question=question,
            answer=answer,
            sources=sources,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "question_length": len(question),
                "answer_length": len(answer),
                "source_count": len(sources)
            }
        )
        
        # Convert to dict for storage
        turn_dict = {
            "question": turn.question,
            "answer": turn.answer,
            "sources": turn.sources,
            "timestamp": turn.timestamp.isoformat(),
            "metadata": turn.metadata
        }
        
        self.history.append(turn_dict)
        
        # Update metadata
        if self.metadata:
            self.metadata.total_turns += 1
            self.metadata.total_tokens += self._estimate_tokens(question + answer)
            self.metadata.modified = datetime.now(timezone.utc)
        
        # Check if we need to trigger summarization
        if (self.llm and 
            self.metadata.total_turns > 0 and 
            self.metadata.total_turns % self.max_turns_before_summary == 0):
            
            self.logger.info(f"Triggering automatic summarization at {self.metadata.total_turns} turns")
            asyncio.create_task(self._async_summarize())
        
        # Record metrics
        self.metrics.record_metric("turn_recorded", 1.0)
        self.metrics.record_metric("turn_length", len(question) + len(answer))
    
    async def _async_summarize(self):
        """Asynchronously generate conversation summary"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._generate_summary
            )
        except Exception as e:
            self.logger.error(f"Async summarization failed: {e}")
    
    def _generate_summary(self) -> Optional[str]:
        """Generate a summary of the conversation"""
        if not self.llm or not self.memory:
            return None
        
        try:
            # If using ConversationSummaryBufferMemory, it handles this automatically
            if isinstance(self.memory, ConversationSummaryBufferMemory):
                summary = self.memory.moving_summary_buffer
            else:
                # Manual summarization for regular memory
                messages = self.memory.chat_memory.messages
                
                # Create a condensed version of the conversation
                conversation_text = "\n".join([
                    f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
                    for msg in messages[:-10]  # Summarize all but last 10 messages
                ])
                
                summary_prompt = (
                    "Please provide a concise summary of the following conversation, "
                    "highlighting key topics, questions asked, and important information provided:\n\n"
                    f"{conversation_text}\n\n"
                    "Summary:"
                )
                
                summary = self.llm.predict(summary_prompt)
            
            # Save summary
            if summary and self.session_id:
                summary_path = self.summary_dir / f"{self.session_id}_summary.json"
                summary_data = {
                    "session_id": self.session_id,
                    "summary": summary,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "turn_count": self.metadata.total_turns,
                    "model_used": self.llm.__class__.__name__
                }
                
                with open(summary_path, "w") as f:
                    json.dump(summary_data, f, indent=2)
                
                self.metadata.summary_available = True
                self.logger.info(f"Generated summary of {len(summary)} characters")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            return None
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    @handle_errors(logger=None, raise_on_error=False)
    def save_snapshot(self, force_compression: bool = None) -> bool:
        """Save the current session snapshot"""
        if not self.session_path or not self.metadata:
            self.logger.error("No active session to save")
            return False
        
        try:
            # Update metadata
            self.metadata.modified = datetime.now(timezone.utc)
            
            # Prepare session data
            session_data = {
                "metadata": asdict(self.metadata),
                "history": self.history,
                "version": "2.0"  # Version for compatibility
            }
            
            # Determine if we should compress
            compress = force_compression if force_compression is not None else self.compression_enabled
            
            # Save session file
            self._save_session_file(self.session_path, session_data, compress=compress)
            
            # Update metadata
            if compress:
                actual_path = Path(str(self.session_path) + ".gz")
                self.metadata.compressed = True
            else:
                actual_path = self.session_path
                self.metadata.compressed = False
            
            # Get file size
            if actual_path.exists():
                self.metadata.size_bytes = actual_path.stat().st_size
            
            # Generate summary if needed
            if self.llm and self.metadata.total_turns >= self.max_turns_before_summary:
                self._generate_summary()
            
            # Update metadata files
            self.sessions_meta[self.session_id] = asdict(self.metadata)
            self.alias_map[self.metadata.alias] = self.session_id
            
            self._save_json(self.session_file, self.sessions_meta)
            self._save_json(self.alias_file, self.alias_map)
            
            self.logger.info(
                f"Saved snapshot: {self.session_id} "
                f"({self.metadata.total_turns} turns, "
                f"{self.metadata.size_bytes / 1024:.1f}KB"
                f"{' compressed' if compress else ''})"
            )
            
            # Record metrics
            self.metrics.record_metric("snapshot_saved", 1.0)
            self.metrics.record_metric("snapshot_size_kb", self.metadata.size_bytes / 1024)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")
            return False
    
    def export_session(
        self, 
        session_id: str,
        format: str = "json",
        include_summary: bool = True
    ) -> Optional[str]:
        """Export a session in various formats"""
        session_id = self.alias_map.get(session_id, session_id)
        
        if session_id not in self.sessions_meta:
            self.logger.error(f"Session not found: {session_id}")
            return None
        
        try:
            # Load session data
            meta = self._dict_to_metadata(self.sessions_meta[session_id])
            file_path = Path(meta.file_path)
            if meta.compressed:
                file_path = Path(str(file_path) + ".gz")
            
            session_data = self._load_session_file(file_path)
            
            # Load summary if requested
            summary = None
            if include_summary and meta.summary_available:
                summary_path = self.summary_dir / f"{session_id}_summary.json"
                if summary_path.exists():
                    with open(summary_path, "r") as f:
                        summary_data = json.load(f)
                        summary = summary_data.get("summary")
            
            # Export based on format
            export_path = self.snapshot_dir / "exports" / f"{session_id}.{format}"
            ensure_directory(export_path.parent)
            
            if format == "json":
                export_data = {
                    "session": session_data,
                    "summary": summary
                }
                with open(export_path, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format == "txt":
                with open(export_path, "w") as f:
                    f.write(f"Session: {meta.alias}\n")
                    f.write(f"Created: {meta.created}\n")
                    f.write(f"Total turns: {meta.total_turns}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    if summary:
                        f.write("SUMMARY:\n")
                        f.write(summary + "\n\n")
                        f.write("=" * 60 + "\n\n")
                    
                    for turn in session_data["history"]:
                        f.write(f"Q: {turn['question']}\n")
                        f.write(f"A: {turn['answer']}\n")
                        f.write("-" * 40 + "\n")
            
            elif format == "md":
                with open(export_path, "w") as f:
                    f.write(f"# Conversation: {meta.alias}\n\n")
                    f.write(f"**Created:** {meta.created}\n")
                    f.write(f"**Total turns:** {meta.total_turns}\n\n")
                    
                    if summary:
                        f.write("## Summary\n\n")
                        f.write(f"{summary}\n\n")
                    
                    f.write("## Conversation\n\n")
                    for i, turn in enumerate(session_data["history"], 1):
                        f.write(f"### Turn {i}\n\n")
                        f.write(f"**Question:** {turn['question']}\n\n")
                        f.write(f"**Answer:** {turn['answer']}\n\n")
                        if turn.get("sources"):
                            f.write("**Sources:**\n")
                            for src in turn["sources"]:
                                f.write(f"- {src.get('file', 'Unknown')}\n")
                        f.write("\n---\n\n")
            
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return None
            
            self.logger.info(f"Exported session {session_id} to {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export session: {e}")
            return None
    
    def search_sessions(
        self,
        query: str,
        search_in: List[str] = ["questions", "answers"],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search through session history"""
        results = []
        query_lower = query.lower()
        
        for session_id, meta_dict in self.sessions_meta.items():
            try:
                meta = self._dict_to_metadata(meta_dict)
                file_path = Path(meta.file_path)
                if meta.compressed:
                    file_path = Path(str(file_path) + ".gz")
                
                if not file_path.exists():
                    continue
                
                # Load session
                session_data = self._load_session_file(file_path)
                history = session_data.get("history", [])
                
                # Search through history
                matches = []
                for i, turn in enumerate(history):
                    match_found = False
                    
                    if "questions" in search_in and query_lower in turn["question"].lower():
                        match_found = True
                    elif "answers" in search_in and query_lower in turn["answer"].lower():
                        match_found = True
                    
                    if match_found:
                        matches.append({
                            "turn_index": i,
                            "question": turn["question"][:100],
                            "answer": turn["answer"][:100]
                        })
                
                if matches:
                    results.append({
                        "session_id": session_id,
                        "alias": meta.alias,
                        "created": meta.created,
                        "match_count": len(matches),
                        "matches": matches[:3]  # First 3 matches
                    })
                
            except Exception as e:
                self.logger.error(f"Error searching session {session_id}: {e}")
                continue
        
        # Sort by match count
        results.sort(key=lambda x: x["match_count"], reverse=True)
        
        return results[:limit]
    
    def _cleanup_old_sessions(self):
        """Clean up old sessions based on retention policy"""
        if self.auto_cleanup_days <= 0:
            return
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.auto_cleanup_days)
        sessions_to_remove = []
        
        for session_id, meta_dict in self.sessions_meta.items():
            try:
                meta = self._dict_to_metadata(meta_dict)
                if meta.modified < cutoff_date:
                    sessions_to_remove.append(session_id)
            except Exception:
                continue
        
        for session_id in sessions_to_remove:
            self.delete_session(session_id)
        
        if sessions_to_remove:
            self.logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all associated files"""
        session_id = self.alias_map.get(session_id, session_id)
        
        if session_id not in self.sessions_meta:
            return False
        
        try:
            meta = self._dict_to_metadata(self.sessions_meta[session_id])
            
            # Delete session file
            file_path = Path(meta.file_path)
            for ext in ["", ".gz"]:
                full_path = Path(str(file_path) + ext)
                if full_path.exists():
                    full_path.unlink()
            
            # Delete summary file
            summary_path = self.summary_dir / f"{session_id}_summary.json"
            if summary_path.exists():
                summary_path.unlink()
            
            # Remove from metadata
            del self.sessions_meta[session_id]
            if meta.alias in self.alias_map:
                del self.alias_map[meta.alias]
            
            # Save updated metadata
            self._save_json(self.session_file, self.sessions_meta)
            self._save_json(self.alias_file, self.alias_map)
            
            self.logger.info(f"Deleted session: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about all sessions"""
        total_sessions = len(self.sessions_meta)
        total_turns = 0
        total_tokens = 0
        total_size_bytes = 0
        
        models_used = {}
        tags_count = {}
        
        for meta_dict in self.sessions_meta.values():
            try:
                meta = self._dict_to_metadata(meta_dict)
                total_turns += meta.total_turns
                total_tokens += meta.total_tokens
                total_size_bytes += meta.size_bytes
                
                if meta.model_used:
                    models_used[meta.model_used] = models_used.get(meta.model_used, 0) + 1
                
                for tag in meta.tags:
                    tags_count[tag] = tags_count.get(tag, 0) + 1
                    
            except Exception:
                continue
        
        return {
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "total_tokens": total_tokens,
            "total_size_mb": total_size_bytes / (1024 * 1024),
            "average_turns_per_session": total_turns / max(total_sessions, 1),
            "average_tokens_per_session": total_tokens / max(total_sessions, 1),
            "models_used": models_used,
            "popular_tags": sorted(tags_count.items(), key=lambda x: x[1], reverse=True)[:10]
        }


# Example usage
if __name__ == "__main__":
    from utils import load_config
    from get_llm import get_local_llm
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Get LLM for summarization (optional)
    try:
        llm = get_local_llm(config)
    except Exception:
        llm = None
        print("Warning: No LLM available for summarization")
    
    # Create snapshot manager
    manager = SnapshotManager(
        snapshot_dir=config.get("snapshot_path", "./snapshots"),
        llm=llm,
        config=config
    )
    
    # Get statistics
    stats = manager.get_session_statistics()
    print(f"Session Statistics:")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Total turns: {stats['total_turns']}")
    print(f"  Total size: {stats['total_size_mb']:.2f}MB")
    
    # List recent sessions
    sessions = manager.list_sessions(limit=5)
    if sessions:
        print(f"\nRecent Sessions:")
        for session in sessions:
            print(f"  - {session['alias']} ({session['turns']} turns, {session['size_mb']:.2f}MB)")
            print(f"    Created: {session['created']}")
            print(f"    Preview: {session['first_msg'][:50]}...")
    
    # Example: Search sessions
    results = manager.search_sessions("Python", search_in=["questions", "answers"])
    if results:
        print(f"\nSearch results for 'Python':")
        for result in results[:3]:
            print(f"  - {result['alias']} ({result['match_count']} matches)")