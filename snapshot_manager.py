import os
import json
import uuid
import gzip
import shutil
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.language_models.chat_models import BaseChatModel

# Import from enhanced utils
from utils import (setup_logging, handle_errors, time_operation, ensure_directory)
from monitoring import get_monitoring_instance


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
    """
    Manages the saving and loading of conversation snapshots, including the state
    of ConversationSummaryBufferMemory.
    """
    
    def __init__(
        self, 
        snapshot_dir: str = "./snapshots",
        llm: Optional[BaseChatModel] = None, # LLM is needed to re-init memory
        config: Dict[str, Any] = None
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.session_dir = self.snapshot_dir / "sessions"
        self.metadata_dir = self.snapshot_dir / "metadata"
        self.export_dir = self.snapshot_dir / "exports" # Added for exports
        
        # The summary_dir is no longer needed
        for dir_path in [self.session_dir, self.metadata_dir, self.export_dir]:
            ensure_directory(dir_path)
        
        # File paths
        self.alias_file = self.metadata_dir / "aliases.json"
        self.session_file = self.metadata_dir / "sessions.json"
        
        # Configuration
        self.config = config or {}
        memory_config = self.config.get("memory", {})
        self.compression_enabled = memory_config.get("enable_compression", True)
        self.auto_cleanup_days = memory_config.get("auto_cleanup_days", 0) # Default to off
        
        # Setup logging
        self.logger = setup_logging("rag_chat.SnapshotManager")
        
        # LLM for re-initializing ConversationSummaryBufferMemory
        self.llm = llm
        
        # Load metadata
        self.alias_map = self._load_json(self.alias_file)
        self.sessions_meta = self._load_json(self.session_file)
        
        # Initialize monitoring
        self.metrics = get_monitoring_instance(config)
        
        # Current session state
        self.session_id = None
        self.session_path = None
        self.metadata = None
        
        if self.auto_cleanup_days > 0:
            self._cleanup_old_sessions()
        
        self.logger.info(f"SnapshotManager initialized with {len(self.sessions_meta)} sessions")
    
    def _load_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, "r") as f: return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load {path}: {e}")
        return {}
    
    def _save_json(self, path: Path, data: dict):
        try:
            ensure_directory(path.parent)
            with open(path, "w") as f: json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save {path}: {e}")
    
    @handle_errors(logger=None, raise_on_error=True)
    def start_new_session(self, alias: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """Start a new conversation session"""
        self.session_id = str(uuid.uuid4())
        self.session_path = self.session_dir / f"{self.session_id}.json"
        
        now = datetime.now(timezone.utc)
        alias = alias or f"session_{now.strftime('%Y%m%d_%H%M%S')}"
        
        self.metadata = SessionMetadata(
            session_id=self.session_id,
            alias=alias,
            created=now,
            modified=now,
            file_path=str(self.session_path),
            model_used=model_name
        )
        
        self.logger.info(f"Started new session: {self.session_id} (alias: {alias})")
        self.metrics.record_metric("session_started", 1.0)
        return self.session_id
    
    @time_operation("session_resume")
    def resume_session(self, identifier: str) -> Optional[ConversationSummaryBufferMemory]:
        """
        Resumes an existing session by loading its state into a 
        ConversationSummaryBufferMemory object.
        """
        session_id = self.alias_map.get(identifier, identifier)
        meta_dict = self.sessions_meta.get(session_id)
        
        if not meta_dict:
            self.logger.error(f"Session not found: {identifier}")
            return None
        
        if not self.llm:
            self.logger.error("LLM instance is required to resume a session with summary memory.")
            return None

        meta = self._dict_to_metadata(meta_dict)
        self.session_id = session_id
        self.session_path = Path(meta.file_path)
        self.metadata = meta
        
        file_path = Path(str(self.session_path) + ".gz") if meta.compressed else self.session_path
        if not file_path.exists():
            self.logger.error(f"Session file missing: {file_path}")
            return None
        
        try:
            session_data = self._load_session_file(file_path)
            history_dicts = session_data.get("history", [])
            summary = session_data.get("summary", "")
            
            # Recreate the memory object
            memory_config = self.config.get("memory", {})
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=memory_config.get("summary_token_limit", 500),
                memory_key="chat_history",
                output_key='answer',
                return_messages=True
            )
            
            # Load the state into the new memory object
            memory.chat_memory.messages = messages_from_dict(history_dicts)
            memory.moving_summary_buffer = summary
            
            self.logger.info(f"Resumed session {session_id} with {len(history_dicts)} messages and a summary of {len(summary)} chars.")
            self.metrics.record_metric("session_resumed", 1.0)
            
            return memory
            
        except Exception as e:
            self.logger.error(f"Failed to resume session: {e}", exc_info=True)
            return None

    def _load_session_file(self, file_path: Path) -> Dict[str, Any]:
        """Load session file, handling compression."""
        open_func = gzip.open if str(file_path).endswith(".gz") else open
        mode = "rt" if str(file_path).endswith(".gz") else "r"
        with open_func(file_path, mode) as f:
            return json.load(f)
    
    def _save_session_file(self, file_path: Path, data: Dict[str, Any], compress: bool):
        """Save session file with optional compression."""
        target_path = Path(str(file_path) + ".gz") if compress else file_path
        open_func = gzip.open if compress else open
        mode = "wt" if compress else "w"
        with open_func(target_path, mode) as f:
            json.dump(data, f, indent=2, default=str)

    def _dict_to_metadata(self, data: Dict[str, Any]) -> SessionMetadata:
        """Convert dictionary to SessionMetadata object."""
        # Ensure only keys present in the dataclass are used for instantiation
        valid_keys = {f.name for f in SessionMetadata.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        for field in ["created", "modified"]:
            if field in filtered_data and isinstance(filtered_data[field], str):
                filtered_data[field] = datetime.fromisoformat(filtered_data[field])
                
        return SessionMetadata(**filtered_data)

    def resume_latest(self) -> Optional[ConversationSummaryBufferMemory]:
        """Resume the most recent session."""
        sessions = self.list_sessions()
        if not sessions:
            self.logger.warning("No sessions found")
            return None
        return self.resume_session(sessions[0]["id"])
    
    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List available sessions."""
        sessions = [self._dict_to_metadata(meta) for meta in self.sessions_meta.values()]
        sessions.sort(key=lambda x: x.modified, reverse=True)
        
        session_list = []
        for meta in sessions[:limit]:
            # Add preview of first message
            first_msg = "(missing session file)"
            file_path = Path(meta.file_path)
            if meta.compressed: file_path = Path(str(file_path) + ".gz")

            if file_path.exists():
                try:
                    data = self._load_session_file(file_path)
                    history = data.get("history", [])
                    # The first message is the first human message
                    if history and history[0]['type'] == 'human':
                        first_msg = history[0]['data']['content']
                except Exception: first_msg = "(error loading preview)"

            meta_dict = asdict(meta)
            meta_dict["first_msg"] = first_msg
            session_list.append(meta_dict)
            
        return session_list

    @handle_errors(logger=None, raise_on_error=False)
    def save_snapshot(self, memory: ConversationSummaryBufferMemory) -> bool:
        """
        Saves the current session snapshot from a ConversationSummaryBufferMemory object.
        """
        if not self.session_path or not self.metadata:
            self.logger.error("No active session to save")
            return False
        
        try:
            # Extract state from memory object
            messages = memory.chat_memory.messages
            summary = memory.moving_summary_buffer
            
            # Serialize messages to a JSON-friendly format
            history_dicts = messages_to_dict(messages)

            # Update metadata
            self.metadata.modified = datetime.now(timezone.utc)
            self.metadata.total_turns = len(messages) // 2
            self.metadata.summary_available = bool(summary)
            self.metadata.total_tokens = self._estimate_tokens(memory.buffer_as_str)

            # Prepare session data for saving
            session_data = {
                "metadata": asdict(self.metadata),
                "history": history_dicts,
                "summary": summary,
                "version": "3.0"
            }
            
            compress = self.compression_enabled
            self._save_session_file(self.session_path, session_data, compress=compress)
            
            actual_path = Path(str(self.session_path) + ".gz") if compress else self.session_path
            self.metadata.compressed = compress
            if actual_path.exists(): self.metadata.size_bytes = actual_path.stat().st_size
            
            self.sessions_meta[self.session_id] = asdict(self.metadata)
            self.alias_map[self.metadata.alias] = self.session_id
            self._save_json(self.session_file, self.sessions_meta)
            self._save_json(self.alias_file, self.alias_map)
            
            self.logger.info(f"Saved snapshot: {self.session_id} ({self.metadata.total_turns} turns, {self.metadata.size_bytes / 1024:.1f}KB)")
            self.metrics.record_metric("snapshot_saved", 1.0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}", exc_info=True)
            return False
    
    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def export_session(self, session_id: str, format: str = "json", include_summary: bool = True) -> Optional[str]:
        """
        Exports a session in various formats, updated for the new data structure.
        """
        session_id = self.alias_map.get(session_id, session_id)
        meta_dict = self.sessions_meta.get(session_id)
        if not meta_dict:
            self.logger.error(f"Session not found: {session_id}")
            return None
        
        try:
            meta = self._dict_to_metadata(meta_dict)
            file_path = Path(str(meta.file_path) + ".gz") if meta.compressed else Path(meta.file_path)
            session_data = self._load_session_file(file_path)
            
            export_path = self.export_dir / f"{session_id}.{format}"
            
            history = session_data.get("history", [])
            summary = session_data.get("summary") if include_summary else None

            if format == "json":
                with open(export_path, "w") as f:
                    json.dump(session_data, f, indent=2, default=str)
            
            elif format == "txt":
                with open(export_path, "w", encoding="utf-8") as f:
                    f.write(f"Session: {meta.alias}\nCreated: {meta.created}\nTotal turns: {meta.total_turns}\n{'='*60}\n\n")
                    if summary: f.write(f"SUMMARY:\n{summary}\n\n{'='*60}\n\n")
                    # Process history in pairs
                    for i in range(0, len(history), 2):
                        question = history[i]['data']['content']
                        answer = history[i+1]['data']['content'] if (i+1) < len(history) else "[No answer]"
                        f.write(f"Q: {question}\nA: {answer}\n{'-'*40}\n")
            
            elif format == "md":
                with open(export_path, "w", encoding="utf-8") as f:
                    f.write(f"# Conversation: {meta.alias}\n\n**Created:** {meta.created}\n**Total turns:** {meta.total_turns}\n\n")
                    if summary: f.write(f"## Summary\n\n{summary}\n\n")
                    f.write("## Conversation\n\n")
                    # Process history in pairs
                    for i in range(0, len(history), 2):
                        question = history[i]['data']['content']
                        answer = history[i+1]['data']['content'] if (i+1) < len(history) else "[No answer]"
                        f.write(f"### Turn {i//2 + 1}\n\n**Question:** {question}\n\n**Answer:** {answer}\n\n---\n\n")
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return None
            
            self.logger.info(f"Exported session {session_id} to {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export session {session_id}: {e}", exc_info=True)
            return None
    
    def search_sessions(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Searches through session history, updated for the new data structure.
        """
        results = []
        query_lower = query.lower()
        
        for session_id, meta_dict in self.sessions_meta.items():
            try:
                meta = self._dict_to_metadata(meta_dict)
                file_path = Path(str(meta.file_path) + ".gz") if meta.compressed else Path(meta.file_path)
                if not file_path.exists(): continue
                
                session_data = self._load_session_file(file_path)
                history = session_data.get("history", [])
                
                matches = []
                for i, msg in enumerate(history):
                    if query_lower in msg['data']['content'].lower():
                        turn_index = i // 2
                        # Reconstruct the turn for context
                        if msg['type'] == 'human':
                            question = msg['data']['content']
                            answer = history[i+1]['data']['content'] if (i+1) < len(history) else "[No answer]"
                        else: # AI message
                            question = history[i-1]['data']['content'] if (i-1) >= 0 else "[No question]"
                            answer = msg['data']['content']
                        
                        matches.append({"turn_index": turn_index, "question": question[:100], "answer": answer[:100]})
                
                if matches:
                    # Remove duplicate turns from matches
                    unique_matches = [dict(t) for t in {tuple(d.items()) for d in matches}]
                    results.append({
                        "session_id": session_id, "alias": meta.alias, "created": meta.created,
                        "match_count": len(unique_matches), "matches": unique_matches[:3]
                    })
            except Exception as e:
                self.logger.error(f"Error searching session {session_id}: {e}")
        
        results.sort(key=lambda x: x["match_count"], reverse=True)
        return results[:limit]

    def _cleanup_old_sessions(self):
        """Clean up old sessions based on retention policy."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.auto_cleanup_days)
        to_remove = [sid for sid, meta in self.sessions_meta.items() if self._dict_to_metadata(meta).modified < cutoff]
        for sid in to_remove: self.delete_session(sid)
        if to_remove: self.logger.info(f"Cleaned up {len(to_remove)} old sessions.")
    
    def delete_session(self, session_id: str) -> bool:
        """
        Deletes a session and all associated files. Now simpler without a separate summary file.
        """
        session_id = self.alias_map.get(session_id, session_id)
        meta_dict = self.sessions_meta.get(session_id)
        if not meta_dict: return False
        
        try:
            meta = self._dict_to_metadata(meta_dict)
            
            # Delete session file (.json or .json.gz)
            for ext in ["", ".gz"]:
                full_path = Path(str(meta.file_path) + ext)
                if full_path.exists(): full_path.unlink()
            
            # The separate summary file is gone, so no need to delete it.

            # Remove from metadata
            del self.sessions_meta[session_id]
            if meta.alias in self.alias_map: del self.alias_map[meta.alias]
            
            self._save_json(self.session_file, self.sessions_meta)
            self._save_json(self.alias_file, self.alias_map)
            
            self.logger.info(f"Deleted session: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all sessions. No changes needed here as it reads
        from the pre-calculated metadata.
        """
        total_sessions = len(self.sessions_meta)
        total_turns = sum(self._dict_to_metadata(m).total_turns for m in self.sessions_meta.values())
        total_tokens = sum(self._dict_to_metadata(m).total_tokens for m in self.sessions_meta.values())
        total_size_bytes = sum(self._dict_to_metadata(m).size_bytes for m in self.sessions_meta.values())
        
        return {
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "total_tokens": total_tokens,
            "total_size_mb": total_size_bytes / (1024 * 1024),
            "average_turns_per_session": total_turns / max(total_sessions, 1),
        }