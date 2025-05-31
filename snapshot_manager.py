import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory


class SnapshotManager:
    def __init__(self, snapshot_dir: str = "./snapshots"):
        self.snapshot_dir = snapshot_dir
        self.session_dir = os.path.join(snapshot_dir, "sessions")
        self.metadata_dir = os.path.join(snapshot_dir, "metadata")
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        self.alias_file = os.path.join(self.metadata_dir, "aliases.json")
        self.session_file = os.path.join(self.metadata_dir, "sessions.json")
        
        self.alias_map = self._load_json(self.alias_file)
        self.sessions_meta = self._load_json(self.session_file)
        
        self.session_id = None
        self.session_path = None
        self.history = []
        self.metadata = {}
        
    def _load_json(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_json(self, path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
    def start_new_session(self, alias: Optional[str] = None) -> str:
        self.session_id = str(uuid.uuid4())
        self.session_path = os.path.join(self.session_dir, f"{self.session_id}.json")
        self.history = []
        
        now = datetime.now(timezone.utc).isoformat()
        alias = alias or self.session_id

        self.metadata = {
            "session_id": self.session_id,
            "alias": alias,
            "created": now,
            "modified": now,
            "file": self.session_path
        }
        
        #self.sessions_meta[self.session_id] = self.metadata
        #self.alias_map[alias] = self.session_id
        #self._save_json(self.session_file, self.sessions_meta)
        #self._save_json(self.alias_file, self.alias_map)
        
        return self.session_id

    def resume_session(self, identifier: str) -> Optional[ConversationBufferMemory]:
        session_id = self.alias_map.get(identifier, identifier)
        meta = self.sessions_meta.get(session_id)

        if not meta:
            print("❌ Session not found.")
            return None
        
        self.session_id = session_id
        self.session_path = meta["file"]
        self.metadata = meta

        if not os.path.exists(self.session_path):
            print("❌ Session file missing.")
            return None

        try:
            with open(self.session_path, "r") as f:
                data = json.load(f)
                self.history = data.get("history", [])
        except Exception:
            print("⚠️ Failed to load session history.")
            return None

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        for item in self.history:
            memory.chat_memory.add_user_message(HumanMessage(content=item["question"]))
            memory.chat_memory.add_ai_message(AIMessage(content=item["answer"]))
        
        return memory
    
    def resume_latest(self) -> Optional[ConversationBufferMemory]:
        sessions = self.list_sessions()
        if not sessions:
            print("❌ No sessions found.")
            return None
        return self.resume_session(sessions[0]["id"])
    
    def list_sessions(self) -> List[Dict]:
        sessions = []
        for sid, meta in self.sessions_meta.items():
            try:
                created = meta.get("created")
                modified = meta.get("modified")
                alias = meta.get("alias", sid)
                file_path = meta.get("file")
                
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    history = data.get("history", [])
                    first_msg = history[0]["question"] if history else ""
                else:
                    first_msg = "(missing session file)"

            except Exception:
                created = modified = None
                alias = sid
                first_msg = "(corrupt or empty)"
                
            sessions.append({
                "id": sid,
                "alias": alias,
                "created": created,
                "modified": modified,
                "first_msg": first_msg
            })
                
        def sort_key(s):
            return s.get("modified") or s.get("created")
  
        return sorted(sessions, key=sort_key, reverse=True)

    def record_turn(self, question: str, answer: str, sources: List[Dict]):
        self.history.append({
            "question": question,
            "answer": answer,
            "sources": sources
        })

    def save_snapshot(self):
        if not self.session_path:
            self.session_path = os.path.join(self.session_dir, f"{self.session_id}.json")

        self.metadata["modified"] = datetime.now(timezone.utc).isoformat()
        data = {
            "metadata": self.metadata,
            "history": self.history
        }

        with open(self.session_path, "w") as f:
            json.dump(data, f, indent=2)
            
        self.sessions_meta[self.session_id] = self.metadata
        self.alias_map[self.metadata["alias"]] = self.session_id
        self._save_json(self.session_file, self.sessions_meta)
        self._save_json(self.alias_file, self.alias_map)

        # Update metadata record
        self.sessions_meta[self.session_id] = self.metadata
        self._save_json(self.session_file, self.sessions_meta)
        print(f"💾 Snapshot saved to: {self.session_path}")
