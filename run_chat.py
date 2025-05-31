from vectorstore_manager import VectorstoreManager
from chat_agent import ChatAgent
from snapshot_manager import SnapshotManager
from get_llm import get_local_llm
from document_loader import SmartDocumentLoader
from langchain.memory import ConversationBufferMemory

def load_documents(config):
    loader = SmartDocumentLoader(config=config)
    documents = loader.load()
    return loader.split_documents(documents)

def update_vectorstore(config, chunks, skip_update=False):
    vs_manager = VectorstoreManager(config)
    vs_manager.load_vectorstore()
    if not skip_update and vs_manager.needs_update(chunks):
        vs_manager.add_documents(chunks)
    else:
        print("✅ Vectorstore is up to date.")
    return vs_manager.vs.as_retriever(search_kwargs={"k": 3})

def setup_llm(config, overrides={}):
    return get_local_llm(config, overrides)

def start_session(config, memory):
    return ChatAgent(config=config, llm=config["llm_instance"], retriever=config["retriever"], memory=memory)

def handle_session(config, override=None):
    snap = SnapshotManager(snapshot_dir=config.get("snapshot_path", "./snapshots"))

    if not override:
        print("\n🎯 Choose an option:\n1: Start new session\n2: Resume existing session\n3: Resume latest session")
        choice = input("Choice: ").strip()
    else:
        choice = {"new": "1", "resume": "2"}.get(override, "1")

    # Start new session
    if choice == "1":
        alias = input("Enter optional session alias (or press Enter to skip): ").strip() or None
        session_id = snap.start_new_session(alias=alias)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        print(f"🆕 Started session {session_id} (alias: {alias or session_id})")

    # Resume existing session    
    elif choice == "2":
        sessions = snap.list_sessions()
        if not sessions:
            print("❌ No sessions found. Starting new session.")
            session_id = snap.start_new_session()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        else:
            for i, s in enumerate(sessions): 
                print(f"{i+1}. Alias: {s['alias']}")
                print(f"   ID: {s['id']}")
                print(f"   Time: {s['modified']}")
                print(f"   Preview: {s['first_msg'][:80]}...\n")
            session_key = input("Enter Alias or session ID to resume: ").strip()
            memory = snap.resume_session(session_key)
            if memory is None:
                print("⚠️ Invalid ID/alias. Starting new session.")
                session_id = snap.start_new_session()
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            else:
                session_id = snap.session_id
                print(f"♻️ Resumed session {session_id}")

    #Resume latest session            
    elif choice == "3":
        memory = snap.resume_latest()
        if memory is None:
            print("⚠️ No previous sessions found. Starting new one.")
            session_id = snap.start_new_session()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        else:
            session_id = snap.session_id
            print(f"⏪ Resumed latest session {session_id}")
            
    else:
        print("❌ Invalid input, exiting."); exit(1)
        
    return snap, memory
