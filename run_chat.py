from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
from pathlib import Path

from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel

# Import from enhanced modules
from utils import (
    setup_logging, handle_errors, time_operation, performance_monitor
)
from vectorstore_manager import VectorstoreManager
from chat_agent import ChatAgent, HybridRetriever
from snapshot_manager import SnapshotManager
from get_llm import ModelManager, get_local_llm
from document_loader import SmartDocumentLoader
from monitoring import get_monitoring_instance


class SessionOrchestrator:
    """Orchestrates all components for a chat session"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Setup logging
        log_config = self.config.get("logging", {})
        self.logger = setup_logging(
            name="rag_chat.SessionOrchestrator",
            log_level=log_config.get("level", "INFO"),
            log_file=log_config.get("file_path"),
            console_level=log_config.get("console_level")
        )
        
        # Initialize monitoring
        self.metrics = get_monitoring_instance(config)
        
        # Component instances
        self.document_loader = None
        self.vectorstore_manager = None
        self.model_manager = None
        self.current_llm = None
        self.current_retriever = None
        
        self.logger.info("SessionOrchestrator initialized")


@time_operation("document_loading_orchestration")
@handle_errors(logger=setup_logging("rag_chat.run_chat"), raise_on_error=True)
def load_documents(config: Dict[str, Any]) -> List:
    """Load documents with enhanced loader"""
    logger = setup_logging("rag_chat.load_documents")
    logger.info("Starting document loading orchestration")
    
    try:
        # Create smart document loader
        loader = SmartDocumentLoader(config=config)
        
        # Load documents (will use async if enabled)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        
        # Split into chunks
        logger.info("Splitting documents into chunks")
        chunks = loader.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Record metrics
        metrics = get_monitoring_instance(config)
        metrics.record_metric("documents_loaded", len(documents))
        metrics.record_metric("chunks_created", len(chunks))
        
        return chunks
        
    except Exception as e:
        logger.error(f"Document loading failed: {e}")
        raise


@time_operation("vectorstore_update_orchestration")
@handle_errors(logger=setup_logging("rag_chat.run_chat"), raise_on_error=True)
def update_vectorstore(
    config: Dict[str, Any], 
    chunks: List, 
    skip_update: bool = False
) -> Any:
    """Update vectorstore with enhanced manager"""
    logger = setup_logging("rag_chat.update_vectorstore")
    logger.info("Starting vectorstore update orchestration")
    
    try:
        # Create vectorstore manager
        vs_manager = VectorstoreManager(config)
        
        # Load existing vectorstore
        vs_manager.load_vectorstore()
        
        # Check health
        health = vs_manager.get_health_status()
        logger.info(f"Vectorstore health: {health['is_healthy']}")
        
        # Update if needed
        if not skip_update and vs_manager.needs_update(chunks):
            logger.info("Updating vectorstore with new chunks")
            vs_manager.add_documents(chunks)
            
            # Optimize after update
            vs_manager.optimize_index()
        else:
            logger.info("Vectorstore is up to date")
        
        # Create retriever based on configuration
        retrieval_config = config.get("retrieval", {})
        search_type = retrieval_config.get("search_type", "hybrid")
        
        if search_type == "hybrid":
            logger.info("Creating hybrid retriever")
            base_retriever = vs_manager.vs.as_retriever(
                search_kwargs={"k": retrieval_config.get("top_k", 5) * 2}
            )
            # Note: In production, we'd pass the document list for BM25
            # For now, return the base retriever with a note
            retriever = base_retriever
            logger.warning("Hybrid search requested but document list not available for BM25. Using semantic search only.")
        else:
            logger.info(f"Creating {search_type} retriever")
            retriever = vs_manager.vs.as_retriever(
                search_kwargs={"k": retrieval_config.get("top_k", 5)}
            )
        
        # Record metrics
        metrics = get_monitoring_instance(config)
        metrics.record_metric("vectorstore_documents", health['total_documents'])
        metrics.record_metric("vectorstore_size_mb", health['index_size_mb'])
        
        return retriever
        
    except Exception as e:
        logger.error(f"Vectorstore update failed: {e}")
        raise


@time_operation("llm_setup_orchestration")
@handle_errors(logger=setup_logging("rag_chat.run_chat"), raise_on_error=True)
def setup_llm(
    config: Dict[str, Any], 
    overrides: Dict[str, Any] = None
) -> Union[BaseLLM, BaseChatModel]:
    """Setup LLM with enhanced multi-model support"""
    logger = setup_logging("rag_chat.setup_llm")
    logger.info("Starting LLM setup orchestration")
    
    try:
        # Check if we have multi-model configuration
        if "models" in config and "profiles" in config["models"]:
            logger.info("Using ModelManager for multi-model support")
            
            # Create model manager
            manager = ModelManager(config)
            
            # Get default profile or use override
            profile_name = None
            if overrides and "model_profile" in overrides:
                profile_name = overrides["model_profile"]
            
            # Get model
            llm = manager.get_model(profile_name, use_fallback=True)
            
            # Store manager in config for later use
            config["_model_manager"] = manager
            
            logger.info(f"Loaded model with profile: {profile_name or 'default'}")
            
        else:
            logger.info("Using legacy single-model configuration")
            
            # Apply overrides
            if overrides:
                if "model_path" in overrides and overrides["model_path"]:
                    config.setdefault("llm", {})["local_model_path"] = overrides["model_path"]
                if "temperature" in overrides and overrides["temperature"] is not None:
                    config.setdefault("llm", {})["temperature"] = overrides["temperature"]
            
            # Use legacy function
            llm = get_local_llm(config, overrides)
            logger.info("Loaded local LLM")
        
        # Record metrics
        metrics = get_monitoring_instance(config)
        metrics.record_metric("llm_loaded", 1.0)
        
        return llm
        
    except Exception as e:
        logger.error(f"LLM setup failed: {e}")
        raise


@time_operation("session_start_orchestration")
def start_session(config: Dict[str, Any], memory: ConversationBufferMemory) -> ChatAgent:
    """Start a new chat session with enhanced agent"""
    logger = setup_logging("rag_chat.start_session")
    logger.info("Starting new chat session")
    
    try:
        # Get components from config
        llm = config.get("llm_instance")
        retriever = config.get("retriever")
        
        if not llm or not retriever:
            raise ValueError("LLM and retriever must be initialized before starting session")
        
        # Create enhanced chat agent
        agent = ChatAgent(
            llm=llm,
            retriever=retriever,
            memory=memory,
            config=config
        )
        
        logger.info("Chat session started successfully")
        
        # Record metrics
        metrics = get_monitoring_instance(config)
        metrics.record_metric("session_created", 1.0)
        
        return agent
        
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise


@handle_errors(logger=setup_logging("rag_chat.run_chat"), raise_on_error=True)
def handle_session(
    config: Dict[str, Any], 
    override: Optional[str] = None
) -> Tuple[SnapshotManager, ConversationBufferMemory]:
    """Handle session selection and initialization"""
    logger = setup_logging("rag_chat.handle_session")
    
    # Create snapshot manager with LLM for summarization
    llm = config.get("llm_instance")
    snap = SnapshotManager(
        snapshot_dir=config.get("snapshot_path", "./snapshots"),
        llm=llm,
        config=config
    )
    
    # Determine session mode
    if override:
        choice = {"new": "1", "resume": "2"}.get(override, "1")
    else:
        # Show session statistics
        stats = snap.get_session_statistics()
        if stats["total_sessions"] > 0:
            print(f"\n📊 Session Overview:")
            print(f"   Total sessions: {stats['total_sessions']}")
            print(f"   Total conversations: {stats['total_turns']}")
            print(f"   Storage used: {stats['total_size_mb']:.2f}MB")
        
        print("\n🎯 Choose an option:")
        print("1: Start new session")
        print("2: Resume existing session")
        print("3: Resume latest session")
        print("4: Search sessions")
        
        choice = input("Choice (1-4): ").strip()
    
    # Handle choice
    if choice == "1":
        # Start new session
        alias = None
        if not override:
            alias = input("Enter session alias (optional): ").strip() or None
        
        # Get model name for metadata
        model_name = None
        if hasattr(config.get("llm_instance"), "__class__"):
            model_name = config["llm_instance"].__class__.__name__
        
        session_id = snap.start_new_session(alias=alias, model_name=model_name)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        logger.info(f"Started new session: {session_id}")
        print(f"🆕 Started session: {alias or session_id}")
        
    elif choice == "2" or (choice == "4" and not override):
        # Search or list sessions
        if choice == "4":
            query = input("Search query (or press Enter to list all): ").strip()
            if query:
                sessions = snap.search_sessions(query)
                print(f"\n🔍 Search results for '{query}':")
            else:
                sessions = snap.list_sessions()
                print("\n📋 Available sessions:")
        else:
            sessions = snap.list_sessions()
            print("\n📋 Available sessions:")
        
        if not sessions:
            print("❌ No sessions found. Starting new session.")
            session_id = snap.start_new_session()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        else:
            # Display sessions
            for i, s in enumerate(sessions[:20], 1):  # Show max 20
                print(f"\n{i}. {s['alias']}")
                print(f"   ID: {s['id']}")
                print(f"   Created: {s['created'].strftime('%Y-%m-%d %H:%M')}")
                print(f"   Turns: {s['turns']}, Tokens: {s['tokens']:,}")
                if s.get('model'):
                    print(f"   Model: {s['model']}")
                if s.get('summary_available'):
                    print(f"   📝 Summary available")
                print(f"   Preview: {s['first_msg'][:80]}...")
            
            # Get selection
            if choice == "4":
                choice = "2"  # Switch to resume mode
            
            selection = input("\nEnter number or session ID/alias: ").strip()
            
            # Handle selection
            if selection.isdigit() and 1 <= int(selection) <= len(sessions):
                session_key = sessions[int(selection) - 1]["id"]
            else:
                session_key = selection
            
            # Resume session
            memory = snap.resume_session(session_key)
            if memory is None:
                print("⚠️ Invalid selection. Starting new session.")
                session_id = snap.start_new_session()
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            else:
                session_id = snap.session_id
                print(f"♻️ Resumed session: {snap.metadata.alias}")
                
                # Show session info
                print(f"   Total turns: {snap.metadata.total_turns}")
                if snap.metadata.summary_available:
                    print(f"   📝 Summary loaded")
    
    elif choice == "3":
        # Resume latest session
        memory = snap.resume_latest()
        if memory is None:
            print("⚠️ No previous sessions found. Starting new session.")
            session_id = snap.start_new_session()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        else:
            session_id = snap.session_id
            print(f"⏪ Resumed latest session: {snap.metadata.alias}")
    
    else:
        print("❌ Invalid choice. Starting new session.")
        session_id = snap.start_new_session()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    return snap, memory


def switch_model(config: Dict[str, Any], profile_name: str) -> Union[BaseLLM, BaseChatModel]:
    """Switch to a different model profile during runtime"""
    logger = setup_logging("rag_chat.switch_model")
    
    try:
        manager = config.get("_model_manager")
        if not manager:
            logger.error("ModelManager not initialized")
            return config.get("llm_instance")
        
        logger.info(f"Switching to model profile: {profile_name}")
        
        # Get new model
        new_llm = manager.get_model(profile_name, use_fallback=True)
        
        # Update config
        config["llm_instance"] = new_llm
        
        logger.info(f"Successfully switched to model: {profile_name}")
        return new_llm
        
    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        return config.get("llm_instance")


async def run_async_orchestration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete orchestration asynchronously"""
    logger = setup_logging("rag_chat.async_orchestration")
    logger.info("Starting async orchestration")
    
    try:
        # Create orchestrator
        orchestrator = SessionOrchestrator(config)
        
        # Run components in parallel where possible
        async def load_docs():
            return await asyncio.get_event_loop().run_in_executor(
                None, load_documents, config
            )
        
        async def setup_model():
            return await asyncio.get_event_loop().run_in_executor(
                None, setup_llm, config, {}
            )
        
        # Load documents and model in parallel
        logger.info("Loading documents and model in parallel")
        chunks, llm = await asyncio.gather(load_docs(), setup_model())
        
        # Update vectorstore (depends on chunks)
        logger.info("Updating vectorstore")
        retriever = await asyncio.get_event_loop().run_in_executor(
            None, update_vectorstore, config, chunks, False
        )
        
        # Store in config
        config["chunks"] = chunks
        config["llm_instance"] = llm
        config["retriever"] = retriever
        
        logger.info("Async orchestration complete")
        return config
        
    except Exception as e:
        logger.error(f"Async orchestration failed: {e}")
        raise


# Example usage for testing
if __name__ == "__main__":
    from utils import load_config
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Test document loading
    print("Testing document loading...")
    chunks = load_documents(config)
    print(f"Loaded {len(chunks)} chunks")
    
    # Test vectorstore update
    print("\nTesting vectorstore update...")
    retriever = update_vectorstore(config, chunks)
    print("Vectorstore updated")
    
    # Test LLM setup
    print("\nTesting LLM setup...")
    llm = setup_llm(config)
    print(f"LLM loaded: {llm.__class__.__name__}")
    
    # Store in config
    config["llm_instance"] = llm
    config["retriever"] = retriever
    
    # Test session handling
    print("\nTesting session handling...")
    snap, memory = handle_session(config, override="new")
    
    # Test agent creation
    print("\nTesting agent creation...")
    agent = start_session(config, memory)
    print("Agent created successfully")
    
    # Test a query
    print("\nTesting query...")
    answer, sources = agent.ask("What is RAG?")
    print(f"Answer: {answer[:200]}...")
    print(f"Sources: {len(sources)} documents")