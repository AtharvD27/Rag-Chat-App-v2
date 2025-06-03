from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass
import uuid
from datetime import datetime
import numpy as np
from collections import Counter
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableMap, Runnable
from langchain_core.documents import Document
from langchain_core.memory import BaseMemory
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import BaseRetriever
import yaml

# Import from enhanced utils and monitoring
from utils import (
    setup_logging, load_config, handle_errors, time_operation,
    ConfigurationError, ModelError, performance_monitor
)
from monitoring import get_monitoring_instance, QueryMetrics


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with metadata"""
    documents: List[Document]
    scores: List[float]
    retrieval_method: str
    query_expansion: Optional[List[str]] = None
    total_candidates: int = 0
    filtering_applied: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryAnalysis:
    """Analysis of user query for optimized retrieval"""
    original_query: str
    cleaned_query: str
    query_type: str  # factual, analytical, conversational, etc.
    entities: List[str]
    keywords: List[str]
    temporal_context: Optional[str] = None
    intent: str = "general"
    expansion_terms: List[str] = None
    
    def __post_init__(self):
        if self.expansion_terms is None:
            self.expansion_terms = []


class ChatAgent:
    """Enhanced chat agent with hybrid retrieval and advanced features"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: Union[VectorStoreRetriever, HybridRetriever],
        memory: BaseMemory,
        config: dict
    ):
        self.llm = llm
        self.memory = memory
        self.config = config
        
        # Setup logging
        log_config = self.config.get("logging", {})
        self.logger = setup_logging(
            name="rag_chat.ChatAgent",
            log_level=log_config.get("level", "INFO"),
            log_file=log_config.get("file_path"),
            console_level=log_config.get("console_level")
        )
        
        # Load prompts
        self.prompts = self._load_prompts(config.get("prompt_path", "./prompts.yaml"))
        
        # Setup retriever (upgrade to hybrid if needed)
        if isinstance(retriever, VectorStoreRetriever) and config.get("retrieval", {}).get("search_type") != "semantic":
            self.logger.info("Upgrading to hybrid retriever")
            # Note: In production, we'd need access to the document list
            # For now, we'll use the retriever as-is
            self.retriever = retriever
        else:
            self.retriever = retriever
        
        # Initialize monitoring
        self.metrics = get_monitoring_instance(config)
        
        # Create chain
        self.chain = self._create_chain()
        
        # Cache for response validation
        self.response_cache = {}
        
        self.logger.info("ChatAgent initialized with enhanced retrieval")

    @handle_errors(logger=None, raise_on_error=True)
    def _load_prompts(self, prompt_path: str) -> dict:
        """Load prompts with validation"""
        try:
            with open(prompt_path) as f:
                prompts = yaml.safe_load(f)
            
            # Validate required prompts
            required = ["answer_prompt_system", "answer_prompt_human"]
            for req in required:
                if req not in prompts:
                    raise ConfigurationError(
                        f"Missing required prompt: {req}",
                        error_code="MISSING_PROMPT",
                        details={"prompt_path": prompt_path, "required": required}
                    )
            
            return prompts
            
        except Exception as e:
            self.logger.error(f"Failed to load prompts from {prompt_path}: {e}")
            # Return default prompts
            return {
                "answer_prompt_system": "You are a helpful assistant. Answer based on the context provided.",
                "answer_prompt_human": "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            }

    def _create_chain(self) -> Runnable:
        """Create the enhanced QA chain"""
        # Create answer prompt
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts["answer_prompt_system"]),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", self.prompts["answer_prompt_human"]),
        ])

        # Create document chain
        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=answer_prompt
        )

        # Create retrieval chain with monitoring
        def monitored_retrieval(x):
            query = x["question"]
            self.logger.debug(f"Retrieving documents for: {query}")
            
            start_time = time.time()
            docs = self.retriever.get_relevant_documents(query)
            retrieval_time = time.time() - start_time
            
            # Record retrieval metrics
            self.metrics.record_metric("retrieval_time", retrieval_time)
            self.metrics.record_metric("chunks_retrieved", len(docs))
            
            return docs

        # Build the chain
        question_only = RunnableLambda(lambda x: x["question"])
        rag_chain = RunnableMap({
            "context": question_only | RunnableLambda(monitored_retrieval),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }) | combine_docs_chain

        return rag_chain

    @time_operation("chat_agent_ask")
    def ask(self, query: str) -> Tuple[str, List[Dict]]:
        """Process a query with enhanced retrieval and monitoring"""
        query_id = str(uuid.uuid4())
        
        # Start monitoring
        query_metrics = self.metrics.start_query(
            query_id, 
            query, 
            model=self.llm.__class__.__name__
        )
        
        try:
            self.logger.info(f"Processing query: {query[:100]}...")
            
            # Retrieve documents with timing
            start_time = time.time()
            retrieved_docs = self.retriever.get_relevant_documents(query)
            retrieval_time = time.time() - start_time
            
            self.logger.debug(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s")
            
            # Generate answer with timing
            start_time = time.time()
            result = self.chain.invoke({
                "question": query,
                "chat_history": self.memory.chat_memory.messages,
                "context": retrieved_docs
            })
            generation_time = time.time() - start_time
            
            # Extract answer
            answer = result if isinstance(result, str) else str(result)
            
            # Validate response
            if not self._validate_response(answer, query):
                self.logger.warning("Response validation failed, using fallback")
                answer = self._generate_fallback_response(query)
            
            # Extract and enhance sources
            sources = self._extract_sources(retrieved_docs)
            
            # Update metrics
            self.metrics.end_query(
                query_id,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                chunks_retrieved=len(retrieved_docs),
                tokens_used=self._estimate_tokens(query + answer)
            )
            
            # Add to conversation memory
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(answer)
            
            self.logger.info(f"Query processed successfully in {retrieval_time + generation_time:.2f}s")
            
            return answer, sources
            
        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            self.metrics.end_query(query_id, error=str(e))
            
            # Return a helpful error message
            error_response = (
                "I apologize, but I encountered an error while processing your question. "
                "Please try rephrasing your question or ask something else."
            )
            
            return error_response, []

    def _validate_response(self, response: str, query: str) -> bool:
        """Validate that the response is appropriate"""
        # Basic validation rules
        if not response or len(response.strip()) < 10:
            return False
        
        # Check if response is just a repetition of the question
        if response.strip().lower() == query.strip().lower():
            return False
        
        # Check for common LLM failure patterns
        failure_patterns = [
            "i don't have access to",
            "i cannot access",
            "no information available",
            "error:",
            "exception:"
        ]
        
        response_lower = response.lower()
        if any(pattern in response_lower for pattern in failure_patterns):
            # Unless the question was actually about errors/exceptions
            if not any(word in query.lower() for word in ["error", "exception", "access"]):
                return False
        
        return True

    def _generate_fallback_response(self, query: str) -> str:
        """Generate a fallback response when validation fails"""
        return (
            "I understand you're asking about: " + query + ". "
            "While I don't have specific information about that in my current context, "
            "I'd be happy to help if you could provide more details or ask a related question."
        )

    def _extract_sources(self, docs: List[Document]) -> List[Dict]:
        """Extract and enhance source information"""
        sources = []
        
        for idx, doc in enumerate(docs):
            source_info = {
                "file": doc.metadata.get("file", "unknown"),
                "page": doc.metadata.get("page", -1),
                "chunk": doc.metadata.get("chunk", -1),
                "text": doc.page_content.strip()[:200] + "...",  # First 200 chars
                "relevance_score": doc.metadata.get("score", 0.0),
                "retrieval_rank": idx + 1
            }
            
            # Add additional metadata if available
            for key in ["type", "date", "author", "section"]:
                if key in doc.metadata:
                    source_info[key] = doc.metadata[key]
            
            sources.append(source_info)
        
        return sources

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the current conversation"""
        messages = self.memory.chat_memory.messages
        
        return {
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if m.type == "human"]),
            "ai_messages": len([m for m in messages if m.type == "ai"]),
            "conversation_length": sum(len(m.content) for m in messages)
        }

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.chat_memory.clear()
        self.logger.info("Conversation memory cleared")
