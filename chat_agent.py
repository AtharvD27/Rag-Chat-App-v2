from typing import Tuple, List, Dict
import uuid
import time

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationSummaryBufferMemory

# Import from enhanced utils and monitoring
from utils import (
    setup_logging, load_config, handle_errors, time_operation,
    ConfigurationError, ModelError, performance_monitor
)
from monitoring import get_monitoring_instance, QueryMetrics

class ChatAgent:
    """
    Enhanced chat agent with history-aware hybrid retrieval and summary memory.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: BaseRetriever,
        config: dict
    ):
        self.llm = llm
        self.retriever = retriever
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

        memory_config = self.config.get("memory", {})
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=memory_config.get("summary_token_limit", 500),
            memory_key="chat_history",
            output_key='answer', # Required for this memory type
            return_messages=True
        )
        
        # Initialize monitoring
        self.metrics = get_monitoring_instance(config)
        
        # --- NEW: Create the enhanced, history-aware chain ---
        self.chain = self._create_chain()
        
        self.logger.info("ChatAgent initialized with history-aware retrieval and summary memory.")

    @handle_errors(logger=None, raise_on_error=True)
    def _load_prompts(self, prompt_path: str) -> dict:
        """Load prompts with validation. Now includes a rephrase prompt."""
        try:
            prompts = load_config(prompt_path) # Assumes load_config can handle yaml
            
            # Validate required prompts
            required = [
                "rephrase_prompt_system",
                "answer_prompt_system",
                "answer_prompt_human"
            ]
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
            # Return default prompts as a fallback
            return {
                "rephrase_prompt_system": "Given a chat history and a follow up question, rephrase the follow up question to be a standalone question.",
                "answer_prompt_system": "You are a helpful assistant. Answer based on the context provided.",
                "answer_prompt_human": "Context: {context}\n\nQuestion: {input}\n\nAnswer:"
            }

    def _create_chain(self) -> Runnable:
        """
        Creates an enhanced retrieval chain that is history-aware.
        
        This chain consists of three main parts:
        1. A history-aware retriever that rephrases the input question based on chat history.
        2. A document stuffing chain that combines retrieved documents into a context.
        3. A final retrieval chain that links the two.
        """
        # 1. Create a chain to rephrase the question
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts["rephrase_prompt_system"]),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.retriever,
            prompt=rephrase_prompt
        )

        # 2. Create the main answering chain
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts["answer_prompt_system"]),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", self.prompts["answer_prompt_human"]),
        ])

        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=answer_prompt
        )
        
        # 3. Combine them into the final retrieval chain
        retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain
        )
        
        self.logger.info("Created history-aware retrieval chain.")
        return retrieval_chain

    @time_operation("chat_agent_ask")
    def ask(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Process a query using the history-aware retrieval chain and summary memory.
        """
        query_id = str(uuid.uuid4())
        self.metrics.start_query(query_id, query, model=self.llm.__class__.__name__)
        
        try:
            self.logger.info(f"Processing query: {query[:100]}...")
            
            # Load chat history from memory
            chat_history = self.memory.chat_memory.messages
            
            start_time = time.time()
            result = self.chain.invoke({
                "input": query,
                "chat_history": chat_history,
            })
            total_time = time.time() - start_time
            
            # Extract answer and context (retrieved documents)
            answer = result.get("answer", "")
            retrieved_docs = result.get("context", [])
            
            self.logger.debug(f"Retrieved {len(retrieved_docs)} documents in {total_time:.2f}s")
            
            # Validate response
            if not self._validate_response(answer, query):
                self.logger.warning("Response validation failed, using fallback")
                answer = self._generate_fallback_response(query)
            
            # Extract and enhance sources
            sources = self._extract_sources(retrieved_docs)
            
            self.memory.save_context({"input": query}, {"answer": answer})
            
            # Update metrics
            self.metrics.end_query(
                query_id,
                generation_time=total_time, # Simplified timing
                chunks_retrieved=len(retrieved_docs),
                tokens_used=self._estimate_tokens(query + answer)
            )
            
            self.logger.info(f"Query processed successfully in {total_time:.2f}s")
            
            return answer, sources
            
        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            self.metrics.end_query(query_id, error=str(e))
            
            error_response = (
                "I apologize, but I encountered an error while processing your question. "
                "Please try rephrasing your question or ask something else."
            )
            
            return error_response, []

    def _validate_response(self, response: str, query: str) -> bool:
        """Validate that the response is appropriate"""
        if not response or len(response.strip()) < 10:
            return False
        
        if response.strip().lower() == query.strip().lower():
            return False
        
        failure_patterns = [
            "i don't have access to", "i cannot access", "no information available",
            "error:", "exception:", "i am not able to", "i am a large language model"
        ]
        
        response_lower = response.lower()
        if any(pattern in response_lower for pattern in failure_patterns):
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
        """Extract and enhance source information from retrieved documents."""
        sources = []
        if not docs:
            return sources
            
        for idx, doc in enumerate(docs):
            source_info = {
                "file": doc.metadata.get("file", "unknown"),
                "page": doc.metadata.get("page", -1),
                "chunk": doc.metadata.get("chunk", -1),
                "text": doc.page_content.strip()[:200] + "...",
                "relevance_score": doc.metadata.get("score", 0.0), # May not be present in all retrievers
                "retrieval_rank": idx + 1
            }
            sources.append(source_info)
        
        return sources

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text) // 4

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        self.logger.info("Conversation memory cleared")