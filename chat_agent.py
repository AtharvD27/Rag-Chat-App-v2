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


class HybridRetriever(BaseRetriever):
    """Custom hybrid retriever combining semantic and keyword search"""
    
    def __init__(
        self,
        vectorstore_retriever: VectorStoreRetriever,
        documents: List[Document],
        config: Dict[str, Any],
        logger=None
    ):
        self.vectorstore_retriever = vectorstore_retriever
        self.documents = documents
        self.config = config
        
        # Retrieval configuration
        retrieval_config = config.get("retrieval", {})
        self.search_type = retrieval_config.get("search_type", "hybrid")
        self.semantic_weight = retrieval_config.get("semantic_weight", 0.7)
        self.keyword_weight = retrieval_config.get("keyword_weight", 0.3)
        self.top_k = retrieval_config.get("top_k", 5)
        self.rerank = retrieval_config.get("rerank", True)
        self.diversity_threshold = retrieval_config.get("diversity_threshold", 0.7)
        
        # Setup logging
        self.logger = logger or setup_logging("rag_chat.HybridRetriever")
        
        # Initialize keyword retriever (BM25)
        if self.search_type in ["hybrid", "keyword"]:
            self.keyword_retriever = BM25Retriever.from_documents(
                documents, 
                k=self.top_k * 2  # Get more candidates for reranking
            )
            self.logger.info("Initialized BM25 keyword retriever")
        
        # Cache for query analysis
        self.query_cache = {}
        
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Main retrieval method"""
        return self.retrieve(query)
    
    @time_operation("hybrid_retrieval")
    def retrieve(self, query: str) -> List[Document]:
        """Perform hybrid retrieval with advanced features"""
        try:
            # Analyze query
            query_analysis = self._analyze_query(query)
            self.logger.debug(f"Query analysis: {query_analysis}")
            
            # Get candidates from different retrievers
            if self.search_type == "semantic":
                results = self._semantic_search(query_analysis)
            elif self.search_type == "keyword":
                results = self._keyword_search(query_analysis)
            else:  # hybrid
                results = self._hybrid_search(query_analysis)
            
            # Apply reranking if enabled
            if self.rerank and len(results.documents) > 0:
                results = self._rerank_results(results, query_analysis)
            
            # Apply diversity filtering
            if self.diversity_threshold < 1.0:
                results = self._apply_diversity_filter(results)
            
            # Log retrieval metrics
            self._log_retrieval_metrics(query, results)
            
            return results.documents[:self.top_k]
            
        except Exception as e:
            self.logger.error(f"Retrieval failed for query '{query}': {e}")
            # Fallback to simple semantic search
            return self.vectorstore_retriever.get_relevant_documents(query)
    
    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to optimize retrieval strategy"""
        # Check cache
        if query in self.query_cache:
            return self.query_cache[query]
        
        # Clean query
        cleaned = query.strip().lower()
        
        # Extract entities (simple regex-based for now)
        entities = []
        # Look for capitalized words (potential entities)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(entity_pattern, query)
        
        # Extract keywords (important terms)
        # Remove common words
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with", "to", "for"}
        words = [w for w in cleaned.split() if w not in stop_words and len(w) > 2]
        keywords = words[:5]  # Top 5 keywords
        
        # Detect query type
        query_type = self._classify_query_type(cleaned)
        
        # Detect temporal context
        temporal_context = None
        temporal_patterns = [
            (r'\b(today|yesterday|tomorrow)\b', "relative"),
            (r'\b(this|last|next)\s+(week|month|year)\b', "relative"),
            (r'\b\d{4}\b', "absolute"),
            (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', "absolute")
        ]
        
        for pattern, context_type in temporal_patterns:
            if re.search(pattern, cleaned):
                temporal_context = context_type
                break
        
        # Generate expansion terms
        expansion_terms = self._generate_expansion_terms(keywords)
        
        analysis = QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned,
            query_type=query_type,
            entities=entities,
            keywords=keywords,
            temporal_context=temporal_context,
            expansion_terms=expansion_terms
        )
        
        # Cache the analysis
        self.query_cache[query] = analysis
        
        return analysis
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        # Factual questions
        if any(word in query_lower for word in ["what", "when", "where", "who", "which", "how many", "how much"]):
            return "factual"
        
        # Analytical questions
        if any(word in query_lower for word in ["why", "how", "analyze", "explain", "compare", "evaluate"]):
            return "analytical"
        
        # Definition questions
        if any(phrase in query_lower for phrase in ["what is", "what are", "define", "meaning of"]):
            return "definition"
        
        # Instructional questions
        if any(word in query_lower for word in ["how to", "steps", "guide", "tutorial", "instruction"]):
            return "instructional"
        
        return "general"
    
    def _generate_expansion_terms(self, keywords: List[str]) -> List[str]:
        """Generate query expansion terms"""
        expansions = []
        
        # Simple synonym expansion (in production, use WordNet or similar)
        synonym_map = {
            "create": ["make", "build", "construct", "develop"],
            "delete": ["remove", "erase", "clear", "destroy"],
            "update": ["modify", "change", "edit", "revise"],
            "error": ["bug", "issue", "problem", "fault"],
            "fast": ["quick", "rapid", "speed", "performance"],
        }
        
        for keyword in keywords:
            if keyword in synonym_map:
                expansions.extend(synonym_map[keyword][:2])  # Add top 2 synonyms
        
        return expansions
    
    def _semantic_search(self, query_analysis: QueryAnalysis) -> RetrievalResult:
        """Perform semantic search"""
        # Use expanded query if available
        search_query = query_analysis.original_query
        if query_analysis.expansion_terms:
            search_query = f"{search_query} {' '.join(query_analysis.expansion_terms[:2])}"
        
        docs = self.vectorstore_retriever.get_relevant_documents(search_query)
        
        # Extract scores if available
        scores = [doc.metadata.get("score", 1.0) for doc in docs]
        
        return RetrievalResult(
            documents=docs,
            scores=scores,
            retrieval_method="semantic",
            query_expansion=query_analysis.expansion_terms,
            total_candidates=len(docs)
        )
    
    def _keyword_search(self, query_analysis: QueryAnalysis) -> RetrievalResult:
        """Perform keyword search"""
        # Use keywords and entities for BM25 search
        keyword_query = " ".join(query_analysis.keywords + query_analysis.entities)
        
        docs = self.keyword_retriever.get_relevant_documents(keyword_query)
        
        # BM25 doesn't provide scores directly, so we'll assign based on rank
        scores = [1.0 / (i + 1) for i in range(len(docs))]
        
        return RetrievalResult(
            documents=docs,
            scores=scores,
            retrieval_method="keyword",
            total_candidates=len(docs)
        )
    
    def _hybrid_search(self, query_analysis: QueryAnalysis) -> RetrievalResult:
        """Perform hybrid search combining semantic and keyword results"""
        # Get results from both retrievers
        semantic_results = self._semantic_search(query_analysis)
        keyword_results = self._keyword_search(query_analysis)
        
        # Combine results using reciprocal rank fusion
        combined_docs, combined_scores = self._reciprocal_rank_fusion(
            [(semantic_results.documents, semantic_results.scores, self.semantic_weight),
             (keyword_results.documents, keyword_results.scores, self.keyword_weight)]
        )
        
        return RetrievalResult(
            documents=combined_docs,
            scores=combined_scores,
            retrieval_method="hybrid",
            query_expansion=query_analysis.expansion_terms,
            total_candidates=len(semantic_results.documents) + len(keyword_results.documents),
            metadata={
                "semantic_count": len(semantic_results.documents),
                "keyword_count": len(keyword_results.documents)
            }
        )
    
    def _reciprocal_rank_fusion(
        self, 
        result_sets: List[Tuple[List[Document], List[float], float]]
    ) -> Tuple[List[Document], List[float]]:
        """Combine multiple result sets using reciprocal rank fusion"""
        doc_scores = {}
        doc_map = {}
        
        k = 60  # Constant for RRF
        
        for docs, scores, weight in result_sets:
            for rank, (doc, score) in enumerate(zip(docs, scores)):
                doc_id = doc.metadata.get("id", str(hash(doc.page_content)))
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_map[doc_id] = doc
                
                # RRF score
                rrf_score = weight * (1.0 / (k + rank + 1))
                doc_scores[doc_id] += rrf_score
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        combined_docs = [doc_map[doc_id] for doc_id, _ in sorted_docs]
        combined_scores = [score for _, score in sorted_docs]
        
        return combined_docs, combined_scores
    
    def _rerank_results(
        self, 
        results: RetrievalResult, 
        query_analysis: QueryAnalysis
    ) -> RetrievalResult:
        """Rerank results based on query analysis and relevance signals"""
        reranked_docs = []
        reranked_scores = []
        
        for doc, base_score in zip(results.documents, results.scores):
            # Calculate relevance boost factors
            boost = 1.0
            
            # Boost if entities are found in the document
            if query_analysis.entities:
                entity_matches = sum(
                    1 for entity in query_analysis.entities 
                    if entity.lower() in doc.page_content.lower()
                )
                boost += 0.1 * entity_matches
            
            # Boost if keywords are found in the document
            keyword_density = sum(
                doc.page_content.lower().count(keyword) 
                for keyword in query_analysis.keywords
            ) / max(len(doc.page_content.split()), 1)
            boost += min(keyword_density * 10, 0.3)
            
            # Boost based on metadata relevance
            if query_analysis.query_type == "factual" and doc.metadata.get("type") == "reference":
                boost += 0.2
            elif query_analysis.query_type == "instructional" and doc.metadata.get("type") == "tutorial":
                boost += 0.2
            
            # Apply temporal relevance if applicable
            if query_analysis.temporal_context and doc.metadata.get("date"):
                # Simple recency boost (in production, use proper date parsing)
                boost += 0.1
            
            final_score = base_score * boost
            reranked_docs.append(doc)
            reranked_scores.append(final_score)
        
        # Sort by new scores
        sorted_pairs = sorted(
            zip(reranked_docs, reranked_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        results.documents = [doc for doc, _ in sorted_pairs]
        results.scores = [score for _, score in sorted_pairs]
        results.metadata["reranked"] = True
        
        return results
    
    def _apply_diversity_filter(self, results: RetrievalResult) -> RetrievalResult:
        """Apply diversity filtering to reduce redundancy"""
        if len(results.documents) <= 1:
            return results
        
        filtered_docs = [results.documents[0]]
        filtered_scores = [results.scores[0]]
        
        for doc, score in zip(results.documents[1:], results.scores[1:]):
            # Check similarity with already selected documents
            is_diverse = True
            
            for selected_doc in filtered_docs:
                similarity = self._calculate_similarity(doc.page_content, selected_doc.page_content)
                if similarity > self.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        results.documents = filtered_docs
        results.scores = filtered_scores
        results.filtering_applied = True
        
        return results
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard)"""
        # In production, use better similarity metrics
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / max(len(union), 1)
    
    def _log_retrieval_metrics(self, query: str, results: RetrievalResult):
        """Log detailed retrieval metrics"""
        self.logger.info(
            f"Retrieval complete: "
            f"method={results.retrieval_method}, "
            f"candidates={results.total_candidates}, "
            f"results={len(results.documents)}, "
            f"filtered={results.filtering_applied}, "
            f"reranked={results.metadata.get('reranked', False)}"
        )
        

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
