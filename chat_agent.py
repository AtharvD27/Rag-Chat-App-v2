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
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: VectorStoreRetriever,
        memory: BaseMemory,
        config: dict
    ):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.config = config
        self.prompts = self._load_prompts(config.get("prompt_path", "./prompts.yaml"))
        self.chain = self._create_chain()

    def _load_prompts(self, prompt_path: str) -> dict:
        with open(prompt_path) as f:
            return yaml.safe_load(f)

    def _create_chain(self) -> Runnable:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts["answer_prompt_system"]),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", self.prompts["answer_prompt_human"]),
        ])

        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=answer_prompt
        )

        question_only = RunnableLambda(lambda x: x["question"])
        rag_chain = RunnableMap({
            "context": question_only | self.retriever,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }) | combine_docs_chain

        return rag_chain

    def ask(self, query: str) -> Tuple[str, List[Dict]]:
        # Run retriever manually for access to source docs
        retrieved_docs = self.retriever.invoke(query)

        result = self.chain.invoke({
            "question": query,
            "chat_history": self.memory.chat_memory.messages,
            "context": retrieved_docs
        })

        answer = result
        sources = self._extract_sources(retrieved_docs)
        return answer, sources

    def _extract_sources(self, docs: List[Document]) -> List[Dict]:
        return [{
            "file": doc.metadata.get("file", "unknown"),
            "page": doc.metadata.get("page", -1),
            "chunk": doc.metadata.get("chunk", -1),
            "text": doc.page_content.strip()
        } for doc in docs]
