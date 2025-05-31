from typing import Tuple, List, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableMap, Runnable
from langchain_core.documents import Document
from langchain_core.memory import BaseMemory
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import yaml


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
