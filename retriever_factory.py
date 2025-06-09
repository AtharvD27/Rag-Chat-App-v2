from typing import List, Dict, Any, Optional
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain.vectorstores.base import VectorStore
from langchain.schema import BaseRetriever

# Import from enhanced utils
from utils import setup_logging, ConfigurationError

class RetrieverFactory:
    """
    A factory class for creating various types of retrievers.

    This class centralizes the logic for instantiating different retriever
    implementations based on the application's configuration. It supports
    semantic, keyword (BM25), and hybrid search retrievers.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        documents: List[Document],
        config: Dict[str, Any],
        logger=None
    ):
        """
        Initializes the RetrieverFactory.

        Args:
            vectorstore (VectorStore): The initialized vector store for semantic search.
            documents (List[Document]): The list of all document chunks, required for BM25.
            config (Dict[str, Any]): The application configuration dictionary.
            logger: An optional logger instance.
        """
        self.vectorstore = vectorstore
        self.documents = documents
        self.config = config.get("retrieval", {})
        self.logger = logger or setup_logging("rag_chat.RetrieverFactory")
        
        # Validate inputs
        if not self.vectorstore:
            raise ConfigurationError("Vectorstore must be provided.", "MISSING_VECTORSTORE")
        if not self.documents:
            self.logger.warning("Document list is empty. BM25 and hybrid retrievers will not be effective.")

    def create_retriever(self) -> BaseRetriever:
        """
        Creates a retriever based on the 'search_type' in the configuration.

        This is the main public method that acts as a dispatcher to the
        specific retriever creation methods.

        Returns:
            BaseRetriever: An initialized retriever instance.
        
        Raises:
            ConfigurationError: If the specified search_type is invalid.
        """
        search_type = self.config.get("search_type", "hybrid")
        self.logger.info(f"Creating retriever of type: {search_type}")

        if search_type == "hybrid":
            return self._create_hybrid_retriever()
        elif search_type == "semantic":
            return self._create_semantic_retriever()
        elif search_type == "keyword":
            return self._create_bm25_retriever()
        else:
            raise ConfigurationError(
                f"Invalid search_type '{search_type}'. Must be 'hybrid', 'semantic', or 'keyword'.",
                error_code="INVALID_SEARCH_TYPE"
            )

    def _create_semantic_retriever(self) -> BaseRetriever:
        """
        Creates a retriever for dense, semantic search using the vector store.
        """
        top_k = self.config.get("top_k", 5)
        self.logger.info(f"Initializing semantic retriever with k={top_k}")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": top_k})

    def _create_bm25_retriever(self) -> BM25Retriever:
        """
        Creates a retriever for sparse, keyword-based search using BM25.
        """
        top_k = self.config.get("top_k", 5)
        self.logger.info(f"Initializing BM25 (keyword) retriever with k={top_k}")

        if not self.documents:
            self.logger.error("Cannot create BM25 retriever without documents. Fallback to semantic search.")
            # Fallback to semantic search to prevent crashing
            return self._create_semantic_retriever()

        return BM25Retriever.from_documents(
            self.documents,
            k=top_k
        )

    def _create_hybrid_retriever(self) -> EnsembleRetriever:
        """
        Creates a hybrid retriever that combines semantic and keyword search results.

        This uses LangChain's EnsembleRetriever to combine results from a dense
        retriever (vector search) and a sparse retriever (BM25) using a weighted
        reciprocal rank fusion algorithm.
        """
        self.logger.info("Initializing hybrid retriever")
        
        # Get individual retrievers
        semantic_retriever = self._create_semantic_retriever()
        bm25_retriever = self._create_bm25_retriever()

        # Get weights from config with defaults
        semantic_weight = self.config.get("semantic_weight", 0.7)
        keyword_weight = self.config.get("keyword_weight", 0.3)
        
        if (semantic_weight + keyword_weight) != 1.0:
            self.logger.warning("Hybrid retriever weights do not sum to 1.0. This may cause unexpected ranking.")

        self.logger.info(f"Hybrid weights: semantic={semantic_weight}, keyword={keyword_weight}")

        return EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[semantic_weight, keyword_weight]
        )

# Example usage for testing purposes
if __name__ == "__main__":
    from langchain.vectorstores.faiss import FAISS
    from langchain_community.embeddings.fake import FakeEmbeddings

    # 1. Setup mock components
    print("--- Setting up mock components ---")
    mock_docs = [
        Document(page_content="RAG stands for Retrieval-Augmented Generation."),
        Document(page_content="LangChain provides tools to build LLM applications."),
        Document(page_content="Hybrid search combines keyword and semantic methods."),
        Document(page_content="BM25 is a popular keyword-based retrieval algorithm.")
    ]
    
    mock_embeddings = FakeEmbeddings(size=128)
    mock_vectorstore = FAISS.from_documents(mock_docs, mock_embeddings)
    
    # 2. Test different retriever creations
    configs = {
        "hybrid": {"retrieval": {"search_type": "hybrid", "semantic_weight": 0.6, "keyword_weight": 0.4}},
        "semantic": {"retrieval": {"search_type": "semantic", "top_k": 3}},
        "keyword": {"retrieval": {"search_type": "keyword", "top_k": 2}},
    }

    for name, config in configs.items():
        print(f"\n--- Testing '{name}' retriever creation ---")
        factory = RetrieverFactory(
            vectorstore=mock_vectorstore,
            documents=mock_docs,
            config=config
        )
        retriever = factory.create_retriever()
        print(f"Successfully created retriever: {retriever.__class__.__name__}")
        
        if isinstance(retriever, EnsembleRetriever):
            print(f"  Ensemble weights: {retriever.weights}")

        # Test a query
        results = retriever.get_relevant_documents("What is hybrid search?")
        print(f"  Retrieved {len(results)} documents for a test query.")
        print(f"  Top result: '{results[0].page_content}'")