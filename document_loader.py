import os
import json
import yaml
import asyncio
import aiofiles
import aiohttp
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timezone

from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import (
    setup_logging, compute_sha1, SecurityValidator, DocumentProcessingError,
    handle_errors, retry_with_backoff, time_operation, get_file_info,
    batch_iterator, performance_monitor, ensure_directory
)

@dataclass
class LoaderStats:
    """Statistics for document loading operations"""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_size_mb: float = 0.0
    processing_time: float = 0.0
    errors: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def success_rate(self) -> float:
        return (self.successful_files / self.total_files * 100) if self.total_files > 0 else 0


class BaseDocumentLoader(ABC):
    """Base class for all document loaders with async support and batch processing"""

    def __init__(self, config_path: str = "config.yaml", config: dict = None):
        self.config = config or self._load_config(config_path)

        # Chunking configuration
        self.chunk_size = self.config.get("chunk", {}).get("size", 800)
        self.chunk_overlap = self.config.get("chunk", {}).get("overlap", 80)

        # Performance configuration
        perf_config = self.config.get("performance", {})
        self.batch_size = perf_config.get("batch_size", 100)
        self.max_concurrent = perf_config.get("max_concurrent_operations", 4)
        self.enable_async = perf_config.get("enable_async", True)

        # Setup logging
        log_config = self.config.get("logging", {})
        self.logger = setup_logging(
            name=f"rag_chat.{self.__class__.__name__}",
            log_level=log_config.get("level", "INFO"),
            log_file=log_config.get("file_path"),
            console_level=log_config.get("console_level")
        )

        # Security validator
        self.security_validator = self.config.get("_security_validator")
        if not self.security_validator:
            self.security_validator = SecurityValidator(self.config.get("security", {}))
        
        # Statistics tracking
        self.stats = LoaderStats()
        
        self.logger.info(
            f"Initialized {self.__class__.__name__} with batch_size={self.batch_size}, "
            f"max_concurrent={self.max_concurrent}, async={self.enable_async}"
        )

    def _load_config(self, path: str) -> dict:
            """Load configuration file with error handling"""
            try:
                with open(path) as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger = setup_logging("document_loader")
                self.logger.warning(f"Failed to load config from {path}: {e}. Using defaults.")
                return {
                    "chunk": {"size": 800, "overlap": 80},
                    "logging": {"level": "INFO"},
                    "performance": {"batch_size": 100, "max_concurrent_operations": 4}
                }

    @abstractmethod
    async def load_async(self) -> List[Document]:
        """Async document loading - must be implemented by subclasses"""
        pass

    @abstractmethod
    def load(self) -> List[Document]:
        """Sync document loading - must be implemented by subclasses"""
        pass

    @time_operation("document_splitting")
    async def split_documents_async(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks asynchronously with batch processing"""
        self.logger.info(f"Splitting {len(documents)} documents into chunks")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        all_chunks = []
        
        # Process in batches to manage memory
        async for batch_chunks in self._process_batches_async(documents, splitter):
            all_chunks.extend(batch_chunks)
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    async def _process_batches_async(
        self, 
        documents: List[Document], 
        splitter: RecursiveCharacterTextSplitter
    ) -> AsyncIterator[List[Document]]:
        
        """Process documents in batches asynchronously"""
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            for batch in batch_iterator(documents, self.batch_size):
                # Process batch in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                batch_chunks = await loop.run_in_executor(
                    executor,
                    self._process_batch_sync,
                    batch,
                    splitter
                )
                yield batch_chunks

    def _process_batch_sync(
        self, 
        batch: List[Document], 
        splitter: RecursiveCharacterTextSplitter
    ) -> List[Document]:
        """Process a batch of documents synchronously"""
        chunks = splitter.split_documents(batch)
        return self.assign_chunk_ids(chunks)

    @staticmethod
    def assign_chunk_ids(chunks: List[Document]) -> List[Document]:
        """Assign unique IDs to chunks with error handling"""

        for idx, doc in enumerate(chunks):
            try:
                file = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", -1)
                chunk_hash = compute_sha1(f"{file}:{page}:{doc.page_content}")[:20]
                
                doc.metadata.update({
                    "file": file,
                    "page": page,
                    "chunk": chunk_hash,
                    "id": f"{file}:{page}:{chunk_hash}",
                    "chunk_index": idx,
                    "processed_at": datetime.now(timezone.utc).isoformat()
                })
            except Exception:
                # Fallback ID
                doc.metadata["id"] = f"unknown:{idx}"
                doc.metadata["chunk_index"] = idx
        
        return chunks


class PDFLoader(BaseDocumentLoader):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def load(self) -> List[Document]:
        docs = []
        for file in os.listdir(self.path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.path, file))
                docs.extend(loader.load())
        return docs


class JSONLoader(BaseDocumentLoader):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def load(self) -> List[Document]:
        docs = []
        for file in os.listdir(self.path):
            if file.endswith(".json"):
                with open(os.path.join(self.path, file), "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for entry in data:
                            content = entry.get("text") or json.dumps(entry)
                            docs.append(Document(page_content=content, metadata={"source": file}))
                    elif isinstance(data, dict):
                        content = data.get("text") or json.dumps(data)
                        docs.append(Document(page_content=content, metadata={"source": file}))
        return docs


class WebPageLoader(BaseDocumentLoader):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def load(self) -> List[Document]:
        docs = []
        for file in os.listdir(self.path):
            full_path = os.path.join(self.path, file)
            if file.endswith(".txt"):  # each line is a URL
                with open(full_path, "r") as f:
                    for url in f.readlines():
                        url = url.strip()
                        try:
                            html = requests.get(url, timeout=10).text
                            text = self.extract_text(html)
                            docs.append(Document(page_content=text, metadata={"source": url}))
                        except Exception as e:
                            print(f"⚠️ Failed to load {url}: {e}")
            elif file.endswith(".html"):
                with open(full_path, "r", encoding="utf-8") as f:
                    html = f.read()
                    text = self.extract_text(html)
                    docs.append(Document(page_content=text, metadata={"source": file}))
        return docs

    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)


class SmartDocumentLoader(BaseDocumentLoader):
    def __init__(self, config_path: str = "config.yaml", config: dict = None):
        super().__init__(config_path=config_path, config=config)
        self.path = self.config.get("data_path", "./data")

    def load(self) -> List[Document]:
        docs = []
        file_types = os.listdir(self.path)
        print(f"📄 Loaded {len(file_types)} documents from {self.path}")

        if any(f.endswith(".pdf") for f in file_types):
            docs.extend(PDFLoader(self.path, config=self.config).load())
        if any(f.endswith(".json") for f in file_types):
            docs.extend(JSONLoader(self.path, config=self.config).load())
        if any(f.endswith((".txt", ".html")) for f in file_types):
            docs.extend(WebPageLoader(self.path, config=self.config).load())

        return docs

