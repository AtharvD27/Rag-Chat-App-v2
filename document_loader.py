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
    """PDF document loader with async support and memory optimization"""
    
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    async def load_async(self) -> List[Document]:
        """Load PDFs asynchronously with progress tracking"""
        self.logger.info(f"Starting async PDF loading from {self.path}")
        performance_monitor.start_timer("pdf_loading")
        
        try:
            # Validate directory
            self.security_validator.validate_file_path(self.path)
        except Exception as e:
            self.logger.error(f"Invalid path {self.path}: {e}")
            raise DocumentProcessingError(
                f"Invalid document path: {self.path}",
                error_code="INVALID_PATH",
                details={"path": self.path, "error": str(e)}
            )
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(self.path) if f.endswith(".pdf")]
        self.stats.total_files = len(pdf_files)
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {self.path}")
            return []
        
        # Process PDFs concurrently
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_pdf(file: str) -> Optional[List[Document]]:
            async with semaphore:
                return await self._load_single_pdf_async(os.path.join(self.path, file))
        
        # Process with progress bar
        tasks = [process_pdf(file) for file in pdf_files]
        results = []
        
        with tqdm(total=len(pdf_files), desc="Loading PDFs") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    results.extend(result)
                pbar.update(1)
        
        # Flatten results
        all_docs = []
        for docs in results:
            if docs:
                all_docs.extend(docs)
        
        # Update statistics
        self.stats.processing_time = performance_monitor.end_timer("pdf_loading")
        self.stats.successful_files = len([r for r in results if r])
        self.stats.failed_files = self.stats.total_files - self.stats.successful_files
        
        self._log_statistics()
        return all_docs

    async def _load_single_pdf_async(self, file_path: str) -> Optional[List[Document]]:
        """Load a single PDF file asynchronously"""
        try:
            # Validate file
            self.security_validator.validate_file_path(file_path)
            file_info = get_file_info(file_path)
            
            self.logger.debug(f"Processing PDF: {file_path} ({file_info['size_mb']:.2f}MB)")
            
            # Run PDF loading in thread pool (PyPDFLoader is not async)
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                docs = await loop.run_in_executor(
                    executor,
                    self._load_pdf_sync,
                    file_path
                )
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    "loader": "PDFLoader",
                    "file_size_mb": file_info['size_mb'],
                    "load_timestamp": datetime.utcnow().isoformat()
                })
            
            self.stats.total_size_mb += file_info['size_mb']
            return docs
            
        except Exception as e:
            self.logger.error(f"Failed to load PDF {file_path}: {e}")
            self.stats.errors.append({
                "file": file_path,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return None

    def _load_pdf_sync(self, file_path: str) -> List[Document]:
        """Synchronous PDF loading"""
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load(self) -> List[Document]:
        """Synchronous loading interface"""
        if self.enable_async:
            return asyncio.run(self.load_async())
        else:
            return self._load_sync()

    def _load_sync(self) -> List[Document]:
        """Traditional synchronous loading"""
        self.logger.info(f"Loading PDFs from {self.path}")
        docs = []
        
        try:
            self.security_validator.validate_file_path(self.path)
        except Exception as e:
            self.logger.error(f"Invalid path {self.path}: {e}")
            raise
        
        pdf_files = [f for f in os.listdir(self.path) if f.endswith(".pdf")]
        
        for file in tqdm(pdf_files, desc="Loading PDFs"):
            file_path = os.path.join(self.path, file)
            try:
                result = self._load_single_pdf_async(file_path)
                if asyncio.iscoroutine(result):
                    result = asyncio.run(result)
                if result:
                    docs.extend(result)
            except Exception as e:
                self.logger.error(f"Failed to load {file}: {e}")
        
        return docs

    def _log_statistics(self):
        """Log loading statistics"""
        self.logger.info(
            f"PDF Loading Complete:\n"
            f"  - Total files: {self.stats.total_files}\n"
            f"  - Successful: {self.stats.successful_files}\n"
            f"  - Failed: {self.stats.failed_files}\n"
            f"  - Total size: {self.stats.total_size_mb:.2f}MB\n"
            f"  - Processing time: {self.stats.processing_time:.2f}s\n"
            f"  - Success rate: {self.stats.success_rate:.1f}%"
        )


class JSONLoader(BaseDocumentLoader):
    """JSON document loader with streaming support for large files"""
    
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    async def load_async(self) -> List[Document]:
        """Load JSON files asynchronously with streaming for large files"""
        self.logger.info(f"Starting async JSON loading from {self.path}")
        
        json_files = [f for f in os.listdir(self.path) if f.endswith(".json")]
        if not json_files:
            self.logger.warning(f"No JSON files found in {self.path}")
            return []
        
        all_docs = []
        
        async def process_json(file: str) -> List[Document]:
            file_path = os.path.join(self.path, file)
            try:
                # Check file size
                file_info = get_file_info(file_path)
                
                if file_info['size_mb'] > 10:  # Stream large files
                    return await self._load_large_json_async(file_path)
                else:
                    return await self._load_json_async(file_path)
                    
            except Exception as e:
                self.logger.error(f"Failed to load JSON {file}: {e}")
                return []
        
        # Process concurrently
        tasks = [process_json(file) for file in json_files]
        results = await asyncio.gather(*tasks)
        
        for docs in results:
            all_docs.extend(docs)
        
        self.logger.info(f"Loaded {len(all_docs)} documents from {len(json_files)} JSON files")
        return all_docs

    async def _load_json_async(self, file_path: str) -> List[Document]:
        """Load a regular JSON file asynchronously"""
        self.security_validator.validate_file_path(file_path)
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)
        
        return self._process_json_data(data, file_path)

    async def _load_large_json_async(self, file_path: str) -> List[Document]:
        """Stream large JSON files to avoid memory issues"""
        self.logger.info(f"Streaming large JSON file: {file_path}")
        docs = []
        
        # For large files, we'll process in chunks
        # This is a simplified approach - for production, consider using ijson
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)
        
        # Process in batches if it's a list
        if isinstance(data, list):
            for batch in batch_iterator(data, self.batch_size):
                batch_docs = self._process_json_batch(batch, file_path)
                docs.extend(batch_docs)
        else:
            docs = self._process_json_data(data, file_path)
        
        return docs

    def _process_json_data(self, data: Any, source: str) -> List[Document]:
        """Process JSON data into documents"""
        docs = []
        
        if isinstance(data, list):
            for idx, entry in enumerate(data):
                content = entry.get("text") if isinstance(entry, dict) else json.dumps(entry)
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "loader": "JSONLoader",
                        "entry_index": idx,
                        "type": "list_entry"
                    }
                ))
        elif isinstance(data, dict):
            content = data.get("text", json.dumps(data))
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": source,
                    "loader": "JSONLoader",
                    "type": "object"
                }
            ))
        
        return docs

    def _process_json_batch(self, batch: List[Any], source: str) -> List[Document]:
        """Process a batch of JSON entries"""
        docs = []
        for idx, entry in enumerate(batch):
            content = entry.get("text") if isinstance(entry, dict) else json.dumps(entry)
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": source,
                    "loader": "JSONLoader",
                    "batch_processed": True
                }
            ))
        return docs

    def load(self) -> List[Document]:
        """Synchronous loading interface"""
        if self.enable_async:
            return asyncio.run(self.load_async())
        else:
            return self._load_sync()

    def _load_sync(self) -> List[Document]:
        """Traditional synchronous loading"""
        docs = []
        json_files = [f for f in os.listdir(self.path) if f.endswith(".json")]
        
        for file in json_files:
            file_path = os.path.join(self.path, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                docs.extend(self._process_json_data(data, file_path))
            except Exception as e:
                self.logger.error(f"Failed to load {file}: {e}")
        
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

