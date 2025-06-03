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
    """Web page loader with async HTTP requests and retry logic"""
    
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.timeout = self.config.get("web_timeout", 10)
        self.max_retries = 3

    async def load_async(self) -> List[Document]:
        """Load web pages asynchronously"""
        self.logger.info(f"Starting async web content loading from {self.path}")
        
        all_docs = []
        
        # Process URL files
        url_files = [f for f in os.listdir(self.path) if f.endswith(".txt")]
        for file in url_files:
            docs = await self._load_urls_from_file(os.path.join(self.path, file))
            all_docs.extend(docs)
        
        # Process HTML files
        html_files = [f for f in os.listdir(self.path) if f.endswith(".html")]
        for file in html_files:
            doc = await self._load_html_file(os.path.join(self.path, file))
            if doc:
                all_docs.append(doc)
        
        self.logger.info(f"Loaded {len(all_docs)} web documents")
        return all_docs

    async def _load_urls_from_file(self, file_path: str) -> List[Document]:
        """Load URLs from a text file and fetch content"""
        docs = []
        
        async with aiofiles.open(file_path, 'r') as f:
            urls = [line.strip() for line in await f.readlines() if line.strip()]
        
        # Create aiohttp session for concurrent requests
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_url_async(session, url) for url in urls]
            
            # Process with progress bar
            results = []
            for coro in async_tqdm.as_completed(tasks, desc="Fetching URLs"):
                result = await coro
                if result:
                    results.append(result)
        
        for url, content in results:
            text = self.extract_text(content)
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": url,
                    "loader": "WebPageLoader",
                    "fetch_time": datetime.utcnow().isoformat()
                }
            ))
        
        return docs

    async def _fetch_url_async(
        self, 
        session: aiohttp.ClientSession, 
        url: str
    ) -> Optional[Tuple[str, str]]:
        """Fetch URL with retry logic"""
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, timeout=self.timeout) as response:
                    response.raise_for_status()
                    content = await response.text()
                    return (url, content)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None

    async def _load_html_file(self, file_path: str) -> Optional[Document]:
        """Load local HTML file"""
        try:
            self.security_validator.validate_file_path(file_path)
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                html = await f.read()
            
            text = self.extract_text(html)
            return Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "loader": "WebPageLoader",
                    "file_name": os.path.basename(file_path)
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to load HTML file {file_path}: {e}")
            return None

    def extract_text(self, html: str) -> str:
        """Extract clean text from HTML with better parsing"""
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "meta", "link", "noscript"]):
                element.decompose()
            
            # Extract text with better formatting
            text_parts = []
            for element in soup.find_all(text=True):
                text = element.strip()
                if text and element.parent.name not in ["script", "style"]:
                    text_parts.append(text)
            
            return "\n".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from HTML: {e}")
            return ""

    def load(self) -> List[Document]:
        """Synchronous loading interface"""
        if self.enable_async:
            return asyncio.run(self.load_async())
        else:
            return self._load_sync()

    @retry_with_backoff(retries=3, exceptions=(requests.RequestException,))
    def _fetch_url_sync(self, url: str) -> Optional[str]:
        """Synchronous URL fetching with retry"""
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def _load_sync(self) -> List[Document]:
        """Traditional synchronous loading"""
        docs = []
        
        # Process URL files
        for file in os.listdir(self.path):
            if file.endswith(".txt"):
                with open(os.path.join(self.path, file), 'r') as f:
                    urls = [line.strip() for line in f.readlines() if line.strip()]
                
                for url in urls:
                    try:
                        html = self._fetch_url_sync(url)
                        if html:
                            text = self.extract_text(html)
                            docs.append(Document(
                                page_content=text,
                                metadata={"source": url, "loader": "WebPageLoader"}
                            ))
                    except Exception as e:
                        self.logger.error(f"Failed to fetch {url}: {e}")
            
            elif file.endswith(".html"):
                file_path = os.path.join(self.path, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html = f.read()
                    text = self.extract_text(html)
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "loader": "WebPageLoader"}
                    ))
                except Exception as e:
                    self.logger.error(f"Failed to load {file}: {e}")
        
        return docs
    

class SmartDocumentLoader(BaseDocumentLoader):
    """Smart loader with automatic format detection and optimized loading"""
    
    def __init__(self, config_path: str = "config.yaml", config: dict = None):
        super().__init__(config_path=config_path, config=config)
        self.path = self.config.get("data_path", "./data")
        
        # Validate data directory
        try:
            ensure_directory(self.path)
            if self.security_validator:
                self.security_validator.validate_file_path(self.path)
        except Exception as e:
            self.logger.error(f"Invalid data path: {e}")
            raise

    async def load_async(self) -> List[Document]:
        """Load documents asynchronously with intelligent routing"""
        self.logger.info(f"Starting smart async document loading from {self.path}")
        performance_monitor.start_timer("smart_loading")
        
        # Scan directory
        file_types = self._scan_directory()
        if not any(file_types.values()):
            self.logger.warning(f"No supported documents found in {self.path}")
            return []
        
        # Load different file types concurrently
        tasks = []
        
        if file_types["pdf"]:
            self.logger.info(f"Scheduling {len(file_types['pdf'])} PDF files")
            tasks.append(self._load_pdfs_async())
        
        if file_types["json"]:
            self.logger.info(f"Scheduling {len(file_types['json'])} JSON files")
            tasks.append(self._load_json_files_async())
        
        if file_types["web"]:
            self.logger.info(f"Scheduling {len(file_types['web'])} web sources")
            tasks.append(self._load_web_content_async())
        
        # Execute all loaders concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_docs = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Loader failed: {result}")
            elif isinstance(result, list):
                all_docs.extend(result)
        
        # Log performance metrics
        loading_time = performance_monitor.end_timer("smart_loading")
        self._log_final_statistics(all_docs, loading_time)
        
        return all_docs

    async def _load_pdfs_async(self) -> List[Document]:
        """Load PDFs asynchronously"""
        loader = PDFLoader(self.path, config=self.config)
        return await loader.load_async()

    async def _load_json_files_async(self) -> List[Document]:
        """Load JSON files asynchronously"""
        loader = JSONLoader(self.path, config=self.config)
        return await loader.load_async()

    async def _load_web_content_async(self) -> List[Document]:
        """Load web content asynchronously"""
        loader = WebPageLoader(self.path, config=self.config)
        return await loader.load_async()

    def load(self) -> List[Document]:
        """Synchronous loading interface"""
        if self.enable_async:
            return asyncio.run(self.load_async())
        else:
            return self._load_sync()

    @time_operation("smart_document_loading")
    @handle_errors(logger=None, raise_on_error=True)
    def _load_sync(self) -> List[Document]:
        """Traditional synchronous loading with timing"""
        self.logger.info(f"Starting smart document loading from {self.path}")
        
        if not os.path.exists(self.path):
            raise DocumentProcessingError(
                f"Data directory not found: {self.path}",
                error_code="DATA_DIR_NOT_FOUND",
                details={"path": self.path}
            )
        
        # Scan directory
        file_types = self._scan_directory()
        if not any(file_types.values()):
            self.logger.warning(f"No supported documents found in {self.path}")
            return []
        
        docs = []
        
        # Load each type
        if file_types["pdf"]:
            try:
                loader = PDFLoader(self.path, config=self.config)
                docs.extend(loader.load())
            except Exception as e:
                self.logger.error(f"PDF loading failed: {e}")
        
        if file_types["json"]:
            try:
                loader = JSONLoader(self.path, config=self.config)
                docs.extend(loader.load())
            except Exception as e:
                self.logger.error(f"JSON loading failed: {e}")
        
        if file_types["web"]:
            try:
                loader = WebPageLoader(self.path, config=self.config)
                docs.extend(loader.load())
            except Exception as e:
                self.logger.error(f"Web loading failed: {e}")
        
        return docs

    def _scan_directory(self) -> Dict[str, List[str]]:
        """Scan directory and categorize files"""
        file_types = {"pdf": [], "json": [], "web": []}
        
        try:
            for file in os.listdir(self.path):
                if file.endswith(".pdf"):
                    file_types["pdf"].append(file)
                elif file.endswith(".json"):
                    file_types["json"].append(file)
                elif file.endswith((".txt", ".html")):
                    file_types["web"].append(file)
            
            self.logger.debug(
                f"Directory scan complete: "
                f"{len(file_types['pdf'])} PDFs, "
                f"{len(file_types['json'])} JSON files, "
                f"{len(file_types['web'])} web sources"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to scan directory: {e}")
            raise DocumentProcessingError(
                f"Directory scan failed: {self.path}",
                error_code="SCAN_FAILED",
                details={"path": self.path, "error": str(e)}
            )
        
        return file_types

    def _log_final_statistics(self, docs: List[Document], loading_time: float):
        """Log comprehensive loading statistics"""
        # Count by loader type
        loader_counts = {}
        total_size = 0
        
        for doc in docs:
            loader = doc.metadata.get("loader", "Unknown")
            loader_counts[loader] = loader_counts.get(loader, 0) + 1
            
            # Sum file sizes if available
            if "file_size_mb" in doc.metadata:
                total_size += doc.metadata["file_size_mb"]
        
        # Get system metrics
        system_metrics = performance_monitor.get_system_metrics()
        
        self.logger.info(
            f"\n{'='*60}\n"
            f"Document Loading Complete:\n"
            f"  Total documents: {len(docs)}\n"
            f"  Loading time: {loading_time:.2f}s\n"
            f"  Documents/second: {len(docs)/loading_time:.2f}\n"
            f"  Total size: {total_size:.2f}MB\n"
            f"\nBreakdown by loader:\n" +
            "\n".join([f"  - {loader}: {count}" for loader, count in loader_counts.items()]) +
            f"\n\nSystem metrics:\n"
            f"  CPU usage: {system_metrics['cpu_percent']}%\n"
            f"  Memory usage: {system_metrics['memory']['percent']}%\n"
            f"  Available memory: {system_metrics['memory']['available_gb']:.2f}GB\n"
            f"{'='*60}"
        )
       
        
# Example usage demonstrating new features
if __name__ == "__main__":
    import argparse
    from utils import load_config
    
    parser = argparse.ArgumentParser(description="Test document loading with async support")
    parser.add_argument("--async", action="store_true", help="Use async loading")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--path", help="Override data path")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override settings if specified
    if args.path:
        config["data_path"] = args.path
    if args.async:
        config["performance"]["enable_async"] = True
    
    # Create loader
    loader = SmartDocumentLoader(config=config)
    
    try:
        # Load documents
        print("Starting document loading...")
        documents = loader.load()
        print(f"\nSuccessfully loaded {len(documents)} documents")
        
        # Test chunking
        print("\nTesting document chunking...")
        chunks = loader.split_documents(documents[:5])  # Test with first 5 docs
        print(f"Created {len(chunks)} chunks from {min(5, len(documents))} documents")
        
        # Show sample chunk
        if chunks:
            print(f"\nSample chunk metadata: {chunks[0].metadata}")
        
    except Exception as e:
        print(f"Error during loading: {e}")
        import traceback
        traceback.print_exc()

