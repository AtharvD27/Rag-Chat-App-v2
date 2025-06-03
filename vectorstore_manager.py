import os
import shutil
import asyncio
import pickle
import json
from typing import List, Set, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
import chromadb

# Import from enhanced utils
from utils import (
    setup_logging, VectorStoreError, handle_errors, retry_with_backoff,
    time_operation, batch_iterator, performance_monitor, ensure_directory,
    get_file_info
)


class VectorstoreHealth:
    """Track vectorstore health metrics"""
    
    def __init__(self):
        self.last_check = None
        self.is_healthy = True
        self.total_documents = 0
        self.total_embeddings = 0
        self.index_size_mb = 0
        self.errors = []
        self.performance_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health status to dictionary"""
        return {
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "is_healthy": self.is_healthy,
            "total_documents": self.total_documents,
            "total_embeddings": self.total_embeddings,
            "index_size_mb": self.index_size_mb,
            "error_count": len(self.errors),
            "recent_errors": self.errors[-5:],  # Last 5 errors
            "performance_metrics": self.performance_metrics
        }


class VectorstoreManager:
    """Enhanced vectorstore manager with batch operations and reliability features"""
    
    def __init__(self, config: dict):
        self.config = config
        self.chroma_path = Path(self.config.get("vector_db_path", "./vector_db"))
        self.backup_path = Path(self.config.get("vector_backup_path", "./vector_db_backup"))
        
        # Performance settings
        perf_config = self.config.get("performance", {})
        self.batch_size = perf_config.get("batch_size", 100)
        self.max_concurrent = perf_config.get("max_concurrent_operations", 4)
        self.enable_async = perf_config.get("enable_async", True)
        
        # Embedding configuration
        embedding_config = self.config.get("embedding", {})
        model_name = embedding_config.get("model_name", "all-MiniLM-L6-v2")
        
        # Setup logging
        log_config = self.config.get("logging", {})
        self.logger = setup_logging(
            name="rag_chat.VectorstoreManager",
            log_level=log_config.get("level", "INFO"),
            log_file=log_config.get("file_path"),
            console_level=log_config.get("console_level")
        )
        
        # Initialize embedding function with caching
        self.logger.info(f"Initializing embeddings with model: {model_name}")
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=str(ensure_directory(Path("./models_cache")))
        )
        
        # ChromaDB settings for better performance
        self.chroma_settings = Settings(
            persist_directory=str(self.chroma_path),
            anonymized_telemetry=False,
            allow_reset=True
        )
        
        self.vs = None
        self.health = VectorstoreHealth()
        
        # Cache for embedding operations
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.logger.info(
            f"Initialized VectorstoreManager with batch_size={self.batch_size}, "
            f"max_concurrent={self.max_concurrent}"
        )

    @handle_errors(logger=None, raise_on_error=True)
    def load_vectorstore(self) -> None:
        """Load vectorstore with health checking"""
        self.logger.info(f"Loading vectorstore from {self.chroma_path}")
        
        try:
            self.vs = Chroma(
                persist_directory=str(self.chroma_path),
                embedding_function=self.embedding_function,
                client_settings=self.chroma_settings
            )
            
            # Perform health check
            self._check_health()
            
            if not self.health.is_healthy:
                self.logger.warning("Vectorstore health check failed, attempting recovery")
                self._attempt_recovery()
            
        except Exception as e:
            self.logger.error(f"Failed to load vectorstore: {e}")
            raise VectorStoreError(
                f"Failed to load vectorstore from {self.chroma_path}",
                error_code="LOAD_FAILED",
                details={"path": str(self.chroma_path), "error": str(e)}
            )

    async def add_documents_async(self, chunks: List[Document]) -> None:
        """Add documents asynchronously with batch processing"""
        if self.vs is None:
            self.load_vectorstore()
        
        self.logger.info(f"Starting async document addition: {len(chunks)} chunks")
        performance_monitor.start_timer("vectorstore_add")
        
        try:
            # Get existing IDs
            existing_ids = await self._get_existing_ids_async()
            
            # Filter new chunks
            new_chunks, new_ids = self._filter_new_chunks(chunks, existing_ids)
            
            if not new_chunks:
                self.logger.info("No new chunks to add")
                return
            
            self.logger.info(f"Adding {len(new_chunks)} new chunks in batches of {self.batch_size}")
            
            # Process in batches
            total_added = 0
            errors = []
            
            with tqdm(total=len(new_chunks), desc="Adding to vectorstore") as pbar:
                for batch_chunks, batch_ids in self._batch_documents(new_chunks, new_ids):
                    try:
                        await self._add_batch_async(batch_chunks, batch_ids)
                        total_added += len(batch_chunks)
                        pbar.update(len(batch_chunks))
                    except Exception as e:
                        self.logger.error(f"Failed to add batch: {e}")
                        errors.append({"batch_size": len(batch_chunks), "error": str(e)})
                        # Continue with next batch
            
            # Log results
            elapsed = performance_monitor.end_timer("vectorstore_add")
            self._log_addition_results(total_added, len(new_chunks), elapsed, errors)
            
            # Update health metrics
            self._check_health()
            
        except Exception as e:
            self.logger.error(f"Document addition failed: {e}")
            raise VectorStoreError(
                "Failed to add documents to vectorstore",
                error_code="ADD_FAILED",
                details={"chunk_count": len(chunks), "error": str(e)}
            )

    async def _get_existing_ids_async(self) -> Set[str]:
        """Get existing document IDs asynchronously"""
        loop = asyncio.get_event_loop()
        
        def get_ids():
            try:
                store_data = self.vs.get(include=["ids"])
                return set(store_data["ids"])
            except Exception:
                return set()
        
        return await loop.run_in_executor(None, get_ids)

    async def _add_batch_async(self, batch_chunks: List[Document], batch_ids: List[str]) -> None:
        """Add a batch of documents asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Generate embeddings in parallel
        texts = [doc.page_content for doc in batch_chunks]
        embeddings = await self._generate_embeddings_async(texts)
        
        # Add to vectorstore
        def add_to_store():
            metadatas = [doc.metadata for doc in batch_chunks]
            self.vs._collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=batch_ids
            )
        
        await loop.run_in_executor(None, add_to_store)

    async def _generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with caching and parallel processing"""
        embeddings = []
        texts_to_embed = []
        cached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if text_hash in self.embedding_cache:
                embeddings.append(self.embedding_cache[text_hash])
                cached_indices.append(i)
                self.cache_hits += 1
            else:
                texts_to_embed.append(text)
                self.cache_misses += 1
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                # Process in smaller sub-batches for memory efficiency
                sub_batch_size = 10
                new_embeddings = []
                
                for sub_batch in batch_iterator(texts_to_embed, sub_batch_size):
                    sub_embeddings = await loop.run_in_executor(
                        executor,
                        self.embedding_function.embed_documents,
                        sub_batch
                    )
                    new_embeddings.extend(sub_embeddings)
                
                # Cache new embeddings
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    text_hash = hash(text)
                    self.embedding_cache[text_hash] = embedding
        
        # Reconstruct full embeddings list in original order
        final_embeddings = []
        new_idx = 0
        
        for i in range(len(texts)):
            if i in cached_indices:
                # Use cached embedding
                text_hash = hash(texts[i])
                final_embeddings.append(self.embedding_cache[text_hash])
            else:
                # Use newly generated embedding
                final_embeddings.append(new_embeddings[new_idx])
                new_idx += 1
        
        return final_embeddings

    def add_documents(self, chunks: List[Document]) -> None:
        """Synchronous document addition with batch processing"""
        if self.enable_async:
            asyncio.run(self.add_documents_async(chunks))
        else:
            self._add_documents_sync(chunks)

    @time_operation("vectorstore_add_sync")
    def _add_documents_sync(self, chunks: List[Document]) -> None:
        """Traditional synchronous document addition"""
        if self.vs is None:
            self.load_vectorstore()
        
        try:
            # Get existing IDs
            existing_ids = set()
            try:
                store_data = self.vs.get(include=["ids"])
                existing_ids = set(store_data["ids"])
            except Exception:
                pass
            
            # Filter new chunks
            new_chunks, new_ids = self._filter_new_chunks(chunks, existing_ids)
            
            if not new_chunks:
                self.logger.info("No new chunks to add")
                return
            
            self.logger.info(f"Adding {len(new_chunks)} new chunks")
            
            # Add in batches
            for batch_chunks, batch_ids in tqdm(
                self._batch_documents(new_chunks, new_ids),
                total=len(new_chunks) // self.batch_size + 1,
                desc="Adding chunks"
            ):
                self.vs.add_documents(batch_chunks, ids=batch_ids)
            
            self.logger.info(f"Successfully added {len(new_chunks)} chunks")
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise

    def _filter_new_chunks(
        self, 
        chunks: List[Document], 
        existing_ids: Set[str]
    ) -> Tuple[List[Document], List[str]]:
        """Filter out chunks that already exist"""
        new_chunks = []
        new_ids = []
        
        for doc in chunks:
            doc_id = doc.metadata.get("id")
            if doc_id and doc_id not in existing_ids:
                new_chunks.append(doc)
                new_ids.append(doc_id)
        
        return new_chunks, new_ids

    def _batch_documents(
        self, 
        chunks: List[Document], 
        ids: List[str]
    ) -> List[Tuple[List[Document], List[str]]]:
        """Batch documents for processing"""
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]
            batches.append((batch_chunks, batch_ids))
        return batches

    def needs_update(self, chunks: List[Document]) -> bool:
        """Check if vectorstore needs updating"""
        if self.vs is None:
            return True
        
        try:
            store_data = self.vs.get(include=["metadatas"])
            existing_ids = {
                meta.get("id") for meta in store_data["metadatas"]
                if meta.get("id") is not None
            }
        except Exception:
            return True
        
        new_ids = {doc.metadata["id"] for doc in chunks if doc.metadata.get("id")}
        return not existing_ids.issuperset(new_ids)

    @handle_errors(logger=None, raise_on_error=False)
    def delete_vectorstore(self) -> None:
        """Delete vectorstore with backup option"""
        if os.path.exists(self.chroma_path):
            # Create backup before deletion
            if self.config.get("backup_before_delete", True):
                self._create_backup()
            
            shutil.rmtree(self.chroma_path)
            self.logger.info(f"Deleted vectorstore at {self.chroma_path}")
            
            # Clear cache
            self.embedding_cache.clear()
            self.vs = None
        else:
            self.logger.warning("Vectorstore directory does not exist")

    def _create_backup(self) -> None:
        """Create backup of vectorstore"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.backup_path / f"backup_{timestamp}"
            
            self.logger.info(f"Creating backup at {backup_dir}")
            shutil.copytree(self.chroma_path, backup_dir)
            
            # Keep only last 3 backups
            self._cleanup_old_backups()
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")

    def _cleanup_old_backups(self) -> None:
        """Remove old backups, keeping only the most recent ones"""
        if not self.backup_path.exists():
            return
        
        backups = sorted(
            [d for d in self.backup_path.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Keep only last 3 backups
        for backup in backups[3:]:
            try:
                shutil.rmtree(backup)
                self.logger.debug(f"Removed old backup: {backup}")
            except Exception as e:
                self.logger.warning(f"Failed to remove backup {backup}: {e}")

    def _check_health(self) -> None:
        """Perform health check on vectorstore"""
        try:
            self.health.last_check = datetime.now()
            
            # Get collection stats
            collection = self.vs._collection
            count = collection.count()
            
            self.health.total_documents = count
            self.health.is_healthy = True
            
            # Check index size
            if self.chroma_path.exists():
                size_bytes = sum(
                    f.stat().st_size 
                    for f in self.chroma_path.rglob('*') 
                    if f.is_file()
                )
                self.health.index_size_mb = size_bytes / (1024 * 1024)
            
            # Log cache statistics
            cache_total = self.cache_hits + self.cache_misses
            if cache_total > 0:
                cache_hit_rate = self.cache_hits / cache_total * 100
                self.logger.debug(f"Embedding cache hit rate: {cache_hit_rate:.1f}%")
            
        except Exception as e:
            self.health.is_healthy = False
            self.health.errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
            self.logger.error(f"Health check failed: {e}")

    def _attempt_recovery(self) -> None:
        """Attempt to recover from vectorstore corruption"""
        self.logger.warning("Attempting vectorstore recovery")
        
        # Try to load from backup
        if self.backup_path.exists():
            backups = sorted(
                [d for d in self.backup_path.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if backups:
                latest_backup = backups[0]
                self.logger.info(f"Restoring from backup: {latest_backup}")
                
                try:
                    # Remove corrupted store
                    if self.chroma_path.exists():
                        shutil.rmtree(self.chroma_path)
                    
                    # Restore from backup
                    shutil.copytree(latest_backup, self.chroma_path)
                    
                    # Reload
                    self.load_vectorstore()
                    
                    if self.health.is_healthy:
                        self.logger.info("Recovery successful")
                    else:
                        self.logger.error("Recovery failed")
                        
                except Exception as e:
                    self.logger.error(f"Recovery failed: {e}")
                    raise VectorStoreError(
                        "Failed to recover from backup",
                        error_code="RECOVERY_FAILED",
                        details={"backup": str(latest_backup), "error": str(e)}
                    )

    def _log_addition_results(
        self, 
        added: int, 
        total: int, 
        elapsed: float, 
        errors: List[Dict]
    ) -> None:
        """Log detailed results of document addition"""
        docs_per_sec = added / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"\nDocument Addition Complete:\n"
            f"  - Total new chunks: {total}\n"
            f"  - Successfully added: {added}\n"
            f"  - Failed: {len(errors)}\n"
            f"  - Time elapsed: {elapsed:.2f}s\n"
            f"  - Rate: {docs_per_sec:.2f} docs/sec\n"
            f"  - Cache hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%"
        )
        
        if errors:
            self.logger.warning(f"Encountered {len(errors)} errors during addition")

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.health.to_dict()

    def optimize_index(self) -> None:
        """Optimize vectorstore index for better performance"""
        self.logger.info("Optimizing vectorstore index")
        
        try:
            # Persist any pending changes
            if hasattr(self.vs, 'persist'):
                self.vs.persist()
            
            # Clear embedding cache if it's too large
            if len(self.embedding_cache) > 10000:
                self.logger.info("Clearing embedding cache")
                self.embedding_cache.clear()
                self.cache_hits = 0
                self.cache_misses = 0
            
            self.logger.info("Index optimization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to optimize index: {e}")


# Example usage
if __name__ == "__main__":
    from utils import load_config
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Create manager
    manager = VectorstoreManager(config)
    
    # Load vectorstore
    manager.load_vectorstore()
    
    # Check health
    health = manager.get_health_status()
    print(f"Vectorstore health: {json.dumps(health, indent=2)}")
    
    # Example: Add some test documents
    test_docs = [
        Document(
            page_content=f"Test document {i}",
            metadata={"id": f"test_{i}", "source": "test"}
        )
        for i in range(10)
    ]
    
    # Add documents
    manager.add_documents(test_docs)
    
    # Optimize index
    manager.optimize_index()