import argparse
import warnings
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import json

from tqdm import tqdm

# Import from enhanced modules
from utils import (
    load_config, setup_logging, handle_errors, time_operation,
    performance_monitor, ConfigurationError
)
from vectorstore_manager import VectorstoreManager
from document_loader import SmartDocumentLoader
from monitoring import get_monitoring_instance, HealthChecker


class VectorstoreUpdateCLI:
    """Enhanced CLI for vectorstore management"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        
        # Load configuration
        try:
            self.config = load_config(self.args.config)
        except ConfigurationError as e:
            print(f"❌ Configuration error: {e}")
            sys.exit(1)
        
        # Setup logging
        log_config = self.config.get("logging", {})
        self.logger = setup_logging(
            name="rag_chat.vectorstore_update",
            log_level=self.args.log_level or log_config.get("level", "INFO"),
            log_file=log_config.get("file_path"),
            console_level=self.args.log_level or log_config.get("console_level", "INFO")
        )
        
        # Initialize monitoring
        self.metrics = get_monitoring_instance(self.config)
        self.health_checker = HealthChecker(self.config)
        
        # Component instances
        self.vs_manager = None
        self.doc_loader = None
        
        self.logger.info(f"VectorstoreUpdateCLI initialized with action: {self._get_action()}")
    
    def _get_action(self) -> str:
        """Determine which action to perform"""
        if self.args.delete:
            return "delete"
        elif self.args.reset:
            return "reset"
        elif self.args.info:
            return "info"
        elif self.args.optimize:
            return "optimize"
        elif self.args.check:
            return "check"
        elif self.args.export:
            return "export"
        elif self.args.import_from:
            return "import"
        else:
            return "update"
    
    @handle_errors(logger=None, raise_on_error=True)
    async def run_async(self):
        """Run the CLI command asynchronously"""
        action = self._get_action()
        
        self.logger.info(f"Executing action: {action}")
        
        # Initialize components
        self.vs_manager = VectorstoreManager(self.config)
        
        # Execute action
        if action == "delete":
            await self._delete_vectorstore()
        elif action == "reset":
            await self._reset_vectorstore()
        elif action == "info":
            await self._show_info()
        elif action == "optimize":
            await self._optimize_vectorstore()
        elif action == "check":
            await self._check_health()
        elif action == "export":
            await self._export_vectorstore()
        elif action == "import":
            await self._import_vectorstore()
        else:  # update
            await self._update_vectorstore()
    
    def run(self):
        """Run the CLI command"""
        if self.config.get("performance", {}).get("enable_async", True) and not self.args.no_async:
            asyncio.run(self.run_async())
        else:
            self._run_sync()
    
    def _run_sync(self):
        """Run synchronously"""
        action = self._get_action()
        
        self.logger.info(f"Executing action (sync): {action}")
        
        # Initialize components
        self.vs_manager = VectorstoreManager(self.config)
        
        # Execute action
        if action == "delete":
            self._delete_vectorstore_sync()
        elif action == "reset":
            self._reset_vectorstore_sync()
        elif action == "info":
            self._show_info_sync()
        elif action == "optimize":
            self._optimize_vectorstore_sync()
        elif action == "check":
            self._check_health_sync()
        else:  # update
            self._update_vectorstore_sync()
    
    async def _delete_vectorstore(self):
        """Delete the vectorstore"""
        print("\n⚠️  WARNING: This will delete the entire vectorstore!")
        
        if not self.args.force:
            confirm = input("Are you sure? Type 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                print("❌ Deletion cancelled")
                return
        
        print("🗑️  Deleting vectorstore...")
        self.vs_manager.delete_vectorstore()
        print("✅ Vectorstore deleted successfully")
        
        # Record metrics
        self.metrics.record_metric("vectorstore_deleted", 1.0)
    
    def _delete_vectorstore_sync(self):
        """Delete vectorstore (sync)"""
        asyncio.run(self._delete_vectorstore())
    
    async def _reset_vectorstore(self):
        """Reset (delete and rebuild) the vectorstore"""
        print("\n🔄 Resetting vectorstore...")
        
        # Delete existing
        self.vs_manager.delete_vectorstore()
        print("✅ Old vectorstore deleted")
        
        # Rebuild
        await self._update_vectorstore(force=True)
    
    def _reset_vectorstore_sync(self):
        """Reset vectorstore (sync)"""
        print("\n🔄 Resetting vectorstore...")
        self.vs_manager.delete_vectorstore()
        print("✅ Old vectorstore deleted")
        self._update_vectorstore_sync(force=True)
    
    async def _update_vectorstore(self, force: bool = False):
        """Update the vectorstore with new documents"""
        print("\n📄 Loading documents...")
        
        # Initialize document loader
        self.doc_loader = SmartDocumentLoader(config=self.config)
        
        # Load documents
        start_time = performance_monitor.start_timer("document_loading")
        documents = await asyncio.get_event_loop().run_in_executor(
            None, self.doc_loader.load
        )
        loading_time = performance_monitor.end_timer("document_loading")
        
        print(f"✅ Loaded {len(documents)} documents in {loading_time:.2f}s")
        
        if not documents:
            print("⚠️  No documents found to process")
            return
        
        # Split into chunks
        print("\n🔪 Splitting documents into chunks...")
        chunks = await asyncio.get_event_loop().run_in_executor(
            None, self.doc_loader.split_documents, documents
        )
        print(f"✅ Created {len(chunks)} chunks")
        
        # Load vectorstore
        print("\n🗄️  Loading vectorstore...")
        self.vs_manager.load_vectorstore()
        
        # Check if update needed
        if not force and not self.vs_manager.needs_update(chunks):
            print("✅ Vectorstore is already up to date")
            
            # Show current stats
            health = self.vs_manager.get_health_status()
            print(f"\n📊 Current Statistics:")
            print(f"   Documents: {health['total_documents']:,}")
            print(f"   Size: {health['index_size_mb']:.2f}MB")
            return
        
        # Add documents
        print("\n📥 Adding documents to vectorstore...")
        if self.config.get("performance", {}).get("enable_async", True):
            await self.vs_manager.add_documents_async(chunks)
        else:
            self.vs_manager.add_documents(chunks)
        
        # Optimize if requested
        if self.args.optimize_after:
            print("\n🔧 Optimizing vectorstore...")
            self.vs_manager.optimize_index()
        
        # Show final stats
        health = self.vs_manager.get_health_status()
        print(f"\n✅ Update Complete!")
        print(f"   Total documents: {health['total_documents']:,}")
        print(f"   Index size: {health['index_size_mb']:.2f}MB")
        
        # Show performance metrics
        if self.args.verbose:
            print(f"\n📊 Performance Metrics:")
            metrics_summary = self.metrics.get_metrics_summary("vectorstore_add", hours=0.1)
            if metrics_summary.get("count", 0) > 0:
                print(f"   Add operations: {metrics_summary['count']}")
                print(f"   Average time: {metrics_summary['mean']:.2f}s")
    
    def _update_vectorstore_sync(self, force: bool = False):
        """Update vectorstore (sync)"""
        print("\n📄 Loading documents...")
        
        self.doc_loader = SmartDocumentLoader(config=self.config)
        documents = self.doc_loader.load()
        print(f"✅ Loaded {len(documents)} documents")
        
        if not documents:
            print("⚠️  No documents found")
            return
        
        print("\n🔪 Splitting documents...")
        chunks = self.doc_loader.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        print("\n🗄️  Loading vectorstore...")
        self.vs_manager.load_vectorstore()
        
        if not force and not self.vs_manager.needs_update(chunks):
            print("✅ Already up to date")
            return
        
        print("\n📥 Adding documents...")
        self.vs_manager.add_documents(chunks)
        
        if self.args.optimize_after:
            self.vs_manager.optimize_index()
        
        print("\n✅ Update complete!")
    
    async def _show_info(self):
        """Show vectorstore information"""
        print("\n📊 Vectorstore Information")
        print("=" * 50)
        
        try:
            # Load vectorstore
            self.vs_manager.load_vectorstore()
            
            # Get health status
            health = self.vs_manager.get_health_status()
            
            print(f"\n🏥 Health Status: {'✅ Healthy' if health['is_healthy'] else '❌ Unhealthy'}")
            print(f"   Last check: {health['last_check'] or 'Never'}")
            
            print(f"\n📈 Statistics:")
            print(f"   Total documents: {health['total_documents']:,}")
            print(f"   Total embeddings: {health['total_embeddings']:,}")
            print(f"   Index size: {health['index_size_mb']:.2f}MB")
            
            if health['error_count'] > 0:
                print(f"\n⚠️  Recent Errors: {health['error_count']}")
                for error in health['recent_errors'][:3]:
                    print(f"   - {error}")
            
            # Performance metrics
            if health['performance_metrics']:
                print(f"\n⚡ Performance:")
                for metric, value in health['performance_metrics'].items():
                    print(f"   {metric}: {value}")
            
            # Configuration
            print(f"\n⚙️  Configuration:")
            print(f"   Path: {self.config.get('vector_db_path')}")
            print(f"   Embedding model: {self.config.get('embedding', {}).get('model_name')}")
            print(f"   Batch size: {self.config.get('performance', {}).get('batch_size')}")
            
        except Exception as e:
            print(f"\n❌ Error loading vectorstore: {e}")
            print("   The vectorstore may not exist or is corrupted")
    
    def _show_info_sync(self):
        """Show info (sync)"""
        asyncio.run(self._show_info())
    
    async def _optimize_vectorstore(self):
        """Optimize the vectorstore"""
        print("\n🔧 Optimizing vectorstore...")
        
        try:
            # Load vectorstore
            self.vs_manager.load_vectorstore()
            
            # Get initial stats
            health_before = self.vs_manager.get_health_status()
            
            # Optimize
            self.vs_manager.optimize_index()
            
            # Get final stats
            health_after = self.vs_manager.get_health_status()
            
            print("\n✅ Optimization complete!")
            print(f"   Size before: {health_before['index_size_mb']:.2f}MB")
            print(f"   Size after: {health_after['index_size_mb']:.2f}MB")
            
            reduction = health_before['index_size_mb'] - health_after['index_size_mb']
            if reduction > 0:
                print(f"   Space saved: {reduction:.2f}MB ({reduction/health_before['index_size_mb']*100:.1f}%)")
            
        except Exception as e:
            print(f"\n❌ Optimization failed: {e}")
    
    def _optimize_vectorstore_sync(self):
        """Optimize (sync)"""
        asyncio.run(self._optimize_vectorstore())
    
    async def _check_health(self):
        """Run comprehensive health checks"""
        print("\n🏥 Running Health Checks")
        print("=" * 50)
        
        # System health
        health_status = await self.health_checker.run_health_checks()
        
        print("\n📋 System Health:")
        for check, status in health_status.items():
            icon = "✅" if status["status"] == "healthy" else "❌"
            print(f"   {icon} {check}: {status['status']}")
            if status.get("warning"):
                print(f"      ⚠️  {status['warning']}")
        
        # Vectorstore health
        try:
            self.vs_manager.load_vectorstore()
            vs_health = self.vs_manager.get_health_status()
            
            print(f"\n📋 Vectorstore Health:")
            print(f"   Status: {'✅ Healthy' if vs_health['is_healthy'] else '❌ Unhealthy'}")
            print(f"   Documents: {vs_health['total_documents']:,}")
            print(f"   Size: {vs_health['index_size_mb']:.2f}MB")
            
        except Exception as e:
            print(f"\n📋 Vectorstore Health: ❌ Error")
            print(f"   {e}")
        
        # Data directory check
        data_path = Path(self.config.get("data_path", "./data"))
        if data_path.exists():
            files = list(data_path.iterdir())
            print(f"\n📋 Data Directory:")
            print(f"   Path: {data_path}")
            print(f"   Files: {len(files)}")
            
            # Count by type
            type_counts = {}
            for file in files:
                ext = file.suffix.lower()
                type_counts[ext] = type_counts.get(ext, 0) + 1
            
            for ext, count in sorted(type_counts.items()):
                print(f"   {ext}: {count} files")
        else:
            print(f"\n📋 Data Directory: ❌ Not found")
    
    def _check_health_sync(self):
        """Check health (sync)"""
        asyncio.run(self._check_health())
    
    async def _export_vectorstore(self):
        """Export vectorstore metadata"""
        output_file = self.args.export
        
        print(f"\n📤 Exporting vectorstore metadata to {output_file}...")
        
        try:
            self.vs_manager.load_vectorstore()
            
            # Get all metadata
            export_data = {
                "health": self.vs_manager.get_health_status(),
                "config": {
                    "path": str(self.config.get("vector_db_path")),
                    "embedding_model": self.config.get("embedding", {}).get("model_name"),
                    "chunk_size": self.config.get("chunk", {}).get("size"),
                    "chunk_overlap": self.config.get("chunk", {}).get("overlap")
                },
                "exported_at": datetime.now().isoformat()
            }
            
            # Save to file
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Export complete: {output_file}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    async def _import_vectorstore(self):
        """Import vectorstore from another location"""
        source_path = Path(self.args.import_from)
        
        if not source_path.exists():
            print(f"❌ Source path not found: {source_path}")
            return
        
        print(f"\n📥 Importing vectorstore from {source_path}...")
        
        if not self.args.force:
            print("⚠️  This will replace the current vectorstore!")
            confirm = input("Continue? (yes/no): ")
            if confirm.lower() != 'yes':
                print("❌ Import cancelled")
                return
        
        try:
            # Create backup of current vectorstore
            if Path(self.config.get("vector_db_path")).exists():
                print("💾 Creating backup of current vectorstore...")
                self.vs_manager._create_backup()
            
            # Copy new vectorstore
            import shutil
            dest_path = Path(self.config.get("vector_db_path"))
            
            if dest_path.exists():
                shutil.rmtree(dest_path)
            
            shutil.copytree(source_path, dest_path)
            
            # Verify import
            self.vs_manager = VectorstoreManager(self.config)
            self.vs_manager.load_vectorstore()
            health = self.vs_manager.get_health_status()
            
            print(f"\n✅ Import complete!")
            print(f"   Documents: {health['total_documents']:,}")
            print(f"   Size: {health['index_size_mb']:.2f}MB")
            
        except Exception as e:
            print(f"❌ Import failed: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for vectorstore management"""
    parser = argparse.ArgumentParser(
        description="Manage RAG Chat vectorstore - Update, optimize, or reset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update vectorstore with new documents
  python run_vectorstore_update.py --update
  
  # Delete and rebuild vectorstore
  python run_vectorstore_update.py --reset
  
  # Show vectorstore information
  python run_vectorstore_update.py --info
  
  # Optimize vectorstore
  python run_vectorstore_update.py --optimize
  
  # Run health checks
  python run_vectorstore_update.py --check
  
  # Export vectorstore metadata
  python run_vectorstore_update.py --export vectorstore_info.json
        """
    )
    
    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--update", 
        action="store_true", 
        help="Update vectorstore with new documents (default)"
    )
    action_group.add_argument(
        "--delete", 
        action="store_true", 
        help="Delete the existing vectorstore"
    )
    action_group.add_argument(
        "--reset", 
        action="store_true", 
        help="Delete and rebuild the vectorstore"
    )
    action_group.add_argument(
        "--info", 
        action="store_true", 
        help="Show vectorstore information"
    )
    action_group.add_argument(
        "--optimize", 
        action="store_true", 
        help="Optimize vectorstore index"
    )
    action_group.add_argument(
        "--check", 
        action="store_true", 
        help="Run comprehensive health checks"
    )
    action_group.add_argument(
        "--export", 
        type=str, 
        metavar="FILE",
        help="Export vectorstore metadata to file"
    )
    action_group.add_argument(
        "--import-from", 
        type=str, 
        metavar="PATH",
        help="Import vectorstore from another location"
    )
    
    # Configuration arguments
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to config file (default: config.yaml)"
    )
    config_group.add_argument(
        "--data-path", 
        type=str, 
        help="Override data directory path"
    )
    config_group.add_argument(
        "--vector-path", 
        type=str, 
        help="Override vectorstore path"
    )
    
    # Performance arguments
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--batch-size", 
        type=int, 
        help="Override batch size for processing"
    )
    perf_group.add_argument(
        "--no-async", 
        action="store_true", 
        help="Disable async operations"
    )
    perf_group.add_argument(
        "--optimize-after", 
        action="store_true", 
        help="Optimize index after update"
    )
    
    # Display arguments
    display_group = parser.add_argument_group("Display Options")
    display_group.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose output"
    )
    display_group.add_argument(
        "--quiet", "-q", 
        action="store_true", 
        help="Minimal output"
    )
    display_group.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        help="Set log level"
    )
    
    # Safety arguments
    safety_group = parser.add_argument_group("Safety Options")
    safety_group.add_argument(
        "--force", "-f", 
        action="store_true", 
        help="Skip confirmation prompts"
    )
    safety_group.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be done without doing it"
    )
    safety_group.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode and show warnings"
    )
    
    return parser


def show_summary(args: argparse.Namespace):
    """Show summary of actions to be performed"""
    print("\n📋 Vectorstore Update Summary")
    print("=" * 50)
    
    # Determine action
    if args.delete:
        print("🗑️  Action: DELETE vectorstore")
    elif args.reset:
        print("🔄 Action: RESET (delete and rebuild)")
    elif args.info:
        print("📊 Action: Show information")
    elif args.optimize:
        print("🔧 Action: Optimize index")
    elif args.check:
        print("🏥 Action: Run health checks")
    elif args.export:
        print(f"📤 Action: Export metadata to {args.export}")
    elif args.import_from:
        print(f"📥 Action: Import from {args.import_from}")
    else:
        print("📝 Action: UPDATE with new documents")
    
    print(f"\n⚙️  Configuration:")
    print(f"   Config file: {args.config}")
    
    if args.data_path:
        print(f"   Data path override: {args.data_path}")
    if args.vector_path:
        print(f"   Vector path override: {args.vector_path}")
    
    if args.batch_size:
        print(f"   Batch size: {args.batch_size}")
    if args.no_async:
        print(f"   Async: Disabled")
    if args.optimize_after:
        print(f"   Optimize after: Yes")
    
    if args.dry_run:
        print(f"\n⚠️  DRY RUN MODE - No changes will be made")
    
    print("=" * 50 + "\n")


def main():
    """Main entry point"""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # If no action specified, default to update
    if not any([args.delete, args.reset, args.info, args.optimize, 
                args.check, args.export, args.import_from]):
        args.update = True
    
    # Configure warnings
    if not args.debug:
        warnings.filterwarnings("ignore")
    
    # Show summary unless quiet
    if not args.quiet:
        show_summary(args)
    
    # Handle dry run
    if args.dry_run:
        print("✅ Dry run complete. No changes were made.")
        return
    
    # Create and run CLI
    try:
        cli = VectorstoreUpdateCLI(args)
        cli.run()
        
        # Show performance summary if verbose
        if args.verbose:
            perf_metrics = performance_monitor.get_metrics()
            if perf_metrics:
                print("\n⚡ Performance Summary:")
                for op, stats in perf_metrics.items():
                    print(f"   {op}: {stats['average']:.2f}s avg ({stats['count']} calls)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"\n❌ Error: {e}")
            print("\nRun with --debug for full traceback")
        sys.exit(1)


if __name__ == "__main__":
    main()