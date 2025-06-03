import argparse
import warnings
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

# Import from enhanced modules
from utils import (
    load_config, setup_logging, handle_errors, performance_monitor,
    ConfigurationError, ensure_directory
)
from run_chat import (
    load_documents, update_vectorstore,
    setup_llm, handle_session, start_session
)
from monitoring import (
    get_monitoring_instance, MetricsCollector, HealthChecker, MonitoringDashboard
)


class RAGChatCLI:
    """Enhanced CLI for RAG Chat Application"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        
        # Load configuration with validation
        try:
            self.config = load_config(
                self.args.config,
                allow_env_override=not self.args.no_env_override
            )
        except ConfigurationError as e:
            print(f"❌ Configuration error: {e}")
            print(f"   Details: {e.details}")
            sys.exit(1)
        
        # Apply CLI overrides
        self._apply_overrides()
        
        # Setup logging
        log_config = self.config.get("logging", {})
        self.logger = setup_logging(
            name="rag_chat.main",
            log_level=log_config.get("level", "INFO"),
            log_file=log_config.get("file_path"),
            console_level=self.args.log_level or log_config.get("console_level", "INFO")
        )
        
        # Initialize monitoring
        self.metrics = get_monitoring_instance(self.config)
        self.health_checker = HealthChecker(self.config)
        self.dashboard = None
        
        # Show startup info
        self._show_startup_info()
    
    def _apply_overrides(self):
        """Apply command-line overrides to configuration"""
        overrides = {
            "model_path": self.args.model_path,
            "temperature": self.args.temperature,
            "model_profile": self.args.model,
            "batch_size": self.args.batch_size,
            "async_enabled": not self.args.no_async
        }
        
        # Apply model overrides
        if overrides["model_path"]:
            self.config.setdefault("llm", {})["local_model_path"] = overrides["model_path"]
            self.logger.info(f"Overriding model path: {overrides['model_path']}")
        
        if overrides["temperature"] is not None:
            self.config.setdefault("llm", {})["temperature"] = overrides["temperature"]
            self.logger.info(f"Overriding temperature: {overrides['temperature']}")
        
        if overrides["model_profile"]:
            self.config.setdefault("models", {})["default_profile"] = overrides["model_profile"]
            self.logger.info(f"Using model profile: {overrides['model_profile']}")
        
        # Apply performance overrides
        if overrides["batch_size"]:
            self.config.setdefault("performance", {})["batch_size"] = overrides["batch_size"]
        
        if not overrides["async_enabled"]:
            self.config.setdefault("performance", {})["enable_async"] = False
    
    def _show_startup_info(self):
        """Display startup information"""
        if not self.args.quiet:
            print("\n🤖 RAG Chat Agent v2.0")
            print("=" * 60)
            
            # Show configuration summary
            print(f"📁 Data path: {self.config.get('data_path', './data')}")
            print(f"🗄️  Vector DB: {self.config.get('vector_db_path', './vector_db')}")
            print(f"💾 Snapshots: {self.config.get('snapshot_path', './snapshots')}")
            
            # Show model info
            model_profile = self.config.get("models", {}).get("default_profile", "default")
            print(f"🧠 Model profile: {model_profile}")
            
            # Show feature status
            features = []
            if self.config.get("performance", {}).get("enable_async", True):
                features.append("async")
            if self.config.get("retrieval", {}).get("search_type", "hybrid") == "hybrid":
                features.append("hybrid search")
            if self.args.monitoring:
                features.append("monitoring")
            
            if features:
                print(f"✨ Features: {', '.join(features)}")
            
            print("=" * 60 + "\n")
    
    @handle_errors(logger=None, raise_on_error=True)
    async def run_async(self):
        """Run the application asynchronously"""
        try:
            # Run health checks
            if not self.args.skip_health_check:
                await self._run_health_checks()
            
            # Initialize components
            await self._initialize_components()
            
            # Start monitoring dashboard if requested
            if self.args.monitoring:
                self.dashboard = MonitoringDashboard(self.metrics, self.health_checker)
                asyncio.create_task(self._update_dashboard())
            
            # Run main chat loop
            await self._run_chat_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise
        finally:
            await self._cleanup()
    
    def run(self):
        """Run the application (sync wrapper)"""
        if self.config.get("performance", {}).get("enable_async", True) and not self.args.no_async:
            asyncio.run(self.run_async())
        else:
            self._run_sync()
    
    def _run_sync(self):
        """Run the application synchronously"""
        try:
            # Initialize components
            self._initialize_components_sync()
            
            # Run chat loop
            self._run_chat_loop_sync()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise
        finally:
            self._cleanup_sync()
    
    async def _run_health_checks(self):
        """Run system health checks"""
        self.logger.info("Running health checks...")
        
        health_status = await self.health_checker.run_health_checks()
        
        unhealthy = [
            name for name, status in health_status.items() 
            if status["status"] != "healthy"
        ]
        
        if unhealthy:
            self.logger.warning(f"Health check warnings: {', '.join(unhealthy)}")
            if not self.args.ignore_warnings:
                response = input("⚠️  Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    sys.exit(1)
        else:
            self.logger.info("All health checks passed")
    
    async def _initialize_components(self):
        """Initialize application components asynchronously"""
        self.logger.info("Initializing components...")
        
        # Load documents
        if not self.args.skip_update:
            print("📄 Loading documents...")
            self.chunks = await asyncio.get_event_loop().run_in_executor(
                None, load_documents, self.config
            )
        else:
            self.chunks = []
        
        # Update vectorstore
        print("🔍 Preparing vectorstore...")
        self.retriever = await asyncio.get_event_loop().run_in_executor(
            None,
            update_vectorstore,
            self.config,
            self.chunks,
            self.args.skip_update
        )
        
        # Setup LLM
        print("🧠 Loading language model...")
        self.llm = setup_llm(self.config, {})
        
        # Store in config for other components
        self.config["retriever"] = self.retriever
        self.config["llm_instance"] = self.llm
        
        self.logger.info("All components initialized")
    
    def _initialize_components_sync(self):
        """Initialize components synchronously"""
        self.logger.info("Initializing components (sync mode)...")
        
        # Load documents
        if not self.args.skip_update:
            print("📄 Loading documents...")
            self.chunks = load_documents(self.config)
        else:
            self.chunks = []
        
        # Update vectorstore
        print("🔍 Preparing vectorstore...")
        self.retriever = update_vectorstore(
            self.config,
            self.chunks,
            self.args.skip_update
        )
        
        # Setup LLM
        print("🧠 Loading language model...")
        self.llm = setup_llm(self.config, {})
        
        # Store in config
        self.config["retriever"] = self.retriever
        self.config["llm_instance"] = self.llm
    
    async def _run_chat_loop(self):
        """Run the main chat loop asynchronously"""
        # Handle session
        snap, memory = handle_session(self.config, override=self.args.session_mode)
        agent = start_session(self.config, memory)
        
        print("\n🔍 Ask questions. Type 'exit' to quit, or use commands:")
        print("   ::new     - Start new session")
        print("   ::resume  - Resume a session")
        print("   ::export  - Export current session")
        print("   ::search  - Search sessions")
        print("   ::stats   - Show statistics")
        print("   ::help    - Show all commands\n")
        
        has_activity = False
        
        while True:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "You: "
                )
                user_input = user_input.strip()
                
                # Handle commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    if has_activity:
                        snap.save_snapshot()
                        print("✅ Session saved before exit.\n")
                    else:
                        print("🗑️ No activity detected. Session discarded.\n")
                    break
                
                elif user_input.startswith("::"):
                    handled = await self._handle_command(user_input, snap, agent, has_activity)
                    if handled == "new_session":
                        snap, memory = handle_session(self.config, override="new")
                        agent = start_session(self.config, memory)
                        has_activity = False
                    elif handled == "resume_session":
                        snap, memory = handle_session(self.config, override="resume")
                        agent = start_session(self.config, memory)
                        has_activity = False
                    continue
                
                # Process query
                if user_input:
                    # Show thinking indicator
                    if not self.args.quiet:
                        print("🤔 Thinking...", end="\r")
                    
                    # Get answer
                    answer, sources = await asyncio.get_event_loop().run_in_executor(
                        None, agent.ask, user_input
                    )
                    
                    # Clear thinking indicator and show response
                    if not self.args.quiet:
                        print(" " * 20, end="\r")  # Clear the line
                    
                    print(f"\n💬 Answer:\n{answer}\n")
                    
                    if self.args.show_sources and sources:
                        print("📚 Sources:")
                        for i, s in enumerate(sources, 1):
                            print(f" {i}. {s['file']} (Page {s['page']}, Score: {s.get('relevance_score', 0):.3f})")
                            if self.args.verbose:
                                print(f"    {s['text'][:100]}...")
                        print()
                    
                    # Record turn
                    snap.record_turn(user_input, answer, sources)
                    has_activity = True
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type 'exit' to quit or continue chatting.")
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                print(f"\n❌ Error: {e}\n")
    
    def _run_chat_loop_sync(self):
        """Run the main chat loop synchronously"""
        # Similar to async version but without await
        snap, memory = handle_session(self.config, override=self.args.session_mode)
        agent = start_session(self.config, memory)
        
        print("\n🔍 Ask questions. Type 'exit' to quit.\n")
        
        has_activity = False
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    if has_activity:
                        snap.save_snapshot()
                        print("✅ Session saved before exit.\n")
                    break
                
                elif user_input.startswith("::"):
                    # Handle commands (simplified for sync version)
                    if user_input == "::new":
                        if has_activity:
                            snap.save_snapshot()
                        snap, memory = handle_session(self.config, override="new")
                        agent = start_session(self.config, memory)
                        has_activity = False
                        continue
                
                # Process query
                if user_input:
                    answer, sources = agent.ask(user_input)
                    print(f"\n💬 Answer:\n{answer}\n")
                    
                    if self.args.show_sources and sources:
                        print("📚 Sources:")
                        for s in sources[:3]:
                            print(f" - {s['file']} (Page {s['page']})")
                    
                    snap.record_turn(user_input, answer, sources)
                    has_activity = True
                    
            except KeyboardInterrupt:
                print("\n⚠️  Use 'exit' to quit.")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    async def _handle_command(
        self, 
        command: str, 
        snap, 
        agent, 
        has_activity: bool
    ) -> Optional[str]:
        """Handle special commands"""
        cmd = command.lower().strip()
        
        if cmd == "::help":
            print("\n📖 Available Commands:")
            print("  ::new     - Start a new session")
            print("  ::resume  - Resume an existing session")
            print("  ::export  - Export current session (json/txt/md)")
            print("  ::search  - Search through all sessions")
            print("  ::stats   - Show session statistics")
            print("  ::list    - List recent sessions")
            print("  ::model   - Show/change model info")
            print("  ::health  - Run health checks")
            print("  ::monitor - Toggle monitoring display")
            print("  ::help    - Show this help message\n")
            
        elif cmd == "::stats":
            stats = snap.get_session_statistics()
            print("\n📊 Session Statistics:")
            print(f"  Total sessions: {stats['total_sessions']}")
            print(f"  Total conversations: {stats['total_turns']}")
            print(f"  Total tokens used: {stats['total_tokens']:,}")
            print(f"  Storage used: {stats['total_size_mb']:.2f}MB")
            
            if stats['models_used']:
                print("\n  Models used:")
                for model, count in stats['models_used'].items():
                    print(f"    - {model}: {count} sessions")
            print()
            
        elif cmd == "::list":
            sessions = snap.list_sessions(limit=10)
            if sessions:
                print("\n📋 Recent Sessions:")
                for i, session in enumerate(sessions, 1):
                    print(f"\n  {i}. {session['alias']}")
                    print(f"     Created: {session['created'].strftime('%Y-%m-%d %H:%M')}")
                    print(f"     Turns: {session['turns']}, Size: {session['size_mb']:.2f}MB")
                    print(f"     Preview: {session['first_msg'][:60]}...")
            else:
                print("\n  No sessions found.")
            print()
            
        elif cmd.startswith("::search"):
            query = command[8:].strip()
            if not query:
                query = input("Search query: ").strip()
            
            if query:
                results = snap.search_sessions(query)
                if results:
                    print(f"\n🔍 Search results for '{query}':")
                    for result in results[:5]:
                        print(f"\n  📁 {result['alias']} ({result['match_count']} matches)")
                        print(f"     Created: {result['created'].strftime('%Y-%m-%d %H:%M')}")
                        for match in result['matches']:
                            print(f"     Q: {match['question']}")
                else:
                    print(f"\n  No results found for '{query}'")
            print()
            
        elif cmd.startswith("::export"):
            parts = command.split()
            format = parts[1] if len(parts) > 1 else "json"
            
            if format not in ["json", "txt", "md"]:
                print("  ❌ Invalid format. Use: json, txt, or md")
            else:
                path = snap.export_session(snap.session_id, format=format)
                if path:
                    print(f"  ✅ Session exported to: {path}")
                else:
                    print("  ❌ Export failed")
            print()
            
        elif cmd == "::model":
            from get_llm import ModelManager
            manager = ModelManager(self.config)
            info = manager.get_model_info()
            
            print("\n🧠 Model Information:")
            current = self.config.get("models", {}).get("default_profile", "default")
            print(f"  Current profile: {current}")
            
            print("\n  Available profiles:")
            for name, details in info.items():
                print(f"    - {name}: {details['provider']} / {details['model']}")
                if details.get('cost_per_1k_tokens', 0) > 0:
                    print(f"      Cost: ${details['cost_per_1k_tokens']}/1k tokens")
            print()
            
        elif cmd == "::health":
            health_status = await self.health_checker.run_health_checks()
            print("\n🏥 Health Status:")
            for check, status in health_status.items():
                icon = "✅" if status["status"] == "healthy" else "❌"
                print(f"  {icon} {check}: {status['status']}")
                if status.get("warning"):
                    print(f"     ⚠️  {status['warning']}")
            print()
            
        elif cmd == "::monitor":
            self.args.monitoring = not self.args.monitoring
            status = "enabled" if self.args.monitoring else "disabled"
            print(f"\n  📊 Monitoring {status}\n")
            
        elif cmd == "::new":
            if has_activity:
                snap.save_snapshot()
                print("💾 Previous session saved.\n")
            return "new_session"
            
        elif cmd == "::resume":
            if has_activity:
                snap.save_snapshot()
                print("💾 Previous session saved.\n")
            return "resume_session"
            
        else:
            print(f"\n  ❓ Unknown command: {command}")
            print("  Type ::help for available commands\n")
        
        return None
    
    async def _update_dashboard(self):
        """Periodically update monitoring dashboard"""
        while self.args.monitoring:
            try:
                # Clear screen and show dashboard
                print("\033[2J\033[H")  # Clear screen
                self.dashboard.display_summary()
                
                # Wait before next update
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        
        # Save any pending metrics
        if hasattr(self.metrics, 'stop'):
            self.metrics.stop()
        
        # Persist performance metrics
        perf_metrics = performance_monitor.get_metrics()
        if perf_metrics:
            self.logger.info("Performance summary:")
            for op, stats in perf_metrics.items():
                self.logger.info(
                    f"  {op}: {stats['count']} calls, "
                    f"avg={stats['average']:.2f}s, "
                    f"total={stats['total']:.2f}s"
                )
    
    def _cleanup_sync(self):
        """Cleanup resources (sync version)"""
        self.logger.info("Cleaning up resources...")
        
        if hasattr(self.metrics, 'stop'):
            self.metrics.stop()


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="RAG Chat Interface - Enhanced with hybrid search and multi-model support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings
  python main.py
  
  # Use a specific model profile
  python main.py --model quality
  
  # Start a new session with custom temperature
  python main.py --new --temperature 0.5
  
  # Resume latest session with monitoring
  python main.py --resume --monitoring
  
  # Use different config file
  python main.py --config production.yaml
        """
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model", 
        type=str, 
        help="Model profile to use (fast/balanced/quality)"
    )
    model_group.add_argument(
        "--model_path", 
        type=str, 
        help="Override local model path"
    )
    model_group.add_argument(
        "--temperature", 
        type=float, 
        help="Override LLM temperature (0.0-1.0)"
    )
    
    # Session arguments
    session_group = parser.add_argument_group("Session Management")
    session_group.add_argument(
        "--new", 
        action="store_const", 
        const="new", 
        dest="session_mode",
        help="Start a new session"
    )
    session_group.add_argument(
        "--resume", 
        action="store_const", 
        const="resume", 
        dest="session_mode",
        help="Resume existing session"
    )
    session_group.add_argument(
        "--session", 
        type=str, 
        help="Resume specific session by ID or alias"
    )
    
    # Performance arguments
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--batch_size", 
        type=int, 
        help="Override batch size for document processing"
    )
    perf_group.add_argument(
        "--no_async", 
        action="store_true", 
        help="Disable async operations"
    )
    perf_group.add_argument(
        "--skip_update", 
        action="store_true", 
        help="Skip vectorstore update"
    )
    
    # Display arguments
    display_group = parser.add_argument_group("Display Options")
    display_group.add_argument(
        "--quiet", "-q", 
        action="store_true", 
        help="Minimal output"
    )
    display_group.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    display_group.add_argument(
        "--show_sources", "-s", 
        action="store_true", 
        help="Show source documents"
    )
    display_group.add_argument(
        "--monitoring", "-m", 
        action="store_true", 
        help="Enable monitoring dashboard"
    )
    
    # System arguments
    system_group = parser.add_argument_group("System Options")
    system_group.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Configuration file path"
    )
    system_group.add_argument(
        "--log_level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        help="Console log level"
    )
    system_group.add_argument(
        "--no_env_override", 
        action="store_true", 
        help="Disable environment variable overrides"
    )
    system_group.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    system_group.add_argument(
        "--skip_health_check", 
        action="store_true", 
        help="Skip startup health checks"
    )
    system_group.add_argument(
        "--ignore_warnings", 
        action="store_true", 
        help="Continue despite health check warnings"
    )
    
    return parser


def main():
    """Main entry point"""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure warnings
    if not args.debug:
        warnings.filterwarnings("ignore")
    
    # Create and run CLI
    try:
        cli = RAGChatCLI(args)
        cli.run()
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"\n❌ Fatal error: {e}")
            print("\nRun with --debug for full traceback")
        sys.exit(1)


if __name__ == "__main__":
    main()