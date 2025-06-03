"""
Comprehensive monitoring system for the RAG Chat Application
Tracks performance metrics, system health, and usage statistics
"""

import time
import json
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict
import statistics

from utils import setup_logging, ensure_directory


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    query_id: str
    query_text: str
    start_time: datetime
    end_time: Optional[datetime] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    total_time: Optional[float] = None
    tokens_used: Optional[int] = None
    chunks_retrieved: Optional[int] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        return self.end_time is not None
    
    def complete(self):
        """Mark query as complete"""
        self.end_time = datetime.now()
        self.total_time = (self.end_time - self.start_time).total_seconds()


class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, retention_hours: int = 24, persist_path: Optional[str] = None):
        self.retention_hours = retention_hours
        self.persist_path = Path(persist_path) if persist_path else None
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.query_metrics: Dict[str, QueryMetrics] = {}
        
        # System metrics
        self.system_metrics_interval = 30  # seconds
        self.system_metrics_thread = None
        self.running = False
        
        # Setup logging
        self.logger = setup_logging("rag_chat.MetricsCollector")
        
        # Ensure persistence directory
        if self.persist_path:
            ensure_directory(self.persist_path.parent)
        
        # Load existing metrics
        self._load_metrics()

    def start(self):
        """Start metrics collection"""
        self.running = True
        self.system_metrics_thread = threading.Thread(
            target=self._collect_system_metrics,
            daemon=True
        )
        self.system_metrics_thread.start()
        self.logger.info("Metrics collection started")

    def stop(self):
        """Stop metrics collection"""
        self.running = False
        if self.system_metrics_thread:
            self.system_metrics_thread.join()
        self._persist_metrics()
        self.logger.info("Metrics collection stopped")

    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a single metric value"""
        metric = MetricPoint(datetime.now(), value, tags or {})
        self.metrics[name].append(metric)
        self._cleanup_old_metrics(name)

    def start_query(self, query_id: str, query_text: str, model: str = None) -> QueryMetrics:
        """Start tracking a new query"""
        metrics = QueryMetrics(
            query_id=query_id,
            query_text=query_text[:100],  # Truncate for storage
            start_time=datetime.now(),
            model_used=model
        )
        self.query_metrics[query_id] = metrics
        return metrics

    def end_query(self, query_id: str, **kwargs):
        """Complete query tracking"""
        if query_id in self.query_metrics:
            metrics = self.query_metrics[query_id]
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            metrics.complete()

    def get_metrics_summary(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m.value for m in self.metrics.get(metric_name, [])
            if m.timestamp > cutoff
        ]
        
        if not recent_metrics:
            return {"count": 0}
        
        return {
            "count": len(recent_metrics),
            "mean": statistics.mean(recent_metrics),
            "median": statistics.median(recent_metrics),
            "std_dev": statistics.stdev(recent_metrics) if len(recent_metrics) > 1 else 0,
            "min": min(recent_metrics),
            "max": max(recent_metrics),
            "p95": statistics.quantiles(recent_metrics, n=20)[18] if len(recent_metrics) > 20 else max(recent_metrics),
            "p99": statistics.quantiles(recent_metrics, n=100)[98] if len(recent_metrics) > 100 else max(recent_metrics)
        }

    def get_query_analytics(self, hours: int = 1) -> Dict[str, Any]:
        """Get analytics on recent queries"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_queries = [
            q for q in self.query_metrics.values()
            if q.start_time > cutoff and q.is_complete
        ]
        
        if not recent_queries:
            return {"total_queries": 0}
        
        # Calculate statistics
        response_times = [q.total_time for q in recent_queries if q.total_time]
        retrieval_times = [q.retrieval_time for q in recent_queries if q.retrieval_time]
        generation_times = [q.generation_time for q in recent_queries if q.generation_time]
        tokens_used = [q.tokens_used for q in recent_queries if q.tokens_used]
        
        # Error analysis
        errors = [q for q in recent_queries if q.error]
        error_rate = len(errors) / len(recent_queries) * 100
        
        # Model usage
        model_usage = defaultdict(int)
        for q in recent_queries:
            if q.model_used:
                model_usage[q.model_used] += 1
        
        return {
            "total_queries": len(recent_queries),
            "successful_queries": len(recent_queries) - len(errors),
            "error_rate": error_rate,
            "response_time": self._calculate_stats(response_times),
            "retrieval_time": self._calculate_stats(retrieval_times),
            "generation_time": self._calculate_stats(generation_times),
            "tokens_per_query": self._calculate_stats(tokens_used),
            "model_usage": dict(model_usage),
            "common_errors": self._get_common_errors(errors)
        }

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values"""
        if not values:
            return {}
        
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "p95": statistics.quantiles(values, n=20)[18] if len(values) > 20 else max(values)
        }

    def _get_common_errors(self, error_queries: List[QueryMetrics]) -> List[Dict[str, Any]]:
        """Analyze common error patterns"""
        error_counts = defaultdict(int)
        for q in error_queries:
            if q.error:
                # Simplify error for grouping
                error_type = q.error.split(':')[0].strip()
                error_counts[error_type] += 1
        
        return [
            {"error": error, "count": count}
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

    def _collect_system_metrics(self):
        """Background thread to collect system metrics"""
        while self.running:
            try:
                # CPU metrics
                self.record_metric("system.cpu.percent", psutil.cpu_percent(interval=1))
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_metric("system.memory.percent", memory.percent)
                self.record_metric("system.memory.used_gb", memory.used / (1024**3))
                self.record_metric("system.memory.available_gb", memory.available / (1024**3))
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.record_metric("system.disk.percent", disk.percent)
                self.record_metric("system.disk.free_gb", disk.free / (1024**3))
                
                # Process-specific metrics
                process = psutil.Process()
                self.record_metric("process.cpu_percent", process.cpu_percent())
                self.record_metric("process.memory_mb", process.memory_info().rss / (1024**2))
                self.record_metric("process.threads", process.num_threads())
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
            
            # Wait for next collection
            time.sleep(self.system_metrics_interval)

    def _cleanup_old_metrics(self, metric_name: str):
        """Remove metrics older than retention period"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        metrics = self.metrics[metric_name]
        
        # Remove old metrics from the left
        while metrics and metrics[0].timestamp < cutoff:
            metrics.popleft()

    def _persist_metrics(self):
        """Save metrics to disk"""
        if not self.persist_path:
            return
        
        try:
            data = {
                "metrics": {
                    name: [
                        {
                            "timestamp": m.timestamp.isoformat(),
                            "value": m.value,
                            "tags": m.tags
                        }
                        for m in metrics
                    ]
                    for name, metrics in self.metrics.items()
                },
                "query_metrics": {
                    qid: asdict(qm) for qid, qm in self.query_metrics.items()
                }
            }
            
            # Convert datetime objects
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, default=convert_datetime, indent=2)
            
            self.logger.debug(f"Persisted metrics to {self.persist_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to persist metrics: {e}")

    def _load_metrics(self):
        """Load metrics from disk"""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            
            # Restore metrics
            for name, metric_list in data.get("metrics", {}).items():
                for m in metric_list:
                    metric = MetricPoint(
                        timestamp=datetime.fromisoformat(m["timestamp"]),
                        value=m["value"],
                        tags=m.get("tags", {})
                    )
                    self.metrics[name].append(metric)
            
            # Restore query metrics
            for qid, qm_data in data.get("query_metrics", {}).items():
                # Convert ISO strings back to datetime
                for field in ["start_time", "end_time"]:
                    if qm_data.get(field):
                        qm_data[field] = datetime.fromisoformat(qm_data[field])
                
                self.query_metrics[qid] = QueryMetrics(**qm_data)
            
            self.logger.info(f"Loaded metrics from {self.persist_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")


class HealthChecker:
    """System health monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging("rag_chat.HealthChecker")
        
        # Health check functions
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        
        # Register default health checks
        self._register_default_checks()

    def _register_default_checks(self):
        """Register built-in health checks"""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory_usage", self._check_memory_usage)

    def register_check(self, name: str, check_func: Callable):
        """Register a new health check"""
        self.health_checks[name] = check_func

    async def run_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = {
                    "status": "healthy" if result.get("healthy", True) else "unhealthy",
                    "timestamp": datetime.now().isoformat(),
                    **result
                }
            except Exception as e:
                self.logger.error(f"Health check '{name}' failed: {e}")
                results[name] = {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
        
        self.health_status = results
        return results

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        healthy = cpu_percent < 90 and memory.percent < 90
        
        return {
            "healthy": healthy,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "warning": "High resource usage" if not healthy else None
        }

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        disk = psutil.disk_usage('/')
        min_free_gb = self.config.get("min_disk_free_gb", 1.0)
        free_gb = disk.free / (1024**3)
        
        healthy = free_gb > min_free_gb
        
        return {
            "healthy": healthy,
            "free_gb": free_gb,
            "used_percent": disk.percent,
            "warning": f"Low disk space: {free_gb:.2f}GB free" if not healthy else None
        }

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check process memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024**2)
        max_memory_mb = self.config.get("max_process_memory_mb", 4096)
        
        healthy = memory_mb < max_memory_mb
        
        return {
            "healthy": healthy,
            "process_memory_mb": memory_mb,
            "warning": f"High process memory: {memory_mb:.0f}MB" if not healthy else None
        }


class MonitoringDashboard:
    """Simple console dashboard for monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector, health_checker: HealthChecker):
        self.metrics = metrics_collector
        self.health = health_checker
        self.logger = setup_logging("rag_chat.MonitoringDashboard")

    def display_summary(self):
        """Display monitoring summary to console"""
        print("\n" + "="*60)
        print("RAG Chat Monitoring Dashboard")
        print("="*60)
        
        # System metrics
        sys_metrics = {
            "CPU": self.metrics.get_metrics_summary("system.cpu.percent", hours=0.1),
            "Memory": self.metrics.get_metrics_summary("system.memory.percent", hours=0.1),
            "Process Memory": self.metrics.get_metrics_summary("process.memory_mb", hours=0.1)
        }
        
        print("\n📊 System Metrics (last 6 minutes):")
        for name, stats in sys_metrics.items():
            if stats.get("count", 0) > 0:
                print(f"  {name}: {stats['mean']:.1f} (min: {stats['min']:.1f}, max: {stats['max']:.1f})")
        
        # Query analytics
        query_stats = self.metrics.get_query_analytics(hours=1)
        if query_stats["total_queries"] > 0:
            print(f"\n📈 Query Performance (last hour):")
            print(f"  Total queries: {query_stats['total_queries']}")
            print(f"  Success rate: {100 - query_stats['error_rate']:.1f}%")
            
            if query_stats.get("response_time"):
                rt = query_stats["response_time"]
                print(f"  Response time: {rt['mean']:.2f}s (p95: {rt['p95']:.2f}s)")
        
        # Health status
        health_status = asyncio.run(self.health.run_health_checks())
        print(f"\n🏥 Health Status:")
        for check, status in health_status.items():
            icon = "✅" if status["status"] == "healthy" else "❌"
            print(f"  {icon} {check}: {status['status']}")
            if status.get("warning"):
                print(f"     ⚠️  {status['warning']}")
        
        print("="*60)


# Global monitoring instance
_monitoring_instance = None


def get_monitoring_instance(config: Dict[str, Any] = None) -> MetricsCollector:
    """Get or create global monitoring instance"""
    global _monitoring_instance
    
    if _monitoring_instance is None:
        if config is None:
            raise ValueError("Config required for first monitoring initialization")
        
        persist_path = config.get("monitoring", {}).get("persist_path", "./metrics/metrics.json")
        _monitoring_instance = MetricsCollector(
            retention_hours=config.get("monitoring", {}).get("retention_hours", 24),
            persist_path=persist_path
        )
        _monitoring_instance.start()
    
    return _monitoring_instance


# Example usage
if __name__ == "__main__":
    from utils import load_config
    
    # Load config
    config = load_config("config.yaml")
    
    # Create monitoring components
    metrics = MetricsCollector(persist_path="./metrics/metrics.json")
    metrics.start()
    
    health = HealthChecker(config)
    
    # Create dashboard
    dashboard = MonitoringDashboard(metrics, health)
    
    try:
        # Simulate some metrics
        import random
        import uuid
        
        for _ in range(5):
            # Simulate a query
            query_id = str(uuid.uuid4())
            query_metrics = metrics.start_query(query_id, "Sample query", "gpt-3.5")
            
            # Simulate processing
            time.sleep(random.uniform(0.1, 0.5))
            
            # Complete query
            metrics.end_query(
                query_id,
                retrieval_time=random.uniform(0.1, 0.3),
                generation_time=random.uniform(0.5, 1.0),
                tokens_used=random.randint(100, 500),
                chunks_retrieved=random.randint(3, 10)
            )
        
        # Display dashboard
        dashboard.display_summary()
        
    finally:
        metrics.stop()