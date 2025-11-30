"""
Observability Module for Fraud Detection Agent System

Provides logging, metrics, and tracing capabilities.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
from contextlib import contextmanager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and tracks metrics for the fraud detection system.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)
        self.start_time = datetime.now()

    def increment_counter(self, metric_name: str, value: int = 1, tags: Dict[str, str] = None):
        """
        Increment a counter metric.

        Args:
            metric_name: Name of the metric
            value: Value to increment by
            tags: Optional tags for the metric
        """
        key = self._make_key(metric_name, tags)
        self.counters[key] += value
        logger.debug(f"Counter {key}: {self.counters[key]}")

    def set_gauge(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """
        Set a gauge metric.

        Args:
            metric_name: Name of the metric
            value: Value to set
            tags: Optional tags for the metric
        """
        key = self._make_key(metric_name, tags)
        self.gauges[key] = value
        logger.debug(f"Gauge {key}: {value}")

    def record_histogram(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """
        Record a histogram value.

        Args:
            metric_name: Name of the metric
            value: Value to record
            tags: Optional tags for the metric
        """
        key = self._make_key(metric_name, tags)
        self.histograms[key].append(value)
        logger.debug(f"Histogram {key}: recorded {value}")

    def record_timer(self, metric_name: str, duration_ms: float, tags: Dict[str, str] = None):
        """
        Record a timer metric.

        Args:
            metric_name: Name of the metric
            duration_ms: Duration in milliseconds
            tags: Optional tags for the metric
        """
        key = self._make_key(metric_name, tags)
        self.timers[key].append(duration_ms)
        logger.debug(f"Timer {key}: {duration_ms:.2f}ms")

    @contextmanager
    def time_operation(self, metric_name: str, tags: Dict[str, str] = None):
        """
        Context manager to time an operation.

        Args:
            metric_name: Name of the metric
            tags: Optional tags for the metric

        Example:
            with metrics.time_operation("fraud_detection"):
                detect_fraud(transaction)
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_timer(metric_name, duration_ms, tags)

    def _make_key(self, metric_name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key for the metric with tags."""
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{metric_name}[{tag_str}]"
        return metric_name

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary with metric summaries
        """
        summary = {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                key: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "mean": sum(values) / len(values) if values else 0
                }
                for key, values in self.histograms.items()
            },
            "timers": {
                key: {
                    "count": len(durations),
                    "min_ms": min(durations) if durations else 0,
                    "max_ms": max(durations) if durations else 0,
                    "mean_ms": sum(durations) / len(durations) if durations else 0
                }
                for key, durations in self.timers.items()
            }
        }
        return summary

    def reset_metrics(self):
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timers.clear()
        self.start_time = datetime.now()
        logger.info("Metrics reset")


class FraudDetectionObservability:
    """
    Observability system for fraud detection agents.
    """

    def __init__(self, enable_debug: bool = False):
        """
        Initialize observability system.

        Args:
            enable_debug: Enable debug-level logging
        """
        self.metrics = MetricsCollector()

        # Configure logging level
        if enable_debug:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        logger.info("Fraud Detection Observability initialized")

    def log_transaction_processed(
        self,
        transaction_id: str,
        customer_id: str,
        fraud_score: float,
        is_fraud: bool
    ):
        """Log a processed transaction."""
        self.metrics.increment_counter("transactions_processed")

        if is_fraud:
            self.metrics.increment_counter("fraud_detected")
            logger.warning(
                f"FRAUD DETECTED: Transaction {transaction_id} "
                f"(Customer: {customer_id}, Score: {fraud_score:.3f})"
            )
        else:
            logger.info(
                f"Transaction {transaction_id} processed "
                f"(Customer: {customer_id}, Score: {fraud_score:.3f})"
            )

        self.metrics.record_histogram("fraud_scores", fraud_score)

    def log_agent_execution(
        self,
        agent_name: str,
        transaction_id: str,
        duration_ms: float,
        success: bool = True
    ):
        """Log agent execution."""
        tags = {"agent": agent_name, "status": "success" if success else "error"}

        self.metrics.increment_counter("agent_executions", tags=tags)
        self.metrics.record_timer("agent_execution_time", duration_ms, tags={"agent": agent_name})

        if success:
            logger.info(f"{agent_name} completed for {transaction_id} in {duration_ms:.2f}ms")
        else:
            logger.error(f"{agent_name} failed for {transaction_id} after {duration_ms:.2f}ms")

    def log_rule_triggered(self, rule_name: str, transaction_id: str):
        """Log a triggered fraud rule."""
        self.metrics.increment_counter("rules_triggered", tags={"rule": rule_name})
        logger.info(f"Rule '{rule_name}' triggered for transaction {transaction_id}")

    def log_ml_prediction(
        self,
        transaction_id: str,
        ml_score: float,
        inference_time_ms: float
    ):
        """Log ML model prediction."""
        self.metrics.increment_counter("ml_predictions")
        self.metrics.record_timer("model_inference_time", inference_time_ms)
        self.metrics.record_histogram("ml_scores", ml_score)

        logger.debug(
            f"ML prediction for {transaction_id}: "
            f"score={ml_score:.3f}, inference_time={inference_time_ms:.2f}ms"
        )

    def log_enrichment(self, transaction_id: str, sources_queried: int, duration_ms: float):
        """Log enrichment activity."""
        self.metrics.increment_counter("enrichments_performed")
        self.metrics.record_timer("enrichment_time", duration_ms)
        self.metrics.record_histogram("enrichment_sources", sources_queried)

        logger.info(
            f"Enrichment for {transaction_id}: "
            f"{sources_queried} sources queried in {duration_ms:.2f}ms"
        )

    def log_investigation_brief(self, case_id: str, priority: str, duration_ms: float):
        """Log investigation brief generation."""
        self.metrics.increment_counter("investigation_briefs_generated", tags={"priority": priority})
        self.metrics.record_timer("brief_generation_time", duration_ms)

        logger.info(
            f"Investigation brief {case_id} generated: "
            f"priority={priority}, generation_time={duration_ms:.2f}ms"
        )

    def log_error(self, component: str, error_message: str, transaction_id: str = None):
        """Log an error."""
        self.metrics.increment_counter("errors", tags={"component": component})

        if transaction_id:
            logger.error(f"Error in {component} for {transaction_id}: {error_message}")
        else:
            logger.error(f"Error in {component}: {error_message}")

    def log_session_activity(
        self,
        customer_id: str,
        transaction_count: int,
        alert_count: int
    ):
        """Log session activity."""
        self.metrics.set_gauge("active_customer_sessions", 1, tags={"customer": customer_id})
        self.metrics.record_histogram("customer_transaction_count", transaction_count)
        self.metrics.record_histogram("customer_alert_count", alert_count)

        logger.debug(
            f"Session activity for customer {customer_id}: "
            f"transactions={transaction_count}, alerts={alert_count}"
        )

    def get_metrics_report(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics report.

        Returns:
            Metrics report dictionary
        """
        return self.metrics.get_metrics_summary()

    def print_metrics_report(self):
        """Print formatted metrics report."""
        report = self.get_metrics_report()

        print("\n" + "="*60)
        print("FRAUD DETECTION SYSTEM METRICS REPORT")
        print("="*60)

        print(f"\nUptime: {report['uptime_seconds']:.2f} seconds")

        print("\n--- Counters ---")
        for key, value in sorted(report['counters'].items()):
            print(f"  {key}: {value}")

        print("\n--- Gauges ---")
        for key, value in sorted(report['gauges'].items()):
            print(f"  {key}: {value:.2f}")

        print("\n--- Histograms ---")
        for key, stats in sorted(report['histograms'].items()):
            print(f"  {key}:")
            print(f"    count: {stats['count']}")
            print(f"    min: {stats['min']:.3f}")
            print(f"    max: {stats['max']:.3f}")
            print(f"    mean: {stats['mean']:.3f}")

        print("\n--- Timers ---")
        for key, stats in sorted(report['timers'].items()):
            print(f"  {key}:")
            print(f"    count: {stats['count']}")
            print(f"    min: {stats['min_ms']:.2f}ms")
            print(f"    max: {stats['max_ms']:.2f}ms")
            print(f"    mean: {stats['mean_ms']:.2f}ms")

        print("="*60 + "\n")


# Global observability instance
_global_observability = None


def get_observability(enable_debug: bool = False) -> FraudDetectionObservability:
    """
    Get global observability instance.

    Args:
        enable_debug: Enable debug logging

    Returns:
        FraudDetectionObservability instance
    """
    global _global_observability
    if _global_observability is None:
        _global_observability = FraudDetectionObservability(enable_debug=enable_debug)
    return _global_observability
