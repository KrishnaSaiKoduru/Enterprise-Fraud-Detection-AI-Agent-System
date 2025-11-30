"""
Enterprise AI Fraud Detection Agent System

Main orchestration system that coordinates the Detector, Enrichment, and Analyst agents.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from .agents import create_detector_agent, create_enrichment_agent, create_analyst_agent
from .models import FraudDetectionSessionManager, get_observability


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionSystem:
    """
    Main fraud detection system orchestrating multiple agents.

    Architecture:
    1. Detector Agent → Detects fraud using rules + ML
    2. Enrichment Agent → Enriches flagged transactions with context
    3. Analyst Agent → Generates investigation briefs using Gemini
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        enable_debug: bool = False,
        auto_enrich_threshold: float = 0.5,
        auto_brief_threshold: float = 0.7
    ):
        """
        Initialize the fraud detection system.

        Args:
            model_name: Gemini model to use for agents
            enable_debug: Enable debug logging
            auto_enrich_threshold: Fraud score threshold to trigger enrichment
            auto_brief_threshold: Fraud score threshold to trigger investigation brief
        """
        self.model_name = model_name
        self.auto_enrich_threshold = auto_enrich_threshold
        self.auto_brief_threshold = auto_brief_threshold

        # Initialize observability
        self.observability = get_observability(enable_debug=enable_debug)

        # Initialize session manager
        self.session_manager = FraudDetectionSessionManager()

        # Initialize agents
        logger.info("Initializing fraud detection agents...")
        self.detector = create_detector_agent(model_name=model_name)
        self.enrichment = create_enrichment_agent(model_name=model_name)
        self.analyst = create_analyst_agent(model_name=model_name)

        logger.info("Fraud Detection System initialized successfully")

    def process_transaction(
        self,
        transaction: Dict[str, Any],
        auto_enrich: bool = True,
        auto_brief: bool = True
    ) -> Dict[str, Any]:
        """
        Process a transaction through the complete fraud detection pipeline.

        Args:
            transaction: Transaction data
            auto_enrich: Automatically enrich if fraud score exceeds threshold
            auto_brief: Automatically generate investigation brief if needed

        Returns:
            Complete processing result including detection, enrichment, and analysis
        """
        start_time = time.time()
        customer_id = transaction.get("customer_id")
        transaction_id = transaction.get("transaction_id")

        logger.info(f"Processing transaction {transaction_id} for customer {customer_id}")

        result = {
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "processing_timestamp": datetime.now().isoformat(),
            "stages": {}
        }

        # Get customer session and history
        session = self.session_manager.get_or_create_session(customer_id)
        customer_history = self.session_manager.get_customer_history(customer_id)

        # Stage 1: Fraud Detection
        detection_result = self._run_detection(transaction, customer_history, session)
        result["stages"]["detection"] = detection_result

        fraud_score = detection_result.get("combined_assessment", {}).get("combined_score", 0)
        is_fraud = fraud_score >= 0.5

        # Log transaction
        self.observability.log_transaction_processed(
            transaction_id, customer_id, fraud_score, is_fraud
        )

        # Add to session history
        self.session_manager.add_transaction(customer_id, transaction)

        # Stage 2: Enrichment (if threshold exceeded)
        if auto_enrich and fraud_score >= self.auto_enrich_threshold:
            enrichment_result = self._run_enrichment(transaction, detection_result, session)
            result["stages"]["enrichment"] = enrichment_result

            # Stage 3: Investigation Brief (if threshold exceeded)
            if auto_brief and fraud_score >= self.auto_brief_threshold:
                brief_result = self._run_analysis(enrichment_result, session)
                result["stages"]["investigation_brief"] = brief_result

                # Add fraud alert to session
                alert = {
                    "transaction_id": transaction_id,
                    "fraud_score": fraud_score,
                    "priority": brief_result.get("investigation_brief", {}).get("priority"),
                    "timestamp": datetime.now()
                }
                self.session_manager.add_fraud_alert(customer_id, alert)

        # Calculate total processing time
        processing_time_ms = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time_ms

        # Add summary
        result["summary"] = {
            "fraud_detected": is_fraud,
            "fraud_score": fraud_score,
            "priority": detection_result.get("combined_assessment", {}).get("priority", "low"),
            "stages_completed": list(result["stages"].keys()),
            "processing_time_ms": processing_time_ms
        }

        logger.info(
            f"Transaction {transaction_id} processed in {processing_time_ms:.2f}ms: "
            f"fraud_score={fraud_score:.3f}, stages={list(result['stages'].keys())}"
        )

        return result

    def _run_detection(
        self,
        transaction: Dict[str, Any],
        customer_history: List[Dict[str, Any]],
        session
    ) -> Dict[str, Any]:
        """Run the Detector Agent."""
        start_time = time.time()

        try:
            result = self.detector.detect_fraud(transaction, customer_history, session)

            # Log triggered rules
            for rule in result.get("rule_analysis", {}).get("triggered_rules", []):
                self.observability.log_rule_triggered(
                    rule["rule"],
                    transaction.get("transaction_id")
                )

            # Log ML prediction
            ml_score = result.get("ml_analysis", {}).get("ml_score", 0)
            self.observability.log_ml_prediction(
                transaction.get("transaction_id"),
                ml_score,
                10.0  # Mock inference time
            )

            duration_ms = (time.time() - start_time) * 1000
            self.observability.log_agent_execution(
                "FraudDetectorAgent",
                transaction.get("transaction_id"),
                duration_ms,
                success=True
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.observability.log_agent_execution(
                "FraudDetectorAgent",
                transaction.get("transaction_id"),
                duration_ms,
                success=False
            )
            self.observability.log_error(
                "FraudDetectorAgent",
                str(e),
                transaction.get("transaction_id")
            )
            raise

    def _run_enrichment(
        self,
        transaction: Dict[str, Any],
        fraud_detection_result: Dict[str, Any],
        session
    ) -> Dict[str, Any]:
        """Run the Enrichment Agent."""
        start_time = time.time()

        try:
            result = self.enrichment.enrich_transaction(
                transaction,
                fraud_detection_result,
                session
            )

            duration_ms = (time.time() - start_time) * 1000
            sources_queried = len(result.get("enrichment_quality", {}).get("sources_queried", []))

            self.observability.log_enrichment(
                transaction.get("transaction_id"),
                sources_queried,
                duration_ms
            )

            self.observability.log_agent_execution(
                "EnrichmentAgent",
                transaction.get("transaction_id"),
                duration_ms,
                success=True
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.observability.log_agent_execution(
                "EnrichmentAgent",
                transaction.get("transaction_id"),
                duration_ms,
                success=False
            )
            self.observability.log_error(
                "EnrichmentAgent",
                str(e),
                transaction.get("transaction_id")
            )
            raise

    def _run_analysis(self, enriched_event: Dict[str, Any], session) -> Dict[str, Any]:
        """Run the Analyst Agent."""
        start_time = time.time()

        try:
            result = self.analyst.analyze_event(enriched_event, session)

            duration_ms = (time.time() - start_time) * 1000
            case_id = result.get("investigation_brief", {}).get("case_id")
            priority = result.get("investigation_brief", {}).get("priority", "medium")

            self.observability.log_investigation_brief(case_id, priority, duration_ms)

            self.observability.log_agent_execution(
                "AnalystAgent",
                enriched_event.get("transaction_id"),
                duration_ms,
                success=True
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.observability.log_agent_execution(
                "AnalystAgent",
                enriched_event.get("transaction_id"),
                duration_ms,
                success=False
            )
            self.observability.log_error(
                "AnalystAgent",
                str(e),
                enriched_event.get("transaction_id")
            )
            raise

    def process_batch(
        self,
        transactions: List[Dict[str, Any]],
        auto_enrich: bool = True,
        auto_brief: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of transactions.

        Args:
            transactions: List of transactions to process
            auto_enrich: Auto-enrich flagged transactions
            auto_brief: Auto-generate investigation briefs

        Returns:
            List of processing results
        """
        logger.info(f"Processing batch of {len(transactions)} transactions")
        results = []

        for transaction in transactions:
            try:
                result = self.process_transaction(transaction, auto_enrich, auto_brief)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing transaction {transaction.get('transaction_id')}: {e}")
                results.append({
                    "transaction_id": transaction.get("transaction_id"),
                    "error": str(e),
                    "status": "failed"
                })

        logger.info(f"Batch processing complete: {len(results)} transactions processed")
        return results

    def get_customer_session_state(self, customer_id: str) -> Dict[str, Any]:
        """
        Get current session state for a customer.

        Args:
            customer_id: Customer identifier

        Returns:
            Session state dictionary
        """
        return self.session_manager.get_session_state(customer_id)

    def get_metrics_report(self) -> Dict[str, Any]:
        """
        Get system metrics report.

        Returns:
            Metrics dictionary
        """
        metrics = self.observability.get_metrics_report()
        session_stats = self.session_manager.get_statistics()

        return {
            "metrics": metrics,
            "session_statistics": session_stats,
            "timestamp": datetime.now().isoformat()
        }

    def print_metrics(self):
        """Print formatted metrics report."""
        self.observability.print_metrics_report()

        print("\n" + "="*60)
        print("SESSION STATISTICS")
        print("="*60)

        stats = self.session_manager.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print("="*60 + "\n")

    def cleanup(self):
        """Cleanup old sessions and resources."""
        self.session_manager.cleanup_old_sessions()
        logger.info("System cleanup completed")


def create_fraud_detection_system(
    model_name: str = "gemini-2.0-flash-exp",
    enable_debug: bool = False
) -> FraudDetectionSystem:
    """
    Factory function to create a fraud detection system.

    Args:
        model_name: Gemini model to use
        enable_debug: Enable debug logging

    Returns:
        Initialized FraudDetectionSystem
    """
    return FraudDetectionSystem(model_name=model_name, enable_debug=enable_debug)
