"""
Session Manager for Fraud Detection Agent System

Manages customer sessions for velocity tracking and state persistence.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd

try:
    from google.adk.sessions import InMemorySessionService, Session
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    # Provide mock classes if ADK not available
    class Session:
        def __init__(self, session_id: str):
            self.id = session_id
            self.data = {}

    class InMemorySessionService:
        def __init__(self):
            self.sessions = {}

        def get_session(self, session_id: str) -> Session:
            if session_id not in self.sessions:
                self.sessions[session_id] = Session(session_id)
            return self.sessions[session_id]


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_datetime(timestamp: Union[str, datetime]) -> datetime:
    """
    Convert timestamp to datetime object if it's a string.

    Args:
        timestamp: Timestamp as string or datetime

    Returns:
        datetime object
    """
    if isinstance(timestamp, str):
        return pd.to_datetime(timestamp)
    return timestamp


class FraudDetectionSessionManager:
    """
    Manages customer sessions for fraud detection.

    Tracks:
    - Customer transaction history (for velocity calculation)
    - Recent fraud alerts
    - Session state and metadata
    """

    def __init__(self, velocity_window_hours: int = 1, max_session_age_hours: int = 24):
        """
        Initialize the session manager.

        Args:
            velocity_window_hours: Time window for velocity tracking
            max_session_age_hours: Maximum age of session before cleanup
        """
        self.velocity_window_hours = velocity_window_hours
        self.max_session_age_hours = max_session_age_hours
        self.adk_available = ADK_AVAILABLE
        self.app_name = "fraud_detection_system"

        # Initialize session service
        if ADK_AVAILABLE:
            self.session_service = InMemorySessionService()
            logger.info("Using ADK InMemorySessionService")
        else:
            self.session_service = InMemorySessionService()  # Mock version
            logger.info("Using mock InMemorySessionService (ADK not available)")

        # Customer data storage
        self.customer_transactions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.customer_alerts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.session_metadata: Dict[str, Dict[str, Any]] = {}

    def get_or_create_session(self, customer_id: str) -> Session:
        """
        Get existing session or create new one for customer.

        Args:
            customer_id: Customer identifier

        Returns:
            Session object
        """
        session_id = f"customer_{customer_id}"
        user_id = str(customer_id)

        if self.adk_available:
            # Use ADK v1.19.0 API - use sync methods (not async)
            session = self.session_service.get_session_sync(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )

            if session is None:
                # Create new session
                session = self.session_service.create_session_sync(
                    app_name=self.app_name,
                    user_id=user_id,
                    session_id=session_id,
                    state={}
                )
        else:
            # Use mock API
            session = self.session_service.get_session(session_id)

        # Initialize metadata if new session
        if session_id not in self.session_metadata:
            self.session_metadata[session_id] = {
                "customer_id": customer_id,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "transaction_count": 0,
                "alert_count": 0
            }

        # Update last activity
        self.session_metadata[session_id]["last_activity"] = datetime.now()

        return session

    def add_transaction(self, customer_id: str, transaction: Dict[str, Any]):
        """
        Add a transaction to customer history.

        Args:
            customer_id: Customer identifier
            transaction: Transaction data
        """
        # Add timestamp if not present
        if "timestamp" not in transaction:
            transaction["timestamp"] = datetime.now()

        self.customer_transactions[customer_id].append(transaction)

        # Update session metadata
        session_id = f"customer_{customer_id}"
        if session_id in self.session_metadata:
            self.session_metadata[session_id]["transaction_count"] += 1
            self.session_metadata[session_id]["last_activity"] = datetime.now()

        # Clean old transactions
        self._cleanup_old_transactions(customer_id)

        logger.debug(f"Added transaction for customer {customer_id}. Total: {len(self.customer_transactions[customer_id])}")

    def get_customer_history(
        self,
        customer_id: str,
        window_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get customer transaction history within time window.

        Args:
            customer_id: Customer identifier
            window_hours: Time window in hours (uses velocity_window_hours if None)

        Returns:
            List of transactions within window
        """
        if window_hours is None:
            window_hours = self.velocity_window_hours

        cutoff_time = datetime.now() - timedelta(hours=window_hours)

        transactions = self.customer_transactions.get(customer_id, [])

        # Filter by time window
        recent_transactions = [
            txn for txn in transactions
            if _ensure_datetime(txn.get("timestamp", datetime.now())) >= cutoff_time
        ]

        return recent_transactions

    def add_fraud_alert(self, customer_id: str, alert: Dict[str, Any]):
        """
        Add a fraud alert for customer.

        Args:
            customer_id: Customer identifier
            alert: Alert data
        """
        if "timestamp" not in alert:
            alert["timestamp"] = datetime.now()

        self.customer_alerts[customer_id].append(alert)

        # Update session metadata
        session_id = f"customer_{customer_id}"
        if session_id in self.session_metadata:
            self.session_metadata[session_id]["alert_count"] += 1

        logger.info(f"Fraud alert added for customer {customer_id}")

    def get_customer_alerts(
        self,
        customer_id: str,
        window_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent fraud alerts for customer.

        Args:
            customer_id: Customer identifier
            window_hours: Time window in hours

        Returns:
            List of recent alerts
        """
        if window_hours is None:
            window_hours = self.max_session_age_hours

        cutoff_time = datetime.now() - timedelta(hours=window_hours)

        alerts = self.customer_alerts.get(customer_id, [])

        recent_alerts = [
            alert for alert in alerts
            if _ensure_datetime(alert.get("timestamp", datetime.now())) >= cutoff_time
        ]

        return recent_alerts

    def get_session_state(self, customer_id: str) -> Dict[str, Any]:
        """
        Get current session state for customer.

        Args:
            customer_id: Customer identifier

        Returns:
            Session state dictionary
        """
        session_id = f"customer_{customer_id}"
        metadata = self.session_metadata.get(session_id, {})

        recent_transactions = self.get_customer_history(customer_id)
        recent_alerts = self.get_customer_alerts(customer_id)

        state = {
            "customer_id": customer_id,
            "session_id": session_id,
            "metadata": metadata,
            "recent_transaction_count": len(recent_transactions),
            "recent_alert_count": len(recent_alerts),
            "has_recent_alerts": len(recent_alerts) > 0,
            "velocity_1h": len(recent_transactions)
        }

        return state

    def _cleanup_old_transactions(self, customer_id: str):
        """Remove transactions older than max session age."""
        cutoff_time = datetime.now() - timedelta(hours=self.max_session_age_hours)

        transactions = self.customer_transactions.get(customer_id, [])
        self.customer_transactions[customer_id] = [
            txn for txn in transactions
            if _ensure_datetime(txn.get("timestamp", datetime.now())) >= cutoff_time
        ]

    def cleanup_old_sessions(self):
        """Clean up expired sessions."""
        cutoff_time = datetime.now() - timedelta(hours=self.max_session_age_hours)

        expired_sessions = [
            session_id for session_id, metadata in self.session_metadata.items()
            if metadata.get("last_activity", datetime.now()) < cutoff_time
        ]

        for session_id in expired_sessions:
            customer_id = self.session_metadata[session_id].get("customer_id")

            # Clean up data
            if customer_id:
                if customer_id in self.customer_transactions:
                    del self.customer_transactions[customer_id]
                if customer_id in self.customer_alerts:
                    del self.customer_alerts[customer_id]

            del self.session_metadata[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall session manager statistics.

        Returns:
            Statistics dictionary
        """
        total_transactions = sum(len(txns) for txns in self.customer_transactions.values())
        total_alerts = sum(len(alerts) for alerts in self.customer_alerts.values())

        return {
            "active_sessions": len(self.session_metadata),
            "total_customers": len(self.customer_transactions),
            "total_transactions": total_transactions,
            "total_alerts": total_alerts,
            "average_transactions_per_customer": (
                total_transactions / len(self.customer_transactions)
                if self.customer_transactions else 0
            )
        }