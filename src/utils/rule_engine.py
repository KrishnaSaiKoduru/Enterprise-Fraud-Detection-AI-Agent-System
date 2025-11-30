"""
Rule-Based Fraud Detection Engine

This module implements rule-based fraud detection logic to complement ML models.
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime
import yaml
from pathlib import Path


class FraudRuleEngine:
    """
    Rule-based fraud detection engine.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the rule engine.

        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'agent_config.yaml'

        self.config = self._load_config(config_path)
        self.rules = self.config.get('rules', {})

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'rules': {
                'high_amount': {
                    'threshold': 5000,
                    'score': 0.4,
                    'description': 'Transaction amount exceeds typical threshold'
                },
                'unusual_distance': {
                    'threshold_km': 100,
                    'score': 0.3,
                    'description': 'Transaction location far from home'
                },
                'high_velocity': {
                    'transactions_per_hour': 5,
                    'score': 0.5,
                    'description': 'Multiple transactions in short time window'
                },
                'risky_merchant': {
                    'categories': ['gambling', 'crypto', 'high-risk-international'],
                    'score': 0.3,
                    'description': 'Merchant in high-risk category'
                },
                'unusual_time': {
                    'hours': [0, 1, 2, 3, 4, 5],
                    'score': 0.2,
                    'description': 'Transaction during unusual hours'
                }
            }
        }

    def evaluate_rules(
        self,
        transaction: Dict[str, Any],
        customer_history: List[Dict[str, Any]] = None
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate all fraud detection rules.

        Args:
            transaction: Transaction data
            customer_history: Customer transaction history

        Returns:
            Tuple of (total_rule_score, list of triggered rules)
        """
        triggered_rules = []
        total_score = 0.0

        # Rule 1: High Amount
        if self._check_high_amount(transaction):
            rule_info = {
                'rule': 'high_amount',
                'score': self.rules['high_amount']['score'],
                'description': self.rules['high_amount']['description'],
                'details': f"Amount ${transaction.get('amount', 0):.2f} exceeds ${self.rules['high_amount']['threshold']}"
            }
            triggered_rules.append(rule_info)
            total_score += rule_info['score']

        # Rule 2: Unusual Distance
        if self._check_unusual_distance(transaction):
            geo_distance = self._calculate_geo_distance(transaction)
            rule_info = {
                'rule': 'unusual_distance',
                'score': self.rules['unusual_distance']['score'],
                'description': self.rules['unusual_distance']['description'],
                'details': f"Distance {geo_distance:.1f} km exceeds {self.rules['unusual_distance']['threshold_km']} km"
            }
            triggered_rules.append(rule_info)
            total_score += rule_info['score']

        # Rule 3: High Velocity
        if customer_history and self._check_high_velocity(transaction, customer_history):
            velocity = self._calculate_velocity(transaction, customer_history)
            rule_info = {
                'rule': 'high_velocity',
                'score': self.rules['high_velocity']['score'],
                'description': self.rules['high_velocity']['description'],
                'details': f"{velocity} transactions in last hour exceeds {self.rules['high_velocity']['transactions_per_hour']}"
            }
            triggered_rules.append(rule_info)
            total_score += rule_info['score']

        # Rule 4: Risky Merchant
        if self._check_risky_merchant(transaction):
            rule_info = {
                'rule': 'risky_merchant',
                'score': self.rules['risky_merchant']['score'],
                'description': self.rules['risky_merchant']['description'],
                'details': f"Merchant category '{transaction.get('merchant_category')}' is high-risk"
            }
            triggered_rules.append(rule_info)
            total_score += rule_info['score']

        # Rule 5: Unusual Time
        if self._check_unusual_time(transaction):
            hour = self._get_transaction_hour(transaction)
            rule_info = {
                'rule': 'unusual_time',
                'score': self.rules['unusual_time']['score'],
                'description': self.rules['unusual_time']['description'],
                'details': f"Transaction at hour {hour} is unusual"
            }
            triggered_rules.append(rule_info)
            total_score += rule_info['score']

        # Normalize score to 0-1 range
        # Maximum possible score if all rules trigger
        max_score = sum(rule['score'] for rule in self.rules.values())
        normalized_score = min(total_score / max_score, 1.0) if max_score > 0 else 0.0

        return normalized_score, triggered_rules

    def _check_high_amount(self, transaction: Dict[str, Any]) -> bool:
        """Check if transaction amount is unusually high."""
        amount = float(transaction.get('amount', 0))
        threshold = self.rules['high_amount']['threshold']
        return amount > threshold

    def _check_unusual_distance(self, transaction: Dict[str, Any]) -> bool:
        """Check if transaction is far from home location."""
        geo_distance = self._calculate_geo_distance(transaction)
        threshold = self.rules['unusual_distance']['threshold_km']
        return geo_distance > threshold

    def _calculate_geo_distance(self, transaction: Dict[str, Any]) -> float:
        """Calculate distance from home location."""
        from .feature_engineering import haversine_distance

        home_lat = transaction.get('home_latitude', 37.7749)
        home_lon = transaction.get('home_longitude', -122.4194)
        txn_lat = transaction.get('latitude', home_lat)
        txn_lon = transaction.get('longitude', home_lon)

        return haversine_distance(home_lat, home_lon, txn_lat, txn_lon)

    def _check_high_velocity(
        self,
        transaction: Dict[str, Any],
        customer_history: List[Dict[str, Any]]
    ) -> bool:
        """Check if customer has too many recent transactions."""
        velocity = self._calculate_velocity(transaction, customer_history)
        threshold = self.rules['high_velocity']['transactions_per_hour']
        return velocity >= threshold

    def _calculate_velocity(
        self,
        transaction: Dict[str, Any],
        customer_history: List[Dict[str, Any]]
    ) -> int:
        """Calculate number of transactions in last hour."""
        from datetime import timedelta
        import pandas as pd

        if not customer_history:
            return 0

        current_time = pd.to_datetime(transaction.get('timestamp', datetime.now()))
        time_window = current_time - timedelta(hours=1)

        velocity = sum(
            1 for txn in customer_history
            if pd.to_datetime(txn.get('timestamp', current_time)) >= time_window
            and pd.to_datetime(txn.get('timestamp', current_time)) < current_time
        )

        return velocity

    def _check_risky_merchant(self, transaction: Dict[str, Any]) -> bool:
        """Check if merchant is in high-risk category."""
        merchant_category = transaction.get('merchant_category', '')
        risky_categories = self.rules['risky_merchant']['categories']
        return merchant_category in risky_categories

    def _check_unusual_time(self, transaction: Dict[str, Any]) -> bool:
        """Check if transaction occurs during unusual hours."""
        hour = self._get_transaction_hour(transaction)
        unusual_hours = self.rules['unusual_time']['hours']
        return hour in unusual_hours

    def _get_transaction_hour(self, transaction: Dict[str, Any]) -> int:
        """Extract hour from transaction timestamp."""
        import pandas as pd

        timestamp = transaction.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        return timestamp.hour
