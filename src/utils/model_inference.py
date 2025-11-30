"""
Model Inference Utilities for Fraud Detection

This module provides functions to load trained models and perform fraud scoring.
"""

import joblib
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

from .feature_engineering import extract_feature_vector


class FraudDetectionModel:
    """
    Wrapper for trained fraud detection model.
    """

    def __init__(self, model_dir: str = None):
        """
        Initialize the fraud detection model.

        Args:
            model_dir: Directory containing model artifacts
        """
        if model_dir is None:
            # Default to models directory
            self.model_dir = Path(__file__).parent.parent.parent / 'models'
        else:
            self.model_dir = Path(model_dir)

        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.config = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, scaler, and configuration."""
        try:
            # Load model
            model_path = self.model_dir / 'iso_fraud.pkl'
            self.model = joblib.load(model_path)

            # Load scaler
            scaler_path = self.model_dir / 'scaler.pkl'
            self.scaler = joblib.load(scaler_path)

            # Load feature columns
            feature_path = self.model_dir / 'feature_columns.json'
            with open(feature_path, 'r') as f:
                self.feature_columns = json.load(f)

            # Load config
            config_path = self.model_dir / 'model_config.json'
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        except FileNotFoundError as e:
            print(f"Warning: Model artifacts not found. Using mock mode. Error: {e}")
            self._init_mock_mode()

    def _init_mock_mode(self):
        """Initialize in mock mode if artifacts not available."""
        self.feature_columns = [
            'amount', 'hour', 'merchant_risk_score',
            'velocity_1h', 'geo_distance_km', 'amount_z_score', 'is_weekend'
        ]
        self.config = {'best_threshold': 0.6}

    def predict_fraud_score(
        self,
        transaction: Dict[str, Any],
        customer_history: List[Dict[str, Any]] = None
    ) -> float:
        """
        Predict fraud score for a transaction.

        Args:
            transaction: Transaction data
            customer_history: Customer transaction history

        Returns:
            Fraud probability score (0-1, higher = more likely fraud)
        """
        if self.model is None or self.scaler is None:
            # Mock mode: simple heuristic
            return self._mock_predict(transaction)

        # Extract features
        X = extract_feature_vector(transaction, customer_history, self.feature_columns)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get anomaly score
        score = self.model.score_samples(X_scaled)[0]

        # Convert to probability (0-1, higher = more fraud)
        # IsolationForest returns negative scores, more negative = more anomalous
        fraud_prob = self._score_to_probability(score)

        return fraud_prob

    def _score_to_probability(self, score: float) -> float:
        """
        Convert IsolationForest anomaly score to fraud probability.

        Args:
            score: Anomaly score from model

        Returns:
            Fraud probability (0-1)
        """
        # Invert and normalize
        # Typical range is around -0.5 to 0.5
        # More negative = more anomalous
        inverted = -score

        # Apply sigmoid to get 0-1 range
        prob = 1 / (1 + np.exp(-inverted * 5))  # Scale factor of 5 for sensitivity

        return float(np.clip(prob, 0, 1))

    def _mock_predict(self, transaction: Dict[str, Any]) -> float:
        """
        Mock prediction when model not available.

        Args:
            transaction: Transaction data

        Returns:
            Mock fraud score
        """
        # Simple heuristic based on amount and merchant risk
        score = 0.0

        amount = float(transaction.get('amount', 0))
        if amount > 5000:
            score += 0.3
        elif amount > 1000:
            score += 0.1

        merchant_risk = transaction.get('merchant_risk_category', 'low_risk')
        if merchant_risk == 'high_risk':
            score += 0.4
        elif merchant_risk == 'medium_risk':
            score += 0.2

        # Add some randomness
        score += np.random.uniform(0, 0.1)

        return float(np.clip(score, 0, 1))

    def get_threshold(self) -> float:
        """Get the optimal classification threshold."""
        if self.config:
            return self.config.get('best_threshold', 0.6)
        return 0.6

    def classify(
        self,
        transaction: Dict[str, Any],
        customer_history: List[Dict[str, Any]] = None,
        threshold: float = None
    ) -> Tuple[bool, float]:
        """
        Classify transaction as fraud or legitimate.

        Args:
            transaction: Transaction data
            customer_history: Customer transaction history
            threshold: Classification threshold (uses optimal if None)

        Returns:
            Tuple of (is_fraud, fraud_score)
        """
        if threshold is None:
            threshold = self.get_threshold()

        fraud_score = self.predict_fraud_score(transaction, customer_history)
        is_fraud = fraud_score >= threshold

        return is_fraud, fraud_score
