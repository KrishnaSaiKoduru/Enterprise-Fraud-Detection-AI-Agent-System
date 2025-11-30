"""
Custom Tools for Fraud Detection Agent

These tools are used by the Detector Agent to analyze transactions.
"""

from typing import Dict, Any, List
import json
from datetime import datetime

from ..utils.feature_engineering import engineer_transaction_features, explain_features
from ..utils.model_inference import FraudDetectionModel
from ..utils.rule_engine import FraudRuleEngine


# Initialize models (singleton pattern)
_ml_model = None
_rule_engine = None


def get_ml_model() -> FraudDetectionModel:
    """Get or initialize ML model."""
    global _ml_model
    if _ml_model is None:
        _ml_model = FraudDetectionModel()
    return _ml_model


def get_rule_engine() -> FraudRuleEngine:
    """Get or initialize rule engine."""
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = FraudRuleEngine()
    return _rule_engine


def analyze_transaction_rules(
    transaction: Dict[str, Any],
    customer_history: List[Dict[str, Any]] = None
) -> str:
    """
    Analyze a transaction using rule-based fraud detection.

    Args:
        transaction: Transaction data as JSON string or dict
        customer_history: Optional list of previous customer transactions

    Returns:
        JSON string with rule analysis results including:
        - rule_score: Overall rule-based fraud score (0-1)
        - triggered_rules: List of rules that were triggered
        - rule_count: Number of rules triggered
    """
    # Parse transaction if it's a string
    if isinstance(transaction, str):
        transaction = json.loads(transaction)

    if customer_history is None:
        customer_history = []

    # Get rule engine and evaluate
    engine = get_rule_engine()
    rule_score, triggered_rules = engine.evaluate_rules(transaction, customer_history)

    result = {
        "rule_score": round(rule_score, 4),
        "triggered_rules": triggered_rules,
        "rule_count": len(triggered_rules),
        "details": {
            "transaction_id": transaction.get("transaction_id"),
            "customer_id": transaction.get("customer_id"),
            "amount": transaction.get("amount"),
            "merchant_category": transaction.get("merchant_category")
        }
    }

    return json.dumps(result, indent=2)


def analyze_transaction_ml(
    transaction: Dict[str, Any],
    customer_history: List[Dict[str, Any]] = None
) -> str:
    """
    Analyze a transaction using machine learning anomaly detection.

    Args:
        transaction: Transaction data as JSON string or dict
        customer_history: Optional list of previous customer transactions

    Returns:
        JSON string with ML analysis results including:
        - ml_score: ML-based fraud probability (0-1)
        - is_anomaly: Whether transaction is classified as anomalous
        - threshold: Classification threshold used
        - features: Engineered features used for prediction
    """
    # Parse transaction if it's a string
    if isinstance(transaction, str):
        transaction = json.loads(transaction)

    if customer_history is None:
        customer_history = []

    # Get ML model and predict
    model = get_ml_model()
    ml_score = model.predict_fraud_score(transaction, customer_history)
    threshold = model.get_threshold()
    is_anomaly = ml_score >= threshold

    # Get engineered features for transparency
    features = engineer_transaction_features(transaction, customer_history)

    result = {
        "ml_score": round(ml_score, 4),
        "is_anomaly": is_anomaly,
        "threshold": threshold,
        "confidence": "high" if abs(ml_score - threshold) > 0.2 else "medium" if abs(ml_score - threshold) > 0.1 else "low",
        "features": {k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()},
        "details": {
            "transaction_id": transaction.get("transaction_id"),
            "customer_id": transaction.get("customer_id")
        }
    }

    return json.dumps(result, indent=2)


def compute_combined_fraud_score(
    rule_score: float,
    ml_score: float,
    rule_weight: float = 0.4,
    ml_weight: float = 0.6
) -> str:
    """
    Combine rule-based and ML scores into a final fraud score.

    Args:
        rule_score: Score from rule-based detection (0-1)
        ml_score: Score from ML model (0-1)
        rule_weight: Weight for rule score (default 0.4)
        ml_weight: Weight for ML score (default 0.6)

    Returns:
        JSON string with combined score and classification
    """
    # Weighted combination
    combined_score = (rule_score * rule_weight) + (ml_score * ml_weight)

    # Classification thresholds
    if combined_score >= 0.85:
        priority = "critical"
        recommendation = "Block transaction and alert fraud team immediately"
    elif combined_score >= 0.7:
        priority = "high"
        recommendation = "Flag for manual review before processing"
    elif combined_score >= 0.5:
        priority = "medium"
        recommendation = "Monitor and require additional verification"
    else:
        priority = "low"
        recommendation = "Process normally with standard monitoring"

    result = {
        "combined_score": round(combined_score, 4),
        "priority": priority,
        "recommendation": recommendation,
        "score_breakdown": {
            "rule_score": round(rule_score, 4),
            "ml_score": round(ml_score, 4),
            "rule_weight": rule_weight,
            "ml_weight": ml_weight
        },
        "timestamp": datetime.now().isoformat()
    }

    return json.dumps(result, indent=2)


def explain_fraud_detection(
    transaction: Dict[str, Any],
    rule_score: float,
    ml_score: float,
    combined_score: float,
    triggered_rules: List[Dict[str, Any]],
    customer_history: List[Dict[str, Any]] = None
) -> str:
    """
    Generate human-readable explanation of fraud detection results.

    Args:
        transaction: Transaction data
        rule_score: Rule-based fraud score
        ml_score: ML fraud score
        combined_score: Combined fraud score
        triggered_rules: List of triggered rule details
        customer_history: Customer transaction history

    Returns:
        JSON string with detailed explanation
    """
    if isinstance(transaction, str):
        transaction = json.loads(transaction)

    if customer_history is None:
        customer_history = []

    # Get feature explanations
    feature_explanations = explain_features(transaction, customer_history)

    # Build explanation
    explanation = {
        "transaction_summary": {
            "transaction_id": transaction.get("transaction_id"),
            "customer_id": transaction.get("customer_id"),
            "amount": f"${transaction.get('amount', 0):.2f}",
            "merchant": transaction.get("merchant_category"),
            "timestamp": str(transaction.get("timestamp"))
        },
        "fraud_assessment": {
            "combined_score": round(combined_score, 4),
            "risk_level": "high" if combined_score >= 0.7 else "medium" if combined_score >= 0.5 else "low",
            "rule_contribution": f"{rule_score * 100:.1f}%",
            "ml_contribution": f"{ml_score * 100:.1f}%"
        },
        "triggered_rules": [
            {
                "rule": rule["rule"],
                "impact": f"{rule['score'] * 100:.1f}%",
                "reason": rule["details"]
            }
            for rule in triggered_rules
        ],
        "key_features": {
            name: {
                "value": round(feat["value"], 2) if isinstance(feat["value"], float) else feat["value"],
                "explanation": feat["explanation"]
            }
            for name, feat in list(feature_explanations.items())[:5]  # Top 5 features
        },
        "explanation_text": _generate_explanation_text(
            transaction, combined_score, triggered_rules, feature_explanations
        )
    }

    return json.dumps(explanation, indent=2)


def _generate_explanation_text(
    transaction: Dict[str, Any],
    combined_score: float,
    triggered_rules: List[Dict[str, Any]],
    feature_explanations: Dict[str, Any]
) -> str:
    """Generate natural language explanation."""
    text_parts = []

    # Overall assessment
    if combined_score >= 0.85:
        text_parts.append("🚨 CRITICAL FRAUD ALERT:")
    elif combined_score >= 0.7:
        text_parts.append("⚠️  HIGH FRAUD RISK:")
    elif combined_score >= 0.5:
        text_parts.append("⚡ MEDIUM FRAUD RISK:")
    else:
        text_parts.append("✓ LOW FRAUD RISK:")

    text_parts.append(f"Transaction {transaction.get('transaction_id')} has a fraud score of {combined_score:.2%}.")

    # Rule triggers
    if triggered_rules:
        text_parts.append(f"\n\n{len(triggered_rules)} fraud indicators detected:")
        for rule in triggered_rules:
            text_parts.append(f"  • {rule['details']}")

    # Key features
    text_parts.append("\n\nKey transaction characteristics:")
    for name, feat in list(feature_explanations.items())[:3]:
        text_parts.append(f"  • {feat['explanation']}")

    return "\n".join(text_parts)
