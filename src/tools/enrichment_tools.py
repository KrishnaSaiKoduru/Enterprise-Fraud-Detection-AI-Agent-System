"""
Enrichment Tools for Fraud Detection

These tools provide additional context about customers, merchants, and transactions.
"""

from typing import Dict, Any, List
import json
import random
import numpy as np
from datetime import datetime, timedelta


# Mock databases (in production, these would be real database queries)
_CUSTOMER_PROFILES = {}
_MERCHANT_PROFILES = {}
_DISPUTE_HISTORY = {}


def get_customer_profile(customer_id: str) -> str:
    """
    Retrieve customer profile information.

    Args:
        customer_id: Customer identifier

    Returns:
        JSON string with customer profile including:
        - customer_id: Customer identifier
        - account_age_days: Days since account creation
        - total_transactions: Total number of transactions
        - average_transaction_amount: Average transaction amount
        - fraud_history_count: Number of previous fraud cases
        - customer_tier: Customer tier (bronze/silver/gold/platinum)
        - risk_score: Overall customer risk score (0-1)
    """
    # Check cache
    if customer_id in _CUSTOMER_PROFILES:
        return json.dumps(_CUSTOMER_PROFILES[customer_id], indent=2)

    # Generate mock profile
    profile = {
        "customer_id": customer_id,
        "account_age_days": random.randint(30, 3650),  # 1 month to 10 years
        "total_transactions": random.randint(10, 5000),
        "average_transaction_amount": round(random.uniform(50, 500), 2),
        "fraud_history_count": random.choice([0, 0, 0, 0, 1, 1, 2]),  # Most customers have no fraud
        "customer_tier": str(np.random.choice(["bronze", "silver", "gold", "platinum"], p=[0.4, 0.3, 0.2, 0.1])),
        "risk_score": round(random.uniform(0.0, 0.3), 4),  # Most customers are low risk
        "email_verified": bool(np.random.choice([True, False], p=[0.9, 0.1])),
        "phone_verified": bool(np.random.choice([True, False], p=[0.85, 0.15])),
        "last_login": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
        "home_location": {
            "city": "San Francisco",
            "state": "CA",
            "country": "USA"
        }
    }

    # Cache for future use
    _CUSTOMER_PROFILES[customer_id] = profile

    return json.dumps(profile, indent=2)


def get_dispute_history(customer_id: str) -> str:
    """
    Retrieve dispute history for a customer.

    Args:
        customer_id: Customer identifier

    Returns:
        JSON string with dispute history including:
        - customer_id: Customer identifier
        - total_disputes: Total number of disputes
        - disputes: List of dispute details
        - dispute_rate: Percentage of transactions disputed
        - resolution_summary: Summary of dispute resolutions
    """
    # Check cache
    if customer_id in _DISPUTE_HISTORY:
        return json.dumps(_DISPUTE_HISTORY[customer_id], indent=2)

    # Generate mock dispute history
    num_disputes = random.choice([0, 0, 0, 1, 1, 2, 3])  # Most customers have few disputes
    disputes = []

    for i in range(num_disputes):
        dispute = {
            "dispute_id": f"DISP{random.randint(10000, 99999)}",
            "transaction_id": f"TXN{random.randint(10000000, 99999999):08d}",
            "amount": round(random.uniform(50, 2000), 2),
            "dispute_date": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
            "reason": random.choice([
                "unauthorized_transaction",
                "product_not_received",
                "product_defective",
                "duplicate_charge",
                "incorrect_amount"
            ]),
            "status": str(np.random.choice(["resolved", "pending", "closed"], p=[0.7, 0.2, 0.1])),
            "resolution": random.choice([
                "refunded",
                "merchant_liable",
                "customer_liable",
                "investigating"
            ]),
            "fraud_confirmed": bool(np.random.choice([True, False], p=[0.3, 0.7]))
        }
        disputes.append(dispute)

    # Count fraud-confirmed disputes
    fraud_confirmed_count = sum(1 for d in disputes if d.get("fraud_confirmed", False))

    history = {
        "customer_id": customer_id,
        "total_disputes": num_disputes,
        "fraud_confirmed_disputes": fraud_confirmed_count,
        "disputes": disputes,
        "dispute_rate": round(num_disputes / max(random.randint(100, 1000), 1) * 100, 2),
        "resolution_summary": {
            "resolved": sum(1 for d in disputes if d["status"] == "resolved"),
            "pending": sum(1 for d in disputes if d["status"] == "pending"),
            "closed": sum(1 for d in disputes if d["status"] == "closed")
        }
    }

    # Cache for future use
    _DISPUTE_HISTORY[customer_id] = history

    return json.dumps(history, indent=2)


def get_merchant_info(merchant_id: str, merchant_category: str = None) -> str:
    """
    Retrieve merchant information and risk profile.

    Args:
        merchant_id: Merchant identifier
        merchant_category: Optional merchant category

    Returns:
        JSON string with merchant information including:
        - merchant_id: Merchant identifier
        - merchant_name: Merchant business name
        - category: Business category
        - risk_level: Risk level (low/medium/high)
        - fraud_rate: Historical fraud rate for this merchant
        - chargeback_rate: Chargeback rate percentage
        - years_in_business: Years merchant has been in business
        - location: Merchant location
    """
    # Check cache
    if merchant_id in _MERCHANT_PROFILES:
        return json.dumps(_MERCHANT_PROFILES[merchant_id], indent=2)

    # Determine risk level based on category
    if merchant_category:
        high_risk_categories = ["gambling", "crypto", "high-risk-international"]
        medium_risk_categories = ["travel", "entertainment", "online"]

        if merchant_category in high_risk_categories:
            risk_level = "high"
            fraud_rate = round(random.uniform(5.0, 15.0), 2)
        elif merchant_category in medium_risk_categories:
            risk_level = "medium"
            fraud_rate = round(random.uniform(1.0, 5.0), 2)
        else:
            risk_level = "low"
            fraud_rate = round(random.uniform(0.1, 1.0), 2)
    else:
        risk_level = str(np.random.choice(["low", "medium", "high"], p=[0.7, 0.2, 0.1]))
        fraud_rate = round(random.uniform(0.1, 10.0), 2)

    # Generate mock merchant profile
    profile = {
        "merchant_id": merchant_id,
        "merchant_name": f"Business {merchant_id}",
        "category": merchant_category or random.choice([
            "retail", "restaurant", "grocery", "gas", "online"
        ]),
        "risk_level": risk_level,
        "fraud_rate": fraud_rate,
        "chargeback_rate": round(random.uniform(0.1, 3.0), 2),
        "years_in_business": random.randint(1, 25),
        "total_transactions_processed": random.randint(1000, 1000000),
        "average_transaction_amount": round(random.uniform(25, 500), 2),
        "location": {
            "city": random.choice(["San Francisco", "New York", "Los Angeles", "Chicago", "Miami"]),
            "state": random.choice(["CA", "NY", "IL", "FL", "TX"]),
            "country": "USA"
        },
        "compliance_status": str(np.random.choice(["compliant", "under_review"], p=[0.9, 0.1])),
        "last_audit_date": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat()
    }

    # Cache for future use
    _MERCHANT_PROFILES[merchant_id] = profile

    return json.dumps(profile, indent=2)


def get_transaction_context(
    transaction_id: str,
    customer_id: str,
    merchant_id: str
) -> str:
    """
    Get comprehensive transaction context by combining multiple data sources.

    Args:
        transaction_id: Transaction identifier
        customer_id: Customer identifier
        merchant_id: Merchant identifier

    Returns:
        JSON string with enriched transaction context
    """
    # Get all related data
    customer_profile = json.loads(get_customer_profile(customer_id))
    dispute_history = json.loads(get_dispute_history(customer_id))
    merchant_info = json.loads(get_merchant_info(merchant_id))

    # Compute contextual risk factors
    risk_factors = []

    if customer_profile.get("fraud_history_count", 0) > 0:
        risk_factors.append("Customer has previous fraud history")

    if dispute_history.get("fraud_confirmed_disputes", 0) > 0:
        risk_factors.append("Customer has confirmed fraud disputes")

    if merchant_info.get("fraud_rate", 0) > 5.0:
        risk_factors.append("Merchant has high fraud rate")

    if merchant_info.get("risk_level") == "high":
        risk_factors.append("Merchant is in high-risk category")

    if customer_profile.get("account_age_days", 0) < 90:
        risk_factors.append("New customer account (less than 90 days)")

    # Calculate combined risk score
    base_risk = customer_profile.get("risk_score", 0)
    merchant_risk = merchant_info.get("fraud_rate", 0) / 100
    dispute_risk = min(dispute_history.get("fraud_confirmed_disputes", 0) * 0.1, 0.3)

    combined_risk = min(base_risk + merchant_risk + dispute_risk, 1.0)

    context = {
        "transaction_id": transaction_id,
        "enrichment_timestamp": datetime.now().isoformat(),
        "customer": {
            "id": customer_id,
            "account_age_days": customer_profile.get("account_age_days"),
            "tier": customer_profile.get("customer_tier"),
            "fraud_history": customer_profile.get("fraud_history_count"),
            "total_disputes": dispute_history.get("total_disputes"),
            "verified": customer_profile.get("email_verified") and customer_profile.get("phone_verified")
        },
        "merchant": {
            "id": merchant_id,
            "risk_level": merchant_info.get("risk_level"),
            "fraud_rate": merchant_info.get("fraud_rate"),
            "years_in_business": merchant_info.get("years_in_business")
        },
        "risk_assessment": {
            "combined_risk_score": round(combined_risk, 4),
            "risk_factors": risk_factors,
            "risk_factor_count": len(risk_factors)
        },
        "full_customer_profile": customer_profile,
        "full_merchant_info": merchant_info,
        "dispute_history": dispute_history
    }

    return json.dumps(context, indent=2)


def get_geo_location_info(latitude: float, longitude: float) -> str:
    """
    Get geographic location information and risk assessment.

    Args:
        latitude: Transaction latitude
        longitude: Transaction longitude

    Returns:
        JSON string with location information and risk
    """
    # Mock geolocation service
    # In production, this would use a real IP geolocation or address service

    location_info = {
        "latitude": latitude,
        "longitude": longitude,
        "estimated_location": {
            "city": "San Francisco",
            "state": "CA",
            "country": "USA",
            "postal_code": "94102"
        },
        "location_risk": {
            "risk_level": str(np.random.choice(["low", "medium", "high"], p=[0.8, 0.15, 0.05])),
            "known_fraud_hotspot": bool(np.random.choice([True, False], p=[0.1, 0.9])),
            "vpn_detected": bool(np.random.choice([True, False], p=[0.05, 0.95])),
            "proxy_detected": bool(np.random.choice([True, False], p=[0.05, 0.95]))
        },
        "timezone": "America/Los_Angeles",
        "local_time": datetime.now().isoformat()
    }

    return json.dumps(location_info, indent=2)