"""
Feature Engineering Utilities for Fraud Detection

This module provides functions to engineer features from raw transaction data
for use in fraud detection models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on earth.

    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)

    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def engineer_transaction_features(
    transaction: Dict[str, Any],
    customer_history: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Engineer features for a single transaction.

    Args:
        transaction: Transaction data dictionary
        customer_history: List of previous transactions for velocity calculation

    Returns:
        Dictionary with engineered features
    """
    features = {}

    # Extract timestamp
    if isinstance(transaction.get('timestamp'), str):
        timestamp = pd.to_datetime(transaction['timestamp'])
    else:
        timestamp = transaction.get('timestamp', datetime.now())

    # Time features
    features['hour'] = timestamp.hour
    features['day_of_week'] = timestamp.dayofweek
    features['is_weekend'] = int(timestamp.dayofweek in [5, 6])

    # Amount
    features['amount'] = float(transaction.get('amount', 0))

    # Merchant risk score
    risk_mapping = {'low_risk': 0.0, 'medium_risk': 0.5, 'high_risk': 1.0}
    merchant_risk = transaction.get('merchant_risk_category', 'low_risk')
    features['merchant_risk_score'] = risk_mapping.get(merchant_risk, 0.0)

    # Geo distance from home
    home_lat = transaction.get('home_latitude', 37.7749)
    home_lon = transaction.get('home_longitude', -122.4194)
    txn_lat = transaction.get('latitude', home_lat)
    txn_lon = transaction.get('longitude', home_lon)

    features['geo_distance_km'] = haversine_distance(
        home_lat, home_lon, txn_lat, txn_lon
    )

    # Velocity (transactions in last hour)
    features['velocity_1h'] = 0
    if customer_history:
        time_window = timestamp - timedelta(hours=1)
        features['velocity_1h'] = sum(
            1 for txn in customer_history
            if pd.to_datetime(txn.get('timestamp', timestamp)) >= time_window
            and pd.to_datetime(txn.get('timestamp', timestamp)) < timestamp
        )

    # Amount z-score (if we have history)
    if customer_history and len(customer_history) > 0:
        amounts = [float(txn.get('amount', 0)) for txn in customer_history]
        mean_amount = np.mean(amounts)
        std_amount = np.std(amounts) if len(amounts) > 1 else 1.0
        std_amount = max(std_amount, 0.01)  # Avoid division by zero
        features['amount_z_score'] = (features['amount'] - mean_amount) / std_amount
    else:
        features['amount_z_score'] = 0.0

    return features


def extract_feature_vector(
    transaction: Dict[str, Any],
    customer_history: List[Dict[str, Any]] = None,
    feature_columns: List[str] = None
) -> np.ndarray:
    """
    Extract feature vector for model inference.

    Args:
        transaction: Transaction data
        customer_history: Customer transaction history
        feature_columns: Ordered list of feature names

    Returns:
        numpy array of features
    """
    if feature_columns is None:
        feature_columns = [
            'amount',
            'hour',
            'merchant_risk_score',
            'velocity_1h',
            'geo_distance_km',
            'amount_z_score',
            'is_weekend'
        ]

    features = engineer_transaction_features(transaction, customer_history)
    return np.array([features.get(col, 0.0) for col in feature_columns]).reshape(1, -1)


def explain_features(
    transaction: Dict[str, Any],
    customer_history: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate human-readable feature explanations.

    Args:
        transaction: Transaction data
        customer_history: Customer transaction history

    Returns:
        Dictionary with feature values and explanations
    """
    features = engineer_transaction_features(transaction, customer_history)

    explanations = {
        'amount': {
            'value': features['amount'],
            'explanation': f"Transaction amount: ${features['amount']:.2f}"
        },
        'hour': {
            'value': features['hour'],
            'explanation': f"Hour of day: {features['hour']:02d}:00"
        },
        'merchant_risk_score': {
            'value': features['merchant_risk_score'],
            'explanation': f"Merchant risk: {features['merchant_risk_score']:.1f} (0=low, 0.5=medium, 1=high)"
        },
        'velocity_1h': {
            'value': features['velocity_1h'],
            'explanation': f"Recent transactions: {features['velocity_1h']} in last hour"
        },
        'geo_distance_km': {
            'value': features['geo_distance_km'],
            'explanation': f"Distance from home: {features['geo_distance_km']:.1f} km"
        },
        'amount_z_score': {
            'value': features['amount_z_score'],
            'explanation': f"Amount deviation: {features['amount_z_score']:.2f} standard deviations from customer average"
        },
        'is_weekend': {
            'value': features['is_weekend'],
            'explanation': f"Weekend transaction: {'Yes' if features['is_weekend'] else 'No'}"
        }
    }

    return explanations
