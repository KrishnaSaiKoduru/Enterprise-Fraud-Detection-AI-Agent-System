"""
Demo Script for Enterprise AI Fraud Detection Agent System

This script demonstrates the complete fraud detection pipeline.
"""

import json
import os
from pathlib import Path
from datetime import datetime

from src.fraud_detection_system import create_fraud_detection_system


def load_sample_transactions(sample_file: str = None):
    """Load sample transactions from file."""
    if sample_file is None:
        sample_file = Path(__file__).parent / "data" / "processed" / "sample_transactions.json"

    if not os.path.exists(sample_file):
        print(f"Warning: Sample file {sample_file} not found.")
        print("Please run the notebook: notebooks/01_data_generation_and_model_training.ipynb first")
        return create_mock_transactions()

    with open(sample_file, 'r') as f:
        transactions = json.load(f)

    return transactions[:10]  # Return first 10 for demo


def create_mock_transactions():
    """Create mock transactions if data not available."""
    print("Creating mock transactions for demo...")

    return [
        {
            "transaction_id": "TXN00000001",
            "customer_id": 1001,
            "timestamp": datetime.now().isoformat(),
            "amount": 1500.00,
            "merchant_id": "MERCH00123",
            "merchant_category": "retail",
            "merchant_risk_category": "low_risk",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "home_latitude": 37.7749,
            "home_longitude": -122.4194
        },
        {
            "transaction_id": "TXN00000002",
            "customer_id": 1002,
            "timestamp": datetime.now().isoformat(),
            "amount": 8500.00,  # High amount
            "merchant_id": "MERCH00456",
            "merchant_category": "crypto",  # Risky category
            "merchant_risk_category": "high_risk",
            "latitude": 40.7128,  # Far from home
            "longitude": -74.0060,
            "home_latitude": 37.7749,
            "home_longitude": -122.4194
        },
        {
            "transaction_id": "TXN00000003",
            "customer_id": 1003,
            "timestamp": datetime.now().isoformat(),
            "amount": 250.00,
            "merchant_id": "MERCH00789",
            "merchant_category": "grocery",
            "merchant_risk_category": "low_risk",
            "latitude": 37.7849,
            "longitude": -122.4094,
            "home_latitude": 37.7749,
            "home_longitude": -122.4194
        }
    ]


def demo_single_transaction():
    """Demo: Process a single high-risk transaction."""
    print("\n" + "="*80)
    print("DEMO 1: Single Transaction Processing")
    print("="*80 + "\n")

    # Create system
    system = create_fraud_detection_system(enable_debug=False)

    # Create a suspicious transaction
    transaction = {
        "transaction_id": "TXN_DEMO_001",
        "customer_id": 9999,
        "timestamp": datetime.now().isoformat(),
        "amount": 12000.00,  # Very high amount
        "merchant_id": "MERCH99999",
        "merchant_category": "gambling",  # High-risk category
        "merchant_risk_category": "high_risk",
        "latitude": 51.5074,  # London (far from SF)
        "longitude": -0.1278,
        "home_latitude": 37.7749,  # San Francisco
        "home_longitude": -122.4194
    }

    print("Transaction Details:")
    print(json.dumps(transaction, indent=2))
    print("\n" + "-"*80 + "\n")

    # Process transaction
    result = system.process_transaction(transaction)

    # Display results
    print("FRAUD DETECTION RESULTS:")
    print(f"  Fraud Score: {result['summary']['fraud_score']:.3f}")
    print(f"  Priority: {result['summary']['priority']}")
    print(f"  Fraud Detected: {result['summary']['fraud_detected']}")
    print(f"  Stages Completed: {', '.join(result['summary']['stages_completed'])}")
    print(f"  Processing Time: {result['summary']['processing_time_ms']:.2f}ms")

    # Show detection details
    if "detection" in result["stages"]:
        detection = result["stages"]["detection"]
        print("\n  Rule Analysis:")
        for rule in detection.get("rule_analysis", {}).get("triggered_rules", []):
            print(f"    - {rule['rule']}: {rule['details']}")

        print(f"\n  ML Score: {detection.get('ml_analysis', {}).get('ml_score', 0):.3f}")
        print(f"  Combined Score: {detection.get('combined_assessment', {}).get('combined_score', 0):.3f}")

    # Show investigation brief
    if "investigation_brief" in result["stages"]:
        brief = result["stages"]["investigation_brief"].get("investigation_brief", {})
        print("\n  Investigation Brief:")
        print(f"    Case ID: {brief.get('case_id')}")
        print(f"    Priority: {brief.get('priority')}")
        print(f"    Summary: {brief.get('summary')}")
        print(f"    Suggested Actions:")
        for action in brief.get('suggested_actions', [])[:3]:
            print(f"      - {action}")

    print("\n" + "="*80 + "\n")


def demo_batch_processing():
    """Demo: Process a batch of transactions."""
    print("\n" + "="*80)
    print("DEMO 2: Batch Transaction Processing")
    print("="*80 + "\n")

    # Create system
    system = create_fraud_detection_system(enable_debug=False)

    # Load sample transactions
    transactions = load_sample_transactions()
    print(f"Processing {len(transactions)} transactions...\n")

    # Process batch
    results = system.process_batch(transactions)

    # Show summary
    print("BATCH PROCESSING SUMMARY:")
    print(f"  Total Transactions: {len(results)}")

    fraud_count = sum(1 for r in results if r.get('summary', {}).get('fraud_detected', False))
    print(f"  Fraud Detected: {fraud_count}")

    total_time = sum(r.get('summary', {}).get('processing_time_ms', 0) for r in results)
    avg_time = total_time / len(results) if results else 0
    print(f"  Total Processing Time: {total_time:.2f}ms")
    print(f"  Average Time per Transaction: {avg_time:.2f}ms")

    # Show fraud cases
    print("\n  Fraud Cases:")
    for result in results:
        if result.get('summary', {}).get('fraud_detected', False):
            txn_id = result.get('transaction_id')
            score = result.get('summary', {}).get('fraud_score', 0)
            priority = result.get('summary', {}).get('priority')
            print(f"    - {txn_id}: score={score:.3f}, priority={priority}")

    print("\n" + "="*80 + "\n")


def demo_metrics_and_observability():
    """Demo: Show metrics and observability."""
    print("\n" + "="*80)
    print("DEMO 3: Metrics and Observability")
    print("="*80 + "\n")

    # Create system
    system = create_fraud_detection_system(enable_debug=False)

    # Process some transactions
    transactions = load_sample_transactions()[:5]
    system.process_batch(transactions)

    # Show metrics
    system.print_metrics()


def demo_customer_session():
    """Demo: Customer session tracking."""
    print("\n" + "="*80)
    print("DEMO 4: Customer Session Tracking")
    print("="*80 + "\n")

    # Create system
    system = create_fraud_detection_system(enable_debug=False)

    customer_id = 5555

    # Simulate multiple transactions from same customer
    for i in range(6):  # 6 transactions to trigger velocity rule
        transaction = {
            "transaction_id": f"TXN_SESSION_{i:03d}",
            "customer_id": customer_id,
            "timestamp": datetime.now().isoformat(),
            "amount": 500.00 + (i * 100),
            "merchant_id": f"MERCH{i:05d}",
            "merchant_category": "online",
            "merchant_risk_category": "medium_risk",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "home_latitude": 37.7749,
            "home_longitude": -122.4194
        }

        print(f"Processing transaction {i+1}/6...")
        result = system.process_transaction(transaction)
        velocity = result['stages']['detection']['rule_analysis'].get('details', {})

        # Check if velocity rule triggered
        triggered_rules = result['stages']['detection']['rule_analysis']['triggered_rules']
        velocity_triggered = any(r['rule'] == 'high_velocity' for r in triggered_rules)

        if velocity_triggered:
            print(f"  ⚠️  HIGH VELOCITY DETECTED on transaction {i+1}!")

    # Show session state
    print("\nCustomer Session State:")
    session_state = system.get_customer_session_state(customer_id)
    print(json.dumps(session_state, indent=2, default=str))

    print("\n" + "="*80 + "\n")


def main():
    """Run all demos."""
    print("\n" + "🔍 " * 20)
    print("ENTERPRISE AI FRAUD DETECTION AGENT SYSTEM - DEMO")
    print("🔍 " * 20 + "\n")

    print("This demo showcases the multi-agent fraud detection system built with Google ADK.")
    print("\nThe system includes:")
    print("  1. Detector Agent: Rule-based + ML fraud detection")
    print("  2. Enrichment Agent: Contextual data gathering")
    print("  3. Analyst Agent: AI-powered investigation briefs")
    print("\nFeatures:")
    print("  - IsolationForest ML model for anomaly detection")
    print("  - Rule-based fraud detection (amount, velocity, location, merchant)")
    print("  - Session management for per-customer state")
    print("  - Observability with metrics and logging")
    print("  - Explainable AI with feature importance")

    input("\nPress Enter to start the demos...")

    # Run demos
    demo_single_transaction()
    input("Press Enter to continue to next demo...")

    demo_batch_processing()
    input("Press Enter to continue to next demo...")

    demo_customer_session()
    input("Press Enter to continue to next demo...")

    demo_metrics_and_observability()

    print("\n" + "✅ " * 20)
    print("DEMO COMPLETE!")
    print("✅ " * 20 + "\n")

    print("Next Steps:")
    print("  1. Run the notebook: notebooks/01_data_generation_and_model_training.ipynb")
    print("     to train the ML model and generate test data")
    print("  2. Explore the evaluation pipeline in: src/evaluation/")
    print("  3. Check the comprehensive README for architecture details")


if __name__ == "__main__":
    main()
