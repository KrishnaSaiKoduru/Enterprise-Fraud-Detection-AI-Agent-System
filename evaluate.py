"""
Evaluation Script for Fraud Detection System

Evaluates the fraud detection system on test data and computes metrics.
"""

import json
import pandas as pd
from pathlib import Path

from src.fraud_detection_system import create_fraud_detection_system
from src.evaluation import FraudDetectionEvaluator


def load_test_data(test_file: str = None):
    """Load test data with true fraud labels."""
    if test_file is None:
        test_file = Path(__file__).parent / "data" / "processed" / "test_set_with_predictions.csv"

    if not test_file.exists():
        print(f"Error: Test file {test_file} not found.")
        print("Please run the notebook: notebooks/01_data_generation_and_model_training.ipynb first")
        return None

    df = pd.read_csv(test_file)
    return df


def evaluate_system():
    """Run comprehensive evaluation of the fraud detection system."""
    print("\n" + "="*80)
    print("FRAUD DETECTION SYSTEM EVALUATION")
    print("="*80 + "\n")

    # Load test data
    print("Loading test data...")
    df = load_test_data()

    if df is None:
        print("\nSkipping evaluation - no test data available.")
        print("Run the data generation notebook first:")
        print("  jupyter notebook notebooks/01_data_generation_and_model_training.ipynb")
        return

    print(f"Loaded {len(df)} test transactions")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")

    # Create fraud detection system
    print("\nInitializing fraud detection system...")
    system = create_fraud_detection_system(enable_debug=False)

    # Create evaluator
    evaluator = FraudDetectionEvaluator()

    # Process each transaction
    print("\nProcessing transactions...")
    processed = 0
    for idx, row in df.iterrows():
        # Convert row to transaction dict
        transaction = {
            "transaction_id": row["transaction_id"],
            "customer_id": int(row["customer_id"]),
            "timestamp": row["timestamp"],
            "amount": float(row["amount"]),
            "merchant_id": row["merchant_id"],
            "merchant_category": row["merchant_category"],
            "merchant_risk_category": row["merchant_risk_category"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "home_latitude": float(row["home_latitude"]),
            "home_longitude": float(row["home_longitude"])
        }

        # Process transaction
        result = system.process_transaction(transaction, auto_enrich=False, auto_brief=False)

        # Add to evaluator
        fraud_score = result["summary"]["fraud_score"]
        true_label = int(row["is_fraud"])

        evaluator.add_result(
            transaction_id=transaction["transaction_id"],
            true_label=true_label,
            fraud_score=fraud_score
        )

        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(df)} transactions...")

    print(f"Completed processing {processed} transactions\n")

    # Compute and display metrics
    print("Computing evaluation metrics...\n")

    # Find optimal threshold
    optimal_threshold, optimal_f1 = evaluator.find_optimal_threshold(metric="f1")
    print(f"Optimal Threshold (F1): {optimal_threshold:.3f} (F1={optimal_f1:.4f})")

    # Compute metrics at different thresholds
    print("\n" + "-"*80)
    print("METRICS AT THRESHOLD 0.5:")
    print("-"*80)
    evaluator.print_metrics_report(threshold=0.5)

    print("\n" + "-"*80)
    print("METRICS AT THRESHOLD 0.7 (Default):")
    print("-"*80)
    evaluator.print_metrics_report(threshold=0.7)

    print("\n" + "-"*80)
    print(f"METRICS AT OPTIMAL THRESHOLD {optimal_threshold:.3f}:")
    print("-"*80)
    evaluator.print_metrics_report(threshold=optimal_threshold)

    # Plot curves (optional - comment out if running headless)
    try:
        print("\nGenerating evaluation plots...")

        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        evaluator.plot_roc_curve(save_path=str(output_dir / "eval_roc_curve.png"))
        evaluator.plot_precision_recall_curve(save_path=str(output_dir / "eval_pr_curve.png"))
        evaluator.plot_confusion_matrix(
            threshold=optimal_threshold,
            save_path=str(output_dir / "eval_confusion_matrix.png")
        )

        print("Plots saved to data/processed/")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    # Save metrics to file
    metrics = evaluator.compute_metrics(threshold=optimal_threshold)
    metrics_file = Path(__file__).parent / "data" / "processed" / "evaluation_metrics.json"

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_file}")

    # Print system performance summary
    print("\n" + "="*80)
    print("SYSTEM PERFORMANCE SUMMARY")
    print("="*80)

    system.print_metrics()

    print("\n" + "✅ " * 20)
    print("EVALUATION COMPLETE!")
    print("✅ " * 20 + "\n")


if __name__ == "__main__":
    evaluate_system()
