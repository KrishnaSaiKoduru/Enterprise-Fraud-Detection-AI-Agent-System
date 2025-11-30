"""
Evaluation Metrics for Fraud Detection System

Computes performance metrics including AUC, precision@k, recall@k, and more.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class FraudDetectionEvaluator:
    """
    Evaluator for fraud detection system performance.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.results = []

    def add_result(
        self,
        transaction_id: str,
        true_label: int,
        fraud_score: float,
        prediction: int = None
    ):
        """
        Add a prediction result for evaluation.

        Args:
            transaction_id: Transaction identifier
            true_label: True fraud label (0 or 1)
            fraud_score: Predicted fraud score (0-1)
            prediction: Binary prediction (0 or 1), computed from score if not provided
        """
        if prediction is None:
            prediction = 1 if fraud_score >= 0.7 else 0

        self.results.append({
            "transaction_id": transaction_id,
            "true_label": true_label,
            "fraud_score": fraud_score,
            "prediction": prediction
        })

    def compute_metrics(self, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        if not self.results:
            raise ValueError("No results added for evaluation")

        df = pd.DataFrame(self.results)
        y_true = df["true_label"].values
        y_scores = df["fraud_score"].values
        y_pred = (y_scores >= threshold).astype(int)

        # Classification metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ROC AUC
        try:
            auc_roc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc_roc = 0.0  # If only one class present

        # PR AUC (Average Precision)
        try:
            auc_pr = average_precision_score(y_true, y_scores)
        except ValueError:
            auc_pr = 0.0

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Precision@k and Recall@k
        precision_at_k = self.compute_precision_at_k(y_true, y_scores, k_values=[10, 50, 100])
        recall_at_k = self.compute_recall_at_k(y_true, y_scores, k_values=[10, 50, 100])

        metrics = {
            "threshold": threshold,
            "sample_size": len(y_true),
            "positive_samples": int(y_true.sum()),
            "negative_samples": int((1 - y_true).sum()),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            },
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        }

        return metrics

    def compute_precision_at_k(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        k_values: List[int] = [10, 50, 100]
    ) -> Dict[str, float]:
        """
        Compute precision@k for different k values.

        Args:
            y_true: True labels
            y_scores: Predicted scores
            k_values: List of k values to evaluate

        Returns:
            Dictionary mapping k to precision@k
        """
        precision_at_k = {}

        # Sort by score descending
        sorted_indices = np.argsort(y_scores)[::-1]

        for k in k_values:
            if k > len(y_true):
                k = len(y_true)

            top_k_indices = sorted_indices[:k]
            top_k_labels = y_true[top_k_indices]

            precision = top_k_labels.sum() / k if k > 0 else 0.0
            precision_at_k[f"p@{k}"] = float(precision)

        return precision_at_k

    def compute_recall_at_k(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        k_values: List[int] = [10, 50, 100]
    ) -> Dict[str, float]:
        """
        Compute recall@k for different k values.

        Args:
            y_true: True labels
            y_scores: Predicted scores
            k_values: List of k values to evaluate

        Returns:
            Dictionary mapping k to recall@k
        """
        recall_at_k = {}
        total_positives = y_true.sum()

        if total_positives == 0:
            return {f"r@{k}": 0.0 for k in k_values}

        # Sort by score descending
        sorted_indices = np.argsort(y_scores)[::-1]

        for k in k_values:
            if k > len(y_true):
                k = len(y_true)

            top_k_indices = sorted_indices[:k]
            top_k_labels = y_true[top_k_indices]

            recall = top_k_labels.sum() / total_positives
            recall_at_k[f"r@{k}"] = float(recall)

        return recall_at_k

    def find_optimal_threshold(
        self,
        metric: str = "f1",
        thresholds: np.ndarray = None
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold.

        Args:
            metric: Metric to optimize ('f1', 'precision', 'recall')
            thresholds: Array of thresholds to test

        Returns:
            Tuple of (optimal_threshold, optimal_metric_value)
        """
        if not self.results:
            raise ValueError("No results added for evaluation")

        df = pd.DataFrame(self.results)
        y_true = df["true_label"].values
        y_scores = df["fraud_score"].values

        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 50)

        best_threshold = 0.5
        best_metric_value = 0.0

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)

            if metric == "f1":
                metric_value = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                metric_value = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                metric_value = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold

        return float(best_threshold), float(best_metric_value)

    def plot_roc_curve(self, save_path: str = None):
        """
        Plot ROC curve.

        Args:
            save_path: Path to save plot (optional)
        """
        from sklearn.metrics import roc_curve

        df = pd.DataFrame(self.results)
        y_true = df["true_label"].values
        y_scores = df["fraud_score"].values

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Fraud Detection System')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.show()

    def plot_precision_recall_curve(self, save_path: str = None):
        """
        Plot Precision-Recall curve.

        Args:
            save_path: Path to save plot (optional)
        """
        df = pd.DataFrame(self.results)
        y_true = df["true_label"].values
        y_scores = df["fraud_score"].values

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {auc_pr:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Fraud Detection System')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, threshold: float = 0.7, save_path: str = None):
        """
        Plot confusion matrix.

        Args:
            threshold: Classification threshold
            save_path: Path to save plot (optional)
        """
        df = pd.DataFrame(self.results)
        y_true = df["true_label"].values
        y_scores = df["fraud_score"].values
        y_pred = (y_scores >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix (Threshold={threshold:.2f})')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def print_metrics_report(self, threshold: float = 0.7):
        """
        Print formatted metrics report.

        Args:
            threshold: Classification threshold
        """
        metrics = self.compute_metrics(threshold)

        print("\n" + "="*60)
        print("FRAUD DETECTION EVALUATION REPORT")
        print("="*60)

        print(f"\nDataset:")
        print(f"  Total Samples: {metrics['sample_size']}")
        print(f"  Positive (Fraud): {metrics['positive_samples']}")
        print(f"  Negative (Legitimate): {metrics['negative_samples']}")
        print(f"  Fraud Ratio: {metrics['positive_samples']/metrics['sample_size']*100:.2f}%")

        print(f"\nThreshold: {metrics['threshold']:.3f}")

        print(f"\nClassification Metrics:")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  True Negatives: {cm['true_negatives']}")
        print(f"  False Positives: {cm['false_positives']}")
        print(f"  False Negatives: {cm['false_negatives']}")
        print(f"  True Positives: {cm['true_positives']}")

        print(f"\nError Rates:")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")

        print(f"\nPrecision@k:")
        for key, value in metrics['precision_at_k'].items():
            print(f"  {key}: {value:.4f}")

        print(f"\nRecall@k:")
        for key, value in metrics['recall_at_k'].items():
            print(f"  {key}: {value:.4f}")

        print("="*60 + "\n")

    def clear_results(self):
        """Clear all evaluation results."""
        self.results = []
