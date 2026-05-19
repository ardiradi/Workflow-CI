"""
modelling.py (MLProject Version)
Model Training Script untuk dijalankan di dalam MLflow Project / CI Pipeline.

Dataset: Heart Disease (Cleveland) - UCI ML Repository
Target: Binary classification (0 = no disease, 1 = disease)

Author: ardir
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(data_dir='heart_disease_preprocessing'):
    """Memuat data yang sudah dipreproses."""
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    """Melatih model dan melakukan logging ke MLflow."""
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("model_type", "RandomForestClassifier")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_test = model.predict(X_test)
    
    # Log metrics
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, average='weighted')
    test_rec = recall_score(y_test, y_pred_test, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_prec)
    mlflow.log_metric("test_recall", test_rec)
    mlflow.log_metric("test_f1_score", test_f1)
    
    # Log confusion matrix artifact
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['no_disease', 'disease'],
                yticklabels=['no_disease', 'disease'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    mlflow.log_artifact(cm_path)
    
    # Log classification report
    report = classification_report(y_test, y_pred_test, output_dict=True)
    report_path = "artifacts/classification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(report_path)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heart Disease Model Training')
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--min_samples_split', type=int, default=5)
    parser.add_argument('--min_samples_leaf', type=int, default=2)
    args = parser.parse_args()
    
    train_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf
    )
