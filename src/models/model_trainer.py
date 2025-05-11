import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score, 
                           confusion_matrix, classification_report, precision_score, 
                           recall_score, f1_score, matthews_corrcoef, cohen_kappa_score,
                           roc_curve, auc)
import os

def get_models():
    """Return a list of models to train."""
    return [
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('SVM', SVC(probability=True, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=42))
    ]

def calculate_metrics(y_test, y_pred, model_name, fold, y_pred_prob=None):
    """Calculate and return performance metrics for a model."""
    metrics = {
        'model': model_name,
        'fold': fold + 1,
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'kappa': cohen_kappa_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    if y_pred_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_test, y_pred_prob)

    return metrics

def train_and_evaluate_models(feature, target, output_dir):
    """Train and evaluate all models using k-fold cross-validation."""
    models = get_models()
    all_metrics = []
    
    # Perform k-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in models:
        print(f"\nTraining {model_name}...")
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(feature)):
            X_train, X_test = feature.iloc[train_idx], feature.iloc[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred, model_name, fold, y_pred_prob)
            fold_metrics.append(metrics)
        
        all_metrics.extend(fold_metrics)
    
    # Convert metrics to DataFrame, handling numpy arrays
    metrics_list = []
    for metric in all_metrics:
        metric_dict = {k: v for k, v in metric.items() if k != 'confusion_matrix'}
        metric_dict['confusion_matrix'] = metric['confusion_matrix'].tolist()
        metrics_list.append(metric_dict)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Save results
    metrics_df.to_excel(os.path.join(output_dir, 'model_metrics.xlsx'), index=False)
    
    return metrics_df, models 