import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score, 
                           confusion_matrix, classification_report, precision_score, 
                           recall_score, f1_score, matthews_corrcoef, cohen_kappa_score,
                           roc_curve, auc)
import warnings
import time
import os
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# Create Output directory if it doesn't exist
os.makedirs('Output', exist_ok=True)

def prepare_target_feature(unitig_file_path, metadata_file_path, target_column):
    """Prepares the target and feature datasets for a given antibiotic resistance data."""
    try:
        # Load the sequence (unitig) data
        print(f"Loading unitig data from: {unitig_file_path}")
        sequence_data = pd.read_csv(unitig_file_path, sep=" ", index_col=0, low_memory=False)
        sequence_data_transposed = sequence_data.transpose()
        
        # Load the metadata
        print(f"Loading metadata from: {metadata_file_path}")
        metadata = pd.read_csv(metadata_file_path, index_col=0)
        
        # Drop rows with null values in the target column
        cleaned_metadata = metadata.dropna(subset=[target_column])
        
        # Convert the target column to binary: Sensitive=0, Resistant=1
        cleaned_metadata[f'{target_column}_binary'] = cleaned_metadata[target_column].map({'Sensitive': 0, 'Resistant': 1})
        
        # Keep only the binary target column
        target_data = cleaned_metadata[[f'{target_column}_binary']]
        
        # Align the sequence data index with the target dataset index
        aligned_sequence_data = sequence_data_transposed[sequence_data_transposed.index.isin(target_data.index)]
        aligned_target_data = target_data[target_data.index.isin(aligned_sequence_data.index)]
        
        # Convert the target data to a Series
        target = aligned_target_data.squeeze()
        feature = aligned_sequence_data
        
        print(f"Data preparation complete. Feature shape: {feature.shape}, Target shape: {target.shape}")
        return target, feature, cleaned_metadata
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data preparation: {e}")
        sys.exit(1)

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

def plot_model_comparison(metrics_df, metric_name, title, filename):
    """Plot comparison of models based on a specific metric."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='model', y=metric_name)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('Output', filename))
    plt.close()

def plot_roc_curves(models, X, y, filename):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models:
        y_pred_prob = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('Output', filename))
    plt.close()

def main():
    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Construct file paths relative to project root
    unitig_file = os.path.join(project_root, 'data', 'azm_sr_gwas_filtered_unitigs.Rtab')
    metadata_file = os.path.join(project_root, 'data', 'metadata_remane_original.csv')
    
    print(f"Project root: {project_root}")
    print(f"Unitig file path: {unitig_file}")
    print(f"Metadata file path: {metadata_file}")
    
    # Load and prepare data
    target, feature, metadata = prepare_target_feature(
        unitig_file,
        metadata_file,
        'azm_sr'
    )
    
    # Define models
    models = [
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('SVM', SVC(probability=True, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=42))
    ]
    
    # Initialize lists to store results
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
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save results
    output_dir = os.path.join(project_root, 'Output')
    os.makedirs(output_dir, exist_ok=True)
    metrics_df.to_excel(os.path.join(output_dir, 'model_metrics.xlsx'), index=False)
    
    # Generate plots
    plot_model_comparison(metrics_df, 'accuracy', 'Model Accuracy Comparison', 'accuracy_comparison.png')
    plot_model_comparison(metrics_df, 'balanced_accuracy', 'Model Balanced Accuracy Comparison', 'balanced_accuracy_comparison.png')
    plot_roc_curves(models, feature, target, 'roc_curves.png')
    
    print("\nAnalysis complete. Results saved in the Output directory.")

if __name__ == "__main__":
    main()