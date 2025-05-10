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

def plot_continent_distribution(metadata, output_dir):
    """Plot the distribution of samples across continents."""
    plt.figure(figsize=(10, 8))
    continent_counts = metadata['Continent'].value_counts()
    plt.pie(continent_counts, labels=continent_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=sns.color_palette("Set2", n_colors=len(continent_counts)))
    plt.title('Distribution of Samples Across Continents')
    plt.savefig(os.path.join(output_dir, 'continent_distribution.png'))
    plt.close()

def plot_antibiotic_distribution(metadata, output_dir):
    """Plot the distribution of antibiotic resistance data."""
    antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
    
    # Count non-null values
    antibiotic_counts = metadata[antibiotics].notna().sum()
    percentages = (antibiotic_counts / len(metadata) * 100).round(2)
    
    # Create DataFrame for plotting
    dist_df = pd.DataFrame({
        'Count': antibiotic_counts,
        'Percentage': percentages
    }).sort_values('Count', ascending=False)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=dist_df.index, y='Count', data=dist_df, 
                    palette=sns.color_palette("Set2", n_colors=len(antibiotics)))
    
    plt.title('Distribution of Antibiotic Data')
    plt.xlabel('Antibiotics')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for i, v in enumerate(dist_df['Count']):
        ax.text(i, v, f"{v} ({dist_df['Percentage'].iloc[i]}%)", 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'antibiotic_distribution.png'))
    plt.close()

def plot_sensitivity_distribution(metadata, output_dir):
    """Plot the distribution of sensitive and resistant cases for each antibiotic."""
    antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
    
    # Create a dictionary to store the counts
    counts = {}
    for antibiotic in antibiotics:
        value_counts = metadata[antibiotic].value_counts()
        counts[antibiotic] = {
            'Sensitive': value_counts.get('Sensitive', 0),
            'Resistant': value_counts.get('Resistant', 0)
        }
    
    # Prepare data for plotting
    antibiotics_names = list(counts.keys())
    sensitive_counts = [counts[ab]['Sensitive'] for ab in antibiotics_names]
    resistant_counts = [counts[ab]['Resistant'] for ab in antibiotics_names]
    
    plt.figure(figsize=(12, 6))
    x = range(len(antibiotics_names))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], sensitive_counts, width, 
            label='Sensitive', color='lightblue')
    plt.bar([i + width/2 for i in x], resistant_counts, width, 
            label='Resistant', color='lightcoral')
    
    plt.xlabel('Antibiotics')
    plt.ylabel('Count')
    plt.title('Distribution of Resistance and Sensitive Cases')
    plt.xticks(x, antibiotics_names, rotation=45)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(sensitive_counts):
        plt.text(i - width/2, v, str(v), ha='center', va='bottom')
    for i, v in enumerate(resistant_counts):
        plt.text(i + width/2, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_distribution.png'))
    plt.close()

def plot_feature_importance(model, feature_names, model_name, output_dir):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.barh(range(20), importances[indices[:20]], align='center')
        plt.yticks(range(20), [feature_names[i] for i in indices[:20]])
        plt.gca().invert_yaxis()
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_importance_{model_name}.png'))
        plt.close()

def plot_azm_trends(metadata, output_dir):
    """Plot trends in Azithromycin resistance over time by continent."""
    azm_df = metadata[metadata['azm_sr'].notna()][['Year', 'Continent', 'azm_sr']]
    
    # Create pivot table
    pivot_df = pd.pivot_table(azm_df, index=['Year', 'Continent'], 
                             columns='azm_sr', aggfunc='size', fill_value=0)
    pivot_df = pivot_df.reset_index()
    
    # Create subplots for each continent
    continents = pivot_df['Continent'].unique()
    fig, axes = plt.subplots(3, 2, figsize=(20, 30))
    fig.suptitle('Azithromycin Resistance Trends by Continent', fontsize=24, y=0.95)
    
    for i, continent in enumerate(continents):
        ax = axes[i // 2, i % 2]
        continent_data = pivot_df[pivot_df['Continent'] == continent]
        
        sns.lineplot(data=continent_data, x='Year', y='Resistant', 
                    ax=ax, label='Resistant', marker='o')
        sns.lineplot(data=continent_data, x='Year', y='Sensitive', 
                    ax=ax, label='Sensitive', marker='o')
        
        ax.set_title(f'{continent}', fontsize=18)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='x', rotation=45)
    
    # Remove the empty subplot
    axes[2, 1].remove()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'azm_trends.png'))
    plt.close()

def plot_confusion_matrices(metrics_df, output_dir):
    """Plot confusion matrices for each model."""
    models = metrics_df['model'].unique()
    
    for model in models:
        model_metrics = metrics_df[metrics_df['model'] == model].iloc[0]
        conf_matrix = model_metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model}.png'))
        plt.close()

def plot_learning_curves(model, X, y, model_name, output_dir):
    """Plot learning curves for a model."""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='balanced_accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    
    plt.title(f'Learning Curves - {model_name}')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'learning_curve_{model_name}.png'))
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
    
    # Create output directory
    output_dir = os.path.join(project_root, 'Output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    target, feature, metadata = prepare_target_feature(
        unitig_file,
        metadata_file,
        'azm_sr'
    )
    
    # Generate additional plots
    print("\nGenerating additional visualizations...")
    plot_continent_distribution(metadata, output_dir)
    plot_antibiotic_distribution(metadata, output_dir)
    plot_sensitivity_distribution(metadata, output_dir)
    plot_azm_trends(metadata, output_dir)
    
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
        
        # Plot learning curves
        plot_learning_curves(model, feature, target, model_name, output_dir)
        
        # Plot feature importance for tree-based models
        if model_name in ['Random Forest', 'Gradient Boosting']:
            plot_feature_importance(model, feature.columns, model_name, output_dir)
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save results
    metrics_df.to_excel(os.path.join(output_dir, 'model_metrics.xlsx'), index=False)
    
    # Generate plots
    plot_model_comparison(metrics_df, 'accuracy', 'Model Accuracy Comparison', 'accuracy_comparison.png')
    plot_model_comparison(metrics_df, 'balanced_accuracy', 'Model Balanced Accuracy Comparison', 'balanced_accuracy_comparison.png')
    plot_roc_curves(models, feature, target, 'roc_curves.png')
    plot_confusion_matrices(metrics_df, output_dir)
    
    print("\nAnalysis complete. Results saved in the Output directory.")

if __name__ == "__main__":
    main()