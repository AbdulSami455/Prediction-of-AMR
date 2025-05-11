import pandas as pd
import numpy as np
import os

def analyze_model_metrics(metrics_df, output_dir):
    """Analyze and save model performance metrics."""
    # Calculate average metrics by model
    avg_metrics = metrics_df.groupby('model').agg({
        'accuracy': 'mean',
        'balanced_accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'mcc': 'mean',
        'kappa': 'mean',
        'auc_roc': 'mean'
    }).round(4)
    
    # Calculate standard deviation of metrics
    std_metrics = metrics_df.groupby('model').agg({
        'accuracy': 'std',
        'balanced_accuracy': 'std',
        'precision': 'std',
        'recall': 'std',
        'f1_score': 'std',
        'mcc': 'std',
        'kappa': 'std',
        'auc_roc': 'std'
    }).round(4)
    
    # Save average metrics
    avg_metrics.to_csv(os.path.join(output_dir, 'model_average_metrics.csv'))
    
    # Save standard deviation metrics
    std_metrics.to_csv(os.path.join(output_dir, 'model_std_metrics.csv'))
    
    # Create detailed performance report
    performance_report = []
    for model in metrics_df['model'].unique():
        model_metrics = metrics_df[metrics_df['model'] == model]
        
        # Calculate confidence intervals (95%)
        ci_metrics = {}
        for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'kappa', 'auc_roc']:
            mean = model_metrics[metric].mean()
            std = model_metrics[metric].std()
            ci_lower = mean - 1.96 * std
            ci_upper = mean + 1.96 * std
            ci_metrics[metric] = {
                'mean': mean,
                'std': std,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        
        performance_report.append({
            'model': model,
            'metrics': ci_metrics
        })
    
    # Save detailed performance report
    with open(os.path.join(output_dir, 'model_performance_report.txt'), 'w') as f:
        for report in performance_report:
            f.write(f"\nModel: {report['model']}\n")
            f.write("-" * 50 + "\n")
            for metric, values in report['metrics'].items():
                f.write(f"{metric}:\n")
                f.write(f"  Mean: {values['mean']:.4f}\n")
                f.write(f"  Std: {values['std']:.4f}\n")
                f.write(f"  95% CI: [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}]\n")
            f.write("\n")
    
    return avg_metrics, std_metrics, performance_report

def analyze_fold_performance(metrics_df, output_dir):
    """Analyze model performance across different folds."""
    # Calculate fold-wise metrics
    fold_metrics = metrics_df.groupby(['model', 'fold']).agg({
        'accuracy': 'mean',
        'balanced_accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'mcc': 'mean',
        'kappa': 'mean',
        'auc_roc': 'mean'
    }).round(4)
    
    # Save fold-wise metrics
    fold_metrics.to_csv(os.path.join(output_dir, 'fold_performance_metrics.csv'))
    
    # Calculate fold stability (standard deviation across folds)
    fold_stability = metrics_df.groupby('model').agg({
        'accuracy': 'std',
        'balanced_accuracy': 'std',
        'precision': 'std',
        'recall': 'std',
        'f1_score': 'std',
        'mcc': 'std',
        'kappa': 'std',
        'auc_roc': 'std'
    }).round(4)
    
    # Save fold stability metrics
    fold_stability.to_csv(os.path.join(output_dir, 'fold_stability_metrics.csv'))
    
    return fold_metrics, fold_stability

def analyze_confusion_matrices(metrics_df, output_dir):
    """Analyze confusion matrices for each model."""
    confusion_matrix_stats = []
    
    for model in metrics_df['model'].unique():
        model_metrics = metrics_df[metrics_df['model'] == model]
        
        # Extract confusion matrices for each fold
        conf_matrices = []
        for _, row in model_metrics.iterrows():
            if isinstance(row['confusion_matrix'], list):
                conf_matrices.append(np.array(row['confusion_matrix']))
        
        if conf_matrices:
            # Calculate average confusion matrix
            avg_conf_matrix = np.mean(conf_matrices, axis=0)
            
            # Calculate metrics from confusion matrix
            tn, fp, fn, tp = avg_conf_matrix.ravel()
            total = tn + fp + fn + tp
            
            stats = {
                'model': model,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn,
                'true_positive': tp,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
                'accuracy': (tp + tn) / total if total > 0 else 0
            }
            
            confusion_matrix_stats.append(stats)
    
    # Create and save confusion matrix statistics
    conf_matrix_df = pd.DataFrame(confusion_matrix_stats)
    conf_matrix_df.to_csv(os.path.join(output_dir, 'confusion_matrix_stats.csv'), index=False)
    
    return conf_matrix_df 