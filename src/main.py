import os
import sys
from data_processing.data_loader import load_unitig_data, load_metadata, prepare_target_feature
from visualization.plotting import (plot_continent_distribution, plot_antibiotic_distribution,
                                  plot_sensitivity_distribution, plot_antibiotic_trends,
                                  plot_model_comparison, plot_roc_curves, plot_confusion_matrices,
                                  plot_learning_curves, plot_feature_importance,
                                  plot_geographical_distribution)
from models.model_trainer import train_and_evaluate_models
from analysis.by_continent.continent_analyzer import analyze_continent_distribution, analyze_continent_trends
from analysis.by_antibiotic.antibiotic_analyzer import (analyze_antibiotic_distribution,
                                                      analyze_antibiotic_trends,
                                                      analyze_antibiotic_correlations,
                                                      analyze_antibiotic_combinations)
from analysis.metrics.metrics_analyzer import (analyze_model_metrics, analyze_fold_performance,
                                             analyze_confusion_matrices)

def main():
    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Create output directories
    output_dir = os.path.join(project_root, 'Output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different types of analysis
    analysis_dirs = {
        'continent': os.path.join(output_dir, 'continent_analysis'),
        'antibiotic': os.path.join(output_dir, 'antibiotic_analysis'),
        'metrics': os.path.join(output_dir, 'model_metrics'),
        'plots': os.path.join(output_dir, 'plots'),
        'maps': os.path.join(output_dir, 'maps')  # New directory for geographical maps
    }
    
    for dir_path in analysis_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Load data
    unitig_file = os.path.join(project_root, 'data', 'azm_sr_gwas_filtered_unitigs.Rtab')
    metadata_file = os.path.join(project_root, 'data', 'metadata_remane_original.csv')
    
    print("Loading data...")
    unitig_data = load_unitig_data(unitig_file)
    metadata = load_metadata(metadata_file)
    
    # Prepare target and feature data
    target, feature, cleaned_metadata = prepare_target_feature(
        unitig_data,
        metadata,
        'azm_sr'
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_continent_distribution(cleaned_metadata, analysis_dirs['plots'])
    plot_antibiotic_distribution(cleaned_metadata, analysis_dirs['plots'])
    plot_sensitivity_distribution(cleaned_metadata, analysis_dirs['plots'])
    plot_antibiotic_trends(cleaned_metadata, analysis_dirs['plots'])
    
    # Generate geographical visualizations
    print("\nGenerating geographical visualizations...")
    plot_geographical_distribution(cleaned_metadata, analysis_dirs['maps'])
    
    # Analyze continent-specific data
    print("\nAnalyzing continent-specific data...")
    continent_dist, continent_ab_stats = analyze_continent_distribution(
        cleaned_metadata, analysis_dirs['continent'])
    continent_trends = analyze_continent_trends(cleaned_metadata, analysis_dirs['continent'])
    
    # Analyze antibiotic-specific data
    print("\nAnalyzing antibiotic-specific data...")
    ab_stats = analyze_antibiotic_distribution(cleaned_metadata, analysis_dirs['antibiotic'])
    ab_trends = analyze_antibiotic_trends(cleaned_metadata, analysis_dirs['antibiotic'])
    ab_correlations = analyze_antibiotic_correlations(cleaned_metadata, analysis_dirs['antibiotic'])
    ab_patterns = analyze_antibiotic_combinations(cleaned_metadata, analysis_dirs['antibiotic'])
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    metrics_df, models = train_and_evaluate_models(feature, target, analysis_dirs['metrics'])
    
    # Generate model-related visualizations
    print("\nGenerating model visualizations...")
    plot_model_comparison(metrics_df, 'accuracy', 'Model Accuracy Comparison',
                         'accuracy_comparison.png', analysis_dirs['plots'])
    plot_model_comparison(metrics_df, 'balanced_accuracy', 'Model Balanced Accuracy Comparison',
                         'balanced_accuracy_comparison.png', analysis_dirs['plots'])
    plot_roc_curves(models, feature, target, 'roc_curves.png', analysis_dirs['plots'])
    plot_confusion_matrices(metrics_df, analysis_dirs['plots'])
    
    # Analyze model metrics
    print("\nAnalyzing model metrics...")
    avg_metrics, std_metrics, performance_report = analyze_model_metrics(
        metrics_df, analysis_dirs['metrics'])
    fold_metrics, fold_stability = analyze_fold_performance(
        metrics_df, analysis_dirs['metrics'])
    conf_matrix_stats = analyze_confusion_matrices(
        metrics_df, analysis_dirs['metrics'])
    
    # Generate learning curves and feature importance plots
    print("\nGenerating learning curves and feature importance plots...")
    for model_name, model in models:
        plot_learning_curves(model, feature, target, model_name, analysis_dirs['plots'])
        if model_name in ['Random Forest', 'Gradient Boosting']:
            plot_feature_importance(model, feature.columns, model_name, analysis_dirs['plots'])
    
    print("\nAnalysis complete. Results saved in the Output directory.")

if __name__ == "__main__":
    main() 