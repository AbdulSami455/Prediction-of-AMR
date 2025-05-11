import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
import shap

def analyze_clinical_impact(model, feature, target, output_dir):
    """Analyze the potential clinical impact of the model predictions."""
    try:
        # Get model predictions
        y_pred = model.predict(feature)
        y_pred_prob = model.predict_proba(feature)[:, 1]
        
        # Calculate confusion matrix
        cm = confusion_matrix(target, y_pred)
        
        # Calculate potential impact metrics
        total_patients = len(target)
        true_positives = cm[1, 1]
        false_positives = cm[0, 1]
        true_negatives = cm[0, 0]
        false_negatives = cm[1, 0]
        
        # Calculate potential impact
        impact_metrics = {
            'Total Patients': total_patients,
            'Correctly Identified Resistant Cases': true_positives,
            'Correctly Identified Sensitive Cases': true_negatives,
            'False Resistant Predictions': false_positives,
            'False Sensitive Predictions': false_negatives,
            'Accuracy': (true_positives + true_negatives) / total_patients,
            'Potential Unnecessary Antibiotics Avoided': true_positives,
            'Potential Missed Resistant Cases': false_negatives
        }
        
        # Save impact metrics
        pd.DataFrame([impact_metrics]).to_csv(
            os.path.join(output_dir, 'clinical_impact_metrics.csv'), index=False)
        
        # Create impact visualization
        plt.figure(figsize=(10, 6))
        impact_data = pd.DataFrame({
            'Metric': ['Correct Resistant', 'Correct Sensitive', 'False Resistant', 'False Sensitive'],
            'Count': [true_positives, true_negatives, false_positives, false_negatives]
        })
        sns.barplot(data=impact_data, x='Metric', y='Count')
        plt.title('Clinical Impact of Model Predictions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clinical_impact.png'))
        plt.close()
        
        # Generate simple report
        report = f"""Clinical Impact Report
===================

Model Performance Summary:
------------------------
Total Patients Analyzed: {impact_metrics['Total Patients']}
Overall Accuracy: {impact_metrics['Accuracy']:.2%}

Clinical Impact:
--------------
- Correctly Identified Resistant Cases: {impact_metrics['Correctly Identified Resistant Cases']}
- Correctly Identified Sensitive Cases: {impact_metrics['Correctly Identified Sensitive Cases']}
- Potential Unnecessary Antibiotics Avoided: {impact_metrics['Potential Unnecessary Antibiotics Avoided']}
- Potential Missed Resistant Cases: {impact_metrics['Potential Missed Resistant Cases']}
"""
        
        # Save report
        with open(os.path.join(output_dir, 'clinical_impact_report.txt'), 'w') as f:
            f.write(report)
        
        return impact_metrics
    except Exception as e:
        print(f"Error in clinical impact analysis: {str(e)}")
        return None

def analyze_feature_importance(model, feature, output_dir):
    """Analyze feature importance using SHAP values."""
    try:
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature)
        
        # Handle both binary and multi-class cases
        if isinstance(shap_values, list):
            # For multi-class case, use the class 1 values (resistant class)
            shap_values = shap_values[1]
        
        # Plot summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, feature, show=False)
        plt.title('Feature Importance (SHAP Values)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_shap.png'))
        plt.close()
        
        # Calculate mean absolute SHAP values
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': feature.columns,
            'Importance': mean_shap_values
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Save feature importance
        feature_importance.to_csv(
            os.path.join(output_dir, 'feature_importance_shap.csv'), index=False)
        
        return feature_importance
    except Exception as e:
        print(f"Error in feature importance analysis: {str(e)}")
        # Fallback to model's built-in feature importance if available
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': feature.columns,
                    'Importance': importances
                })
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                sns.barplot(data=feature_importance.head(20), x='Importance', y='Feature')
                plt.title('Top 20 Feature Importances')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importance_fallback.png'))
                plt.close()
                
                # Save feature importance
                feature_importance.to_csv(
                    os.path.join(output_dir, 'feature_importance_fallback.csv'), index=False)
                
                return feature_importance
        except Exception as e2:
            print(f"Error in fallback feature importance analysis: {str(e2)}")
            return None

def analyze_prediction_confidence(model, feature, target, output_dir):
    """Analyze model prediction confidence and its relationship with accuracy."""
    # Get prediction probabilities
    y_pred_prob = model.predict_proba(feature)[:, 1]
    
    # Create confidence bins
    bins = np.linspace(0, 1, 11)
    confidence_data = []
    
    for i in range(len(bins)-1):
        mask = (y_pred_prob >= bins[i]) & (y_pred_prob < bins[i+1])
        if np.sum(mask) > 0:
            accuracy = np.mean(target[mask] == model.predict(feature[mask]))
            confidence_data.append({
                'Confidence Range': f'{bins[i]:.1f}-{bins[i+1]:.1f}',
                'Accuracy': accuracy,
                'Count': np.sum(mask)
            })
    
    confidence_df = pd.DataFrame(confidence_data)
    
    # Plot confidence vs accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(data=confidence_df, x='Confidence Range', y='Accuracy')
    plt.title('Model Accuracy by Prediction Confidence')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_confidence.png'))
    plt.close()
    
    # Save confidence analysis
    confidence_df.to_csv(
        os.path.join(output_dir, 'prediction_confidence.csv'), index=False)
    
    return confidence_df

def generate_clinical_report(model, feature, target, output_dir):
    """Generate a comprehensive clinical impact report."""
    # Analyze clinical impact
    impact_metrics = analyze_clinical_impact(model, feature, target, output_dir)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, feature, output_dir)
    
    # Analyze prediction confidence
    confidence_analysis = analyze_prediction_confidence(model, feature, target, output_dir)
    
    # Generate report
    report = f"""Clinical Impact Report
===================

Model Performance Summary:
------------------------
Total Patients Analyzed: {impact_metrics['Total Patients']}
Overall Accuracy: {impact_metrics['Accuracy']:.2%}

Clinical Impact:
--------------
- Correctly Identified Resistant Cases: {impact_metrics['Correctly Identified Resistant Cases']}
- Correctly Identified Sensitive Cases: {impact_metrics['Correctly Identified Sensitive Cases']}
- Potential Unnecessary Antibiotics Avoided: {impact_metrics['Potential Unnecessary Antibiotics Avoided']}
- Potential Missed Resistant Cases: {impact_metrics['Potential Missed Resistant Cases']}

Top 5 Most Important Features:
---------------------------
{feature_importance.head().to_string()}

Prediction Confidence Analysis:
---------------------------
{confidence_analysis.to_string()}
"""
    
    # Save report
    with open(os.path.join(output_dir, 'clinical_impact_report.txt'), 'w') as f:
        f.write(report)
    
    return report 