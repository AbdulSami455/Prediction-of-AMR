import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_snps(unitig_data, target, output_dir):
    """Analyze SNPs and their relationship with antibiotic resistance."""
    try:
        # Check if data shapes match
        if len(unitig_data) != len(target):
            print(f"Warning: Data shape mismatch. Unitig data: {len(unitig_data)}, Target data: {len(target)}")
            print("Skipping SNP analysis due to data mismatch.")
            return None
        
        # Select top k most important SNPs
        selector = SelectKBest(f_classif, k=100)
        selector.fit(unitig_data, target)
        
        # Get feature scores
        scores = pd.DataFrame({
            'SNP': unitig_data.columns,
            'Score': selector.scores_
        })
        scores = scores.sort_values('Score', ascending=False)
        
        # Plot top SNPs
        plt.figure(figsize=(12, 6))
        sns.barplot(data=scores.head(20), x='Score', y='SNP')
        plt.title('Top 20 Most Important SNPs')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_snps.png'))
        plt.close()
        
        return scores
    except Exception as e:
        print(f"Error in SNP analysis: {str(e)}")
        return None

def analyze_args(unitig_data, target, output_dir):
    """Analyze Antibiotic Resistance Genes (ARGs) patterns."""
    try:
        # Check if data shapes match
        if len(unitig_data) != len(target):
            print(f"Warning: Data shape mismatch. Unitig data: {len(unitig_data)}, Target data: {len(target)}")
            print("Skipping ARG analysis due to data mismatch.")
            return None
        
        # Calculate correlation between ARGs and resistance
        correlations = []
        for col in unitig_data.columns:
            corr = np.corrcoef(unitig_data[col], target)[0, 1]
            correlations.append({
                'ARG': col,
                'Correlation': corr
            })
        
        correlations_df = pd.DataFrame(correlations)
        correlations_df = correlations_df.sort_values('Correlation', ascending=False)
        
        # Plot top ARGs
        plt.figure(figsize=(12, 6))
        sns.barplot(data=correlations_df.head(20), x='Correlation', y='ARG')
        plt.title('Top 20 Most Correlated ARGs')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_args.png'))
        plt.close()
        
        return correlations_df
    except Exception as e:
        print(f"Error in ARG analysis: {str(e)}")
        return None

def analyze_genetic_patterns(unitig_data, target, metadata, output_dir):
    """Analyze genetic patterns and their relationship with resistance."""
    try:
        # Check if data shapes match
        if len(unitig_data) != len(target) or len(unitig_data) != len(metadata):
            print(f"Warning: Data shape mismatch. Unitig data: {len(unitig_data)}, Target data: {len(target)}, Metadata: {len(metadata)}")
            print("Skipping genetic patterns analysis due to data mismatch.")
            return None
        
        # Combine genetic data with metadata
        combined_data = pd.concat([unitig_data, metadata], axis=1)
        
        # Analyze genetic patterns by continent
        continent_patterns = combined_data.groupby('Continent').agg({
            col: ['mean', 'std'] for col in unitig_data.columns
        })
        
        # Save patterns
        continent_patterns.to_csv(os.path.join(output_dir, 'genetic_patterns_by_continent.csv'))
        
        # Create heatmap of genetic patterns
        plt.figure(figsize=(15, 10))
        sns.heatmap(continent_patterns.xs('mean', level=1, axis=1).T, 
                    cmap='YlOrRd', 
                    center=0)
        plt.title('Genetic Patterns by Continent')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'genetic_patterns_heatmap.png'))
        plt.close()
        
        return continent_patterns
    except Exception as e:
        print(f"Error in genetic patterns analysis: {str(e)}")
        return None

def analyze_resistance_mechanisms(unitig_data, target, output_dir):
    """Analyze potential resistance mechanisms based on genetic data."""
    try:
        # Check if data shapes match
        if len(unitig_data) != len(target):
            print(f"Warning: Data shape mismatch. Unitig data: {len(unitig_data)}, Target data: {len(target)}")
            print("Skipping resistance mechanisms analysis due to data mismatch.")
            return None
        
        # Calculate feature importance for resistance prediction
        selector = SelectKBest(f_classif, k=50)
        selector.fit(unitig_data, target)
        
        # Get feature importance scores
        importance_scores = pd.DataFrame({
            'Feature': unitig_data.columns,
            'Importance': selector.scores_
        })
        importance_scores = importance_scores.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_scores.head(20), x='Importance', y='Feature')
        plt.title('Top 20 Most Important Genetic Features for Resistance Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resistance_mechanisms.png'))
        plt.close()
        
        # Save importance scores
        importance_scores.to_csv(os.path.join(output_dir, 'resistance_mechanisms.csv'), index=False)
        
        return importance_scores
    except Exception as e:
        print(f"Error in resistance mechanisms analysis: {str(e)}")
        return None 