import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
import folium
from folium.plugins import HeatMap

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

def plot_antibiotic_trends(metadata, output_dir):
    """Plot trends in antibiotic resistance over time by continent for all antibiotics."""
    antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
    
    for antibiotic in antibiotics:
        print(f"\nAnalyzing trends for {antibiotic}...")
        # Filter data for current antibiotic
        ab_df = metadata[metadata[antibiotic].notna()][['Year', 'Continent', antibiotic]]
        
        # Create pivot table
        pivot_df = pd.pivot_table(ab_df, index=['Year', 'Continent'], 
                                columns=antibiotic, aggfunc='size', fill_value=0)
        pivot_df = pivot_df.reset_index()
        
        # Create subplots for each continent
        continents = pivot_df['Continent'].unique()
        fig, axes = plt.subplots(3, 2, figsize=(20, 30))
        fig.suptitle(f'{antibiotic.upper()} Resistance Trends by Continent', fontsize=24, y=0.95)
        
        for i, continent in enumerate(continents):
            ax = axes[i // 2, i % 2]
            continent_data = pivot_df[pivot_df['Continent'] == continent]
            
            # Plot trends
            sns.lineplot(data=continent_data, x='Year', y='Resistant', 
                        ax=ax, label='Resistant', marker='o', color='red')
            sns.lineplot(data=continent_data, x='Year', y='Sensitive', 
                        ax=ax, label='Sensitive', marker='o', color='blue')
            
            # Calculate and display statistics
            total_cases = continent_data['Resistant'].sum() + continent_data['Sensitive'].sum()
            resistance_rate = (continent_data['Resistant'].sum() / total_cases * 100).round(2)
            
            ax.set_title(f'{continent}\nResistance Rate: {resistance_rate}%', fontsize=18)
            ax.set_xlabel('Year', fontsize=14)
            ax.set_ylabel('Count', fontsize=14)
            ax.legend(fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Remove empty subplots if any
        for i in range(len(continents), 6):
            fig.delaxes(axes[i // 2, i % 2])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'{antibiotic}_trends.png'))
        plt.close()
        
        # Create summary statistics
        summary_stats = []
        for continent in continents:
            continent_data = pivot_df[pivot_df['Continent'] == continent]
            total_cases = continent_data['Resistant'].sum() + continent_data['Sensitive'].sum()
            resistance_rate = (continent_data['Resistant'].sum() / total_cases * 100).round(2)
            summary_stats.append({
                'Continent': continent,
                'Total Cases': total_cases,
                'Resistance Rate (%)': resistance_rate,
                'Resistant Cases': continent_data['Resistant'].sum(),
                'Sensitive Cases': continent_data['Sensitive'].sum()
            })
        
        # Save summary statistics to CSV
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(os.path.join(output_dir, f'{antibiotic}_trends_summary.csv'), index=False)

def plot_model_comparison(metrics_df, metric_name, title, filename, output_dir):
    """Plot comparison of models based on a specific metric."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='model', y=metric_name)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_roc_curves(models, X, y, filename, output_dir):
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
    plt.savefig(os.path.join(output_dir, filename))
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

def plot_geographical_distribution(metadata, output_dir):
    """Create geographical plots showing antibiotic resistance distribution."""
    antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
    
    # Create a base map centered at the mean coordinates
    center_lat = metadata['Latitude'].mean()
    center_lon = metadata['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
    
    # Create a separate map for each antibiotic
    for antibiotic in antibiotics:
        # Filter data for current antibiotic
        ab_data = metadata[metadata[antibiotic].notna()].copy()
        
        # Create a new map for this antibiotic
        m_ab = folium.Map(location=[center_lat, center_lon], zoom_start=2)
        
        # Add markers for each location
        for idx, row in ab_data.iterrows():
            # Determine color based on resistance status
            color = 'red' if row[antibiotic] == 'Resistant' else 'green'
            
            # Create popup content
            popup_content = f"""
            <b>Location:</b> {row['Country']}<br>
            <b>Year:</b> {row['Year']}<br>
            <b>Status:</b> {row[antibiotic]}<br>
            <b>Latitude:</b> {row['Latitude']}<br>
            <b>Longitude:</b> {row['Longitude']}
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m_ab)
        
        # Add a legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    padding: 10px;
                    ">
        <p><strong>{antibiotic.upper()} Resistance</strong></p>
        <p><span style="color:red;">●</span> Resistant</p>
        <p><span style="color:green;">●</span> Sensitive</p>
        </div>
        '''
        m_ab.get_root().html.add_child(folium.Element(legend_html))
        
        # Save the map
        m_ab.save(os.path.join(output_dir, f'geographical_distribution_{antibiotic}.html'))
        
        # Create a heatmap version
        m_heat = folium.Map(location=[center_lat, center_lon], zoom_start=2)
        
        # Prepare heatmap data
        heat_data = []
        for idx, row in ab_data.iterrows():
            if row[antibiotic] == 'Resistant':
                heat_data.append([row['Latitude'], row['Longitude'], 1])
        
        # Add heatmap layer
        HeatMap(heat_data).add_to(m_heat)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 60px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    padding: 10px;
                    ">
        <p><strong>{antibiotic.upper()} Resistance Heatmap</strong></p>
        <p>Darker areas indicate higher resistance</p>
        </div>
        '''
        m_heat.get_root().html.add_child(folium.Element(legend_html))
        
        # Save the heatmap
        m_heat.save(os.path.join(output_dir, f'geographical_heatmap_{antibiotic}.html'))
    
    # Create a combined map with all antibiotics
    m_combined = folium.Map(location=[center_lat, center_lon], zoom_start=2)
    
    # Add markers for each location with multiple antibiotic resistance
    for idx, row in metadata.iterrows():
        # Count resistant antibiotics
        resistant_count = sum(1 for ab in antibiotics if row.get(ab) == 'Resistant')
        if resistant_count > 0:
            # Create popup content
            popup_content = f"""
            <b>Location:</b> {row['Country']}<br>
            <b>Year:</b> {row['Year']}<br>
            <b>Resistant Antibiotics:</b> {resistant_count}<br>
            """
            for ab in antibiotics:
                if pd.notna(row.get(ab)):
                    popup_content += f"<b>{ab}:</b> {row[ab]}<br>"
            
            # Add marker with size based on number of resistant antibiotics
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5 + resistant_count * 2,  # Size increases with resistance count
                popup=folium.Popup(popup_content, max_width=300),
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7
            ).add_to(m_combined)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                ">
    <p><strong>Multiple Antibiotic Resistance</strong></p>
    <p>Circle size indicates number of resistant antibiotics</p>
    <p>Click markers for detailed information</p>
    </div>
    '''
    m_combined.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the combined map
    m_combined.save(os.path.join(output_dir, 'geographical_distribution_combined.html')) 