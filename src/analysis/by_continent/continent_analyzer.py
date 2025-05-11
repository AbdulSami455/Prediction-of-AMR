import pandas as pd
import numpy as np
import os

def analyze_continent_distribution(metadata, output_dir):
    """Analyze and save continent-specific statistics."""
    # Calculate basic statistics
    continent_stats = metadata['Continent'].value_counts()
    total_samples = len(metadata)
    
    # Calculate resistance rates by continent
    antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
    continent_antibiotic_stats = {}
    
    for continent in continent_stats.index:
        continent_data = metadata[metadata['Continent'] == continent]
        stats = {}
        
        for antibiotic in antibiotics:
            if antibiotic in continent_data.columns:
                resistance_rate = (continent_data[antibiotic] == 'Resistant').mean() * 100
                stats[antibiotic] = {
                    'resistance_rate': resistance_rate,
                    'total_cases': continent_data[antibiotic].count(),
                    'resistant_cases': (continent_data[antibiotic] == 'Resistant').sum(),
                    'sensitive_cases': (continent_data[antibiotic] == 'Sensitive').sum()
                }
        
        continent_antibiotic_stats[continent] = stats
    
    # Save continent distribution statistics
    continent_dist = pd.DataFrame({
        'Continent': continent_stats.index,
        'Sample_Count': continent_stats.values,
        'Percentage': (continent_stats.values / total_samples * 100).round(2)
    })
    continent_dist.to_csv(os.path.join(output_dir, 'continent_distribution_stats.csv'), index=False)
    
    # Save detailed antibiotic resistance statistics by continent
    detailed_stats = []
    for continent, stats in continent_antibiotic_stats.items():
        for antibiotic, values in stats.items():
            detailed_stats.append({
                'Continent': continent,
                'Antibiotic': antibiotic,
                'Resistance_Rate': values['resistance_rate'],
                'Total_Cases': values['total_cases'],
                'Resistant_Cases': values['resistant_cases'],
                'Sensitive_Cases': values['sensitive_cases']
            })
    
    detailed_stats_df = pd.DataFrame(detailed_stats)
    detailed_stats_df.to_csv(os.path.join(output_dir, 'continent_antibiotic_stats.csv'), index=False)
    
    return continent_dist, detailed_stats_df

def analyze_continent_trends(metadata, output_dir):
    """Analyze trends in antibiotic resistance by continent over time."""
    antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
    trend_stats = []
    
    for antibiotic in antibiotics:
        # Filter data for current antibiotic
        ab_df = metadata[metadata[antibiotic].notna()][['Year', 'Continent', antibiotic]]
        
        # Group by Year and Continent
        yearly_stats = ab_df.groupby(['Year', 'Continent'])[antibiotic].agg([
            ('total_cases', 'count'),
            ('resistant_cases', lambda x: (x == 'Resistant').sum()),
            ('sensitive_cases', lambda x: (x == 'Sensitive').sum())
        ]).reset_index()
        
        # Calculate resistance rate
        yearly_stats['resistance_rate'] = (yearly_stats['resistant_cases'] / 
                                         yearly_stats['total_cases'] * 100).round(2)
        
        # Add antibiotic column
        yearly_stats['antibiotic'] = antibiotic
        
        trend_stats.append(yearly_stats)
    
    # Combine all trend statistics
    all_trends = pd.concat(trend_stats, ignore_index=True)
    
    # Save trend statistics
    all_trends.to_csv(os.path.join(output_dir, 'continent_trends_stats.csv'), index=False)
    
    return all_trends 