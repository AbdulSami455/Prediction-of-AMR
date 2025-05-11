import pandas as pd
import numpy as np
import os

def analyze_antibiotic_distribution(metadata, output_dir):
    """Analyze and save antibiotic-specific statistics."""
    antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
    
    # Calculate basic statistics for each antibiotic
    antibiotic_stats = []
    for antibiotic in antibiotics:
        if antibiotic in metadata.columns:
            total_cases = metadata[antibiotic].count()
            resistant_cases = (metadata[antibiotic] == 'Resistant').sum()
            sensitive_cases = (metadata[antibiotic] == 'Sensitive').sum()
            resistance_rate = (resistant_cases / total_cases * 100).round(2) if total_cases > 0 else 0
            
            antibiotic_stats.append({
                'Antibiotic': antibiotic,
                'Total_Cases': total_cases,
                'Resistant_Cases': resistant_cases,
                'Sensitive_Cases': sensitive_cases,
                'Resistance_Rate': resistance_rate,
                'Data_Availability': (total_cases / len(metadata) * 100).round(2)
            })
    
    # Create and save antibiotic statistics
    stats_df = pd.DataFrame(antibiotic_stats)
    stats_df.to_csv(os.path.join(output_dir, 'antibiotic_distribution_stats.csv'), index=False)
    
    return stats_df

def analyze_antibiotic_trends(metadata, output_dir):
    """Analyze trends in antibiotic resistance over time."""
    antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
    trend_stats = []
    
    for antibiotic in antibiotics:
        if antibiotic in metadata.columns:
            # Filter data for current antibiotic
            ab_df = metadata[metadata[antibiotic].notna()][['Year', antibiotic]]
            
            # Group by Year
            yearly_stats = ab_df.groupby('Year')[antibiotic].agg([
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
    all_trends.to_csv(os.path.join(output_dir, 'antibiotic_trends_stats.csv'), index=False)
    
    return all_trends

def analyze_antibiotic_correlations(metadata, output_dir):
    """Analyze correlations between different antibiotics."""
    antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
    
    # Convert resistance status to binary (1 for Resistant, 0 for Sensitive)
    binary_data = metadata[antibiotics].apply(lambda x: (x == 'Resistant').astype(int))
    
    # Calculate correlation matrix
    correlation_matrix = binary_data.corr()
    
    # Save correlation matrix
    correlation_matrix.to_csv(os.path.join(output_dir, 'antibiotic_correlations.csv'))
    
    return correlation_matrix

def analyze_antibiotic_combinations(metadata, output_dir):
    """Analyze patterns of resistance across multiple antibiotics."""
    antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
    
    # Convert resistance status to binary
    binary_data = metadata[antibiotics].apply(lambda x: (x == 'Resistant').astype(int))
    
    # Calculate resistance patterns
    resistance_patterns = []
    for i in range(len(antibiotics)):
        for j in range(i + 1, len(antibiotics)):
            ab1, ab2 = antibiotics[i], antibiotics[j]
            pattern_stats = {
                'Antibiotic_1': ab1,
                'Antibiotic_2': ab2,
                'Both_Resistant': ((binary_data[ab1] == 1) & (binary_data[ab2] == 1)).sum(),
                'Both_Sensitive': ((binary_data[ab1] == 0) & (binary_data[ab2] == 0)).sum(),
                'Mixed': ((binary_data[ab1] != binary_data[ab2])).sum()
            }
            resistance_patterns.append(pattern_stats)
    
    # Create and save pattern statistics
    patterns_df = pd.DataFrame(resistance_patterns)
    patterns_df.to_csv(os.path.join(output_dir, 'antibiotic_resistance_patterns.csv'), index=False)
    
    return patterns_df 