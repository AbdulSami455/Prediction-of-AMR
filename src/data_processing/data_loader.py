import pandas as pd
import os
import sys

def load_unitig_data(file_path):
    """Load and prepare unitig data."""
    try:
        print(f"Loading unitig data from: {file_path}")
        sequence_data = pd.read_csv(file_path, sep=" ", index_col=0, low_memory=False)
        return sequence_data.transpose()
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data loading: {e}")
        sys.exit(1)

def load_metadata(file_path):
    """Load and prepare metadata."""
    try:
        print(f"Loading metadata from: {file_path}")
        return pd.read_csv(file_path, index_col=0)
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during metadata loading: {e}")
        sys.exit(1)

def prepare_target_feature(unitig_data, metadata, target_column):
    """Prepare target and feature datasets for a given antibiotic resistance data."""
    try:
        # Drop rows with null values in the target column
        cleaned_metadata = metadata.dropna(subset=[target_column])

        # Convert the target column to binary: Sensitive=0, Resistant=1
        cleaned_metadata[f'{target_column}_binary'] = cleaned_metadata[target_column].map({'Sensitive': 0, 'Resistant': 1})

        # Keep only the binary target column
        target_data = cleaned_metadata[[f'{target_column}_binary']]

        # Align the sequence data index with the target dataset index
        aligned_sequence_data = unitig_data[unitig_data.index.isin(target_data.index)]
        aligned_target_data = target_data[target_data.index.isin(aligned_sequence_data.index)]

        # Convert the target data to a Series
        target = aligned_target_data.squeeze()
        feature = aligned_sequence_data

        print(f"Data preparation complete. Feature shape: {feature.shape}, Target shape: {target.shape}")
        return target, feature, cleaned_metadata

    except Exception as e:
        print(f"Error during data preparation: {e}")
        sys.exit(1) 