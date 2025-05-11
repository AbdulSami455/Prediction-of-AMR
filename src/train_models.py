import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def get_models():
    """Return a list of models to train."""
    return [
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('SVM', SVC(probability=True, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=42))
    ]

def train_and_save_models(feature, target, output_dir, antibiotic_name):
    """Train and save models for a specific antibiotic."""
    print(f"\nTraining models for {antibiotic_name}...")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(output_dir, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create and save label encoder for target
    target_encoder = LabelEncoder()
    target_encoded = target_encoder.fit_transform(target)
    target_encoder_path = os.path.join(models_dir, f'{antibiotic_name}_encoder.joblib')
    joblib.dump(target_encoder, target_encoder_path)
    print(f"Saved target encoder for {antibiotic_name}")
    
    # Create and save feature encoder
    feature_encoder = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Country'])
        ],
        remainder='drop'
    )
    
    # Fit and transform features
    feature_encoded = feature_encoder.fit_transform(feature)
    
    # Save feature encoder
    feature_encoder_path = os.path.join(models_dir, f'{antibiotic_name}_feature_encoder.joblib')
    joblib.dump(feature_encoder, feature_encoder_path)
    print(f"Saved feature encoder for {antibiotic_name}")
    
    # Train and save each model
    for model_name, model in get_models():
        print(f"Training {model_name}...")
        
        # Train the model
        model.fit(feature_encoded, target_encoded)
        
        # Save the model
        model_path = os.path.join(models_dir, f'{antibiotic_name}_{model_name}.joblib')
        joblib.dump(model, model_path)
        print(f"Saved {model_name} model for {antibiotic_name}")

def main():
    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Set up output directory
    output_dir = os.path.join(project_root, 'Output', 'model_metrics')
    os.makedirs(output_dir, exist_ok=True)
    
    # List of antibiotics to train models for
    antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
    
    # Load data
    print("Loading data...")
    metadata_file = os.path.join(project_root, 'data', 'metadata_remane_original.csv')
    metadata = pd.read_csv(metadata_file)
    
    # Train models for each antibiotic
    for ab in antibiotics:
        print(f"\nProcessing {ab}...")
        
        # Prepare target and feature data
        target = metadata[ab]
        feature = metadata[['Gender', 'Country']]
        
        # Train and save models
        train_and_save_models(feature, target, output_dir, ab)
    
    print("\nModel training complete. Models saved in Output/model_metrics/saved_models")

if __name__ == "__main__":
    main() 