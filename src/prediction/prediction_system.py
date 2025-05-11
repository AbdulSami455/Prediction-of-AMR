import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ResistancePredictor:
    def __init__(self, models_dir):
        """Initialize the predictor with trained models."""
        logger.info(f"Initializing ResistancePredictor with models_dir: {models_dir}")
        self.models_dir = models_dir
        self.models = {}
        self.label_encoders = {}
        self.feature_encoders = {}
        self.load_models()
        
    def load_models(self):
        """Load all trained models and their associated encoders."""
        logger.info("Starting to load models and encoders...")
        antibiotics = ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']
        model_types = ['Random Forest', 'SVM', 'Gradient Boosting']
        
        for ab in antibiotics:
            logger.info(f"\nProcessing antibiotic: {ab}")
            self.models[ab] = {}
            self.label_encoders[ab] = {}
            
            # Load feature encoder
            feature_encoder_path = os.path.join(self.models_dir, f'{ab}_feature_encoder.joblib')
            logger.debug(f"Looking for feature encoder at: {feature_encoder_path}")
            if os.path.exists(feature_encoder_path):
                try:
                    self.feature_encoders[ab] = joblib.load(feature_encoder_path)
                    logger.info(f"Successfully loaded feature encoder for {ab}")
                except Exception as e:
                    logger.error(f"Error loading feature encoder for {ab}: {str(e)}")
            else:
                logger.warning(f"Feature encoder not found for {ab}")
            
            # Load label encoder first
            encoder_path = os.path.join(self.models_dir, f'{ab}_encoder.joblib')
            if os.path.exists(encoder_path):
                try:
                    self.label_encoders[ab] = joblib.load(encoder_path)
                    logger.info(f"Successfully loaded label encoder for {ab}")
                except Exception as e:
                    logger.error(f"Error loading label encoder for {ab}: {str(e)}")
            
            for model_type in model_types:
                model_path = os.path.join(self.models_dir, f'{ab}_{model_type}.joblib')
                logger.debug(f"Looking for model at: {model_path}")
                
                if os.path.exists(model_path):
                    try:
                        self.models[ab][model_type] = joblib.load(model_path)
                        logger.info(f"Successfully loaded {model_type} model for {ab}")
                    except Exception as e:
                        logger.error(f"Error loading model for {ab} {model_type}: {str(e)}")
                else:
                    logger.warning(f"Model file not found for {ab} {model_type}")
    
    def prepare_input(self, gender, country, age):
        """Prepare input data for prediction."""
        logger.info(f"Preparing input data - Gender: {gender}, Country: {country}")
        # Create a DataFrame with the input features
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Country': [country]
        })
        logger.debug(f"Input data prepared:\n{input_data}")
        return input_data
    
    def predict_resistance(self, gender, country, age):
        """Predict resistance for all antibiotics using all available models."""
        logger.info("Starting resistance prediction...")
        if not any(self.models.values()):
            logger.error("No models are available for prediction")
            raise RuntimeError("No models are available for prediction")
            
        input_data = self.prepare_input(gender, country, age)
        predictions = {}
        confidence_scores = {}
        
        for ab in self.models.keys():
            if not self.models[ab]:
                logger.warning(f"No models available for {ab}, skipping...")
                continue
                
            logger.info(f"\nProcessing predictions for {ab}")
            predictions[ab] = {}
            confidence_scores[ab] = {}
            
            # Transform input features using the feature encoder
            if ab in self.feature_encoders:
                try:
                    logger.debug(f"Transforming input features for {ab}")
                    input_encoded = self.feature_encoders[ab].transform(input_data)
                    logger.debug(f"Encoded features shape: {input_encoded.shape}")
                except Exception as e:
                    logger.error(f"Error encoding features for {ab}: {str(e)}")
                    raise RuntimeError(f"Feature encoding failed for {ab}: {str(e)}")
            else:
                logger.error(f"Feature encoder not found for {ab}")
                raise RuntimeError(f"Feature encoder not found for {ab}")
            
            for model_type, model in self.models[ab].items():
                try:
                    logger.debug(f"Making prediction with {model_type} for {ab}")
                    # Get prediction and probability
                    pred = model.predict(input_encoded)[0]
                    prob = model.predict_proba(input_encoded)[0]
                    
                    logger.debug(f"Raw prediction: {pred}")
                    logger.debug(f"Raw probabilities: {prob}")
                    
                    # Convert prediction using label encoder
                    if ab in self.label_encoders:
                        try:
                            # Check if the prediction is within the range of known classes
                            known_classes = self.label_encoders[ab].classes_
                            if pred < len(known_classes):
                                pred = self.label_encoders[ab].inverse_transform([pred])[0]
                                logger.debug(f"Transformed prediction: {pred}")
                            else:
                                logger.warning(f"Prediction {pred} is outside known classes {known_classes} for {ab}")
                                pred = "Unknown"
                        except Exception as e:
                            logger.error(f"Error transforming prediction for {ab}: {str(e)}")
                            pred = "Unknown"  # Fallback value
                    else:
                        logger.warning(f"No label encoder found for {ab}, using raw prediction")
                        pred = "Unknown"  # Fallback value
                    
                    predictions[ab][model_type] = pred
                    confidence_scores[ab][model_type] = float(max(prob))  # Convert to float
                    logger.info(f"Successfully made prediction with {model_type} for {ab}")
                except Exception as e:
                    logger.error(f"Error making prediction for {ab} {model_type}: {str(e)}")
                    continue
        
        if not predictions:
            logger.error("No predictions could be made with available models")
            raise RuntimeError("No predictions could be made with available models")
            
        logger.info("Prediction process completed successfully")
        return predictions, confidence_scores
    
    def get_best_model_predictions(self, predictions, confidence_scores):
        """Get predictions from the most confident model for each antibiotic."""
        logger.info("Finding best model predictions...")
        best_predictions = {}
        best_models = {}
        
        for ab in predictions.keys():
            if not predictions[ab]:
                logger.warning(f"No predictions available for {ab}, skipping...")
                continue
                
            # Find model with highest confidence
            best_model = max(confidence_scores[ab].items(), key=lambda x: x[1])[0]
            best_predictions[ab] = predictions[ab][best_model]
            best_models[ab] = best_model
            logger.info(f"Best model for {ab}: {best_model} with confidence {confidence_scores[ab][best_model]}")
        
        return best_predictions, best_models
    
    def format_results(self, predictions, confidence_scores, best_predictions, best_models):
        """Format the results in a user-friendly way."""
        logger.info("Formatting results...")
        results = {
            'summary': {},
            'detailed': {}
        }
        
        # Get best predictions
        for ab, pred in best_predictions.items():
            model = best_models[ab]
            confidence = confidence_scores[ab][model]
            
            results['summary'][ab] = {
                'prediction': pred,
                'confidence': float(confidence),  # Ensure float
                'best_model': model
            }
            
            # Detailed results for each model
            results['detailed'][ab] = {
                model_type: {
                    'prediction': predictions[ab][model_type],
                    'confidence': float(confidence_scores[ab][model_type])  # Ensure float
                }
                for model_type in predictions[ab].keys()
            }
            
            logger.info(f"Formatted results for {ab}: {results['summary'][ab]}")
        
        return results

def predict_resistance_for_user(gender, country, age, models_dir):
    """Main function to get resistance predictions for a user."""
    logger.info(f"Starting prediction for user - Gender: {gender}, Country: {country}")
    predictor = ResistancePredictor(models_dir)
    
    # Get predictions from all models
    predictions, confidence_scores = predictor.predict_resistance(gender, country, age)
    
    # Get best model predictions
    best_predictions, best_models = predictor.get_best_model_predictions(
        predictions, confidence_scores)
    
    # Format results
    results = predictor.format_results(
        predictions, confidence_scores, best_predictions, best_models)
    
    logger.info("Prediction process completed successfully")
    return results 