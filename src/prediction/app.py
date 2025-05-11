from flask import Flask, render_template, request, jsonify, send_from_directory
from prediction_system import predict_resistance_for_user
import os

# Get the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# Create Flask app with correct static folder path
app = Flask(__name__, 
            static_folder=os.path.join(project_root, 'Output'),
            static_url_path='/static')

# Get the models directory
models_dir = os.path.join(project_root, 'Output', 'model_metrics', 'saved_models')

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Check for required model files
required_models = []
for ab in ['azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr']:
    for model_type in ['Random Forest', 'SVM', 'Gradient Boosting']:
        required_models.append(f'{ab}_{model_type}.joblib')
        required_models.append(f'{ab}_encoder.joblib')

missing_models = [model for model in required_models if not os.path.exists(os.path.join(models_dir, model))]

if missing_models:
    app.logger.warning(f"Missing model files: {', '.join(missing_models)}")
    app.logger.warning("Please ensure all model files are present in the models directory")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if models are available
        if missing_models:
            return jsonify({
                'success': False,
                'error': 'Prediction models are not available. Please contact the administrator.'
            }), 503

        # Get user input
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Validate required fields
        required_fields = ['gender', 'country', 'age']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Validate gender
        gender = data['gender']
        if gender not in ['Male', 'Female']:
            return jsonify({
                'success': False,
                'error': 'Gender must be either "Male" or "Female"'
            }), 400

        # Validate country
        country = data['country']
        if not isinstance(country, str) or not country.strip():
            return jsonify({
                'success': False,
                'error': 'Country must be a non-empty string'
            }), 400

        # Validate age
        try:
            age = int(data['age'])
            if age < 0 or age > 120:
                return jsonify({
                    'success': False,
                    'error': 'Age must be between 0 and 120'
                }), 400
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Age must be a valid number'
            }), 400
        
        # Get predictions
        results = predict_resistance_for_user(gender, country, age, models_dir)
        
        # Format results for display
        formatted_results = {
            'summary': {},
            'detailed': {}
        }
        
        # Process summary results
        for ab, info in results['summary'].items():
            antibiotic_name = {
                'azm_sr': 'Azithromycin',
                'cip_sr': 'Ciprofloxacin',
                'cro_sr': 'Ceftriaxone',
                'cfx_sr': 'Cefixime',
                'tet_sr': 'Tetracycline',
                'pen_sr': 'Penicillin'
            }.get(ab, ab)
            
            formatted_results['summary'][antibiotic_name] = {
                'prediction': info['prediction'],
                'confidence': info['confidence'],
                'best_model': info['best_model']
            }
        
        # Process detailed results
        for ab, models in results['detailed'].items():
            antibiotic_name = {
                'azm_sr': 'Azithromycin',
                'cip_sr': 'Ciprofloxacin',
                'cro_sr': 'Ceftriaxone',
                'cfx_sr': 'Cefixime',
                'tet_sr': 'Tetracycline',
                'pen_sr': 'Penicillin'
            }.get(ab, ab)
            
            formatted_results['detailed'][antibiotic_name] = {
                model: {
                    'prediction': info['prediction'],
                    'confidence': info['confidence']
                }
                for model, info in models.items()
            }
        
        return jsonify({
            'success': True,
            'results': formatted_results
        })
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while making the prediction. Please try again.'
        }), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/maps/<path:filename>')
def serve_maps(filename):
    maps_dir = os.path.join(app.static_folder, 'maps')
    return send_from_directory(maps_dir, filename)

if __name__ == '__main__':
    app.run(debug=True) 