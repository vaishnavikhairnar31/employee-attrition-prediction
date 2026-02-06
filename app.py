"""
Flask Web Application for Employee Attrition Prediction
Author: Senior ML Engineer
Description: REST API and web interface for employee attrition predictions
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Initialize Flask app
app = Flask(__name__, template_folder='frontend', static_folder='frontend')

# Global variables for model and preprocessor
model = None
scaler = None
label_encoders = None
feature_columns = None
model_metadata = None


def load_model_artifacts():
    """Load all necessary model artifacts"""
    global model, scaler, label_encoders, feature_columns, model_metadata
    
    print("🔄 Loading model artifacts...")
    
    try:
        model_dir = 'model'
        
        # Load the trained model
        model = joblib.load(os.path.join(model_dir, 'attrition_model.pkl'))
        print("  ✓ Model loaded")
        
        # Load the scaler
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        print("  ✓ Scaler loaded")
        
        # Load label encoders
        label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
        print("  ✓ Label encoders loaded")
        
        # Load feature columns
        feature_columns = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))
        print("  ✓ Feature columns loaded")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
        if os.path.exists(metadata_path):
            model_metadata = joblib.load(metadata_path)
            print(f"  ✓ Model metadata loaded: {model_metadata['model_name']}")
        
        print("✅ All artifacts loaded successfully!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model artifacts: {str(e)}")
        return False


def preprocess_input(input_data):
    """
    Preprocess input data to match training data format
    
    Args:
        input_data (dict): Dictionary containing employee features
        
    Returns:
        np.array: Preprocessed and scaled feature array
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Encode categorical variables using saved encoders
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                # Handle the specific categorical values
                df[col] = encoder.transform(df[col].astype(str))
            except ValueError:
                # If unknown category, use the most frequent class
                df[col] = encoder.transform([encoder.classes_[0]])[0]
    
    # Ensure all required features are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Default value for missing features
    
    # Select only the features used during training
    df = df[feature_columns]
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    return df_scaled


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions
    Accepts JSON data and returns prediction result
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        required_fields = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany', 'OverTime']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'status': 'error'
                }), 400
        
        # Prepare complete input data with default values
        input_data = {
            'Age': int(data.get('Age', 30)),
            'BusinessTravel': data.get('BusinessTravel', 'Travel_Rarely'),
            'DailyRate': int(data.get('DailyRate', 800)),
            'Department': data.get('Department', 'Research & Development'),
            'DistanceFromHome': int(data.get('DistanceFromHome', 10)),
            'Education': int(data.get('Education', 3)),
            'EducationField': data.get('EducationField', 'Life Sciences'),
            'EnvironmentSatisfaction': int(data.get('EnvironmentSatisfaction', 3)),
            'Gender': data.get('Gender', 'Male'),
            'HourlyRate': int(data.get('HourlyRate', 65)),
            'JobInvolvement': int(data.get('JobInvolvement', 3)),
            'JobLevel': int(data.get('JobLevel', 2)),
            'JobRole': data.get('JobRole', 'Laboratory Technician'),
            'JobSatisfaction': int(data.get('JobSatisfaction', 3)),
            'MaritalStatus': data.get('MaritalStatus', 'Single'),
            'MonthlyIncome': int(data.get('MonthlyIncome', 5000)),
            'MonthlyRate': int(data.get('MonthlyRate', 15000)),
            'NumCompaniesWorked': int(data.get('NumCompaniesWorked', 2)),
            'OverTime': data.get('OverTime', 'No'),
            'PercentSalaryHike': int(data.get('PercentSalaryHike', 13)),
            'PerformanceRating': int(data.get('PerformanceRating', 3)),
            'RelationshipSatisfaction': int(data.get('RelationshipSatisfaction', 3)),
            'StockOptionLevel': int(data.get('StockOptionLevel', 0)),
            'TotalWorkingYears': int(data.get('TotalWorkingYears', 8)),
            'TrainingTimesLastYear': int(data.get('TrainingTimesLastYear', 3)),
            'WorkLifeBalance': int(data.get('WorkLifeBalance', 3)),
            'YearsAtCompany': int(data.get('YearsAtCompany', 5)),
            'YearsInCurrentRole': int(data.get('YearsInCurrentRole', 3)),
            'YearsSinceLastPromotion': int(data.get('YearsSinceLastPromotion', 1)),
            'YearsWithCurrManager': int(data.get('YearsWithCurrManager', 3))
        }
        
        # Preprocess input
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Prepare result
        result = {
            'prediction': 'Yes' if prediction == 1 else 'No',
            'prediction_label': 'Employee likely to LEAVE 🚪' if prediction == 1 else 'Employee likely to STAY ✅',
            'confidence': float(max(probability) * 100),
            'attrition_probability': float(probability[1] * 100),
            'retention_probability': float(probability[0] * 100),
            'status': 'success'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Endpoint to get model information
    """
    try:
        info = {
            'model_name': model_metadata.get('model_name', 'Unknown') if model_metadata else 'Unknown',
            'train_accuracy': model_metadata.get('train_accuracy', 0) if model_metadata else 0,
            'test_accuracy': model_metadata.get('test_accuracy', 0) if model_metadata else 0,
            'test_auc': model_metadata.get('test_auc', 0) if model_metadata else 0,
            'number_of_features': len(feature_columns) if feature_columns else 0,
            'status': 'success'
        }
        return jsonify(info), 200
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Employee Attrition Prediction API is running!'
    }), 200


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500


# Main execution
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 EMPLOYEE ATTRITION PREDICTION API")
    print("="*60 + "\n")
    
    # Load model artifacts
    if not load_model_artifacts():
        print("⚠️  Failed to load model artifacts. Please train the model first.")
        print("   Run: python src/model_training.py")
        sys.exit(1)
    
    print("🌐 Starting Flask server...")
    print("📍 Access the application at: http://localhost:5000")
    print("📍 API Health Check: http://localhost:5000/health")
    print("📍 Model Info: http://localhost:5000/model-info")
    print("\n" + "="*60 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
