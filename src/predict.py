"""
Prediction Module
Author: Senior ML Engineer
Description: Makes predictions using the trained model
"""

import pandas as pd
import numpy as np
import joblib
import os


class AttritionPredictor:
    """
    A class to make employee attrition predictions using the trained model
    """
    
    def __init__(self, model_dir='../model'):
        """
        Initialize the predictor by loading the trained model and preprocessor
        
        Args:
            model_dir (str): Directory containing the saved model and preprocessor objects
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.metadata = None
        
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """Load all necessary model artifacts"""
        print("🔄 Loading model artifacts...")
        
        try:
            # Load the trained model
            model_path = os.path.join(self.model_dir, 'attrition_model.pkl')
            self.model = joblib.load(model_path)
            print("  ✓ Model loaded")
            
            # Load the scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            print("  ✓ Scaler loaded")
            
            # Load label encoders
            encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
            self.label_encoders = joblib.load(encoders_path)
            print("  ✓ Label encoders loaded")
            
            # Load feature columns
            features_path = os.path.join(self.model_dir, 'feature_columns.pkl')
            self.feature_columns = joblib.load(features_path)
            print("  ✓ Feature columns loaded")
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'model_metadata.pkl')
            if os.path.exists(metadata_path):
                self.metadata = joblib.load(metadata_path)
                print(f"  ✓ Model metadata loaded: {self.metadata['model_name']}")
            
            print("✅ All artifacts loaded successfully!\n")
            
        except Exception as e:
            print(f"❌ Error loading model artifacts: {str(e)}")
            raise
    
    def preprocess_input(self, input_data):
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
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # If unknown category, use the most frequent class
                    df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Select only the features used during training
        df = df[self.feature_columns]
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        
        return df_scaled
    
    def predict(self, input_data):
        """
        Make prediction for a single employee
        
        Args:
            input_data (dict): Dictionary containing employee features
            
        Returns:
            dict: Prediction result with probability
        """
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0]
            
            # Prepare result
            result = {
                'prediction': 'Yes' if prediction == 1 else 'No',
                'prediction_label': 'Employee likely to LEAVE' if prediction == 1 else 'Employee likely to STAY',
                'confidence': float(max(probability) * 100),
                'attrition_probability': float(probability[1] * 100),
                'retention_probability': float(probability[0] * 100)
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            raise
    
    def predict_batch(self, input_data_list):
        """
        Make predictions for multiple employees
        
        Args:
            input_data_list (list): List of dictionaries containing employee features
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for input_data in input_data_list:
            result = self.predict(input_data)
            results.append(result)
        
        return results
    
    def display_prediction(self, result):
        """Display prediction result in a formatted way"""
        print("\n" + "="*60)
        print("🎯 PREDICTION RESULT")
        print("="*60)
        print(f"\n  Prediction: {result['prediction_label']}")
        print(f"  Confidence: {result['confidence']:.2f}%")
        print(f"\n  Attrition Probability: {result['attrition_probability']:.2f}%")
        print(f"  Retention Probability: {result['retention_probability']:.2f}%")
        print("\n" + "="*60)


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "🎯"*30)
    print("EMPLOYEE ATTRITION PREDICTION - PREDICTION MODULE")
    print("🎯"*30 + "\n")
    
    # Initialize predictor
    predictor = AttritionPredictor(model_dir='../model')
    
    # Example 1: Employee likely to leave
    print("Example 1: High-risk employee profile")
    employee_high_risk = {
        'Age': 28,
        'BusinessTravel': 'Travel_Frequently',
        'DailyRate': 500,
        'Department': 'Sales',
        'DistanceFromHome': 25,
        'Education': 2,
        'EducationField': 'Life Sciences',
        'EnvironmentSatisfaction': 1,
        'Gender': 'Male',
        'HourlyRate': 45,
        'JobInvolvement': 2,
        'JobLevel': 1,
        'JobRole': 'Sales Executive',
        'JobSatisfaction': 1,
        'MaritalStatus': 'Single',
        'MonthlyIncome': 2500,
        'MonthlyRate': 8000,
        'NumCompaniesWorked': 5,
        'OverTime': 'Yes',
        'PercentSalaryHike': 11,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 1,
        'StockOptionLevel': 0,
        'TotalWorkingYears': 6,
        'TrainingTimesLastYear': 1,
        'WorkLifeBalance': 1,
        'YearsAtCompany': 1,
        'YearsInCurrentRole': 0,
        'YearsSinceLastPromotion': 0,
        'YearsWithCurrManager': 0
    }
    
    result_1 = predictor.predict(employee_high_risk)
    predictor.display_prediction(result_1)
    
    # Example 2: Employee likely to stay
    print("\n\nExample 2: Low-risk employee profile")
    employee_low_risk = {
        'Age': 45,
        'BusinessTravel': 'Travel_Rarely',
        'DailyRate': 1200,
        'Department': 'Research & Development',
        'DistanceFromHome': 5,
        'Education': 4,
        'EducationField': 'Medical',
        'EnvironmentSatisfaction': 4,
        'Gender': 'Female',
        'HourlyRate': 85,
        'JobInvolvement': 4,
        'JobLevel': 4,
        'JobRole': 'Manager',
        'JobSatisfaction': 4,
        'MaritalStatus': 'Married',
        'MonthlyIncome': 12000,
        'MonthlyRate': 20000,
        'NumCompaniesWorked': 1,
        'OverTime': 'No',
        'PercentSalaryHike': 18,
        'PerformanceRating': 4,
        'RelationshipSatisfaction': 4,
        'StockOptionLevel': 2,
        'TotalWorkingYears': 20,
        'TrainingTimesLastYear': 5,
        'WorkLifeBalance': 4,
        'YearsAtCompany': 15,
        'YearsInCurrentRole': 8,
        'YearsSinceLastPromotion': 3,
        'YearsWithCurrManager': 10
    }
    
    result_2 = predictor.predict(employee_low_risk)
    predictor.display_prediction(result_2)
