"""
Data Preprocessing Module
Author: Senior Data Scientist
Description: Handles all data preprocessing tasks including encoding, scaling, and splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for HR Employee Attrition data
    """
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor with data path
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self):
        """Load the dataset from CSV file"""
        print("📊 Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Data loaded successfully! Shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Display basic information about the dataset"""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of Features: {self.df.shape[1]}")
        print(f"Number of Samples: {self.df.shape[0]}")
        
        print("\n📋 Column Names and Types:")
        print(self.df.dtypes)
        
        print("\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✅ No missing values found!")
        else:
            print(missing[missing > 0])
        
        print("\n📊 Attrition Distribution:")
        print(self.df['Attrition'].value_counts())
        print(f"Attrition Rate: {(self.df['Attrition'] == 'Yes').sum() / len(self.df) * 100:.2f}%")
        
        return self.df.describe()
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n🔧 Handling missing values...")
        
        # Check for missing values
        missing_counts = self.df.isnull().sum()
        
        if missing_counts.sum() == 0:
            print("✅ No missing values to handle!")
            return
        
        # For numerical columns: fill with median
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                median_value = self.df[col].median()
                self.df[col].fillna(median_value, inplace=True)
                print(f"  Filled {col} with median: {median_value}")
        
        # For categorical columns: fill with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode()[0]
                self.df[col].fillna(mode_value, inplace=True)
                print(f"  Filled {col} with mode: {mode_value}")
        
        print("✅ Missing values handled!")
    
    def encode_categorical_variables(self):
        """Encode categorical variables using Label Encoding"""
        print("\n🔤 Encoding categorical variables...")
        
        # Identify categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from encoding list if present
        if 'Attrition' in categorical_cols:
            categorical_cols.remove('Attrition')
        
        # Encode each categorical column
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f"  ✓ Encoded {col}: {len(le.classes_)} unique values")
        
        # Encode target variable (Attrition)
        print("\n🎯 Encoding target variable (Attrition)...")
        self.df['Attrition'] = self.df['Attrition'].map({'Yes': 1, 'No': 0})
        print("  ✓ Yes → 1, No → 0")
        
        print("\n✅ All categorical variables encoded!")
        return self.df
    
    def remove_irrelevant_features(self):
        """Remove features that don't contribute to prediction"""
        print("\n🗑️  Removing irrelevant features...")
        
        # Features to remove (EmployeeCount, EmployeeNumber, Over18, StandardHours are constants)
        cols_to_drop = []
        
        # Check for constant columns
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                cols_to_drop.append(col)
        
        # Also remove employee identifiers
        identifier_cols = ['EmployeeNumber', 'EmployeeCount']
        for col in identifier_cols:
            if col in self.df.columns and col not in cols_to_drop:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            print(f"  Dropped columns: {cols_to_drop}")
        else:
            print("  No irrelevant features found!")
        
        print("✅ Feature removal complete!")
        return self.df
    
    def prepare_features_and_target(self):
        """Separate features and target variable"""
        print("\n🎯 Preparing features and target...")
        
        # Separate target variable
        y = self.df['Attrition']
        X = self.df.drop('Attrition', axis=1)
        
        self.feature_columns = X.columns.tolist()
        
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Number of features: {len(self.feature_columns)}")
        
        return X, y
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features using StandardScaler"""
        print("\n📏 Scaling features...")
        
        # Fit scaler on training data and transform both train and test
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("  ✓ Features scaled using StandardScaler")
        print(f"  Training data shape: {X_train_scaled.shape}")
        print(f"  Testing data shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print(f"\n✂️  Splitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Testing set: {X_test.shape[0]} samples")
        print(f"  Train attrition rate: {y_train.sum() / len(y_train) * 100:.2f}%")
        print(f"  Test attrition rate: {y_test.sum() / len(y_test) * 100:.2f}%")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, model_dir='../model'):
        """Save the scaler and encoders for later use"""
        print("\n💾 Saving preprocessor objects...")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        print("  ✓ Scaler saved")
        
        # Save label encoders
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
        print("  ✓ Label encoders saved")
        
        # Save feature columns
        joblib.dump(self.feature_columns, os.path.join(model_dir, 'feature_columns.pkl'))
        print("  ✓ Feature columns saved")
        
        print("✅ Preprocessor saved successfully!")
    
    def preprocess_pipeline(self, save_preprocessor=True):
        """Complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Encode categorical variables
        self.encode_categorical_variables()
        
        # Step 5: Remove irrelevant features
        self.remove_irrelevant_features()
        
        # Step 6: Prepare features and target
        X, y = self.prepare_features_and_target()
        
        # Step 7: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Step 8: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Step 9: Save preprocessor
        if save_preprocessor:
            self.save_preprocessor()
        
        print("\n" + "="*60)
        print("✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test


# Main execution
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('../data/hr_employee_data.csv')
    
    # Run complete preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    print(f"\n📊 Final Dataset Summary:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    print(f"  Number of features: {X_train.shape[1]}")
    print(f"  Training labels distribution: {np.bincount(y_train)}")
    print(f"  Testing labels distribution: {np.bincount(y_test)}")
