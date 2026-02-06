"""
Complete Pipeline Runner
Runs the entire ML pipeline: preprocessing, training, and starts the app
"""

import os
import sys
import subprocess

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")

def run_preprocessing():
    """Run data preprocessing"""
    print_banner("STEP 1: DATA PREPROCESSING")
    
    try:
        from src.data_preprocessing import DataPreprocessor
        
        # Get absolute path to data
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, 'data', 'hr_employee_data.csv')
        model_dir = os.path.join(base_dir, 'model')
        
        # Run preprocessing
        preprocessor = DataPreprocessor(data_path)
        preprocessor.model_dir = model_dir
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
        
        print("\n✅ Preprocessing completed successfully!")
        return X_train, X_test, y_train, y_test, preprocessor
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        sys.exit(1)

def run_training(X_train, X_test, y_train, y_test, feature_names):
    """Run model training"""
    print_banner("STEP 2: MODEL TRAINING")
    
    try:
        from src.model_training import ModelTrainer
        
        # Get absolute path for model directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'model')
        
        # Train models
        trainer = ModelTrainer()
        trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test, feature_names)
        
        # Update save path
        trainer.save_best_model(model_dir)
        
        print("\n✅ Training completed successfully!")
        print(f"✅ Best model: {trainer.best_model_name}")
        print(f"✅ Test Accuracy: {trainer.results[trainer.best_model_name]['test_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main pipeline runner"""
    print("\n" + "🎯"*30)
    print("EMPLOYEE ATTRITION PREDICTION - COMPLETE PIPELINE")
    print("🎯"*30 + "\n")
    
    # Step 1: Preprocessing
    X_train, X_test, y_train, y_test, preprocessor = run_preprocessing()
    
    # Step 2: Training
    success = run_training(X_train, X_test, y_train, y_test, preprocessor.feature_columns)
    
    if success:
        print_banner("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("\n📝 Next Steps:")
        print("   1. Review model performance in model/plots/")
        print("   2. Start the web app: python app.py")
        print("   3. Open browser: http://localhost:5000")
        print("\n" + "="*60 + "\n")
    else:
        print_banner("❌ PIPELINE FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
