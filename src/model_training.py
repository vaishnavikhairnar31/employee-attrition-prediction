"""
Model Training Module
Author: Senior ML Engineer
Description: Trains and evaluates Logistic Regression and Random Forest models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path to import data_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import DataPreprocessor


class ModelTrainer:
    """
    A comprehensive model training class for Employee Attrition Prediction
    """
    
    def __init__(self):
        """Initialize the model trainer"""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("\n" + "="*60)
        print("🤖 TRAINING LOGISTIC REGRESSION MODEL")
        print("="*60)
        
        # Initialize model
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear'
        )
        
        # Train model
        print("Training in progress...")
        lr_model.fit(X_train, y_train)
        
        # Store model
        self.models['Logistic Regression'] = lr_model
        
        print("✅ Logistic Regression training completed!")
        return lr_model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("🌲 TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        # Initialize model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            n_jobs=-1
        )
        
        # Train model
        print("Training in progress...")
        rf_model.fit(X_train, y_train)
        
        # Store model
        self.models['Random Forest'] = rf_model
        
        print("✅ Random Forest training completed!")
        return rf_model
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evaluate a trained model"""
        print(f"\n{'='*60}")
        print(f"📊 EVALUATING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Probabilities for AUC
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Classification Report
        report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Store results
        self.results[model_name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_test_pred,
            'probabilities': y_test_proba
        }
        
        # Display results
        print(f"\n📈 Performance Metrics:")
        print(f"  Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  Testing Accuracy:    {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  ROC-AUC Score:       {test_auc:.4f}")
        
        print(f"\n📊 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 No    Yes")
        print(f"  Actual No    {cm[0][0]:5d} {cm[0][1]:5d}")
        print(f"  Actual Yes   {cm[1][0]:5d} {cm[1][1]:5d}")
        
        print(f"\n📋 Classification Report:")
        print(f"              Precision  Recall  F1-Score  Support")
        print(f"  No (Stay)     {report['0']['precision']:.4f}   {report['0']['recall']:.4f}    {report['0']['f1-score']:.4f}    {int(report['0']['support'])}")
        print(f"  Yes (Leave)   {report['1']['precision']:.4f}   {report['1']['recall']:.4f}    {report['1']['f1-score']:.4f}    {int(report['1']['support'])}")
        
        return self.results[model_name]
    
    def plot_confusion_matrix(self, model_name, save_path=None):
        """Plot confusion matrix heatmap"""
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Stay (No)', 'Leave (Yes)'],
            yticklabels=['Stay (No)', 'Leave (Yes)']
        )
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved confusion matrix plot: {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, y_test, save_path=None):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 6))
        
        for model_name, results in self.results.items():
            y_proba = results['probabilities']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = results['test_auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved ROC curve plot: {save_path}")
        
        plt.close()
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("🏆 MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train Accuracy': [self.results[m]['train_accuracy'] for m in self.results],
            'Test Accuracy': [self.results[m]['test_accuracy'] for m in self.results],
            'ROC-AUC': [self.results[m]['test_auc'] for m in self.results]
        })
        
        print("\n", comparison_df.to_string(index=False))
        
        # Determine best model based on test accuracy
        best_idx = comparison_df['Test Accuracy'].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🎯 Best Model: {self.best_model_name}")
        print(f"   Test Accuracy: {comparison_df.loc[best_idx, 'Test Accuracy']:.4f}")
        print(f"   ROC-AUC: {comparison_df.loc[best_idx, 'ROC-AUC']:.4f}")
        
        return comparison_df
    
    def get_feature_importance(self, feature_names, top_n=10):
        """Get feature importance from the best model"""
        if self.best_model_name == 'Random Forest':
            importance = self.best_model.feature_importances_
        elif self.best_model_name == 'Logistic Regression':
            importance = np.abs(self.best_model.coef_[0])
        else:
            return None
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\n📊 Top {top_n} Most Important Features ({self.best_model_name}):")
        print(feature_importance_df.head(top_n).to_string(index=False))
        
        return feature_importance_df
    
    def save_best_model(self, model_dir='../model'):
        """Save the best performing model"""
        print(f"\n💾 Saving best model ({self.best_model_name})...")
        
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'attrition_model.pkl')
        joblib.dump(self.best_model, model_path)
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'train_accuracy': self.results[self.best_model_name]['train_accuracy'],
            'test_accuracy': self.results[self.best_model_name]['test_accuracy'],
            'test_auc': self.results[self.best_model_name]['test_auc']
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"  ✓ Model saved to: {model_path}")
        print(f"  ✓ Metadata saved to: {metadata_path}")
        print("✅ Best model saved successfully!")
    
    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test, feature_names):
        """Complete training and evaluation pipeline"""
        print("\n" + "="*60)
        print("🚀 STARTING MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Train models
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        
        # Evaluate models
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
        
        # Compare models
        comparison = self.compare_models()
        
        # Feature importance
        self.get_feature_importance(feature_names)
        
        # Generate plots
        print("\n📊 Generating evaluation plots...")
        os.makedirs('../model/plots', exist_ok=True)
        
        for model_name in self.models.keys():
            self.plot_confusion_matrix(
                model_name, 
                save_path=f'../model/plots/confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
            )
        
        self.plot_roc_curve(y_test, save_path='../model/plots/roc_curve_comparison.png')
        
        # Save best model
        self.save_best_model()
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING PIPELINE COMPLETED!")
        print("="*60)
        
        return self.best_model


# Main execution
if __name__ == "__main__":
    print("\n" + "🎯"*30)
    print("EMPLOYEE ATTRITION PREDICTION - MODEL TRAINING")
    print("🎯"*30 + "\n")
    
    # Step 1: Preprocess data
    print("Step 1: Data Preprocessing")
    preprocessor = DataPreprocessor('../data/hr_employee_data.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # Get feature names
    feature_names = preprocessor.feature_columns
    
    # Step 2: Train and evaluate models
    print("\nStep 2: Model Training & Evaluation")
    trainer = ModelTrainer()
    best_model = trainer.train_and_evaluate_all(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    print("\n✨ Training process completed successfully!")
    print(f"Best model ({trainer.best_model_name}) is ready for deployment!")
