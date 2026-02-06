# 🎯 Employee Attrition Prediction System

A complete, production-ready Machine Learning project that predicts employee attrition using IBM HR Employee data. This system uses advanced ML algorithms and provides a user-friendly Flask web interface for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## 🎓 Project Overview

Employee attrition is a critical challenge for organizations, leading to significant costs in recruitment, training, and lost productivity.  
This project leverages machine learning to:

- Predict which employees are at risk of leaving
- Identify key factors driving attrition
- Provide actionable insights for HR decision-making
- Enable proactive retention strategies

---

## ✨ Features

### Core Functionality
- ✅ Logistic Regression & Random Forest models
- ✅ Real-time predictions via Flask web app
- ✅ REST API for integration
- ✅ Feature importance analysis
- ✅ End-to-end ML pipeline

### Technical Highlights
- ✅ Data preprocessing & scaling
- ✅ Model evaluation (Accuracy, ROC-AUC, Confusion Matrix)
- ✅ Model persistence using joblib
- ✅ Clean modular codebase

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn

### Web
- Flask
- HTML / CSS / JavaScript

### Tools
- Jupyter Notebook
- Git & GitHub

---

## 📁 Project Structure

```
employee-attrition-prediction/
│
├── data/
│   └── hr_employee_data.csv          # IBM HR Employee Attrition dataset
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # EDA with visualizations
│
├── src/
│   ├── data_preprocessing.py         # Data cleaning and transformation
│   ├── model_training.py             # Model training and evaluation
│   └── predict.py                    # Prediction module
│
├── model/
│   ├── attrition_model.pkl          # Trained model
│   ├── scaler.pkl                   # Feature scaler
│   ├── label_encoders.pkl           # Categorical encoders
│   ├── feature_columns.pkl          # Feature names
│   ├── model_metadata.pkl           # Model metadata
│   └── plots/                       # Evaluation plots
│       ├── confusion_matrix_*.png
│       └── roc_curve_comparison.png
│
├── frontend/
│   └── index.html                   # Web interface
│
├── app.py                           # Flask application
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/vaishnavikhairnar31/employee-attrition-prediction.git
cd employee-attrition-prediction
```
### Step 2: Create Virtual Environment
python -m venv venv
venv\Scripts\activate

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Train the Model
python run_all.py

### Step 5: Run the Web App
python app.py


### Open in browser:

http://localhost:5000
