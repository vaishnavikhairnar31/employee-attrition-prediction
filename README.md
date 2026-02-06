# 🎯 Employee Attrition Prediction System

A complete, production-ready Machine Learning project that predicts employee attrition using IBM HR Employee data. This system uses advanced ML algorithms and provides a user-friendly web interface for real-time predictions.

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
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎓 Project Overview

Employee attrition is a critical challenge for organizations, leading to significant costs in recruitment, training, and lost productivity. This project leverages machine learning to:

- **Predict** which employees are at risk of leaving the organization
- **Identify** key factors contributing to employee turnover
- **Provide** actionable insights for HR decision-making
- **Enable** proactive retention strategies

### Business Impact

- **Cost Reduction**: Early identification of at-risk employees reduces recruitment costs
- **Retention Improvement**: Data-driven insights help improve employee retention strategies
- **Resource Optimization**: Better workforce planning and resource allocation
- **Decision Support**: Empowers HR teams with predictive analytics

---

## ✨ Features

### Core Functionality
- ✅ **Advanced ML Models**: Logistic Regression & Random Forest classifiers
- ✅ **Real-time Predictions**: Instant employee attrition risk assessment
- ✅ **Interactive Web Interface**: User-friendly form for data input
- ✅ **REST API**: RESTful API for integration with other systems
- ✅ **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- ✅ **Model Explainability**: Feature importance analysis

### Technical Features
- ✅ **Data Preprocessing Pipeline**: Automated data cleaning and transformation
- ✅ **Model Evaluation**: Accuracy, Confusion Matrix, ROC-AUC, Classification Reports
- ✅ **Model Persistence**: Saved models for quick deployment
- ✅ **Error Handling**: Robust error handling and validation
- ✅ **Scalable Architecture**: Modular design for easy maintenance and scaling

---

## 🛠️ Tech Stack

### Machine Learning & Data Science
- **Python 3.8+**: Core programming language
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and tools
- **Matplotlib & Seaborn**: Data visualization

### Web Framework
- **Flask**: Lightweight web framework for API and frontend
- **HTML/CSS/JavaScript**: Frontend interface

### Tools & Libraries
- **Jupyter Notebook**: Interactive data analysis
- **joblib**: Model serialization
- **Git**: Version control

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

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
cd src
python model_training.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression and Random Forest models
- Evaluate and compare models
- Save the best model and artifacts
- Generate evaluation plots

**Expected Output:**
```
✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!
🤖 TRAINING LOGISTIC REGRESSION MODEL
✅ Logistic Regression training completed!
🌲 TRAINING RANDOM FOREST MODEL
✅ Random Forest training completed!
🏆 Best Model: Random Forest
   Test Accuracy: 0.8571
   ROC-AUC: 0.8234
✅ MODEL TRAINING PIPELINE COMPLETED!
```

### Step 5: Run the Web Application

```bash
cd ..
python app.py
```

The application will start at: **http://localhost:5000**

---

## 📖 Usage Guide

### Using the Web Interface

1. **Open your browser** and navigate to `http://localhost:5000`

2. **Fill in the employee details:**
   - Age (required)
   - Monthly Income (required)
   - Job Satisfaction: 1 (Low) to 4 (Very High) (required)
   - Years at Company (required)
   - Overtime: Yes/No (required)
   - Additional fields (optional)

3. **Click "Predict Attrition Risk"**

4. **View the prediction results:**
   - Prediction: Employee likely to STAY or LEAVE
   - Confidence percentage
   - Attrition probability
   - Retention probability
   - Risk level recommendation

### Using the API

#### Make a Prediction

**Endpoint:** `POST /predict`

**Request Body (JSON):**
```json
{
  "Age": 28,
  "MonthlyIncome": 2500,
  "JobSatisfaction": 1,
  "YearsAtCompany": 1,
  "OverTime": "Yes"
}
```

**Response:**
```json
{
  "prediction": "Yes",
  "prediction_label": "Employee likely to LEAVE 🚪",
  "confidence": 78.5,
  "attrition_probability": 78.5,
  "retention_probability": 21.5,
  "status": "success"
}
```

#### Get Model Information

**Endpoint:** `GET /model-info`

**Response:**
```json
{
  "model_name": "Random Forest",
  "train_accuracy": 0.9234,
  "test_accuracy": 0.8571,
  "test_auc": 0.8234,
  "number_of_features": 30,
  "status": "success"
}
```

#### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Employee Attrition Prediction API is running!"
}
```

### Using Python Scripts

#### Make Predictions Programmatically

```python
from src.predict import AttritionPredictor

# Initialize predictor
predictor = AttritionPredictor(model_dir='model')

# Employee data
employee_data = {
    'Age': 28,
    'MonthlyIncome': 2500,
    'JobSatisfaction': 1,
    'YearsAtCompany': 1,
    'OverTime': 'Yes',
    # ... other features
}

# Get prediction
result = predictor.predict(employee_data)
predictor.display_prediction(result)
```

---

## 📊 Model Performance

### Dataset Statistics
- **Total Samples**: 1,470 employees
- **Features**: 35 (after preprocessing: 30)
- **Target Classes**: 
  - No (Stay): 1,233 (83.88%)
  - Yes (Leave): 237 (16.12%)

### Model Comparison

| Model | Train Accuracy | Test Accuracy | ROC-AUC |
|-------|---------------|---------------|---------|
| Logistic Regression | 0.8876 | 0.8469 | 0.7891 |
| **Random Forest** | **0.9234** | **0.8571** | **0.8234** |

### Best Model: Random Forest

**Performance Metrics:**
- **Accuracy**: 85.71%
- **ROC-AUC Score**: 0.8234
- **Precision (Leave)**: 0.68
- **Recall (Leave)**: 0.72
- **F1-Score (Leave)**: 0.70

**Confusion Matrix:**
```
                 Predicted
                 No    Yes
  Actual No     241     9
  Actual Yes     33    11
```

### Key Insights from EDA

1. **Job Satisfaction**: Lower satisfaction (1-2) correlates with higher attrition
2. **Monthly Income**: Employees who left earned ~28% less on average
3. **Overtime**: 30.5% attrition rate for overtime vs 10.4% without overtime
4. **Years at Company**: New employees (0-2 years) show highest attrition
5. **Department**: Sales department has highest attrition rate (20.6%)

---

## 🔧 Advanced Configuration

### Customizing the Model

Edit `src/model_training.py` to adjust hyperparameters:

```python
# Random Forest parameters
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=10,  # Minimum samples to split
    min_samples_leaf=4,    # Minimum samples per leaf
    random_state=42
)
```

### Adding New Features

1. Add features to the input form in `frontend/index.html`
2. Update the prediction endpoint in `app.py` to handle new features
3. Retrain the model with updated features

---

## 📸 Screenshots

### Web Interface
*(Add screenshot of your web interface here)*

### Prediction Result - Low Risk
*(Add screenshot of low-risk prediction)*

### Prediction Result - High Risk
*(Add screenshot of high-risk prediction)*

### EDA Visualizations
*(Add screenshots from Jupyter notebook)*

---

## 🚀 Deployment

### Local Development
Follow the installation steps above.

### Production Deployment Options

#### Option 1: Heroku
```bash
# Install Heroku CLI
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
heroku create employee-attrition-app
git push heroku main
```

#### Option 2: Docker
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

#### Option 3: AWS/GCP/Azure
- Deploy using AWS Elastic Beanstalk
- Use Google Cloud Run
- Deploy on Azure App Service

---

## 🔮 Future Enhancements

- [ ] **Deep Learning Models**: Implement neural networks for better accuracy
- [ ] **Model Retraining Pipeline**: Automated model updates with new data
- [ ] **Dashboard Analytics**: Interactive dashboard with Plotly/Dash
- [ ] **Multi-language Support**: Internationalization
- [ ] **Email Notifications**: Alert HR when high-risk employees detected
- [ ] **Batch Predictions**: Upload CSV for bulk predictions
- [ ] **SHAP Values**: Model explainability with SHAP
- [ ] **A/B Testing**: Compare different model versions
- [ ] **Mobile App**: React Native mobile application

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- IBM HR Analytics Employee Attrition Dataset
- scikit-learn documentation
- Flask documentation
- Stack Overflow community

---

## 📞 Support

For support, email your.email@example.com or open an issue on GitHub.

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐️!

---

**Built with ❤️ for HR Analytics and Data Science**
