# 🚀 Complete Setup Guide - Employee Attrition Prediction System

This guide will walk you through setting up the project on your laptop from scratch.

---

## 📋 Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.8 or higher installed
- [ ] pip (Python package manager)
- [ ] Git installed
- [ ] A text editor or IDE (VS Code, PyCharm, etc.)
- [ ] At least 500MB free disk space

### Check Python Installation

Open terminal/command prompt and run:
```bash
python --version
# or
python3 --version
```

You should see something like: `Python 3.8.0` or higher

### Check pip Installation
```bash
pip --version
# or
pip3 --version
```

---

## 🔧 Step-by-Step Setup on Your Laptop

### Step 1: Download/Clone the Project

**Option A: If you have Git**
```bash
# Navigate to where you want to store the project
cd Desktop  # or any folder you prefer

# Clone the repository
git clone https://github.com/yourusername/employee-attrition-prediction.git

# Navigate into the project
cd employee-attrition-prediction
```

**Option B: If you don't have Git**
- Download the project as a ZIP file
- Extract it to your preferred location
- Open terminal/command prompt in that folder

---

### Step 2: Create Virtual Environment

**Why use a virtual environment?**
It keeps project dependencies isolated and prevents conflicts with other projects.

**On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# You should see (venv) at the start of your command prompt
```

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) at the start of your terminal
```

**✅ Success indicator:** Your command prompt should now start with `(venv)`

---

### Step 3: Install Required Packages

```bash
# Make sure you're in the project directory
# Install all dependencies at once
pip install -r requirements.txt
```

This will install:
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- scikit-learn (machine learning)
- Flask (web framework)
- jupyter (for notebooks)
- joblib (model saving)

**⏱️ Expected time:** 2-5 minutes depending on internet speed

**✅ Success indicator:** You should see "Successfully installed..." messages

---

### Step 4: Verify Installation

```bash
# Test Python packages
python -c "import pandas; import sklearn; import flask; print('All packages installed successfully!')"
```

**✅ Success indicator:** You should see "All packages installed successfully!"

---

### Step 5: Explore the Data (Optional but Recommended)

```bash
# Start Jupyter Notebook
jupyter notebook

# This will open a browser window
# Navigate to: notebooks/01_exploratory_data_analysis.ipynb
# Click "Run All" to see the data analysis
```

**What you'll see:**
- Dataset overview
- Attrition trends
- Beautiful visualizations
- Key insights

---

### Step 6: Train the Model

```bash
# Navigate to src directory
cd src

# Run the training script
python model_training.py

# Wait for training to complete (2-3 minutes)
```

**What happens during training:**
1. ✅ Data is loaded from `data/hr_employee_data.csv`
2. ✅ Data preprocessing (encoding, scaling, splitting)
3. ✅ Logistic Regression model is trained
4. ✅ Random Forest model is trained
5. ✅ Models are evaluated and compared
6. ✅ Best model is saved to `model/` directory
7. ✅ Evaluation plots are generated

**✅ Success indicator:**
```
✅ MODEL TRAINING PIPELINE COMPLETED!
🎯 Best Model: Random Forest
   Test Accuracy: 0.8571
   ROC-AUC: 0.8234
```

**📁 Files created:**
- `model/attrition_model.pkl` - Trained model
- `model/scaler.pkl` - Feature scaler
- `model/label_encoders.pkl` - Categorical encoders
- `model/feature_columns.pkl` - Feature names
- `model/model_metadata.pkl` - Model info
- `model/plots/` - Evaluation plots

---

### Step 7: Test Prediction Module (Optional)

```bash
# Still in src directory
python predict.py
```

This will run example predictions and show:
- High-risk employee profile → Prediction
- Low-risk employee profile → Prediction

---

### Step 8: Start the Web Application

```bash
# Navigate back to project root
cd ..

# Start Flask server
python app.py
```

**✅ Success indicator:**
```
🚀 EMPLOYEE ATTRITION PREDICTION API
✓ Model loaded
✓ Scaler loaded
✓ Label encoders loaded
🌐 Starting Flask server...
📍 Access the application at: http://localhost:5000
```

---

### Step 9: Use the Web Interface

1. **Open your web browser**
2. **Navigate to:** `http://localhost:5000`
3. **Fill in employee details:**
   - Age: 28
   - Monthly Income: 2500
   - Job Satisfaction: 1 (Low)
   - Years at Company: 1
   - Overtime: Yes
   - (Other fields are optional)
4. **Click "Predict Attrition Risk"**
5. **See the prediction result!**

**🎯 Expected Result:**
- Prediction: Employee likely to LEAVE 🚪
- Confidence: ~75-85%
- Attrition Probability: High
- Risk Level: 🚨 High Risk

**Try different combinations to see different predictions!**

---

## 🧪 Testing Different Scenarios

### Scenario 1: High-Risk Employee
```
Age: 28
Monthly Income: $2,500
Job Satisfaction: 1 (Low)
Years at Company: 1
Overtime: Yes
→ Expected: HIGH attrition risk
```

### Scenario 2: Low-Risk Employee
```
Age: 45
Monthly Income: $12,000
Job Satisfaction: 4 (Very High)
Years at Company: 15
Overtime: No
→ Expected: LOW attrition risk
```

### Scenario 3: Medium-Risk Employee
```
Age: 35
Monthly Income: $6,000
Job Satisfaction: 3 (High)
Years at Company: 5
Overtime: No
→ Expected: MEDIUM attrition risk
```

---

## 🔄 Stopping and Restarting

### To Stop the Flask Server
Press `Ctrl + C` in the terminal

### To Deactivate Virtual Environment
```bash
deactivate
```

### To Restart Everything Later

1. **Activate virtual environment:**
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

2. **Start Flask server:**
   ```bash
   python app.py
   ```

3. **Open browser:** `http://localhost:5000`

---

## 📤 Pushing to GitHub

### Step 1: Create GitHub Repository
1. Go to GitHub.com
2. Click "New Repository"
3. Name it: `employee-attrition-prediction`
4. Don't initialize with README (we already have one)
5. Click "Create Repository"

### Step 2: Initialize Git (if not already done)
```bash
# In project directory
git init
git add .
git commit -m "Initial commit: Employee Attrition Prediction System"
```

### Step 3: Connect to GitHub
```bash
# Replace 'yourusername' with your GitHub username
git remote add origin https://github.com/yourusername/employee-attrition-prediction.git
git branch -M main
git push -u origin main
```

### Step 4: Verify on GitHub
- Refresh your GitHub repository page
- You should see all files uploaded!

---

## 🎓 Understanding the Project Structure

```
employee-attrition-prediction/
│
├── 📂 data/                    # Dataset storage
│   └── hr_employee_data.csv    # IBM HR dataset
│
├── 📂 notebooks/               # Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb
│
├── 📂 src/                     # Source code
│   ├── data_preprocessing.py   # Data cleaning
│   ├── model_training.py       # ML training
│   └── predict.py              # Predictions
│
├── 📂 model/                   # Saved models
│   ├── attrition_model.pkl     # Trained model
│   ├── scaler.pkl              # Scaler
│   └── plots/                  # Visualizations
│
├── 📂 frontend/                # Web interface
│   └── index.html              # UI
│
├── app.py                      # Flask application
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## 🐛 Troubleshooting Common Issues

### Issue 1: "Python not found"
**Solution:** Install Python from python.org or use python3 instead of python

### Issue 2: "pip not found"
**Solution:** 
```bash
python -m pip install --upgrade pip
```

### Issue 3: "ModuleNotFoundError"
**Solution:**
```bash
# Make sure virtual environment is activated
pip install -r requirements.txt
```

### Issue 4: "Port 5000 already in use"
**Solution:**
```bash
# Stop other Flask apps or change port in app.py
# Edit app.py: app.run(port=5001)
```

### Issue 5: "Model not found"
**Solution:**
```bash
# Train the model first
cd src
python model_training.py
cd ..
python app.py
```

### Issue 6: Cannot access localhost:5000
**Solution:**
- Make sure Flask is running
- Check if firewall is blocking
- Try http://127.0.0.1:5000 instead

---

## ✅ Verification Checklist

Before considering setup complete, verify:

- [ ] Virtual environment created and activated
- [ ] All packages installed successfully
- [ ] Model trained successfully
- [ ] Model files exist in `model/` directory
- [ ] Flask server starts without errors
- [ ] Web interface loads at localhost:5000
- [ ] Prediction works with sample data
- [ ] No error messages in terminal

---

## 📚 Next Steps

1. **Experiment** with different employee profiles
2. **Explore** the Jupyter notebook for insights
3. **Modify** the model hyperparameters
4. **Add** new features to the prediction form
5. **Deploy** to Heroku or AWS (see README.md)
6. **Share** with friends and on GitHub!

---

## 🆘 Need Help?

- Check the troubleshooting section
- Review the README.md file
- Search for error messages online
- Open an issue on GitHub
- Contact: your.email@example.com

---

## 🎉 Congratulations!

You've successfully set up a complete, production-ready ML project!

**What you've learned:**
- ✅ Setting up Python projects
- ✅ Working with virtual environments
- ✅ Training ML models
- ✅ Building Flask APIs
- ✅ Creating web interfaces
- ✅ Git and GitHub basics

**This project demonstrates:**
- Data Science skills
- Machine Learning expertise
- Full-stack development
- Production deployment readiness

**Perfect for:**
- Final year projects
- Job interviews
- Portfolio showcase
- Learning ML end-to-end

---

**Happy Predicting! 🎯**
