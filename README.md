# 🩺 Breast Cancer Diagnosis App  

This project is a **Machine Learning-based Breast Cancer Diagnosis App** built with **Streamlit**. It allows users to predict whether a tumor is **Benign (0) or Malignant (1)** using five different ML algorithms.

## ✨ Features
- **Supports five ML algorithms**:  
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Machine (SVM)  
  - Random Forest  
  - C4.5 (Decision Tree)  
- **Dual Input Methods**:  
  - **Manual Entry** (Enter values for features)  
  - **CSV Upload** (Batch predictions)  
- **Visualizations**: Displays a **radar chart** for feature comparison.  
- **Switch Models**: Users can easily switch between different ML models.  

---

## 📂 Project Structure

📁 Breast-Cancer-Prediction │── 📄 main.py # Main application file
│── 📄 requirements.txt # Required dependencies
│── 📄 environment.yml # Conda environment file
│── 📄 .gitignore # Files to exclude from version control
│── 📂 dataset # Contains the dataset
│ ├── cancer-diagnosis.csv


install dependencies using
'pip install -r requirements.txt'

Install Dependencies (Using Conda)  

'conda env create -f environment.yml'

'conda activate breast-cancer-diagnosis'

after installing the requirement file you can run with

streamlit run main.py




