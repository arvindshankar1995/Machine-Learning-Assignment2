import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Title
st.title("HR Candidate Joining Prediction App")
st.markdown("""
This app predicts whether a candidate will **Join** or **Not Join** based on HR data.
Upload your test dataset to evaluate the model performance.
""")

# --- 1. Load and Train Models (Cached) ---
@st.cache_resource
def load_and_train_models():
    # Load Training Data
    try:
        df = pd.read_csv('hr_data.csv')
        
        # FIX: Clean column names to remove any leading/trailing spaces
        df.columns = df.columns.str.strip()
        
        # Check if 'Status' exists
        if 'Status' not in df.columns:
            st.error(f"CRITICAL ERROR: 'Status' column not found in hr_data.csv.\nFound columns: {list(df.columns)}")
            return None, None, None, None
            
    except FileNotFoundError:
        st.error("Error: 'hr_data.csv' not found. Please ensure it is in the GitHub repository.")
        return None, None, None, None

    # Preprocessing
    # Drop IDs
    df = df.drop(['SLNO', 'Candidate Ref'], axis=1, errors='ignore')
    
    # Target Encoding
    # Map 'Joined' to 1 and 'Not Joined' to 0
    df['Status'] = df['Status'].map({'Joined': 1, 'Not Joined': 0})
    
    # Feature Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Separation
    X = df_encoded.drop('Status', axis=1)
    y = df_encoded['Status']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10
