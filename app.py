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
st.title("Credit Card Default Prediction App")
st.markdown("""
This app predicts whether a customer will **Default** (1) or **Not Default** (0) next month.
Upload your test dataset to evaluate model performance.
""")

# --- 1. Load and Train Models (Cached) ---
@st.cache_resource
def load_and_train_models():
    # Load Training Data
    try:
        df = pd.read_csv('UCI_Credit_Card.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Target Column Name in UCI dataset
        target_col = 'default.payment.next.month'
        
        # Check if target exists
        if target_col not in df.columns:
            st.error(f"CRITICAL ERROR: '{target_col}' column not found in CSV.\nFound columns: {list(df.columns)}")
            return None, None, None, None
            
    except FileNotFoundError:
        st.error("Error: 'UCI_Credit_Card.csv' not found. Please ensure it is in the GitHub repository.")
        return None, None, None, None

    # Preprocessing
    # Drop ID column if it exists
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Feature Separation
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Scaling (Important for Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=10, max_depth=8, min_samples_leaf=20, random_state=42)
    }
    
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_scaled, y)
        else:
            model.fit(X, y)
            
    return models, X.columns, scaler, target_col

# Load models
models, train_columns, scaler, target_col = load_and_train_models()

# Stop execution if models failed to load
if models is None:
    st.stop()

# --- 2. Sidebar: Model Selection ---
st.sidebar.header("User Input")
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# --- 3. Main Interface: File Upload ---
uploaded_file = st.file_uploader(f"Upload your Test CSV (Must contain '{target_col}' column for evaluation)", type=["csv"])

if uploaded_file is not None:
    # Read Data
    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip() # Clean test data columns
    
    st.write("### Uploaded Data Preview")
    st.dataframe(test_df.head())
    
    # Preprocessing for Test Data
    try:
        # Drop ID if exists
        test_df_clean = test_df.drop('ID', axis=1, errors='ignore')
        
        # Handle Target if exists
        has_labels = False
        if target_col in test_df_clean.columns:
            y_test = test_df_clean[target_col]
            test_df_clean = test_df_clean.drop(target_col, axis=1)
            has_labels = True
        
        # Align Columns (Crucial!)
        # Ensure test data has the same columns as training data
        test_df_clean = test_df_clean.reindex(columns=train_columns, fill_value=0)
        
        # Select Model
        model = models[model_name]
        
        # Predict
        if model_name == "Logistic Regression":
            X_test_scaled = scaler.transform(test_df_clean)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(test_df_clean)
            
        # Display Results
        st.subheader(f"Results using {model_name}")
        
        if has_labels:
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.2%}")
            
            st.write("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            st.write("#### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
        else:
            st.success("Predictions generated!")
            test_df['Predicted_Default'] = y_pred
            st.write(test_df[['Predicted_Default']])
            
    except Exception as e:
        st.error(f"Error processing data: {e}")

else:
    st.info("Please upload a CSV file to proceed.")
