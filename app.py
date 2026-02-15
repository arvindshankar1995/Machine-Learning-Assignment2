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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix

# Title
st.title("HR Candidate Joining Prediction App")
st.markdown("""
This app predicts whether a candidate will **Join** or **Not Join** based on HR data.
Upload your test dataset to evaluate the model performance.
""")

# --- 1. Load and Train Models (Cached) ---
@st.cache_resource
def load_and_train_models():
    # Load Training Data (Assumes hr_data.csv is in the same folder)
    try:
        df = pd.read_csv('hr_data.csv')
    except FileNotFoundError:
        st.error("Error: 'hr_data.csv' not found. Please ensure it is in the GitHub repository.")
        return None, None, None, None

    # Preprocessing
    # Drop IDs
    df = df.drop(['SLNO', 'Candidate Ref'], axis=1, errors='ignore')
    
    # Target Encoding
    if 'Status' in df.columns:
        df['Status'] = df['Status'].map({'Joined': 1, 'Not Joined': 0})
    
    # Feature Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop('Status', axis=1)
    y = df_encoded['Status']
    
    # Scaling
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
            model.fit(X, y) # Trees don't need scaling
            
    return models, X.columns, scaler, list(df.columns)

# Load models
models, train_columns, scaler, original_columns = load_and_train_models()

# --- 2. Sidebar: Model Selection ---
st.sidebar.header("User Input")
model_name = st.sidebar.selectbox("Select Model", list(models.keys()) if models else [])

# --- 3. Main Interface: File Upload ---
uploaded_file = st.file_uploader("Upload your Test CSV (Must contain 'Status' column for evaluation)", type=["csv"])

if uploaded_file is not None and models:
    # Read Data
    test_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(test_df.head())
    
    # Preprocessing for Test Data
    try:
        # Drop IDs
        test_df_clean = test_df.drop(['SLNO', 'Candidate Ref'], axis=1, errors='ignore')
        
        # separate target if exists
        if 'Status' in test_df_clean.columns:
            # map target
            y_test = test_df_clean['Status'].map({'Joined': 1, 'Not Joined': 0})
            test_df_clean = test_df_clean.drop('Status', axis=1)
            has_labels = True
        else:
            has_labels = False
        
        # Encode Categoricals
        categorical_cols_test = test_df_clean.select_dtypes(include=['object']).columns
        test_encoded = pd.get_dummies(test_df_clean, columns=categorical_cols_test, drop_first=True)
        
        # Align Columns with Training Data (Crucial Step!)
        # This ensures the test data has exactly the same columns as the model expects
        test_encoded = test_encoded.reindex(columns=train_columns, fill_value=0)
        
        # Select Model
        model = models[model_name]
        
        # Predict
        if model_name == "Logistic Regression":
            X_test_scaled = scaler.transform(test_encoded)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(test_encoded)
            
        # Display Predictions
        st.subheader(f"Results using {model_name}")
        
        if has_labels:
            # --- Metrics ---
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.2%}")
            
            # --- Confusion Matrix ---
            st.write("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=['Not Joined', 'Joined'], yticklabels=['Not Joined', 'Joined'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # --- Classification Report ---
            st.write("#### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
        else:
            st.success("Predictions generated successfully!")
            test_df['Predicted_Status'] = ["Joined" if p==1 else "Not Joined" for p in y_pred]
            st.write(test_df[['Predicted_Status']])
            
    except Exception as e:
        st.error(f"Error processing data: {e}")

elif not models:
    st.warning("Models could not be trained. Check if 'hr_data.csv' is in the repo.")
else:
    st.info("Please upload a CSV file to proceed.")
