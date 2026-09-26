import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(page_title="Planetary Defense Analytics", page_icon="🪐", layout="wide")
st.title("🪐 Planetary Defense Analytics Dashboard")
st.markdown("Predicting and Analyzing Potentially Hazardous Asteroids (PHAs)")

# 2. Load Models
@st.cache_resource
def load_models():
    clf = joblib.load('models/best_classification_model.pkl')
    reg = joblib.load('models/best_regression_model.pkl')
    clus = joblib.load('models/kmeans_model.pkl')
    return clf, reg, clus

try:
    clf_model, reg_model, clus_model = load_models()
except Exception as e:
    st.error(f"Model Loading Error. Ensure models are saved in the 'models' folder. Error: {e}")
    st.stop()

# 3. Load Data
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned_pha_dataset.csv')

try:
    df = load_data()
except Exception as e:
    st.error(f"Data Loading Error. Ensure 'cleaned_pha_dataset.csv' is in 'data' folder. Error: {e}")
    st.stop()

# Get model feature names directly from trained models
clf_features = getattr(clf_model, 'feature_names_in_', None)
reg_features = getattr(reg_model, 'feature_names_in_', None)
clus_features = getattr(clus_model, 'feature_names_in_', None)

# 4. Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a Task", ["Dataset Overview", "Classification (PHA)", "Regression (Magnitude)", "Clustering"])

# 5. Application Modes
if app_mode == "Dataset Overview":
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(10))
    st.write(f"Total records in dataset: {df.shape[0]}")

elif app_mode == "Classification (PHA)":
    st.subheader("🛡️ Asteroid Hazard Classification")
    st.write("Select an asteroid index from the dataset to check if it is Potentially Hazardous (PHA).")
    
    index = st.number_input("Enter Asteroid Index (Row Number)", min_value=0, max_value=len(df)-1, value=0)
    selected_asteroid = df.iloc[[index]]
    
    if clf_features is not None:
        X_clf = selected_asteroid[clf_features].fillna(0)
    else:
        X_clf = selected_asteroid.select_dtypes(include=['float64', 'int64']).drop(columns=['pha'], errors='ignore').fillna(0)

    st.write("Selected Asteroid Data:")
    st.dataframe(X_clf)
    
    if st.button("Predict Hazard Status"):
        prediction = clf_model.predict(X_clf)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.error("⚠️ ALERT: This is a Potentially Hazardous Asteroid (PHA)!")
        else:
            st.success("✅ SAFE: This asteroid is not considered hazardous.")
            
        # --- Fixed SHAP Plot for Classification ---
        st.subheader("💡 Explainable AI (SHAP Plot)")
        st.write("This plot displays which features most strongly influenced the classification result:")
        try:
            explainer = shap.TreeExplainer(clf_model)
            shap_values = explainer.shap_values(X_clf)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Handle array dimension properly to avoid blank plot
            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif len(np.shape(shap_values)) == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values

            shap.summary_plot(sv, X_clf, plot_type="bar", show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Unable to generate SHAP plot: {e}")

elif app_mode == "Regression (Magnitude)":
    st.subheader("📏 Predict Absolute Magnitude (H)")
    index = st.number_input("Enter Asteroid Index", min_value=0, max_value=len(df)-1, value=0)
    selected_asteroid = df.iloc[[index]]
    
    if reg_features is not None:
        X_reg = selected_asteroid[reg_features].fillna(0)
    else:
        X_reg = selected_asteroid.select_dtypes(include=['float64', 'int64']).drop(columns=['H', 'pha'], errors='ignore').fillna(0)

    st.write("Selected Asteroid Data:")
    st.dataframe(X_reg)

    if st.button("Predict Magnitude"):
        pred_H = reg_model.predict(X_reg)
        
        st.markdown("---")
        st.info(f"Predicted Absolute Magnitude (H): **{pred_H[0]:.4f}**")
        
        if 'H' in df.columns:
            st.write(f"Actual Absolute Magnitude (H) in dataset: **{selected_asteroid['H'].values[0]}**")
            
        # --- SHAP Plot for Regression ---
        st.subheader("💡 Explainable AI (SHAP Plot)")
        st.write("This plot shows the feature contributions toward the predicted magnitude:")
        try:
            explainer = shap.TreeExplainer(reg_model)
            shap_values = explainer(X_reg)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.summary_plot(shap_values, X_reg, plot_type="bar", show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Unable to generate SHAP plot: {e}")

elif app_mode == "Clustering":
    st.subheader("🌌 Asteroid Grouping (Clustering)")
    index = st.number_input("Enter Asteroid Index to find its group", min_value=0, max_value=len(df)-1, value=0)
    selected_asteroid = df.iloc[[index]]
    
    if clus_features is not None:
        X_clus = selected_asteroid[clus_features].fillna(0)
    else:
        X_clus = selected_asteroid.select_dtypes(include=['float64', 'int64']).fillna(0)

    st.write("Selected Asteroid Data:")
    st.dataframe(X_clus)

    if st.button("Find Cluster Group"):
        cluster_id = clus_model.predict(X_clus)
        st.markdown("---")
        st.success(f"This asteroid belongs to **Cluster Group {cluster_id[0]}**")