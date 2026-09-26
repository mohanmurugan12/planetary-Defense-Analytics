# 🪐 Planetary Defense Analytics Dashboard

An interactive Web Application built with **Streamlit** and **Machine Learning** to analyze and predict Potentially Hazardous Asteroids (PHAs) and estimate their Absolute Magnitude.

---

## 📌 Features
* **Dataset Overview:** Explore cleaned PHA asteroid data.
* **Hazard Classification:** Predict whether an asteroid is hazardous or safe using Machine Learning (Classification) accompanied by **SHAP Explainable AI** plots.
* **Magnitude Regression:** Estimate Absolute Magnitude (H) with feature contribution analysis.
* **Asteroid Grouping:** Cluster asteroids into categories using K-Means Clustering.

---

## 🛠️ Project Structure
```text
├── data/
│   └── cleaned_pha_dataset.csv
├── models/
│   ├── best_classification_model.pkl
│   ├── best_regression_model.pkl
│   └── kmeans_model.pkl
├── app.py
├── requirements.txt
└── README.md

git clone [https://github.com/YOUR_USERNAME/Planetary-Defense-Analytics.git](https://github.com/YOUR_USERNAME/Planetary-Defense-Analytics.git)
cd Planetary-Defense-Analytics

pip install -r requirements.txt

streamlit run app.py

