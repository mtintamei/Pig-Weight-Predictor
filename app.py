import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load Data & Model
# ------------------------------
@st.cache_data

def load_data():
    url = "https://raw.githubusercontent.com/mtintamei/Pig-Weight-Predictor/main/pig_fattening_data.csv"
    return pd.read_csv(url)

summary_df = load_data()

# Model setup
X = summary_df[['start_weight', 'total_feed_intake_kg', 'days_in_fattening', 'feed_type']]
y = summary_df['end_weight']

categorical_features = ['feed_type']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ------------------------------
# 2. Streamlit App UI
# ------------------------------
st.title("🐖 Pig End Weight Predictor")
st.write("""
Estimate the final weight of a pig based on its starting weight, total feed intake, days in fattening, and feed type.
This model is a demo built for learning purposes only.
""")

# Metrics
st.markdown(f"**Model Performance**")
st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
st.metric(label="R² Score", value=f"{r2:.2%}")

# Distribution Plot
st.subheader("📊 End Weight Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(summary_df['end_weight'], bins=20, kde=True, ax=ax1)
ax1.set_title("Distribution of End Weights")
ax1.set_xlabel("End Weight (kg)")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# Scatter Plot
st.subheader("📈 Feed Intake vs End Weight")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=summary_df, x='total_feed_intake_kg', y='end_weight', hue='feed_type', ax=ax2)
ax2.set_title("Feed Intake vs End Weight")
st.pyplot(fig2)

# ------------------------------
# 3. Interactive Prediction
# ------------------------------
st.subheader("🔍 Predict a Pig's End Weight")
start_weight = st.number_input("Start Weight (kg)", min_value=10.0, max_value=60.0, value=30.0)
total_feed = st.number_input("Total Feed Intake (kg)", min_value=100.0, max_value=250.0, value=160.0)
days = st.number_input("Days in Fattening", min_value=50, max_value=100, value=75)
feed_type = st.selectbox("Feed Type", options=summary_df['feed_type'].unique())

if st.button("Predict End Weight"):
    input_df = pd.DataFrame({
        'start_weight': [start_weight],
        'total_feed_intake_kg': [total_feed],
        'days_in_fattening': [days],
        'feed_type': [feed_type]
    })
    predicted_weight = pipeline.predict(input_df)[0]
    st.success(f"✅ Predicted End Weight: {predicted_weight:.2f} kg")

    # Optional ROI Estimator
    st.subheader("💰 Estimate Sale Revenue")
    price_per_kg = st.number_input("Market Price per Kg (KES)", min_value=100.0, max_value=1000.0, value=450.0)
    est_price = predicted_weight * price_per_kg
    st.write(f"Estimated Sale Price: **KES {est_price:,.0f}**")

# ------------------------------
# 4. Footer
# ------------------------------
st.markdown("""
---
**Author:** Musa Tintamei | [GitHub Repo](https://github.com/mtintamei/Pig-Weight-Predictor)
""")
