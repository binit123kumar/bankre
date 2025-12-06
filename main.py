import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Bankruptcy Dashboard", layout="wide", page_icon="📊")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("Bankruptcy.xlsx")

df = load_data()

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
@st.cache_resource
def train_model():

    df_enc = df.copy()

    le = LabelEncoder()
    df_enc["class_num"] = le.fit_transform(df_enc["class"])

    feature_cols = [
        "industrial_risk", "management_risk", "financial_flexibility",
        "credibility", "competitiveness", "operating_risk"
    ]

    X = df_enc[feature_cols]
    y = df_enc["class_num"]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.07,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, feature_cols, acc, cm, report, le


model, feature_cols, acc, cm, report, le = train_model()


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Dataset", "EDA", "Model Training", "Prediction"]
)

# ---------------------------------------------------------
# HOME PAGE UI
# ---------------------------------------------------------
if page == "Home":

    # Custom CSS
    st.markdown("""
        <style>
            .title {
                font-size: 44px;
                font-weight: 700;
                text-align: center;
                color: #1F4E79;
                margin-top: -20px;
            }
            .subtitle {
                text-align: center;
                font-size: 20px;
                color: #4a4a4a;
                margin-bottom: 30px;
            }
            .card {
                background-color: #ffffff;
                padding: 25px;
                border-radius: 16px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                margin-top: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>💼 Bankruptcy Prediction System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-powered Financial Risk Prediction using SMOTE + XGBoost</div>", unsafe_allow_html=True)

    # Lottie Animation
    import requests
    try:
        from streamlit_lottie import st_lottie
        def load_lottie(url):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
        ani = load_lottie("https://assets10.lottiefiles.com/packages/lf20_puciaact.json")
        if ani:
            st_lottie(ani, height=250)
    except:
        st.info("⚠ Lottie animation module not installed. Run:  pip install streamlit-lottie")

    # About Section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📘 About the System")
    st.write("""
        This platform predicts **company bankruptcy risk** using advanced ML techniques.
        
        ### 🔍 Key Features
        ✔ Interactive EDA  
        ✔ SMOTE Balanced Dataset  
        ✔ High-Accuracy XGBoost Model  
        ✔ Real-Time Probability Score  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Workflow
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🚀 How It Works")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
            **Step 1:** Load dataset  
            **Step 2:** EDA visualization  
            **Step 3:** SMOTE data balancing  
            **Step 4:** XGBoost training  
            **Step 5:** Bankruptcy prediction  
        """)

    with col2:
        st.image("https://i.imgur.com/jxWqIBf.png", use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Sample Output
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Sample Prediction Output")
    st.image("https://i.imgur.com/HdhW0wL.png", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><center>💙 Developed by Binit</center>", unsafe_allow_html=True)


# ---------------------------------------------------------
# DATASET PAGE
# ---------------------------------------------------------
elif page == "Dataset":
    st.header("📄 Dataset Preview")
    st.dataframe(df)


# ---------------------------------------------------------
# EDA PAGE (FIXED HEATMAP)
# ---------------------------------------------------------
elif page == "EDA":
    st.header("📊 Exploratory Data Analysis (Advanced)")

    st.markdown("""
        <style>
            .card {
                background-color: #ffffff;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
                margin-bottom: 25px;
            }
        </style>
    """, unsafe_allow_html=True)

    # 1️⃣ PIE CHART
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📌 Class Distribution")
    fig = px.pie(df, names="class", title="Safe vs Bankrupt")
    st.plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # 2️⃣ BAR CHART
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📌 Class Count")
    df_class = df["class"].value_counts().reset_index()
    df_class.columns = ["Class", "Count"]
    fig = px.bar(df_class, x="Class", y="Count", title="Class Count")
    st.plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # 3️⃣ ⭐ FIXED HEATMAP
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔥 Correlation Heatmap (Numeric Only)")
    df_corr = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)
    st.markdown("</div>", unsafe_allow_html=True)

    # 4️⃣ BOX PLOT
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📦 Box Plot")
    numeric_cols = df_corr.columns.tolist()
    selected = st.selectbox("Select Feature", numeric_cols)
    fig = px.box(df, y=selected, points="all", title=f"Box Plot: {selected}")
    st.plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # 5️⃣ MEAN BAR CHART
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Average Feature Values")
    mean_df = df_corr.mean().reset_index()
    mean_df.columns = ["Feature", "Average"]
    fig = px.bar(mean_df, x="Feature", y="Average", title="Mean Feature Values")
    st.plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# MODEL TRAINING PAGE
# ---------------------------------------------------------
elif page == "Model Training":
    st.header("🤖 Model Performance")
    st.success(f"Accuracy: {acc:.4f}")

    st.subheader("📌 Confusion Matrix")
    st.write(cm)

    st.subheader("📌 Classification Report")
    st.text(report)

# ---------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------
elif page == "Prediction":
    st.header("🔮 Predict Bankruptcy")

    risk_map = {"Low": 0.0, "Medium": 0.5, "High": 1.0}

    col1, col2, col3 = st.columns(3)
    industrial = col1.selectbox("Industrial Risk", risk_map.keys())
    management = col2.selectbox("Management Risk", risk_map.keys())
    financial = col3.selectbox("Financial Flexibility", risk_map.keys())

    col4, col5, col6 = st.columns(3)
    credibility = col4.selectbox("Credibility", risk_map.keys())
    competitive = col5.selectbox("Competitiveness", risk_map.keys())
    operating = col6.selectbox("Operating Risk", risk_map.keys())

    input_df = pd.DataFrame({
        "industrial_risk": [risk_map[industrial]],
        "management_risk": [risk_map[management]],
        "financial_flexibility": [risk_map[financial]],
        "credibility": [risk_map[credibility]],
        "competitiveness": [risk_map[competitive]],
        "operating_risk": [risk_map[operating]],
    })

    st.subheader("User Input")
    st.write(input_df)

    if st.button("Predict"):
        proba = model.predict_proba(input_df)[0][1]
        if proba >= 0.65:
            st.error(f"⚠ Company is *Bankrupt* (Prob: {proba:.2f})")
        else:
            st.success(f"✔ Company is *Safe* (Prob: {proba:.2f})")
