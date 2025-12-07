import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)
from sklearn.exceptions import NotFittedError # Added for robustness

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor # XGBRegressor added for Auto ML
from fpdf import FPDF
from datetime import datetime

# Import auto_eda_engine safely for use in the 'Auto Insights' page
try:
    # Try importing from the same directory if it's the main app
    from auto_insights import auto_eda_engine 
except ImportError:
    # Handle module structure (e.g., if used with 'pages' folder in Streamlit)
    try:
        from pages.auto_insights import auto_eda_engine
    except ImportError:
        # Define a fallback function if the file cannot be found
        def auto_eda_engine():
            st.error("Error: The 'auto_insights.py' file could not be found or imported.")
            st.info("Please ensure 'auto_insights.py' is in the same directory as the main script.")


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bankruptcy Dashboard",
    layout="wide",
    page_icon="📊",
)

# ---------------------------------------------------------
# LOAD DATA (Make sure Bankruptcy.xlsx exists)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Assuming the file is named "Bankruptcy.xlsx" as per the original code
        return pd.read_excel("Bankruptcy.xlsx")
    except FileNotFoundError:
        st.error("Error: 'Bankruptcy.xlsx' not found. Please place the file in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df = load_data()


# ---------------------------------------------------------
# MODEL TRAINING (Wrapped in try/except for robustness)
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    data_copy = df.copy()

    # Preprocessing
    enc = LabelEncoder()
    data_copy["class_num"] = enc.fit_transform(data_copy["class"])

    feature_cols = [
        "industrial_risk",
        "management_risk",
        "financial_flexibility",
        "credibility",
        "competitiveness",
        "operating_risk",
    ]

    X = data_copy[feature_cols]
    y = data_copy["class_num"]

    # SMOTE for class balancing
    try:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
    except ValueError as e:
        # Handle case where SMOTE can't be applied (e.g., single class or very small data)
        st.warning(f"SMOTE skipped: {e}. Training on original data distribution.")
        X_res, y_res = X, y

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # XGBoost Model
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.07,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, zero_division=0) # zero_division added

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    feat_importance = model.feature_importances_

    return (
        model,
        feature_cols,
        acc,
        cm,
        rep,
        (fpr, tpr, auc_score),
        (feature_cols, feat_importance),
    )


# Run model training and unpack results
try:
    (
        model,
        feature_cols,
        acc,
        cm,
        report,
        roc_data,
        importance_data,
    ) = train_model()
except Exception as e:
    st.error(f"FATAL: Model training failed. Check data file or features. Error: {e}")
    # Define safe defaults to prevent code crash
    acc, cm, report, roc_data, importance_data = 0.0, np.array([[0, 0], [0, 0]]), "", ([], [], 0.0), ([], [])
    model = None # Set model to None if training failed
    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("⚙ Settings")

lang = st.sidebar.radio("Language", ["English", "Hindi"], index=0)
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Choose Page:",
    ["Home", "Dataset", "EDA", "Model Training", "Prediction", "Auto Insights", "Auto ML"],
)

# ---------------------------------------------------------
# THEME CSS (Unchanged - assumes your CSS works correctly)
# ---------------------------------------------------------
if theme == "Light":
    st.markdown(
        """
        <style>
        body { background-color: #f3f4f6; font-family: 'Segoe UI', sans-serif; }
        .fin-card {
            background: #ffffff;
            padding: 22px;
            border-radius: 16px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.10);
            margin-bottom: 22px;
        }
        .fin-title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            margin-bottom: -5px;
            color: #1F4E79;
        }
        .fin-subtitle {
            font-size: 18px;
            text-align: center;
            color: #4b5563;
            margin-bottom: 25px;
        }
        .topbar {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
            font-size: 18px;
            margin-bottom: 10px;
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body { background-color: #020617; color: #e5e7eb; font-family: 'Segoe UI', sans-serif; }
        .fin-card {
            background: #020617;
            padding: 22px;
            border-radius: 16px;
            border: 1px solid #1f2937;
            margin-bottom: 22px;
            box-shadow: 0 4px 24px rgba(15,23,42,0.85);
        }
        .fin-title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            margin-bottom: -5px;
            color: #e5e7eb;
        }
        .fin-subtitle {
            font-size: 18px;
            text-align: center;
            color: #9ca3af;
            margin-bottom: 25px;
        }
        .topbar {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
            font-size: 18px;
            margin-bottom: 10px;
            color: #e5e7eb;
        }
        .stMetric label, .stMetric span {
            color: #e5e7eb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# LANGUAGE HELPER
# ---------------------------------------------------------
def T(en, hi):
    return en if lang == "English" else hi

# ---------------------------------------------------------
# PDF REPORT
# ---------------------------------------------------------
def generate_pdf_report(input_df, probability, bankrupt_flag, accuracy, auc_value):
    # Safe PDF generation using 'latin-1' encoding for output
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, T("Bankruptcy Prediction Report", "दिवालियापन पूर्वानुमान रिपोर्ट"), ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", "", 11)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 8, f"{T('Date & Time', 'दिनांक व समय')}: {ts}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, T("Prediction Summary", "पूर्वानुमान सारांश"), ln=True)

    pdf.set_font("Arial", "", 11)
    result_text = T("Bankrupt", "दिवालिया") if bankrupt_flag else T("Not Bankrupt", "दिवालिया नहीं")
    pdf.cell(0, 8, f"{T('Prediction', 'पूर्वानुमान')}: {result_text}", ln=True)
    pdf.cell(0, 8, f"{T('Probability', 'संभावना')}: {probability:.2f}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, T("Model Details", "मॉडल विवरण"), ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"{T('Accuracy', 'शुद्धता')}: {accuracy*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"ROC AUC: {auc_value:.3f}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, T("Input Features", "इनपुट विशेषताएँ"), ln=True)

    pdf.set_font("Arial", "", 11)
    row = input_df.iloc[0]
    for col, val in row.items():
        # Use col and val.astype(str) to avoid float/int to string conversion errors
        pdf.cell(0, 8, f"{str(col)}: {str(val)}", ln=True) 

    pdf.ln(6)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(
        0,
        6,
        T(
            "Report generated using FinShield Prediction Tool",
            "यह रिपोर्ट FinShield Prediction Tool द्वारा तैयार की गई है",
        ),
    )

    return pdf.output(dest="S").encode("latin-1")

# ---------------------------------------------------------
# TOPBAR
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="topbar">
        <span>🧠💰</span>
        <span>{T('FinShield – Bankruptcy Risk Dashboard', 'FinShield – दिवालियापन जोखिम डैशबोर्ड')}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
if page == "Home":

    st.markdown(
        f"<div class='fin-title'>💼 {T('Bankruptcy Prediction System', 'दिवालियापन पूर्वानुमान प्रणाली')}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='fin-subtitle'>AI-driven bankruptcy risk estimation powered by SMOTE and XGBoost</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    total_records = len(df)
    # Safely get bankrupt count
    bankrupt_count = int(df["class"].value_counts().get("bankruptcy", 0))
    bank_pct = (bankrupt_count / total_records * 100) if total_records > 0 else 0

    with c1:
        st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
        st.metric(T("Total Records", "कुल रिकॉर्ड"), f"{total_records:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
        # Hindi translation corrected to 'दिवालिया केस' (Bankrupt Cases)
        st.metric(T("Bankrupt Cases", "दिवालिया केस"), str(bankrupt_count), f"{bank_pct:.1f}%") 
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
        st.metric(T("Model Accuracy", "मॉडल शुद्धता"), f"{acc*100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
    st.subheader(T("📘 About the System", "📘 सिस्टम के बारे में"))
    st.write(
        T(
            """
This tool estimates a company’s bankruptcy probability using a simple ML pipeline:

- Balanced training with SMOTE 
- XGBoost classifier 
- Interactive EDA 
- Real-time prediction with probability 
            """,
            """
यह टूल ML pipeline की मदद से कंपनी का दिवालियापन जोखिम अनुमानित करता है:

- SMOTE के साथ balanced training 
- XGBoost classifier 
- इंटरएक्टिव EDA 
- रीयल-टाइम संभावना आधारित पूर्वानुमान 
            """,
        )
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATASET PAGE
# ---------------------------------------------------------
elif page == "Dataset":
    st.header(T("📄 Dataset Preview", "📄 डाटासेट पूर्वावलोकन"))
    st.dataframe(df)

# ---------------------------------------------------------
# EDA PAGE
# ---------------------------------------------------------
elif page == "EDA":
    st.header(T("📊 Exploratory Data Analysis", "📊 खोजपरक डेटा विश्लेषण"))

    st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
    st.subheader(T("📌 Class Distribution", "📌 वर्ग वितरण"))
    fig_pie = px.pie(df, names="class", title="Safe vs Bankrupt")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
    st.subheader(T("🔥 Correlation Heatmap", "🔥 सहसम्बंध हीटमैप"))
    # Check if 'class_num' exists (it should, from train_model)
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)
    else:
        st.info("No numeric columns found for correlation.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
    st.subheader(T("📦 Box Plot", "📦 बॉक्स प्लॉट"))
    numeric_cols = numeric_df.columns.tolist()
    if numeric_cols:
        sel = st.selectbox(T("Choose Feature", "फीचर चुनें"), numeric_cols, key="eda_box_sel")
        fig_box = px.box(df, y=sel, points="all", title=f"Box Plot: {sel}")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No numeric columns available for Box Plot.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL TRAINING PAGE
# ---------------------------------------------------------
elif page == "Model Training":
    st.header(T("🤖 Model Performance", "🤖 मॉडल प्रदर्शन"))

    if model is None:
        st.error(T("Model not trained successfully.", "मॉडल सफलतापूर्वक प्रशिक्षित नहीं हुआ।"))
        st.stop()
        
    st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
    st.subheader(T("✅ Accuracy", "✅ शुद्धता"))
    st.success(f"{T('Accuracy', 'शुद्धता')}: {acc:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
    st.subheader(T("📌 Confusion Matrix", "📌 कन्फ्यूज़न मैट्रिक्स"))
    st.write(cm)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
    st.subheader(T("📌 Classification Report", "📌 वर्गीकरण रिपोर्ट"))
    st.text(report)
    st.markdown("</div>", unsafe_allow_html=True)

    fpr, tpr, auc_score = roc_data
    st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
    st.subheader(f"{T('📈 ROC Curve', '📈 ROC कर्व')} (AUC = {auc_score:.3f})")
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig_roc)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fin-card'>", unsafe_allow_html=True)
    st.subheader(T("📊 Feature Importance", "📊 फीचर महत्ता"))
    feat_names, imps = importance_data
    # Use try/except to handle case where feature importance is not available or empty
    try:
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": imps}).sort_values(
            "Importance", ascending=False
        )
        fig_imp = px.bar(imp_df, x="Feature", y="Importance", title=T("Feature Importance (XGBoost)", "फीचर महत्ता (XGBoost)"))
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info(T("Feature importance data is not available.", "फीचर महत्ता डेटा उपलब्ध नहीं है।"))
        
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------
elif page == "Prediction":
    st.header(T("🔮 Predict Bankruptcy", "🔮 दिवालियापन पूर्वानुमान"))

    if model is None:
        st.error(T("Prediction is unavailable as the model failed to train.", "मॉडल प्रशिक्षित न होने के कारण पूर्वानुमान अनुपलब्ध है।"))
        st.stop()
        
    risk_map = {"Low": 0, "Medium": 1, "High": 2}

    c1, c2, c3 = st.columns(3)
    industrial = c1.selectbox(T("Industrial Risk", "औद्योगिक जोखिम"), risk_map.keys())
    management = c2.selectbox(T("Management Risk", "प्रबंधन जोखिम"), risk_map.keys())
    financial = c3.selectbox(T("Financial Flexibility", "वित्तीय लचीलापन"), risk_map.keys())

    c4, c5, c6 = st.columns(3)
    credibility = c4.selectbox(T("Credibility", "विश्वसनीयता"), risk_map.keys())
    competitive = c5.selectbox(T("Competitiveness", "प्रतिस्पर्धात्मकता"), risk_map.keys())
    operating = c6.selectbox(T("Operating Risk", "संचालन जोखिम"), risk_map.keys())

    input_df = pd.DataFrame(
        {
            "industrial_risk": [risk_map[industrial]],
            "management_risk": [risk_map[management]],
            "financial_flexibility": [risk_map[financial]],
            "credibility": [risk_map[credibility]],
            "competitiveness": [risk_map[competitive]],
            "operating_risk": [risk_map[operating]],
        }
    )

    st.subheader(T("User Input", "उपयोगकर्ता इनपुट"))
    st.write(input_df)

    pdf_bytes = None
    fpr, tpr, auc_score = roc_data

    if st.button(T("Predict", "पूर्वानुमान करें")):
        try:
            # Predict probability for the positive class (1: Bankrupt)
            proba = model.predict_proba(input_df)[0][1]
            is_bankrupt = proba >= 0.6 # Use the same threshold as defined earlier

            if is_bankrupt:
                st.error(
                    f"⚠ {T('Company is Bankrupt', 'कंपनी दिवालिया है')} ({T('Prob', 'संभावना')}: {proba:.2f})"
                )
            else:
                st.success(
                    f"✔ {T('Company is Safe', 'कंपनी सुरक्षित है')} ({T('Prob', 'संभावना')}: {proba:.2f})"
                )

            # Generate PDF only on successful prediction
            pdf_bytes = generate_pdf_report(input_df, proba, is_bankrupt, acc, auc_score)
        
        except NotFittedError:
            st.error(T("Model not trained or features are missing.", "मॉडल प्रशिक्षित नहीं है या इनपुट सुविधाएँ गायब हैं।"))
        except Exception as e:
            st.error(f"Prediction Error: {e}")

    if pdf_bytes:
        st.download_button(
            label=T("⬇ Download PDF Report", "⬇ PDF रिपोर्ट डाउनलोड करें"),
            data=pdf_bytes,
            file_name="bankruptcy_report.pdf",
            mime="application/pdf",
        )

# ---------------------------------------------------------
# AUTO INSIGHTS PAGE
# ---------------------------------------------------------
elif page == "Auto Insights":
    # Call the imported function
    auto_eda_engine()
    
# ---------------------------------------------------------
# AUTO ML PAGE
# ---------------------------------------------------------
elif page == "Auto ML":
    st.header("🤖 Auto ML – Upload CSV/Excel & Train Model")

    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], key="auto_ml_uploader")

    if uploaded is not None:
        try:
            # Load file (using simple pandas read for Auto ML page)
            if uploaded.name.endswith(".csv"):
                df_auto = pd.read_csv(uploaded, on_bad_lines="skip")
            else:
                df_auto = pd.read_excel(uploaded)

        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

        st.success("File uploaded successfully!")
        st.subheader("📄 Dataset Preview")
        
        # Ensure column names are strings for later use
        df_auto.columns = df_auto.columns.astype(str)
        st.dataframe(df_auto.head())

        # ----------------------------
        # CLEAN DATA
        # ----------------------------
        st.subheader("🧹 Data Cleaning")
        initial_rows = len(df_auto)
        df_auto = df_auto.dropna()
        final_rows = len(df_auto)
        st.write(f"Removed {initial_rows - final_rows} rows with missing values.")

        # ----------------------------
        # TARGET SELECTION
        # ----------------------------
        st.subheader("🎯 Choose Target Column")
        target = st.selectbox("Target", df_auto.columns, key="auto_target")

        X = df_auto.drop(columns=[target])
        y = df_auto[target]
        
        if len(df_auto.columns) <= 1:
            st.error("Dataset has only one column after dropping NaNs. Cannot perform ML.")
            st.stop()

        # ----------------------------
        # DETECT PROBLEM TYPE
        # ----------------------------
        is_classification = False

        # Classification if target is object/string, or if unique values <= 20
        if y.dtype == "object":
            is_classification = True
        elif y.dtype in ["int64", "float64"] and len(y.unique()) <= min(20, len(y) * 0.5):
            is_classification = True

        st.info(f"Detected Problem Type: **{'Classification' if is_classification else 'Regression'}**")
        
        # ----------------------------
        # PREPROCESSING
        # ----------------------------
        st.subheader("⚙ Preprocessing")
        
        # Label Encoding for Classification Target
        le_target = None
        if is_classification:
            try:
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                st.write("Classification target encoded.")
            except:
                st.warning("Could not encode classification target. Proceeding with original values.")
                is_classification = False # Switch to regression model if encoding fails

        # One-Hot Encoding for Features
        X_enc = pd.get_dummies(X, drop_first=True, dtype=int)
        st.write(f"Features One-Hot Encoded. Total features: {X_enc.shape[1]}")
        
        # Drop any remaining non-numeric columns and columns with NaN (if any slipped through)
        X_enc = X_enc.select_dtypes(include=[np.number]).fillna(0)
        
        if X_enc.shape[1] == 0:
            st.error("No numeric features remaining after encoding/cleaning. Cannot train model.")
            st.stop()

        # ----------------------------
        # MODEL TRAINING
        # ----------------------------
        st.subheader("📚 Model Training")

        try:
            # Check for stratification eligibility
            can_stratify = False
            if is_classification:
                 unique_classes = np.unique(y)
                 # Stratify only if test set size for each class is at least 1
                 if all(count >= 2 for count in pd.Series(y).value_counts()):
                      can_stratify = True
                      
            if can_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_enc, y, test_size=0.2, random_state=42, stratify=y
                )
                st.write("Data split with stratification (20% Test Set).")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_enc, y, test_size=0.2, random_state=42
                )
                st.write("Data split without stratification (20% Test Set).")

            # Initialize Model
            if is_classification:
                model_auto = XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    eval_metric="logloss",
                    use_label_encoder=False,
                    random_state=42
                )
            else:
                # Use XGBRegressor for regression problem
                model_auto = XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )

            model_auto.fit(X_train, y_train)
            y_pred = model_auto.predict(X_test)

            # ----------------------------
            # RESULTS
            # ----------------------------
            st.subheader("📊 Results")
            if is_classification:
                st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred, zero_division=0))
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                st.success(f"Regression Model Trained Successfully.")
                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
                st.metric("R-squared (R2)", f"{r2:.4f}")
                
                # Show sample predictions vs actual
                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(10)
                st.write("Top 10 Actual vs Predicted Values:")
                st.dataframe(results_df)


        except Exception as e:
            st.error(f"Model Training Error: {e}")
            st.stop()

        # ---------------------------------------------------------
        # AUTO CHART ENGINE (Condensed version)
        # ---------------------------------------------------------
        def generate_auto_charts(df_in):
            st.header("📊 Auto Generated Insights (Key Charts)")
            
            # Use original dataframe for charts before OHE
            df_in.columns = df_in.columns.astype(str)
            num_cols = df_in.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = df_in.select_dtypes(include=['object']).columns.tolist()
            
            st.markdown("### Numeric Features")
            for col in num_cols[:4]: # Limit to top 4 for clarity
                st.subheader(f"📌 Distribution of {col}")
                st.plotly_chart(px.histogram(df_in, x=col), use_container_width=True)
            
            st.markdown("### Categorical Features")
            for col in cat_cols[:4]: # Limit to top 4 for clarity
                st.subheader(f"📌 Value Counts of {col}")
                tmp = df_in[col].value_counts().head(10).reset_index()
                tmp.columns = ["value", "count"]
                st.plotly_chart(px.bar(tmp, x="value", y="count"), use_container_width=True)
            
            if len(num_cols) > 1:
                st.markdown("### Correlation Heatmap")
                st.plotly_chart(px.imshow(df_in[num_cols].corr(), text_auto=".2f", title="Feature Correlation"), use_container_width=True)


        # ---------------------------------------------------------
        # USER SELECTED CHART ENGINE (Condensed version)
        # ---------------------------------------------------------
        def user_selected_charts(df_in):
            st.header("📌 User Selected Column Charts")

            cols = df_in.columns.tolist()
            selected = st.selectbox("Choose a column", cols, key="user_chart_sel")
            
            if selected and selected in df_in.columns:
                col_type = df_in[selected].dtype
                
                if col_type in ['int64', 'float64']:
                    st.plotly_chart(px.histogram(df_in, x=selected, title=f"Histogram: {selected}"))
                    st.plotly_chart(px.box(df_in, y=selected, title=f"Box Plot: {selected}"))
                else:
                    st.plotly_chart(px.pie(df_in, names=selected, title=f"Pie Chart: {selected}"))
                    st.plotly_chart(px.bar(df_in[selected].value_counts().head(10).reset_index(), 
                                           x='index', y=selected, title=f"Bar Chart: {selected}"))

        # ---------------------------------------------------------
        # SHOW BUTTONS
        # ---------------------------------------------------------
        st.markdown("## 📈 Auto Visualization")

        if st.checkbox("Show Auto Insights (Key Charts)"):
            generate_auto_charts(df_auto)

        if st.checkbox("Show User Selected Column Charts"):
            user_selected_charts(df_auto)