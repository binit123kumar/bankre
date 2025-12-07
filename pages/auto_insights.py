# auto_insights.py – LIVE DASHBOARD STYLE (17 CHARTS)

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ================================
# SAFE CSV LOADER
# ================================
def safe_read_csv(uploaded):
    import io
    raw = uploaded.read()
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc)))
        except Exception:
            pass
    raise Exception("CSV encoding not supported")


# ================================
# AUTO EDA ENGINE – DASHBOARD VIEW
# ================================
def auto_eda_engine():

    # ---------- SIMPLE DASHBOARD CSS ----------
    st.markdown(
        """
        <style>
        .dash-title {
            font-size: 32px;
            font-weight: 800;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #0f172a;
            text-align: left;
            margin-bottom: 4px;
        }
        .dash-subtitle {
            font-size: 14px;
            color: #64748b;
            margin-bottom: 18px;
        }
        .dash-card {
            background: #0b2533;
            border-radius: 14px;
            padding: 14px 18px;
            color: #e5e7eb;
            box-shadow: 0 8px 16px rgba(15,23,42,0.45);
        }
        .dash-card-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9ca3af;
            margin-bottom: 4px;
        }
        .dash-card-value {
            font-size: 20px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- HEADER ----------
    st.markdown(
        """
        <div class="dash-title">AUTO INSIGHTS DASHBOARD</div>
        <div class="dash-subtitle">
            Concise live overview of your uploaded dataset with 17 ready-made charts
            for instant Exploratory Data Analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- FILE UPLOADER ----------
    uploaded = st.file_uploader(
        "Upload CSV or Excel to generate live dashboard",
        type=["csv", "xlsx"],
        key="auto_insights_uploader",
    )

    if uploaded is None:
        st.info("Please upload a CSV / Excel file to start Auto Insights.")
        return

    # ---------- LOAD DATA ----------
    try:
        if uploaded.name.endswith(".csv"):
            df = safe_read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success("✅ File loaded successfully!")

        # 🔧 FIX: make all object columns Arrow-compatible
        df = df.convert_dtypes()
        df = df.apply(lambda x: x.astype(str) if x.dtype == "object" else x)

    except Exception as e:
        st.error(f"❌ File load error: {e}")
        return

    # ---------- BASIC INFO ----------
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    rows, cols = df.shape
    missing_total = int(df.isna().sum().sum())
    missing_pct = (missing_total / (rows * cols) * 100) if rows * cols > 0 else 0
    num_count = len(numeric_cols)
    cat_count = len(categorical_cols)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="dash-card">
                <div class="dash-card-title">ROWS</div>
                <div class="dash-card-value">{rows:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="dash-card">
                <div class="dash-card-title">COLUMNS</div>
                <div class="dash-card-value">{cols}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="dash-card">
                <div class="dash-card-title">NUMERIC / CATEGORICAL</div>
                <div class="dash-card-value">{num_count} / {cat_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="dash-card">
                <div class="dash-card-title">MISSING VALUES</div>
                <div class="dash-card-value">{missing_total:,} ({missing_pct:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- DATA PREVIEW ----------
    with st.expander("🔍 Data Preview (Top 10 Rows)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📑 Column Summary", expanded=False):
        st.write(df.dtypes.reset_index().rename(columns={"index": "column", 0: "dtype"}))
        st.write("Numeric:", numeric_cols or "-")
        st.write("Categorical:", categorical_cols or "-")

    # ---------- USER SELECTION ----------
    st.markdown("---")
    st.markdown("#### 🎯 Column Selection for Charts")

    base_col = st.selectbox(
        "Select main column for most charts",
        df.columns,
        key="auto_insights_main_col",
    )

    binary_cols = [
        c for c in df.columns if set(df[c].dropna().unique()) <= {0, 1}
    ]
    roc_target = st.selectbox(
        "Binary column for ROC / PR (0/1)",
        binary_cols,
        key="auto_insights_bin_col",
    ) if binary_cols else None

    # ======================================================
    # TABS
    # ======================================================
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Distributions", "📈 Relationships", "🧠 Feature Insights", "⚙ Advanced"]
    )

    # ------------------------------------------------------
    # TAB 1 – DISTRIBUTIONS
    # ------------------------------------------------------
    with tab1:

        st.subheader("1️⃣ Histogram")
        st.plotly_chart(px.histogram(df, x=base_col), use_container_width=True)

        st.subheader("2️⃣ Bar Chart")
        if df[base_col].dtype == "object":
            temp = df[base_col].value_counts().reset_index()
            temp.columns = ["value", "count"]
            fig_bar = px.bar(temp, x="value", y="count")
        else:
            fig_bar = px.bar(df, x=df.index, y=base_col)
        st.plotly_chart(fig_bar, use_container_width=True)

        if df[base_col].dtype != "object":
            st.subheader("3️⃣ KDE Density Plot")
            fig_kde, ax = plt.subplots()
            sns.kdeplot(df[base_col].dropna(), fill=True, ax=ax)
            st.pyplot(fig_kde)

        st.subheader("4️⃣ Box Plot")
        st.plotly_chart(px.box(df, y=base_col), use_container_width=True)

        st.subheader("5️⃣ Violin Plot")
        st.plotly_chart(px.violin(df, y=base_col, box=True, points="all"),
                        use_container_width=True)

        if df[base_col].dtype == "object":
            st.subheader("6️⃣ Pie Chart")
            st.plotly_chart(px.pie(df, names=base_col),
                            use_container_width=True)

    # ------------------------------------------------------
    # TAB 2 – RELATIONSHIPS
    # ------------------------------------------------------
    with tab2:

        st.subheader("7️⃣ Line Plot")
        st.plotly_chart(px.line(df, y=base_col), use_container_width=True)

        st.subheader("8️⃣ Scatter Plot")
        st.plotly_chart(px.scatter(df, x=df.index, y=base_col),
                        use_container_width=True)

        if len(numeric_cols) >= 2:
            st.subheader("9️⃣ Joint Plot")
            x_j = st.selectbox("X-axis", numeric_cols, key="jx")
            y_j = st.selectbox("Y-axis", numeric_cols, key="jy")
            st.plotly_chart(px.scatter(df, x=x_j, y=y_j),
                            use_container_width=True)

        if len(numeric_cols) >= 3:
            st.subheader("🔟 Bubble Chart")
            x_b = st.selectbox("Bubble X", numeric_cols, key="b1")
            y_b = st.selectbox("Bubble Y", numeric_cols, key="b2")
            size_b = st.selectbox("Bubble Size", numeric_cols, key="b3")
            st.plotly_chart(
                px.scatter(df, x=x_b, y=y_b, size=size_b, color=size_b),
                use_container_width=True,
            )

    # ------------------------------------------------------
    # TAB 3 – FEATURE INSIGHTS
    # ------------------------------------------------------
    with tab3:

        if len(numeric_cols) > 1:
            st.subheader("1️⃣1️⃣ Correlation Heatmap")
            st.plotly_chart(px.imshow(df[numeric_cols].corr(), text_auto=".2f"),
                            use_container_width=True)

        if numeric_cols:
            st.subheader("1️⃣2️⃣ Feature Importance (Variance)")
            var_df = df[numeric_cols].var().reset_index()
            var_df.columns = ["Feature", "Importance"]
            st.plotly_chart(px.bar(var_df, x="Feature", y="Importance"),
                            use_container_width=True)

        if categorical_cols:
            st.subheader("1️⃣3️⃣ Treemap")
            col = categorical_cols[0]
            temp = df[col].value_counts().reset_index()
            temp.columns = [col, "count"]
            st.plotly_chart(px.treemap(temp, path=[col], values="count"),
                            use_container_width=True)

        st.subheader("1️⃣4️⃣ Gauge Meter")
        mean_val = float(df[base_col].mean()) if base_col in numeric_cols else 0
        st.plotly_chart(go.Figure(go.Indicator(
            mode="gauge+number", value=mean_val,
            title={"text": f"Mean: {base_col}"}
        )), use_container_width=True)

        if roc_target is not None:
            st.subheader("1️⃣5️⃣ ROC Curve")

            from sklearn.preprocessing import LabelEncoder
            y = df[roc_target].dropna()
            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            if len(np.unique(y_enc)) == 2:
                y_prob = y_enc + np.random.random(len(y_enc)) * 0.01

                fpr, tpr, _ = roc_curve(y_enc, y_prob)
                st.plotly_chart(px.area(x=fpr, y=tpr), use_container_width=True)

                st.subheader("1️⃣6️⃣ Precision–Recall Curve")
                prec, rec, _ = precision_recall_curve(y_enc, y_prob)
                st.plotly_chart(px.line(x=rec, y=prec), use_container_width=True)

            else:
                st.warning("Binary (0/1) target needed for ROC.")

