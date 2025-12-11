# app.py — ChurnGuard AI (production-style, inspired by provided app)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Page config (first Streamlit call)
# -------------------------
st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")

# -------------------------
# Styling (simple dark-ish theme)
# -------------------------
st.markdown("""
<style>
body { background-color: #0f172a; color: #e6eef8; }
.card { padding:16px; border-radius:12px; background: rgba(255,255,255,0.03); margin-bottom:12px; }
.kpi { padding:14px; border-radius:10px; background: linear-gradient(90deg,#3A7BD5,#00d2ff); color:white; text-align:center; }
.smallmuted { color:#9aa8c4; font-size:0.95rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------
def load_models():
    """Load model artifacts from models/ directory."""
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/transformer.pkl", "rb") as f:
        transformer = pickle.load(f)
    # features may be used by SHAP plotting or display
    try:
        with open("models/features.pkl", "rb") as f:
            features = pickle.load(f)
    except Exception:
        features = None
    return model, transformer, features

def load_sample_data(path="data/sample_customers.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def preprocess_local(df):
    """Apply the same preprocessing/feature-engineering used during training."""
    df = df.copy()
    # drop id if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    # TotalCharges numeric
    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns and "tenure" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"], inplace=True)
    # engineered
    if {"TotalCharges","tenure","MonthlyCharges"}.issubset(df.columns):
        df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(df["tenure"], bins=[-1,12,36,100], labels=["New","Mid","Loyal"])
    # service_count
    svc_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    present = [c for c in svc_cols if c in df.columns]
    if present:
        df["service_count"] = (df[present] == "Yes").sum(axis=1)
    if "InternetService" in df.columns:
        df["is_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    return df

# -------------------------
# Load artifacts (cached)
# -------------------------
@st.cache_resource
def _load():
    try:
        model, transformer, features = load_models()
        return model, transformer, features
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

model, transformer, feature_names = _load()

# -------------------------
# Tabs: Upload & Predict | Visual Analysis | Explain (SHAP) | Model Info | About
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload & Predict",
    "📊 Visual Analysis",
    "🔬 Explain (SHAP)",
    "🧠 Model Info",
    "📘 About"
])

# -------------------------
# TAB 1 — Upload & Predict
# -------------------------
with tab1:
    st.header("📤 Upload customer CSV or try sample")
    colA, colB = st.columns([2,1])
    uploaded = colA.file_uploader("Upload CSV (same schema as training)", type=["csv"])
    sample_btn = colB.button("📄 Try sample (data/sample_customers.csv)")

    if sample_btn:
        df_raw = load_sample_data("data/sample_customers.csv")
        if df_raw is None:
            st.error("Sample file not found at data/sample_customers.csv")
            st.stop()
        st.success(f"Loaded sample data — {len(df_raw)} rows")
        uploaded_file_obj = io.StringIO(df_raw.to_csv(index=False))
        file_to_read = uploaded_file_obj
    elif uploaded:
        file_to_read = uploaded
    else:
        file_to_read = None

    if file_to_read:
        df_raw = pd.read_csv(file_to_read)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Preview (first 8 rows)")
        st.dataframe(df_raw.head(8))
        st.markdown("</div>", unsafe_allow_html=True)

        # Basic validation for required columns
        required = ["tenure","MonthlyCharges","TotalCharges","InternetService","Contract","PaymentMethod","OnlineSecurity"]
        missing = [c for c in required if c not in df_raw.columns]
        if missing:
            st.warning(f"Input missing some expected columns used for features: {missing}. Preprocessing will try to continue but results may be lower quality.")

        # Preprocess
        df_proc = preprocess_local(df_raw)
        # transform with saved transformer
        try:
            X_trans = transformer.transform(df_proc)
        except Exception as e:
            st.error(f"Failed to transform input with saved transformer: {e}")
            st.stop()

        # predict probabilities
        try:
            probs = model.predict_proba(X_trans)[:,1]
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.stop()

        df_out = df_raw.copy()
        df_out["churn_prob"] = np.round(probs, 4)
        df_out["churn_pred"] = (probs >= 0.5).astype(int)

        # Save to session for other tabs
        st.session_state["predictions"] = df_out

        # KPIs
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### Summary KPIs")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df_out)}")
        c2.metric("Avg churn prob", f"{df_out['churn_prob'].mean():.3f}")
        c3.metric("High risk (>=0.7)", int((df_out['churn_prob'] >= 0.7).sum()))
        st.markdown("</div>", unsafe_allow_html=True)

        # Show top-risk table
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### Top 20 highest-risk customers")
        st.dataframe(df_out.sort_values("churn_prob", ascending=False).head(20))
        st.markdown("</div>", unsafe_allow_html=True)

        # Download button
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download predictions CSV", csv_bytes, "churn_predictions.csv", "text/csv")

# -------------------------
# TAB 2 — Visual Analysis
# -------------------------
with tab2:
    st.header("📊 Visual analysis & segmentation")
    if "predictions" not in st.session_state:
        st.info("Please upload data in the 'Upload & Predict' tab or try the sample.")
    else:
        df = st.session_state["predictions"]

        # risk segmentation controls
        st.sidebar.subheader("Segmentation thresholds")
        high_thresh = st.sidebar.slider("High risk threshold", 0.6, 0.95, 0.7, 0.01)
        med_thresh = st.sidebar.slider("Medium risk threshold", 0.2, 0.6, 0.4, 0.01)

        df["risk_segment"] = pd.cut(df["churn_prob"],
                                    bins=[-0.01, med_thresh, high_thresh, 1.01],
                                    labels=["Low","Medium","High"])

        # KPI row
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("🔴 High risk", int((df["risk_segment"]=="High").sum()))
        b.metric("🟠 Medium risk", int((df["risk_segment"]=="Medium").sum()))
        c.metric("🟢 Low risk", int((df["risk_segment"]=="Low").sum()))
        st.markdown("</div>", unsafe_allow_html=True)

        # churn probability distribution
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### Churn probability distribution")
        fig, ax = plt.subplots(figsize=(8,3))
        sns.histplot(df["churn_prob"], bins=25, kde=True, color="#3A7BD5", ax=ax)
        ax.set_xlabel("Churn probability")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        # simple breakdown by contract
        if "Contract" in df.columns:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("### Average churn probability by Contract")
            agg = df.groupby("Contract")["churn_prob"].mean().sort_values(ascending=False)
            st.bar_chart(agg)
            st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# TAB 3 — Explain (SHAP)
# -------------------------
with tab3:
    st.header("🔬 Explainability (SHAP) — plain English")
    # lazy-shap import
    try:
        import shap
        shap_ok = True
    except Exception:
        shap_ok = False

    if not shap_ok:
        st.warning("SHAP is not installed in this environment. Install `shap==0.41.0` to enable explanations.")
        st.info("You can still view simple business insights in Model Info tab.")
    else:
        st.success("SHAP available ✔")
        # global summary image if saved
        shap_img_path = "models/shap_summary.png"
        if os.path.exists(shap_img_path):
            st.subheader("Global feature importance")
            st.image(shap_img_path)
        else:
            st.info("No precomputed SHAP summary found. Computing a small sample summary may take a moment.")
            if st.button("Compute SHAP summary on sample"):
                # compute SHAP summary using a small sample (best-effort)
                if "predictions" in st.session_state:
                    sample_df = st.session_state["predictions"]
                else:
                    sample_df = load_sample_data("data/sample_customers.csv")
                df_proc = preprocess_local(sample_df)
                X = transformer.transform(df_proc)
                X_arr = X.toarray() if hasattr(X, "toarray") else X
                sample_idx = np.random.choice(X_arr.shape[0], min(400, X_arr.shape[0]), replace=False)
                X_sample = X_arr[sample_idx]
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                plt.figure(figsize=(10,6))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                st.pyplot(plt.gcf()); plt.close()

        # per-customer explanation
        if "predictions" not in st.session_state:
            st.info("Upload data to explain a customer.")
        else:
            df = st.session_state["predictions"]
            st.subheader("Explain a single customer (choose by index or customerID)")
            method = st.radio("Select by", ["Index", "customerID"], horizontal=True)
            if method == "Index":
                idx = st.number_input("Row index (0-based)", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
            else:
                if "customerID" not in df.columns:
                    st.warning("customerID column not present in uploaded data; use Index mode.")
                    idx = 0
                else:
                    cust = st.selectbox("CustomerID", df["customerID"].astype(str).tolist())
                    idx = int(df[df["customerID"].astype(str)==cust].index[0])

            if st.button("Explain this customer"):
                try:
                    df_proc = preprocess_local(df)
                    X = transformer.transform(df_proc)
                    X_arr = X.toarray() if hasattr(X, "toarray") else X
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(X_arr[idx:idx+1])
                    # waterfall or bar fallback
                    plt.figure(figsize=(8,4))
                    try:
                        shap.plots.waterfall(sv[0], max_display=12)
                    except Exception:
                        vals = np.abs(sv[0])
                        ix = np.argsort(-vals)[:12]
                        names = [feature_names[i] for i in ix]
                        plt.barh(names[::-1], vals[ix][::-1])
                    st.pyplot(plt.gcf()); plt.close()

                    # human readable top reasons
                    top_ix = np.argsort(-np.abs(sv[0]))[:5]
                    st.markdown("### Plain-English reasons (top contributors)")
                    for i in top_ix:
                        fname = feature_names[i]
                        contrib = float(sv[0][i])
                        direction = "increases" if contrib > 0 else "decreases"
                        st.write(f"- **{fname}** {direction} the churn probability (impact {contrib:.3f})")
                    # simple retention suggestions (rule-based)
                    st.markdown("### Suggested retention actions")
                    suggestions = []
                    if "Contract" in df.columns and "Month-to-month" in df.loc[idx].values:
                        suggestions.append("Offer a 3-month discount to encourage switching to annual plan.")
                    if df.loc[idx].get("InternetService","") == "Fiber optic":
                        suggestions.append("Offer bundle discount or price lock for 6 months.")
                    if "Electronic check" in str(df.loc[idx].get("PaymentMethod","")):
                        suggestions.append("Remind about autopay and offer small incentive to switch.")
                    if len(suggestions) == 0:
                        suggestions.append("Assign a quick customer success outreach call to understand issues.")
                    for s in suggestions:
                        st.write("- " + s)

                except Exception as e:
                    st.error(f"Could not compute SHAP: {e}")

# -------------------------
# TAB 4 — Model Info
# -------------------------
with tab4:
    st.header("🧠 Model Info & Business Notes")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Model:** XGBoost classifier for churn probability")
    try:
        st.write(f"**Params:** trees={model.get_params().get('n_estimators')}, depth={model.get_params().get('max_depth')}, lr={model.get_params().get('learning_rate')}")
    except Exception:
        st.write("Model parameter details not available")
    st.markdown("</div>", unsafe_allow_html=True)

    # quick feature importances
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Top features (by gain)")
    try:
        imp = model.get_booster().get_score(importance_type="gain")
        imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:12]
        df_imp = pd.DataFrame(imp_sorted, columns=["feature","gain"]).set_index("feature")
        st.bar_chart(df_imp)
    except Exception as e:
        st.write("Feature importance unavailable:", e)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Business notes")
    st.write("- Model is tuned for recall on churn class (catch as many churners as possible).")
    st.write("- Use retention suggestions for targeted outreach and A/B test campaigns to measure lift.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# TAB 5 — About
# -------------------------
with tab5:
    st.header("📘 About ChurnGuard AI")
    st.markdown("""
**ChurnGuard AI** helps business teams identify customers at risk of leaving and gives simple, actionable reasons and suggested actions.

**How to use**
1. Upload your customer CSV (same schema as training) or try the sample.
2. Go to Visual Analysis to see distribution & segments.
3. Use Explain to get human-friendly reasons for an individual customer.
4. Use Model Info to understand top drivers.

Made by **Salman Mhaskar** — © 2025
    """)
