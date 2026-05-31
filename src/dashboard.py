"""
MLOps Monitoring Dashboard
Live view of the Tetouan power consumption pipeline.
CL04 G05 - AI for Engineering
"""
import json
import os
import subprocess
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# Optional imports - degrade gracefully if not present at runtime
try:
    import numpy as np
    import joblib
except Exception:
    np = None
    joblib = None

# -----------------------------
# Paths
# -----------------------------
ARTIFACTS_METRICS = "artifacts/metrics"
ARTIFACTS_METADATA = "artifacts/metadata"
ARTIFACTS_MODELS = "artifacts/models"
ARTIFACTS_DATA = "artifacts/data"
REPORTS = "reports"

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="MLOps Dashboard - CL04 G05",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    header[data-testid="stHeader"] { display: none; }
    [data-testid="stDecoration"] { display: none; }
    .block-container { padding-top: 2rem !important; }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    .main-header {
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        padding: 2rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.5rem; font-weight: 700; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0 0; font-size: 1.1rem; }

    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700;
        color: #f1f5f9;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .stMarkdown, .stCaption, p { color: #cbd5e1; }

    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-success { background: rgba(16, 185, 129, 0.15); color: #10b981; }
    .badge-warn    { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .badge-danger  { background: rgba(239, 68, 68, 0.15); color: #ef4444; }

    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_text(path):
    if not os.path.exists(path):
        return "N/A"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "N/A"


def style_chart(ax, fig):
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_color("#64748b")
    ax.tick_params(colors="#cbd5e1")
    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    ax.title.set_color("#f1f5f9")
    ax.grid(alpha=0.15, color="#94a3b8")
    if ax.get_legend():
        ax.get_legend().get_frame().set_facecolor("#0f172a")
        ax.get_legend().get_frame().set_edgecolor("#334155")
        for text in ax.get_legend().get_texts():
            text.set_color("#cbd5e1")


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### Project Info")
    st.markdown("**Unit:** AI for Engineering")
    st.markdown("**Class:** CL04")
    st.markdown("**Group:** G05")
    st.markdown("**Task:** 3 - Retraining & Monitoring")
    st.markdown("---")
    st.markdown("### Team")
    st.markdown("- **Sandaru** - Data Engineer")
    st.markdown("- **Binara** - ML Engineer")
    st.markdown("- **Matthew** - MLOps / DevOps")
    st.markdown("- **Ashen** - Project Manager")
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("[GitHub Repo](https://github.com/Sandaru757/Task-3-Automated-retraining-and-monitoring-pipeline.-CL04-G05)")
    st.markdown("[DagsHub Storage](https://dagshub.com/Sandaru757/Task-3-Automated-retraining-and-monitoring-pipeline.-CL04-G05)")
    st.markdown("---")
    if st.button("Refresh Dashboard", use_container_width=True):
        st.rerun()
    st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")


# -----------------------------
# Load artifacts
# -----------------------------
training_history = load_json(f"{ARTIFACTS_METRICS}/training_history.json")
evaluation_metrics = load_json(f"{ARTIFACTS_METRICS}/evaluation_metrics.json")
drift_report = load_json(f"{REPORTS}/drift_report.json")
monitoring_metrics = load_json(f"{ARTIFACTS_METRICS}/monitoring_metrics.json")
last_retrain = load_text(f"{ARTIFACTS_METADATA}/last_retrain.txt")
model_version = load_text(f"{ARTIFACTS_METADATA}/model_version.txt")


# -----------------------------
# Hero Header
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>Power Consumption MLOps Dashboard</h1>
    <p>Automated retraining, drift detection and monitoring for the Tetouan power consumption prediction system</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# System Status
# -----------------------------
st.markdown("### System Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Version", model_version if model_version != "N/A" else "v1.0")

with col2:
    if training_history and "training_timestamp" in training_history:
        st.metric("Last Trained", training_history["training_timestamp"])
    else:
        st.metric("Last Retrain", last_retrain[:19] if last_retrain != "N/A" else "N/A")

with col3:
    if evaluation_metrics:
        r2 = evaluation_metrics.get("metrics", {}).get("r2_score", 0)
        st.metric("Test R-squared", f"{r2:.3f}")
    else:
        st.metric("Test R-squared", "N/A")

with col4:
    if drift_report:
        overall = drift_report.get("overall_drift", {})
        drifted = overall.get("drifted_features", 0)
        total = overall.get("total_features", 0)
        st.metric("Features Drifted", f"{drifted} / {total}")
    else:
        st.metric("Features Drifted", "N/A")


# -----------------------------
# Health Banner
# -----------------------------
if drift_report and evaluation_metrics:
    share = drift_report.get("overall_drift", {}).get("drift_percentage", 0)
    r2 = evaluation_metrics.get("metrics", {}).get("r2_score", 0)

    if share > 0.5 or r2 < 0.3:
        st.markdown(
            '<div style="padding: 1rem; background: rgba(239, 68, 68, 0.15); '
            'border-left: 4px solid #ef4444; border-radius: 8px; margin-top: 1rem;">'
            '<span class="badge badge-danger">RETRAIN RECOMMENDED</span>'
            '<span style="margin-left: 1rem; color: #cbd5e1;">Significant drift detected. The automated workflow should trigger retraining.</span>'
            '</div>',
            unsafe_allow_html=True
        )
    elif share > 0.2 or r2 < 0.6:
        st.markdown(
            '<div style="padding: 1rem; background: rgba(245, 158, 11, 0.15); '
            'border-left: 4px solid #f59e0b; border-radius: 8px; margin-top: 1rem;">'
            '<span class="badge badge-warn">MONITOR</span>'
            '<span style="margin-left: 1rem; color: #cbd5e1;">Some drift detected. Pipeline operating within tolerance.</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="padding: 1rem; background: rgba(16, 185, 129, 0.15); '
            'border-left: 4px solid #10b981; border-radius: 8px; margin-top: 1rem;">'
            '<span class="badge badge-success">HEALTHY</span>'
            '<span style="margin-left: 1rem; color: #cbd5e1;">Pipeline operating normally. No retraining needed.</span>'
            '</div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------
# Training & Evaluation Metrics
# -----------------------------
left, right = st.columns(2)

with left:
    st.markdown("### Training Metrics")
    if training_history:
        tm = training_history.get("training_metrics", {})
        sub1, sub2, sub3 = st.columns(3)
        with sub1:
            st.metric("MAE", f"{tm.get('mae', 0):.0f}")
        with sub2:
            st.metric("RMSE", f"{tm.get('rmse', 0):.0f}")
        with sub3:
            st.metric("R-squared", f"{tm.get('r2_score', 0):.3f}")
        with st.expander("Hyperparameters"):
            st.json(training_history.get("hyperparameters", {}))
        st.caption(f"Trained on {training_history.get('training_samples', 0):,} samples, "
                   f"{training_history.get('number_of_features', 0)} features")
    else:
        st.warning("No training history found.")

with right:
    st.markdown("### Evaluation Metrics")
    if evaluation_metrics:
        em = evaluation_metrics.get("metrics", {})
        sub1, sub2, sub3 = st.columns(3)
        with sub1:
            st.metric("MAE", f"{em.get('mae', 0):.0f}")
        with sub2:
            st.metric("RMSE", f"{em.get('rmse', 0):.0f}")
        with sub3:
            st.metric("R-squared", f"{em.get('r2_score', 0):.3f}")
        st.caption(f"Tested on {evaluation_metrics.get('test_samples', 0):,} samples - "
                   "data the model never saw during training")
    else:
        st.warning("No evaluation metrics found.")

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------
# Drift Detection
# -----------------------------
st.markdown("### Drift Detection")

if drift_report:
    overall = drift_report.get("overall_drift", {})
    feature_drift_dict = drift_report.get("feature_drift", {})

    # Sort features by their key order (feature_0, feature_1, ...) numerically
    def _key_num(k):
        try:
            return int(k.split("_")[-1])
        except Exception:
            return 0

    feature_keys = sorted(feature_drift_dict.keys(), key=_key_num)
    feature_list = [feature_drift_dict[k] for k in feature_keys]

    drift_col1, drift_col2 = st.columns([1, 2])

    with drift_col1:
        st.metric("Method", "KS Test")
        st.metric("p-value Threshold", "0.05")
        share = overall.get("drift_percentage", 0)
        st.metric("Drift Share", f"{share * 100:.1f}%")
        st.caption(f"{overall.get('drifted_features', 0)} of {overall.get('total_features', 0)} features drifted")

    with drift_col2:
        if feature_list:
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ["#ef4444" if r.get("drift_detected") else "#10b981" for r in feature_list]
            ax.bar(range(len(feature_list)),
                   [r.get("ks_statistic", 0) for r in feature_list],
                   color=colors, edgecolor="none")
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("KS Statistic")
            ax.set_title("Per-Feature Drift (red = drifted, green = stable)")
            style_chart(ax, fig)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No per-feature drift data available.")
else:
    st.warning("No drift report found.")

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------
# Predicted vs Actual
# -----------------------------
st.markdown("### Predicted vs Actual")

model_path = f"{ARTIFACTS_MODELS}/model.pkl"
x_test_path = f"{ARTIFACTS_DATA}/X_test.npy"
y_test_path = f"{ARTIFACTS_DATA}/y_test.npy"

predictions_data = load_json(f"{ARTIFACTS_METRICS}/predictions_sample.json")

if predictions_data:
    predicted_arr = predictions_data.get("predicted", [])
    actual_arr = predictions_data.get("actual", [])
    total_samples = predictions_data.get("total_samples", len(actual_arr))

    if predicted_arr and actual_arr:
        max_n = len(actual_arr)
        n_samples = st.slider(
            "Window size (test samples)",
            min_value=50, max_value=max_n,
            value=min(500, max_n), step=50
        )

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(actual_arr[:n_samples], label="Actual", color="#3b82f6", linewidth=1.8)
        ax.plot(predicted_arr[:n_samples], label="Predicted", color="#f59e0b", linewidth=1.8, alpha=0.85)
        ax.set_xlabel("Time step (10-minute intervals)")
        ax.set_ylabel("Zone 1 Power Consumption")
        ax.set_title(f"Predicted vs Actual - First {n_samples} Test Samples")
        ax.legend(loc="upper right")
        style_chart(ax, fig)
        st.pyplot(fig, use_container_width=True)

        pred_mean = predictions_data.get("mean_predicted_all", 0)
        actual_mean = predictions_data.get("mean_actual_all", 0)
        bias = pred_mean - actual_mean

        cap_col1, cap_col2, cap_col3 = st.columns(3)
        with cap_col1:
            st.metric("Mean Predicted", f"{pred_mean:,.0f}")
        with cap_col2:
            st.metric("Mean Actual", f"{actual_mean:,.0f}")
        with cap_col3:
            st.metric("Bias", f"{bias:+,.0f}")

        st.caption(f"Showing first {n_samples} of {total_samples:,} total test samples (pre-computed)")
    else:
        st.info("Prediction sample is empty.")
else:
    st.info("Predicted vs Actual chart requires the predictions sample file. Run the local pipeline first.")


# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    Built with Streamlit | Powered by DVC + DagsHub + GitHub Actions<br>
    AI for Engineering | Swinburne University | CL04 G05
</div>
""", unsafe_allow_html=True)