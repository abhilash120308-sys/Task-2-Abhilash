"""
app.py - DecodeLabs AI Project 2: Iris Flower Classification
Industrial Training Kit | Batch 2026 | Powered by DecodeLabs

Interactive Streamlit Application featuring:
- Supervised Machine Learning Workflow (Load -> Explore -> Split -> Scale -> Train -> Predict -> Evaluate)
- Live Flower Measurement Predictor with confidence breakdown & neighbor inspection
- Comprehensive Exploratory Data Analysis & Feature Explanations
- K-Nearest Neighbors Hyperparameter Tuning Curve (Elbow Plot)
- Held-Out Test Set Evaluation & Interactive Confusion Matrix
- The "99% Accuracy Mirage" Interactive Deep-Dive
- Methodology & Theoretical Foundations (IPO framework, Heuristic vs ML, Precision/Recall trade-offs)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from model import (
    load_iris_data,
    split_data,
    create_pipeline,
    tune_k_cross_validation,
    train_final_model,
    evaluate_model,
    predict_flower,
    save_pipeline,
    load_pipeline,
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    SPECIES_METADATA,
    RANDOM_STATE,
    DEFAULT_K
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Iris AI Classifier | DecodeLabs Project 2",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global Font & Header adjustments */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .badge-kit {
        background-color: #1e293b;
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #0284c7;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #0f172a;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 14px;
        border: 2px solid #3b82f6;
        background: linear-gradient(145deg, #0f172a 0%, #1e1b4b 100%);
        margin-bottom: 1.5rem;
    }
    .step-box {
        background: #1e293b;
        color: #f8fafc;
        border-left: 4px solid #38bdf8;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .mirage-box {
        background: #450a0a;
        border: 1px solid #dc2626;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA & MODEL CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_data():
    X, y, target_names, feature_names, full_df = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=RANDOM_STATE)
    return X, y, target_names, feature_names, full_df, X_train, X_test, y_train, y_test

@st.cache_resource
def get_trained_pipeline(k_neighbors=DEFAULT_K):
    _, _, _, _, _, X_train, _, y_train, _ = get_cached_data()
    pipeline = train_final_model(X_train, y_train, n_neighbors=k_neighbors)
    return pipeline

@st.cache_data
def get_cv_tuning_data():
    _, _, _, _, _, X_train, _, y_train, _ = get_cached_data()
    cv_df, best_k, best_score = tune_k_cross_validation(X_train, y_train, k_range=range(1, 26), cv_folds=5)
    return cv_df, best_k, best_score


# Load cached items
X, y, target_names, feature_names, full_df, X_train, X_test, y_train, y_test = get_cached_data()
cv_df, best_k, best_score = get_cv_tuning_data()
pipeline_k5 = get_trained_pipeline(DEFAULT_K)
pipeline_tuned = get_trained_pipeline(best_k)

eval_results_k5 = evaluate_model(pipeline_k5, X_test, y_test, target_names)
eval_results_tuned = evaluate_model(pipeline_tuned, X_test, y_test, target_names)


# -----------------------------------------------------------------------------
# HEADER & SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.markdown('<span class="badge-kit">🛡 INDUSTRIAL TRAINING KIT | BATCH 2026</span>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Data Classification Using AI — Iris Flower Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">DecodeLabs Project 2 &bull; Supervised Learning Milestone &bull; K-Nearest Neighbors Pipeline</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo-without-subtitle.svg", width=140)
    st.markdown("### 🌸 Navigation")
    section = st.radio(
        "Select Module:",
        [
            "🔮 Live Flower Predictor",
            "📊 Exploratory Data Analysis",
            "⚙️ ML Workflow & K-Tuning",
            "🎯 Evaluation & Metrics",
            "⚠️ The '99% Accuracy Mirage'",
            "📖 Methodology & Architecture"
        ]
    )
    st.markdown("---")
    st.markdown("#### ⚙️ Active Inference Model")
    active_choice = st.selectbox(
        "Inference Model:",
        [f"Baseline Model (K={DEFAULT_K})", f"CV-Tuned Model (K={best_k})"],
        index=0,
        help="Select which trained pipeline is used for live predictions."
    )
    active_pipeline = pipeline_k5 if "Baseline" in active_choice else pipeline_tuned
    active_k = DEFAULT_K if "Baseline" in active_choice else best_k
    
    st.info(
        f"**Algorithm:** K-Nearest Neighbors\n\n"
        f"**Active K:** {active_k}\n\n"
        f"**Preprocessing:** StandardScaler (Leakage-free)\n\n"
        f"**Split:** 80% Train ({len(X_train)}), 20% Test ({len(X_test)})\n\n"
        f"**Reproducibility Seed:** {RANDOM_STATE}"
    )
    st.markdown("---")
    st.caption("Powered by **DecodeLabs** | Greater Lucknow, India")


# =============================================================================
# MODULE 1: LIVE FLOWER PREDICTOR
# =============================================================================
if section == "🔮 Live Flower Predictor":
    st.header(f"🔮 Interactive Flower Measurement Classifier ({active_choice.split()[0]} K={active_k})")
    st.markdown(
        f"Enter sepal and petal dimensions below to classify an unknown Iris flower using the **{active_choice}**. "
        f"The model scales inputs using parameters learned **only** from training history "
        f"and projects the sample into 4D Euclidean space to locate its **{active_k} nearest neighbors**."
    )
    
    col_input, col_pred = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("1. Flower Measurements (cm)")
        
        # Preset buttons for quick demonstrations
        st.markdown("**Quick Presets:**")
        preset_cols = st.columns(3)
        preset_choice = None
        if preset_cols[0].button("🌸 Setosa Archetype"):
            preset_choice = [5.0, 3.5, 1.4, 0.2]
        if preset_cols[1].button("🌿 Versicolor Archetype"):
            preset_choice = [5.9, 2.8, 4.3, 1.3]
        if preset_cols[2].button("🌺 Virginica Archetype"):
            preset_choice = [6.7, 3.1, 5.6, 2.2]
            
        def_vals = preset_choice if preset_choice else [5.8, 3.0, 4.0, 1.3]
        
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=float(def_vals[0]),
            step=0.1,
            help=FEATURE_DESCRIPTIONS["sepal length (cm)"]
        )
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=2.0,
            max_value=4.5,
            value=float(def_vals[1]),
            step=0.1,
            help=FEATURE_DESCRIPTIONS["sepal width (cm)"]
        )
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=float(def_vals[2]),
            step=0.1,
            help=FEATURE_DESCRIPTIONS["petal length (cm)"]
        )
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=2.6,
            value=float(def_vals[3]),
            step=0.1,
            help=FEATURE_DESCRIPTIONS["petal width (cm)"]
        )
        
        input_data = [sepal_length, sepal_width, petal_length, petal_width]
        
    with col_pred:
        st.subheader("2. AI Prediction & Rationale")
        prediction_result = predict_flower(active_pipeline, input_data, target_names)
        
        predicted_species = prediction_result["predicted_species"]
        conf = prediction_result["confidence"]
        meta = SPECIES_METADATA[predicted_species]
        
        st.markdown(f"""
        <div class="prediction-card">
            <span style="font-size: 0.85rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Species</span>
            <div style="font-size: 2.2rem; font-weight: 800; color: {meta['badge_color']};">
                {meta['display_name']}
            </div>
            <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
                Confidence: <strong style="color: #f8fafc;">{conf * 100:.1f}%</strong> 
                <span style="color: #38bdf8; font-size: 0.88rem;">&bull; KNN Neighbour-Vote Proportion: <strong>{prediction_result['vote_count']}/{prediction_result['total_k']}</strong></span>
            </div>
            <div style="margin-top: 12px; font-size: 0.92rem; color: #e2e8f0; line-height: 1.4;">
                {prediction_result['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilities Distribution Bar Chart
        st.markdown("**Class Probability Breakdown:**")
        prob_df = pd.DataFrame({
            "Species": [SPECIES_METADATA[s]["display_name"] for s in target_names],
            "Probability": [prediction_result["probabilities"][s] for s in target_names]
        })
        
        fig, ax = plt.subplots(figsize=(6, 2.2))
        colors = [SPECIES_METADATA[s]["badge_color"] for s in target_names]
        bars = ax.barh(prob_df["Species"], prob_df["Probability"], color=colors, height=0.55)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Predicted Probability")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%",
                    va='center', ha='left', fontsize=9, fontweight='bold', color='#e2e8f0')
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        ax.tick_params(colors='#cbd5e1')
        ax.xaxis.label.set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("3. Feature Comparison: User Input vs Species Averages")
    
    # Calculate species average measurements
    species_means = full_df.groupby("species")[FEATURE_NAMES].mean()
    comp_df = species_means.copy()
    comp_df.loc["👉 Your Sample"] = input_data
    
    fig_comp, ax_comp = plt.subplots(figsize=(10, 3.5))
    x_indices = np.arange(len(FEATURE_NAMES))
    width = 0.2
    
    palette = ["#22c55e", "#3b82f6", "#8b5cf6", "#f59e0b"]
    labels = ["Iris setosa avg", "Iris versicolor avg", "Iris virginica avg", "Your Flower Input"]
    
    for i, (idx_label, color) in enumerate(zip(["setosa", "versicolor", "virginica", "👉 Your Sample"], palette)):
        vals = comp_df.loc[idx_label].values
        ax_comp.bar(x_indices + (i - 1.5) * width, vals, width, label=labels[i], color=color)
        
    ax_comp.set_xticks(x_indices)
    ax_comp.set_xticklabels([f.replace(" (cm)", "") for f in FEATURE_NAMES], fontsize=10, color="#f8fafc")
    ax_comp.set_ylabel("Centimeters (cm)", color="#cbd5e1")
    ax_comp.set_title("Morphological Comparison with Benchmark Archetypes", color="#f8fafc", fontweight="bold")
    ax_comp.legend(loc="upper left")
    ax_comp.grid(axis="y", alpha=0.2, linestyle="--")
    fig_comp.patch.set_facecolor('#0f172a')
    ax_comp.set_facecolor('#0f172a')
    ax_comp.tick_params(colors='#cbd5e1')
    ax_comp.spines['bottom'].set_color('#334155')
    ax_comp.spines['left'].set_color('#334155')
    ax_comp.spines['top'].set_visible(False)
    ax_comp.spines['right'].set_visible(False)
    st.pyplot(fig_comp)


# =============================================================================
# MODULE 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
elif section == "📊 Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis: The Iris Benchmark")
    st.markdown(
        "The Iris dataset is the golden benchmark in supervised classification. "
        "It consists of **150 samples** equally distributed across **3 species classes** "
        "(50 samples each) measured across **4 continuous morphological dimensions**."
    )
    
    # Dataset Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown('<div class="metric-card"><div class="metric-val">150</div><div class="metric-lbl">Total Samples</div></div>', unsafe_allow_html=True)
    m2.markdown('<div class="metric-card"><div class="metric-val">4</div><div class="metric-lbl">Features (Dimensions)</div></div>', unsafe_allow_html=True)
    m3.markdown('<div class="metric-card"><div class="metric-val">3</div><div class="metric-lbl">Species Classes</div></div>', unsafe_allow_html=True)
    m4.markdown('<div class="metric-card"><div class="metric-val">50 : 50 : 50</div><div class="metric-lbl">Balanced Distribution</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("1. Detailed Feature Explanations")
    f_cols = st.columns(4)
    for i, col_name in enumerate(FEATURE_NAMES):
        with f_cols[i]:
            st.markdown(f"**🌿 `{col_name}`**")
            st.caption(FEATURE_DESCRIPTIONS[col_name])
            st.write(f"- **Mean:** `{full_df[col_name].mean():.2f} cm`")
            st.write(f"- **Range:** `[{full_df[col_name].min():.1f} - {full_df[col_name].max():.1f} cm]`")
            st.write(f"- **Std Dev:** `± {full_df[col_name].std():.2f}`")

    st.markdown("---")
    st.subheader("2. Interactive 2D Feature Space Scatter Plot")
    st.markdown("Explore how pairs of features separate the three species classes:")
    
    col_feat1, col_feat2 = st.columns(2)
    feat_x = col_feat1.selectbox("Select X-Axis Feature:", FEATURE_NAMES, index=2)  # Petal length default
    feat_y = col_feat2.selectbox("Select Y-Axis Feature:", FEATURE_NAMES, index=3)  # Petal width default
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 4.5))
    species_colors = {"setosa": "#22c55e", "versicolor": "#3b82f6", "virginica": "#8b5cf6"}
    
    for sp_name, grp in full_df.groupby("species"):
        ax_scatter.scatter(
            grp[feat_x],
            grp[feat_y],
            label=f"Iris {sp_name}",
            color=species_colors[sp_name],
            alpha=0.85,
            edgecolors="w",
            s=65
        )
        
    ax_scatter.set_xlabel(feat_x, color="#cbd5e1", fontsize=11)
    ax_scatter.set_ylabel(feat_y, color="#cbd5e1", fontsize=11)
    ax_scatter.set_title(f"Class Separation: {feat_x} vs {feat_y}", color="#f8fafc", fontweight="bold")
    ax_scatter.legend(loc="upper left")
    ax_scatter.grid(True, alpha=0.2, linestyle="--")
    fig_scatter.patch.set_facecolor('#0f172a')
    ax_scatter.set_facecolor('#0f172a')
    ax_scatter.tick_params(colors='#cbd5e1')
    ax_scatter.spines['bottom'].set_color('#334155')
    ax_scatter.spines['left'].set_color('#334155')
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)
    st.pyplot(fig_scatter)
    
    st.markdown("---")
    st.subheader("3. Feature Correlation Matrix")
    corr = full_df[FEATURE_NAMES].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(7, 3.8))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=1,
        ax=ax_corr,
        cbar_kws={'label': 'Pearson Correlation'}
    )
    ax_corr.set_title("Correlation Heatmap Across Iris Dimensions", color="#f8fafc", fontweight="bold")
    fig_corr.patch.set_facecolor('#0f172a')
    ax_corr.set_facecolor('#0f172a')
    ax_corr.tick_params(colors='#cbd5e1')
    st.pyplot(fig_corr)
    st.caption("Notice the very high correlation (0.96) between Petal Length and Petal Width, making them primary discriminators.")


# =============================================================================
# MODULE 3: ML WORKFLOW & K-TUNING
# =============================================================================
elif section == "⚙️ ML Workflow & K-Tuning":
    st.header("⚙️ Machine Learning Pipeline & Hyperparameter Tuning")
    st.markdown(
        "Supervised learning requires rigorous discipline. In this module we illustrate "
        "the 7-step engineering architecture and demonstrate why **K must be tuned strictly "
        "on training cross-validation**, never on the held-out test set."
    )
    
    st.subheader("1. The 7-Stage Machine Learning Pipeline")
    steps = [
        ("1. Load", "Load benchmark Iris dataset (150 samples, 4 continuous features, 3 species)."),
        ("2. Explore", "Verify class balance (50 each), feature distributions, and morphological characteristics."),
        ("3. Split", "Shuffle & partition into 80% Train (120 samples) and 20% Test (30 samples) with Stratification."),
        ("4. Scale", "Fit StandardScaler on training data ONLY; transform test data to eliminate data leakage."),
        ("5. Train", "Fit KNeighborsClassifier (Baseline K=5) and explore K values using 5-fold cross-validation."),
        ("6. Predict", "Apply learned Euclidean decision boundaries to infer species for unknown flower measurements."),
        ("7. Evaluate", "Measure held-out generalization performance via Confusion Matrix, Precision, Recall, and F1.")
    ]
    for name, desc in steps:
        st.markdown(f'<div class="step-box"><strong>{name}:</strong> {desc}</div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("2. Tuning the Engine: Choosing 'K' via Cross-Validation")
    st.markdown("""
    - **Small K (e.g., K=1):** Sensitive to noise, jagged decision boundaries, prone to **Overfitting** (high variance).
    - **Large K (e.g., K=25+):** Blurs localized boundaries, dominated by general population density, prone to **Underfitting** (high bias).
    - **The Elbow (Optimal K):** The sweet spot where training cross-validation error rate is minimized.
    """)
    
    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 4.5))
    ax_elbow.plot(cv_df["k"], cv_df["error_rate"], marker='o', color='#38bdf8', linewidth=2.2, label="CV Error Rate (1 - Accuracy)")
    ax_elbow.axvline(best_k, color="#22c55e", linestyle="--", linewidth=1.8, label=f"Optimal K = {best_k} (Min Error = {1-best_score:.4f})")
    ax_elbow.axvline(DEFAULT_K, color="#f59e0b", linestyle=":", linewidth=1.8, label=f"Baseline K = {DEFAULT_K}")
    
    ax_elbow.set_xlabel("Value of K (Number of Nearest Neighbors)", color="#cbd5e1", fontsize=11)
    ax_elbow.set_ylabel("Cross-Validation Error Rate", color="#cbd5e1", fontsize=11)
    ax_elbow.set_title("K-Hyperparameter Tuning Curve (5-Fold Stratified CV on Training Data)", color="#f8fafc", fontweight="bold")
    ax_elbow.set_xticks(range(1, 26))
    ax_elbow.grid(True, alpha=0.2, linestyle="--")
    ax_elbow.legend(loc="upper right")
    fig_elbow.patch.set_facecolor('#0f172a')
    ax_elbow.set_facecolor('#0f172a')
    ax_elbow.tick_params(colors='#cbd5e1')
    ax_elbow.spines['bottom'].set_color('#334155')
    ax_elbow.spines['left'].set_color('#334155')
    ax_elbow.spines['top'].set_visible(False)
    ax_elbow.spines['right'].set_visible(False)
    st.pyplot(fig_elbow)
    
    st.markdown("---")
    st.subheader("3. Interactive K-Value Simulator")
    selected_k = st.slider("Select candidate K to evaluate on training folds:", min_value=1, max_value=25, value=5, step=1)
    k_row = cv_df[cv_df["k"] == selected_k].iloc[0]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected K", f"{int(k_row['k'])}")
    c2.metric("Mean CV Accuracy", f"{k_row['mean_cv_accuracy']*100:.2f}%")
    c3.metric("CV Error Rate", f"{k_row['error_rate']:.4f}")


# =============================================================================
# MODULE 4: EVALUATION & METRICS
# =============================================================================
elif section == "🎯 Evaluation & Metrics":
    st.header("🎯 Held-Out Test Set Evaluation")
    st.markdown(
        "To rigorously validate model generalization, evaluation is performed "
        "strictly on the **held-out test set of 30 unseen samples** (10 per class) "
        "that was locked away prior to training."
    )
    
    eval_mode = st.radio(
        "Select Model to Inspect on Locked Test Set:",
        [f"🔹 Baseline Model (K={DEFAULT_K})", f"🔸 CV-Tuned Model (K={best_k})", "⚖️ Direct Comparison (Baseline vs Tuned)"],
        horizontal=True
    )
    
    if "Direct Comparison" in eval_mode:
        st.subheader("Model Comparison on Untouched Test Set (30 Samples)")
        comp_metrics_df = pd.DataFrame([
            {
                "Model": f"Baseline Model (K={DEFAULT_K})",
                "Accuracy": f"{eval_results_k5['accuracy']*100:.2f}%",
                "Macro Precision": f"{eval_results_k5['precision_macro']:.4f}",
                "Macro Recall": f"{eval_results_k5['recall_macro']:.4f}",
                "Macro F1-Score": f"{eval_results_k5['f1_macro']:.4f}",
                "Weighted F1": f"{eval_results_k5['f1_weighted']:.4f}",
                "Hyperparameter Origin": "Assignment Baseline Mandate"
            },
            {
                "Model": f"CV-Tuned Model (K={best_k})",
                "Accuracy": f"{eval_results_tuned['accuracy']*100:.2f}%",
                "Macro Precision": f"{eval_results_tuned['precision_macro']:.4f}",
                "Macro Recall": f"{eval_results_tuned['recall_macro']:.4f}",
                "Macro F1-Score": f"{eval_results_tuned['f1_macro']:.4f}",
                "Weighted F1": f"{eval_results_tuned['f1_weighted']:.4f}",
                "Hyperparameter Origin": "5-Fold CV on Training Set (Lowest Error)"
            }
        ])
        st.dataframe(comp_metrics_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.markdown(f"**Baseline Confusion Matrix (K={DEFAULT_K}):**")
            fig_cm1, ax_cm1 = plt.subplots(figsize=(5, 3.8))
            sns.heatmap(
                eval_results_k5["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=[name.capitalize() for name in target_names],
                yticklabels=[name.capitalize() for name in target_names],
                cbar=False,
                ax=ax_cm1,
                annot_kws={"size": 13, "weight": "bold"}
            )
            ax_cm1.set_xlabel("Predicted", color="#cbd5e1")
            ax_cm1.set_ylabel("Actual", color="#cbd5e1")
            fig_cm1.patch.set_facecolor('#0f172a')
            ax_cm1.set_facecolor('#0f172a')
            ax_cm1.tick_params(colors='#cbd5e1')
            st.pyplot(fig_cm1)
            
        with col_cm2:
            st.markdown(f"**CV-Tuned Confusion Matrix (K={best_k}):**")
            fig_cm2, ax_cm2 = plt.subplots(figsize=(5, 3.8))
            sns.heatmap(
                eval_results_tuned["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Purples",
                xticklabels=[name.capitalize() for name in target_names],
                yticklabels=[name.capitalize() for name in target_names],
                cbar=False,
                ax=ax_cm2,
                annot_kws={"size": 13, "weight": "bold"}
            )
            ax_cm2.set_xlabel("Predicted", color="#cbd5e1")
            ax_cm2.set_ylabel("Actual", color="#cbd5e1")
            fig_cm2.patch.set_facecolor('#0f172a')
            ax_cm2.set_facecolor('#0f172a')
            ax_cm2.tick_params(colors='#cbd5e1')
            st.pyplot(fig_cm2)
            
    else:
        current_eval = eval_results_k5 if "Baseline" in eval_mode else eval_results_tuned
        active_k_name = f"Baseline K={DEFAULT_K}" if "Baseline" in eval_mode else f"CV-Tuned K={best_k}"
        
        # High-level metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-val">{current_eval["accuracy"]*100:.1f}%</div><div class="metric-lbl">Overall Accuracy ({active_k_name})</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-val">{current_eval["precision_macro"]:.4f}</div><div class="metric-lbl">Macro Precision</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-val">{current_eval["recall_macro"]:.4f}</div><div class="metric-lbl">Macro Recall</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-val">{current_eval["f1_macro"]:.4f}</div><div class="metric-lbl">Macro F1-Score</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        col_cm, col_diag = st.columns([1, 1], gap="large")
        
        with col_cm:
            st.subheader(f"Confusion Matrix ({active_k_name})")
            cm = current_eval["confusion_matrix"]
            
            fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
            cmap_choice = "Blues" if "Baseline" in eval_mode else "Purples"
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap=cmap_choice,
                xticklabels=[name.capitalize() for name in target_names],
                yticklabels=[name.capitalize() for name in target_names],
                cbar=False,
                ax=ax_cm,
                annot_kws={"size": 14, "weight": "bold"}
            )
            ax_cm.set_xlabel("Predicted Species", color="#cbd5e1", fontsize=11)
            ax_cm.set_ylabel("Actual Species", color="#cbd5e1", fontsize=11)
            ax_cm.set_title(f"Confusion Matrix ({active_k_name})", color="#f8fafc", fontweight="bold")
            fig_cm.patch.set_facecolor('#0f172a')
            ax_cm.set_facecolor('#0f172a')
            ax_cm.tick_params(colors='#cbd5e1')
            st.pyplot(fig_cm)
            
        with col_diag:
            st.subheader("Per-Class Diagnostics")
            st.markdown("""
            - **TP (True Positive):** Correctly classified as the target species.
            - **FP (False Positive / Type I Error):** False Alarm (other species misclassified as this).
            - **FN (False Negative / Type II Error):** Missed Detection (this species misclassified as another).
            - **TN (True Negative):** Correctly rejected non-members.
            """)
            
            diag_rows = []
            for name, d in current_eval["per_class_diagnostics"].items():
                diag_rows.append({
                    "Species": name.capitalize(),
                    "TP": d["tp"],
                    "FP (Type I)": d["fp"],
                    "FN (Type II)": d["fn"],
                    "TN": d["tn"],
                    "Precision": f"{d['precision']:.4f}",
                    "Recall": f"{d['recall']:.4f}",
                    "F1-Score": f"{d['f1']:.4f}"
                })
            st.dataframe(pd.DataFrame(diag_rows), hide_index=True, use_container_width=True)
            
        st.markdown("---")
        st.subheader(f"Detailed Classification Report ({active_k_name})")
        st.code(current_eval["classification_report_str"], language="text")


# =============================================================================
# MODULE 5: THE 99% ACCURACY MIRAGE
# =============================================================================
elif section == "⚠️ The '99% Accuracy Mirage'":
    st.header("⚠️ The '99% Accuracy Mirage' Deep-Dive")
    st.markdown(
        "Requirement 8 from DecodeLabs Project 2: **Explain why accuracy alone may not tell the whole story.** "
        "In balanced datasets like Iris, accuracy appears intuitive. However, in real-world scenarios, "
        "accuracy can be completely misleading."
    )
    
    st.markdown("""
    <div class="mirage-box">
        <h4 style="color: #f87171; margin-top: 0;">🚨 Why High Accuracy Can Be a Lie</h4>
        <p style="color: #fecaca; margin-bottom: 0;">
            Consider a cancer screening test with 1,000 patients where 990 are healthy (Negative)
            and 10 have cancer (Positive). A trivial model that simply predicts <em>'Healthy' for every single person</em>
            achieves <strong>99.0% Accuracy</strong>, but its <strong>Recall for detecting cancer is 0.0%</strong>!
            Every cancer patient is missed, creating a catastrophic failure masked by a seemingly near-perfect accuracy score.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Interactive Imbalance Simulator")
    st.markdown("Experiment with class imbalance below to observe how Accuracy decouples from Precision and Recall:")
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        total_pop = st.number_input("Total Patient Population:", min_value=100, max_value=10000, value=1000, step=100)
        cancer_rate = st.slider("Actual Disease Incidence Rate (%):", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
        
    num_positive = int(total_pop * (cancer_rate / 100.0))
    num_negative = total_pop - num_positive
    
    # Scenario A: Trivial Dummy Classifier (always predicts Negative)
    acc_dummy = (num_negative) / total_pop
    rec_dummy = 0.0
    prec_dummy = 0.0
    f1_dummy = 0.0
    
    # Scenario B: Realistic Diagnostic AI Model
    # Detects 9 out of 10 positive cases, but produces 15 false alarms
    tp_ai = max(1, int(num_positive * 0.90))
    fn_ai = num_positive - tp_ai
    fp_ai = int(num_negative * 0.02)
    tn_ai = num_negative - fp_ai
    
    acc_ai = (tp_ai + tn_ai) / total_pop
    prec_ai = tp_ai / (tp_ai + fp_ai)
    rec_ai = tp_ai / (tp_ai + fn_ai)
    f1_ai = 2 * (prec_ai * rec_ai) / (prec_ai + rec_ai)
    
    with col_sim2:
        st.markdown(f"**Cohort Breakdown ({total_pop} individuals):**")
        st.write(f"- 🔴 Positive (Has Condition): **{num_positive} patients**")
        st.write(f"- 🟢 Negative (Healthy): **{num_negative} patients**")
        
    st.markdown("#### Performance Comparison: Naive Dummy vs Diagnostic AI")
    sim_data = [
        {
            "Model": "🚫 Naive Model (Always Predicts Healthy)",
            "Accuracy": f"{acc_dummy*100:.2f}%",
            "Precision": "0.00%",
            "Recall (Sensitivity)": "0.00% ⚠️ (ALL MISSED)",
            "F1-Score": "0.000"
        },
        {
            "Model": "🩺 Trained Diagnostic Classifier",
            "Accuracy": f"{acc_ai*100:.2f}%",
            "Precision": f"{prec_ai*100:.2f}%",
            "Recall (Sensitivity)": f"{rec_ai*100:.2f}% ✅",
            "F1-Score": f"{f1_ai:.3f}"
        }
    ]
    st.dataframe(pd.DataFrame(sim_data), hide_index=True, use_container_width=True)
    
    st.info(
        "💡 **Conclusion:** This is why DecodeLabs engineers use **Precision**, **Recall**, "
        "**F1-Score (Harmonic Mean)**, and the **Confusion Matrix** to ensure models deliver "
        "trustworthy, clinical-grade decision-making."
    )


# =============================================================================
# MODULE 6: METHODOLOGY & ARCHITECTURE
# =============================================================================
elif section == "📖 Methodology & Architecture":
    st.header("📖 Methodology & Architectural Paradigms")
    st.markdown(
        "A formal review of the concepts covered in DecodeLabs Project 2, bridging "
        "the gap between heuristic rules and algorithmic machine learning."
    )
    
    # 1. Old Way vs New Way
    st.subheader("1. The Logic Skeleton: Heuristic vs Supervised Learning")
    col_old, col_new = st.columns(2)
    with col_old:
        st.markdown("""
        **Old Way: Heuristics & Rules (Project 1)**
        - Hardcoded nested `if / else` rules.
        - Brittle; quickly breaks as feature dimensions increase.
        - Cannot generalize to novel or ambiguous edge cases.
        - Human programmer defines the explicit logic.
        """)
    with col_new:
        st.markdown("""
        **New Way: Supervised Learning (Project 2)**
        - *'We do not write the rules. We provide history, and the machine derives the logic.'*
        - Learns mathematical decision boundaries in $n$-dimensional space.
        - Adapts seamlessly to multi-variable interactions.
        - Rigorous statistical verification via cross-validation.
        """)
        
    st.markdown("---")
    # 2. Master Blueprint: IPO Framework
    st.subheader("2. The Master Blueprint: IPO Framework")
    st.markdown("""
    | Stage | Phase | Components in Project 2 |
    | :--- | :--- | :--- |
    | **INPUT** | Raw Material & Preprocessing | Iris Benchmark (150 samples), Feature Scaling (StandardScaler) |
    | **PROCESS** | Logic & Optimization | Stratified 80/20 Split, K-Nearest Neighbors (KNN Baseline K=5) |
    | **OUTPUT** | Verification & Deployment | Confusion Matrix, Macro Precision, Recall, F1 Harmonic Mean |
    """)
    
    st.markdown("---")
    # 3. StandardScaler & Leakage
    st.subheader("3. The Gatekeeper Rule: Feature Scaling & Data Leakage Prevention")
    st.markdown(r"""
    The K-Nearest Neighbors algorithm relies on calculating distances in multidimensional space:
    $$d(p, q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$$
    
    If one feature ranges from $0$ to $1000$ and another ranges from $0$ to $1$, the large feature 
    completely dominates the distance calculation.
    
    **StandardScaler Formula:**
    $$z = \frac{x - \mu}{\sigma} \quad \text{such that } \text{Mean } \mu = 0, \text{ Variance } \sigma^2 = 1$$
    
    **Data Leakage Rule:**
    The scaler parameters ($\mu, \sigma$) must be calculated **exclusively from the training set**. 
    Applying scaler fitting across the whole dataset before splitting causes information from 
    the test set to leak into training, artificially inflating performance estimates.
    """)
    
    st.markdown("---")
    # 4. Strategic Trade-offs
    st.subheader("4. Strategic Trade-offs: Precision vs Recall")
    st.markdown(r"""
    - **Precision ($\frac{TP}{TP + FP}$):** *Trustworthiness*. Crucial when False Positives are costly (e.g. email spam filters — never send an important email to spam).
    - **Recall ($\frac{TP}{TP + FN}$):** *Sensitivity*. Crucial when False Negatives are catastrophic (e.g. medical diagnosis or airport security — never miss a dangerous condition).
    - **F1-Score ($2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$):** The harmonic mean providing a balanced single metric when class distributions vary.
    """)
    
    st.markdown("---")
    # 5. Emerging Horizons
    st.subheader("5. Emerging Horizons: Tabular Data to Computer Vision")
    st.markdown("""
    While Project 2 utilizes 4 tabular measurements, the identical supervised learning principles 
    extend to **Computer Vision & Convolutional Neural Networks (CNNs)**:
    - Tabular columns $\to$ Image pixel matrices.
    - Euclidean Distance $\to$ Convolutional feature maps.
    - Botanical dimensions $\to$ Visual Iris petal segmentation.
    """)
