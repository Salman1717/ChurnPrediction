# utils/shap_utils.py
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def make_explainer(model):
    # For tree-based models, TreeExplainer is best
    return shap.TreeExplainer(model)

def global_summary_plot(shap_values, X_sample, feature_names, outpath: str = "models/shap_summary.png"):
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

def save_force_plot_html(explainer, shap_values_row, X_row, feature_names, outpath_html):
    """
    Save an interactive force plot as HTML (useful to open locally).
    """
    # create the force plot (JS) and save as html
    force = shap.force_plot(explainer.expected_value, shap_values_row, X_row, feature_names=feature_names, matplotlib=False)
    # save HTML
    shap.save_html(outpath_html, force)
    return outpath_html

def save_waterfall_png(explainer, shap_values_row, X_row, feature_names, outpath_png):
    """
    Save a waterfall (matplotlib) plot as PNG for a single observation.
    """
    plt.figure(figsize=(8,4))
    try:
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values_row, feature_names=feature_names, max_display=12)
    except Exception:
        # fallback: use waterfall from shap if API newer
        shap.plots.waterfall((shap_values_row), max_display=12)
    plt.tight_layout()
    plt.savefig(outpath_png, dpi=150)
    plt.close()
    return outpath_png

def batch_save_top_k_force_pngs(model, transformer, X_original_df, k=10, out_dir="models/forces"):
    """
    Picks top-k highest risk customers (by prob) and saves waterfall PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    Xp = X_original_df.copy()
    X_trans = transformer.transform(Xp)
    probs = model.predict_proba(X_trans)[:,1]
    idxs = np.argsort(-probs)[:k]  # top-k indices
    explainer = make_explainer(model)
    # convert to dense rows if sparse
    X_trans_dense = X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans
    for rank, i in enumerate(idxs, start=1):
        shap_vals = explainer.shap_values(X_trans_dense[i:i+1])[0]
        fname = os.path.join(out_dir, f"customer_rank_{rank}_idx_{i}.png")
        save_waterfall_png(explainer, shap_vals, X_trans_dense[i:i+1], transformer.get_feature_names_out(Xp.columns), fname)
    return out_dir
