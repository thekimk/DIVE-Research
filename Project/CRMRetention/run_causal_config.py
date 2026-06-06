# [Research-Grade CausalForest Feature Ablation Estimator]
# Designed for process-isolated sequential execution to prevent memory OOM crashes.
# Fits a CausalForest (GRF) for a specified feature configuration and saves results to disk.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from econml.grf import CausalForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import os
import argparse

# =============================================================================
# 1. SETUP COMMAND LINE ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description="Process-isolated CausalForest estimator.")
parser.add_argument(
    "--config", 
    type=str, 
    required=True, 
    choices=["v0", "v0_v1", "v0_v2", "v0_v3", "v0_v4", "v0_v1_v2_v3_v4"],
    help="Feature configuration to train."
)
args = parser.parse_args()

SEED = 42
np.random.seed(SEED)
N_ESTIMATORS = 2000

DATA_PATH = "data/featured_causal_balanced(1:10).parquet"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "featured_causal_balanced(1:10).parquet"

OUTPUT_DIR = "results/comparison_ab"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

# Helper function to enforce clean white background and gridlines for academic papers
def apply_academic_style(fig, ax):
    sns.set_theme(style="whitegrid")
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(True, color='#E5E5E5', linestyle='-', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')

# =============================================================================
# 2. FEATURE GROUP DEFINITIONS
# =============================================================================
cols_v0 = [
    'avg_campaign_duration', 'avg_time_since_complaint', 'avg_time_since_first_purchase',
    'avg_time_since_last_click', 'avg_time_since_last_open', 'avg_time_since_unsubscribe',
    'camp_campaign_typebulk', 'camp_campaign_typetransactional', 'camp_campaign_typetrigger',
    'camp_channelemail', 'camp_channelmobile_push', 'camp_channelmultichannel', 'camp_channelsms',
    'camp_topicevent', 'camp_topichappy.birthday', 'camp_topicleave.review',
    'camp_topicoffer.after.purchase', 'camp_topicother', 'camp_topicsale.out',
    'channel_email', 'channel_mobile_push', 'channel_web_push',
    'email_provider_gmail.com', 'email_provider_mail.ru', 'email_provider_other',
    'is_holiday',
    'message_type_bulk', 'message_type_transactional', 'message_type_trigger',
    'platform.', 'platform.desktop', 'platform.phablet', 'platform.smartphone', 'platform.tablet',
    'prev_is_clicked', 'prev_is_complained', 'prev_is_opened', 'prev_is_unsubscribed',
    'total_campaigns', 'total_messages', 'total_purchases'
]

cols_v1 = ['days_since_last_purchase', 'feat_rtb_hazard', 'feat_postbuy_refrac']
cols_v2 = ['cal_is_weekend', 'cal_week_of_month', 'feat_dow_shift', 'feat_eoq_bump', 'feat_hour_shift', 'feat_payday_bump']
cols_v3 = ['ctx_tc_open_rate_30d', 'feat_fatigue', 'feat_last_any_hours', 'feat_last_email_hours', 'feat_last_mobile_push_hours', 'u_cadence_std_30d', 'u_click_rate_30d', 'u_open_cnt_30d', 'u_open_rate_30d']
cols_v4 = ['feat_like_last_success', 'feat_path_align', 'feat_topic_novelty', 'topic_N7', 'topic_t_since_hours']

# Select feature list based on config choice
config_mapping = {
    "v0": cols_v0,
    "v0_v1": cols_v0 + cols_v1,
    "v0_v2": cols_v0 + cols_v2,
    "v0_v3": cols_v0 + cols_v3,
    "v0_v4": cols_v0 + cols_v4,
    "v0_v1_v2_v3_v4": cols_v0 + cols_v1 + cols_v2 + cols_v3 + cols_v4
}
feature_list = config_mapping[args.config]

# Clean data across all features combined to ensure consistent test set alignment
all_possible_features = list(set(cols_v0 + cols_v1 + cols_v2 + cols_v3 + cols_v4))
valid_all_possible = [c for c in all_possible_features if c in df.columns]

target_col = 'is_purchased' 
treatment_col = 'ab_test'
df[treatment_col] = df[treatment_col].fillna(0).astype(int)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=valid_all_possible + [treatment_col, target_col])

# Train/Test split (kept naturally unbalanced as per instructions)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

T_train = df_train[treatment_col].values
T_test = df_test[treatment_col].values
Y_train = df_train[target_col].values
Y_test = df_test[target_col].values

# Save target variables once (from the v0 execution) to align evaluation
if args.config == "v0":
    np.save(f"{OUTPUT_DIR}/y_test.npy", Y_test)
    np.save(f"{OUTPUT_DIR}/t_test.npy", T_test)

# =============================================================================
# 3. FIT ESTIMATOR & SAVE CATE PREDICTIONS
# =============================================================================
valid_features = [c for c in feature_list if c in df.columns]
X_train = df_train[valid_features].values
X_test = df_test[valid_features].values

print(f"\n[*] Training configuration: {args.config}")
print(f"[*] Features count: {len(valid_features)}")
print(f"[*] Fitting CausalForest (GRF) on {X_train.shape[0]:,} rows...")

est = CausalForest(
    n_estimators=N_ESTIMATORS,
    criterion='het',
    min_samples_leaf=10,
    min_samples_split=20,
    honest=True,
    inference=False,
    random_state=SEED
)
est.fit(X_train, T_train, Y_train)
cate_pred = est.predict(X_test).flatten()

# Save CATE predictions to disk
np.save(f"{OUTPUT_DIR}/pred_{args.config}.npy", cate_pred)
print(f"[OK] CATE predictions saved to: {OUTPUT_DIR}/pred_{args.config}.npy")

# =============================================================================
# 4. INDIVIDUAL DIAGNOSTIC VISUALIZATIONS
# =============================================================================
sns.set_theme(style="whitegrid")

# A. Positivity (Overlap) Plot
try:
    propensity_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    propensity_model.fit(X_train, T_train)
    estimated_propensity = propensity_model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(x=estimated_propensity, hue=T_test, bins=30, kde=True, palette="Set1", alpha=0.6, ax=ax)
    apply_academic_style(fig, ax)
    plt.title(f"Positivity Overlap Check ({args.config})", fontsize=14, fontweight='bold')
    plt.xlabel("Estimated Propensity Score P(T=1|X)")
    plt.ylabel("Count")
    plt.savefig(f"{OUTPUT_DIR}/causal_forest_grf_overlap_{args.config}.png", dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"[!] Overlap plot failed: {e}")

# B. Top 10 Feature Importance Plot
if hasattr(est, 'feature_importances_'):
    try:
        importances = est.feature_importances_() if callable(est.feature_importances_) else est.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:10]
        top_importances = importances[sorted_indices]
        top_features = [valid_features[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=top_importances, y=top_features, hue=top_features, palette="viridis", legend=False, ax=ax)
        apply_academic_style(fig, ax)
        plt.title(f"Top 10 Feature Importance ({args.config})", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Score", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/causal_forest_grf_importance_{args.config}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[!] Feature importance plot failed: {e}")

# C. SHAP Summary Plot (via Surrogate Model)
try:
    surrogate = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=SEED)
    surrogate.fit(X_test, cate_pred)
    
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X_test[:500])
    
    fig, ax = plt.subplots(figsize=(8, 10))
    shap.summary_plot(shap_values, X_test[:500], feature_names=valid_features, show=False)
    apply_academic_style(fig, plt.gca())
    plt.title(f"SHAP Summary Plot for Predicted CATE (Surrogate - {args.config})", fontsize=12, fontweight='bold')
    plt.savefig(f"{OUTPUT_DIR}/causal_forest_grf_shap_{args.config}.png", dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"[!] SHAP visualization failed: {e}")

# D. Individual Uplift Curves (Gain, Qini, Lift)
try:
    from causalml.metrics import plot_gain, plot_qini, plot_lift
    indiv_df = pd.DataFrame({
        'y': Y_test.flatten().astype(float),
        't': T_test.flatten().astype(int),
        args.config: cate_pred
    })
    
    # Gain Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_gain(indiv_df, outcome_col='y', treatment_col='t')
    apply_academic_style(fig, plt.gca())
    plt.title(f"Cumulative Gain Curve ({args.config})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/causal_forest_grf_gain_{args.config}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Qini Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_qini(indiv_df, outcome_col='y', treatment_col='t')
    apply_academic_style(fig, plt.gca())
    plt.title(f"Qini Curve ({args.config})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/causal_forest_grf_qini_{args.config}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Lift Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_lift(indiv_df, outcome_col='y', treatment_col='t')
    apply_academic_style(fig, plt.gca())
    plt.title(f"Cumulative Lift Curve ({args.config})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/causal_forest_grf_lift_{args.config}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Individual curves and diagnostic plots saved successfully.")
except Exception as e:
    print(f"[!] Curves generation failed: {e}")
