# =============================================================
#   ECO-SCORE CAMPUS SUSTAINABILITY — ML MODEL
#   Model 1 : Gradient Boosting Regressor  → Forecast energy_kwh
#   Model 2 : Gradient Boosting Classifier → Detect Anomalies
#   Dataset : campus_utility_full.csv
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────
print("=" * 60)
print("  ECO-SCORE ML MODEL TRAINING")
print("=" * 60)

print("\n[1/7] Loading dataset...")
df = pd.read_csv(os.path.join(OUTPUT_DIR, "campus_utility_full.csv"))
print(f"      Rows    : {len(df):,}")
print(f"      Columns : {df.columns.tolist()}")

# ─────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────
print("\n[2/7] Engineering features...")

# Extract time features from date and time columns
df["datetime"]    = pd.to_datetime(df["date"] + " " + df["time"])
df["hour"]        = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek   # 0=Monday, 6=Sunday
df["month_num"]   = df["datetime"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

# Convert is_anomaly to int (True→1, False→0)
df["is_anomaly"] = df["is_anomaly"].astype(str).map({"True": 1, "False": 0})

# Encode categorical columns
le_building  = LabelEncoder()
le_day_type  = LabelEncoder()

df["building_encoded"] = le_building.fit_transform(df["building_id"])
df["day_type_encoded"] = le_day_type.fit_transform(df["day_type"])

print("      New features added: hour, day_of_week, month_num,")
print("                          is_weekend, building_encoded, day_type_encoded")

# ─────────────────────────────────────────
# STEP 3 — DEFINE FEATURES AND TARGETS
# ─────────────────────────────────────────
print("\n[3/7] Defining features and targets...")

# Features used for training (same for both models)
FEATURES = [
    "hour",
    "day_of_week",
    "month_num",
    "is_weekend",
    "building_encoded",
    "day_type_encoded",
    "temperature_c",
    "occupancy",
    "peak_demand_kw",
    "water_liters"
]

TARGET_REGRESSION    = "energy_kwh"     # Model 1 — What value to predict
TARGET_CLASSIFICATION = "is_anomaly"    # Model 2 — Anomaly yes/no

X = df[FEATURES]
y_reg  = df[TARGET_REGRESSION]
y_clf  = df[TARGET_CLASSIFICATION]

print(f"      Features  : {FEATURES}")
print(f"      Target 1  : {TARGET_REGRESSION}")
print(f"      Target 2  : {TARGET_CLASSIFICATION}")

# ─────────────────────────────────────────
# STEP 4 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────
print("\n[4/7] Splitting data into train and test sets (80% / 20%)...")

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

print(f"      Training rows : {len(X_train):,}")
print(f"      Testing rows  : {len(X_test):,}")

# ─────────────────────────────────────────
# STEP 5 — TRAIN MODEL 1: REGRESSOR
# ─────────────────────────────────────────
print("\n[5/7] Training Model 1 — Gradient Boosting Regressor (Energy Forecast)...")

reg_model = GradientBoostingRegressor(
    n_estimators    = 200,    # Number of trees
    learning_rate   = 0.1,    # How much each tree contributes
    max_depth       = 5,      # Depth of each tree
    min_samples_split = 10,
    subsample       = 0.8,    # Use 80% of data per tree (prevents overfitting)
    random_state    = 42
)

reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

# Evaluation Metrics
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
mae  = mean_absolute_error(y_reg_test, y_reg_pred)
r2   = r2_score(y_reg_test, y_reg_pred)

print(f"\n      ── REGRESSOR RESULTS ──────────────────")
print(f"      R² Score  : {r2:.4f}  (closer to 1.0 = better)")
print(f"      RMSE      : {rmse:.4f} kWh (avg prediction error)")
print(f"      MAE       : {mae:.4f} kWh (mean absolute error)")
print(f"      ────────────────────────────────────────")

# ─────────────────────────────────────────
# STEP 5B — TRAIN MODEL 2: CLASSIFIER
# ─────────────────────────────────────────
print("\n[5b] Training Model 2 — Gradient Boosting Classifier (Anomaly Detection)...")

clf_model = GradientBoostingClassifier(
    n_estimators  = 200,
    learning_rate = 0.1,
    max_depth     = 4,
    subsample     = 0.8,
    random_state  = 42
)

clf_model.fit(X_train, y_clf_train)
y_clf_pred = clf_model.predict(X_test)

acc  = accuracy_score(y_clf_test, y_clf_pred)
prec = precision_score(y_clf_test, y_clf_pred, zero_division=0)
rec  = recall_score(y_clf_test, y_clf_pred, zero_division=0)
f1   = f1_score(y_clf_test, y_clf_pred, zero_division=0)

print(f"\n      ── CLASSIFIER RESULTS ─────────────────")
print(f"      Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"      Precision : {prec:.4f}")
print(f"      Recall    : {rec:.4f}")
print(f"      F1 Score  : {f1:.4f}")
print(f"      ────────────────────────────────────────")
print(f"\n      Classification Report:")
print(classification_report(y_clf_test, y_clf_pred,
                             target_names=["Normal", "Anomaly"]))

# ─────────────────────────────────────────
# STEP 6 — VISUALIZATIONS
# ─────────────────────────────────────────
print("\n[6/7] Generating plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Eco-Score Campus — ML Model Results", fontsize=16, fontweight="bold")

# ── Plot 1: Actual vs Predicted Energy ──
ax1 = axes[0, 0]
sample = min(500, len(y_reg_test))
ax1.scatter(y_reg_test[:sample], y_reg_pred[:sample],
            alpha=0.4, color="#2196F3", s=15)
max_val = max(y_reg_test.max(), y_reg_pred.max())
ax1.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction")
ax1.set_xlabel("Actual Energy (kWh)")
ax1.set_ylabel("Predicted Energy (kWh)")
ax1.set_title(f"Actual vs Predicted Energy\nR² = {r2:.4f}")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Plot 2: Residuals Distribution ──
ax2 = axes[0, 1]
residuals = y_reg_test - y_reg_pred
ax2.hist(residuals, bins=50, color="#4CAF50", edgecolor="white", alpha=0.8)
ax2.axvline(x=0, color="red", linestyle="--", linewidth=2)
ax2.set_xlabel("Residual (Actual - Predicted)")
ax2.set_ylabel("Frequency")
ax2.set_title("Residuals Distribution\n(Centered near 0 = Good)")
ax2.grid(True, alpha=0.3)

# ── Plot 3: Feature Importance (Regressor) ──
ax3 = axes[0, 2]
feat_importance = pd.Series(reg_model.feature_importances_, index=FEATURES)
feat_importance = feat_importance.sort_values(ascending=True)
colors = ["#FF5722" if v > 0.15 else "#2196F3" for v in feat_importance]
feat_importance.plot(kind="barh", ax=ax3, color=colors)
ax3.set_title("Feature Importance\n(Regressor)")
ax3.set_xlabel("Importance Score")
ax3.grid(True, alpha=0.3)

# ── Plot 4: Confusion Matrix (Classifier) ──
ax4 = axes[1, 0]
cm = confusion_matrix(y_clf_test, y_clf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4,
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"])
ax4.set_title(f"Confusion Matrix\nAccuracy = {acc*100:.2f}%")
ax4.set_ylabel("Actual")
ax4.set_xlabel("Predicted")

# ── Plot 5: Avg Energy Per Building ──
ax5 = axes[1, 1]
building_energy = df.groupby("building_id")["energy_kwh"].mean().sort_values(ascending=False)
building_energy.plot(kind="bar", ax=ax5, color="#9C27B0", edgecolor="white")
ax5.set_title("Average Energy Usage\nper Building")
ax5.set_xlabel("Building ID")
ax5.set_ylabel("Avg Energy (kWh)")
ax5.tick_params(axis="x", rotation=45)
ax5.grid(True, alpha=0.3, axis="y")

# ── Plot 6: Hourly Avg Energy Pattern ──
ax6 = axes[1, 2]
hourly = df.groupby("hour")["energy_kwh"].mean()
ax6.plot(hourly.index, hourly.values, color="#FF9800",
         linewidth=2.5, marker="o", markersize=4)
ax6.fill_between(hourly.index, hourly.values, alpha=0.2, color="#FF9800")
ax6.set_title("Average Energy by Hour of Day\n(All Buildings)")
ax6.set_xlabel("Hour")
ax6.set_ylabel("Avg Energy (kWh)")
ax6.set_xticks(range(0, 24, 2))
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"      Saved: model_results.png")

# ─────────────────────────────────────────
# STEP 6B — ECO SCORE VISUALIZATION
# ─────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Eco-Score Dashboard Preview", fontsize=14, fontweight="bold")

# Eco Score per building
BENCHMARK_ENERGY = {"B001":1200,"B002":600,"B003":800,"B004":750,
                    "B005":400,"B006":900,"B007":500,"B008":1000}
BENCHMARK_WATER  = {"B001":6000,"B002":2000,"B003":8000,"B004":7500,
                    "B005":1500,"B006":10000,"B007":5000,"B008":3000}

daily = df.groupby(["building_id","date"]).agg(
    total_energy=("energy_kwh","sum"),
    total_water=("water_liters","sum")
).reset_index()

daily["benchmark_e"] = daily["building_id"].map(BENCHMARK_ENERGY)
daily["benchmark_w"] = daily["building_id"].map(BENCHMARK_WATER)

daily["energy_score"] = (100 - ((daily["total_energy"] / daily["benchmark_e"]) * 100)).clip(0, 100)
daily["water_score"]  = (100 - ((daily["total_water"]  / daily["benchmark_w"])  * 100)).clip(0, 100)
daily["eco_score"]    = ((daily["energy_score"] + daily["water_score"]) / 2).round(2)

avg_eco = daily.groupby("building_id")["eco_score"].mean().sort_values(ascending=False)

colors_eco = ["#4CAF50" if v >= 50 else "#FF5722" for v in avg_eco.values]
avg_eco.plot(kind="bar", ax=axes2[0], color=colors_eco, edgecolor="white")
axes2[0].set_title("Average Eco-Score per Building\n(Green ≥ 50, Red < 50)")
axes2[0].set_xlabel("Building ID")
axes2[0].set_ylabel("Eco-Score (0–100)")
axes2[0].axhline(y=50, color="orange", linestyle="--", linewidth=2, label="Benchmark")
axes2[0].legend()
axes2[0].tick_params(axis="x", rotation=45)
axes2[0].grid(True, alpha=0.3, axis="y")

# Monthly energy trend (all buildings avg)
monthly = df.groupby("month_num")["energy_kwh"].mean()
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
axes2[1].plot(monthly.index, monthly.values,
              color="#2196F3", linewidth=2.5, marker="s", markersize=6)
axes2[1].fill_between(monthly.index, monthly.values, alpha=0.2, color="#2196F3")
axes2[1].set_xticks(range(1,13))
axes2[1].set_xticklabels(month_labels, rotation=45)
axes2[1].set_title("Monthly Energy Trend\n(Average All Buildings)")
axes2[1].set_ylabel("Avg Energy (kWh)")
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eco_score_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"      Saved: eco_score_dashboard.png")

# ─────────────────────────────────────────
# STEP 7 — SAVE MODELS + PREDICTION DEMO
# ─────────────────────────────────────────
print("\n[7/7] Saving trained models...")

joblib.dump(reg_model, f"{OUTPUT_DIR}/energy_forecast_model.pkl")
joblib.dump(clf_model, f"{OUTPUT_DIR}/anomaly_detector_model.pkl")
joblib.dump(le_building, f"{OUTPUT_DIR}/building_encoder.pkl")
joblib.dump(le_day_type, f"{OUTPUT_DIR}/day_type_encoder.pkl")

print(f"      Saved: energy_forecast_model.pkl")
print(f"      Saved: anomaly_detector_model.pkl")
print(f"      Saved: building_encoder.pkl")
print(f"      Saved: day_type_encoder.pkl")

# ─────────────────────────────────────────
# PREDICTION UTILITY FUNCTION
# ─────────────────────────────────────────

def predict_energy(building_id, hour, day_type, month_num,
                   temperature_c, occupancy, peak_demand_kw, water_liters):
    """
    Predict energy usage and check for anomaly for a new reading.
    """
    day_of_week = 0 if day_type == "Weekday" else 5
    is_weekend  = 1 if day_type == "Weekend" else 0

    b_enc = le_building.transform([building_id])[0]
    d_enc = le_day_type.transform([day_type])[0]

    input_data = pd.DataFrame([{
        "hour"             : hour,
        "day_of_week"      : day_of_week,
        "month_num"        : month_num,
        "is_weekend"       : is_weekend,
        "building_encoded" : b_enc,
        "day_type_encoded" : d_enc,
        "temperature_c"    : temperature_c,
        "occupancy"        : occupancy,
        "peak_demand_kw"   : peak_demand_kw,
        "water_liters"     : water_liters
    }])

    predicted_energy = reg_model.predict(input_data)[0]
    is_anomaly       = clf_model.predict(input_data)[0]
    anomaly_prob     = clf_model.predict_proba(input_data)[0][1]

    return {
        "building_id"      : building_id,
        "predicted_energy" : round(predicted_energy, 2),
        "is_anomaly"       : bool(is_anomaly),
        "anomaly_prob_%"   : round(anomaly_prob * 100, 2)
    }

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING COMPLETE — SUMMARY")
print("=" * 60)
print(f"\n  REGRESSOR  (Energy Forecast)")
print(f"    R² Score : {r2:.4f}")
print(f"    RMSE     : {rmse:.4f} kWh")
print(f"    MAE      : {mae:.4f} kWh")
print(f"\n  CLASSIFIER (Anomaly Detection)")
print(f"    Accuracy : {acc*100:.2f}%")
print(f"    F1 Score : {f1:.4f}")
print(f"\n  OUTPUT FILES")
print(f"    energy_forecast_model.pkl  → Use to predict energy")
print(f"    anomaly_detector_model.pkl → Use to detect anomalies")
print(f"    model_results.png          → Training graphs")
print(f"    eco_score_dashboard.png    → Eco score charts")
print("=" * 60)
