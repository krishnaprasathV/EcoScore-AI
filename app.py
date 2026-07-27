# =============================================================
#   ECO-SCORE CAMPUS SUSTAINABILITY — FLASK WEB PORTAL
#   Run: python app.py
#   Open: http://127.0.0.1:5000
# =============================================================

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = "ecoscore_secret_2024"

# ─────────────────────────────────────────
# LOAD MODELS & ENCODERS
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

reg_model   = joblib.load(os.path.join(BASE_DIR, "energy_forecast_model.pkl"))
clf_model   = joblib.load(os.path.join(BASE_DIR, "anomaly_detector_model.pkl"))
le_building = joblib.load(os.path.join(BASE_DIR, "building_encoder.pkl"))
le_day_type = joblib.load(os.path.join(BASE_DIR, "day_type_encoder.pkl"))

# ─────────────────────────────────────────
# LOAD DATASET FOR DASHBOARD
# ─────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE_DIR, "campus_utility_full.csv")).sample(3000)
df["datetime"]    = pd.to_datetime(df["date"] + " " + df["time"])
df["hour"]        = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month_num"]   = df["datetime"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["is_anomaly"]  = df["is_anomaly"].astype(str).map({"True": 1, "False": 0})

BUILDINGS = {
    "B001": "Main Academic Block",
    "B002": "Library",
    "B003": "Hostel Block A",
    "B004": "Hostel Block B",
    "B005": "Admin Block",
    "B006": "Sports Complex",
    "B007": "Canteen",
    "B008": "Lab Block"
}

BENCHMARK_ENERGY = {"B001":1200,"B002":600,"B003":800,"B004":750,
                    "B005":400,"B006":900,"B007":500,"B008":1000}
BENCHMARK_WATER  = {"B001":6000,"B002":2000,"B003":8000,"B004":7500,
                    "B005":1500,"B006":10000,"B007":5000,"B008":3000}

def compute_eco_scores():
    daily = df.groupby(["building_id","date"]).agg(
        total_energy=("energy_kwh","sum"),
        total_water=("water_liters","sum")
    ).reset_index()
    daily["benchmark_e"] = daily["building_id"].map(BENCHMARK_ENERGY)
    daily["benchmark_w"] = daily["building_id"].map(BENCHMARK_WATER)
    daily["energy_score"] = (100 - ((daily["total_energy"] / daily["benchmark_e"]) * 100)).clip(0, 100)
    daily["water_score"]  = (100 - ((daily["total_water"]  / daily["benchmark_w"])  * 100)).clip(0, 100)
    daily["eco_score"]    = ((daily["energy_score"] + daily["water_score"]) / 2).round(2)
    return daily

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

# HOME — DASHBOARD
@app.route("/")
def index():
    daily = compute_eco_scores()
    avg_eco = daily.groupby("building_id")["eco_score"].mean().round(2).to_dict()

    # Building cards data
    building_data = []
    for bid, bname in BUILDINGS.items():
        score = avg_eco.get(bid, 0)
        avg_energy = df[df["building_id"] == bid]["energy_kwh"].mean()
        avg_water  = df[df["building_id"] == bid]["water_liters"].mean()
        anomaly_count = int(df[df["building_id"] == bid]["is_anomaly"].sum())
        status = "Excellent" if score >= 70 else "Good" if score >= 50 else "Warning" if score >= 30 else "Critical"
        building_data.append({
            "id": bid,
            "name": bname,
            "score": round(score, 1),
            "avg_energy": round(avg_energy, 2),
            "avg_water": round(avg_water, 2),
            "anomalies": anomaly_count,
            "status": status
        })

    # Sort by eco score descending
    building_data.sort(key=lambda x: x["score"], reverse=True)

    # Monthly trend data for chart
    monthly = df.groupby("month_num")["energy_kwh"].mean().round(2)
    monthly_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_data   = [float(monthly.get(i, 0)) for i in range(1, 13)]

    # Overall stats
    total_anomalies = int(df["is_anomaly"].sum())
    avg_score_all   = round(sum(b["score"] for b in building_data) / len(building_data), 1)
    top_building    = building_data[0]["name"]
    worst_building  = building_data[-1]["name"]

    return render_template("index.html",
        buildings=building_data,
        buildings_map=json.dumps(BUILDINGS),
        buildings_map_dict=BUILDINGS,
        monthly_labels=json.dumps(monthly_labels),
        monthly_data=json.dumps(monthly_data),
        total_anomalies=total_anomalies,
        avg_score=avg_score_all,
        top_building=top_building,
        worst_building=worst_building
    )

# PREDICT PAGE
@app.route("/predict", methods=["GET","POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            building_id    = request.form["building_id"]
            hour           = int(request.form["hour"])
            day_type       = request.form["day_type"]
            month_num      = int(request.form["month_num"])
            temperature_c  = float(request.form["temperature_c"])
            occupancy      = int(request.form["occupancy"])
            peak_demand_kw = float(request.form["peak_demand_kw"])
            water_liters   = float(request.form["water_liters"])

            day_of_week = 0 if day_type == "Weekday" else 5
            is_weekend  = 1 if day_type == "Weekend" else 0

            b_enc = le_building.transform([building_id])[0]
            d_enc = le_day_type.transform([day_type])[0]

            input_data = pd.DataFrame([{
                "hour": hour, "day_of_week": day_of_week,
                "month_num": month_num, "is_weekend": is_weekend,
                "building_encoded": b_enc, "day_type_encoded": d_enc,
                "temperature_c": temperature_c, "occupancy": occupancy,
                "peak_demand_kw": peak_demand_kw, "water_liters": water_liters
            }])

            predicted_energy = round(float(reg_model.predict(input_data)[0]), 2)
            is_anomaly       = bool(clf_model.predict(input_data)[0])
            anomaly_prob     = round(float(clf_model.predict_proba(input_data)[0][1]) * 100, 2)

            result = {
                "building_id": building_id,
                "building_name": BUILDINGS[building_id],
                "predicted_energy": predicted_energy,
                "is_anomaly": is_anomaly,
                "anomaly_prob": anomaly_prob,
                "status": "ANOMALY DETECTED" if is_anomaly else "Normal Reading"
            }
        except Exception as e:
            flash(f"Error: {str(e)}", "error")

    return render_template("predict.html", buildings=BUILDINGS, result=result)

# ANOMALIES PAGE
@app.route("/anomalies")
def anomalies():
    anomaly_df = df[df["is_anomaly"] == 1].copy()
    anomaly_df = anomaly_df.sort_values("datetime", ascending=False).head(100)
    anomaly_df["building_name"] = anomaly_df["building_id"].map(BUILDINGS)

    records = anomaly_df[[
        "building_id","building_name","date","time",
        "energy_kwh","water_liters","temperature_c","occupancy"
    ]].to_dict("records")

    # Anomaly count per building for chart
    anomaly_counts = df[df["is_anomaly"]==1].groupby("building_id").size().to_dict()
    chart_labels = list(BUILDINGS.keys())
    chart_data   = [int(anomaly_counts.get(b, 0)) for b in chart_labels]

    return render_template("anomalies.html",
        records=records,
        total=len(anomaly_df),
        chart_labels=json.dumps(chart_labels),
        chart_data=json.dumps(chart_data)
    )

# UPLOAD PAGE
@app.route("/upload", methods=["GET","POST"])
def upload():
    preview = None
    stats   = None
    if request.method == "POST":
        file = request.files.get("csv_file")
        if file and file.filename.endswith(".csv"):
            try:
                uploaded_df = pd.read_csv(file)
                required_cols = ["building_id","date","time","energy_kwh","water_liters"]
                missing = [c for c in required_cols if c not in uploaded_df.columns]
                if missing:
                    flash(f"Missing columns: {missing}", "error")
                else:
                    stats = {
                        "rows": len(uploaded_df),
                        "buildings": uploaded_df["building_id"].nunique(),
                        "date_range": f"{uploaded_df['date'].min()} to {uploaded_df['date'].max()}",
                        "avg_energy": round(uploaded_df["energy_kwh"].mean(), 2),
                        "avg_water": round(uploaded_df["water_liters"].mean(), 2)
                    }
                    preview = uploaded_df.head(10).to_dict("records")
                    flash("File uploaded and validated successfully!", "success")
            except Exception as e:
                flash(f"Error reading file: {str(e)}", "error")
        else:
            flash("Please upload a valid CSV file.", "error")

    return render_template("upload.html", preview=preview, stats=stats)

# API ENDPOINT — for live data
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    try:
        b_enc = le_building.transform([data["building_id"]])[0]
        d_enc = le_day_type.transform([data["day_type"]])[0]
        input_data = pd.DataFrame([{
            "hour": data["hour"], "day_of_week": data.get("day_of_week", 0),
            "month_num": data["month_num"], "is_weekend": data.get("is_weekend", 0),
            "building_encoded": b_enc, "day_type_encoded": d_enc,
            "temperature_c": data["temperature_c"], "occupancy": data["occupancy"],
            "peak_demand_kw": data["peak_demand_kw"], "water_liters": data["water_liters"]
        }])
        predicted = round(float(reg_model.predict(input_data)[0]), 2)
        anomaly   = bool(clf_model.predict(input_data)[0])
        return jsonify({"predicted_energy": predicted, "is_anomaly": anomaly, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400

# API ENDPOINT — filtered dashboard data by selected buildings
@app.route("/api/dashboard", methods=["POST"])
def api_dashboard():
    data = request.get_json()
    selected = data.get("buildings", list(BUILDINGS.keys()))
    
    if not selected:
        selected = list(BUILDINGS.keys())
    
    daily = compute_eco_scores()
    filtered_daily = daily[daily["building_id"].isin(selected)]
    filtered_df = df[df["building_id"].isin(selected)]
    
    # Building eco-scores
    avg_eco = filtered_daily.groupby("building_id")["eco_score"].mean().round(2)
    
    building_data = []
    for bid in selected:
        if bid in BUILDINGS:
            bname = BUILDINGS[bid]
            score = float(avg_eco.get(bid, 0))
            bdf = filtered_df[filtered_df["building_id"] == bid]
            avg_energy = float(bdf["energy_kwh"].mean()) if len(bdf) > 0 else 0
            avg_water = float(bdf["water_liters"].mean()) if len(bdf) > 0 else 0
            anomaly_count = int(bdf["is_anomaly"].sum())
            status = "Excellent" if score >= 70 else "Good" if score >= 50 else "Warning" if score >= 30 else "Critical"
            building_data.append({
                "id": bid,
                "name": bname,
                "score": round(score, 1),
                "avg_energy": round(avg_energy, 2),
                "avg_water": round(avg_water, 2),
                "anomalies": anomaly_count,
                "status": status
            })
    
    building_data.sort(key=lambda x: x["score"], reverse=True)
    
    # Monthly trend — per building
    monthly_per_building = {}
    for bid in selected:
        if bid in BUILDINGS:
            bdf = filtered_df[filtered_df["building_id"] == bid]
            monthly = bdf.groupby("month_num")["energy_kwh"].mean().round(2)
            monthly_per_building[bid] = {
                "name": BUILDINGS[bid],
                "data": [float(monthly.get(i, 0)) for i in range(1, 13)]
            }
    
    # Monthly average across all selected
    monthly_avg = filtered_df.groupby("month_num")["energy_kwh"].mean().round(2)
    monthly_avg_data = [float(monthly_avg.get(i, 0)) for i in range(1, 13)]
    
    # Overall stats
    total_anomalies = int(filtered_df["is_anomaly"].sum())
    avg_score_all = round(sum(b["score"] for b in building_data) / max(len(building_data), 1), 1)
    top_building = building_data[0]["name"] if building_data else "N/A"
    worst_building = building_data[-1]["name"] if building_data else "N/A"
    
    return jsonify({
        "buildings": building_data,
        "monthly_labels": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "monthly_avg_data": monthly_avg_data,
        "monthly_per_building": monthly_per_building,
        "total_anomalies": total_anomalies,
        "avg_score": avg_score_all,
        "top_building": top_building,
        "worst_building": worst_building
    })

if __name__ == "__main__":
    print("=" * 50)
    print("  ECO-SCORE PORTAL RUNNING")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True)
