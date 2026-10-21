from src.infer import HealthScorer

scorer = HealthScorer()                 # loads the model and metadata from config paths

packet = {"input_current_A": 3.1, "output_current_A": 82.0, "output_voltage_V": 414.0,
          "ambient_temp_C": 29.5, "humidity_pct": 64.0, "vibration_rms_ms2": 0.62,
          "h2_ppm": 34.0, "oil_temp_C": 52.0}        # one packet every 15 minutes

result = scorer.update(packet)
print(result["health_score"], result["state_name"], result["warming_up"])
for r in result["top_reasons"]:
    print(r["label"], round(r["impact_points"], 1), r["direction"])