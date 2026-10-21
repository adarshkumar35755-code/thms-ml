# Dataset

The training dataset `thms_dataset.csv` is not included in this repository.
To train or to run the data-dependent tests, place it here as
`data/thms_dataset.csv`, or set the environment variable `THMS_DATA_PATH` to its location.

Expected format: 94,080 rows = 70 runs x 1,344 samples (14 days at 15-minute spacing).

| Group | Columns |
|---|---|
| Meta (never model inputs) | `run_id`, `split` (train / val / test, assigned per run), `t_hours`, `hour_of_day` |
| Sensor inputs | `input_current_A`, `output_current_A`, `output_voltage_V`, `ambient_temp_C`, `humidity_pct`, `vibration_rms_ms2`, `h2_ppm`, `oil_temp_C` |
| Labels (never model inputs) | `fault_type`, `health_state`, `health_state_name`, `health_score` |

`src/data.py` validates all of this on load and refuses to run if the schema, value
ranges or run-level split are wrong.
