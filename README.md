# Brighton Renewable Energy Forecasting

**24-hour-ahead probabilistic forecasting of solar irradiance and wind speed for Brighton, UK**, benchmarked across 6 models on 14 years of hourly weather data.

Rebuild of a university project with proper methodology: real baselines, engineered features, chronological splits, calibrated uncertainty, and business translation grounded in real Brighton renewable assets.

---

## Headline results

Evaluated on 17,328 hours of held-out test data (2022-2023):

| Target | Model | MAE | R² | vs Climatology baseline |
|---|---|---|---|---|
| **Solar** (MJ/m²) | Climatology | 0.216 | 0.71 | — |
| **Solar** | LightGBM | **0.176** | **0.77** | **18% MAE reduction** |
| **Solar** | LSTM (P50) | 0.175 | 0.77 | 19% MAE reduction |
| **Wind** (km/h) | Climatology | 6.62 | 0.09 | — |
| **Wind** | LightGBM | 6.22 | 0.20 | 6% MAE reduction |
| **Wind** | LSTM (P50) | **6.06** | **0.22** | **8% MAE reduction** |

**Probabilistic forecasts:** LSTM quantile outputs produced calibrated 80% prediction intervals with empirical coverage of **85.8% (solar)** and **78.8% (wind)** on test data.

---

## Business translation

Forecasts were applied to Brighton Energy Coop's real 3 MW rooftop portfolio ([62 sites, 20-year PPAs](https://www.thenews.coop/brighton-energy-co-op-adds-three-sites-to-its-portfolio/)) using the UK MCS standard conversion factor (0.85 kWh per kWp per MJ/m²):

- **Actual 2022-2023 generation:** 19,572 MWh
- **Forecast generation:** 20,689 MWh
- **Aggregate error:** 5.7%
- **Estimated annual imbalance exposure** at £120/MWh UK System Buy Price ([Elexon BMRS](https://bmrs.elexon.co.uk/)): ~£467,000 ± £140,000

> ⚠️ Rampion Offshore Wind (400 MW, 116 turbines 13 km off Brighton) was considered for wind translation but excluded from final results: Brighton ground-level wind is a poor proxy for offshore wind at 80 m hub height. Operational wind forecasting requires offshore buoy measurements or ECMWF numerical weather prediction data.

---

## Why this project exists

This is a rebuild of my MSc Data Science module project (University of Essex, 2024). The original got 93% as a grade but had several methodological flaws I've since learned to spot:

| Original | Rebuilt |
|---|---|
| ARMA(3,4) baseline (cannot represent 24h seasonality) | SARIMAX(1,0,1)(1,0,1,24) + 4 naive baselines + climatology |
| Univariate LSTM (past solar → future solar only) | Multivariate LSTM with 36 features including weather |
| 1 year test set (2023) | 2 year test set (2022-2023) |
| No uncertainty estimates | Calibrated P10-P90 prediction intervals |
| Business impact: fabricated 168,000 m² panel array × guessed tariff = £92k | Real Brighton Energy Coop 3 MW portfolio × MCS conversion × Elexon reference prices |
| Cubic polynomial interpolation (overshoot risk) | Linear (time-weighted) interpolation |
| Negative R² on wind baseline (model worse than mean) | Positive R² across all ML models |

The rebuild taught me more than the original grade did.

---

## Approach

### Data
- **Source:** Visual Crossing Weather API, Brighton UK, 2010-01-01 to 2024-01-06
- **Resolution:** hourly, 122,844 rows × 16 columns
- **Target variables:** `solarenergy` (MJ/m²), `windspeed` (km/h)
- **Noted anomalies:** 28 duplicate timestamps from DST fall-back hours (resolved), 14 missing timestamps from DST spring-forward hours (interpolated)

### Split (chronological, NEVER random for time series)
- Train: 2010-2020 (11 years, ~96k hours)
- Validation: 2021 (for tuning)
- Test: 2022-2023 (touched exactly once, at the end)

### Feature engineering (36 features)
- Lag features: 1h, 24h, 168h (weekly) for solar, wind, cloudcover, temp, humidity, pressure
- Rolling 24h and 168h means
- Fourier time encoding: sin/cos of hour-of-day and day-of-year (captures cyclical patterns without discontinuity at midnight/year-end)
- Wind direction as sin/cos pair (circular encoding)
- Dropped as leakage: `solarradiation` and `uvindex` (0.999 correlation with solar target)

### Model ladder
1. **Persistence** (tomorrow = today) — trivial baseline
2. **Seasonal naive 24h** (tomorrow = yesterday same hour) — captures daily cycle
3. **Weekly naive 168h** — captures weekly cycle
4. **Climatology** (mean by hour-of-year) — strongest classical baseline
5. **SARIMAX**(1,0,1)(1,0,1,24) — proper seasonal time-series model
6. **LightGBM** with all 36 features — typically wins on tabular time series
7. **Multivariate LSTM** with quantile loss for P10/P50/P90 — gives probabilistic output

### Evaluation
- MAE, RMSE, R² on validation set for model selection
- Final metrics computed once on test set (2022-2023)
- Coverage check for prediction intervals: does the 80% interval actually contain 80% of observations?

---

## Key findings

1. **Solar is highly predictable, wind is fundamentally hard.** Solar's diurnal and annual cycles are nearly deterministic (sun position), leaving cloud cover as the main noise source. Wind is dominated by synoptic-scale weather with weak cyclical structure. Every model confirms this: solar R² ≈ 0.77, wind R² ≈ 0.22.

2. **Climatology is a shockingly hard baseline to beat.** Predicting "the long-term average for this hour-of-year" already captures the dominant signal. ML gains come from weather corrections on top, not from replacing climatology.

3. **Feature importance confirms physical intuition.** Solar is dominated by `solarenergy_lag24` (same hour yesterday) with humidity and pressure as secondary features. Wind has no single dominant feature and relies on combinations of recent observations, pressure, and seasonal timing.

4. **LightGBM matches or beats LSTM on point forecasts** for tabular time series with engineered features. The LSTM's value is its probabilistic output (prediction intervals), not point accuracy.

5. **Calibrated uncertainty is more useful than a perfect point forecast.** On cloudy summer days the point forecast misses sharp drops, but the P10-P90 interval still contains the actual observation. That's what decision-makers need.

---

## Repository structure


**Planned v3 refactor:** split into proper `src/` modules with tests, CI, a FastAPI serving layer, and Streamlit dashboard.

---

## Stack

Python 3.11 · pandas · NumPy · scikit-learn · LightGBM ·  LSTM · TensorFlow/Keras · statsmodels · matplotlib · seaborn

Ran on Google Colab with T4 GPU for LSTM training. LightGBM and SARIMAX train on CPU in under a minute each.

---

## Data sources and references

- Brighton Energy Coop portfolio: [Co-operative News, 2023](https://www.thenews.coop/brighton-energy-co-op-adds-three-sites-to-its-portfolio/)
- Rampion Offshore Wind Farm specs: [RWE UK](https://uk.rwe.com/locations/rampion-offshore-wind-farm/)
- UK solar PV conversion factor: [Microgeneration Certification Scheme (MCS)](https://mcscertified.com/)
- Imbalance pricing reference: [Elexon BMRS](https://bmrs.elexon.co.uk/)

---

## Author

**Shoaib Junaeid** ([GitHub](https://github.com/Junaeid-Shoaib))
MSc Artificial Intelligence, University of Essex (Distinction, 2024)


Open to data science and ML engineering roles with a focus on energy, forecasting, or LLM applications.

---

## Licence

MIT
