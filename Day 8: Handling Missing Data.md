# Handling Missing Data

Time series often have missing observations: maybe a sensor went offline for a day, or sales weren’t recorded on a holiday, etc. Missing data can be a big issue, because many models (ARIMA, for example) can’t handle gaps naturally, and even those that can (like state space models using Kalman filter – discussed later) still require careful handling. Here’s how to tackle missing data:

- **Forward fill (carry last observation forward):**  
  This replaces missing values with the last known value. In pandas, `df.fillna(method='ffill')` does this. It’s simple and keeps the series continuity. It makes sense when you believe the value didn’t change much during the gap or if the gap is short. For example, if a daily stock price is missing on a holiday, you might carry forward the last price (since the market was closed, the price didn’t actually change). However, forward fill can introduce bias if used indiscriminately (it creates a “flat” period that might not be real). It’s best when missing data is truly an absence of change or you explicitly want to assume “no news means nothing changed.”

- **Backward fill:**  
  Opposite of forward fill – use the next valid observation to fill backward. For instance, if you have a gap and you know the next measured value, you assume the missing period was at that next value early. This is less commonly used alone, but sometimes combined (like forward then backward) if missing in middle. Typically, forward fill is more common because in many scenarios it’s logical that a process holds its last value until new data comes. But backward fill might be used in some contexts (say you have end-of-month balances missing some mid-month, maybe you backfill with next known if you assume a quick drop happened on the last missing day).

- **Linear interpolation:**  
  Draw a straight line between the last known value before the gap and the first known value after the gap. This assumes a linear change in between. This can be reasonable for something that likely changed smoothly. E.g., if temperature data missing for 2 hours, linear interpolation between 1 PM and 4 PM values will give a plausible gradual change at 2 PM, 3 PM. It avoids sudden jumps that forward/backward fill produce. Most libraries offer this: `df.interpolate(method='linear')`. This works well if the gap is not too long and you expect no sudden shocks in between. It won’t capture nonlinear patterns though (if actual changed in a curve, linear is an approximation).

- **Spline or higher-order interpolation:**  
  This fits a polynomial or spline curve through the known points to fill the gap. A spline can capture some curvature (e.g. maybe a daily cycle shape missing a segment). It can produce more realistic intermediate values if the series is smooth. `df.interpolate(method='spline', order=3)` for example can fit a cubic spline. Just be cautious: polynomials can overshoot or oscillate. But splines (piecewise polynomials) are usually more stable.

- **Seasonal interpolation:**  
  If data is seasonal and missing an entire period, sometimes you might use last year’s same period value as an estimate (if that pattern is stable). E.g., missing one week of sales, maybe use last year’s same week as a placeholder (scaled perhaps by overall growth). That’s more ad-hoc but in some domains they do that.

- **Kalman filter or model-based imputation:**  
  State space models (like those underlying ARIMA and exponential smoothing) can handle missing data via Kalman filtering, which essentially does an optimal interpolation based on the model dynamics. For example, statsmodels’ SARIMAX can handle missing values in the input by automatically using the Kalman filter to estimate them (by treating them as unobserved state). Similarly, prophet can handle missing days – it basically just doesn’t penalize anything for those days and will still produce a forecast (since it operates on a continuous time model). Kalman smoothing essentially gives a statistically optimal fill-in (like if the model expects a certain trend and noise level, it will infer the likely missing value). This is great if you trust a model structure; it avoids manual filling. There’s also the imputeTS package in R (and similar logic in Python) which offers Kalman smoothing for time series imputation.

- **Interpolation by regression or ML:**  
  Sometimes you can predict missing values using other correlated series. E.g., if one sensor is out, maybe use readings from a nearby sensor to estimate it (through a regression). Or use a ML model trained to impute one series from others. That’s advanced but can be effective if relationships exist.

- **Dropping missing periods:**  
  If the missingness is extensive or the period is irrecoverable, sometimes you exclude those periods from analysis entirely. E.g., if one month of data is completely missing and you can’t impute confidently, you might not use that month for model training or treat it as a special case (maybe treat that month like a holiday effect or something). However, most models require regular frequency, so you’d have to fill the gap in timeline – maybe with NaNs that model can skip if it supports, or fill with something neutral (like average) but perhaps downweight that period in error.

- **Frequency re-sampling:**  
  If you have irregular timestamps (not missing per se, but not evenly spaced), you often resample to a regular frequency and impute missing slots as needed. For example, you have transactions at random times – you could resample to daily counts, and days with no transactions become zeros (which is a form of missing turned to 0 which in this context is correct: missing because there were none). Or if you have a time series with slight timing jitter, you might align to a grid and fill missing with interpolation.

- **Holidays/zero-demand days:**  
  Sometimes missing isn’t random – e.g., a store closed on Sunday means sales = 0, not really missing. So distinguish between “no observation because there was nothing” and “no observation but there was something”. If missing means “should be 0” (like no stock trading on weekend -> volume is 0), you fill with 0, not actually missing.

- **Advanced: Multiple Imputation:**  
  In some statistical contexts, rather than filling with one value, they do multiple imputation (generate several possible fills by sampling noise). But that’s more for inference problems. For forecasting, usually a single imputation strategy is fine.

- **Evaluate impact of missing data:**  
  It’s good to check how sensitive your model is if you forward-fill vs interpolate. E.g., an ARIMA model might treat forward-filled flat segments as real data, which could bias trend estimation (if a big drop happened in missing interval, forward-fill hides it). Interpolation would distribute that drop. In critical cases, you might even scenario test “if missing was high vs low” to see impact.

- **Documentation and flags:**  
  If you fill data, it can be helpful to keep a mask of what was filled to possibly treat those differently. For example, you might not want to count imputed values when calculating certain stats.

---

### In Python, example using pandas:
```python
ts = ts.asfreq('D')  # set daily frequency, introduce NaNs for missing days
ts_ffill = ts.fillna(method='ffill')
ts_linear = ts.interpolate(method='linear')
```

And for a more advanced approach, one could use statsmodels.tsa.statespace.tools.missing_data.KalmanFilter or simply fit a SARIMAX with missing and it will handle internally.

**Recap:**

Handle missing data before modeling (except models that can incorporate it like Prophet or certain Kalman approaches). Forward-fill is simple and often okay, linear/spline for more smooth assumptions. Use domain knowledge: e.g., if missing because machine was off, maybe the value truly was zero during off state – fill with zero. Always be mindful that imputation is an assumption – try to test if different methods yield different model results. Missing data imputation is part of the data cleaning that can significantly affect your forecasts, so treat it seriously.
