# Evaluation Metrics

When we build a forecasting model, we need ways to quantify its accuracy (or error) on test data (or via cross-validation). There are numerous metrics, and the choice can depend on business context (is % error more important or absolute error?) and characteristics (is zero an important value? Are we aggregating across series?). Here are commonly used metrics:

---

### • MAE (Mean Absolute Error)
This is the average of absolute errors:  
$MAE = \frac{1}{n}\sum |y_t - \hat{y}_t|$  

It’s a straightforward measure in the same units as the data (e.g. if predicting sales units, MAE=5 means on average you’re off by 5 units). MAE is easy to interpret and is less sensitive to outliers than MSE because it doesn’t square errors. It corresponds to optimizing the median forecast (if you minimize MAE, your forecast tends to be median-unbiased). MAE is good when all errors are equally weighted linearly.

---

### • RMSE (Root Mean Square Error)
This is  
$RMSE = \sqrt{\frac{1}{n}\sum (y_t - \hat{y}_t)^2}$  

It’s the square root of MSE. RMSE penalizes large errors more strongly (squaring emphasizes outliers). It’s perhaps the most common metric in literature due to nice mathematical properties (MSE is differentiable, etc.) and it aligns with optimizing mean (least squares gives mean prediction). RMSE is in the same units as data, but because of squaring, an RMSE of (say) 10 can be more influenced by a few large errors than MAE of 10 would be. If your application really dislikes big errors, RMSE is useful (it’ll heavily punish those). But if outliers are maybe anomalies you don’t want to over-penalize, MAE might be better. RMSE is also convenient for normally-distributed error assumptions (one standard deviation if errors ~ normal).

---

### • MAPE (Mean Absolute Percentage Error)
$MAPE = \frac{100\%}{n}\sum \left|\frac{y_t - \hat{y}_t}{y_t}\right|$  

It expresses error as a percentage of the actual value, averaged. This is great for communicating relative error: e.g. “on average, our forecast is off by 8%”. MAPE is scale-independent, so you can compare performance across series of different scales with it. However, it has issues: if any actual $y_t$ is zero, you get a division by zero (so often we exclude those or add a small number). Also, it heavily penalizes under-forecasts vs over-forecasts when actual is small (e.g. actual 5, predicted 10: abs% error = 100%; actual 10, predicted 5: abs% error = 50% – same magnitude error 5 units, but different %). For very low-volume series, MAPE can blow up or be overly harsh. Despite drawbacks, MAPE is widely used in business because a percentage is intuitive. Just be careful if your data has zeros or near-zeros – one approach is to exclude or cap extremely high percentage errors (or use a different metric in those cases).

---

### • SMAPE (Symmetric MAPE)
This modifies MAPE to be symmetric:  
$SMAPE = \frac{100\%}{n}\sum \frac{|y_t - \hat{y}_t|}{(|y_t| + |\hat{y}_t|)/2}$  

It basically divides the error by the average of actual and forecast. SMAPE ensures the metric is bounded between 0% and 200% (if either forecast or actual is zero, the denominator is half of the other, yielding at most 200%). It tends to penalize over- and under-forecasts more evenly. SMAPE is used in some competitions as an objective. It still has the issue of being undefined if both forecast and actual are zero (0/0), but usually that’s resolved by defining 0/0=0 in that context. SMAPE is a bit less intuitive than MAPE but more stable when actual values can be very small.

---

### • WAPE (Weighted Absolute Percentage Error)
Or sometimes called MAD/Mean or just “Mean Absolute Scaled Error” in some contexts:  

WAPE is usually defined as  
$WAPE = \frac{\sum |y - \hat{y}|}{\sum |y|} \times 100\%$  

In other words, total absolute error divided by total actual quantity. It’s like a global MAPE (weighted by actual values). WAPE is useful for intermittent demand where computing an average of percentages might be dominated by tiny denominators. WAPE gives an overall indication like “we mis-forecasted X% of the total volume.” Some prefer WAPE because it’s simple and avoids the instability of MAPE on small values. If you see “normalized MAE” that could be similar concept (normalized by the mean of actuals or sum).

---

### • MSLE / RMSLE (Mean Squared Log Error)
This is useful when you care about relative errors and you don’t want to blow up on large absolute errors.  

$MSLE = \frac{1}{n}\sum (\ln(1+y_t) - \ln(1+\hat{y}_t))^2$  

By taking logs, it means if you under-forecast by a factor of 2 or over-forecast by factor 2, it’s the same error in log terms. RMSLE (root of MSLE) is also common. This metric is good when the scale of the series is large or you care more about ratio errors than absolute differences. For example, in population forecasting, predicting 110 vs 100 (10 absolute error) is small percentage (10%), whereas predicting 1010 vs 1000 (also 10 absolute error) is 1% – RMSLE would treat these more fairly, whereas MAE would give equal weight. RMSLE is also often used when data ranges over several orders of magnitude and you want to not let huge values dominate the metric. One must be careful with zero/negative values (the +1 in log helps avoid log(0)).

---

### • MSPE / RMSPE
Mean squared percentage error (not as common) or Root mean squared percentage error.  

These are like MAPE but squaring the percentage error, which heavily penalizes some outliers. Not widely used.

---

### • MASE (Mean Absolute Scaled Error)
This is a metric that scales the MAE by the MAE of a naive baseline (like last value or seasonal naive).  

$MASE = \frac{MAE_{model}}{MAE_{naive}}$  

If MASE < 1, your model is better than the naive on MAE; if > 1, worse. It was proposed by Hyndman to address some issues with MAPE and others, and to provide a scale-free metric that is also comparable across series and doesn’t blow up. It’s quite interpretable in a relative sense. If you see MASE = 0.8, it means your errors are 80% of the naive error (20% improvement). MASE can be used for intermittent series (since it doesn’t involve dividing by zero; the naive forecast error would be something like using last observation or seasonal last).

---

### • Others
There are many, like MedAE (Median AE), sMAPE (we did), MAAPE (mean arctangent absolute percentage error – a newer smooth version of MAPE), OWA (overall weighted average, used in M4 competition combining sMAPE and MASE relative to naive), etc., and business-specific ones (e.g. service level metrics).  

For example, in supply chain, one might use bias (mean forecast error) to see if consistently over/under predicting. Or use a “target service level” metric where if forecast is within ±X% it’s fine.

---
## Comparison Table

| Metric | Scale | Pros | Cons |
|--------|-------|------|------|
| **MAE** (Mean Absolute Error) | Same units as data | Easy to interpret, robust to outliers, linear penalty | Doesn’t emphasize large errors |
| **RMSE** (Root Mean Square Error) | Same units as data | Penalizes large errors, standard in literature, works with normal error assumptions | Sensitive to outliers, harder to explain |
| **MAPE** (Mean Absolute Percentage Error) | Percentage (%) | Scale-independent, intuitive for business use | Division by zero issue, unstable for small actuals |
| **SMAPE** (Symmetric MAPE) | Percentage (%) | Bounded (0–200%), symmetric penalty for over/under-forecast | Less intuitive, undefined if both actual & forecast = 0 |

---


## Choosing Metrics

Often you’ll compute several. It’s good to check **MAE and RMSE together** – if they are very far apart, that indicates some big outliers in error (RMSE >> MAE). **MAPE/SMAPE** give relative error which is important for context (5 unit error might be trivial for a series with level 1000, but huge for a series with level 5 – MAPE shows that). If you have multiple series, you might average MAPE across series (beware of small denominators), or better, compute a global **WAPE** across all series.  

Competitions often pick one, like **sMAPE** or **MASE**, but internally businesses may track multiple (e.g. *“We have a 10% MAPE overall, with an RMSE of X units”*).

---

## Plotting Forecast vs Actual

Beyond numeric metrics, a visual plot of predictions against actuals over time is invaluable. It can show when and where errors occur – e.g.:

- Did the model miss seasonal peaks?  
- Does it consistently lag behind turning points?  
- Are there particular periods where it fails (maybe due to anomaly not in training)?  

A common practice: produce a **forecast plot with confidence intervals**, and actuals overlaid – to inspect goodness of fit. Another diagnostic is the **residual plot** (actual - forecast over time) and its **ACF** to see if residuals are white noise (see section 12).

---

## Example in Code

In code, computing these metrics is straightforward, e.g.:

```python
import numpy as np

y = np.array(actuals)
y_hat = np.array(forecasts)

mae = np.mean(np.abs(y - y_hat))
rmse = np.sqrt(np.mean((y - y_hat)**2))
mape = 100 * np.mean(np.abs((y - y_hat) / y))  # Assuming no zeros in y
```

---

## Takeaway

Choose metrics that align with your goals:
- If all errors in dollars matter equally → use MAE
- If large mistakes are exponentially worse → use RMSE
- If the client/boss thinks in percent terms → provide MAPE (with caution around zeros)

It’s often wise to report multiple metrics to get a fuller picture. And remember, metrics like MAPE/SMAPE can be high if denominators are low – e.g. if actual sales = 1 and forecast = 3, that’s 200% error for a 2-unit miss; in absolute terms that might be okay, but percentage looks huge.
