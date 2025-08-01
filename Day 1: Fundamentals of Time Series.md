# Day 1: Fundamentals of Time Series

## 1. What is Time Series Data?

A time series is a sequence of data points indexed in time order (e.g., daily stock prices, hourly website traffic). Unlike typical ML datasets, time series are temporally ordered, meaning the order of observations matters and past values influence future values. You can’t randomly shuffle time series data without losing information because autocorrelation (correlation of a series with its past values) is often present.

### Key characteristics:

- **Temporal Ordering:**  
  Time is the primary index. Each observation is linked to a timestamp (e.g., a date or second). This ordering means observations are not independent and identically distributed (i.i.d.) as in standard ML; instead, each value may depend on previous values.

  Examples: Daily stock prices, monthly sales, hourly temperature readings, yearly population figures — anything recorded over time in sequence.

- **Not Typical Supervised ML:**  
  In standard ML, we assume data points are independent. In time series, the temporal dependency violates this assumption. You must be careful to avoid “training on future data” when modeling or validating — the model should never see data from the future when predicting (to prevent data leakage).

- **Autocorrelation:**  
  It’s common for time series to show autocorrelation — meaning the series correlates with lagged versions of itself. For example, yesterday’s temperature might be a good predictor for today’s. Patterns like these violate independent observations but provide signal we can exploit.

- **Seasonality:**  
  Many time series have seasonality — regular patterns that repeat at fixed periods (daily, weekly, yearly). For example, web traffic might spike every weekend, or electricity demand might be higher every Monday morning. These periodic patterns are a key difference from typical ML data.

### Why time series ≠ typical supervised learning

In time series forecasting, we use previous time steps as features to predict future values. The order, lags, and pattern over time are crucial. You cannot just throw time-series data into a standard regression without accounting for time; you must respect the sequence and typically create features like “previous value” or “day of week.” Also, model evaluation must mimic real-time forecasting — training on past and testing on future, never randomly splitting.

---

## 2. Stationarity

A stationary time series is one whose statistical properties do not change over time. Formally, it has a constant mean, constant variance, and constant autocovariance (correlation structure) over time. This matters because many classical models (like ARIMA) assume stationarity for making forecasts. If a series is non-stationary (e.g., has a trend or changing variance), those models may produce unreliable forecasts.

### Key points:

- **Definition:**  
  Stationarity means the series doesn’t wander off with time — no overall trend, no changing volatility, no seasonality (or those have been removed). For example, white noise (random fluctuations with fixed mean 0 and variance) is stationary. In contrast, a trending series (e.g., steadily increasing sales) is non-stationary because its mean changes over time.

- **Why it matters:**  
  Many forecasting methods (ARIMA, ARMA) assume a stationary process to confidently extrapolate the future. If the data isn’t stationary, these models can misestimate parameters and give poor predictions. Thus, a common first step is to test for stationarity and transform the data if needed.

### How to detect stationarity:

- **Visual inspection:**  
  Plot the series. Does it show a trend (upward/downward) or obvious seasonality? Does variability increase over time? If yes, it’s likely non-stationary.

- **Rolling statistics:**  
  Calculate a rolling mean and rolling standard deviation and plot them against time. For a stationary series, the rolling mean and std should be roughly horizontal (flat) lines. If they shift over time, that’s a sign of non-stationarity.

<img width="554" height="455" alt="image" src="https://github.com/user-attachments/assets/ff6b95d3-bc89-40b1-98f0-1e67c1dddbc5" />

A stationary series would show no systematic change in the mean (orange) or spread (green) over time.

- **Statistical tests:**  
  The Augmented Dickey-Fuller (ADF) test is commonly used. It tests the null hypothesis that the series has a unit root (i.e., is non-stationary). A low p-value (p < 0.05) leads us to reject that null hypothesis, suggesting the series is stationary. Another test is KPSS, which has opposite hypotheses (null = stationary). In practice, you can run these tests using libraries (e.g., statsmodels).




### How to make a series stationary (if it’s not):

- **Differencing:**  
  Take the difference between consecutive observations, i.e., use  
  `Y'_t = Y_t - Y_{t-1}`. Differencing can remove trends and make mean constant. Sometimes one difference isn’t enough (series might need second differencing). Many models (ARIMA’s “I” term) incorporate differencing.

- **Log or power transformations:**  
  If variance grows with level, a log transform can stabilize the variance (e.g., a series with exponential growth can be tamed by log). Taking logs makes fluctuations at high values more comparable to those at low values. For example, if you have a series of counts or sales,  
  `Y'_t = log(Y_t + 1)` (add 1 if zeros present) can stabilize increasing variance.

- **De-trending and De-seasonalizing:**  
  You can subtract a moving average or use regression to remove a trend. To remove seasonality, subtract the seasonal average or use seasonal differencing (difference the series with itself from one season ago). After removing trend/seasonality, the remainder is often stationary.

- **Seasonal decomposition:**  
  Methods like STL (Seasonal-Trend decomposition using Loess) can separate a series into Trend, Seasonal, and Residual components. Removing the trend and seasonal components leaves the residual, which should be stationary noise if the model captured the structure.

### Summary

Stationarity means a series is stable over time in behavior. It’s crucial because many forecasting techniques either require it or are easier to apply on stationary data. If your series isn’t stationary, apply transformations until it is (at least approximately). After modeling/forecasting, you can always invert those transformations to get forecasts back on the original scale.

---

## 3. Components of Time Series

Most time series can be thought of as a combination of several components: **Trend**, **Seasonality**, **Cycles**, and **Noise**. Identifying these components helps you understand the data and build better models (e.g., include a trend term, adjust for seasonality, etc.).

### Main components:

- **Trend:**  
  The long-term upward or downward movement in the series. It shows the overall direction over a long period. For example, a company’s revenue may have an upward trend over years. A trend can be linear or nonlinear (e.g., exponential growth) and can change (e.g., a series might trend up then level off). If a series has a trend, its mean is changing over time (violating stationarity).

- **Seasonality:**  
  A repeating short-term pattern at fixed, known intervals. This is usually tied to calendar or regular cycles. Examples: daily seasonality (e.g., traffic peaks every rush hour), weekly seasonality (weekends vs weekdays), yearly seasonality (retail sales spike every holiday season). Seasonal effects have a fixed period (e.g., 7 days, 12 months). If you plot data and see a pattern repeating every year, that’s seasonality. Seasonality is predictable; you can add seasonal terms or dummy variables (e.g., a “month” feature) to model it.

- **Cyclicality:**  
  Cycles are patterns like seasonality but not strictly fixed period or caused by calendar. They are longer-term oscillations, often influenced by economic or other external factors, and the cycle length might vary. For example, a business cycle or a boom-bust cycle in economics — it repeats but not on a rigid schedule like seasonality. Cycles tend to be irregular or of longer duration than seasonal patterns.

- **Noise (Irregular component):**  
  The random variation left over after accounting for trend and seasonality. This is essentially the “residual” part of the series — the unpredictable part. We usually assume this noise is random (white noise). If a model captures trend and seasonality well, the residuals should ideally be just noise (with no patterns).

### Autocorrelation

Autocorrelation refers to correlation of the series with its own lagged values. Seasonality and trend can create strong autocorrelation at certain lags (e.g., an annual cycle causes a spike in autocorrelation at lag 12 for monthly data). Plotting the autocorrelation function (ACF) helps reveal seasonality and cycles (you’ll see peaks at the seasonal lag). Autocorrelation that remains in residuals indicates a pattern still present in the data (which modelers aim to eliminate by including appropriate terms).

---

## Visualizing Components with Decomposition

You can use decomposition techniques to visualize components. For instance, Python’s `seasonal_decompose` or STL methods split a time series into additive components: Trend, Seasonal, and Residual (noise).

<img width="975" height="647" alt="image" src="https://github.com/user-attachments/assets/387ac633-6816-4a66-a650-0be020548045" />

We can clearly see an upward trend in the first half of the year (trend panel) and a regular monthly seasonal pattern (third panel), with the residuals fluctuating around zero (last panel). This confirms the series = Trend + Seasonal + Residual.

Another powerful method is STL decomposition (Seasonal-Trend Loess) which is more robust and allows multiple seasonalities. These tools are useful for exploratory analysis and for preprocessing (e.g. you might remove the seasonal component before modeling).

Understanding these components is critical. If you know your data has a seasonality and a trend, you might choose models (or features) that explicitly address those (e.g. use SARIMA for seasonality, or Prophet which includes trend and seasonal terms, or add Fourier terms for seasonality). In summary, Trend tells you if the series is growing/changing overall, Seasonality tells you the regular cycles, Cycles capture longer irregular swings, and Noise is what’s left that we can’t explain (hopefully just random). Identifying these makes your forecasting more informed

### Example in Python:

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(automotive_monthly_sales_indexed['sales'], model='additive', period=12)
result.plot();


