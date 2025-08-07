# Day 2: Classical Time Series Models

## Classical Models (Most Popular)

These are the time-tested statistical models for forecasting. They often rely on the structure of the time series itself (its lags and past errors) and come with strong theoretical grounding. Key classical models include: **AR, MA, ARMA, ARIMA, SARIMA, Exponential Smoothing (ETS), and Prophet**. Let’s break them down:

---

### AR (Autoregressive) Model

This model uses its own past values to predict the future. An AR(p) model means the value at time t is a linear combination of the previous p values plus noise. It’s essentially a regression of the time series onto itself (lagged). AR models are good at detecting patterns like:

- **Momentum** (if the values keep going up or down)
- **Mean reversion** (if the values tend to bounce back to a certain level)
- **Oscillations** (if values go up and down in a wave-like fashion)

**Autocorrelation plots (ACF, PACF)** help choose the order `p` by seeing how many lags have significant correlation.

- **ACF (Autocorrelation Function)**: This shows how strongly your current value is related to previous values (lags). You plot it to see: “How correlated is today’s value with 1 day ago? 2 days ago? 3 days ago?”. If many lags are significantly correlated, your data has memory — past values matter. If the ACF drops off gradually, it might mean a higher-order AR model is needed.

- **PACF (Partial Autocorrelation Function)**: This is a bit smarter. It tells you how much a specific lag (say, 3 days ago) influences today’s value after removing the effects of the previous lags (1 and 2 days ago). You use PACF to decide how many lags to include in your AR model. For example, if the PACF spikes at lag 1 but then drops off, an AR(1) model might be enough.

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/5a62cba6-2d30-465e-8915-9d3d6d799c06" />
Here p = 1 (maybe 2), q = 1


---

### MA (Moving Average) Model

Sometimes, a value in your time series isn’t just based on a trend or pattern — it’s influenced by random events, like a sudden spike in sales due to a flash sale or bad weather reducing footfall. The MA model tries to capture and correct for those shocks by learning from how wrong the model was in the past.

**Example**:  
Imagine you're tracking daily coffee shop sales. On a rainy day, fewer people show up — your prediction was too high. The next day, if it's sunny, your model might still be “recovering” from the shock of yesterday’s error. An MA model helps adjust for that past shock. It says: “Okay, yesterday we overestimated by 20 customers, so today, let’s tone down the forecast a bit.”

In a **MA(1)** model, the forecast depends on:

- The average level of the series (a constant)
- The error (difference between actual and predicted) from 1 day ago

A **MA(2)** model would also consider the error from 2 days ago, and so on.

> **Note**:  
> Moving Average Smoothing (used in plots and dashboards) is about reducing noise by averaging recent values. You might average the past 3 days to get a smoothed trend line.  
> Don’t confuse this with the **Moving Average Model**.

MA models are good for modeling short-term dependencies (like weather or one-time events that don’t persist but do influence nearby values). The term “moving average” comes from the idea that output is a moving average of random shocks.  
The order `q` can be suggested by the **ACF plot** (ACF cutting off after q lags). In practice, AR and MA are often combined for flexibility.

---

### ARMA (Autoregressive Moving Average)

This combines **AR(p)** and **MA(q)**: $y_t$ depends on both its past values and past errors. It’s a **stationary model** (no differencing), suitable if the series has **no trend or seasonality** (after preprocessing).
ARMA can model a wide range of autocorrelation patterns with relatively few parameters. However, if data is non-stationary (e.g., trending), we need to extend it to **ARIMA**.

---

### ARIMA (Autoregressive Integrated Moving Average)

The **I** stands for “Integrated”, referring to **differencing**.  
**ARIMA(p, d, q)** means an ARMA(p, q) model on the differenced data (d differences applied):

- `d = 0` implies no differencing (so ARMA),
- `d = 1` means we model the **first difference** of the series as ARMA, etc.

ARIMA is a powerful model that can handle trends (with d > 0) and some forms of seasonality (though seasonal patterns often require a seasonal extension).

The famous **Box-Jenkins methodology** is about identifying p, d, q by looking at plots and tests:

- **ADF test** to pick `d` for stationarity
- **PACF** for `p`
- **ACF** for `q`

ARIMA models are quite interpretable and were a forecasting workhorse for decades.  
**Example**:  
ARIMA(1,1,1) means we take first difference of the series and then model it as an ARMA(1,1).

Many software (like **pmdarima’s `auto_arima`**) can automatically select p, d, q by minimizing **AIC/BIC**.

> **Note on AIC and BIC**:
>
> - **AIC (Akaike Information Criterion)** and **BIC (Bayesian Information Criterion)** help compare models to see which fits best — without overfitting. AIC tends to be more forgiving of complex models. BIC is stricter — it penalizes complexity more heavily, especially when you have more data.
> Both of them check:
> - How well the model fits the data (lower error is better)
> - Model complexity (more parameters = more risk of overfitting)

```python
from pmdarima import auto_arima

# auto_arima needs a 1D series
stepwise_model = auto_arima(automotive_ts,
                             start_p=1, start_q=1,
                             max_p=2, max_q=2,
                             d=1,           # let it test for d
                             start_P=0, seasonal=True,
                             D=None,           # seasonal differencing
                             max_P=2, max_Q=2,
                             m=12,              # seasonality period (12 = monthly)
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)

# Summary of the model
print(stepwise_model.summary())
```



---

### SARIMA (Seasonal ARIMA)

This extends ARIMA to handle **seasonality** by adding seasonal terms.  
**Notation**: ARIMA(p,d,q) × (P,D,Q)<sub>m</sub>, where `m` is the **seasonal period**.

**Example**:  
ARIMA(1,0,1) × (0,1,1)<sub>12</sub> for monthly data means: it includes ARIMA(1,0,1) for short-term and one seasonal difference (D=1) plus an MA(1) seasonality at lag 12.

SARIMA adds seasonal autoregressive terms and seasonal moving average terms that operate at **multiples of the season length** (and possibly seasonal differencing).

This is very useful for data with strong seasonality (e.g., electricity demand with yearly seasonality).  
SARIMA models can be complex to tune (six parameters plus period), but tools exist to auto-fit them.

It’s more powerful than ARIMA for seasonal data – e.g., ARIMA can’t directly model “every December sales jump” unless you use seasonal terms.

---

### ETS (Exponential Smoothing)

A different approach than ARIMA, focusing on **smoothing levels, trends, and seasonals**.  
Exponential Smoothing methods (like **Holt-Winters**) create forecasts by **weighted averages** of past observations, where weights **decay exponentially** into the past.

- **Simple Exponential Smoothing (SES)**: For no trend, no seasonality – just smooths the level.
- **Holt’s Linear**: Adds a trend component (two smoothing parameters: one for level, one for trend).
- **Holt-Winters**: Adds a seasonal component (three parameters: level α, trend β, seasonality γ).

**Intuition**:  
Recent observations are given more weight than older ones (hence “exponential” decay of influence).

These methods:

- Are effective for many forecasting problems
- Were widely used in industry (and still are as benchmarks or when data is scarce)
- Automatically adapt to changes (e.g., if trend shifts, the model adjusts)
- Can provide **prediction intervals** (assuming residuals are Gaussian)

Statistical software like `statsmodels.tsa.holtwinters` can fit these.

---

### Prophet (by Meta/Facebook)

**Prophet** is a modern tool (released 2017 by Facebook) that **automates** much of the forecasting process. It’s an **additive model**: forecast = trend + seasonality + holidays + noise

Prophet:

- Fits **piecewise linear or logistic trend** with changepoints
- Includes **weekly, yearly** seasonality (uses Fourier series)
- Lets you specify **holiday effects**
- Handles missing observations gracefully (assumes they are just missing, not zero or repeated)

**Changepoints**: Points in time where the underlying trend changes (suddenly or gradually). Prophet inserts candidate changepoints (default: 25 in the first 80% of data) and selects only the ones that significantly improve the model. You can customize this behavior.

**When to use Prophet**:  
When you want a quick, reasonably accurate forecast with minimal effort, especially for business time series with human-understandable trend/season patterns.  
It’s not always the most accurate, but often a strong **baseline**.  
Prophet also provides **interpretable components**: you can plot trend, yearly and weekly seasonality, etc.

---

## Which to Choose?

- If data is **stationary** (no trend/seasonality): **ARMA** may suffice.
- If data has a **trend or a unit root**: **ARIMA** (with differencing) is appropriate.
- If data has **seasonality**:
  - Use **SARIMA**
  - Or add **seasonal dummy features**
  - Or use **Prophet**/**Holt-Winters**, which handle seasonality directly:
    - **SARIMA** explicitly models seasonal lags
    - **Holt-Winters** directly models seasonal patterns
    - **Prophet** uses Fourier terms

> **Hybrid Approaches**: You can also **combine these models**, e.g. Forecast with **ARIMA** for short-term and use another model for long-term Or **combine their outputs** (we’ll discuss ensembling later)

---

Classical models provide a **solid starting point**. They are relatively **interpretable** (you can see coefficients and understand components), and for many structured time series (especially with not too many external regressors), they produce **good forecasts**.

However, they might struggle if:
- The series has **complex nonlinear patterns**
- There are **many related series**
That’s where **Machine Learning (ML)** and **Deep Learning (DL)** can help — which we’ll cover soon.

**Always consider trying:**
- A simple model like **Naive forecast** = last value  
- Or **Seasonal Naive** = last year’s same period  
- And a **classical model** as **baselines**
