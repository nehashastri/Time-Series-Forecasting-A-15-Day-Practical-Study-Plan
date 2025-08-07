# Day 3: Feature Engineering and Machine Learning

# Feature Engineering for Time Series

In machine learning-based, you need to convert the time series into a supervised learning format – i.e. create features from past observations to predict the future. Feature engineering is key for ML models to capture time patterns. Important techniques include:

- **Lag Features:** These are the most basic features – prior values of the time series. For example, to predict sales today, you might include sales yesterday (`lag_1`), a week ago (`lag_7` for daily data), etc., as features. Including enough lag features allows ML models (like regression or trees) to mimic an AR model. In practice, you decide how many lags based on domain knowledge or data (e.g. if autocorrelation cuts off after 5 lags, maybe use lags 1–5). In a pandas DataFrame, you can create them like:

  ```python
  df['lag_1'] = df['y'].shift(1)
  ```
> Note: with seasonal data, include seasonal lags (e.g. 24-hour ago for hourly, 12-month ago for yearly cycle). Lag features essentially let an ML model do what AR terms do in ARIMA.

- **Rolling Window Statistics:** These aggregate statistics over a moving window of past values – e.g. the past 7-day average, past 30-day sum, past 3-day standard deviation, etc. Rolling features smooth out noise and capture local trends or volatility. 

  For instance, a 7-day rolling mean feature can help a model understand the underlying trend/week-average level, while a rolling standard deviation feature can indicate how volatile the series has been recently. 

  **Example:**
  ```python
  df['roll_mean_7'] = df['y'].rolling(window=7).mean() #(Centered on current time or using last 7 observations)
  ```
  These features provide a sense of momentum or seasonal baseline to the model. For intermittent or bursty data, rolling sums (e.g., sum of last 3 observations) can be useful.

- **Time-based Calendar Features:** These are deterministic features derived from the timestamp (useful especially for capturing seasonality and calendar effects). 

  **Examples:**
  - **Day of week / Month of year:** If you have daily data, create a feature for day-of-week (`Monday=0…Sunday=6`) – you can one-hot encode it or just keep numeric if using trees. This helps capture weekday vs weekend effects. Similarly, month of year can capture annual seasonality for monthly data (or you could use quarter).
  - **Holiday or weekend indicators:** A binary feature for whether a day is a holiday (or weekend) can help capture special events. For example, predict higher sales on Black Friday by having a `BlackFriday` flag.
  - **Cyclical encoding:** If you use numeric encoding for cyclical features like day-of-week or month, it’s often helpful to encode them cyclically (since, for example, December (12) and January (1) are actually adjacent in time). This is done via sine/cosine transformations:
    ```python
    day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)
    ```
    This maps the circular feature into two dimensions where closeness in the calendar is preserved. Similarly for hour of day (24h cycle). Many find this improves modeling of periodic features.
    By adding time-based features, you are giving the ML model knowledge of the calendar, enabling it to learn, for example, that “Fridays have higher traffic” or “sales dip in February” – rather than expecting it to infer that purely from lagged target values (which might require many lags). These features essentially inject seasonality information in a way the model can use.

- **Fourier Features for Seasonality:** An alternative to one-hot encoding seasonality (especially for long seasonal periods) is using Fourier terms. This is what Prophet does under the hood for yearly seasonality. These create continuous cyclic features that can approximate seasonal patterns. They are useful in regression models to handle seasonality smoothly. 

  **Example:** To capture a yearly cycle, you might include sine and cosine of the day of year (effectively a Fourier series representation of a periodic function). If your seasonality isn’t well captured by a single frequency (like complex seasonal curves), adding a few pairs of Fourier terms can help a linear model fit a curve.  

  In Python, **fbprophet** automatically does this for yearly, weekly, and daily seasons, but you can manually do it with:
  ```python
  df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
  df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
  ```
- **External / Exogenous Features:** Related data (exogenous variables) can improve forecasts. For instance: Weather can affect energy load or sales. Promotions or price changes can affect product sales. Include these as input features aligned in time.  
  **Example:** If predicting electricity demand, include temperature forecasts as a feature.

- **Interactions and Derived Features:** Combine features for special effects.  
  **Example:** `isWeekend * isHoliday` can capture differences when a holiday falls on a weekend versus a weekday. These are more advanced and situation-specific.

- **Trend as a Feature:** If a long-term trend isn’t captured by other features, add a **time index** (`t = 1, 2, 3…`). A regression can partially fit a trend through this.  
    **Caution:** If you fit a normal linear trend (y = m*t + b), you’re assuming: The trend grows or declines at the same rate forever. Real-world data often breaks that rule. Prophet’s piecewise trend is more robust, but in ML, you can mimic it by adding: “Time since a particular event” or Polynomial time features
  > Note: Prophet splits the trend into segments using changepoints — specific dates where the growth rate is allowed to change. Between changepoints, the trend is linear (constant slope). At changepoints, the slope can increase, decrease, or even flip direction.

---

Feature engineering for time series is crucial because unlike models like ARIMA which inherently use past values, a machine learning model won’t know to look back in time unless you explicitly give it features to do so. 

The good news: a lot of this can be automated or assisted by libraries. For example:  
- **tsfresh**: A Python library that can automatically compute many time-series features (statistics, frequencies, etc.).  
- **Facebook’s Kats**: Has modules to generate features (and even uses models like Prophet internally to generate changepoint features).  
- **Nixtla’s StatsForecast** and **NeuralForecast**: Allow auto-features in some contexts.  

Still, understanding the above core techniques (**lags**, **rolling**, **calendar features**) is essential, as they cover most of what you need for classical ML approaches.

---

**In summary:**  Good feature engineering often determines success in time series ML models. It compensates for the model’s lack of built-in memory by handing the relevant history to it *on a silver platter*.

---


