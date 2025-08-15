# Forecasting Horizons

The “forecasting horizon” means how far ahead you are forecasting – e.g. one step (tomorrow) vs multiple steps (next 12 months). Different strategies are used for one-step vs multi-step forecasting:

### One-step (recursive) forecasting
You predict the next time step only. This is simpler – your model directly outputs $\hat{y}_{t+1}$ given data up to $t$. Many models (like ARIMA by default) produce one-step forecasts.  
If you need multiple steps ahead, you feed forecasts back in recursively: i.e., predict $t+1$, then treat that as an observed value and predict $t+2$, and so on. This is called **recursive multi-step forecasting**.

**Advantages:**
- You only have to build a model for one-step ahead.

**Disadvantages:**
- Errors compound – any mistake in early predictions becomes input for later predictions, potentially causing error to grow.

### Direct multi-step forecasting
Here, you build a separate model for each horizon, or a model that outputs multiple horizons at once. For example, train one model to predict 1-step ahead, another for 2-steps ahead (using appropriate lag features that skip one ahead), etc.  
Or in deep learning, you can have the network output a vector of the next H values directly.

**Advantages:**
- The model can optimize specifically for that horizon.
- Doesn’t propagate its own errors.

**Disadvantages:**
- You need more complexity (H models or a more complex output).
- Requires more training data for longer horizons (since e.g. for 10-steps ahead model, your training pairs are further apart in time).

In practice, **Nixtla’s StatsForecast** library can do direct forecasting with ARIMA/ETS by iterating internally.  
For ML, you can create target variables shifted by different lengths.

### Which to use?
If your horizon is short, **recursive** is fine and simpler.  
If horizon is long and your model is flexible (like a neural net), **direct** often yields better accuracy for far-out predictions because it doesn’t accumulate errors.  

Some advanced strategies exist like forecasting with an iterative model plus error correction, or using multiple output models (e.g. an MLP that outputs a vector of next H values). Also, some specialized losses (e.g. optimizing jointly across multi-step forecast) can be used.

---

## Static vs Dynamic Forecasting

### Static Forecast
A **static forecast** (sometimes called **one-off** or **“in-sample” forecast**) uses actual known values for any required inputs beyond the forecast origin.  

For example:
- If you forecast 5 days ahead **statically**, you use the actual *yesterday* → forecast **day1**.
- When forecasting **day2**, you might still use the actual **day1** (if it was observed in real life).

### Dynamic Forecast
A **dynamic forecast** uses its own **predicted values** for previous steps when forecasting multiple steps ahead.  

In other words:
- Once you predict **day1**, you treat that as reality for predicting **day2**, and so on.
- This is essentially the **recursive approach**.

### Key Differences
- **Static forecasting** is only possible in hindsight (or for evaluation on a test set where you “cheat” by using actual future values for the intermediate points).
- **Dynamic forecasting** is the true simulation of real usage — predictions feed into themselves.

### Impact on Model Evaluation
- When evaluating models, if you use **actual intermediate values** (static), the error will be **lower** because you’re not compounding mistakes.
- In practice, when we say a model’s performance, we usually mean **dynamic forecast error** since that’s what you’d face in real deployment.

Some software (like **EViews** or **Stata**) have explicit *static* vs *dynamic* forecast options when evaluating models on historical data.

> Simplified Summary
> - **Dynamic** = The model’s predictions feed into itself.  
> - **Static** = We always reset to **ground truth** when available.  

**Static** is useful for diagnosing model errors assuming perfect inputs (e.g., how well would my model do if it only had to predict one step at a time with no feedback).  
**Dynamic** is the **realistic** scenario.

---

## Rolling Forecasts vs Expanding Window

This concept is about how you use data when forecasting over time (especially in backtesting or model updating):

### **Rolling (moving) window forecast:**  
  You keep a fixed-size training window that moves forward through time. For example, you always use the last 3 years of data to forecast next month, then move ahead one month (dropping the oldest month, adding the new). This ensures roughly consistent amount of training data and can adapt to changes (because very old data drops out). It’s useful if the process is non-stationary or if you suspect only recent data is relevant (maybe due to concept drift).

### **Expanding window (growing):**  
  You start with an initial training period, then for each forecast, you expand the training set to include new data as it becomes available. So the training size grows over time, never discarding data. This assumes older data still contains signal (and you don’t worry about computation growth). Expanding window gives the model more and more data – often leading to more stable parameter estimates for statistical models. Most academic time series analysis uses expanding (i.e. use all data up to time T to forecast T+1).

In cross-validation, these correspond to different fold strategies:

- An **expanding window CV** means each fold uses data from start up to some point as train, and next chunk as test.  
- A **rolling window CV** might use a sliding window of fixed width for train for each fold.  

Both avoid using future data to predict past, unlike standard CV.

### Which to choose?

- If you have a long stable series, **expanding window** makes sense (why throw away data?).  
- If the series distribution changes over time (structural breaks, trends that your model might not handle explicitly), a **rolling window** might adapt better (the model trained only on the latest data, not burdened by outdated patterns).  
- Sometimes a hybrid is used (e.g. use at most last N years even as you expand, to cap size).

---

## Forecast horizon length considerations

- Often, the appropriate model or features depend on how far out you need to forecast.  
  - For very short horizon (next step), models can lean heavily on recent lags, and being super precise with near-term patterns is key.  
  - For a long horizon (say 12 months out), the model will likely need to rely on capturing seasonal cycle and trend, since individual day-to-day fluctuations average out and aren’t predictable that far out. You might also incorporate expected future changes (like known events) for long horizon.

- Evaluation metrics might change with horizon:  
  - For example, MAPE might be fine for short-term, but for very long term you might focus on overall trend accuracy or use measures like relative error compared to a naive seasonal forecast.  
  - We often evaluate multi-step forecasts by looking at error at each horizon (e.g. how error grows from 1-step to 6-steps ahead).  
  - Some models (like some neural nets) try to minimize the overall error across all horizons at once.

- If the forecasting horizon is beyond the range of your training data’s patterns, it gets tricky (extrapolation).  
  - E.g., predicting 10 years out with only 3 years of data – essentially impossible to do reliably without external info or assumptions.  
  - Models like Prophet at least assume trend continues (with possible change) and seasonality repeats; pure ML might just give some mean.  
  - So for very long term, you often incorporate domain knowledge or scenarios rather than purely relying on a model.

## Summary

- Deciding how to forecast multiple steps is an important design choice.  
- **Recursive (dynamic) forecasting** is straightforward but prone to accumulating error.  
- **Direct forecasting** can be more accurate for longer horizons but requires either multiple models or a multi-output model and sufficient data. It’s often beneficial to try both if feasible. Many competition-winning solutions use ensembles of recursive and direct approaches.  
- Also, understand **static vs dynamic**: always evaluate your model in the way it will be used (dynamic) to get a true sense of performance.  
- Use rolling/expanding windows properly during model selection to avoid lookahead bias.  

Essentially, forecasting isn’t just building a model – it’s also deciding how you will generate and update forecasts as time moves on (will you retrain weekly? use an expanding window? etc.). These strategies ensure your model remains robust over the forecasting process.

