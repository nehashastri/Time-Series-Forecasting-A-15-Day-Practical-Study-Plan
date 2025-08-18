# Time-Series Cross-Validation 

## The core rule

* Time matters. You **must** keep past → future order.
* Don’t use random shuffle k-fold like regular ML, or you’ll leak future info into training (temporal leakage).

## The main ways to split

### 1) TimeSeriesSplit / Forward chaining

* You grow the training window over time (or keep it fixed and slide it).
* Example (data: Jan 2018–Dec 2020; 3-month test chunks):

  * **Fold 1:** Train = Jan–Dec 2018; Test = Jan–Mar 2019
  * **Fold 2:** Train = Jan 2018–Mar 2019; Test = Apr–Jun 2019
  * **Fold 3:** Train = Jan 2018–Jun 2019; Test = Jul–Sep 2019
  * …continue until the end.
* Training sets are **expanding supersets** of previous ones by default; or set a **fixed rolling** window (limit the past you keep).
* Why it’s useful: you get multiple evaluations that **respect time order** and can see if performance changes over time.

### 2) Walk-forward validation (rolling origin)

* You mimic real life:

  1. Train on the first block (e.g., first 2 years).
  2. Forecast the **next** step(s) (e.g., next month).
  3. Add that new month to training, **retrain/update**, forecast the next month.
  4. Repeat.
* This is common in **backtesting**. It’s more work, but it best matches how you’ll deploy the model.
* You get a sequence of out-of-sample forecasts across history; their **average error** is a strong estimate of future performance.

### 3) Blocking by groups or seasons

* With seasonal or yearly data, you might compare years.
* **Avoid** removing a middle year and training on data **after** it—that breaks time order.
* Instead do forward blocks, e.g.: Train 2015–2018 → Test 2019; then Train 2015–2019 → Test 2020; etc.
* On very long series, people sometimes repeat **percentage splits**: 50/50, then 60/40, 70/30, … (always forward).

## Using scikit-learn

### `TimeSeriesSplit`

* Gives you split **indices** that move forward in time.
* By default it’s **expanding**; set `max_train_size` for a **fixed rolling** window (useful if old data hurts or the series is huge/non-stationary).
* Each fold’s **train** = all data **before** a point; **test** = the block **right after** it.
* By default, test folds are **equal-size**; you can adjust `test_size` and related params.

### No shuffling

* Never shuffle time series.
* If you use `train_test_split`, set `shuffle=False` and split at a **time boundary** (e.g., last 20% as test).

## Backtesting with model updates

* You can **retrain every step** (most realistic), but it can be expensive.
* Training **once** and forecasting far ahead usually **degrades**.
* A compromise: **periodic retraining** (e.g., monthly).

## Metrics and how to read them

* Compute errors on each test fold, then:

  * **Average** them (e.g., average MAPE) for a single score, and/or
  * Check if error **changes over time** (later folds can be harder if the process shifts).
* If data is non-stationary, you **might** weight later folds more heavily (they’re more like “now”), though a simple average or pooled MAPE across all folds is common.

## Avoiding temporal leakage (super important)

* Features must not **peek** into the future:

  * Don’t compute a rolling mean that **includes the current target** when predicting that same target.
  * Fit scalers/encoders **only on the training data** of each fold; then transform the test data.
* Use proper **pipelines** so nothing from the test period leaks into training steps.

## Concrete example

* Daily data (2018–2020); tuning XGBoost hyperparameters.
* Use `TimeSeriesSplit` with 3 splits and `cross_val_score` on **MAPE**:

  * Suppose fold MAPE = **12%**, **9%**, **15%** ⇒ average ≈ **12%**.
  * If the last fold is much worse (e.g., 2020 had a regime change like **COVID**), your model may need **new features** or a **shorter training window**.
* If you had done a **random** split, you might get **too-optimistic** results, because pieces of 2020 could leak into training.

## Train/Validation/Test pattern for forecasting

* Hold out the **latest chunk** as a **test set** (e.g., last 6 months).
* Use the earlier part for **training + validation** via time-aware CV (e.g., `TimeSeriesSplit`) to tune hyperparameters.
* Then **retrain** on all the training (possibly including the validation portion) and **evaluate once** on the untouched final test.
* Alternative: **rolling origin** over the final year—produce a forecast for each month using only data **up to** that month, then compare to actuals. (This is like multi-fold CV focused on the end period.)

## Bottom line

* Time-series CV **must** preserve time order.
* Use **TimeSeriesSplit**, walk-forward, or carefully designed forward blocks.
* Evaluate like you’d **actually deploy** the model, and vigilantly **prevent leakage**.
* This gives trustworthy estimates and better decisions when tuning ML/DL models for forecasting.
