# Ensembling Forecasts

Just as in general machine learning, ensembles of models often perform better than a single model. In forecasting, ensembling can be very powerful because different models capture different aspects of the data. By combining them, you can often reduce overall error and improve robustness. Here are ways to ensemble forecasts:

- **Simple average ensemble:**  The simplest approach – take multiple model forecasts and average them (equally weighted). For example, if you have a Prophet forecast and an XGBoost forecast, you produce both and then do (forecast\_prophet + forecast\_xgb)/2 as the final forecast.

  This often yields a more accurate prediction than either alone, especially if their errors are not perfectly correlated (one might fix the other’s mistakes). The average also tends to stabilize variance (less overfitting risk than a single model). Many competition-winning solutions are just clever averages of many models. You can also median them (robust to outlier forecasts).

- **Weighted average:** Sometimes you may want to give more weight to a model that you trust more on certain range or overall. For instance, maybe ARIMA works well for short-term and a neural net works well for long-term trend.

  You could assign weights (say 0.7 \* ARIMA + 0.3 \* NN) either based on performance metrics (like inverse of validation error) or based on intuition about regime. These weights can even vary by time horizon (maybe weight ARIMA more on 1-day ahead, weight Prophet more on seasonal long horizons). That’s a bit advanced but doable.

## Model blending vs stacking:

- **Blending (averaging):** as above just mixes outputs.

- **Stacking (meta-learning):** Here, you train a meta-model to combine the outputs of base models.

  For example, use the forecasts of Model A, Model B, Model C as input features to a simple regression that tries to predict the actual. You’d typically need a validation set to train this meta-learner (to avoid overfit). A stacking model might learn optimal weights or even nonlinear combination of forecasts. E.g., maybe when volume is high, one model is better, when low, another is – a meta-model could learn that. In forecasting, stacking is less common than just averaging, but it’s used in advanced solutions. It’s essentially a small ML model on top of forecasts.

## Ensemble different model types: 
Combining statistical models (like ARIMA, ETS) with ML models (XGBoost) and DL models (LSTM) can yield very good results. 

Each has different strengths: ARIMA might nail the short-term autocorrelation, XGBoost might capture holiday effects from features, and LSTM might capture some long-term pattern. Their errors might be uncorrelated, so averaging reduces net error. For instance, an ensemble could be: (Prophet + XGBoost)/2. 

A known trick is Prophet handles trend and seasonal well but can’t model irregular short spikes – an XGBoost on Prophet’s residuals can learn remaining patterns. This effectively is an ensemble structured as: Forecast = Prophet(t) + XGBoost(predicted\_residual). This approach has been noted to improve accuracy over Prophet alone, since XGBoost fixes some of Prophet’s systematic biases (maybe Prophet is too smooth, and XGBoost brings some responsiveness). 

Another classic hybrid: ARIMA + ANN where ARIMA models the linear part and an Artificial Neural Network models the non-linear part of residuals. Such hybrid models have been in literature (sometimes called hybrid ARIMA-ANN, or ARIMA for structure + ML for remainder).

In fact, research shows ensembles of simple models can often outperform single fancy model – because the combination can capture a broader set of patterns and reduce noise.

## Different data segments ensemble: 
Sometimes you ensemble by regime: e.g., use model A for weekdays, model B for weekends. Or model A for product category 1, model B for category 2. That’s more like choosing best model for each context – which is a form of ensemble (though more like a switch). This might involve a classification of context then picking model accordingly. For example, an energy forecast might use one model for working days and another for holidays.

## Prediction intervals in ensembles: 
Combining forecasts is easy, but combining uncertainty is trickier. A quick fix: simulate many forecasts from each model (if they provide simulation or bootstrap) then average those simulations to derive an interval. Or assume errors independence and use variance combination formulas. That’s advanced – often people just focus on point forecast ensemble and might not provide combined PI (or they pick one model’s PI to present, albeit that underestimates combined uncertainty).

## Ensemble overhead: 
Running multiple models means more computation and complexity. However, many forecasting tasks are not extremely computational once set up (and parallel computing can train models concurrently). For important tasks, the accuracy gain is often worth it.

## Example code (conceptual):

```python
# Assume we have defined prophet_model and xgb_model and fitted them
prophet_pred = prophet_model.predict(future_dates)['yhat'].values
xgb_pred = xgb_model.predict(X_future)
# simple average ensemble
ensemble_pred = 0.5 * prophet_pred + 0.5 * xgb_pred
```
If you had actuals, you could compare ensemble\_pred vs actual to see improvement.

## Stacking example: 
if you wanted to learn weights:

```python
# create validation set forecasts
prophet_val = prophet_model.predict(dates_val)['yhat'].values
xgb_val = xgb_model.predict(X_val)
# use linear regression to learn combination
meta_features = np.vstack([prophet_val, xgb_val]).T  # shape (n_val, 2)
meta_model = LinearRegression()
meta_model.fit(meta_features, y_val)
print(meta_model.coef_, meta_model.intercept_)
ensemble_val_pred = meta_model.predict(meta_features)
```

This linear meta_model finds coefficients w1, w2 such that w1prophet + w2xgb fits y_val best. Those coefficients can be used on future forecasts as weights. One must be careful to avoid overfit (hence keep a separate val set).

## Downside of ennsembling: 
Harder to interpret (which model is "right"?), and one must maintain multiple models. Also, if models are very similar (like ARIMA and ETS on same data often produce similar forecasts), ensemble adds little.

In conclusion, ensembling is a best practice for improving forecast accuracy. Even something as simple as averaging a statistical forecast and a machine learning forecast can yield a strong result with minimal extra effort. The M4 competition result paper noted that a combination of many methods won (and in general, combination forecasts have theoretical justification for being better in MSE sense if models are unbiased and have uncorrelated errors).
