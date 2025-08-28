# Residual Analysis & Error Diagnostics

After fitting a forecasting model, it’s crucial to examine the residuals (the forecast errors) to see if the model has captured the patterns or if there’s leftover structure. A good model’s residuals should be essentially “white noise” – meaning no autocorrelation, zero mean (no bias), constant variance, and no patterns. Here’s how to do residual analysis:

## Check Autocorrelation of Residuals
Check autocorrelation of residuals: Compute the ACF (and PACF) of the residual series. If your model captured all linear autocorrelations, the residual ACF should have no significant spikes (all within confidence bounds). If you see a significant autocorrelation at lag 1 in residuals, it means there was still some predictable structure at lag 1 the model missed – perhaps an AR(1) term could have helped or a seasonal pattern remains. For example, if you fit an ARIMA and the residual ACF shows a big spike at lag 12, likely a yearly seasonality is still present that the model didn’t account for. You would then consider adding a seasonal term or an external regressor to capture that. Many statistical models come with a Ljung-Box test for residual autocorrelation – it tests the null “residuals are white noise.” A low p-value indicates residuals are not independent (bad). If that happens, your model is missing something (or overfitting weirdly).

## Residual Distribution
Residual distribution: Plot a histogram or Q-Q plot of residuals. Ideally, if you assumed normal errors (common in many models), residuals should roughly follow a normal distribution (bell curve, Q-Q plot near straight line). If they are skewed or heavy-tailed, your prediction intervals based on normal may be off. Also, any extreme outliers in residuals might warrant investigating those time points (maybe an outlier event the model couldn’t handle). If residuals have non-constant variance (e.g. bigger errors when level is higher), that suggests maybe a transformation (like log) could stabilize variance for a better model, or a model that has multiplicative errors would suit better. A plot of residuals vs fitted values can also show if variance changes with level (heteroscedasticity).

> **What is heteroscedasticity?**
> 
> •	In regression, we assume constant error variance - homoscedasticity.
> 
> •	Heteroscedasticity means that variance changes with X or with the fitted value:
> 
> •	Consequences: OLS coefficients remain unbiased, but standard errors are wrong, tests/p-values and CIs become unreliable, and prediction intervals are miscalibrated.

> **What's the meaning of Fitted Value?**
> 
> •	Fitted value: the model’s predicted mean of the target for a given X. It isn’t a single global number. It’s one number per observation.
> 
> •	For each row in your dataset with features xix\_ixi, the model gives a predicted mean y^i=E\[Y∣X=xi]\hat y\_i = \mathbb E\[Y\mid X=x\_i]y^i=E\[Y∣X=xi].
> 
> •	In linear regression: y^i=xi⊤β^\hat y\_i = x\_i^\top \hat\betay^i=xi⊤β^.
> 
> •	The residual for that row is ei=yi−y^ie\_i = y\_i - \hat y\_iei=yi−y^i.

# Residuals Over Time
Plot residuals as a time series. Look for any systematic runs: e.g., are residuals positive for a long stretch then negative for a long stretch? That could indicate the model is not tracking a slow change (maybe a trend shift) – the model might be consistently under-predicting for a period (residuals > 0 means actual > forecast) and over-predicting later. Ideally, residuals fluctuate around zero randomly. You can also check if residuals have any seasonal pattern: e.g., compute average residual by day of week – if Monday has consistently positive residuals, model underestimates Mondays (maybe need a Monday effect). This is like analyzing “error by segment.”

## Patterns in residuals to diagnose issues:

•	If residuals have mean ≠ 0 (consistently positive or negative) -> model has bias (maybe it’s consistently low or high). Could be due to a missing trend or growth rate. This could also happen if you differenced data when maybe you shouldn’t have or vice versa.
•	If residuals variance increases over time -> the process variance wasn’t stationary; maybe a transformation (log) would be better, or a model with time-varying volatility (like ARCH/GARCH for financial data) might be needed if that’s relevant (volatility modeling is a whole field).
•	If residuals correlate with an external factor (maybe plot residuals against temperature, see correlation) -> that factor should perhaps be in the model. E.g., if after modeling, the remaining error still correlates with temperature, you should include temperature as a regressor.


## Another Diagnostic: Underfitting vs Overfitting

•	If residuals still have structure -> underfitting (model too simple to capture all patterns). Solution: add parameters or features.
•	If residuals are white noise but model is very complex, and maybe you used too many parameters, check if you might be overfitting (overfit usually shows up as residuals looking fine on training but maybe not on validation). Overfit models may actually have residuals that are too good (like zero on training) but then on test explode. So always check residual diagnostics on a holdout set if possible, not just training.

## White Noise Residuals

If after your analysis, residuals appear random (no autocorr, zero mean, constant variance), you can be more confident in model. You then essentially have all systematic signal extracted, leaving only unpredictable noise – that’s the goal in forecasting modeling. However, note that “white noise” doesn’t guarantee good forecast – it could be that the model overfit weirdly to achieve that, so pair this with out-of-sample evaluation.

## Seasonal Subseries of Residuals

Sometimes one plots residuals by season to see if certain seasons systematically under/over predict. E.g., take all January residuals vs all July residuals. If a pattern emerges, maybe your seasonal modeling isn’t capturing some seasonal amplitude changes (maybe winters get increasingly harsher but model assumed constant seasonal effect, etc.).

## Error Metrics by Segment

 e.g., MAPE on weekdays vs weekends. If big difference, model might not capture the weekend pattern well.

# Example of Residual ACF Diagnosis

Example of residual ACF diagnosis: You fit ARIMA(0,1,1) on some data and then plot residual ACF. You see big spikes at lags 12, 24... The model’s telling you “there’s something repeating every 12 lags I didn’t model.” That screams seasonality – perhaps you needed a seasonal ARIMA component or a seasonal dummy. Or maybe an ARIMA can’t capture a long seasonal period easily without being explicitly told. So next iteration, you try ARIMA(0,1,1)(0,0,1,12) i.e. add seasonal MA(1) at 12. Now residual ACF hopefully has no spike at 12.

# Example in Python

After fitting, say you have residuals in an array resid. You can do:

```
 	import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.plot(resid); plt.title('Residuals over time')
plot_acf(resid, lags=30)
plt.hist(resid, bins=20); plt.title('Residual distribution')
```

Ljung-Box test:

```
 	from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(resid, lags=[10, 20], return_df=True)
```

This might output p-values for no-autocorr up to lag 10, 20. If p < 0.05, then residuals are not independently distributed (not good).

## Non-linear Patterns in Residuals

If you suspect maybe residuals are bigger when predictions are higher (maybe the model underestimates peaks consistently), you might add an interaction or try a different model that can handle that non-linearity. For example, sometimes residuals of a linear model might show an arch shape when plotted against predicted – might indicate a quadratic term missing or so.

In practice, residual analysis is an iterative model improvement tool. It’s very much like checking assumptions: if something’s off, you refine the model. Eventually, hopefully, you get residuals \~ white noise. If you cannot, it might mean either the process truly has a structure that your class of models can’t capture (maybe try a more complex model or add external data), or there’s an unpredictable component.

Also, remember: no model will have perfect residuals if the data has unpredictable spikes, etc. The white noise assumption is often checked in academic contexts; in practical terms, you want no obvious patterns left. Some patterns might remain, but are too small to worry about or too costly to model further.

To quote a principle: “All models are wrong, but some are useful.” Residual diagnostics help ensure your model is at least not obviously wrong. As a final check, monitoring residuals in production (forecast errors as new data comes in) is important too – if they start showing patterns, maybe relationships changed (this goes to drift detection in deployment, covered later).



