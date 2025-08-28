# Residual Analysis & Error Diagnostics

After fitting a forecasting model, it’s crucial to examine the residuals (the forecast errors) to see if the model has captured the patterns or if there’s leftover structure. A good model’s residuals should be essentially “white noise” – meaning no autocorrelation, zero mean (no bias), constant variance, and no patterns. Here’s how to do residual analysis:

## Check Autocorrelation of Residuals
Check autocorrelation of residuals: Compute the ACF (and PACF) of the residual series. If your model captured all linear autocorrelations, the residual ACF should have no significant spikes (all within confidence bounds). If you see a significant autocorrelation at lag 1 in residuals, it means there was still some predictable structure at lag 1 the model missed – perhaps an AR(1) term could have helped or a seasonal pattern remains. For example, if you fit an ARIMA and residual ACF shows a big spike at lag 12, likely a yearly seasonality is still present that the model didn’t account for. You would then consider adding a seasonal term or an external regressor to capture that. Many statistical models come with a Ljung-Box test for residual autocorrelation – it tests the null “residuals are white noise.” A low p-value indicates residuals are not independent (bad). If that happens, your model is missing something (or overfitting weirdly).

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


