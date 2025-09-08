# Model Selection & Interpretability

Selecting the right model for a forecasting task is both art and science. Interpretability is also crucial, because stakeholders often want to know why the model is predicting certain values. Let’s break down some guidelines:

- **Start simple:** Always begin with simple benchmarks: e.g., a Naive forecast (tomorrow = today’s value) or Seasonal Naive (next year’s same day = this year’s value). These are trivial but often surprisingly hard to beat for certain data (especially naive seasonal for strong seasonal series). Then try simple classical models like Prophet or ARIMA – they’re relatively quick to implement and provide a sanity check. Prophet, in particular, is a good first choice if you have multiple seasonalities and holiday effects – it will automatically model those. ARIMA is good if you suspect strong autocorrelation and a need for differencing. If data is non-seasonal and roughly linear, these might get you close to best.

- **Next: XGBoost with lags/features:** If you need more accuracy or if there are significant external drivers, a gradient boosting model with a rich feature set is a logical next step. It requires more work (feature engineering, cross-val to tune) but can capture nonlinear effects and handle multiple series (via features) easily. Tree models are relatively interpretable via feature importance or SHAP (you can see which features – e.g., lag1, lag7, holiday flag – are most impactful). XGBoost/LightGBM are often used when you have enough data and want something powerful but still fairly fast to train.

- **When to use DL:** Use deep learning if:  
  1) You have a lot of training data (e.g., years of hourly data, or thousands of related series). DL models typically need this to outperform simpler ones.  
  2) The pattern is very complex (maybe multiple seasonalities, nonlinear trends, interactions between several signals) that simpler models can’t easily capture.  
  3) You need to forecast many series jointly and want a single model (like a global model for all time series – RNNs/transformers excel at learning across series).  
  4) You have the resources/time to tune them.  

  For example, if forecasting traffic in all roads in a city, a graph neural network or LSTM that takes all sensors might outperform one ARIMA per sensor by learning spatial patterns. But if you only care about one road’s daily traffic for 2 years, probably an ARIMA or Prophet is enough.  

  Also consider that sometimes combining ML and DL can yield diminishing returns – e.g., in M4 competition, pure statistical combos were on par with ML/DL combos for many frequencies. So don’t jump to DL unless needed.

- **Cost of complexity:** Simpler models are easier to maintain and explain. If Prophet gives acceptable accuracy, you might prefer it over a black-box LSTM even if LSTM is slightly better, because Prophet yields interpretable components (trend, weekly effect, yearly effect), and you can explain those to stakeholders. Also, simpler models run faster, which is beneficial if you need quick updates.

- **Interpretability:** This is crucial in practice. Some ways to interpret:  
  - *Statistical models (ARIMA, ETS):* They have parameters that can be interpreted (AR coefficients show how past values influence current, etc.), though for laypersons these are still abstract. But e.g., AR(1) coefficient of 0.8 means strong persistence, etc. More interpretable are Prophet’s components plots (you can show “this is the weekly pattern the model learned”, “here’s the yearly effect”) – that resonates with people (“Oh, Monday boost of 5%, December decline etc.”).  
  - *Tree-based models:* You can examine feature importances (e.g., “the model relies most on last week’s sales and the marketing spend feature”). You can also use SHAP (Shapley Additive Explanations) to get local interpretability: for a given prediction, SHAP can tell how each feature contributed to pushing the forecast up or down. For example, SHAP could reveal “This week’s forecast was higher mainly because last week’s sales were high and because there’s a holiday (these added to forecast), while an overall downward trend slightly reduced it”. That’s powerful in explaining specific forecasts. Many libraries (SHAP package) support XGBoost and even neural nets partially.  
  - *Neural nets:* Harder, but techniques exist: attention weights in Transformers or TFT can highlight which time steps or features the model attended to for that prediction. E.g., TFT can output that for this week’s forecast, it paid most attention to sales 4 weeks ago and to the “promotion” feature now. That gives insight. If using LSTMs without attention, you have limited interpretability (some try to do sensitivity analysis – vary an input and see effect). For CNNs/TCNs, you might examine which filters activate but that’s research-level.  
  - *Partial dependence plots (for tree or any model):* e.g., plot predicted value as a function of a feature, holding others constant. For instance, partial dependence on “day of week” might show Tuesday baseline is 100 units, Saturday baseline 150 units – so you glean a learned weekly pattern from an ML model. Similarly, partial dependence on “price” might show as price increases, predicted sales drop – giving elasticity.  

- **Model selection process:** Usually:  
  - Try a few candidate models (with appropriate tuning) – e.g., ARIMA, Prophet, LightGBM, maybe a simple LSTM.  
  - Use time series CV or a holdout set to compare their performance (MAE, MAPE, etc.).  
  - Also weigh practical factors: If two models have similar error, choose the simpler or more interpretable. Or if one model takes hours to train and you need frequent retrains, maybe skip it.  
  - Sometimes ensemble them as we discussed – the “model selection” could be that the best is an ensemble of 2 models.  
  - Validate that chosen model’s residuals look random (to ensure no obvious missed pattern).  

- **When things go wrong:** If none of the tried models are good enough, examine residuals and domain knowledge. Perhaps there’s an external factor not included (economy, competitor, etc.). Or the data may be too noisy to forecast accurately (there is irreducible error – like trying to forecast stock prices, where a large part is random). Acknowledge limits – sometimes the best model is one that also quantifies uncertainty well.

- **Communication:** Interpretability helps building trust. Often, forecast users want to know "why is Q3 predicted to drop?" A model like Prophet might answer: "because trend is leveling plus no holiday boost unlike Q2" from its components. A pure ML might be harder to explain succinctly. You might use SHAP summary to say "the model’s predictions are most driven by these factors in general." And track those factors over time to explain a specific change.

- **Balance accuracy vs interpretability:** In some cases a slightly less accurate but more interpretable model is preferable (especially in fields like healthcare or finance where decisions need justification). However, if accuracy is paramount (e.g. automated trading, or where small error improvements save big money in supply chain), then a black-box might be acceptable with some post-hoc interpretation attempts.

- **Modern tools for interpretability:**  
  - SHAP (works well with tree models, also with deep via DeepExplainer for TensorFlow).  
  - LIME (Local Interpretable Model-agnostic Explanations): creates local linear approximations of the model around a prediction to explain it. Could be used for forecasting to interpret a particular forecast outcome by perturbing inputs.  
  - Partial Dependence / ICE (Individual Conditional Expectation) plots to see feature effects.  
  - For global understanding: decision tree surrogate – train a simple decision tree on the model’s predictions to approximate its logic (not always accurate but can reveal structure).  
  - If using a global model for many series, sometimes analyzing one series at a time can help (e.g., how did the model fit series A vs series B).  

---

In summary: Choose the simplest model that meets accuracy requirements. Start with statistical/Prophet, escalate to ML, then DL if needed. Use ensembles if beneficial. Always consider how to explain the model’s behavior to stakeholders – if you can’t, be ready with approximate explanations (like "according to model, factor X and Y are driving the changes"). Over time, you might maintain a few models and revisit if conditions change. Model selection is not one-time: as more data comes or new patterns (like COVID in 2020 c
