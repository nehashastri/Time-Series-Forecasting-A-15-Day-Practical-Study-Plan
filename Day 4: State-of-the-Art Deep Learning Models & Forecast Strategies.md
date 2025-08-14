# Day 4: State-of-the-Art Deep Learning Models & Forecast Strategies

# Deep Learning Models

Deep learning has made significant inroads into time series forecasting, particularly for problems involving long sequences, multiple variables, or complex patterns that are challenging to manually feature engineer. Unlike traditional models, neural networks can automatically learn representations of the data (like nonlinear combinations of lagged values, or latent factors).  

Key types of deep learning models for time series include **RNN variants** (such as LSTM/GRU), **Temporal Convolutional Networks (TCN)**, and **Transformers** specifically designed for time series. Let’s outline each and their use cases:  

---

### **RNN (Recurrent Neural Network)**

RNNs are designed to handle sequential data by maintaining a hidden “state” that is updated as it reads through the sequence.  

In plain RNNs (which are rarely used now due to limitations), at each time step, the network takes the current input and the previous state, and outputs a new state (and possibly a prediction).  

**LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** are improved RNN architectures that address the issues of standard RNNs (like vanishing/exploding gradients) by using gating mechanisms. These gates allow the network to learn what to keep in memory and what to forget over long sequences.  

> If you have many related time series (e.g., sales for 300 products, or readings from 50 sensors), **instead of building 300 separate models (one per series)** like in traditional forecasting, you can train **one deep learning model** on *all* the data at once.
> - The model might take the **series ID** (or other identifiers) as an input feature, so it knows which series a given data point belongs to.
> - Because it’s trained on all series together, it can **“share knowledge”** between them — for example, if weekends tend to have higher values across *many* products, the model can learn that general “weekend effect” and apply it to any series that shows similar behavior.

> **Traditional Model:** separate model for each series (or complicated hierarchical setups).  
> **Deep learning Model:** single “global” model that automatically learns patterns common across series.


In practice, LSTMs/GRUs can capture long-term dependencies far better than naive RNNs. For example, an LSTM can, in theory, learn a seasonal pattern 100 steps long by retaining information in its cell state.  

**Why RNN-type models are natural for time series**:  
- They process data in sequence, mimicking how the data unfolds.  
- Especially useful for long sequences and when you want to train one model on many sequences (e.g., lots of related time series).  
- Can handle **multivariate input** and **multivariate output** easily.  

**Use cases**:  
- IoT sensor readings (e.g., capturing device-specific patterns).  
- Finance (capturing temporal dependencies that ARIMA wouldn’t).  

**Cautions**:  
- Avoid overfitting by using proper sequence input windows.  
- Typically trained to predict **one step at a time** or a **sequence of future steps**.  

**Example** – A simple LSTM in Keras:  
```python
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, num_features)))
model.add(Dense(future_horizon))  # output length
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=20, batch_size=32)
```
Here:
- look_back = number of past time steps given as input.
- num_features = number of series or exogenous features at each time.
- future_horizon = 1 for a one-step forecast or more for a multi-step forecast.

---

### **Temporal Convolutional Networks (TCN)**

TCNs are a class of models that use **1D convolutions with dilations** (skipping more and more timesteps in the convolution) to capture long-range patterns in sequences.  

Instead of recurrence, they use convolutional filters that slide over the time sequence. By stacking layers with increasing dilation, TCNs can achieve a long **“receptive field”** (i.e., lookback) while being easier to train:  
- Convolutions can be parallelized.  
- Avoid RNNs’ sequential dependency during training.  

TCNs also use **causal convolutions** (ensuring output at time *t* only depends on *t* and before, not the future).  

They have been shown to often outperform RNNs like LSTMs on certain sequence tasks while being faster to train and avoiding issues like **vanishing gradients**.  

Think of TCNs like a **very deep moving average model** where the convolution filters learn which past patterns are predictive. They can also incorporate multiple series as different channels (e.g., multivariate input as multiple convolution channels).  

**Use cases**:  
- Very long sequences or high-frequency data where capturing long dependencies matters.  
- Applied in traffic forecasting, audio signal modeling, and other long-context tasks.  
- Many modern forecasting toolkits include TCN implementations (e.g., Unit8’s **Darts** library).  

**Intuition**: Suppose you have daily data with a yearly seasonality (~365 days). A TCN can have a convolution filter that effectively spans 365 days after a few layers, meaning it can learn that “the value a year ago influences today” by having a filter that aligns with that lag.  
- RNNs could also learn this, but it might be harder for them to carry information that long without forgetting.

**Advantages**:  
- TCN architecture tends to be **more stable to train** than RNNs.  
- Better convergence on noisy, complex datasets.

**Example** – A simple TCN in Keras:  
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='causal',
                 dilation_rate=1, activation='relu', input_shape=(look_back, num_features)))
model.add(Conv1D(filters=64, kernel_size=3, padding='causal',
                 dilation_rate=2, activation='relu'))
# Add more layers with increasing dilation_rate for longer range
```
With enough layers, this setup can cover a long time range.

Reference:
Bai, S., Kolter, J. Z., & Koltun, V. (2018).
"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" – Proposed TCN and showed it often outperforms LSTMs in benchmarks.

---

### **Transformers for Time Series**

Transformers revolutionized NLP by using **self-attention mechanisms** to handle long-range dependencies without recurrence. Recently, they’ve been adapted to time series. Transformers can look at an entire sequence and learn what time steps to pay attention to for forecasting.  

However, native transformers have some issues for time series (like **quadratic complexity** with sequence length, and they’re **data-hungry**). But specialized variants have been developed: examples include **Informer**, **Autoformer**, **Temporal Fusion Transformer (TFT)**, **PatchTST**, etc., each addressing certain challenges (like reducing complexity or incorporating inductive biases for time).  

These models are on the cutting edge (papers in 2020–2023). They excel particularly in **long-range forecasting** and **multivariate series** where the relationships between different series and time steps can be complex.  

- **TFT (by Google)** is designed for **interpretable multivariate forecasting** – it uses attention to figure out which predictors are important at which times.  
- **Informer** introduces a mechanism to sparsify attention for long series to be efficient.  
- **PatchTST (2023)** breaks a time series into patches (like image patches) and applies transformers to capture very long-term patterns effectively (it’s one of the current state-of-the-art for long-horizon forecasting).  

These models often outperform classical approaches on very long forecast horizons or very complex multivariate data. However, they require a lot of **data** and **computational power** to train (and tune). They’re more commonly seen in research and large-scale applications (forecasting hundreds of thousands of series, or very high-frequency data, etc.).  

> **Use case – Long-range forecasting:** Research has shown models like **PatchTST** outperform classical methods significantly for horizons like 96 steps ahead on certain benchmarks (electricity, traffic, weather data).

---

### **Other Deep Learning Models**

#### **N-BEATS / N-HiTS**  
These are neural architectures specifically crafted for **univariate forecasting** that had great success in competitions.  

- **N-BEATS (2019)** uses backward and forward residual stacks to learn trend and seasonality bases implicitly and has matched or beaten statistical methods on M4 competition data.  
- **N-HiTS (2022)** is an improved version using **hierarchical interpolation**.  
> Hierarchical Interpolation is like looking at your data at different zoom levels.

> Imagine you have daily sales data. There are short-term patterns (like day-to-day changes) and long-term patterns (like weekly or monthly trends).
> - Step 1 – Coarse view: You first look at the data in a coarser way, like weekly averages or monthly totals. This helps capture the big trends.
> - Step 2 – Fine view: You also keep the original daily data to capture small, day-to-day variations.
> - Step 3 – Combine: You make predictions at both levels. Then you “stretch” the coarse predictions back to daily values and add them together with the fine predictions (step 2).

These are less general-purpose than transformers but highly effective for pure time-series forecasting tasks.  

**Example – Long-range forecasting:**  
Forecasting traffic for the next 6 months at hourly granularity – deep models like transformers or N-BEATS can potentially pick up seasonal patterns at multiple scales (daily, weekly, yearly) automatically.

---

#### **Autoencoder-based / CNN–LSTM hybrids**  
Some approaches use **1D CNNs** to extract features and then **LSTM** to forecast, or use **autoencoders** to learn latent representations of multiple time series and then predict. 
> An autoencoder is a type of neural network used for unsupervised learning, mainly for dimensionality reduction or feature learning.

> Intuition:
> - Encoder: Compresses input data into a smaller representation (latent space). Think of it as summarizing the key information.
> - Latent space: The “compressed” version of your data. This captures the most important features.
> Decoder: Reconstructs the input from this compressed representation.

---

#### **Graph Neural Networks for Time Series**  
For forecasting many related series (like sensor networks, or traffic in roads), sometimes **graph neural nets** are used to capture relationships among series.
