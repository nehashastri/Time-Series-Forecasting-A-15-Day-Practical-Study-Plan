# Day 4: State-of-the-Art Deep Learning Models & Forecast Strategies

# Deep Learning Models

Deep learning has made significant inroads into time series forecasting, particularly for problems involving long sequences, multiple variables, or complex patterns that are challenging to manually feature engineer. Unlike traditional models, neural networks can automatically learn representations of the data (like nonlinear combinations of lagged values, or latent factors).  

Key types of deep learning models for time series include **RNN variants** (such as LSTM/GRU), **Temporal Convolutional Networks (TCN)**, and **Transformers** specifically designed for time series. Let’s outline each and their use cases:  

---

### **RNN (Recurrent Neural Network)**

RNNs are designed to handle sequential data by maintaining a hidden “state” that is updated as it reads through the sequence.  

In plain RNNs (which are rarely used now due to limitations), at each time step, the network takes the current input and the previous state, and outputs a new state (and possibly a prediction).  

**LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** are improved RNN architectures that address the issues of standard RNNs (like vanishing/exploding gradients) by using gating mechanisms. These gates allow the network to learn what to keep in memory and what to forget over long sequences.  

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
