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
