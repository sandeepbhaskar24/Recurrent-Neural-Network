import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Training data
X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
])

y = np.array([
    4,
    5,
    6,
    7,
    8,
    9,
    10
])

# Reshape:
# (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# creating RNN
model = Sequential([
    SimpleRNN(16, activation='tanh', input_shape=(3, 1)),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

# training RNN
model.fit(X,
          y,
          epochs=1000,
          verbose=1)

# Test
test = np.array([7, 8, 9]).reshape(1, 3, 1)
prediction = model.predict(test, verbose=1)

print("Predicted value:", prediction[0][0])