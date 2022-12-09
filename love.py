# Imports
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# NN model
model = Sequential()
# Hidden layer (1st)
model.add(Dense(32, input_dim=25, activation='relu'))
# Output layer
model.add(Dense(4, activation='softmax'))
# Loss function, optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Training data
x = np.array([
    # L
    [[1,0,0,0,0],
     [1,0,0,0,0],
     [1,0,0,0,0],
     [1,0,0,0,0],
     [1,1,1,1,1]],

    # O
    [[1,1,1,1,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [1,1,1,1,1]],

    # V
    [[1,0,0,0,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [0,1,1,1,0],
     [0,0,1,0,0]],

    # E
    [[1,1,1,1,1],
     [1,0,0,0,0],
     [1,1,1,1,1],
     [1,0,0,0,0],
     [1,1,1,1,1]]
    ])

# Labeled training data (one-hot encoding)
y = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
])

# To visualize
import matplotlib.pyplot as plt
plt.imshow(x[0])
plt.show()

# Reshape as 2 dimension
x = x.reshape(4, 25)

# Pass the data x, y and train 10 times
model.fit(x, y, epochs=10)

# Predict
predicts = model.predict(x[0].reshape(-1, 25))

# Result
print(predicts[0])
