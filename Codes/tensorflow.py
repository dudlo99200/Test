import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Load Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess the Data
x_train = x_train / 255.0  # Normalize to [0, 1]
x_test = x_test / 255.0

# 3. Build the Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # Flatten 28x28 images to 1D
    layers.Dense(128, activation='relu'),   # Hidden layer
    layers.Dense(10, activation='softmax')  # Output layer (10 classes)
])

# 4. Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the Model
model.fit(x_train, y_train, epochs=5, batch_size=64)

# 6. Evaluate the Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
