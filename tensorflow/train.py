import tensorflow as tf
from model import create_model

# Define a simple dataset for demonstration
def create_dummy_dataset():
    X = tf.random.normal((100, 32, 32, 3))  # 100 samples of 32x32 RGB images
    y = tf.random.uniform((100,), minval=0, maxval=10, dtype=tf.int32)  # 100 labels for 10 classes
    return tf.data.Dataset.from_tensor_slices((X, y)).batch(10)

# Initialize dataset
dataset = create_dummy_dataset()

# Initialize model, loss function, and optimizer
model = create_model()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Training loop
def train_model(epochs=1):
    model.fit(dataset, epochs=epochs)

if __name__ == "__main__":
    train_model()
