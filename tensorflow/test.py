import tensorflow as tf
from model import create_model

# Define a simple dataset for demonstration
def create_dummy_dataset():
    X = tf.random.normal((10, 32, 32, 3))  # 10 samples of 32x32 RGB images
    y = tf.random.uniform((10,), minval=0, maxval=10, dtype=tf.int32)  # 10 labels for 10 classes
    return tf.data.Dataset.from_tensor_slices((X, y)).batch(10)

# Initialize dataset
dataset = create_dummy_dataset()

# Initialize model
model = create_model()
model.load_weights('model.h5')  # Load pre-trained model

# Testing loop
def test_model():
    loss, accuracy = model.evaluate(dataset)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

if __name__ == "__main__":
    test_model()
