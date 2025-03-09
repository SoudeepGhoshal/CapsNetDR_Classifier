import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
from capsnet import CapsuleLayer
from train import CapsNet  # Import CapsNet from train.py


IMAGE_PATH = 'inputs/img0.png'
MODEL_PATH = 'model/capsnet.keras'


def load_and_preprocess_image(image_path):
    """Load and preprocess an image to match MNIST format: 28x28 grayscale, normalized to [0, 1]."""
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype='float32') / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array


def classify_image(model_path, image_path):
    """Load the model, preprocess the image, and predict the class."""
    # Load the saved model with all custom objects
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'CapsuleLayer': CapsuleLayer,
            'margin_loss': margin_loss,
            'CapsNet': CapsNet  # Add CapsNet to custom_objects
        }
    )

    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)

    # Predict
    predictions = model(img, training=False)  # [1, 10] (vector lengths)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]

    # Print result
    print(f"Predicted class: {predicted_class}")
    return predicted_class


# Define margin_loss (copied from train.py for model loading)
def margin_loss(y_true, y_pred):
    m_plus, m_minus, lambda_ = 0.9, 0.1, 0.5
    T = y_true
    L = T * tf.square(tf.maximum(0., m_plus - y_pred)) + \
        lambda_ * (1 - T) * tf.square(tf.maximum(0., y_pred - m_minus))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


if __name__ == '__main__':
    classify_image(MODEL_PATH, IMAGE_PATH)