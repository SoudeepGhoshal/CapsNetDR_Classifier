import os
import tensorflow as tf
from keras.src.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.optimizers import Adam
from keras.src.utils import plot_model
from tensorflow.keras import Model, layers
from capsnet import CapsuleLayer
from preprocess import gen_train_sets

MODEL_PATH = 'model/capsnet.keras'
MODEL_ARCH_PATH = 'model/model_architecture.png'

class CapsNet(Model):
    def __init__(self, **kwargs):  # Add **kwargs to accept extra arguments
        super(CapsNet, self).__init__(**kwargs)  # Pass them to the parent
        self.conv1 = layers.Conv2D(256, 9, activation='relu')
        self.primary_caps_conv = layers.Conv2D(256, 9, strides=2, activation='relu')
        self.digit_caps = CapsuleLayer(num_capsules=10, capsule_dim=16)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.primary_caps_conv(x)
        x = tf.reshape(x, [-1, 6*6*32, 8])
        x = self.digit_caps(x)
        return tf.norm(x, axis=-1)

def margin_loss(y_true, y_pred):
    m_plus, m_minus, lambda_ = 0.9, 0.1, 0.5
    T = y_true
    L = T * tf.square(tf.maximum(0., m_plus - y_pred)) + \
        lambda_ * (1 - T) * tf.square(tf.maximum(0., y_pred - m_minus))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

def train():
    x_train, y_train, x_test, y_test = gen_train_sets()

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )
    callbacks = [checkpoint, early_stopping, reduce_lr]

    model = CapsNet()
    model.compile(optimizer=Adam(learning_rate=0.001), loss=margin_loss, metrics=['accuracy'])
    model.summary()

    plot_model(model, to_file=MODEL_ARCH_PATH, show_shapes=True, show_layer_names=True)

    hist = model.fit(x_train, y_train,
                     batch_size=128,
                     epochs=10,
                     validation_data=(x_test, y_test),
                     callbacks=callbacks)
    model.save(MODEL_PATH)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc}")

if __name__ == '__main__':
    train()