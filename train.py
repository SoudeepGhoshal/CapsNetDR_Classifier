import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.mixed_precision import set_global_policy
from preprocess import get_train_data

# Enable mixed precision for faster training and reduced memory usage
set_global_policy('mixed_float16')

MODEL_PATH = 'model/capsnet.keras'
MODEL_ARCH_PATH = 'model/model_architecture.png'

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, routing_iterations=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iterations = routing_iterations

    def build(self, input_shape):
        self.input_num_capsules = input_shape[1]  # 576 capsules
        self.input_capsule_dim = input_shape[2]  # 256 dimensions
        self.W = self.add_weight(
            shape=[self.input_num_capsules, self.num_capsules, self.input_capsule_dim, self.capsule_dim],
            initializer='glorot_uniform',
            trainable=True,
            name='W',
            dtype='float32'  # Weights are typically stored in float32
        )
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs_reshaped = tf.reshape(inputs, [batch_size, self.input_num_capsules, self.input_capsule_dim])
        predictions = tf.einsum('bni,njkl->bnjl', inputs_reshaped, self.W)

        # Ensure b is in float16 to match predictions
        b = tf.zeros(shape=[batch_size, self.input_num_capsules, self.num_capsules], dtype=tf.float16)
        for i in range(self.routing_iterations):
            c = tf.nn.softmax(b, axis=2)
            # Ensure c is in float16
            c = tf.cast(c, tf.float16)
            # Ensure the multiplication is compatible
            s = tf.reduce_sum(predictions * tf.expand_dims(c, axis=-1), axis=1)
            v = self.squash(s)
            if i < self.routing_iterations - 1:
                agreement = tf.einsum('bnjl,bjl->bnj', predictions, v)
                b += tf.cast(agreement, tf.float16)  # Ensure agreement is in float16
        return v

    def squash(self, s):
        squared_norm = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + 1e-9)
        scale = squared_norm / (1.0 + squared_norm) / safe_norm
        return scale * s

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsules, self.capsule_dim)

class CapsNet(Model):
    def __init__(self, num_classes, input_shape=(64, 64, 3), **kwargs):
        super(CapsNet, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(128, 9, activation='relu', padding='valid', kernel_initializer='glorot_uniform')
        self.dropout1 = layers.Dropout(0.20)
        self.primary_caps_conv = layers.Conv2D(256, 9, strides=2, padding='valid', kernel_initializer='glorot_uniform')
        self.primary_caps_reshape = layers.Reshape((-1, 256))  # 576 capsules with 256 dimensions
        self.class_caps = CapsuleLayer(num_capsules=num_classes, capsule_dim=8)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.primary_caps_conv(x)
        x = tf.nn.relu(x)
        x = self.primary_caps_reshape(x)
        x = self.class_caps(x)
        return tf.norm(x, axis=-1)

def margin_loss(y_true, y_pred):
    m_plus, m_minus, lambda_ = 0.9, 0.1, 0.5
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    T = y_true
    L = T * tf.square(tf.maximum(0., m_plus - y_pred)) + \
        lambda_ * (1 - T) * tf.square(tf.maximum(0., y_pred - m_minus))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

def train():
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, datagen = get_train_data()

    os.makedirs('model', exist_ok=True)

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )
    callbacks = [checkpoint, early_stopping, reduce_lr]

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU detected:", physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU detected. Training on CPU.")

    model = CapsNet(num_classes)
    model.build(input_shape=(None, 64, 64, 3))

    LEARNING_RATE = 0.001
    optimizer = Adam(learning_rate=LEARNING_RATE, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss=margin_loss, metrics=['accuracy'])

    model(X_train[:1])
    model.summary()

    plot_model(model, to_file=MODEL_ARCH_PATH, show_shapes=True, show_layer_names=True)

    hist = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    model.save(MODEL_PATH)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    train()