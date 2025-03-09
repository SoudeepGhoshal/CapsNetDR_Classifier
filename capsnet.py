import tensorflow as tf
from tensorflow.keras import layers

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, routing_iterations=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iterations = routing_iterations

    def build(self, input_shape):
        self.input_num_capsules = input_shape[1]  # e.g., 1152
        self.input_capsule_dim = input_shape[2]  # e.g., 8
        self.W = self.add_weight(
            shape=[self.input_num_capsules, self.num_capsules, self.input_capsule_dim, self.capsule_dim],
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs_reshaped = tf.reshape(inputs, [batch_size, self.input_num_capsules, self.input_capsule_dim])
        predictions = tf.einsum('bni,njkl->bnjl', inputs_reshaped, self.W)  # [batch, input_caps, num_caps, capsule_dim]

        b = tf.zeros(shape=[batch_size, self.input_num_capsules, self.num_capsules])
        for i in range(self.routing_iterations):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(predictions * tf.expand_dims(c, axis=-1), axis=1)  # [batch, num_caps, capsule_dim]
            v = self.squash(s)  # [batch, num_caps, capsule_dim]
            if i < self.routing_iterations - 1:
                agreement = tf.einsum('bnjl,bjl->bnj', predictions, v)  # [batch, input_caps, num_caps]
                b += agreement

        return v

    def squash(self, s):
        squared_norm = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
        scale = squared_norm / (1.0 + squared_norm) / tf.sqrt(squared_norm + 1e-7)
        return scale * s