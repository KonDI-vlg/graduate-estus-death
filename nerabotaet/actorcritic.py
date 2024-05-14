import os
os.environ["WEBOTS_HOME"] = "/usr/local/webots"
import tensorflow as tf
from keras import layers
from camera import Camera

camera = Camera()
class ActorNetwork(tf.keras.Model):
    def __init__(self, action_size):
        super(ActorNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256,  activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.output_action = tf.keras.layers.Dense(action_size,  activation="softmax")

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_action(x)

class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.value = tf.keras.layers.Dense(1)

    def call(self, state, action):
        x = self.dense1(state)
        x = tf.concat([x, action], axis=1)
        x = self.dense2(x)
        value = self.value(x)
        return value

action_size = 3
state = camera.get_center_row()
state = tf.expand_dims(state, axis=0)
model_actor = ActorNetwork(action_size)
model_critic = CriticNetwork()
action = model_actor(state).numpy()
value = model_critic(state, action)
print(value)
print(action)

import tensorflow as tf
from tensorflow.keras import layers

class ActorCritic(tf.keras.Model):
    def __init__(self, action_size):
        super(ActorCritic, self).__init__()
        self.common = layers.Dense(128, activation='relu')
        self.actor = layers.Dense(action_size, activation='softmax')
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x), self.critic(x)
action_size =3
model = ActorCritic(action_size)
optimizer = tf.keras.optimizers.Adam(learning_rate= 0.01)

def train_step(state, action1, reward, next_state, done):
    with tf.GradientTape() as tape:
        action, value = model(state)
        _, critic_value_next = model(next_state)
        action_log_probs = tf.math.log(action[0, action1])
        td_error = reward + 0.99 * critic_value_next * (1 - done) - value
        actor_loss = -action_log_probs * tf.stop_gradient(td_error)
        critic_loss = td_error**2

    grads = tape.gradient([actor_loss, critic_loss], model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
















