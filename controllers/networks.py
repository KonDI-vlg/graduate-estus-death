import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np

class ActorNetwork(keras.Model):
    def __init__(self, n_actions=3, dense1_dims=128, dense2_dims=128, l2_factor=0.01,
                 name='actor', chkpt_dir='tmp/actor'):
        super(ActorNetwork, self).__init__()
        self.dense1 = Dense(dense1_dims, activation='relu', kernel_regularizer=l2(l2_factor))
        self.dense2 = Dense(dense2_dims, activation='relu', kernel_regularizer=l2(l2_factor))
        self.actor_probs = Dense(n_actions, activation='softmax')
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_actor.weights.h5')

    def call(self, state):
        value = self.dense1(state)
        value = self.dense2(value)
        probs = self.actor_probs(value)
        return probs
    
class CriticNetwork(keras.Model):
    def __init__(self, dense1_dims=128, dense2_dims=128, l2_factor=0.01,
                 name='critic', chkpt_dir='tmp/critic'):
        super(CriticNetwork, self).__init__()
        self.dense1 = Dense(dense1_dims, activation='tanh', kernel_regularizer=l2(l2_factor))
        self.dense2 = Dense(dense2_dims, activation='tanh', kernel_regularizer=l2(l2_factor))
        self.critic_value = Dense(1, activation=None)
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_critic.weights.h5')

    def call(self, state):
        value = self.dense1(state)
        value = self.dense2(value)
        critic_value = self.critic_value(value)
        return critic_value

class Agent:
    def __init__(self, critic_dims=[128,128], actor_dims=[128,128],
                 lr=0.001, gamma=0.95, epsilon=0.1, l2_factor=0.01, n_actions=3):
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.action = 0
        self.action_space = [i for i in range(self.n_actions)]

        self.actor = ActorNetwork(dense1_dims=actor_dims[0],dense2_dims=actor_dims[1],
                                  l2_factor=l2_factor, n_actions=n_actions)
        self.critic = CriticNetwork(dense1_dims=critic_dims[0],dense2_dims=critic_dims[1],
                                    l2_factor=l2_factor)
        
        self.actor.compile(optimizer=Adam(learning_rate=lr))
        self.critic.compile(optimizer=Adam(learning_rate=lr))

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        state = tf.convert_to_tensor([observation])
        probs = self.actor(state)

        action = tf.random.categorical(tf.math.log(probs), 1)
        self.action = action.numpy()[0, 0]

        return self.action

    def save_models(self):
        print("---- saving models ----")
        os.makedirs(os.path.dirname(self.actor.checkpoint_file), exist_ok=True)
        self.actor.save_weights(self.actor.checkpoint_file)
        os.makedirs(os.path.dirname(self.critic.checkpoint_file), exist_ok=True)
        self.critic.save_weights(self.critic.checkpoint_file)

    def load_models(self):
        print("---- loading models ----")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)


    def train_step(self, state, reward, next_state, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        if tf.reduce_any(tf.math.is_nan(state)) or tf.reduce_any(tf.math.is_nan(next_state)) or tf.math.is_nan(reward):
            print("Input contains NaN")
            return

        with tf.GradientTape(persistent=True) as tape:
            state_value = tf.squeeze(self.critic(state))
            next_state_value = tf.squeeze(self.critic(next_state))
            probs = self.actor(state)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma * next_state_value * (1 - int(done)) - state_value

            critic_loss = tf.math.square(delta)

            actor_loss = -log_prob * delta
            actor_loss = tf.math.reduce_mean(actor_loss)

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        del tape                 