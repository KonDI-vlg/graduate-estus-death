import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np

class ActorNetwork(keras.Model):
    def __init__(self, n_actions=3, dense1_dims=64, dense2_dims=64,
                 name='actor', chkpt_dir='tmp\\actor'):
        super(ActorNetwork, self).__init__()
        self.dense1 = Dense(dense1_dims, activation='relu')
        self.dense2 = Dense(dense2_dims, activation='relu')
        self.actor_probs = Dense(n_actions, activation='softmax')
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_actor.weights.h5')

    def call(self, state):
        value = self.dense1(state)
        value = self.dense2(value)
        probs = self.actor_probs(value)
        return probs
    
class CriticNetwork(keras.Model):
    def __init__(self, dense1_dims=64, dense2_dims=64,
                 name='critic', chkpt_dir='tmp\\critic'):
        super(CriticNetwork, self).__init__()
        self.dense1 = Dense(dense1_dims, activation='relu')
        self.dense2 = Dense(dense2_dims, activation='relu')
        self.critic_value = Dense(1, activation=None)
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_critic.weights.h5')

    def call(self, state):
        value = self.dense1(state)
        value = self.dense2(value)
        critic_value = self.critic_value(value)
        return critic_value


class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions=3, dense1_dims=64, dense2_dims=64,
                 name='actor_critic', chkpt_dir='tmp\\actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.dense1_dims = dense1_dims
        self.dense2_dims = dense2_dims

        self.n_action = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac.weights.h5')

        self.dense1 = Dense(self.dense1_dims, activation='relu')
        self.dense2 = Dense(self.dense2_dims, activation='relu')
        self.critic_value = Dense(1, activation=None)
        self.actor_probs = Dense(n_actions, activation='softmax')

    def call(self, state):
        value = self.dense1(state)
        value = self.dense2(value)

        critic_value = self.critic_value(value)
        actor_probs = self.actor_probs(value)

        return critic_value, actor_probs


class Agent:
    def __init__(self, lr=0.001, gamma=0.95, epsilon=0.1, n_actions=3):
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.action = 0
        self.action_space = [i for i in range(self.n_actions)]

        self.actor = ActorNetwork(n_actions=n_actions)
        self.critic = CriticNetwork()
        
        self.actor.compile(optimizer=Adam(learning_rate=lr))
        self.critic.compile(optimizer=Adam(learning_rate=lr))

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        state = tf.convert_to_tensor([observation])
        probs = self.actor(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        self.action = action

        return action.numpy()[0]

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

        with tf.GradientTape() as tape:
            state_value, probs = self.actor_critic(state)
            next_state_value, _ = self.actor_critic(next_state)

            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma * next_state_value * (1 - int(done)) - state_value

            actor_loss = -log_prob * delta
            critic_loss = delta ** 2

            total_loss = actor_loss * 0.8 + critic_loss * 0.6

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        gradient, _ = tf.clip_by_global_norm(gradient, 1.0)
        self.actor_critic.optimizer.apply_gradients(zip(gradient, self.actor_critic.trainable_variables))
    