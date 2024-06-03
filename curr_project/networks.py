import os
import tensorflow as tf
import tensorflow.keras
import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, state_shape, n_actions):
        self.buffer_size = max_size
        self.buffer_cnt = 0
        self.state_buffer = state_shape
        self.action_buffer = n_actions
        self.states = np.zeros((self.buffer_size, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, n_actions), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.bool_)
        print(f"Buff size={self.buffer_size}")

    def add_experience(self, state, action, reward, next_state, done):
        index = self.buffer_cnt % self.buffer_size
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done

        self.buffer_cnt += 1
        #print(f"Buff cnt={self.buffer_cnt}")

    def sample_batch(self, batch_size):
        max_mem = min(self.buffer_cnt, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        states_ = self.next_states[batch]
        dones = self.dones[batch]

        return states, actions, rewards, states_, dones

class ActorNetwork(keras.Model):
    def __init__(self, n_actions=3, dense1_dims=128, dense2_dims=128, l2_factor=0.01, drop_factor=0.2, **kwargs):
        super(ActorNetwork, self).__init__(**kwargs)
        self.dense1 = Dense(dense1_dims, activation='sigmoid', kernel_regularizer=L2(l2_factor))
        self.drop1 = Dropout(drop_factor)
        self.dense2 = Dense(dense2_dims, activation='sigmoid', kernel_regularizer=L2(l2_factor))
        self.drop2 = Dropout(drop_factor)
        self.mu = Dense(n_actions, activation='softmax')

    def call(self, state):
        prob = self.dense1(state)
        prob = self.drop1(prob)
        prob = self.dense2(prob)
        prob = self.drop2(prob)

        mu = self.mu(prob)
        return mu
    
    def get_config(self):
        config = super(ActorNetwork, self).get_config()
        config.update({
            'n_actions': self.mu.units,
            'dense1_dims': self.dense1.units,
            'dense2_dims': self.dense2.units,
            'l2_factor': self.dense1.kernel_regularizer.l2,
            'drop_factor': self.drop1.rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class CriticNetwork(keras.Model):
    def __init__(self, dense1_dims=128, dense2_dims=128, l2_factor=0.01, drop_factor=0.2, **kwargs):
        super(CriticNetwork, self).__init__(**kwargs)
        self.dense1 = Dense(dense1_dims, activation='sigmoid', kernel_regularizer=L2(l2_factor))
        self.drop1 = Dropout(drop_factor)
        self.dense2 = Dense(dense2_dims, activation='sigmoid', kernel_regularizer=L2(l2_factor))
        self.drop2 = Dropout(drop_factor)
        self.q_value = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.dense1(tf.concat([state,action], axis=1))
        action_value = self.drop1(action_value)
        action_value = self.dense2(action_value)
        action_value = self.drop2(action_value)
        q_value = self.q_value(action_value)
        return q_value
    
    def get_config(self):
        config = super(CriticNetwork, self).get_config()
        config.update({
            'dense1_dims': self.dense1.units,
            'dense2_dims': self.dense2.units,
            'l2_factor': self.dense1.kernel_regularizer.l2,
            'drop_factor': self.drop1.rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Agent:
    def __init__(self, params):
        self.gamma = params['GAMMA']
        self.epsilon = params['EPSILON']
        self.tau = params['TAU']
        self.learning_rate = params['LEARNING_RATE']
        self.l2_factor = params['L2_FACTOR']

        self.actor_dims = params['ACTOR_DIMS']
        self.critic_dims = params['CRITIC_DIMS']

        self.batch_size = params['BATCH_SIZE']
        self.buffer_max_size = self.batch_size * 64
        self.n_actions = params['N_ACTIONS']
        self.memory = ReplayBuffer(self.buffer_max_size, params['STATE_SHAPE'], self.n_actions)

        self.actor = ActorNetwork(self.n_actions, self.actor_dims[0], self.actor_dims[1], self.l2_factor, params["DROPOUT"]) # self.n_actions, self.actor_dims[0], self.actor_dims[1], self.l2_factor
        self.critic = CriticNetwork(self.critic_dims[0], self.critic_dims[1], self.l2_factor, params["DROPOUT"]) # self.critic_dims[0], self.critic_dims[1], self.l2_factor
        self.target_actor = ActorNetwork(self.n_actions, self.actor_dims[0], self.actor_dims[1], self.l2_factor, params["DROPOUT"])
        self.target_critic = CriticNetwork(self.critic_dims[0], self.critic_dims[1], self.l2_factor, params["DROPOUT"])
        
        self.actor.compile(optimizer=Adam(learning_rate=self.learning_rate))
        self.critic.compile(optimizer=Adam(learning_rate=self.learning_rate))
        self.target_actor.compile(optimizer=Adam(learning_rate=self.learning_rate))
        self.target_critic.compile(optimizer=Adam(learning_rate=self.learning_rate))

        self.actor_loss = None
        self.critic_loss = None
        self.chkpt_dir='tmp'

        self.update_network_parameters(tau=1)
        
    def add_experience(self, state, action, reward, next_state, done):
        self.memory.add_experience(state,action,reward,next_state,done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            actions_probs = self.actor(state)
            action = np.random.choice(self.n_actions, p=actions_probs.numpy()[0])

        one_hot_action = np.zeros(self.n_actions)
        one_hot_action[action] = 1

        return one_hot_action

    @tf.function
    def train_step(self, state, action, reward, next_state, done):
        
        #state, action, reward, next_state, done = self.memory.sample_batch(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)
        dones = tf.convert_to_tensor(done, dtype=tf.bool)
        dones = tf.cast(dones, dtype=tf.float32)

        #target = tf.reshape(target, (-1,1))

        with tf.GradientTape() as tape:
            target_actions_probs = self.target_actor(next_states)
            critic_next_value = tf.squeeze(self.target_critic(next_states, target_actions_probs),1)
            critic_value = tf.squeeze(self.critic(states,actions),1)
            target = rewards + self.gamma * critic_next_value * (1-dones)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_target_networks()

        return critic_loss, actor_loss

    @tf.function
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)
        

    @tf.function
    def update_target_networks(self):
        for target, current in zip(self.target_actor.trainable_variables, self.actor.trainable_variables):
            target.assign(self.tau * current + (1 - self.tau) * target)

        for target, current in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            target.assign(self.tau * current + (1 - self.tau) * target)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(os.path.join(self.chkpt_dir, 'actor.keras'))
        self.critic.save(os.path.join(self.chkpt_dir, 'critic.keras'))
        self.target_actor.save(os.path.join(self.chkpt_dir, 'target_actor.keras'))
        self.target_critic.save(os.path.join(self.chkpt_dir, 'target_critic.keras'))
    
    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(os.path.join(self.chkpt_dir, 'actor.keras'), custom_objects={'ActorNetwork': ActorNetwork})
        self.critic = keras.models.load_model(os.path.join(self.chkpt_dir, 'critic.keras'), custom_objects={'CriticNetwork': CriticNetwork})
        self.target_actor = keras.models.load_model(os.path.join(self.chkpt_dir, 'target_actor.keras'), custom_objects={'ActorNetwork': ActorNetwork})
        self.target_critic = keras.models.load_model(os.path.join(self.chkpt_dir, 'target_critic.keras'), custom_objects={'CriticNetwork': CriticNetwork})