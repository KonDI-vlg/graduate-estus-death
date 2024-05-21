import numpy as np
import tensorflow as tf
from keras import layers, models
import json


class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.index = 0
        self.size = 0

    def add_experience(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample_batch(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
        return batch

    def save_buffer(self, path):
        buffer_data = {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_states': self.next_states[:self.size],
            'dones': self.dones[:self.size],
            'index': self.index,
            'size': self.size
        }
        np.savez(path, **buffer_data)

    def load_buffer(self, path):
        buffer_data = np.load(path)
        self.states[:len(buffer_data['states'])] = buffer_data['states']
        self.actions[:len(buffer_data['actions'])] = buffer_data['actions']
        self.rewards[:len(buffer_data['rewards'])] = buffer_data['rewards']
        self.next_states[:len(buffer_data['next_states'])] = buffer_data['next_states']
        self.dones[:len(buffer_data['dones'])] = buffer_data['dones']
        self.index = buffer_data['index']
        self.size = buffer_data['size']


class ActorNetwork(tf.keras.Model):
    def __init__(self, action_size = 3):
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


class DDPGAgent:
    def __init__(self, action_dim=2, state_dim=3, layer_dims_actor=[128, 128], actor_activation='relu',
                 layer_dims_critic=[128, 128], actor_lr=1e-5, critic_lr=1e-4, gamma=0.99, tau=1e-3,
                 buffer_size=100000, batch_size=32, start_size=1000, noise_factor=0.1,
                 rand_act_prob=0.3):

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.layer_dims_actor = layer_dims_actor
        self.actor_activation = actor_activation
        self.layer_dims_critic = layer_dims_critic
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.start_size = start_size
        self.noise_factor = noise_factor
        self.rand_act_prob = rand_act_prob
        self.curr_noise_factor = self.noise_factor
        self.curr_rand_act_prob = self.rand_act_prob
        self._init_networks()
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    def _init_networks(self):
        self.actor = ActorNetwork()
        self.critic = CriticNetwork()
        self.target_actor = ActorNetwork()
        self.target_critic = CriticNetwork()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        self.actor.compile(tf.keras.optimizers.Adam(learning_rate=self.actor_lr))
        self.critic.compile(tf.keras.optimizers.Adam(learning_rate=self.critic_lr))
        self.target_actor.compile(tf.keras.optimizers.Adam(learning_rate=self.actor_lr))
        self.target_critic.compile(tf.keras.optimizers.Adam(learning_rate=self.critic_lr))

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        rewards = tf.cast(rewards, tf.float32)  # Приведение к float32, если необходимо
        dones = tf.cast(dones, tf.float32)
        with tf.GradientTape() as tape:

            # actions = tf.expand_dims(actions, axis=0)
            # actions = tf.cast(actions, tf.float32)
            # actions = tf.reshape(actions, [-1, 1])
            target_actions = self.target_actor(next_states)
            new_critic_value = tf.squeeze(self.target_critic(next_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions),1)
            target = rewards + self.gamma * new_critic_value * (1 - dones)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables
        ))

        with tf.GradientTape() as tape:
            next_actions = self.actor(states)
            actor_loss = - self.critic(states, next_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables
        ))
        self.update_target_networks()

    @tf.function
    def update_target_networks(self):
        for target, current in zip(self.target_actor.trainable_variables, self.actor.trainable_variables):
            target.assign(self.tau * current + (1 - self.tau) * target)

        for target, current in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            target.assign(self.tau * current + (1 - self.tau) * target)

    def get_action(self, state, eval=False):
        if not eval:
            if np.random.random() < self.curr_rand_act_prob:
                action = np.random.uniform(self.clip[0], self.clip[1], self.action_dim)
            else:
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                action = self.actor(state)[0]
                action += np.random.normal(0, self.curr_noise_factor, size=self.action_dim)
                action = np.clip(action, self.clip[0], self.clip[1])
            return action
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state)[0]
        return np.array(action)

    def save_model_weights(self, actor_path, critic_path, t_actor_path, t_critic_path, hyperparams_path):
        hyperparams = {
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'layer_dims_actor': self.layer_dims_actor,
            'actor_activation': self.actor_activation,
            'layer_dims_critic': self.layer_dims_critic,
            'clip': self.clip,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'curr_actor_lr': float(self.actor.optimizer.learning_rate),
            'curr_critic_lr': float(self.critic.optimizer.learning_rate),
            'gamma': self.gamma,
            'tau': self.tau,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'start_size': self.start_size,
            'noise_factor': self.noise_factor,
            'rand_act_prob': self.rand_act_prob,
            'curr_noise_factor': self.curr_noise_factor,
            'curr_rand_act_prob': self.curr_rand_act_prob
        }

        with open(hyperparams_path, 'w') as file:
            json.dump(hyperparams, file)
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        self.target_actor.save_weights(t_actor_path)
        self.target_critic.save_weights(t_critic_path)

    def _build_models(self):
        build_state = tf.convert_to_tensor(np.zeros((1, self.state_dim)), dtype=tf.float32)
        build_action = tf.convert_to_tensor(np.zeros((1, self.action_dim)), dtype=tf.float32)

        _ = self.actor(build_state)
        _ = self.critic(build_state, build_action)
        _ = self.target_actor(build_state)
        _ = self.target_critic(build_state, build_action)

    def load_data(self, actor_path, critic_path, t_actor_path, t_critic_path, params_path):
        with open(params_path) as f:
            params = json.load(f)

        self.action_dim = params['action_dim']
        self.state_dim = params['state_dim']
        self.layer_dims_actor = params['layer_dims_actor']
        self.actor_activation = params['actor_activation']
        self.layer_dims_critic = params['layer_dims_critic']
        self.clip = params['clip']
        self.actor_lr = params['actor_lr']
        self.critic_lr = params['critic_lr']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch_size']
        self.start_size = params['start_size']
        self.noise_factor = params['noise_factor']
        self.rand_act_prob = params['rand_act_prob']
        self.curr_noise_factor = params['curr_noise_factor']
        self.curr_rand_act_prob = params['curr_rand_act_prob']

        self._init_networks()
        self._build_models()

        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        self.target_actor.load_weights(t_actor_path)
        self.target_critic.load_weights(t_critic_path)

        self.actor.optimizer.learning_rate = params['curr_actor_lr']
        self.critic.optimizer.learning_rate = params['curr_critic_lr']
        self.target_actor.optimizer.learning_rate = params['curr_actor_lr']
        self.target_critic.optimizer.learning_rate = params['curr_critic_lr']