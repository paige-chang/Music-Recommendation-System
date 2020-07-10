## DDPG model with actor-critic framework

import numpy as np
import random
import tensorflow as tf
from tensorflow.python.framework import ops
import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout

class Actor():
    
    '''
    Policy function approximator
    '''
    
    def __init__(self, sess, state_space_size, action_space_size, batch_size, ra_length, history_length, embedding_size, tau, learning_rate, scope='actor'):
        
        self.sess = sess
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.batch_size = batch_size
        self.ra_length = ra_length
        self.history_length = history_length
        self.embedding_size = embedding_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            # Build Actor network
            self.action_weights, self.state, self.sequence_length = self._build_net('estimator_actor')
            self.network_params = tf.trainable_variables()

            # Build target Actor network
            self.target_action_weights, self.target_state, self.target_sequence_length = self._build_net('target_actor')
            self.target_network_params = tf.trainable_variables()[len(self.network_params):] # TODO: why sublist [len(x):]? Maybe because its equal to network_params + target_network_params

            # Initialize target network weights with network weights
            self.init_target_network_params = [self.target_network_params[i].assign(self.network_params[i]) 
                                               for i in range(len(self.target_network_params))]

            # Update target network weights 
            self.update_target_network_params = [self.target_network_params[i].assign(
            tf.multiply(self.tau, self.network_params[i]) +
            tf.multiply(1 - self.tau, self.target_network_params[i]))for i in range(len(self.target_network_params))]

            # Gradient computation from Critic's action_gradients
            self.action_gradients = tf.placeholder(tf.float32, [None, self.action_space_size])
            gradients = tf.gradients(tf.reshape(self.action_weights, [self.batch_size, self.action_space_size], name = '42'),
                                     self.network_params, self.action_gradients)
            params_gradients = list(map(lambda x: tf.div(x, self.batch_size * self.action_space_size), gradients))

            # Compute ∇_a.Q(s, a|θ^µ).∇_θ^π.f_θ^π(s)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(params_gradients, self.network_params))

    def _build_net(self, scope):
        
        '''
        Build the (target) Actor network
        '''

        def gather_last_output(data, seq_lens):
            def cli_value(x, v):
                y = tf.constant(v, shape = x.get_shape(), dtype = tf.int64)
                x = tf.cast(x, tf.int64)
                return tf.where(tf.greater(x, y), x, y)

            batch_range = tf.range(tf.cast(tf.shape(data)[0], dtype = tf.int64), dtype=tf.int64)
            tmp_end = tf.map_fn(lambda x: cli_value(x, 0), seq_lens - 1, dtype = tf.int64)
            indices = tf.stack([batch_range, tmp_end], axis = 1)
            return tf.gather_nd(data, indices)

        with tf.variable_scope(scope):
            # Inputs: current state, sequence_length
            # Outputs: action weights 
            state = tf.placeholder(tf.float32, [None, self.state_space_size], 'state')
            state_ = tf.reshape(state, [-1, self.history_length, self.embedding_size])
            sequence_length = tf.placeholder(tf.int32, [None], 'sequence_length')
            cell = tf.nn.rnn_cell.GRUCell(self.embedding_size,
                                        activation = tf.nn.relu,
                                        kernel_initializer = tf.initializers.random_normal(),
                                        bias_initializer = tf.zeros_initializer())
            outputs, _ = tf.nn.dynamic_rnn(cell, state_, dtype = tf.float32, sequence_length = sequence_length)
            last_output = gather_last_output(outputs, sequence_length)
            x = tf.keras.layers.Dense(self.ra_length * self.embedding_size)(last_output)
            action_weights = tf.reshape(x, [-1, self.ra_length, self.embedding_size])

        return action_weights, state, sequence_length

    def train(self, state, sequence_length, action_gradients):
        
        '''
        Compute ∇_a.Q(s, a|θ^µ).∇_θ^π.f_θ^π(s)
        '''
        
        self.sess.run(self.optimizer,
                      feed_dict = {
                          self.state: state,
                          self.sequence_length: sequence_length,
                          self.action_gradients: action_gradients})

    def predict(self, state, sequence_length):
        
        return self.sess.run(self.action_weights,
                            feed_dict = {
                                self.state: state,
                                self.sequence_length: sequence_length})

    def predict_target(self, state, sequence_length):
        
        return self.sess.run(self.target_action_weights,
                            feed_dict = {
                                self.target_state: state,
                                self.target_sequence_length: sequence_length})

    def init_target_network(self):
        
        self.sess.run(self.init_target_network_params)

    def update_target_network(self):
        
        self.sess.run(self.update_target_network_params)
      
    def get_recommendation_list(self, ra_length, noisy_state, embeddings, target = False):
        
        '''
        Args:
          ra_length: length of the recommendation list.
          noisy_state: current/remembered environment state with noise.
          embeddings: Embeddings object.
          target: boolean to use Actor's network or target network.
        Returns:
          Recommendation List: list of embedded items as future actions.
        '''

        def get_score(weights, embedding, batch_size):
            
            '''
            Args:
            weights: w_t^k shape = (embedding_size,).
            embedding: e_i shape = (embedding_size,).
            Returns:
            score of the item i: score_i = w_t^k.e_i^T shape = (1,).
            '''

            ret = np.dot(weights, embedding.T)
            return ret

        batch_size = noisy_state.shape[0]

        # Generate w_t = {w_t^1, ..., w_t^K}
        method = self.predict_target if target else self.predict
        weights = method(noisy_state, [ra_length] * batch_size)

        # Score items
        scores = np.array([[[get_score(weights[i][k], embedding, batch_size)
                             for embedding in embeddings.get_embedding_vector()]
                            for k in range(ra_length)] for i in range(batch_size)])

        # return a_t
        return np.array([[embeddings.get_embedding(np.argmax(scores[i][k]))
                          for k in range(ra_length)] for i in range(batch_size)])

class Critic():
    
    '''
    Value function approximator
    '''
    
    def __init__(self, sess, state_space_size, action_space_size, history_length, embedding_size, tau, learning_rate, scope='critic'):
        self.sess = sess
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.history_length = history_length
        self.embedding_size = embedding_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            # Build Critic network
            self.critic_Q_value, self.state, self.action, self.sequence_length = self._build_net('estimator_critic')
            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='estimator_critic')

            # Build target Critic network
            self.target_Q_value, self.target_state, self.target_action, self.target_sequence_length = self._build_net('target_critic')
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')

            # Initialize target network weights with network weights (θ^µ′ ← θ^µ)
            self.init_target_network_params = [self.target_network_params[i].assign(self.network_params[i])
            for i in range(len(self.target_network_params))]

            # Update target network weights (θ^µ′ ← τθ^µ + (1 − τ)θ^µ′)
            self.update_target_network_params = [self.target_network_params[i].assign(
            tf.multiply(self.tau, self.network_params[i]) +
            tf.multiply(1 - self.tau, self.target_network_params[i]))
            for i in range(len(self.target_network_params))]

            # Minimize MSE between Critic's and target Critic's outputed Q-values
            self.expected_reward = tf.placeholder(tf.float32, [None, 1])
            self.loss = tf.reduce_mean(tf.squared_difference(self.expected_reward, self.critic_Q_value))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            # Compute ∇_a.Q(s, a|θ^µ)
            self.action_gradients = tf.gradients(self.critic_Q_value, self.action)

    def _build_net(self, scope):

        '''
        Build the (target) Critic network
        '''

        def gather_last_output(data, seq_lens):
            def cli_value(x, v):
                y = tf.constant(v, shape = x.get_shape(), dtype = tf.int64)
                return tf.where(tf.greater(x, y), x, y)

            this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype = tf.int64), dtype = tf.int64)
            tmp_end = tf.map_fn(lambda x: cli_value(x, 0), seq_lens - 1, dtype = tf.int64)
            indices = tf.stack([this_range, tmp_end], axis = 1)
            return tf.gather_nd(data, indices)

        with tf.variable_scope(scope):
            # Inputs: current state, current action
            # Outputs: predicted Q-value
            state = tf.placeholder(tf.float32, [None, self.state_space_size], 'state')
            state_ = tf.reshape(state, [-1, self.history_length, self.embedding_size])
            action = tf.placeholder(tf.float32, [None, self.action_space_size], 'action')
            sequence_length = tf.placeholder(tf.int64, [None], name = 'critic_sequence_length')
            cell = tf.nn.rnn_cell.GRUCell(self.history_length,
                                        activation = tf.nn.relu,
                                        kernel_initializer = tf.initializers.random_normal(),
                                        bias_initializer = tf.zeros_initializer())
            predicted_state, _ = tf.nn.dynamic_rnn(cell, state_, dtype = tf.float32, sequence_length = sequence_length)
            predicted_state = gather_last_output(predicted_state, sequence_length)

            inputs = tf.concat([predicted_state, action], axis = -1)
            layer1 = tf.layers.Dense(32, activation = tf.nn.relu)(inputs)
            layer2 = tf.layers.Dense(16, activation = tf.nn.relu)(layer1)
            critic_Q_value = tf.layers.Dense(1)(layer2)
            return critic_Q_value, state, action, sequence_length

    def train(self, state, action, sequence_length, expected_reward):
        
        '''
        Minimize MSE between expected reward and target Critic's Q-value
        '''
        
        return self.sess.run([self.critic_Q_value, self.loss, self.optimizer],
                            feed_dict = {
                                self.state: state,
                                self.action: action,
                                self.sequence_length: sequence_length,
                                self.expected_reward: expected_reward})

    def predict(self, state, action, sequence_length):
        
        '''
        Returns Critic's predicted Q-value
        '''
        
        return self.sess.run(self.critic_Q_value,
                            feed_dict = {
                                self.state: state,
                                self.action: action,
                                self.sequence_length: sequence_length})

    def predict_target(self, state, action, sequence_length):
        
        '''
        Returns target Critic's predicted Q-value
        '''
        
        return self.sess.run(self.target_Q_value,
                            feed_dict = {
                                self.target_state: state,
                                self.target_action: action,
                                self.target_sequence_length: sequence_length})

    def get_action_gradients(self, state, action, sequence_length):
        
        '''
        Returns ∇_a.Q(s, a|θ^µ)
        '''
        
        return np.array(self.sess.run(self.action_gradients,
                            feed_dict = {
                                self.state: state,
                                self.action: action,
                                self.sequence_length: sequence_length})[0])

    def init_target_network(self):
        
        self.sess.run(self.init_target_network_params)

    def update_target_network(self):
        
        self.sess.run(self.update_target_network_params)
        
class ReplayMemory():
    
    '''
    Replay memory D in article
    '''
    
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, state, action, reward, n_state):
        self.buffer.append([state, action, reward, n_state])
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
def experience_replay(replay_memory, batch_size, actor, critic, embeddings, ra_length, state_space_size, action_space_size, discount_factor):
    
    '''
    Experience replay.
    Args:
          replay_memory: replay memory D in article.
          batch_size: sample size.
          actor: Actor network.
          critic: Critic network.
          embeddings: Embeddings object.
          state_space_size: dimension of states.
          action_space_size: dimensions of actions.
    Returns:
          Best Q-value, loss of Critic network for printing/recording purpose.
    '''

    # Sample minibatch of N transitions (s, a, r, s′)
    samples = replay_memory.sample_batch(batch_size)
    states = np.array([s[0] for s in samples])
    actions = np.array([s[1] for s in samples])
    rewards = np.array([s[2] for s in samples])
    n_states = np.array([s[3] for s in samples]).reshape(-1, state_space_size)

    # Generate a′ by target Actor network 
    n_actions = actor.get_recommendation_list(ra_length, states, embeddings, target = True).reshape(-1, action_space_size)

    # Calculate predicted Q′(s′, a′|θ^µ′) value
    target_Q_value = critic.predict_target(n_states, n_actions, [ra_length] * batch_size)

    # Set y = r + γQ′(s′, a′|θ^µ′)'
    expected_rewards = rewards + discount_factor * target_Q_value
    
    # Update Critic by minimizing (y − Q(s, a|θ^µ))²'
    critic_Q_value, critic_loss, _ = critic.train(states, actions, [ra_length] * batch_size, expected_rewards)
    
    # Update the Actor using the sampled policy gradient'
    action_gradients = critic.get_action_gradients(states, n_actions, [ra_length] * batch_size)
    actor.train(states, [ra_length] * batch_size, action_gradients)

    # Update the Critic target networks
    critic.update_target_network()

    # Update the Actor target network'
    actor.update_target_network()

    return np.amax(critic_Q_value), critic_loss

