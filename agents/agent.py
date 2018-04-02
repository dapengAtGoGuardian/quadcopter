""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

mostly borrowed from https://github.com/pemami4911/deep-rl/tree/master/ddpg
original author: Patrick Emami

certainly i didn't change its network structure-- i guess all DDPG implementations use the same network structure;
what i changed is some implementation details.
"""

import tensorflow as tf
import numpy as np
import tflearn
import argparse
import pprint as pp
from collections import deque
import random
import numpy as np

from physics_sim import PhysicsSim

replay_buffer_size = 1024 # * 1024
state_dim = 12
action_dim = 4
action_bound = 1000
actor_lr = 0.005
critic_lr = 0.05
gamma = 1
actor_tau = 0.1 # the bigger, the faster to move to new value; the smaller, the more stable in later stage
critic_tau = 0.1
batch_size = 32

class ReplayBuffer(object):
    def __init__(self):
        self.buffer = deque(maxlen=replay_buffer_size)
        random.seed(6666)

    def add(self, s, a, r, t, s2):
        self.buffer.append((s, a, r, t, s2))

    def size(self):
        return len(self.buffer)

    def sample_batch(self):
        '''
        batch = random.sample(self.buffer, batch_size)
        s_batch, a_batch, r_batch, t_batch, s2_batch = zip(*batch)
        return s_batch, a_batch, r_batch, t_batch, s2_batch
        '''
        return zip(*random.sample(self.buffer, batch_size))

    def clear(self):
        self.buffer.clear()


class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        variable_start = len(tf.trainable_variables())
        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.network_params = tf.trainable_variables()[variable_start:]

        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[len(self.network_params) + variable_start:]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) +
                tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        # Combine the gradients here
        '''
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        '''
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient/self.batch_size)
        '''
        self.actor_gradients = tf.scalar_mul(
            1/self.batch_size,
            tf.gradients(self.scaled_out, self.network_params, -self.action_gradient/self.batch_size))
        '''
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init=w_init)
        scaled_out = tf.multiply(out, action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={self.inputs: inputs, self.action_gradient: a_gradient})

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={self.inputs: inputs})

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={self.target_inputs: inputs})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
        '''
        for i in range(len(self.target_network_params)):
            self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) +
                tf.multiply(self.target_network_params[i], 1. - self.tau))
        '''


class CriticNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        variable_start = len(tf.trainable_variables())
        self.inputs, self.action, self.out = self.create_critic_network()
        self.network_params = tf.trainable_variables()[variable_start:]

        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + variable_start):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) +
                tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action) # shape === [batch_size, a_dim]

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        #net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        net = tflearn.activations.relu(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b)

        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train_and_get_out(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={self.inputs: inputs, self.action: action})

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={self.target_inputs: inputs, self.target_action: action})

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={self.inputs: inputs, self.action: actions})

    def update_target_network(self):
        #self.sess.run(self.update_target_network_params)
        for i in range(len(self.target_network_params)):
            self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) +
                tf.multiply(self.target_network_params[i], 1. - self.tau))
        '''
        for target_network_param in range(elf.target_network_params):
            target_network_param.assign(
                tf.multiply(self.network_params[i], self.tau) +
                tf.multiply(target_network_param, 1. - self.tau))
        '''


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        #self.x_repv = self.x0 or np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# ===========================
#   Agent Training
# ===========================

def train(sess, task, actor, critic, actor_noise, epoches, max_steps):

    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    replay_buffer = ReplayBuffer()

    # --original author's comment:
    # Needed to enable BatchNorm. This hurts the performance on Pendulum but could be useful in other environments.
    tflearn.is_training(True)

    qmaxs = []
    xyzs = []
    for i in range(1, epoches+1):
        s = task.reset()
        replay_buffer.clear()
        qmax = -np.inf
        min_distance = np.inf

        for j in range(1, max_steps+1):
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) # + actor_noise()
            s2, r, terminal = task.step(a[0])
            replay_buffer.add(
                #np.reshape(s, (state_dim,)), np.reshape(a, (actor_dim,)), r, terminal, np.reshape(s2, (state_dim,)))
                s.reshape((state_dim,)), a.reshape((action_dim,)), r, terminal, s2.reshape((state_dim,)))

            print('epoch', i, j, ' -->', s2.reshape((state_dim,))[:3], end='\r', flush=True)
            if replay_buffer.size() >= batch_size:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch()

                s2_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))
                y_i = []
                for k in range(batch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * s2_q[k])
                predicted_q_value, _ = critic.train_and_get_out(s_batch, a_batch, np.reshape(y_i, (batch_size, 1)))

                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                actor.update_target_network()
                critic.update_target_network()

                amax = np.amax(predicted_q_value)
                qmax = max(qmax, amax)

            s = s2
            if terminal:
                break
        x, y, z = s[:3]
        print('epoch', i, 'steps', j, 'q max', qmax, 'x, y, z = {:.2f}, {:.2f}, {:.2f}          '.format(x, y, z),
              end='\n', flush=True)
        qmaxs.append(qmax)
        xyzs.append((x, y, z))
    #print(qmaxs)
    #print(xyzs)
    return qmaxs, xyzs

def main(task, epoches=10, max_steps=128):

    with tf.Session() as sess:
        np.random.seed(6666)
        tf.set_random_seed(6666)
        actor = ActorNetwork(sess, state_dim, action_dim, actor_lr, actor_tau, batch_size)
        critic = CriticNetwork(sess, state_dim, action_dim, critic_lr, critic_tau, gamma)
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        return train(sess, task, actor, critic, actor_noise, epoches, max_steps)
