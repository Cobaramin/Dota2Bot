import keras.backend as K
import tensorflow as tf
from keras.initializers import identity, normal
from keras.layers import Activation, Dense, Flatten, Input, Lambda, add
from keras.models import Model, Sequential, load_model, model_from_json
from keras.optimizers import Adam
from setting import cf


class CriticNetwork(object):

    def __init__(self, sess, tf_graph, state_size, action_size, tau, lr):
        self.sess = sess
        self.tf_graph = tf_graph
        self.tau = tau
        self.lr = lr
        self.action_size = action_size
        self.hidden1 = cf.CRITIC_HIDDEN1_UNITS
        self.hidden2 = cf.CRITIC_HIDDEN2_UNITS

        K.set_session(sess)

        # Now create the model
        with self.tf_graph.as_default():
            with tf.name_scope('critic_input'):
                self.state = Input(shape=[state_size], name='state')
                self.action = Input(shape=[action_size], name='action')

            with tf.name_scope('target')
                self.target = tf.placeholder(tf.float32, [None, action_size], name='target')
            variable_summaries(self.target, 'target_summary')

            with tf.name_scope('reward'):
                self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')
            variable_summaries(self.reward, 'reward_summary')

            with tf.name_scope('critic_net'):
                self.model = self.create_critic_network(self.state, self.action, state_size, action_size)
            with tf.name_scope('critic_target_net'):
                self.target_model = self.create_critic_network(self.state, self.action, state_size, action_size)

            with tf.name_scope('critic_gradient_for_actor'):
                self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update

            with tf.name_scope('critic_train'):
                self.loss = tf.losses.mean_squared_error(self.target, self.model.output)
                self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss, name='minimize_loss')
            tf.summary.scalar('mean_squared_error', self.loss)

            self.sess.run(tf.global_variables_initializer())

    def train(self, S, A, target, reward, i, sum_writer):
        merged = tf.summary.merge_all()
        loss, _, summary = self.sess.run([self.loss, self.optimize, merged], feed_dict={
            self.state: S, self.action: A, self.target: target, self.reward: reward})
        sum_writer.add_summary(summary, i)
        return loss

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, S, A, state_size, action_dim):
        w1 = Dense(self.hidden1, activation='relu')(S)
        a1 = Dense(self.hidden2, activation='linear')(A)
        h1 = Dense(self.hidden2, activation='linear')(w1)
        h2 = add([h1, a1])
        h3 = Dense(self.hidden2, activation='relu')(h2)
        V = Dense(action_dim, activation='linear')(h3)
        model = Model(inputs=[S, A], outputs=V)
        return model


def variable_summaries(var, scope_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(scope_name):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
