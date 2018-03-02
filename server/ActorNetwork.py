import keras.backend as K
import tensorflow as tf
from keras.initializers import identity, normal
from keras.layers import Dense, Flatten, Input, Lambda, concatenate
from keras.models import Model, Sequential, model_from_json
from keras.optimizers import Adam
from setting import cf


class ActorNetwork(object):

    def __init__(self, sess, tf_graph, state_size, action_size, tau, lr):
        self.sess = sess
        self.tf_graph = tf_graph
        self.tau = tau
        self.lr = lr
        self.hidden1 = cf.ACTOR_HIDDEN1_UNITS
        self.hidden2 = cf.ACTOR_HIDDEN2_UNITS

        K.set_session(sess)

        # Now create the model
        with self.tf_graph.as_default():
            with tf.name_scope('actor_input'):
                self.state_input = Input(shape=[state_size], name='state')
                self.gradients_of_action = tf.placeholder(tf.float32, [None, action_size], name='action_gradents')

            # Actor Network & Target Create
            with tf.name_scope('actor_net'):
                self.model, self.weights = self.create_actor_network(self.state_input, state_size, action_size)
            with tf.name_scope('actor_target_net'):
                self.target_model, self.target_weights = self.create_actor_network(
                    self.state_input, state_size, action_size)

            with tf.name_scope('actor_gradient'):
                self.params_grad = tf.gradients(self.model.output, self.weights, -self.gradients_of_action)
                grads = zip(self.params_grad, self.weights)

            with tf.name_scope('actor_train'):
                self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads, name='apply_gradients')

            self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state_input: states,
            self.gradients_of_action: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, S, state_size, action_dim):
        h0 = Dense(self.hidden1, activation='relu')(S)
        h1 = Dense(self.hidden2, activation='relu')(h0)
        Y = Dense(action_dim, activation='tanh')(h1)
        model = Model(inputs=S, outputs=Y)
        return model, model.trainable_weights
