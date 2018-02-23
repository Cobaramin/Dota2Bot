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
            self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
            self.target_model, self.target_action, self.target_state = self.create_critic_network(
                state_size, action_size)
            self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
            self.sess.run(tf.global_variables_initializer())

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

    def create_critic_network(self, state_size, action_dim):
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w1 = Dense(self.hidden1, activation='relu')(S)
        a1 = Dense(self.hidden2, activation='linear')(A)
        h1 = Dense(self.hidden2, activation='linear')(w1)
        h2 = add([h1, a1])
        h3 = Dense(self.hidden2, activation='relu')(h2)
        V = Dense(action_dim, activation='linear')(h3)
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.lr)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S
