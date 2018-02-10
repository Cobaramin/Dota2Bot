import os
import tensorflow as tf

from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam

from Memory import Memory

# test
import numpy as np
from sklearn.datasets.samples_generator import make_circles
from keras.optimizers import SGD

class Model:

    def __init__(self):

        # Variable Definition
        self.ep = 0
        self.explore = 5
        self.boot_strap = 0
        self.boot_strap_freq = 5
        self.replace_freq = 10
        self.target_replace_freq = 10
        self.save_freq = 25
        self.save_path = os.path.dirname(os.path.abspath(__file__)) + '/weights/'

        # Net Variable
        self.net_layers = [2, 4, 1]
        self.num_layers = len(self.net_layers)
        self.learning_rate = 0.01

        # Network
        self.policy_network = Net(
            net_layers=self.net_layers,
            learning_rate=self.learning_rate)

        self.memory = Memory()

    def get_model(self):
        weights = self.policy_network.get_weights()
        num_dense = self.num_layers - 1

        data = []
        for i in range(0, 2 * num_dense, 2):
            data.append({
                'dense': i / 2,
                'weights': {'shape': weights[i].shape, 'value': weights[i].tolist()},
                'bias': {'shape': weights[i + 1].shape, 'value': weights[i + 1].tolist()}
            })

        return {'len': num_dense, 'data': data}

    def update(self, data):
        self.policy_network.train()

    def load(self, file):
        try:
            self.policy_network = Net(self.net_layers, self.learning_rate).get_model()
            self.policy_network.load_weights(self.save_path + file, by_name=True)
            print('already loaded weights from file "%s"' % file)
        except OSError:
            print('File does not exist')

    def dump(self):
        if(not self.policy_network):
            self.policy_network = Net(self.net_layers, self.learning_rate).get_model()
            print('created new model')

        self.policy_network.save_weights(self.save_path + 'aaaa.hdf5')
        print('saved weights')
        print(self.save_path)


class Net:

    def __init__(self, net_layers, learning_rate):
        self.tf_session = K.get_session() # this creates a new session since one doesn't exist already.
        self.tf_graph = tf.get_default_graph()

        # self.sess = tf.Session()
        self.net_layers = net_layers
        self.learning_rate = learning_rate

        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                self.model = self.new_model()

    def new_model(self):
        # K.set_session(self.sess)

        # model = Sequential()
        # model.add(Dense(self.net_layers[1], input_dim=self.net_layers[0], activation="relu"))
        # for num_node in self.net_layers[2:-1]:
        #     model.add(Dense(num_node, activation="relu"))
        # model.add(Dense(self.net_layers[-1]))
        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))

        model = Sequential()
        model.add(Dense(4, input_shape=(2,), activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        X, y = make_circles(n_samples=1000,
                    noise=0.1,
                    factor=0.2,
                    random_state=0)

        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                self.model.fit(X, y, epochs=20)

    def get_weights(self):
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                weights = self.model.get_weights()
        return weights
