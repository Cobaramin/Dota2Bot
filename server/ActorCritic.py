import os

from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam


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
        self.net_layers = [10, 5, 3, 2]
        self.num_layers = len(self.net_layers)
        self.learning_rate = 0.01

        # Network
        self.policy_network = None

        # self.memory = Memory()
        # self.policy_net = Net(POLICY_NET)

    def get_weights(self):
        self.policy_network = Net(self.net_layers, self.learning_rate).get_model()
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
        pass

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
        self.net_layers = net_layers
        self.learning_rate = learning_rate

    def get_model(self):
        model = Sequential()
        model.add(Dense(self.net_layers[1], input_dim=self.net_layers[0], activation="relu"))
        for num_node in self.net_layers[2:-1]:
            model.add(Dense(num_node, activation="relu"))
        model.add(Dense(self.net_layers[-1]))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))

        return model


class Memory:

    def __init__(self):
        pass
