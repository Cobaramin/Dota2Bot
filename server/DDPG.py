import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
# from Memory import Memory
from ReplayBuffer import ReplayBuffer

from config import GAMMA, LRA, LRC, TAU, action_dim, state_dim, BUFFER_SIZE, BATCH_SIZE

class DDPG:

    def __init__(self):

        # Variable Definition
        self.ep = 0
        self.replace_freq = 1
        self.save_freq = 25
        self.save_path = os.path.dirname(os.path.abspath(__file__)) + '/weights/'

        # Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Session Setup & graph
        self.sess = tf.Session(config=config)
        from keras import backend as K
        K.set_session(self.sess)
        self.tf_graph = tf.get_default_graph()

        # Network
        self.actor = ActorNetwork(self.sess, self.tf_graph, state_dim, action_dim, TAU, LRA)
        self.critic = CriticNetwork(self.sess, self.tf_graph, state_dim, action_dim, TAU, LRC)
        self.memory = ReplayBuffer(BUFFER_SIZE)

    def get_model(self):
        # helper function
        def get_json_each_model(weights):
            num_dense = self.num_layers - 1
            data = []
            for i in range(0, 2 * num_dense, 2):
                data.append({
                    'dense': i / 2,
                    'weights': {'shape': weights[i].shape, 'value': weights[i].tolist()},
                    'bias': {'shape': weights[i + 1].shape, 'value': weights[i + 1].tolist()}
                })
            return {'len': num_dense, 'data': data}

        actor_weight = self.actor.model.get_weights()
        return {'weights': [dense.tolist() for dense in actor_weight],
                'replace': (self.ep % self.replace_freq) == 0,
                'explore': 2,
                'boot_strap': 0}

    def update(self, data, train_indicator = 1):
        # Mock data
        data = [{'s': [1,1,1], 'a': [1,0.1], 'r': 2, 's1': [1,2,1], 'done': 0},
                {'s': [2,2,2], 'a': [-1,-0.8], 'r': 5, 's1': [1,0,1], 'done': 1}]

        # Duplicate Episode
        if data['ep'] == self.ep:
            print("Duplicate ep %d" % self.ep, file=sys.stderr)
            return
        self.ep = data['ep']
        print("Ep: %d" % self.ep, file=sys.stderr)

        # Data extraction
        s_t = np.array([d['s'] for d in data], dtype=np.float32)
        a_t = np.array([d['a'] for d in data], dtype=np.float32)
        r_t = np.array([d['r'] for d in data], dtype=np.float32)
        s_t1 = np.array([d['s1'] for d in data], dtype=np.float32) # add this
        done = np.array([d['done'] for d in data], dtype=np.bool) # add this

        self.memory.add_multiple(zip(s_t, a_t, r_t, s_t1, done))

        loss = 0
        if(self.memory.count() > BATCH_SIZE):
            # Start training
            states, actions, rewards, next_states, dones = self.memory.getBatch(BATCH_SIZE)

            with self.tf_graph.as_default():
                target_q_values = self.critic.target_model.predict([next_states, self.actor.target_model.predict(next_states)])

            y_t = np.zeros(actions.shape)
            for i in range(BATCH_SIZE):
                if dones[i]:
                    y_t[i] = rewards[i]
                else:
                    y_t[i] = rewards[i] + GAMMA * target_q_values[i]

            if (train_indicator):
                with self.tf_graph.as_default():
                    loss += self.critic.model.train_on_batch([states, actions], y_t)
                    a_for_grad = self.actor.model.predict(states)
                    grads = self.critic.gradients(states, a_for_grad)
                    self.actor.train(states, grads)
                    self.actor.target_train()
                    self.critic.target_train()

        print("Episode", self.ep, "Loss", loss)

    def load(self, actor_file, critic_file):
        try:
            with self.tf_graph.as_default():
                self.actor.model.load_weights(self.save_path + actor_file)
                self.actor.target_model.load_weights(self.save_path + actor_file)
                self.critic.model.load_weights(self.save_path + critic_file)
                self.critic.target_model.load_weights(self.save_path + critic_file)
            print('already loaded weights from file "%s" & "%s"' % (actor_file, critic_file))
        except:
            print('cannot load weights')

    def dump(self):
        self.actor.model.save_weights(self.save_path + 'actor_model_%d.hdf5' % self.ep, overwrite=False)
        self.critic.model.save_weights(self.save_path + 'critic_model_%d.hdf5' % self.ep, overwrite=False)
        print('already saved weights to "%s"' % self.save_path)
