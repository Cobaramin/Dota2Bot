import math
import os
import sys

import numpy as np
import tensorflow as tf

from ActorNetwork import ActorNetwork
from config import (BATCH_SIZE, BOOTSTRAP_FREQ, BUFFER_SIZE, EXPLORE, GAMMA,
                    LRA, LRC, REPLACE_FREQ, SAVE_FREQ, TAU, action_dim,
                    state_dim)
from CriticNetwork import CriticNetwork
# from Memory import Memory
from ReplayBuffer import ReplayBuffer

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DDPG:

    def __init__(self):

        # Variable Definition
        self.ep = 0
        self.replace_freq = REPLACE_FREQ
        self.save_freq = SAVE_FREQ
        self.save_path = os.path.dirname(os.path.abspath(__file__)) + '/weights/'
        self.boot_strap = 0

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
        actor_weight = self.actor.model.get_weights()
        labels = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        res = {a: b.tolist() for a, b in zip(labels, actor_weight)}
        res['replace'] = (self.ep % self.replace_freq) == 0
        # res['explore'] = np.max(0, 20 * (1 - (self.ep / float(1e5))))
        res['explore'] = EXPLORE
        res['boot_strap'] = self.boot_strap

        return res

    def update(self, data, train_indicator=1):
        # Mock data
        # data = [{'s': [1,1,1], 'a': [1,0.1], 'r': 2, 's1': [1,2,1], 'done': 0},
        #         {'s': [2,2,2], 'a': [-1,-0.8], 'r': 5, 's1': [1,0,1], 'done': 1}]

        # Duplicate Episode
        if data['ep'] == self.ep:
            print("Duplicate ep %d" % self.ep, file=sys.stderr)
            return
        self.ep = data['ep']
        print("Ep: %d" % self.ep, file=sys.stderr)

        # toggle boot_strap
        if self.ep % BOOTSTRAP_FREQ == 0:
            self.boot_strap = 100
        else:
            self.boot_strap = 0

        # Data extraction
        # a number of episodes = (len of data) - 1   --------  (1: is len episodes)
        list_episodes = [data[str(i)] for i in range(len(data) - 1)]

        a_t = np.array([e['a'] for e in list_episodes], dtype=np.float32)
        s_t = np.array([e['s'] for e in list_episodes], dtype=np.float32)
        r_t = np.array([e['r'] for e in list_episodes], dtype=np.float32)
        s_t1 = np.array([e['s1'] for e in list_episodes], dtype=np.float32)  # add this
        done = np.array([e['done'] for e in list_episodes], dtype=np.bool)  # add this

        # Normalization
        a_t /= 100.
        s_t /= 1000.
        s_t1 /= 1000.

        self.memory.add_multiple(zip(s_t, a_t, r_t, s_t1, done))

        total_loss = 0
        if(self.memory.count() > BATCH_SIZE):
            # Start training
            states, actions, rewards, next_states, dones = self.memory.getBatch(BATCH_SIZE)

            with self.tf_graph.as_default():
                target_q_values = self.critic.target_model.predict(
                    [next_states, self.actor.target_model.predict(next_states)])

            y_t = np.zeros(actions.shape)
            for i in range(BATCH_SIZE):
                if dones[i]:
                    y_t[i] = rewards[i]
                else:
                    y_t[i] = rewards[i] + GAMMA * target_q_values[i]

            if (train_indicator):
                with self.tf_graph.as_default():

                    loss = self.critic.model.train_on_batch([states, actions], y_t)  # Train critic
                    a_for_grad = self.actor.model.predict(states)
                    grads = self.critic.gradients(states, a_for_grad)  # Cal critic gradients
                    self.actor.train(states, grads)  # Train actor

                    self.actor.target_train()  # train target actor
                    self.critic.target_train()  # train target critic

                if math.isnan(loss):
                    print('State', states, 'Action', actions)
                    import pdb
                    pdb.set_trace()
                    print("Unexpected error:", sys.exc_info()[0])
                    raise

                total_loss += loss

        print("Episode", self.ep, "Buffer", self.memory.count(), '/', BUFFER_SIZE, "Loss", total_loss)

        if self.ep % self.save_freq == 0:
            self.dump()

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
