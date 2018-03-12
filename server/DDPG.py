import math
import pickle
import sys
import time

import numpy as np

import tensorflow as tf
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer
from setting import cf


class DDPG:

    def __init__(self):

        # Variable Definition
        self.ep = 0
        self.replace_freq = cf.REPLACE_FREQ
        self.save_freq = cf.SAVE_FREQ
        self.WEIGHT_PATH = cf.WEIGHT_PATH

        # Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Session Setup & graph
        self.sess = tf.Session(config=config)
        from keras import backend as K
        K.set_session(self.sess)
        self.tf_graph = tf.get_default_graph()

        # Network
        self.actor = ActorNetwork(self.sess, self.tf_graph, cf.STATE_DIM, cf.ACTION_DIM, cf.TAU, cf.LRA)
        self.critic = CriticNetwork(self.sess, self.tf_graph, cf.STATE_DIM, cf.ACTION_DIM, cf.TAU, cf.LRC)
        self.memory = ReplayBuffer(cf.BUFFER_SIZE)

        # write graph
        self.timestamp = int(time.time())
        self.sum_writer = tf.summary.FileWriter(cf.TMP_PATH + '/ddpg' + str(self.timestamp), self.tf_graph)

    def get_model(self):
        actor_weight = self.actor.model.get_weights()
        labels = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        res = {a: b.tolist() for a, b in zip(labels, actor_weight)}

        res['next_ep'] = self.ep + 1
        res['replace'] = (self.ep % self.replace_freq) == 0
        res['explore'] = cf.EXPLORE
        res['boot_strap'] = 0 if self.ep % cf.BOOTSTRAP_FREQ else 100
        res['train_indicator'] = cf.TRAIN

        return res

    def update(self, data, train_indicator=1):

        # Duplicate Episode
        if data['ep'] <= self.ep:
            print("Duplicate ep %d" % self.ep, file=sys.stderr)
            return
        self.ep = data['ep']
        print("Ep: %d" % self.ep, file=sys.stderr)

        # Data extraction
        # a number of episodes = (len of data) - 1   --------  (1: is len of data['ep'])
        list_episodes = [data[str(i)] for i in range(len(data) - 1)]

        a_t = np.array([e['a'] for e in list_episodes], dtype=np.float32)
        s_t = np.array([e['s'] for e in list_episodes], dtype=np.float32)
        r_t = np.array([e['r'] for e in list_episodes], dtype=np.float32)
        s_t1 = np.array([e['s1'] for e in list_episodes], dtype=np.float32)
        done = np.array([e['done'] for e in list_episodes], dtype=np.bool)
        # print('episode size:', len(list_episodes))

        current_reward = r_t.reshape(r_t.shape[0], 1)
        # Normalization
        a_t /= 100.
        s_t /= 1000.
        s_t1 /= 1000.

        if train_indicator:
            self.memory.add_multiple(zip(s_t, a_t, r_t, s_t1, done))
            self.train(current_reward)
            if self.ep % self.save_freq == 0:
                self.dump()
        else:
            # load list of weight file
            # iterate to load each file
                # new sum_writer
                # iterate n episodes
                    # get rewards
                    # summary in TensorBoard
                        # - histrogram & box plot of weights in each episodes
                        # - stat of weights in each episodes
                        # -

            pass

    def train(self, current_reward):
        total_loss = 0
        if(self.memory.count() > cf.BATCH_SIZE):
            # Start training
            states, actions, rewards, next_states, dones = self.memory.getBatch(cf.BATCH_SIZE)
            with self.tf_graph.as_default():
                target_q_values = self.critic.target_model.predict(
                    [next_states, self.actor.target_model.predict(next_states)])
            replay_reward = rewards.reshape(rewards.shape[0], 1)
            y_t = np.zeros(actions.shape)
            for i in range(cf.BATCH_SIZE):
                if dones[i]:
                    y_t[i] = rewards[i]
                else:
                    y_t[i] = rewards[i] + cf.GAMMA * target_q_values[i]

            with self.tf_graph.as_default():
                loss = self.critic.train(states, actions, y_t, current_reward, replay_reward,
                                         self.ep, self.sum_writer)  # Train critic
                a_for_grad = self.actor.model.predict(states)
                grads = self.critic.gradients(states, a_for_grad)  # Cal gradients for that action from critic
                self.actor.train(states, grads)  # Train actor
                self.actor.target_train()  # train target actor
                self.critic.target_train()  # train target critic
                total_loss += loss
        print("Episode", self.ep, "Buffer", self.memory.count(), '/', cf.BUFFER_SIZE, "Loss", total_loss)

    def evaluate(self):
        pass

    def load(self, ep, timestamp):
        actor_file = '/actor_%d_%d.hdf5' % (timestamp, ep)
        critic_file = '/critic_%d_%d.hdf5' % (timestamp, ep)
        try:
            with self.tf_graph.as_default():
                self.actor.model.load_weights(self.WEIGHT_PATH + actor_file)
                self.actor.target_model.load_weights(self.WEIGHT_PATH + actor_file)
                self.critic.model.load_weights(self.WEIGHT_PATH + critic_file)
                self.critic.target_model.load_weights(self.WEIGHT_PATH + critic_file)
            print('.....Already loaded weights from file "%s" & "%s"' % (actor_file, critic_file))
        except Exception as e:
            print('*****Cannot load weights')
            print(e)

        self.ep = ep + 1
        self.timestamp = timestamp
        print('.....Starting with episodes :', self.ep)
        print('.....Starting with timestamp :', self.timestamp)
        if cf.TRAIN:
            # load buffer
            try:
                file_handler = open(cf.BUFF_PATH + '/buff' + str(timestamp) + '.obj', 'rb')
                self.memory = pickle.load(file_handler)
                print('.....Loaded buffer')
            except Exception as e:
                print('*****Cannot Load buffer')
                print(e)

    def dump(self):
        # save weight
        try:
            tf.gfile.MakeDirs(self.WEIGHT_PATH)
            self.actor.model.save_weights(self.WEIGHT_PATH + '/actor_%d_%d.hdf5' % (self.timestamp, self.ep), overwrite=False)
            self.critic.model.save_weights(self.WEIGHT_PATH + '/critic_%d_%d.hdf5' % (self.timestamp, self.ep), overwrite=False)
            print('.....Already saved weights to "%s"' % self.WEIGHT_PATH)
        except Exception as e:
            print('*****Cannot save weights')
            print(e)

        if cf.TRAIN:
            # save buffer
            try:
                tf.gfile.MakeDirs(cf.BUFF_PATH)
                file_handler = open(cf.BUFF_PATH + '/buff' + str(self.timestamp) + '.obj', 'wb')
                pickle.dump(self.memory, file_handler)
                print('.....Saved buffer')
            except Exception as e:
                print('*****Cannot Save buffer')
                print(e)
