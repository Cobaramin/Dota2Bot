import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TEST_TMPDIR'] =  os.path.join(CURRENT_DIR, 'tmp/')


class Config(object):
    # Parameter config
    BUFFER_SIZE = 100000
    BATCH_SIZE = 128
    GAMMA = 0.99  # Discounted Factor
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    # Network config
    ACTION_DIM = 2  # x_pos , y_pos
    STATE_DIM = 3  # of sensors input
    ACTOR_HIDDEN1_UNITS = 150
    ACTOR_HIDDEN2_UNITS = 200
    CRITIC_HIDDEN1_UNITS = 150
    CRITIC_HIDDEN2_UNITS = 200

    # Learning style config
    TRAIN = 1
    EXPLORE = 20
    REPLACE_FREQ = 1
    BOOTSTRAP_FREQ = 5
    SAVE_FREQ = 200

    SAVE_PATH = os.path.join(CURRENT_DIR, 'weights/')
    LOGS_PATH = os.path.join(CURRENT_DIR, 'logs/')
    TMP_PATH = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/dota2/logs/dota2_with_summaries')
    # /home/cobaramin/Documents/Dota2Bot/server/tmp/tensorflow/dota2/logs/dota2_with_summaries

cf = Config()
