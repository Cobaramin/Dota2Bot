# Parameter config
BUFFER_SIZE = 100000
BATCH_SIZE = 128
GAMMA = 0.99 # Discounted Factor
TAU = 0.001  # Target Network HyperParameters
LRA = 0.0001  # Learning rate for Actor
LRC = 0.001  # Lerning rate for Critic
EXPLORE = 10

# Network config
action_dim = 2  # x_pos , y_pos
state_dim = 11  # of sensors input

# Frequency
REPLACE_FREQ = 1
SAVE_FREQ = 200
BOOTSTRAP_FREQ = 5
