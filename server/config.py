# Parameter config
BUFFER_SIZE = 100000
BATCH_SIZE = 32
GAMMA = 0.99 # Discounted Factor
TAU = 0.001  # Target Network HyperParameters
LRA = 0.0001  # Learning rate for Actor
LRC = 0.001  # Lerning rate for Critic

# Network config
action_dim = 2  # x_pos , y_pos
state_dim = 6  # of sensors input
