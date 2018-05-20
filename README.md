# Dota2 Bot
--------------
| **`Reffecence paper`** |
|-----------------|
| *Silver, David, Lever, Guy, Heess, Nicolas, Degris, Thomas, Wierstra, Daan, and Riedmiller, Martin. **Deterministic policy gradient algorithms.** In ICML, 2014.* [![Reffecence paper](https://img.shields.io/badge/api-reference-blue.svg)](http://proceedings.mlr.press/v32/silver14.pdf) |
| *Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra.  **Continuous control with deep reinforcement learning.** CoRR, abs/1509.02971, 2015.* [![Reffecence paper](https://img.shields.io/badge/api-reference-blue.svg)](https://arxiv.org/pdf/1509.02971.pdf) |
--------------
**Dota2 Bot** is an dota2 creep blocking AI bot was implement by reinforcement learning alogorithums via. Deep Deterministic Policy Gradients (DDPG)
## Requirements
- Dota2
- Dota2 Workshop Tools DLC
- python3.6
- pip

## Installation

###### Install libs
```shell
$ pip install -r requirment.txt
```
## Setting (server/setting.py)
Correct settting is nessesery for corect running task such as
- training by uniform noise
- traing by Ornstein-Uhlenbeck
- testing (evaluating)

###### Paramter Setting
```python
>>> BUFFER_SIZE = 100000
>>> GAMMA = 0.99  # Discounted Factor
>>> BATCH_SIZE = 200
>>> TAU = 0.001  # Target Network HyperParameters
>>> LRA = 0.0001  # Learning rate for Actor
>>> LRC = 0.001  # Lerning rate for Critic
```
###### Network config setting
```python
>>> ACTION_DIM = 2  # x_pos , y_pos
>>> STATE_DIM = 11  # of sensors input
>>> ACTOR_HIDDEN1_UNITS = 150
>>> ACTOR_HIDDEN2_UNITS = 200
>>> CRITIC_HIDDEN1_UNITS = 150
>>> CRITIC_HIDDEN2_UNITS = 200
```
###### Lerning behavior setting
```python
>>> TRAIN = 0
>>> EXPLORE = 20
>>> OU = 0
>>> MU = -10
>>> SIGMA = 30
>>> REPLACE_FREQ = 1
>>> BOOTSTRAP_FREQ = 5
>>> SAVE_FREQ = 1000
```
## Run Flask server
```shell
$ python server/app.py
```
## Load weights from pre-trained network
###### ex1. uniform noise with 20 noise scale with boostraping
```shell
$ python server/reload.py --timestamp 1521285961 --ep 151000
```
###### ex2. Ornstein-Uhlenbeck without boostraping
```
$ python server/reload.py --timestamp 1523632535 --ep 118000
```
