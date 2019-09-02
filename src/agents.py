# encoding: utf-8
from collections import deque

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense

np.random.seed(1)
tf.set_random_seed(1)


class DQN_Agent(object):
    def __init__(self, s_dim, a_dim, epsilon_decay, epsilon_min, gamma, replay_batchsize=32, lr=0.002,
                 memory_size=2000):
        self.memory = deque(maxlen=memory_size)

        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.gamma = gamma
        self.replay_batchsize = replay_batchsize

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.step_count = 0
        self.replay_count = 0

        self.model = keras.models.Sequential()
        self.model.add(Dense(64, input_dim=self.s_dim, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.a_dim, activation='linear'))

        self.model.compile(loss='mse',
                           optimizer=keras.optimizers.Adam(lr=lr))

        self.curr_exploration_rate = 1

    def Q_prediction(self, S):
        DQN_state = np.expand_dims(S, axis=0)
        prediction = self.model.predict(DQN_state)[0]
        return prediction

    def choose_action(self, state):
        state_actions = self.Q_prediction(state)
        if np.random.uniform() <= self.curr_exploration_rate:
            action_id = np.random.choice([i for i in range(self.a_dim)], 1)[0]
        else:
            action_id = np.argmax(state_actions)
        return state_actions, action_id

    def remember(self, state, action_id, reward, new_state):
        q_target = reward + self.gamma * np.max(self.Q_prediction(new_state))
        self.step_count += 1
        self.memory.append([state, action_id, q_target])

    def replay(self):
        self.replay_count += 1
        minibatch = np.array(list(self.memory))[np.random.randint(0, len(self.memory), self.replay_batchsize)]
        DQN_X = np.array([line[0] for line in minibatch])
        DQN_y_pred = self.model.predict(DQN_X)
        DQN_Y_gt = []

        for idx, line in enumerate(minibatch):
            _, action_id, q_target = line
            y_gt_line = DQN_y_pred[idx].copy()
            y_gt_line[action_id] = q_target
            DQN_Y_gt.append(y_gt_line)

        DQN_Y_gt = np.array(DQN_Y_gt)

        self.model.fit(DQN_X, DQN_Y_gt, batch_size=self.replay_batchsize, epochs=5, verbose=0)

    def learn(self):
        self.curr_exploration_rate = self.epsilon_min if self.curr_exploration_rate < self.epsilon_min else self.curr_exploration_rate * self.epsilon_decay
        self.replay()
