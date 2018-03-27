import numpy as np
from task import Task
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

'''
this Agent canNOT move up on Z, at least because it only considers first 6 values in state; i guess that i made this decision
because in the instructor's code, in task.py, `self.state_size = self.action_repeat * 6`,
and in step(), next_state only contains 'pose'.

now i believe that a state needs to contain all 12 values, including velocity and angular velocity.
otherwise, state aliasing will be extremely serious; there is no way for the training to go well.

i WILL update the code later. //now just exhausted of going over all code multiple times.
'''

class Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        # from now on, normalize action to [0, action_size)
        
        # nn for Q
        self.learning_rate = 0.001
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=6, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.init_model_weights = self.model.get_weights()
        
        #np.random.normal(
        #    size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
        #    scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        #model.save_weights('best_weights.h5')
        #model.load_weights('best_weights.h5')
        
        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        #self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        # don't restart learning
        #self.model.set_weights(self.init_model_weights)
        self.total_reward = 0.0
        self.count = 0
        self.state = self.task.reset()
        return self.state

    # better to name it learn_from_one_step()
    #def step(self, reward, done):
    def step(self):
        action = self.get_action()
        print('step ', 'state ', self.state, 'action : ', action)
        next_state, reward, done = self.task.step(action[0])

        # Save experience / reward
        #self.total_reward += reward # this is for DP or MC; but not TD
        #self.count += 1

        # learn at each step
        input = self.state.reshape(3, 6)
        output = self.model.predict(input)
        print('step output : ', type(output))
        output[0:, action] = reward
        self.model.fit(input, output)
        
        self.state = next_state
        return done


    # better to name it as get_act()
    #def act(self, state):
        # Choose action based on given state and policy
        #action = np.dot(state, self.w)  # simple linear policy
        #action = self.model.predict(state)
    def get_action(self):
        print('state: ', self.state[:6])
        action = self.model.predict(np.resize(self.state[:6], [1, 6]))
        return action + self.action_low # here self.action_low==0

    def learn(self):
        '''
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        '''
        