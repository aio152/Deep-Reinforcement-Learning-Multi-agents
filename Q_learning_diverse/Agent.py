import matplotlib.pyplot as plt
import numpy as np
#import imageio
import tempfile
import os
import pygame
from PIL import Image  
import pickle



class Agent:
    def __init__(self, state_init, action_init, reward_init,done_init):
        self.state         = state_init
        self.next_state    = 0
        self.action        = action_init
        self.reward        = reward_init
        self.done          = done_init
        self.previous_done = done_init
        self.Q             = [] 
        self.initial_state = state_init
        self.initial_action= action_init
        
    #---------------------------------------------------------------------------------------------
                        # Reset the position of the agent
    #---------------------------------------------------------------------------------------------
    def reset(self):
        """
        Reset the attributes of the agent to initial values.
    
        """
        self.state         = self.initial_state
        self.next_state    = 0
        self.action        = self.initial_action
        self.reward        = 0
        self.done          = False
        self.previous_done = False
    
    #---------------------------------------------------------------------------------------------
        # Chosing the action by the equation in the article(cf Determinist policy)
    #---------------------------------------------------------------------------------------------

    def diverse_action(self,Q, state,n):
        set_action = []
        Q_state = Q[state, :]

        action_max = np.max(Q[state, :])

        for element in Q_state:
            if abs(element - action_max) < n:
                indice = np.where(Q_state == element)[0]

                set_action.append(indice[0])

        return set_action[-1] 

     #---------------------------------------------------------------------------------------------
                                 # Taking action 
    #---------------------------------------------------------------------------------------------

    def take_action(self,epsilon,n):
        rng = np.random.default_rng()
        
        if not(self.previous_done): # look if agent1 is in a hole or on the reward state
            
            # Epsilon Greedy for agent1
            if rng.random() < epsilon:
                self.action = np.random.choice([0, 1, 2, 3, 4])
                
            else:
                self.action = self.diverse_action(Q = self.Q, state = self.state, n = n)
                
        else:
             self.action = 4 # stay in place  


        return self.action
    
    
    #---------------------------------------------------------------------------------------------
                                 # Learning
    #---------------------------------------------------------------------------------------------

    def learn(self, gamma,learning_rate):
        
        if not self.previous_done: # don't update the Q_table while your agent is in a hole or on the treasure state
            
            self.Q[self.state, self.action] = self.Q[self.state, self.action] + learning_rate * (
                        self.reward + gamma * np.max(self.Q[self.next_state, :]) - self.Q[self.state, self.action])
        

    #---------------------------------------------------------------------------------------------
                            # Save Q table to a file
    #---------------------------------------------------------------------------------------------
    def save_q_table(self, filename):
        
        with open(filename, 'wb') as f:
            pickle.dump(self.Q, f)

    #---------------------------------------------------------------------------------------------
                            # Load Q table from a file
    #---------------------------------------------------------------------------------------------
    def load_q_table(self, filename):
        
        with open(filename, 'rb') as f:
            self.Q = pickle.load(f)




