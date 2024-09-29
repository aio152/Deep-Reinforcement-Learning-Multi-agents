import numpy as np
import random
from Agent import *
from Environment import *
from  Game_class import *

import matplotlib.pyplot as plt
import numpy as np
import imageio
import tempfile
import os
import pygame
from PIL import Image  
import pickle

mapp = np.array([
    "FFFFFFFFF",
    "FFFFFFHFF",
    "FSFFHFFFS",
    "FFFFFFHFF",
    "FFFFHFFFF",
    "FFHFFFFSF",
    "FFHFFHFHF",
    "FSFFFHFFF",
    "FFFFFFGFF"
    
])
# Create an environment with the map and two agents
env = Environment(mapp, 9, 3)
#----------------------------------------------------------------------------------
    # Creation of agents and environment
    
#agents = [Agent(np.random.randint(env.size_map**2), np.random.randint(5), 0) for _ in range(env.number_agent)]
agents = [Agent(11,0,0, False), Agent(3,0,0,False), Agent(2,0,0,False)]

# Creation of the game
game = Game(env, agents)

#----------------------------------------------------------------------------------------------------------------
        # Hyperparamters
    
gamma              = 0.9                  # discounting factor
alpha              = 0.1                  # learning rate
n                  = 0.001                # eta of the article
epsilon            = 1                    # 1 = 100% random actions
nbEpisodes         = 50_000               #number of episode
epsilon_decay_rate = 1 / nbEpisodes       # epsilon decay rate. 1 / 0.0001 = 10,000
rng                = np.random.default_rng()
max_step           = 60                   # Max step during an episode
c                  = 0.08                 # the novelty to degrade the reward of the agents
#-----------------------------------------------------------------------------------------------------------
 
action = [0]* env.number_agent
state = [0]* env.number_agent

rew = [0]*env.number_agent # list of sum of reward of each agent during trainning

for i in range(nbEpisodes):
    
    done = [False]* env.number_agent
    terminated = False
    nbSteps = 0

    for j in range(env.number_agent):
        #game.reset(game.agents[j])
        game.agents[j].reset()
        
    while (nbSteps< max_step and (not terminated)):
            #------------------------------------------------------------------------------
                # All agents take an action
            #------------------------------------------------------------------------------
            for m in range(env.number_agent):
                action[m] = game.agents[m].take_action(epsilon,n)
                
                
            #------------------------------------------------------------------------------
                    # A step in our environment
            #------------------------------------------------------------------------------
            game.step(action)

            
            
            #------------------------------------------------------------------------------
                    # Modifying reward for diverse learning
            #------------------------------------------------------------------------------
            game.novelty(n=n, c=0.06)       
                
                
                
            #------------------------------------------------------------------------------
                          # Learning for all agents
            #------------------------------------------------------------------------------
            for m in range(env.number_agent):   
                game.agents[m].learn(gamma = gamma,learning_rate = alpha)        
                    
                    
                    
            #------------------------------------------------------------------------------
                    # Updating next step and  calculate the terminated state
            #------------------------------------------------------------------------------
            nbSteps += 1
        
            terminated = game.game_over() 
            
            #------------------------------------------------------------------------------
                # Store the reward of each agent during an episode
            #------------------------------------------------------------------------------
            for m in range(env.number_agent):   
                rew[m] = game.agents[m].reward
                
                
                
            #------------------------------------------------------------------------------
                # Updating the state and the previous_done for each agent
            #------------------------------------------------------------------------------
            for m in range(env.number_agent):
                game.agents[m].state =  game.agents[m].next_state
                game.agents[m].previous_done = game.agents[m].done 
                
    #------------------------------------------------------------------------------
            # Linear epsilon greedy
    #------------------------------------------------------------------------------

    epsilon = max(epsilon - epsilon_decay_rate, 0)
    
  #------------------------------------------------------------------------------
            # Saving the Q_ table for each agent
  #------------------------------------------------------------------------------

for m in range(env.number_agent):
    game.agents[m].save_q_table('q' + str(m) +'_table.pkl')

