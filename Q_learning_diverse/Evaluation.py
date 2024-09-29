from Agent import *
from Environment import *
from Game_class import *

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

  #------------------------------------------------------------------------------
            # Loading the Q_ table for each agent
  #------------------------------------------------------------------------------

for m in range(env.number_agent):
    game.agents[m].load_q_table('q' + str(m) +'_table.pkl')
 
    
action = [0]* env.number_agent
state = [0]* env.number_agent

rew = [0]*env.number_agent # list of sum of reward of each agent during trainning

max_step = 60
n        = 0.001  
for i in range(1):
    
    done = [False]* env.number_agent
    terminated = False
    nbSteps = 0

    for j in range(env.number_agent):
        game.agents[j].reset()
        
    game.reset_history() # reset the historical state-action of each agent
        
    while (nbSteps< max_step and (not terminated)):
            #------------------------------------------------------------------------------
                # All agents take an action
            #------------------------------------------------------------------------------
            for m in range(env.number_agent):
                action[m] = game.agents[m].take_action(epsilon=0,n=n)
                
                
            #------------------------------------------------------------------------------
                    # A step in our environment
            #------------------------------------------------------------------------------
            game.step(action)

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

            game.save_history()    # Saving the historical position of each agent
            

game.render("3_agents_diverse_learning")

