
from Agent import *
from  Environment import *
import matplotlib.pyplot as plt
import numpy as np
import imageio
import tempfile
import os
import pygame
from PIL import Image  
import pickle

class Game:
    def __init__(self, environnement, agents):
        
        self.environnement = environnement
        self.agents = [Agent(agent.state, agent.action, agent.reward, agent.done) for agent in agents]
        
        for agent in self.agents:
            agent.Q = np.zeros([self.environnement.size_map ** 2, 5])
                               
                    
        self.initial_agent_states = [(agent.state, agent.action, agent.reward, agent.done,agent.next_state, agent.previous_done) for agent in self.agents]
        
        self.history = []
        
      
    
    #---------------------------------------------------------------------------------------------
        # Saving  state-action History for each agent for the pygame render
    #---------------------------------------------------------------------------------------------
    def save_history(self):
        
        self.history.append([[agent.state,agent.action] for agent in self.agents])
        
    def reset_history(self):
        self.history = []
        
       
    
    
    #---------------------------------------------------------------------------------------------
        # Looking if the state of an agent is a valid state
    #---------------------------------------------------------------------------------------------
    def is_valid_state(self, state):
        return 0 <= state < self.environnement.size_map * self.environnement.size_map
    
    
    
    
    #---------------------------------------------------------------------------------------------
        # Looking if the state is a terminated state
    #---------------------------------------------------------------------------------------------
    def is_endOfepisode(self, state):
        
        done = False
        if self.environnement.map[state // self.environnement.size_map][state % self.environnement.size_map] in ["G", "S"]:
            done = True
        return done
    
    
    
    
    #---------------------------------------------------------------------------------------------
        # Looking if the game is over
    #---------------------------------------------------------------------------------------------
    def game_over(self):
        done = [False] * self.environnement.number_agent

        for i in range(self.environnement.number_agent):
            done[i] = self.agents[i].done
        if False in done:
            return False
        else:
            return True
            
         
        
        
    
    #---------------------------------------------------------------------------------------------
        # Step of the agent in the environment
    #---------------------------------------------------------------------------------------------
    def step(self, actions):
        
        new_states = [0] * self.environnement.number_agent
       
        for i in range(self.environnement.number_agent):
            
            self.agents[i].reward = 0
         
            # Calculate the new state knowing is action
            if actions[i] == 0:
                new_states[i] = self.agents[i].state - 1 if self.agents[i].state % self.environnement.size_map != 0 else self.agents[i].state
            elif actions[i] == 1:
                new_states[i] = self.agents[i].state + self.environnement.size_map if self.agents[i].state < (self.environnement.size_map)**2 - self.environnement.size_map else self.agents[i].state
            elif actions[i] == 2:
                new_states[i] = self.agents[i].state + 1 if self.agents[i].state % self.environnement.size_map != self.environnement.size_map  - 1 else self.agents[i].state
            elif actions[i] == 3:
                new_states[i] = self.agents[i].state - self.environnement.size_map if self.agents[i].state >= self.environnement.size_map else self.agents[i].state
            elif actions[i] == 4:
                new_states[i] = self.agents[i].state
        
        for i in range(self.environnement.number_agent):
            
            # Validation of new_state
            if self.is_valid_state(new_states[i]):
                
                # Detection de collision with other agents
                collision = False
                for j in range(self.environnement.number_agent): 
                    if  i != j and new_states[i] == new_states[j]:
                        collision = True
                        
                        # Update the reward of agent involved in the colision 
                        
                        # colission are not count where the reward(treasure) is 
                        if self.environnement.map[new_states[i] // self.environnement.size_map][new_states[i] % self.environnement.size_map] == "G":
                            self.agents[i].reward = self.environnement.get_reward(new_states[i], self.agents[i])
                           # rewards[i] = self.agents[i].reward
                            
                        else:    
                            self.agents[i].reward = self.environnement.agent_rewards["Collision"]

                            self.agents[j].reward = self.environnement.agent_rewards["Collision"]

                        break
                        
                    else:
                        
                        # Update of the reward of the agent knowing his state
                        self.agents[i].reward = self.environnement.get_reward(new_states[i], self.agents[i])
        

                self.agents[i].next_state = new_states[i]
                    
                self.agents[i].action = actions[i]

                # Checking if the episode is finished for this agent
                self.agents[i].done = self.is_endOfepisode(self.agents[i].next_state)
                
            else:
                # the agent can't move in this direction
                self.agents[i].action = actions[i]
                self.agents[i].reward = 0
                self.agents[i].done  = False
                
    
    
    
    
    
    #---------------------------------------------------------------------------------------------
            # modify the reward of an agent if he has the same policy
    #---------------------------------------------------------------------------------------------
    def novelty(self,n,c):
        
        for a in range(self.environnement.number_agent):
            for b in range(self.environnement.number_agent):
                if a != b and self.agents[b].diverse_action( 
                    Q = self.agents[a].Q, state = self.agents[b].state, n = n) == self.agents[b].action:
                    self.agents[b].reward =  self.agents[b].reward - c
                    

       
    
    
    #---------------------------------------------------------------------------------------------
            # matplotlib display of an episode
    #---------------------------------------------------------------------------------------------
    def display(self):
        m = np.zeros([self.environnement.size_map,self.environnement.size_map])
        for i  in range(self.environnement.size_map):
            for j in range(self.environnement.size_map):
                if self.environnement.map[i][j] == "F":
                    m[i][j] = 0
                elif self.environnement.map[i][j] == "H":
                    m[i][j] = 1
                elif self.environnement.map[i][j] == "S":
                    m[i][j] = 2
                elif self.environnement.map[i][j] == "G":
                    m[i][j] = 3
                    
        for i in range(self.environnement.number_agent):
            m[self.agents[i].state // self.environnement.size_map][self.agents[i].state % self.environnement.size_map] = self.environnement.size_map + i
        plt.imshow(m)   
        plt.show()

        
        
               
    #---------------------------------------------------------------------------------------------
        # Saving image to create a Gif for an episode
    #---------------------------------------------------------------------------------------------
    def save_image(self, step):
        
        # Screenshot of the windows of pygame
        pygame.image.save(pygame.display.get_surface(), f"images/screenshot_{step}.png")

    def create_gif(self, gif_filename):
        
       # loading all the picture saved
        images = []
        for step in range(len(self.history)):
            image_path = f"images/screenshot_{step}.png"
            images.append(imageio.imread(image_path))

         # save picture like a GIF 
        gif_filename = os.path.join("Test_gif", gif_filename)  # path to the folder Test
        gif_filename = gif_filename.rstrip(".gif") + ".gif" 
        imageio.mimsave(gif_filename, images, format="GIF", fps=2)

        
        
            
    #---------------------------------------------------------------------------------------------
        # Create a Pygame windows for an episode
    #---------------------------------------------------------------------------------------------
    def render(self,modele_name):
        # Data incoming
        gameMap = self.environnement.map
        stepsGame = self.history
                    
            
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # Metadata for display game
        lenGameMap = len(gameMap)
        numAgent = len(stepsGame[0])
        if numAgent > 8:
            print(f'There are {numAgent} agents. Only 8 agents will display.')
            numAgent = 8
        stepEnd = len(stepsGame)
        lenTiles = 32
        # -----------------------------------------------------------------------------
        # Load Assets
        for agent in [1,2,3,4,5,6,7,8]:
            for k in range(4):
                exec('assetAgent'+str(agent-1)+'p'+str(k)+' = pygame.image.load("assets/Agent_'+str(agent)+'p'+str(k)+'.png")')

        assetEmpty = pygame.image.load("assets/empty.png")
        assetGold = pygame.image.load("assets/gold.png")
        assetLava = pygame.image.load("assets/lava.png")
        assetRock = pygame.image.load("assets/rock.png")
        # -----------------------------------------------------------------------------
        # If map is short, then we scale by a factor 2
        if lenGameMap < 10 :
            lenTiles = 64
            for agent in [1,2,3,4,5,6,7,8]:
                for k in range(4):
                    exec('assetAgent'+str(agent-1)+'p'+str(k)+' = pygame.transform.scale(assetAgent'+str(agent-1)+'p'+str(k)+', (lenTiles,lenTiles))')

            assetEmpty = pygame.transform.scale(assetEmpty, (lenTiles,lenTiles))
            assetGold = pygame.transform.scale(assetGold, (lenTiles,lenTiles))
            assetLava = pygame.transform.scale(assetLava, (lenTiles,lenTiles))
            assetRock = pygame.transform.scale(assetRock, (lenTiles,lenTiles))
        # -----------------------------------------------------------------------------
        # Initialise pygame environnement
        pygame.init()
        nameWindows = " Jeu Apprentissage diversifie "
        pygame.display.set_caption(nameWindows)
        screen = pygame.display.set_mode((lenTiles*lenGameMap,lenTiles*lenGameMap))
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # Starting windows
       
        for step in range(len(stepsGame)):
            # --------------------------
            # Draw le map without agents
            for i in range(lenGameMap):
                for j in range(lenGameMap):
                    if gameMap[i][j] == "F":    
                        screen.blit(assetEmpty, (j*lenTiles,i*lenTiles))
                    elif gameMap[i][j] == "S":    
                        screen.blit(assetLava, (j*lenTiles,i*lenTiles))
                    elif gameMap[i][j] == "H":    
                        screen.blit(assetRock, (j*lenTiles,i*lenTiles))
                    elif gameMap[i][j] == "G":    
                        screen.blit(assetGold, (j*lenTiles,i*lenTiles))
            # --------------------------
            # Display agents (max 8 agents)
            if step < stepEnd:
                for nameAgent in range(numAgent):

                    if stepsGame[step][nameAgent][1] == 4:
                        exec("screen.blit(assetAgent"+str(nameAgent)+"p1, (stepsGame[step][nameAgent][0] % lenGameMap * lenTiles,stepsGame[step][nameAgent][0] // lenGameMap *lenTiles))")    
                    else : 
                        exec("screen.blit(assetAgent"+str(nameAgent)+"p"+str(int(stepsGame[step][nameAgent][1]))+", (stepsGame[step][nameAgent][0] % lenGameMap * lenTiles,stepsGame[step][nameAgent][0] // lenGameMap *lenTiles))")
                # --------------------------   
            
             # --------------------------
                    # Save the picture
            
            self.save_image(step)
               
            #  -----------------------------------    
            # Update the screen
            
            pygame.display.flip()  
            pygame.time.wait(500)
            
        pygame.quit()
                    
        # Créer un GIF à partir des captures d'écran sauvegardées
        self.create_gif(modele_name) 
        # -----------------------------------------------------------------------------  
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------

        
           
        





