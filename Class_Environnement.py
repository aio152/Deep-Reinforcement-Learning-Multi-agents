

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import imageio
import matplotlib.pyplot as plt
import tempfile
import os
import pygame
from PIL import Image  # Importer Image pour redimensionner les images



class Environment:
    """
    Description de l'environnement:

    Dimension : nxn n entier naturel
    
    États : n (de 1 à n-1, de gauche à droite)
    Actions : 5 (0 : gauche, 1 : bas, 2 : droite, 3 : haut, 4 : immobile)
    État initial :
        Agent 1 : (état 1, action1, reward1)
        Agent 2 : (état2, action2, reward2)
        .
        .
        .
        Agent p : (étatp, actionp, rewardp)
    Carte :
        S : Trou
        F : Case libre
        H : Obstacle
        G : Récompense

Fonctionnalités:

    Déplacement des agents :
        L'agent peut se déplacer dans les 4 directions (gauche, bas, droite, haut) ou rester immobile
        Le déplacement est possible uniquement si la case n'est pas un Trou (S)
    Récompense :
        L'agent reçoit une récompense (+1.5) lorsqu'il atteint la case G
    Pénalité :
        L'agent reçoit une pénalité (-0.5) lorsqu'il tombe dans un trou (S)
        L'agent reçoit une pénalité (-0.35) lorsqu'il tombe sur un obstacle (S)

    Fin de l'épisode :
        L'épisode se termine lorsque l'un des agents atteint la case G ou tombe dans un trou S

    """
    def __init__(self, mapp, size_map,number_agent):
        self.map = mapp
        self.size_map = size_map
        self.number_agent = number_agent
        self.agent_rewards = {
            "G": 2,
            "S": -0.1,
            "H": -0.2,
            "default": 0,
            "Collision": -0.3
        }        
    
    def get_reward(self, state, agent):
        
        my_reward = 0
                    
        if self.map[state // self.size_map][state % self.size_map] == "G" :
            my_reward = self.agent_rewards["G"]
            
        elif self.map[state // self.size_map][state % self.size_map] == "S":
            my_reward =  self.agent_rewards["S"] 
        
        elif self.map[state // self.size_map][state % self.size_map] == "H":
            my_reward = self.agent_rewards["H"]
        
        elif self.map[state // self.size_map][state % self.size_map] == "F":
            my_reward = self.agent_rewards["default"]
        
        agent.reward = my_reward
        return agent.reward
  
    def position(self, state):
        pos = [0]* self.size_map**2

        for i in range(self.size_map**2):
            
            if state == i:
                pos[i] = 1
            else:
                pos[i] = 0
        return pos

# # Class Agent

# In[9]:


class Agent:
    def __init__(self, state, action, reward,done):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.initial_state = [(np.random.randint(1**2),0,0,False)]#  for i in range(number_agent)]
    def etat_action(self):
        return self.state, self.action
    


# # Class Game

# In[109]:


class Game:
    def __init__(self, environnement, agents):
        
        self.environnement = environnement
        self.agents = [Agent(agent.state, agent.action, agent.reward, agent.done) for agent in agents]
        self.initial_agent_states = [(agent.state, agent.action, agent.reward, agent.done) for agent in self.agents]
        
        self.history = []
        
        
    def save_history(self):
        
        self.history.append([[agent.state,agent.action] for agent in self.agents])
        
    def reset_history(self):
        self.history = []
        
    def is_valid_state(self, state):
        return 0 <= state < self.environnement.size_map * self.environnement.size_map
    
    def is_endOfepisode(self, state):
        """
        Détermine si l'épisode est terminé pour l'agent se trouvant dans l'état donné.
        
        Args:
        state: L'état de l'agent.
        
        Returns:
        True si l'épisode est terminé, False sinon.
        """
        
        done = False
        if self.environnement.map[state // self.environnement.size_map][state % self.environnement.size_map] in ["G", "S"]:
            done = True
        return done
    
    def game_over(self):
        done = [False] * self.environnement.number_agent

        for i in range(self.environnement.number_agent):
            done[i] = self.agents[i].done
        if False in done:
            return False
        else:
            return True
            
    def reset(self, agent):
        # Réinitialiser la position de l'agent aux valeurs initiales
        initial_state = self.initial_agent_states[self.agents.index(agent)]
        agent.state = initial_state[0]
        agent.action = initial_state[1]
        return agent.state, agent.action, 0, False

    def step(self, actions):
        
        """
        Simule le déplacement de tous les agents.

        Args:
            actions: Liste des actions que chaque agent souhaite effectuer.

        Returns:
            Une liste de nouveaux états, une liste de récompenses, et un booléen indiquant si l'épisode est terminé.
        """
        
        #terminated = False
        new_states = [0] * self.environnement.number_agent
        rewards = [0] * self.environnement.number_agent
        done = [False] * self.environnement.number_agent
        
        for i in range(self.environnement.number_agent):
            #renitialiser les recompenses
            self.agents[i].reward = 0
            #print()
            # Déterminer le nouvel état en fonction de l'action
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
            
            # Validation du nouvel état
            if self.is_valid_state(new_states[i]):
                
                # Détection de collision avec d'autres agents
                collision = False
                for j in range(self.environnement.number_agent): # a modifier avec j de i+1 a nbre agent - 1 
                    if  i != j and new_states[i] == new_states[j]:#self.agents[i] != self.agents[j] and
                        collision = True
                        #mettre à jour les récompenses des agents impliqués
                        
                        # on va pas compter les collisions au niveau de l'etat ou se trouve le gain
                        if self.environnement.map[new_states[i] // self.environnement.size_map][new_states[i] % self.environnement.size_map] == "G":
                            self.agents[i].reward = self.environnement.get_reward(new_states[i], self.agents[i])
                            rewards[i] = self.agents[i].reward
                            
                        else:    
                            self.agents[i].reward = self.environnement.agent_rewards["Collision"]

                            self.agents[j].reward = self.environnement.agent_rewards["Collision"]

                            rewards[i] = self.environnement.agent_rewards["Collision"]
                            rewards[j] = self.environnement.agent_rewards["Collision"]
                        break
                        
                    else:
                        # Mettre à jour la récompense de l'agent en fonction de son nouvel état
                        self.agents[i].reward = self.environnement.get_reward(new_states[i], self.agents[i])
                        rewards[i] = self.agents[i].reward
        

                # Mettre à jour l'état et l'action de l'agent
                #if collision:
                #self.agents[i].state = self.agents[i].state
                #else:
                self.agents[i].state = new_states[i]
                    
                self.agents[i].action = actions[i]

                # Vérifier si l'épisode est terminé pour cet agent
                done[i] = self.is_endOfepisode(self.agents[i].state)
                self.agents[i].done = self.is_endOfepisode(self.agents[i].state)
            else:
                # L'agent ne peut pas se déplacer dans cette direction
                self.agents[i].action = actions[i]
                self.agents[i].reward = 0
                rewards[i] = 0
                done[i] = False
                self.agents[i].done[i]  = False
                
        return new_states, rewards, done
    
    def novelty(self, agent,c):
        if self.environnement.map[agent.state // self.environnement.size_map][agent.state % self.environnement.size_map] == "G":
            return c[0]
        
        elif self.environnement.map[agent.state // self.environnement.size_map][agent.state % self.environnement.size_map] == "F":
            return c[1]
        
        elif self.environnement.map[agent.state // self.environnement.size_map][agent.state % self.environnement.size_map] == "H":
            return c[2]
        
        elif self.environnement.map[agent.state // self.environnement.size_map][agent.state % self.environnement.size_map] == "S":
            return c[3]

            

        
        
        
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
      
    def save_image(self, step):
        # Capture d'écran de la fenêtre pygame
        pygame.image.save(pygame.display.get_surface(), f"images/screenshot_{step}.png")

        
    def create_gif(self, gif_filename, duration):
        # Charger toutes les images sauvegardées
        images = []
        for step in range(len(self.history)):
            image_path = f"images/screenshot_{step}.png"
            images.append(imageio.imread(image_path))

        # Enregistrer les images comme un GIF avec une durée personnalisée
        gif_filename = os.path.join("Test_gif", gif_filename)  # Chemin complet vers le dossier Test
        gif_filename = gif_filename.rstrip(".gif") + ".gif"  # S'assurer que le nom de fichier se termine par .gif
        imageio.mimsave(gif_filename, images, format="GIF", duration=duration)

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
            # Display agents (max 4 agents)
            if step < stepEnd:
                for nameAgent in range(numAgent):
                    
                    if stepsGame[step][nameAgent][1] == 4:
                        exec("screen.blit(assetAgent"+str(nameAgent)+"p1, (stepsGame[step][nameAgent][0] % lenGameMap * lenTiles,stepsGame[step][nameAgent][0] // lenGameMap *lenTiles))")    
                    else : 
                        exec("screen.blit(assetAgent"+str(nameAgent)+"p"+str(int(stepsGame[step][nameAgent][1]))+", (stepsGame[step][nameAgent][0] % lenGameMap * lenTiles,stepsGame[step][nameAgent][0] // lenGameMap *lenTiles))")
                # --------------------------   
            
             
            # Sauvegarde de l'image
            
            self.save_image(step)
               
            
            # Update the screen
            #######################################################################""
            #  -----------------------------------     A enlever si compilation sur serveur de calcul
            pygame.display.flip()  
            pygame.time.wait(500)
            
        pygame.quit()
                    
         # Créer un GIF à partir des captures d'écran sauvegardées
        self.create_gif(modele_name) 
        # -----------------------------------------------------------------------------  
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------

        
           
