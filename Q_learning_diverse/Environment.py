class Environment:
    """
    Description of the environment:

    Dimension: nxn, n natural number
    
    States: n (from 1 to n-1, from left to right)
    Actions: 5 (0: left, 1: down, 2: right, 3: up, 4: stay still)
    Initial state:
        Agent 1: (state1, action1, reward1, done1)
        Agent 2: (state2, action2, reward2, done2)
        .
        .
        .
        Agent p: (statep, actionp, rewardp, donep)
    Map:
        S: Hole
        F: Free space
        H: Obstacle
        G: Treasure

    Features:

    Agent movement:
        The agent can move in 4 directions (left, down, right, up) or stay still
        Movement is only possible if the space is not a Hole (S)
    Reward:
        The agent receives a reward (+1) when it reaches the G space
    Penalty:
        The agent receives a penalty (-0.4) when it falls into a Hole (S)
        The agent receives a penalty (-0.4) when it falls on an obstacle (H)
        the agent in collision receive a penalty of
    End of the episode:
        The episode ends when all the agents reaches the G space or if other agent falls into a Hole (S)
    """
        
          
    def __init__(self, mapp, size_map,number_agent):
        self.map = mapp
        self.size_map = size_map
        self.number_agent = number_agent
        self.agent_rewards = {
            "G": 1,
            "S": -0.4,
            "H": -0.4,
            "default": 0,
            "Collision": 0 #-0.5 
        }        
    
    

    #-------------------------------------------------------------------------------------------------
        # Reward of each agent depending of the state where he is.
    #-------------------------------------------------------------------------------------------------
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
  


    #-------------------------------------------------------------------------------------------------
        # Useful function after for our neural network for knowing where each agent is.
    #-------------------------------------------------------------------------------------------------
    def position(self, state):
        pos = [0]* self.size_map**2
        print(self.size_map**2)
        for i in range(self.size_map**2):
            
            if state == i:
                pos[i] = 1
            else:
                pos[i] = 0
        return pos
    
