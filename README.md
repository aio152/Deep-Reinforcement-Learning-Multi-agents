![MARL](https://github.com/user-attachments/assets/8297f7d6-1062-48d7-b5b7-bd97d21a9027)

Implementation of the Deep Q-Learning approach for multi-agent systems. 
The goal is to ensure that each agent has a distinct policy compared to every other agent. 
Our objective is to encourage diverse policies among these agents, even if they are not optimal.
The emphasis lies in fostering cooperation among agents while acknowledging and embracing the diversity in their individual decision-making strategies.

# Algorithm

<img width="1104" alt="algo_diverse_Qlearning" src="https://github.com/user-attachments/assets/84c72ce5-e7f6-4566-b78b-841a26f3f5ea">


# Environment
Our environment is a square grid of size n x n that can contain p agents. Each agent
has five possible actions:

![direction_agent](https://github.com/user-attachments/assets/eac9cba5-f8cf-4fa8-b961-23aa02450d0e)

files: Class_environnement.py 



![Environment](https://github.com/user-attachments/assets/5df609c8-c619-47b2-93dd-39e440f8f2a5)

# Evaluation
8 agent train on a 16x16 map with a deep Q learning model

![16x16 map with 8 agents(1)](https://github.com/user-attachments/assets/85221775-5834-4d7d-b5a8-8cad7b3eae8f)

file Deep_Q_Learning_diversifié.ipynb



# Files
Q_learning_diverse contain a simple Q learning solution of the environment with two agents 

assets contain the assets for the environment 

Test-gif contain evaluation of some models after trainning 
