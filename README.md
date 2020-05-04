# A Q-learning from sctrach in Python for GridWorld

This repository consists of some Python codes for implementing the Q-Learning algorithm from scratch and training the agent to learn a maze.

`env_map_maker.py`:
This code consists of a helper function for implementing any standard mazes that have a start point, a goal point, and an inner wall.

`mdp_meta_data.py`:
Here, this is a helper function in which we create the metadata, such as the MDP, that we need for defining the Environment class.

`grid_world_general_env.py`
This piece of code defines the Environment class.

`plot_environment.py`:
This is a helper class that consists of some methods to make an image of the maze and to visualize the agent moves within the maze.

`agents.py`:
Here, we implemented a Random Agent and Q-learning Agent classes.

`train_QAgent.py`:
By this code, we can train the Q-learning agent to find the best policy to navigate inside the maze.

`test_QAgent.py`:
This is for testing the trained agent performance.

`visutils.py`:
This includes some helper functions to visualize the agent performance in the training phase.

`training_pipline.py`:
Here, we call the objects we've defined and run the training and test phases.

`RL_QLearningMaze_training_pipline.ipynb`:
This is similir to `training_pipline.py`, but in the Google colab.

`Tutorial_RL_QLearning_for_Maze.ipynb`:
A tutorial fof the programming session of the course.