# Learning-shortcuts-in-chemical-space-using-Reinforcement-Learning

This is part of a research under Prof. Alessandro Lunghi. 

## Problem-

The electron spin is an ideal candidate for the implementation of quantum sensing protocols, but the inherent fragility of its quantum states and the limits on interfacing it in relevant environments pose serious constraints to this field. On the other hand, the design of new materials with optimal characteristics is an incredibly hard challenge, as it requires the ability to explore billions of potential materials. In this project, we will explore the use of reinforcement learning to individuate promising avenues of exploration of the chemical space of molecular qubits. Reinforcement learning is inspired by the way humans learn through trial and error and positive outcomes are reinforced by rewards and vice versa. This branch of computer science has revolutionized problem-solving in areas such as robotics, game playing, and more, as also popularized by the achievements of AlphaGO, i.e. the first AI able to beat the world champion of the game GO.

New molecules will automatically be assembled, and their spin properties estimated with quantum chemistry methods. Molecules with promising properties will provide a positive bias to our AI, thus letting them learn from experience. The student will use standard Python ML libraries (TensorFlow or PyTorch) to implement some reinforcement learning algorithms such as (Deep)-Q-Learning or deterministic policy gradient-based reinforcement learning to individuate the optimal organic ligands to bond a central magnetic ion such as Cobalt.

## Reinforcement Learning
The recent advancements and immense popularity of Machine Learning has opened up endless opportunities to solve complex human problems by training the machine to learn based
on some human data. But making the machine learn like humans based on experiences from the environment is probably the best way to learn from nature. This type of computational learning approach where the machine is learning from interaction with it’s environment and using trial and error to compute good and bad outcome from it’s data is called Reinforcement Learning.

The agent, according to this policy, updates it’s state by taking appropriate actions in the environment and then based on the reward, updates the value function to improve the learning curve in order to maximize reward within optimal steps. The ultimate goal here is to find an optimal policy based on the following approximations of the value function-

• Value function V - The value of a state under a policy, V(s) is given by the expected return when starting in state s and following the given policy.
• Action-Value function Q - The value of taking action a in state s under a policy, Q(s,a) is given by the expected return when starting from state s, taking action a and following the given policy.

As the agent keeps learning, it has to consider past data while predicting future moves. To store all the past data in a compact way in the learning policy of the agent is really important. But as it’s difficult to store all the data, only relevant data necessary for future predictions must be selected and stored. A Markov property in a learning model is a state signal of the model which retains the relevant past information. If the state signal has this Markov property, the response at t+1 should depend only on response at t rather than all the past signals. A learning task that satisfies this Markov property is called a Markov Decision Process (MDP).

## Temporal Difference Learning
Temporal Difference (TD) Learning is a combination of Monte Carlo and Dynamic Programming methods where like Monter Carlo, TD method learns from experience with knowledge
of the environment and like Dynamic Programming, it updates policy in parts rather than at the end of each episode. Based on how the value function is updated, TD Learning follows two major approaches in learning-
• SARSA - This method follows a policy to update the action-value function.
• Q-Learning - This method updates the action-value function independent of the policy.

### SARSA
SARSA is an on-policy TD Control method in which the agent follows a policy π for learning to estimate an action value function qπ(s, a). So when we visualize our agents learning, instead of moving from state to state, as SARSA estimates action value function, the agent moves from state-action pair to state-action pair. The SARSA algorithm then according to some policy decides what action to take next in order to reach the goal with maximum reward, i.e. converging within an optimal time.
As TD Learning updates the policy after every step, the above update is done after each step till we reach the goal step. The algorithm converges based on the greediness of exploring the space which can be chosen while selecting next action. For example, one can choose ϵ-greedy method or ϵ-soft policies to define the greediness of the algorithm to converge.

### Playing with Hyperparameters
There are different parameters which defines the learning curve of a reinforcement learning algorithm. Studying and finding the correct combination of these parameters is another important factor when defining an RL algorithm. The major hyperparameters that drive the SARSA algorithm are-
• Alpha (the learning rate)- This parameters defines how much data to store from previous experience in order to predict future actions. This factor decides how quickly Q value function can change. Having higher value means storing more past data so this factor must be decided carefully in order to avoid contention in the learning process.
• Gamma (Discount factor)- This parameter decides how much value to give to future rewards. The higher the gamma value, more weight is given to future rewards. We can
start with a higher gamma but as we advance in our learning process, this value should be reduced towards the end in order to consider immediate gains.
• Epsilon (Exploration factor)- This factor decided how much to explore and how much to exploit current value function. Its an ideal choice to start with a higher epsilon so that the agent explores in the beginning and then reduce this value towards the end, so that the agent exploits the already explored data to gain better rewards.

## Chemical Space problem

### Mapping the problem to RL
The main idea of the quantum problem that we are trying to solve is to find a combination of molecules to form a ligand with optimal characteristics to bond a central magnetic ion such as Cobalt. The spin properties of such new materials is calculated using quantum chemistry methods. Using these properties with a reinforcement learning model, we will explore the chemical space of molecular qubits. The model will learn by receiving positive feedback from molecules with better properties.

We have used already generated data of combinations of ligands with their spin properties calculated based on quantum chemistry methods. We have a list of 13 molecules and 766 combinations of ligand data. The ligands are formed by combining any 4 molecules with repetitions of the same molecule in any place. For example, we can have a combination with 1 molecule in all four places of the ligand (acetonitrile, acetonitrile, acetonitrile, acetonitrile) or we can have different molecules in all the places as well (chloropyridine, acetonitrile, imidazole, nme3).

One important thing to note here is that the arrangement of the ligands doesn’t matter, i.e. if we have four molecules forming a ligand with some ’xyz’ spin property, then any permutations of those four molecules will still be a ligand with the same ’xyz’ property. To model the quantum problem, I have built three different versions of algorithms using SARSA method and trained and tested them with different parameters.

### Algorithm 1
The algorithm starts by first initializing the Q matrix. We have initialized the Q matrix as a 2-dimensional array, where the rows represent the states and columns represent actions. The Q matrix is initialized with 766 X 13 dimension, where 766 are the combinations of data that we have with their spin properties and 13 are the total number of molecules. If a state is not found in database while running the episode, then it is added at the end of the Q matrix. So everytime the Q matrix expands and we can keep track of all the new combination of ligands. Each episode in the first algorithm works as follows-
• Step 1- Start from first state in our ligand database which is (acetonitrile, acetonitrile, acetonitrile, acetonitrile).
• Step 2- Choose an action by ϵ-greedy method (e.g. water).
• Step 3- Assign the action (molecule) to a random place in ligand combination, e.g. (acetonitrile, acetonitrile, water, acetonitrile). This will form the next state.
• Step 4- If the ligand is found in our 766 combinations data (by checking all the permutations of the 4 molecules in the ligand) then take reward from the database otherwise add the new combination of ligands to database with a reward as -25.
• Step 5- Choose next action.
• Step 6- Update Q value function using SARSA method. If the next state is a new state, then expand the Q matrix to accommodate the new state.
• Step 7- Update the current state and action with next state and action.
• Step 8- Repeat step 3-7 four times. So in each episode, we are taking four steps only. We are doing this so that we train the model to find the best solution by stepping only 4 times for each of the 4 different positions of the ligand.

### Algorithm 1.1
We also built an updated version of the above algorithm to avoid some issues found while testing. In this version, we are starting from a null state (null, null, null, null) and we are repeating steps 2-7 of the above algorithm until a match is found in our original database, i.e. the first 766 ligands. So the steps taken change every time while running each episode.

### Algorithm 2
In this algorithm,we have initialized the Q matrix in a different way. It is again a 2-dimensional array, where the rows represent the states and columns represent actions but it is initialized with (766+1) X 13 dimension. We are adding 1 extra row for state to add all the data of the ligand combinations which were not found in the database. The steps of the algorithm are similar to the first algorithm, except for step 4 where if the ligand is not found in the database, then point the next state to N+1 state in Q matrix. The Q matrix here now has the extra state at the end which will accommodate all the not found combinations and the size of the matrix is fixed.


### Algorithm 2.1
Again, an updated version of this algorithm starts from a null state (null, null, null, null) and we are repeating all the steps of the second algorithm until a match is found in our database. So here again, the steps taken change every time while running each episode.

### Algorithm 3
This algorithm is an N-step SARSA algorithm for the chemical space problem. N-step SARSA is a combination of TD(0) and Monte Carlo method. In this algorithm, the estimate of qπ is not updated after every step, but it is updated after n-step. For example, in 3-step SARSA (TD(3)), the Q matrix is updated after 3 steps rather than updating after every step. So like Monte Carlo which updates qπ at the end of the episode, the learning method waits for n-steps before updating. This method gives an efficient and incremental way to select between Monte Carlo and TD(0) method. As Monte-Carlo methods are said to have advantage in some non-Markovian tasks, TD(λ) methods can be used to overcome drawbacks in TD(0) methods. We did try to implement this algorithm for the chemical space problem, but we didn’t see any improvement in the learning behaviour of the agent. But I felt it is worth mentioning this algorithm as it can be considered for some future work.

## Results and Discussion

### Train and Test with different Hyperparameters
To find the best optimal combination of ligands, we trained the model with above described algorithms and tested with different parameters. We ran around 100 episodes of training with different values of hyperparameters and then tested with 10 episodes. We mapped the data on three different graphs to check the efficiency and compare the different algorithms. We are mapping the train/test data of total rewards, cumulative rewards and maximum of Q(s,a) against number of episodes.

![image](https://github.com/AditiRupade/Learning-shortcuts-in-chemical-space-using-Reinforcement-Learning/assets/30768250/762c4889-812f-4800-8a8a-935221b6ca89)

The first algorithm didn’t show much improvement in terms of rewards gain or improving Q function as seen above, this is when we realised that there were few loopholes
in the way we built the algorithms, so in the updated version, we started from null state and then kept searching till we find a valid ligand combination. The issue with first version was that even though we found a valid ligand combination, as the number of steps taken were constant, the agent was still moving to the next state and then terminating at an invalid ligand combination state. The output from the updated version of first algorithm is shown below where we did observe some improvement in terms of rewards and Q function’s convergence.

![image](https://github.com/AditiRupade/Learning-shortcuts-in-chemical-space-using-Reinforcement-Learning/assets/30768250/678cdc98-31d9-4825-a52d-0a4b5dfdc1cb)

We again observe few better insights in second algorithm in terms of Q function improvement over time and rewards gained with high gamma and epsilon. The blue line in below figure shows some improvements towards the end of training as well as rewards gained over time.

![image](https://github.com/AditiRupade/Learning-shortcuts-in-chemical-space-using-Reinforcement-Learning/assets/30768250/2797110b-9fa3-413a-b9ac-ec89b9d556d9)

The updated version of second algorithm on the other hand, didn’t perform as well as updated version of first algorithm as observed in below figure.

![image](https://github.com/AditiRupade/Learning-shortcuts-in-chemical-space-using-Reinforcement-Learning/assets/30768250/51c5ee7e-0d99-426e-9576-9bcfb69a1090)

### Testing with decaying Epsilon
When we are training a reinforcement learning model, it is always better to have a higher epsilon value so that the agent can explore the environment to gain experience for better future predictions. But as we advance in the algorithm, having higher epsilon will lead to additional costs. If the agent has already gained experience from it’s environment, then it should not keep exploring the environment. It should rather exploit the gained experience to make better predictions. This idea of reducing the value of epsilon as we advance our training is called ’Decaying ϵ’. We have tested our above algorithms by decaying ϵ while training with 1000 episodes and then testing the model with 100 episodes and minimum value of ϵ. 

We didn’t observe any better improvements in Algorithm 1 with this testing as seen in below Figure.

![image](https://github.com/AditiRupade/Learning-shortcuts-in-chemical-space-using-Reinforcement-Learning/assets/30768250/a56143bf-64ae-4eba-8472-82c9281829a4)

 But we did see some improvement in Algorithm 1.2 especially in terms of convergence of Q matrix when gamma is low (Seen in bottom firgure).

![image](https://github.com/AditiRupade/Learning-shortcuts-in-chemical-space-using-Reinforcement-Learning/assets/30768250/3f076dec-9afd-42e3-9929-be6b7056287a)

Similar observations were made for Algorithm 2 and it’s updated version as seen in bottom two figures respectively.

![image](https://github.com/AditiRupade/Learning-shortcuts-in-chemical-space-using-Reinforcement-Learning/assets/30768250/1126757e-e01d-4a49-94c8-50688817761b)

![image](https://github.com/AditiRupade/Learning-shortcuts-in-chemical-space-using-Reinforcement-Learning/assets/30768250/7954877f-d88b-4c6d-9b74-ccca88f1e576)

## Conclusions and Future Work
As the field of electron spin qubits is tremendously growing and showing some amazing advancements in quantum science especially in the implementation of quantum sensing protocols, it has become a major area of research. But due to the fragile nature of spin qubits quantum states and it’s instability when interfaced with other environment, designing these materials is really difficult. But with the help of Reinforcement Learning model, we have developed a machine learning tool to find stable states by combining different molecules to design new materials which are stable and promising. We have built this model using spin properties data of organic ligands to bond a central magnetic ion such as Cobalt. We implemented 3 different algorithms and tested them with different hyperparameters. We observe better convergence of the Q matrix when gamma is low, i.e. when the agent considers immediate rewards rather discounting for future rewards. The updated version of the algorithm where we are starting from null state and stepping till we find an optimal ligand state works better than just stepping for a constant number of steps. We also observed better results when we used ’Decaying ϵ-greedy policy’. So for our problem, an ideal agent should start with a high value of ϵ and then decay the value while advancing in the training process. At the end while testing, the agent should work with lower ϵ value as it should exploit the environment (rather than explore) using the Q matrix generated while training. This agent can similarly be used with any sample of molecule and ligands data to generate an optimal combination of molecules. But as the data size goes on increasing, the model can become slow while training. Maybe some parallel version of Reinforcement learning or a neural network model can be implemented as future to improve the model’s accuracy when scaling the problem size. You can find data on all the current work on this research (as well as for any suggestions on future work) here.
