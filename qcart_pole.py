import gym  
import numpy as np   

# Supress logger warning from gym.make() / Box class 
gym.logger.set_level(40) 

# Intialize intervals for theta, and theta_dot 
# To build 'boxes' for the state space 
# There are 18 boxes 
five_degrees = (5*np.pi)/180 
one_degree = (np.pi)/180 
fifty_rps = (50*np.pi)/180 
state_space_size = 18 

# Intialize hyperparameters. In order, these are: 
# alpha, gamma, number of episodes, horizon(?) 
learning_rate0 = 0.05
learning_rate_decay = 0.1
discount_rate = 0.1  
n_episodes = 175 
n_iterations = 10000 

# Intialize action space, and the start state. 
# For the action space, move_left = 0 and move_right = 1  
action_space = np.array([0, 1])
state = 0 

# Create Cart Pole environment & intialize observations  
env = gym.make('CartPole-v0') 
env.reset()

# State-Action Pairs / Q-table. Note indexing is Q[col, row]
# Only using theta and theta_dot 
Q = np.zeros([state_space_size, action_space.size])  

# We obtain state_space from the observation_space 
# obs[2] is theta, obs[3] is theta_dot  
def get_state(obs): 
	s = 0
	theta = obs[2] 
	theta_dot = obs[3] 
 	
	# Threshold/Discretization for theta. 
	if(theta < -five_degrees):  pass 
	elif(theta < -one_degree):  s = 1
	elif(theta < 0): 	    s = 2 
	elif(theta < one_degree):   s = 3 
	elif(theta < five_degrees): s = 4 
	else: 			     s = 5 
	
	# Threshold/Discretization for theta_dot. 
	if(theta_dot < -fifty_rps):  pass 
	elif(theta_dot < fifty_rps): s += 6
	else: 		             s += 7 	

# Q-Learning too solve Cart Pole problem. 
for episode in range(n_episodes): 
	for iteration in range(n_iterations):
		action = np.random.choice(action_space)
		obs_new, reward, done, _ = env.step(action)
		state_new = get_state(obs_new)
		learning_rate = learning_rate0 / (1 + iteration*learning_rate_decay)    
		Q[state, action] = (1 - learning_rate)*Q[state,action] + learning_rate * (reward + discount_rate*np.max(Q[state_new])) 
		state = state_new 
		if done: break 

# Display the learning results
s = 0
for _ in range(n_episodes):
	env.reset()  
	for _ in range(n_iterations): 
		action = np.argmax(Q[s, :]) 
		obs_new, _, done, _ = env.step(action) 
		sp = get_state(obs_new) 	
		env.render() 
		s = sp 
		if done: break

print(Q) 
 
'''
def get_reward(obs): 
	max_theta = ((15)*np.pi / 180) 
	min_theta = -max_theta 
	return -1 if obs[2] > max_theta or obs[2] < min_theta else 1  
''' 
