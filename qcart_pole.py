import gym 
import math 
import numpy as np  

# Supress logger warning from gym.make() / Box class 
gym.logger.set_level(40)

# Intialize hyperparameters. In order, these are: 
# min alpha, min epsilon gamma, number of episodes
# number of buckets per observation (x, x_dot, theta, theta_dot) 
# stored in a tuple 
min_alpha = 0.1 
min_epsilon = 0.1 
gamma = 1.0 
n_episodes = 10000
buckets = (1,1,6,12,)

# Create Cart Pole environment & intialize observations  
env = gym.make('CartPole-v0')

# State-Action Pairs / Q-table.
# Only using theta and theta_dot 
# Creates 5 dimensional  np.ndarray of shape (1, 1, 6, 12, 2)
# Note for later: Changes this to a 72 x 2 matrix instead and figure out how to work with that. 

Q = np.zeros(buckets + (env.action_space.n,))

def get_alpha(value): 
	return max(min_alpha, min(1.0, 1.0 - math.log10((value + 1) / 25))) 

def get_epsilon(value): 
	return max(min_epsilon, min(1, 1.0 - math.log10((value + 1) / 25))) 

def get_state(obs):
	upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)] 
	lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)] 
	ratios = np.multiply(obs + np.abs(lower_bounds), np.divide(1, np.subtract(upper_bounds, lower_bounds)))
	new_obs = (np.around(np.subtract(buckets, 1)*ratios)).astype(int) 
	new_obs = np.minimum(np.subtract(buckets, 1), np.maximum(0, new_obs))
	return tuple(new_obs)

# array holding total reward per episode 
totalReward = np.zeros([n_episodes , 1]) 
for episode in range(n_episodes):
	# Reset environment and get first new state 
	state = get_state(env.reset()) 
	alpha = get_alpha(episode)
	epsilon = get_epsilon(episode)
	done = False 
	# Tabular Q-Learning algorithm with epislon-greedy 
	while not done:
		# Choose action via epislon-greedy  
		if np.random.random() <= epsilon: 
			action = env.action_space.sample() 
		else: 
			action = np.argmax(Q[state]) 
		
		# Get new observations, reward, and whether we're done from enviornment 
		new_obs, reward, done, _ = env.step(action) 
		new_state = get_state(new_obs) 

		# Update Q-table 
		Q[state][action] += alpha*(reward + gamma * np.max(Q[new_state]) - Q[state][action])  
		
		# Count total reward and update state 
		totalReward[episode] += 1 
		state = new_state

print("The average reward is {}".format(np.mean(totalReward)))
