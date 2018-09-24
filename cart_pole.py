import gym 

gym.logger.set_level(40) # Suppresses warnings 

env = gym.make('CartPole-v0')  
 

def basic_policy(obs): 
	angle = obs[2]
	return 0 if angle < 0 else 1 
 
for _  in range(500):
	obs = env.reset()
	for _ in range(1000): 
	    action = basic_policy(obs)
	    env.render() 
	    obs, reward, done, info  = env.step(action) 
	    if done: 
	      break  

