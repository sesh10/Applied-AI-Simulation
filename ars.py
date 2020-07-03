# AI 2020

# libraries
import os
import numpy as np

# hyper parameters
class Hp():
    
    def __init__(self):
        self.num_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.num_direction = 16
        self.num_best_dir = 16
        self.noise = 0.03   
        self.seed = 1
        self.env_name = ''
        
        assert self.num_best_dir <= self.num_direction


# Normailze states

class Normalize():
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)
        
    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
        
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std_dev = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std_dev
    
    
# Building the AI

class Policy():
    def __init__(self, input_size, output_size):
        self.theta = np.zeros({output_size, input_size})
        
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == 'positive':
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)
        
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.num_direction)]

    def update(self, rollout, std_dev_r):
        step = np.zeros(self.theta.shape)
        for pos_reward, neg_reward, d in rollout:
            step += (pos_reward - neg_reward) * d
            
        self.theta += ( hp.learning_rate / hp.num_best_dir * std_dev_r) * step
        

# exploring the ai(policy) on one specific direction

def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    num_action_plays = 0.
    reward_sum = 0
    
    while not done and num_action_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        reward_sum += reward
        num_action_plays += 1
        
    return reward_sum
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
