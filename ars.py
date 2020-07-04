# AI 2020

# libraries
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

# hyper parameters
class Hyperparam():
    
    def __init__(self):
        self.num_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.num_direction = 16
        self.num_best_dir = 16
        self.noise = 0.03   
        self.seed = 1
        self.env_name = "HalfCheetahBulletEnv-v0"
        
        assert self.num_best_dir <= self.num_direction


# Normailze states

class Normalizer():
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
        self.theta = np.zeros((output_size, input_size))
        
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)
        
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.num_direction)]

    def update(self, rollout, std_dev_r):
        step = np.zeros(self.theta.shape)
        for pos_reward, neg_reward, d in rollout:
            step += (pos_reward - neg_reward) * d
            
        self.theta += (( hp.learning_rate )/ (hp.num_best_dir * std_dev_r)) * step
        

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


# AI Training

def train(env, policy, normalizer, hp):
    for step in range(hp.num_steps):
        deltas = policy.sample_deltas()
        pos_rewards = [0] * hp.num_direction
        neg_rewards = [0] * hp.num_direction
        
        # getting +ve rewards
        for i in range(hp.num_direction):
            pos_rewards[i] = explore(env, normalizer, policy, direction="positive", delta=deltas[i])
            
        # getting -ve rewards
        for i in range(hp.num_direction):
            neg_rewards[i] = explore(env, normalizer, policy, direction="negative", delta=deltas[i])
            
        # Calculate standard deviation of rewards obtained
        all_rewards = np.array(pos_rewards + neg_rewards)
        std_dev_r = all_rewards.std()
        
        # Sorting rollouts and select best direction
        scores = {i: max(r_pos, r_neg) for i, (r_pos, r_neg) in enumerate(zip(pos_rewards, neg_rewards))}
        order = sorted(scores.keys(), key = lambda x: scores[x], reverse = True)[:hp.num_best_dir]
        rollout = [(pos_rewards[i], neg_rewards[i], deltas[i]) for i in order]
        
        # Updating policy values
        policy.update(rollout, std_dev_r)
        
        # Final reward display
        eval_reward = explore(env, normalizer, policy)
        print('Step: ', step, 'Reward: ', eval_reward)
        

# Result folder

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')


# Object creation with main code

hp = Hyperparam()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)
env = wrappers.Monitor(env, monitor_dir, force = True)
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]
policy = Policy(num_inputs, num_outputs)
normalizer = Normalizer(num_inputs)
train(env, policy, normalizer, hp)




