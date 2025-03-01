import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np

import functools
import operator
import random
import gym_duckietown
import argparse
# import pandas as pd
import sys
sys.path.append("../gym-duckietown/")
import learning.reinforcement.pytorch.ddpg as ddpg
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from learning.utils.env import launch_env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPOCriticCNN(ddpg.CriticCNN):
    def __init__(self):
        super(PPOCriticCNN, self).__init__(0)
    def forward(self, states):
        # From learning/reinforcement/pytorch/ddpg.py remove actions
        x = self.bn1(self.lr(self.conv1(states)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(x))
        x = self.lin3(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, max_action):
        super(ActorCritic, self).__init__()
        self.actor = ddpg.ActorCNN(action_dim, max_action)
        self.critic = PPOCriticCNN()
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        # Don't allow duckie to go backwards or spin in place
        action[:,0] = torch.clamp(action[:,0], min=0)
        action[:,1] = torch.clamp(action[:,1], min=-1, max=1)
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = torch.squeeze(self.actor(state))
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, max_action, batch_size):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        
        self.policy = ActorCritic(state_dim, action_dim, action_std, max_action).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std,max_action).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        #if flat
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state = np.array(state)
        state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update_batch(self, memory, rewards):
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(1):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        batches = []
        total_samples = len(memory.states)
        # Create batches
        for start in range(0, total_samples, self.batch_size):
            last = min(total_samples, start+self.batch_size)
            batch = Memory()
            batch.actions = memory.actions[start:last]
            batch.states = memory.states[start:last]
            batch.rewards = memory.rewards[start:last]
            batch.logprobs = memory.logprobs[start:last]
            batch.is_terminals = memory.is_terminals[start:last]
            batches.append([batch, rewards[start:last]])
        random.shuffle(batches)
        for _ in range(self.K_epochs):
            for batch in batches:
                self.update_batch(batch[0], batch[1])
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def _main(args):
    ############## Hyperparameters ##############
    # env_name = "BipedalWalker-v2"
    env_name = 'Duckietown-loop_empty-v0'
    render = False
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = None

    print(args)
    #############################################
    
    # creating environment
    env = launch_env()
    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")
    # state_dim = env.observation_space.shape[0]
    state_dim = env.observation_space.shape
    state_dim = functools.reduce(operator.mul, state_dim, 1)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, args.action_std, lr, betas, args.gamma, args.K_epochs, args.eps_clip, max_action, args.batch_size)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    episode_reward = 0
    # stats = pd.DataFrame(columns = ["Episode", "Length", "Reward"])
    stats = []
    with open("PPO_stats.csv", 'w') as statsfile:
        statsfile.write("Epoch, Timesteps, Reward\n")
    # training loop
    for i_episode in range(1, args.max_episodes+1):
        state = env.reset()
        for t in range(args.max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if time_step % args.update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
        # stats = stats.append({"Episode" : i_episode, "Length" : t, "Reward" : episode_reward}, ignore_index=True)
        stats.append( (i_episode, t, episode_reward) )
        running_reward += episode_reward
        episode_reward = 0
        
        if i_episode % args.store_interval == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
            # stats.to_csv("PPO_stats.csv", index=False) #This line does not work on Google Colab!
            with open("PPO_stats.csv", 'a') as statsfile:
                for eps, ts, rwd in stats:
                    statsfile.write("%d, %d, %f\n"%(eps, ts, rwd) )
            stats = []
            
        # logging
        if i_episode % args.log_interval == 0:
            avg_length = int(avg_length/args.log_interval)
            running_reward = int((running_reward/args.log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_interval", default = 10, type = int, help = "print avg reward in the interval")
    parser.add_argument("--max_episodes", default = 1000, type = int, help = "max training episodes")
    parser.add_argument("--max_timesteps", default = 1500, type = int, help = "max timesteps in one episode")
    parser.add_argument("--update_timestep", default = 1000, type = int, help = "update policy every n timesteps")
    parser.add_argument("--action_std", default = 0.05, type = float, help = "constant std for action distribution (Multivariate Normal)")
    parser.add_argument("--K_epochs", default = 20, type = int, help = "update policy for K epochs")
    parser.add_argument("--eps_clip", default = 0.2, type = float, help = "clip parameter for PPO")
    parser.add_argument("--gamma", default = 0.99, type = float, help = "discount factor")
    parser.add_argument("--batch_size", default = 32, type = int, help = "mini batch size")
    parser.add_argument("--store_interval", default = 200, type = int, help = "interval for storing a backup of the model")

    _main(parser.parse_args())
