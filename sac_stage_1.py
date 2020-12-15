#!/usr/bin/env python

import rospy
import os
import random
import argparse
import numpy as np
from collections import deque
import sys
import gc
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter

from std_msgs.msg import Float32
from environment_stage_1 import Env

dirPath = os.path.dirname(os.path.realpath(__file__))

class Actor(nn.Module):
    def __init__(self, state_size, action_size, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.fc3 = nn.Linear(hidden_size, action_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        mu = self.fc3(x)
        log_std = self.fc4(x)
        
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)

        return mu, std

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()

        # Q1 architecture
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(state_size + action_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, 1)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)

        x1 = torch.relu(self.fc1(x))
        x1 = torch.relu(self.fc2(x1))
        q_value1 = self.fc3(x1)

        x2 = torch.relu(self.fc4(x))
        x2 = torch.relu(self.fc5(x2))
        q_value2 = self.fc6(x2)

        return q_value1, q_value2

def get_action(mu, std): 
    normal = Normal(mu, std)
    #print(mu, std)
    z = normal.rsample() # reparameterization trick (mean + std * N(0,1))
    action = torch.tanh(z)
    action[0] = (action[0]+1) * 0.22 / 2
    action[1] *= 2
    return action.data.numpy()

def eval_action(mu, std, epsilon_c=1e-6):
    normal = Normal(mu, std)
     
    z = normal.rsample() # reparameterization trick (mean + std * N(0,1))
    action = torch.tanh(z)
        
    
    log_prob = normal.log_prob(z)

    # Enforcing Action Bounds
    log_prob -= torch.log(1 - action.pow(2) + epsilon_c)
    log_policy = log_prob.sum(1, keepdim=True)

    return action, log_policy



def hard_target_update(net, target_net):
    target_net.load_state_dict(net.state_dict())

def soft_target_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def save_models(episode_count):
    torch.save(actor.state_dict(), dirPath + '/test_sac_1/' + str(episode_count)+ '_policy_net_512.pth')
    torch.save(critic.state_dict(), dirPath + '/test_sac_1/'+ str(episode_count)+ 'soft_q_net_512.pth')
    torch.save(target_critic.state_dict(), dirPath + '/test_sac_1/' + str(episode_count)+ 'target_value_net_512.pth')
    print("====================================")
    print("Model has been saved...")
    print("====================================")

def load_models(episode):
    actor.load_state_dict(torch.load(dirPath + '/test_sac_1/' + str(episode)+ '_policy_net_512.pth'))
    critic.load_state_dict(torch.load(dirPath + '/test_sac_1/'+ str(episode)+ 'soft_q_net_512.pth'))
    target_critic.load_state_dict(torch.load(dirPath + '/test_sac_1/' + str(episode)+ 'target_value_net_512.pth'))
    print('***Models load***')
"""
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99, action="store_true")
parser.add_argument('--hidden_size', type=int, default=128, action="store_true")
parser.add_argument('--batch_size', type=int, default=128, action="store_true")
parser.add_argument('--actor_lr', type=float, default=1e-4, action="store_true")
parser.add_argument('--critic_lr', type=float, default=1e-3, action="store_true")
parser.add_argument('--alpha_lr', type=float, default=1e-4, action="store_true")
parser.add_argument('--tau', type=float, default=0.005, action="store_true")
parser.add_argument('--log_interval', type=int, default=10, action="store_true")
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory', action="store_true")
args = parser.parse_args()
"""
gamma = 0.99
hidden_size = 512
batch_size = 128
actor_lr = 1e-4
critic_lr = 1e-3
alpha =20.9


tau = 0.005
state_size = 14
action_size = 2

actor = Actor(state_size, action_size)
critic = Critic(state_size, action_size)
target_critic = Critic(state_size, action_size)
load_models(260)

is_training = True

def train_model(actor, critic, target_critic, mini_batch, 
                actor_optimizer, critic_optimizer, 
                target_entropy):
    mini_batch = np.array(mini_batch)
    states = np.vstack(mini_batch[:, 0])
    actions = list(mini_batch[:, 1])
    rewards = list(mini_batch[:, 2])
    next_states = np.vstack(mini_batch[:, 3])
    masks = list(mini_batch[:, 4])

    states = torch.Tensor(states)
    next_states = torch.Tensor(next_states)
    actions = torch.Tensor(actions).squeeze(1)
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)


    # update critic 
    criterion = torch.nn.MSELoss()
    
    # get Q-values using two Q-functions to mitigate overestimation bias
    q_value1, q_value2 = critic(states, actions)

    # get target
    mu, std = actor(next_states)
    next_policy, next_log_policy = eval_action(mu, std)
    target_next_q_value1, target_next_q_value2 = target_critic(next_states, next_policy)
    
    min_target_next_q_value = torch.min(target_next_q_value1, target_next_q_value2)
    min_target_next_q_value = min_target_next_q_value.squeeze(1) - alpha * next_log_policy.squeeze(1)
    target = rewards + masks * gamma * min_target_next_q_value

    critic_loss1 = criterion(q_value1.squeeze(1), target.detach()) 
    critic_optimizer.zero_grad()
    critic_loss1.backward()
    critic_optimizer.step()

    critic_loss2 = criterion(q_value2.squeeze(1), target.detach()) 
    critic_optimizer.zero_grad()
    critic_loss2.backward()
    critic_optimizer.step()

    # update actor 
    mu, std = actor(states)
    policy, log_policy = eval_action(mu, std)

    
    q_value1, q_value2 = critic(states, policy)
    min_q_value = torch.min(q_value1, q_value2)
    
    actor_loss = ((alpha * log_policy) - min_q_value).mean() 
    actor_optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    actor_optimizer.step()
    


    
def main():
    rospy.init_node('sac_stage_1')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env()
    torch.manual_seed(500)
    

    
    
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    hard_target_update(critic, target_critic)
    
    # initialize automatic entropy tuning
    target_entropy = -torch.prod(torch.Tensor(action_size)).item()
    
    writer = SummaryWriter('./house_sac_4')

    replay_buffer = deque(maxlen=100000)
    recent_rewards = []
    
    for episode in range(10001):
        done = False
        score = 0.
        state = env.reset()
        print('Episode: ' + str(episode))
        past_action = np.array([0.,0.])

        for step in range(1000):
            state = np.float32(state)
            #print(state)
            mu, std = actor(torch.Tensor(state))
            action = get_action(mu, std)
            #action = np.array([np.clip(action[0], 0., 0.22),
             #                         np.clip(action[1], -2., 2.)])
           
            next_state, reward, done = env.step(action,past_action)
            print(action, reward)
            past_action = action
            
            next_state = np.float32(next_state)
            mask = 0 if done else 1
            if step > 1:
		score += reward
            	replay_buffer.append((state, action, reward, next_state, mask))

            state = next_state

            if done:
                recent_rewards.append(score)
                break
            
            if len(replay_buffer) >= 2*batch_size and is_training:

                mini_batch = random.sample(replay_buffer, batch_size)
                
                actor.train(), critic.train(), target_critic.train()
                alpha = train_model(actor, critic, target_critic, mini_batch, 
                                    actor_optimizer, critic_optimizer,
                                    target_entropy)
               
                soft_target_update(critic, target_critic, tau)
            
        result = score
        pub_result.publish(result)
        gc.collect()
        print('reward per ep: ' + str(score))
        
        if episode % 10 == 0:
            print('{} episode | score_avg: {:.2f}'.format(episode, np.mean(recent_rewards)))
            writer.add_scalar('log/score', float(np.mean(recent_rewards)),  episode+260)
            #writer.add_scalar('log/alpha', float(alpha.detach().numpy()),episode+260)
            recent_rewards = []
            print("save")
        if episode % 10 == 0:
           save_models(episode+260)
       
if __name__ == '__main__':
    main()
