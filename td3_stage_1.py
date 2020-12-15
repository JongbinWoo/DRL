#!/usr/bin/env python

import rospy
import os
import random
import numpy as np
from collections import deque
import sys
import gc
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from std_msgs.msg import Float32
from environment_stage_1 import Env

dirPath = os.path.dirname(os.path.realpath(__file__))

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor ,self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.fc_a1 = nn.Linear(hidden_size, 1)
        self.fc_a2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        a1 = torch.sigmoid(self.fc_a1(x)) * 0.22
        a2 = torch.tanh(self.fc_a2(x)) * 2
        if list(a1.size())[0] == 1:
            policy = torch.cat([a1,a2])
        else:
            policy = torch.cat([a1,a2],dim=1)            

        return policy

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        #Q1
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        #Q2
        self.fc4 = nn.Linear(state_size+action_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size,hidden_size)
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
    


class OUNoise:
    def __init__(self, action_size, theta, mu, sigma):
        self.action_size = action_size
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.X = np.zeros(self.action_size) 

    def sample(self):
        dx = self.theta * (self.mu - self.X) + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        self.X[0] = self.X[0] / 2.
        
        return self.X

def get_action(policy, ou_noise,episode): 
    noise = ou_noise.sample()
    #print(noise)

    if episode > 80 :
        noise = noise * 0.7
    elif 400> episode > 200:
        noise = noise * 0.8
    elif episode > 400:
        noise = noise * 0.8    

    action = policy.detach().numpy() #+ noise
    action[0] = np.clip(action[0], 0., 0.22)
    action[1] = np.clip(action[1], -2., 2.)
    return action

def get_exploit_action(policy):
    action = policy.detach().numpy()
    action[0] = np.clip(action[0], 0., 0.22)
    action[1] = np.clip(action[1], -2., 2.)
    return action

def hard_target_update(actor, critic, target_actor, target_critic):
    target_critic.load_state_dict(critic.state_dict())
    target_actor.load_state_dict(actor.state_dict())

def soft_target_update(actor, critic, target_actor, target_critic, tau):
    soft_update(critic, target_critic, tau)
    soft_update(actor, target_actor, tau)

def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def train_model(actor, critic, target_actor, target_critic, 
                actor_optimizer, critic_optimizer, mini_batch,step):
    mini_batch = np.array(mini_batch)
    states = np.vstack(mini_batch[:, 0])
    actions = list(mini_batch[:, 1])
    rewards = list(mini_batch[:, 2])
    next_states = np.vstack(mini_batch[:, 3])
    masks = list(mini_batch[:, 4])

    states = torch.Tensor(states)
    actions = torch.Tensor(actions).squeeze(1)
    rewards = torch.Tensor(rewards)#.squeeze(1)
    masks = torch.Tensor(masks)
    next_states = torch.Tensor(next_states)

    # update critic 
    criterion = torch.nn.MSELoss()
    
    # get Q-value
    q_value1, q_value2 = critic(states, actions)
    
    # get target
    target_next_policy = target_actor(next_states)
    target_next_q_value1, target_next_q_value2 = target_critic(next_states, target_next_policy)
    min_target_next_q_value = torch.min(target_next_q_value1, target_next_q_value2)
    target = rewards + masks * gamma * min_target_next_q_value.squeeze(1)

    critic_loss1 = criterion(q_value1.squeeze(1), target.detach())
    critic_optimizer.zero_grad()
    critic_loss1.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), gradient_clip_critic)
    critic_optimizer.step()

    critic_loss2 = criterion(q_value2.squeeze(1), target.detach())
    critic_optimizer.zero_grad()
    critic_loss2.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), gradient_clip_critic)
    critic_optimizer.step()

    if step % 2 ==0:
    # update actor 
        policy = actor(states)
        actor_loss = -critic(states, policy)[0].mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip_actor)
        actor_optimizer.step()

def save_models(episode_count):
    torch.save(target_actor.state_dict(), dirPath +'/test_td3_2/'+str(episode_count)+ '_actor.pth')
    torch.save(target_critic.state_dict(), dirPath + '/test_td3_2/'+str(episode_count)+ '_critic.pth')
    print('****Models saved***')
        
def load_models(episode):
    actor.load_state_dict(torch.load(dirPath + '/test_td3_2/'+str(episode)+ '_actor.pth'))
    critic.load_state_dict(torch.load(dirPath + '/test_td3_2/'+str(episode)+ '_critic.pth'))

    print('***Models load***')

state_size = 14
action_size =2
gamma = 0.99
hidden_size = 512
batch_size = 128
actor_lr = 1e-4
critic_lr = 1e-3
theta = 0.15
mu = 0.0
sigma = 0.3
tau = 0.001
gradient_clip_actor = 0.5
gradient_clip_critic = 1.0

is_training = True

actor = Actor(state_size, action_size)
target_actor = Actor(state_size, action_size)
critic = Critic(state_size, action_size)
target_critic = Critic(state_size, action_size)
load_models(1060)        
def main():
    rospy.init_node('ddpg_stage_1')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env()
    torch.manual_seed(1000)


    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    hard_target_update(actor, critic, target_actor, target_critic)
    ou_noise = OUNoise(action_size, theta, mu, sigma)

    writer = SummaryWriter('./house_td3_4')
    
    replay_buffer = deque(maxlen=100000)
    recent_rewards = []
    

    for episode in range(100001):
        done = False
        score = 0.
        state = env.reset()
        print('Episode: ' + str(episode))
        past_action = np.array([0.,0.])
        
        for step in range(1000):
            
            state = np.float32(state)
            #print(state)
            policy = actor(torch.Tensor(state))
            action = get_action(policy, ou_noise,episode)
            
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
            
            if len(replay_buffer) >= 2*batch_size  and is_training:
		
                mini_batch = random.sample(replay_buffer, batch_size)
                
                actor.train(), critic.train()
                target_actor.train(), target_critic.train()
                train_model(actor, critic, target_actor, target_critic, 
                            actor_optimizer, critic_optimizer, mini_batch,step)
                
                soft_target_update(actor, critic, target_actor, target_critic, tau)
                
        result = score
        pub_result.publish(result)
        gc.collect()
        print('reward per ep: ' + str(score))
                
	#if episode % 10 == 0:
        #    print('{} episode | score_avg: {:.2f}'.format(episode, np.mean(recent_rewards)))
        #    writer.add_scalar('log/score', float(np.mean(recent_rewards)), episode+320)
        #    recent_rewards = []
        #    print("save")
        #if episode % 20 == 0:
        #   save_models(episode+320)


if __name__ == '__main__':
    main()
