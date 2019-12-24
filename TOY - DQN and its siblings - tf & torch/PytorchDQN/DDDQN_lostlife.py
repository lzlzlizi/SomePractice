import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torchvision

import itertools
from collections import namedtuple

import os
import numpy as np
from gym.wrappers import Monitor
import time
import gym
import os
import random
import sys
import plotting

from tensorboardX import SummaryWriter
from attention_augmented_conv import AugmentedConv
from lib.replay_buffer import ReplayBuffer


env = gym.envs.make("Breakout-v0")

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
optimizer_spec = OptimizerSpec(
    constructor=optim.RMSprop,
    kwargs=dict(lr=0.00025, alpha=0.99, eps=10e-6),
)

class DQN_Dueling(nn.Module):
    def __init__(self, in_channels=12, num_action=4):
        super(DQN_Dueling,self).__init__()
        self.num_action = num_action
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.A1 = nn.Linear(7*7*64, 512)
        self.V1 = nn.Linear(7*7*64, 512)

        self.A2 = nn.Linear(512, num_action)
        self.V2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0),-1)

        A = F.relu(self.A1(x))
        V = F.relu(self.V1(x))

        A = self.A2(A)
        V = self.V2(V)
        
        x = V + A - A.mean(1).unsqueeze(1)

        return x



def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal(m.weight,gain=2)
        if m.bias: 
            torch.nn.init.xavier_uniform_(m.bias, gain=2)

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


def select_epilson_greedy_action(model, obs, eps_threshold):
        sample = random.random()
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).permute((2,0,1)).type(dtype).unsqueeze(0) / 255.0
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            return model(Variable(obs)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(model.num_action)]])


from skimage.transform import resize
from skimage.color import rgb2gray


def pre_proc(X):
    return np.uint8(resize(rgb2gray(X), (84, 84), mode='reflect') * 255)


num_episodes = 10000
experiment_dir = '../experiments/torch/DDDQN_LIFE'
replay_memory_size=200000
replay_memory_init_size=20000
update_target_estimator_every=10000
discount_factor=0.99
epsilon_start=1.0
epsilon_end=0.1
epsilon_decay_steps=200000
batch_size=32
record_video_every=50
gamma = 0.99


# init model
num_actions = 4
in_channel = 4
frame_history_len = 4

q_estimator = DQN_Dueling(in_channel, num_actions).type(dtype)
target_estimator = DQN_Dueling(in_channel, num_actions).type(dtype)
weight_init(q_estimator)
weight_init(target_estimator)


# Construct Q network optimizer function
optimizer = optimizer_spec.constructor(q_estimator.parameters(), **optimizer_spec.kwargs)


# Keeps track of useful statistics
stats = plotting.EpisodeStats(
    episode_lengths=np.zeros(num_episodes),
    episode_rewards=np.zeros(num_episodes))

# Create directories for checkpoints and summaries
checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "model")
monitor_path = os.path.join(experiment_dir, "monitor")
summary_path = os.path.join(experiment_dir, "summary")

writer = SummaryWriter(summary_path)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(monitor_path):
    os.makedirs(monitor_path)


# Get the current time step
total_t = 0

# The epsilon decay schedule
epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

# The replay memory
replay_memory = []

# init replay buffer
print('Start initializing replay buffer...')
state = env.reset()
state = pre_proc(state)
state = np.stack( [state] * 4, axis=2 )
life = 0
for t in range(replay_memory_init_size):
    # take a step
    action = random.randrange(num_actions)
    next_state, reward, done, info = env.step(action)
    # clip rewards between -1 and 1 ??
    reward = max(-1.0, min(reward, 1.0))
    next_state = pre_proc(next_state)
    next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
    if info['ale.lives'] < life:
        die = True
    else:
        die = done
    life = info['ale.lives']
    # If our replay memory is full, pop the first element
    replay_memory.append( Transition( state, action, reward, next_state, die))
    if done:
        state = env.reset()
        state = pre_proc(state)
        state = np.stack( [state] * 4, axis=2 )
        life = 0
    else:
        state = next_state
print('Initialize replay buffer: done!')

# Record videos
env= Monitor(env,
                directory=monitor_path,
                resume=True,
                video_callable=lambda count: count % record_video_every == 0)
total_t = 0
for i_episode in range(num_episodes):
    loss = None
    state = env.reset()
    state = pre_proc(state)
    state = np.stack( [state] * 4, axis=2 )
    life = 0
    # One step in the environment
    for t in itertools.count():
        # Choose random action if not yet start learning
        epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
        action = select_epilson_greedy_action(q_estimator, state, epsilon)
        
        next_state, reward, done, info = env.step(action)
        # clip rewards between -1 and 1 ??
        reward = max(-1.0, min(reward, 1.0))
        next_state = pre_proc(next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        if info['ale.lives'] < life:
            die = True
        else:
            die = done
        life = info['ale.lives']
        if len(replay_memory) == replay_memory_size:
            replay_memory.pop(0)
        replay_memory.append( Transition( state, action, reward, next_state, die))

        # Epsilon for this time step
        
        # Add epsilon to Tensorboard
        writer.add_scalar("epsilon", epsilon,total_t)

        ###########################################################

        # Print out which step we're on, useful for debugging.
        print("\rStep {} ({}) @ Episode {}/{}, loss: {} ".format(
                t, total_t, i_episode + 1, num_episodes, loss,), end="")
        sys.stdout.flush()

        # Update statistics
        stats.episode_rewards[i_episode] += reward
        stats.episode_lengths[i_episode] = t


        minibatch = np.array(random.sample(replay_memory, batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*minibatch))
        # Convert numpy nd_array to torch variables for calculation
        state_batch = Variable(torch.from_numpy(state_batch).permute((0,3,1,2)).type(dtype) / 255.0)
        action_batch = Variable(torch.from_numpy(action_batch).long())
        reward_batch = Variable(torch.from_numpy(reward_batch)).type(dtype)
        next_state_batch = Variable(torch.from_numpy(next_state_batch).permute((0,3,1,2)).type(dtype) / 255.0)
        done_batch = np.invert(done_batch)
        done_batch = Variable(torch.from_numpy(done_batch)).type(dtype)

        if USE_CUDA:
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()

        # calculate qt_error 
        q = q_estimator(state_batch).gather(1, action_batch.unsqueeze(1))
        next_max_q = target_estimator(next_state_batch).max(1)[0]
        target_Q_values = reward_batch + gamma * done_batch * next_max_q
  
        optimizer.zero_grad()
        loss = F.smooth_l1_loss(q.reshape(-1), target_Q_values)
        loss.backward()
        

        # and train
        optimizer.step()
        

        # update target_estimator
        if total_t % update_target_estimator_every == 0:
                target_estimator.load_state_dict(q_estimator.state_dict())


        if done:
            state = env.reset()
            state = pre_proc(state)
            state = np.stack( [state] * 4, axis=2 )
            life = 0
            break            
        
        state = next_state
        total_t += 1

     #Add summaries to tensorboard
    
    writer.add_scalar('episode_reward',stats.episode_rewards[i_episode], total_t)
    writer.add_scalar('episode_length',stats.episode_lengths[i_episode],total_t)
   
    print(stats.episode_rewards[i_episode])

env.close()
writer.close()


# change to adam with lr = 10^-6
# hubor loss
# initialize scale = 2
# record if die instead if done
