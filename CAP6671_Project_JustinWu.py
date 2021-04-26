# CAP6671 - Final Project
# Deep Q-learning - Super Mario Bros
# Justin Wu
# Due: April 26th, 2021
# References:
#           - <https://github.com/roclark/super-mario-bros-dqn>
#           - <https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html>
#           - <https://www.statworx.com/de/blog/using-reinforcement-learning-to-play-super-mario-bros-on-nes-using-tensorflow/>
#           - <https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/>
#           - <https://pypi.org/project/gym-super-mario-bros/>
#           - <https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html>
#--------------------------------------------

import random, time, datetime, sys, os, math 

import collections # 
import shutil # 
import glob # 
import winsound # 
import matplotlib.pyplot as plt # 
import numpy as np # 
import cv2 # 

import gym # 
from nes_py.wrappers import JoypadSpace # 
import gym_super_mario_bros # 
from gym_super_mario_bros.actions import RIGHT_ONLY

import torch # 
import torch.nn as nn#

#-------------------------------------------------------------
# Predefined Constraints

ENVIRONMENT = 'SuperMarioBros-1-1-v0'
# ENVIRONMENT = 'SuperMarioBros-1-2-v0'
# ENVIRONMENT = 'SuperMarioBros-1-3-v0'
# ENVIRONMENT = 'SuperMarioBros-1-4-v0'
PRETRAINED_MODELS = 'pretrained_models'
ACTION_SPACE = RIGHT_ONLY

                                # Originial Values
BATCH_SIZE = 32                 # 32 
MEMORY_CAPACITY = 20000         # 20000 
BETA_FRAMES = 10000             # 10000 
BETA_START = 0.4                # 0.4 
EPSILON_START = 1.0             # 1.0 
EPSILON_FINAL = 0.01            # 0.01 
EPSILON_DECAY = 100000          # 100000 
GAMMA = 0.99                    # 0.99 
LEARNING_RATE = 1e-4            # 1e-4 
NUM_EPISODES = 1000             # 50000 

INITIAL_LEARNING = 100          # 10000
TARGET_UPDATE_FREQUENCY = 50    # 1000

TRANSFER = False
RENDER = False

# Global Variables
train_win = 0
step_list = []
episode_reward_list = []
epsilon_list = []
average_list = []

test_count = 0
test_win = 0
test_reward_list = []
test_steps_list = []

#-------------------------------------------------------------

# Every action the agent makes is repeated over 4 frames
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


# The size of each frame is reduced to 84×84
class FrameDownsample(gym.ObservationWrapper):
    def __init__(self, env):
        super(FrameDownsample, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(84, 84, 1),
                                                dtype=np.uint8)
        self._width = 84
        self._height = 84

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame,
                           (self._width, self._height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


# Only every fourth frame is collected by the buffer
class FrameBuffer(gym.ObservationWrapper):
    def __init__(self, env, num_steps, dtype=np.float32):
        super(FrameBuffer, self).__init__(env)
        obs_space = env.observation_space
        self._dtype = dtype
        self.observation_space = gym.spaces.Box(obs_space.low.repeat(num_steps, axis=0),
                                                obs_space.high.repeat(num_steps, axis=0),
                                                dtype=self._dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low,
                                    dtype=self._dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


# The frames are converted to PyTorch tensors
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(obs_shape[::-1]),
                                            dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


# The frames are normalized so that pixel values are between 0 and 1 
class NormalizeFloats(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']

        if done:
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0

        return state, reward / 10.0, done, info


def wrap_env(ENVIRONMENT, ACTION_SPACE , monitor=False, iteration=0):
    env = gym_super_mario_bros.make(ENVIRONMENT)
    env = JoypadSpace(env, ACTION_SPACE)

    # if monitor:
    #     env = gym.wrappers.Monitor(env, 'recording/run%s' % iteration, force=True)

    env = MaxAndSkipEnv(env)
    env = FrameDownsample(env)
    env = ImageToPyTorch(env)
    env = FrameBuffer(env, 4)
    env = NormalizeFloats(env)
    env = CustomReward(env)

    return env

#-------------------------------------------------------------

class CNN_Model(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(CNN_Model, self).__init__()

        self._input_shape = input_shape
        self._num_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_out(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
    
    def act(self, state, epsilon, device):
        if random.random() > epsilon:
            state = torch.FloatTensor(np.float32(state)) .unsqueeze(0).to(device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self._num_actions)
        return action

#-------------------------------------------------------------

class PrioritizedBuffer:
    def __init__(self, capacity, alpha=0.6):
        self._alpha = alpha
        self._capacity = capacity
        self._buffer = []
        self._position = 0
        self._priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self._priorities.max() if self._buffer else 1.0

        batch = (state, action, reward, next_state, done)
        if len(self._buffer) < self._capacity:
            self._buffer.append(batch)
        else:
            self._buffer[self._position] = batch

        self._priorities[self._position] = max_prio
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size, beta=0.4):
        if len(self._buffer) == self._capacity:
            prios = self._priorities
        else:
            prios = self._priorities[:self._position]

        probs = prios ** self._alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self._buffer), batch_size, p=probs)
        samples = [self._buffer[idx] for idx in indices]

        total = len(self._buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self._priorities[idx] = prio

    def __len__(self):
        return len(self._buffer)

#-------------------------------------------------------------

def compute_td_loss(model, target_net, replay_buffer, gamma, device, batch_size, beta):
    batch = replay_buffer.sample(batch_size, beta)
    state, action, reward, next_state, done, indices, weights = batch

    state = torch.autograd.Variable(torch.FloatTensor(np.float32(state))).to(device)
    next_state = torch.autograd.Variable(torch.FloatTensor(np.float32(next_state))).to(device)
    action = torch.autograd.Variable(torch.LongTensor(action)).to(device)
    reward = torch.autograd.Variable(torch.FloatTensor(reward)).to(device)
    done = torch.autograd.Variable(torch.FloatTensor(done)).to(device)
    weights = torch.autograd.Variable(torch.FloatTensor(weights)).to(device)

    q_values = model(state)
    next_q_values = target_net(next_state)

    q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

# Decaying Epsilon Greedy
def update_epsilon(episode):
    epsilon = EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * math.exp(-1 * ((episode + 1) / EPSILON_DECAY))
    return epsilon

def update_beta(episode):
    beta = BETA_START + episode * (1.0 - BETA_START) / BETA_FRAMES
    return min(1.0, beta)

def load_model(model, target_model):
    model_name = os.path.join(PRETRAINED_MODELS, '%s.dat' % ENVIRONMENT)
    model.load_state_dict(torch.load(model_name))
    target_model.load_state_dict(model.state_dict())

    return model, target_model

def initialize_models(env, device):
    model = CNN_Model(env.observation_space.shape, env.action_space.n).to(device)
    target_model = CNN_Model(env.observation_space.shape, env.action_space.n).to(device)

    if TRANSFER:
        model, target_model = load_model(model, target_model)

    return model, target_model

#-------------------------------------------------------------

class TrainInformation:
    def __init__(self):
        self._average = 0.0
        self._best_reward = -float('inf')
        self._best_average = -float('inf')
        self._rewards = []
        self._average_range = 100
        self._index = 0
        self._new_best_counter = 0

    @property
    def best_reward(self):
        return self._best_reward

    @property
    def best_average(self):
        return self._best_average

    @property
    def average(self):
        avg_range = self._average_range * -1
        return sum(self._rewards[avg_range:]) / len(self._rewards[avg_range:])

    @property
    def index(self):
        return self._index

    @property
    def new_best_counter(self):
        return self._new_best_counter

    def update_best_counter(self):
        self._new_best_counter += 1

    def _update_best_reward(self, episode_reward):
        if episode_reward > self.best_reward:
            self._best_reward = episode_reward
            return True
        return False

    def _update_best_average(self):
        if self.average > self.best_average:
            self._best_average = self.average
            return True
        return False

    def update_rewards(self, episode_reward):
        self._rewards.append(episode_reward)
        x = self._update_best_reward(episode_reward)
        y = self._update_best_average()
        if x or y:
            self.update_best_counter()
        return x or y

    def update_index(self):
        self._index += 1

#-------------------------------------------------------------

def test(environment, action_space, iteration):
    global  test_win

    flag = False
    env = wrap_env(ENVIRONMENT, ACTION_SPACE, monitor=True, iteration=iteration)
    net = CNN_Model(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(os.path.join(PRETRAINED_MODELS, '%s.dat' % environment)))

    total_reward = 0.0
    state = env.reset()

    test_steps = 0

    frames_stuck = 30
    pos_checker = 0
    temp_pos_x = 0
    temp_pos_y = 0
    stuck = False

    while True:
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)

        # Capture video
        show_state(env, iteration, test_steps)

        state, reward, done, info = env.step(action)

        # Check if we're stuck
        if pos_checker == 0:
            temp_pos_x = info['x_pos']
            temp_pos_y = info['y_pos']

        if (pos_checker >= 0) and (pos_checker < frames_stuck):
            pos_checker += 1

        else: # Check our position
            if (temp_pos_x == info['x_pos']) and (temp_pos_y == info['y_pos']):
                print("         Test - We got stuck")
                stuck = True
                done = True
            
            # Reset position checker 
            temp_pos_x = 0
            temp_pos_y = 0
            pos_checker = 0

        if stuck:
            total_reward = total_reward + (reward - 20)
        else:
            total_reward += reward

        test_steps += 1

        if info['flag_get']:
            print('WE GOT THE FLAG!')
            flag = True
            test_win += 1
        if done:
            # If test_reward_list is empty then make best run the first run
            if (len(test_reward_list) == 0): 
                if (os.path.isdir('./recording/best_run/images/')):
                    shutil.rmtree("./recording/best_run/images/")

                shutil.copytree('./recording/temp_run/images/', './recording/best_run/images/')
                shutil.rmtree("./recording/temp_run/images/")

            # Is the best total reward so move temp video capture to current best run 
            elif (total_reward > np.max(test_reward_list)):

                if (os.path.isdir('./recording/best_run/images/')):
                    shutil.rmtree("./recording/best_run/images/")

                shutil.copytree('./recording/temp_run/images/', './recording/best_run/images/')
                shutil.rmtree("./recording/temp_run/images/")

            # Not the best total reward so remove temp video capture
            else: 
                shutil.rmtree("./recording/temp_run/images/")

            test_reward_list.append(total_reward)
            test_steps_list.append(test_steps)
            break

    env.close()
    return flag

#------------------------------------------------------------- 

def show_state(env, ep, frame):
    if not os.path.isdir('./recording/temp_run/images/'):
        os.mkdir('./recording/temp_run/images/')

    path_name = os.path.join('./recording/temp_run/images/')
    plt.figure()
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("Episode: %d" %ep)
    plt.axis('off')
    plt.savefig(path_name + str(frame) + '.jpeg') 
    plt.close()

def update_graph(model, target_model, optimizer, replay_buffer, device, info, beta):
    if len(replay_buffer) > INITIAL_LEARNING:
        if not info.index % TARGET_UPDATE_FREQUENCY:
            target_model.load_state_dict(model.state_dict())
        optimizer.zero_grad()
        compute_td_loss(model, target_model, replay_buffer, GAMMA, device, BATCH_SIZE, beta)
        optimizer.step()

def test_new_model(model, environment, info, action_space, episode):
    global  test_count

    record = False
    if episode == NUM_EPISODES - 1:
        record = True

    torch.save(model.state_dict(), os.path.join(PRETRAINED_MODELS, '%s.dat' % environment))
    flag = test(environment, action_space, info.new_best_counter)
    if flag:
        print("We also got the flag")
        copyfile(os.path.join(PRETRAINED_MODELS, '%s.dat' % environment), 'recording/run%s/%s.dat' % (info.new_best_counter,environment))

    print('         Test Episode %s - Reward: %s, Steps: %s'  
            % (test_count,
                round(test_reward_list[-1], 3),
                test_steps_list[-1]
                )
            )

def complete_episode(model, environment, info, episode_reward, episode, epsilon, stats, action_space, steps):
    global  test_count, train_win
    
    new_best = info.update_rewards(episode_reward)
    
    step_list.append(steps)
    episode_reward_list.append(episode_reward)
    epsilon_list.append(epsilon)
    average_list.append(info.average)

    print('Episode %s - Reward: %s, Steps: %s, Epsilon: %s, Best: %s, Average: %s' 
            % (episode,
                round(episode_reward, 3),
                steps,
                round(epsilon, 3),
                round(info.best_reward, 3),
                round(info.average, 3)
                )
            )

    if new_best:
        print('         New best average reward of %s!' % round(info.best_average, 3))
        test_new_model(model, environment, info, action_space, episode)
        test_count += 1

    elif stats['flag_get']:
        train_win += 1
        info.update_best_counter()
        test_new_model(model, environment, info, action_space, episode)
        test_count += 1

def run_episode(env, model, target_model, optimizer, replay_buffer, device, info, episode, training_mode):
    episode_reward = 0.0
    state = env.reset()
    steps = 0

    frames_stuck = 30
    pos_checker = 0
    temp_pos_x = 0
    temp_pos_y = 0
    stuck = False

    while True:
        epsilon = update_epsilon(info.index)
        if len(replay_buffer) > BATCH_SIZE: 
            beta = update_beta(info.index)
        else:
            beta = BETA_START

        action = model.act(state, epsilon, device)

        if RENDER:
            env.render()

        next_state, reward, done, stats = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        steps += 1

        # Check if we're stuck
        if pos_checker == 0:
            temp_pos_x = stats['x_pos']
            temp_pos_y = stats['y_pos']

        if (pos_checker >= 0) and (pos_checker < frames_stuck):
            pos_checker += 1

        else: # Check our position 
            if (temp_pos_x == stats['x_pos']) and (temp_pos_y == stats['y_pos']):
                print("Training - We got struck")
                stuck = True
                done = True
            
            # Reset position checker 
            temp_pos_x = 0
            temp_pos_y = 0
            pos_checker = 0

        if stuck:
            episode_reward = episode_reward + (reward - 20)
        else:
            episode_reward += reward

        info.update_index()

        update_graph(model, target_model, optimizer, replay_buffer, device, info, beta)

        if done:
            complete_episode(model, ENVIRONMENT, info, episode_reward, episode, epsilon, stats, ACTION_SPACE, steps)
            break

def train(env, model, target_model, optimizer, replay_buffer, device):
    global train_win, test_win
    
    info = TrainInformation()

    training_mode = True
    print("*"*50)
    print("Training Model:")
    print("-"*25)
    
    for episode in range(NUM_EPISODES):
        run_episode(env, model, target_model, optimizer, replay_buffer, device, info, episode, training_mode)
        print("-"*25)
    print("End Training:")

    # print("Training Step list:", step_list)
    # print("Training Reward list:", episode_reward_list)
    # print("Training Epsilon list:", epsilon_list)
    # print("Training Average list:", average_list)
    print("Training Flag Reached:", train_win)
    print("---")
    # print("Test Step list:", test_steps_list)
    # print("Test Reward list:", test_reward_list)
    print("Test Flag Reached:", test_win)

    print("*"*50)

#-------------------------------------------------------------

# Log to output files as well as print in terminal 
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("output.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # Flush method is needed for python 3 compatibility.
        pass    

#-------------------------------------------------------------

def main():

    if os.path.isfile("output.txt"):
        os.remove("output.txt")

    # Logging function
    sys.stdout = Logger()
    
    if not os.path.isdir(PRETRAINED_MODELS):
        os.mkdir(PRETRAINED_MODELS)
    
    if not os.path.isdir('recording'):
        os.mkdir('recording')
        os.mkdir('./recording/best_run')
        os.mkdir('./recording/temp_run')
    else: 
        shutil.rmtree("recording")
        os.mkdir("recording")
        os.mkdir('./recording/best_run')
        os.mkdir('./recording/temp_run')

    train_value = True
    test_value = False

    env = wrap_env(ENVIRONMENT, ACTION_SPACE)

    if train_value:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, target_model = initialize_models(env, device)
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        replay_buffer = PrioritizedBuffer(MEMORY_CAPACITY)
        train(env, model, target_model, optimizer, replay_buffer, device)

    if test_value:
        print("="*50)
        for i in range(0, NUM_EPISODES):
            if i % 50:
                print("-"*25)
                print("Episode", i )
            test(ENVIRONMENT, ACTION_SPACE, i)
        
        print("="*50)

    env.close()
    
    # Create Video out of the best run
    files = glob.glob("./recording/best_run/images/*.jpeg")
    files.sort(key=os.path.getmtime)

    img_array = []
    for filename in files:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter('./recording/best_run/video.avi', fourcc, fps, size)
    
    for j in range(len(img_array)):
        video.write(img_array[j])
    video.release()

    print("Average step",np.mean(step_list))
    print("Average score",np.mean(episode_reward_list))
    print("Average test step",np.mean(test_steps_list))
    print("Average test step",np.mean(test_reward_list))

    if train_value: 
        x_axis = [x for x in range(NUM_EPISODES)]
        plt.title("Training - Episodes vs Steps")
        plt.xlabel('# of Episodes')
        plt.ylabel('# of Steps')
        plt.plot(x_axis, step_list)
        plt.savefig("./recording/0_Episodes_vs_Steps.jpeg")
        plt.close()
        
        plt.title("Training - Episodes vs Reward")
        plt.xlabel('# of Episodes')
        plt.ylabel('Reward')
        plt.plot(x_axis, episode_reward_list)
        plt.savefig("./recording/1_Training_Episodes_vs_Reward.jpeg")
        plt.close()

        plt.title("Training - Episodes vs Average Reward")
        plt.xlabel('# of Episodes')
        plt.ylabel('Average Reward')
        plt.plot(x_axis, average_list)
        plt.savefig("./recording/2_Training_Episodes_vs_Average_Reward.jpeg")
        plt.close()

        plt.title("Training - Episodes vs Epsilon")
        plt.xlabel('# of Episodes')
        plt.ylabel('Epsilon')
        plt.plot(x_axis, epsilon_list)
        plt.savefig("./recording/3_Training_Episodes_vs_Epsilon.jpeg")
        plt.close()

        test_x_axis = [test_x for test_x in range(test_count)]
        plt.title("Test - Episodes vs Steps")
        plt.xlabel('# of Episodes')
        plt.ylabel('# of Steps')
        plt.plot(test_x_axis, test_steps_list)
        plt.savefig("./recording/5_Test_Episodes_vs_Steps.jpeg")
        plt.close()

        plt.title("Test - Episodes vs Reward")
        plt.xlabel('# of Episodes')
        plt.ylabel('Reward')
        plt.plot(test_x_axis, test_reward_list)
        plt.savefig("./recording/4_Test_Episodes_vs_Reward.jpeg")
        plt.close()

    if test_value:
        test_x_axis = [test_x for test_x in range(test_count)]
        plt.title("Test - Episodes vs Steps")
        plt.xlabel('# of Episodes')
        plt.ylabel('# of Steps')
        plt.plot(test_x_axis, test_steps_list)
        plt.savefig("./recording/5_Test_Episodes_vs_Steps.jpeg")
        plt.close()

        plt.title("Test - Episodes vs Reward")
        plt.xlabel('# of Episodes')
        plt.ylabel('Reward')
        plt.plot(test_x_axis, test_reward_list)
        plt.savefig("./recording/4_Test_Episodes_vs_Reward.jpeg")
        plt.close()

    # # Make sounds when done
    # freq = 1000
    # duration = 1000
    # winsound.Beep(freq,duration)
    # winsound.Beep(freq,duration)
    # winsound.Beep(freq,duration)
    # winsound.Beep(freq,duration)
    # winsound.Beep(freq,duration)

if __name__ == '__main__':
    main()