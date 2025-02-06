import gymnasium as gym #The gymnasium library is a toolkit for developing and comparing reinforcement learning (RL) algorithms
import math
import random
import matplotlib
import matplotlib.pyplot as plt

#A tuple is a data structure in python that is used to store an ordered, immutable collection of items.
#Tuples group multiple values together, which can be of different data types.
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

env = gym.make("CartPole-v1") #Create and initialise the environment (returns an instance of the requested environment)

#Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion() #Enables interactive mode. When enabled, every call to plt.plot() auto updates and displays the plot without requiring plt.show() at every step.

#If GPU is to be used...
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

#Create a specialised data structure
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
#Creates a class-like object called 'Transition' that stores four fields/attributes.

#state: current state of the environment
#action: the action taken by the agent in this state
#next_state: the resulting state after taking the action
#reward: the reward received after transitioning to the next state

class ReplayMemory(object):

    def __init__(self, capacity): #Initialise
        self.memory = deque([], maxlen=capacity)

    def push(self, *args): #Save a Transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size): #Retrieves a random mini-batch of past transitions for training purposes
        return random.sample(self.memory, batch_size)

    def __len__(self): #Returns the length of transitions in self.memory
        return len(self.memory)

#Use DQN to approximate the Q-value function, helping the agent decide the best action to take in a given state of the environment
class DQN(nn.Module): #nn.Module is the base class for all neural networks in PyTorch

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128) #Input layer
        self.layer2 = nn.Linear(128, 128) #Hidden layer
        self.layer3 = nn.Linear(128, n_actions) #Output layer

    #Called with either one element to determine the next action, or a batch
    #During optimisation. Returns tensor([[left0exp, right0exp]...])
    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

#Hyperparameters (Configuration variables used to define the structure, behaviour and learning process of a machine learning model or algorithm)
BATCH_SIZE = 128 #Number of transitions (state-action-reward tuples) sampled from replay buffer during one training step.
GAMMA = 0.99 #Discount factor for future rewards, controlling the tradeoff between immediate and future rewards.
EPS_START = 0.9 #Initial exploration rate for the epsilon-greedy policy
EPS_END = 0.05 #Final exploration rate for the epsilon-greedy policy
EPS_DECAY = 1000 #Decay rate for the exploration rate (epsilon)
TAU = 0.005 #Soft update rate for updating the target network, determining how much the online network's weights are mixed into the target network every update.
LR = 1e-4 #Learning Rate for the optimiser. Controls the step size at each iteration.

#Get number of actions from gym action space
n_actions = env.action_space.n

#Get the number of state observations
state, info = env.reset()
n_observations = len(state)

#Policy and target networks
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

#The optimizer object is responsible for updating the model's parameters (weights and biases) during training, based on the gradients computed during backpropagation.
#Helps to prevent overfitting by imposing penalty (or decay)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0 #Initialise at step zero

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY) #Calculate the epsilon value (for epsilon-greedy policy)

    steps_done += 1
    #Action Selection Logic...
    if sample > eps_threshold: #The agent exploits
        with torch.no_grad(): #Saves memory and computations while evaluting the policy network
            #t.max(1) will return the largest column of each row
            #second column on max result is index of where max element was
            #found, so we pick action with the larger expected reward
            return policy_net(state).max(1).indices.view(1, 1)
    else: #The agent explores
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        #The torch.tensor() function converts the randomly sampled action into a PyTorch tensor

episode_durations = []

def plot_durations(show_result=False): #Displays the training process in a line graph using matplotlib
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf() #Clears the figure
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    #Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001) # pause for a time so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    #Transpose the batch
    batch = Transition(*zip(*transitions))

    #Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

    #torch.cat() function is a PyTorch function used to concatenate tensors along a specified dimension. (Combines two or more tensors into a single tensor)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #Compute Q(s_t, a), the Q-value for a given state-action pair for every sample in the batch
    state_action_values = policy_net(state_batch).gather(1, action_batch) #.gather() function is used to select the Q-values of the specific actions taken in those

    #Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    #Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    #Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    #Huber Loss is a popular loss function in machine learning. It is a combination of Mean Squared Error (MSE) & Mean Absolute Error (MAE)

    #Optimise the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) #Used for gradient clipping, ensuring that the gradient values during training do not exceed a specified threshold.
    optimizer.step()

def export_model_to_onnx(model, input_size, file_name="policy_net.onnx"):
    """
        Exports the trained PyTorch policy_net to ONNX format.

        Args:
            model (torch.nn.Module): Trained policy network.
            input_size (int): Number of input features (n_observations).
            file_name (str): The file name for the ONNX model.
    """

    dummy_input = torch.randn(1, input_size, device=device) #Batch size = 1

    torch.onnx.export(
        model,
        dummy_input,
        file_name,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Model successfully exported to {file_name}")

#\*****TRAINING LOOP*****/#

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    #Initialise the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0) #Converts an observation into a PyTorch tensor.

        #Store the transition in memory
        memory.push(state, action, next_state, reward)

        #Move to the next state
        state = next_state

        #Soft update of target network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)

# Export trained model to ONNX
export_model_to_onnx(policy_net, n_observations)

plt.ioff()
plt.show()

