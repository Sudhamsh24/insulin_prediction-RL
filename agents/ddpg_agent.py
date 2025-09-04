"""#CPU BASED 
# agents/ddpg_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Deep Deterministic Policy Gradient (DDPG) Network Definitions ---
class Critic(nn.Module): # Q-Network
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

class Actor(nn.Module): # Policy Network
    def __init__(self, state_dim, action_dim, n_latent_var, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        return self.max_action * torch.tanh(self.layer_3(x))

# --- DDPG Agent Implementation (TD3-like for stability) ---
class DDPGAgent:
    def __init__(self, env, state_dim, action_dim, n_latent_var, lr, gamma, tau):
        self.gamma = gamma
        self.tau = tau
        self.max_action = float(env.action_space.high[0])

        self.actor = Actor(state_dim, action_dim, n_latent_var, self.max_action)
        self.actor_target = Actor(state_dim, action_dim, n_latent_var, self.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Using two critics to mitigate overestimation bias
        self.critic_1 = Critic(state_dim, action_dim, n_latent_var)
        self.critic_1_target = Critic(state_dim, action_dim, n_latent_var)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = Critic(state_dim, action_dim, n_latent_var)
        self.critic_2_target = Critic(state_dim, action_dim, n_latent_var)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()
        action = action + noise * np.random.randn(action.shape[0])
        return action.clip(-self.max_action, self.max_action)

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            # Use the minimum of the two critics to avoid overestimation bias
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update actor
        actor_loss = -self.critic_1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update for target networks
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)"""





#GPU BASED T4
# agents/ddpg_agent.py

# agents/ddpg_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Deep Deterministic Policy Gradient (DDPG) Network Definitions ---
class Critic(nn.Module): # Q-Network
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

class Actor(nn.Module): # Policy Network
    def __init__(self, state_dim, action_dim, n_latent_var, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        return self.max_action * torch.tanh(self.layer_3(x))

# --- DDPG Agent Implementation (TD3-like for stability) ---
class DDPGAgent:
    def __init__(self, env, state_dim, action_dim, n_latent_var, lr, gamma, tau, device):
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.max_action = float(env.action_space.high[0])

        self.actor = Actor(state_dim, action_dim, n_latent_var, self.max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, n_latent_var, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action = action + noise * np.random.randn(action.shape[0])
        return action.clip(-self.max_action, self.max_action)

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        actor_loss = -self.critic_1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)