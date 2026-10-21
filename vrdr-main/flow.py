import torch
import torch.nn.functional as F
from torch import nn
import math
import torch.distributions as td
from torch.distributions import MultivariateNormal


class Flow(nn.Module):
    # Flow matching
    def __init__(self, state_dim, action_dim, max_action, flow_steps, device, hidden_dim=750):
        super(Flow, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim + 1, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)
        self.e3 = nn.Linear(hidden_dim, hidden_dim)
        self.e4 = nn.Linear(hidden_dim, action_dim)

        self.action_dim = action_dim
        self.relu = nn.Mish()
        self.max_action = max_action
        self.flow_steps = flow_steps
        self.device = device

    def net(self, state, x_t, t):
        z = self.relu(self.e1(torch.cat([state, x_t, t], -1)))
        z = self.relu(self.e2(z))
        z = self.relu(self.e3(z))

        return self.e4(z)

    @torch.no_grad()
    def forward(self, state, noise=None):
        action = torch.randn_like(state[:, :self.action_dim]).clamp(-1, 1) if noise is None else noise

        # log_ts = torch.log(torch.arange(self.flow_steps + 1, dtype=torch.float32) + 1)
        # ts = log_ts / log_ts[-1]
        # for i in range(self.flow_steps):
        #     t = torch.ones(state.size(0), 1, device=self.device) * ts[i]
        #     dt = ts[i + 1] - ts[i]
        #     vels = self.net(state, action, t)
        #     action = action + vels * dt

        ts = torch.arange(self.flow_steps, dtype=torch.float32) / self.flow_steps
        for i, j in enumerate(ts): # Euler method.
            t = torch.ones(state.size(0), 1, device=self.device) * j
            vels = self.net(state, action, t)
            action = action + vels / self.flow_steps

        return action if self.max_action is None else action.clamp(-self.max_action, self.max_action)

    def elbo_loss(self, state, action, samples=10):
        steps = (torch.arange(samples, device=self.device) / samples).view(samples, 1, 1).expand(-1, action.size(0), -1)
        actions, states = action.repeat(samples, 1, 1), state.repeat(samples, 1, 1)
        x_0 = torch.randn_like(actions).clamp(-1, 1)
        ts = steps.expand(-1, action.size(0), -1)
        x_ts = (1.0 - ts) * x_0 + ts * actions
        vels = actions - x_0

        u = self.net(states, x_ts, ts)
        density_loss = ((u - vels) ** 2).mean([0, 2])
        return density_loss

    def train_density(self, state, action, density_optimizer):
        x_0 = torch.randn_like(action).clamp(-1, 1)
        t = torch.rand(action.size(0), 1, device=self.device)
        x_t = (1 - t) * x_0 + t * action
        vel = action - x_0

        u = self.net(state, x_t, t)
        density_loss = F.mse_loss(u, vel)

        # density_optimizer.zero_grad()
        # density_loss.backward()
        # density_optimizer.step()
        return density_loss

    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.forward(state).cpu().data.numpy().flatten()
        return action


class Flow_one(nn.Module):
    # One-step Flow matching
    def __init__(self, state_dim, action_dim, max_action, device, hidden_dim=256, elbo_dim=750):
        super(Flow_one, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)
        self.e3 = nn.Linear(hidden_dim, hidden_dim)
        self.e4 = nn.Linear(hidden_dim, action_dim)

        self.action_dim = action_dim
        self.relu = nn.ReLU()
        self.max_action = max_action
        self.device = device

    def net(self, state, x_0):
        z = self.relu(self.e1(torch.cat([state, x_0], -1)))
        z = self.relu(self.e2(z))
        z = self.relu(self.e3(z))

        return self.e4(z)

    def forward(self, state, noise=None):
        noise = torch.randn_like(state[:, :self.action_dim]).clamp(-1, 1) if noise is None else noise
        action = self.net(state, noise)

        return action if self.max_action is None else self.max_action * torch.tanh(action)


    def elbo_loss(self, state, action, iw_elbo, K=2):
        mean, std, log_std = self.encode(state, action)
        mean_k, std_k = mean.unsqueeze(0).expand(K, -1, -1), std.unsqueeze(0).expand(K, -1, -1)
        state_k, action_k = state.unsqueeze(0).expand(K, -1, -1), action.unsqueeze(0).expand(K, -1, -1)
        if iw_elbo:
            z_k = torch.randn_like(std_k).clamp(-1, 1)

            dist_z = torch.distributions.Normal(loc=mean_k, scale=std_k)
            log_prob_z = dist_z.log_prob(z_k).sum(dim=-1)
            dist_0 = torch.distributions.Normal(loc=torch.zeros_like(z_k), scale=torch.ones_like(z_k))
            log_prob_0 = dist_0.log_prob(z_k).sum(dim=-1)
            iw = torch.exp(log_prob_z - log_prob_0)

            a_k = self.forward(state_k, z_k)
            recon_loss = (iw * ((a_k - action_k) ** 2).mean(dim=-1)).mean()
        else:
            z_k = mean_k + std_k * torch.randn_like(std_k)

            a_k = self.decode(state_k, z_k)
            recon_loss = F.mse_loss(a_k, action_k)

        KL_loss = -0.5 * (1 + 2 * log_std - mean.pow(2) - std.pow(2)).mean()
        elbo_loss = recon_loss + KL_loss
        return elbo_loss

    # def iwelbo_loss(self, state, action):
    #     mean, std, log_std = self.encode(state, action)
    #     z = torch.randn_like(std).clamp(-1, 1)
    #
    #     dist_z = torch.distributions.Normal(loc=mean, scale=std)
    #     dist_0 = torch.distributions.Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z))
    #     log_prob_z = dist_z.log_prob(z).sum(-1)
    #     log_prob_0 = dist_0.log_prob(z).sum(-1)
    #     iw = (torch.exp(log_prob_z - log_prob_0))
    #
    #     a = self.forward(state, z)
    #     recon_loss = iw * ((a - action) ** 2).mean(-1)
    #     KL_loss = -0.5 * (1 + 2 * log_std - mean.pow(2) - std.pow(2)).mean(-1)
    #     elbo_loss = recon_loss + KL_loss
    #     return elbo_loss

    # def elbo_loss(self, state, action):
    #     mean, std, log_std = self.encode(state, action)
    #     z = mean + std * torch.randn_like(std)
    #
    #     a = self.decode(state, z)
    #     recon_loss = ((a - action) ** 2).mean()
    #     KL_loss = -0.5 * (1 + 2 * log_std - mean.pow(2) - std.pow(2)).mean()
    #     vae_loss = recon_loss + KL_loss
    #     return vae_loss

    def elbo_train(self, state, action):
        mean, std, log_std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        recon_loss = F.mse_loss(u, action) #((u - action) ** 2).mean(-1)
        KL_loss = -0.5 * (1 + 2 * log_std - mean.pow(2) - std.pow(2)).mean()
        elbo_loss = recon_loss + KL_loss
        return elbo_loss

