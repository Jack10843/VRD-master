import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
from utils import ReplayBuffer, calculate_norm
import numpy as np
from flow import Flow_one


def weights_init_(m, init_w=3e-3):
    m.weight.data.uniform_(-init_w, init_w)
    m.bias.data.uniform_(-init_w, init_w)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, num_layers=2, init_w=None):
        super(Actor, self).__init__()

        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), )
        for i in range(num_layers):
            self.trunk.add_module(f"{2 + i * 2}", nn.Linear(hidden_dim, hidden_dim))
            self.trunk.add_module(f"{3 + i * 2}", nn.ReLU())

        self.l = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

        if init_w:
            weights_init_(self.l, init_w=init_w)

    def forward(self, state):
        a = self.l(self.trunk(state))
        if self.max_action is not None:
            action = self.max_action * torch.tanh(a)
        else:
            action = a
        return action

    # def train_pi(self, replay_buffer, optimizer, batch, mod):
    #     state, action, ind = replay_buffer.sample_density(batch, mod)
    #     a = self.forward(state)
    #     loss = F.mse_loss(a, action)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()


class parallelized_layer(nn.Module):
    def __init__(self, in_dim, out_dim, ensemble=2):
        super().__init__()
        w_init, b_init = [], []
        for _ in range(ensemble):
            l = nn.Linear(in_dim, out_dim)
            w_init.append(l.weight.data.transpose(0, 1))
            b_init.append(l.bias.data.unsqueeze(0))
        w_init = torch.stack(w_init, dim=0)
        b_init = torch.stack(b_init, dim=0)

        self.weight = nn.Parameter(w_init, requires_grad=True)
        self.bias = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        return x @ self.weight + self.bias


class Parallelized_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2, init_w=None, ensemble=2):
        super(Parallelized_Critic, self).__init__()
        # parallelized architecture
        self.ensemble = ensemble
        self.t = nn.Sequential(parallelized_layer(state_dim + action_dim, hidden_dim, ensemble),
                               nn.ReLU(),
                               # parallelized_layer_norm(hidden_dim, ensemble),
                               )
        for i in range(num_layers):
            self.t.add_module(f"{3 + i * 3}", parallelized_layer(hidden_dim, hidden_dim, ensemble))
            self.t.add_module(f"{4 + i * 3}", nn.ReLU())
            # self.t.add_module(f"{5 + i * 3}", parallelized_layer_norm(hidden_dim, ensemble))
        self.l = parallelized_layer(hidden_dim, 1, ensemble)

        if init_w:
            weights_init_(self.l, init_w=init_w)

    def forward(self, state, action):
        sa = torch.cat([state, action], -1).repeat(self.ensemble, 1, 1)

        q = self.l(self.t(sa))
        return q[0], q[1]

    def Q1(self, state, action):
        sa = torch.cat([state, action], -1).repeat(self.ensemble, 1, 1)

        q = self.l(self.t(sa))
        return q[0]


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2, init_w=None):
        super(Critic, self).__init__()
        # Q1 architecture
        self.t1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                nn.ReLU(),
                                nn.LayerNorm(hidden_dim),)
        for i in range(num_layers):
            self.t1.add_module(f"{3 + i * 3}", nn.Linear(hidden_dim, hidden_dim))
            self.t1.add_module(f"{4 + i * 3}", nn.ReLU())
            self.t1.add_module(f"{5 + i * 3}", nn.LayerNorm(hidden_dim))
        self.l1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.t2 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                nn.ReLU(),
                                nn.LayerNorm(hidden_dim),)
        for i in range(num_layers):
            self.t2.add_module(f"{3 + i * 3}", nn.Linear(hidden_dim, hidden_dim))
            self.t2.add_module(f"{4 + i * 3}", nn.ReLU())
            self.t2.add_module(f"{5 + i * 3}", nn.LayerNorm(hidden_dim))
        self.l2 = nn.Linear(hidden_dim, 1)

        if init_w:
            weights_init_(self.l1, init_w=init_w)
            weights_init_(self.l2, init_w=init_w)

    def forward(self, state, action):
        sa = torch.cat([state, action], -1)

        q1 = self.l1(self.t1(sa))
        q2 = self.l2(self.t2(sa))
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], -1)

        q1 = self.l1(self.t1(sa))
        return q1


class ADC_TD3(object):
    def __init__(
            self,
            device,
            env,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            warmup_time=1e5,
            switch_time=1e5,
            # actor-critic
            lr=3e-4,
            actor_lr=None,
            num_layers=2,
            actor_hidden_dim=256,
            critic_hidden_dim=256,
            actor_init_w=None,
            critic_init_w=None,
            Vmax=float('inf'),
            Vmin=-float('inf'),
            # density estimation
            density=None,
            density_optimizer=None,
            flow_steps=10,
            lambd=1.0,
            max_type='z',
            iw_elbo=False,
            # finetune
            lambd_cool=False,
            lambd_end=0.2,
    ):
        self.total_it = 0
        self.device = device
        self.env = env
        # Actor
        self.actor = Flow_one(state_dim, action_dim, max_action, device, actor_hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr or lr)
        # Critic
        self.critic = Parallelized_Critic(state_dim, action_dim, critic_hidden_dim, num_layers, critic_init_w).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # RL
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = policy_noise*2.5 # noise_clip
        self.policy_freq = policy_freq
        self._Vmax = Vmax
        self._Vmin = Vmin

        # density estimation
        self.warmup_time = warmup_time
        self.density = density
        self.density_optimizer = density_optimizer
        self.lambd = lambd
        # self.test_pi = 'flow'
        # self.max_type = max_type
        # self.switch_time = switch_time
        self.norm = calculate_norm(tau, 1e5, device)

        # fine-tuning
        self.lambd_cool = lambd_cool
        self.lambd_end = lambd_end

    @torch.no_grad()
    def select_action(self, state):
        self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        noise = torch.randn_like(state[:, :self.action_dim]).clamp(-1, 1) * 1e-5
        action = self.actor(state, noise).cpu().data.numpy().flatten()
        self.actor.train()
        return action

    def train(self, replay_buffer: ReplayBuffer, batch_size=256, logger=None):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select target action according to flow policy or deterministic policy
            next_action = self.actor(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = (target_Q1 + target_Q2) / 2 if 'antmaze' in self.env else torch.min(target_Q1, target_Q2)
            target_Q = (reward + not_done * self.discount * target_Q).clamp(min=self._Vmin, max=self._Vmax)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Log
        logger.log('train/critic_loss', critic_loss, self.total_it)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:  # Compute actor loss
            noise = torch.randn_like(action).clamp(-1, 1)

            bc_pi = self.density(state, noise)
            with torch.no_grad():

                current_Q1, current_Q2 = self.critic(state, bc_pi)
                q_data = torch.min(current_Q1, current_Q2)  # Conservative Q

            w = (q_data - self._Vmin) / (self._Vmax - self._Vmin + 1e-8)
            r_min, r_max = 1e-8, 1.0 
            target_radius = r_min + (r_max - r_min) * (1 - w)
            raw_noise = noise 
            direction = raw_noise / (raw_noise.norm(dim=-1, keepdim=True) + 1e-8)

            structured_noise = target_radius * direction
            noise = structured_noise.clamp(-1, 1)

            pi = self.actor(state, noise)

            neg_log_pi = F.mse_loss(pi, bc_pi) #((u - bc_u) ** 2).mean()

            Q, Q2 = self.critic(state, pi)
            Q = (Q + Q2) / 2 if 'antmaze' in self.env else Q

            self.norm.update(Q.abs().mean().detach())
            actor_loss = - Q.mean() / self.norm.mu + self.lambd * neg_log_pi

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Log
            logger.log('train/Q', Q.mean(), self.total_it)
            logger.log('train/actor_loss', actor_loss, self.total_it)
            logger.log('train/neg_log_pi', neg_log_pi, self.total_it)

            #  kill for diverging
            if Q.abs().mean().item() > 1e4:
                exit(0)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update Flow matching model
        if self.total_it <= self.warmup_time:
            density_loss = self.density.train_density(state, action, self.density_optimizer)

            self.density_optimizer.zero_grad()
            density_loss.backward()
            self.density_optimizer.step()
            logger.log('train/density_loss', density_loss, self.total_it)

    def train_online(self, replay_buffer: ReplayBuffer, density_optimizer, batch_size=256, logger=None):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = (reward + not_done * self.discount * target_Q).clamp(min=self._Vmin, max=self._Vmax)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Compute the current Q value, inspired by ATAC for accelerating training
        if self.DQRA:
            with torch.no_grad():  # Unlike ATAC, we fix current Q target。
                dqra_Q1, dqra_Q2 = self.critic(next_state, next_action)
                dqra_Q1 = (reward + not_done * self.discount * dqra_Q1).clamp(min=self._Vmin, max=self._Vmax)
                dqra_Q2 = (reward + not_done * self.discount * dqra_Q2).clamp(min=self._Vmin, max=self._Vmax)
            critic_loss += F.mse_loss(current_Q1, dqra_Q1) + F.mse_loss(current_Q2, dqra_Q2)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Log
        logger.log('train/critic_loss', critic_loss, self.total_it)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)

            neg_log_pi = self.density.elbo_loss(state, pi)

            if self.lambd_cool:
                lambd = self.lambd * max(self.lambd_end, (1 - self.total_it / (self.lambd_end * 1e6)))
                logger.log('train/lambd', lambd, self.total_it)
            else:
                lambd = self.lambd

            actor_loss = - Q.mean() / Q.abs().mean().detach() + lambd * neg_log_pi.mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Optimize the reference policy
            if not self.lambd_cool:
                recon, mean, std = self.density.forward(state, pi.detach())
                recon_loss = F.mse_loss(recon, pi.detach())
                KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                density_loss = recon_loss + KL_loss

                density_optimizer.zero_grad()
                density_loss.backward()
                density_optimizer.step()
                logger.log('train/density_loss', density_loss, self.total_it)

            # Log
            logger.log('train/Q', Q.mean(), self.total_it)
            logger.log('train/actor_loss', actor_loss, self.total_it)
            logger.log('train/neg_log_pi', neg_log_pi.mean(), self.total_it)
            logger.log('train/neg_log_pi_max', neg_log_pi.max(), self.total_it)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, model_dir, type=None):
        type = type or self.total_it
        torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_s{str(type)}.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(model_dir, f"critic_target_s{str(type)}.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(
            model_dir, f"critic_optimizer_s{str(type)}.pth"))

        torch.save(self.actor.state_dict(), os.path.join(model_dir, f"actor_s{str(type)}.pth"))
        # torch.save(self.actor_target.state_dict(), os.path.join(model_dir, f"actor_target_s{str(type)}.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(
            model_dir, f"actor_optimizer_s{str(type)}.pth"))
        torch.save(self.density.state_dict(), os.path.join(model_dir, f"density_s{str(type)}.pth"))

    def load(self, model_dir, step=1000000):
        self.critic.load_state_dict(
            torch.load(os.path.join(model_dir, f"critic_s{str(step)}.pth"), map_location=self.device))
        self.critic_target.load_state_dict(
            torch.load(os.path.join(model_dir, f"critic_target_s{str(step)}.pth"), map_location=self.device))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(model_dir, f"critic_optimizer_s{str(step)}.pth"), map_location=self.device))

        self.actor.load_state_dict(
            torch.load(os.path.join(model_dir, f"actor_s{str(step)}.pth"), map_location=self.device))
        # self.actor_target.load_state_dict(
        #     torch.load(os.path.join(model_dir, f"actor_target_s{str(step)}.pth"), map_location=self.device))
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(model_dir, f"actor_optimizer_s{str(step)}.pth"), map_location=self.device))
        self.density.load_state_dict(torch.load(os.path.join(model_dir, f"density_s{str(step)}.pth"), map_location=self.device))
