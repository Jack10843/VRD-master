import numpy as np
import torch
import os
import random
import imageio
from tqdm import trange


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.size = 0
        self.ptr = 0
        self.ptrD_ = 0

        self.state = np.empty((max_size, state_dim))
        self.action = np.empty((max_size, action_dim))
        self.next_state = np.empty((max_size, state_dim))
        self.reward = np.empty((max_size, 1))
        self.not_done = np.empty((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):  # RL training sample
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.state[ind]).float().to(self.device),
            torch.from_numpy(self.action[ind]).float().to(self.device),
            torch.from_numpy(self.next_state[ind]).float().to(self.device),
            torch.from_numpy(self.reward[ind]).float().to(self.device),
            torch.from_numpy(self.not_done[ind]).float().to(self.device),)

    def convert_D4RL(self, dataset, standardize=False):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        rewards = dataset['rewards']
        if standardize:
            # self.reward = ((dataset['rewards'] - dataset['rewards'].min()) / (
            #             dataset['rewards'].max() - dataset['rewards'].min())).reshape(-1, 1)

            # Robust Normalization
            median = np.median(rewards)
            Q1 = np.percentile(rewards, 25)
            Q3 = np.percentile(rewards, 75)
            IQR = Q3 - Q1
            if IQR == 0:
                self.reward = (rewards - median).reshape(-1, 1)
            else:
                self.reward = ((rewards - median) / IQR).reshape(-1, 1)
        else:
            self.reward = rewards.reshape(-1, 1)

        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]
        self.ptrD_ = self.state.shape[0]

    import numpy as np

    def convert_mix_D4RL(self, dataset1, dataset2, ratio, standardize=False):

        total_size = dataset1['observations'].shape[0]


        size1 = int(total_size * ratio)
        size2 = total_size - size1


        idx1 = np.random.choice(dataset1['observations'].shape[0], size1, replace=False)
        idx2 = np.random.choice(dataset2['observations'].shape[0], size2, replace=False)


        self.state = np.concatenate([dataset1['observations'][idx1], dataset2['observations'][idx2]], axis=0)
        self.action = np.concatenate([dataset1['actions'][idx1], dataset2['actions'][idx2]], axis=0)
        self.next_state = np.concatenate([dataset1['next_observations'][idx1], dataset2['next_observations'][idx2]],
                                         axis=0)
        rewards = np.concatenate([dataset1['rewards'][idx1], dataset2['rewards'][idx2]], axis=0)
        terminals = np.concatenate([dataset1['terminals'][idx1], dataset2['terminals'][idx2]], axis=0)


        if standardize:
            # Robust Normalization
            median = np.median(rewards)
            Q1 = np.percentile(rewards, 25)
            Q3 = np.percentile(rewards, 75)
            IQR = Q3 - Q1
            if IQR == 0:
                self.reward = (rewards - median).reshape(-1, 1)
            else:
                self.reward = ((rewards - median) / IQR).reshape(-1, 1)
        else:
            self.reward = rewards.reshape(-1, 1)

        self.not_done = 1. - terminals.reshape(-1, 1)


        self.size = self.state.shape[0]
        self.ptrD_ = self.state.shape[0]

    def convert_D4RL_finetune(self, dataset):
        self.ptr = dataset['observations'].shape[0]
        self.size = dataset['observations'].shape[0]
        self.state[:self.ptr] = dataset['observations']
        self.action[:self.ptr] = dataset['actions']
        self.next_state[:self.ptr] = dataset['next_observations']
        self.reward[:self.ptr] = dataset['rewards'].reshape(-1, 1)
        self.not_done[:self.ptr] = 1. - dataset['terminals'].reshape(-1, 1)


    def noretention(self, num, max_num):
        self.size = min(num, self.size)
        self.max_size = min(max_num, self.max_size)
        self.ptr = self.size % self.max_size
        self.state = self.state['observations'][-num:]
        self.action = self.action['actions'][-num:]
        self.next_state = self.next_state['next_observations'][-num:]
        self.reward = self.reward['rewards'][-num:].reshape(-1, 1)
        self.not_done = 1. - self.not_done['terminals'][-num:].reshape(-1, 1)

    def normalize_states(self, eps=1e-3):
        mean = self.state[:self.size].mean(0, keepdims=True)
        std = self.state[:self.size].std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std

    def clip_to_eps(self, eps=1e-6): # due to tanh
        lim = 1 - eps
        self.action = np.clip(self.action, -lim, lim)


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed_everywhere(env, args):
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # torch.backends.cudnn.benchmark = True


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def snapshot_src(src, target, exclude_from):
    make_dir(target)
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")


class VideoRecorder(object):
    def __init__(self, dir_name, height=512, width=512, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.sim.render(
                # mode='rgb_array',
                height=self.height,
                width=self.width,
                # camera_id=self.camera_id
            )
            self.frames.append(frame[::-1,:,:])

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)


def init_density(density_optimizer, args):
    if getattr(init_density, 'has_not_run', True):
        init_density.has_not_run = False
        for pam in density_optimizer.param_groups:
            pam['lr'] = args.density_lr


def warmup(density, density_optimizer, replay_buffer, args):
    print('Warm-up')
    for _ in trange(3000):
        state, action, _, _, _ = replay_buffer.sample(args.batch_size)
        density.train_density(state, action, density_optimizer)

def transform_to_parallelism(parall_critic, critic):
    a = parall_critic.state_dict()
    b = critic.state_dict()

    b1,b2 = [],[]
    for k,v in b.items():
        b1.append(v)
    b2 = b1[int(len(b)/2):]
    b1 = b1[:int(len(b)/2)]

    c = []
    for src1,src2 in zip(b1,b2):
        if len(src1.shape)>1:
            p = torch.stack([src1, src2], dim=0).permute(0, 2, 1)
        else:
            p = torch.stack([src1, src2], dim=0).unsqueeze(1)
        c.append(p)

    with torch.no_grad():
        for trg, src in zip(a.values(), c):
            trg.data.copy_(src)

    parall_critic.load_state_dict(a)
    parall_critic.train()
    return parall_critic

class calculate_norm:
    def __init__(self, alpha, init_mu=None, device='cpu'):
        self.alpha = alpha
        self.mu = torch.tensor(init_mu, device=device)
        self.device = device

    def update(self, current_mu):
        self.mu = self.alpha * current_mu + (1 - self.alpha) * self.mu
