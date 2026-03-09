import torch 
import torch.nn as nn
import numpy as np
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data.dataloader import DataLoader
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, neur_cnt, astr_cnt, in_size, out_size, gamma=0.1, tau=0.01):
        super().__init__()

        self.n = neur_cnt
        self.m = astr_cnt
        self.k = in_size
        self.o = out_size
        
        self.gamma = gamma
        self.tau = tau

        self.C = nn.Parameter(torch.Tensor(self.n * self.n))
        self.D = nn.Parameter(torch.Tensor(self.n * self.n, self.m))
        self.F = nn.Parameter(torch.Tensor(self.m))
        self.H = nn.Parameter(torch.Tensor(self.m, self.n * self.n))

        self.W_in_1 = nn.Parameter(torch.Tensor(self.n, self.k))
        self.W_in_2 = nn.Parameter(torch.Tensor(self.m, self.k))

        with torch.no_grad():
            self.C.normal_(std = 1. / np.sqrt(self.n * self.n))
            self.D.normal_(std = 1. / np.sqrt(self.n * self.n))
            self.F.normal_(std = 1. / np.sqrt(self.m))
            self.H.normal_(std = 1. / np.sqrt(self.m))
            self.W_in_1.uniform_(-1. / np.sqrt(self.k), 1. / np.sqrt(self.k))
            self.W_in_2.uniform_(-1. / np.sqrt(self.k), 1. / np.sqrt(self.k))

    def phi(self, x):
        return torch.sigmoid(x)

    def Phi(self, x):
        return (self.phi(x) @ self.phi(x).reshape(1, -1)).reshape(-1, 1)

    def psi(self, z):
        return torch.tanh(z)

    def forward(self, I, hidden=None):
        if hidden is None:
            curr_device = self.C.device 
            x = torch.zeros(self.n, 1, device=curr_device)
            z = torch.zeros(self.m, 1, device=curr_device)
            W = torch.eye(self.n, device=curr_device) * 0.01 
        else:
            x, W, z = hidden
        
        C_term = self.C.unsqueeze(1) * self.Phi(x)
        F_term = self.F.unsqueeze(1) * self.psi(z)
        
        x = (1 - self.gamma) * x + self.gamma * W @ self.phi(x) + (self.W_in_1 @ I)
        
        W = (1. - self.gamma) * W + self.gamma * (C_term + self.D @ self.psi(z)).reshape(self.n, self.n)
        
        z = (1. - self.gamma * self.tau) * z + self.gamma * self.tau * (F_term + self.H @ self.Phi(x) + self.W_in_2 @ I)

        hidden = (x, W, z)
        return x, hidden


class NetModule(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.net = Model(neur_cnt=128, astr_cnt=64, in_size=in_size, out_size=out_size)
        self.lt = nn.Linear(128, out_size) 

    def forward(self, x, hidden):
        x, hidden = self.net(x, hidden)
        y = self.lt(x.flatten())
        return y, hidden


class Learning():
    def __init__(self, in_size, out_size, device="cpu"):
        self.device = torch.device(device)
        self.model = NetModule(in_size, out_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.actions_vals = torch.ones(out_size, device=self.device)

    def train(self, data, bptt_steps=20):
        self.model.train()

        avg_reward = 0.0
        regret_rec = []

        cumulative_regret = 0.0
        cumulative_regret_rec = []

        state = None
        loss = 0.0
        
        self.optimizer.zero_grad()

        for t, (rewards, def_probs) in enumerate(data):
            # Move data to target device
            rewards = rewards.to(self.device)
            def_probs = def_probs.to(self.device)

            state, c, action, reward, regret = self.get_sample_data(state, def_probs, rewards)

            # REINFORCE objective: accumulate loss
            baseline = avg_reward
            loss += - (reward - baseline) * c.log_prob(action)

            # Update baseline incrementally
            avg_reward = (avg_reward * t + reward) / (t + 1)

            print(t, ": ", regret)

            cumulative_regret += regret
            cumulative_regret_rec.append(cumulative_regret)
            regret_rec.append(regret)

            # FIX: Truncated BPTT - Only update weights and detach state every N steps
            if (t + 1) % bptt_steps == 0 or (t + 1) == len(data):
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss = 0.0  # Reset loss for the next chunk
                
                # Detach state to prevent backpropagating through the entire history
                state = tuple(var.detach() for var in state)

        return cumulative_regret_rec
    
    def get_sample_data(self, state, orig_probs, rewards):
        I = torch.ones(1, 1, device=self.device) 
        logits, state = self.model(I, state)

        c = Categorical(logits=logits)
        action = c.sample()

        reward = rewards[0, action].item() if self.actions_vals[action].item() > 0 else 0.0
        
        # Calculate regret
        max_prob = torch.max(orig_probs).item()
        chosen_prob = orig_probs[0, action].item()
        regret = max_prob - chosen_prob

        return state, c, action, reward, regret


class SBDataset():
    def __init__(self, samples_num, actions_num):
        self.samples_num = samples_num
        self.Mu = torch.rand(actions_num)
        
    def __getitem__(self, item):
        return Bernoulli(probs=self.Mu).sample(), self.Mu
    
    def __len__(self):
        return self.samples_num


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    ACTIONS_NUM = 6
    SAMPLES_NUM = 400 

    dataset = SBDataset(SAMPLES_NUM, ACTIONS_NUM)
    data = DataLoader(dataset, batch_size=1, shuffle=False)

    L = Learning(in_size=1, out_size=ACTIONS_NUM, device=device)

    cumulative_regrets = L.train(data, bptt_steps=20)

    plt.plot(np.arange(len(cumulative_regrets)), cumulative_regrets)
    plt.xlabel("Training Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Multi-Armed Bandit (Neuron-Astrocyte Net)")
    plt.grid(True)
    plt.show()