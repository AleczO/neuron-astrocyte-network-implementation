from collections import defaultdict
import torch 
import torch.nn as nn
import numpy as np
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data.dataloader import DataLoader
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

device = "cpu"


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
            x, W, z =  (torch.zeros(self.n, 1),
                        torch.zeros(self.n, self.n),
                        torch.zeros(self.m, 1))
        else :
            x, W, z = hidden
        
        x = (1 - self.gamma) * x + self.gamma * W @ self.phi(x) + (self.W_in_1 @ I)
        W = (1. - self.gamma) * W + self.gamma * (torch.diag(self.C) @ self.Phi(x) + self.D @ self.psi(z)).reshape(self.n, self.n)
        z = (1. - self.gamma * self.tau) * z + self.gamma * self.tau * (torch.diag(self.F) @ self.psi(z) + self.H @ self.Phi(x) + self.W_in_2 @ I)

        hidden = (x, W, z)
        
        return x, hidden


class NetModule(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.net = Model(neur_cnt=128, astr_cnt=64, in_size=in_size, out_size=out_size)
        self.lt = nn.Linear(128, 3)

    def forward(self, x, hidden):
        x, hidden = self.net(x, hidden)
        y = self.lt(x.flatten())
        return y, hidden


class Learning():
    def __init__(self, in_size, out_size):
        super().__init__()
        self.model = NetModule(in_size, out_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.actions_vals = torch.ones(out_size)


    def train(self, data):

        self.model.train()


        avg_reward = 0
        
        regret = 0
        regret_rec = []

        avg_regret = 0

        cumulative_regret = 0
        cumulative_regret_rec = []

        state = None
        
        Loss = 0
        
        for t, (rewards, def_probs) in enumerate(data):
            state, c, action, reward, regret = self.get_sample_data(state, def_probs, rewards)

            Loss += - (reward - avg_reward)  * c.log_prob(action)
            

            Loss.backward()
            self.optimizer.step()

            if isinstance(state, tuple):
                state = tuple(var.detach() for var in state)
            else:
                state = state.detach()

            Loss = 0
            self.optimizer.zero_grad()



            avg_reward = (avg_reward * (t) + reward) / (t + 1)

            print(regret)

            cumulative_regret += regret
            cumulative_regret_rec.append(cumulative_regret)

            regret_rec.append(regret)
            avg_regret = (avg_regret * (t) + regret) /  (t + 1)
            
        print(def_probs)
        return cumulative_regret_rec
    
    
    def get_sample_data(self, state, orig_probs, rewards):

        I = torch.ones(1).unsqueeze(0)
        logits, state = self.model(I, state)

        c = Categorical(logits=logits)
        action = c.sample()

        print(action)

        reward = rewards[0, action]

        reward = rewards[0, action].item() if self.actions_vals[action].item() > 0 else 0
        regret = np.max(orig_probs.numpy()) - (orig_probs.squeeze().numpy())[action]

        return state, c, action, reward, regret

        

class SBDataset():
    def __init__(self, samples_num, actions_num):
        self.samples_num = samples_num
        self.Mu = torch.rand(actions_num)
        
    def __getitem__(self, item):
        return Bernoulli(probs=self.Mu).sample(), self.Mu
    
    def __len__(self):
        return self.samples_num
    




dataset = SBDataset(60, 6)
data = DataLoader(dataset, batch_size=1, shuffle=False, generator=torch.Generator())

L = Learning(1, 6)

cumulative_regrets = L.train(data)

plt.plot(np.arange(len(cumulative_regrets)), cumulative_regrets)
plt.show()

