import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w_final = nn.Linear(state_dim, embed_dim)
        self.v = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        # agent_qs: [bs, t, n_agents]
        # states: [bs, t, state_dim]
        bs, t, _ = agent_qs.shape

        flat_agent_qs = agent_qs.view(-1, 1, self.n_agents)
        flat_states = states.reshape(-1, self.state_dim)

        w1 = torch.abs(self.hyper_w1(flat_states)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(flat_states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(flat_agent_qs, w1) + b1)

        w_final = torch.abs(self.hyper_w_final(flat_states)).view(-1, self.embed_dim, 1)
        v = self.v(flat_states).view(-1, 1, 1)

        y = torch.bmm(hidden, w_final) + v
        return y.view(bs, t, 1)
