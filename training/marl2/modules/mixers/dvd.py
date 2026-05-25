import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)

        self.proj = nn.Linear(input_dim, self.n_heads * self.hidden_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(1, self.n_heads, 2 * self.hidden_dim))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, hidden):
        # hidden: [batch, n_agents, input_dim]
        batch_size, n_agents, _ = hidden.shape
        projected = self.proj(hidden).view(batch_size, n_agents, self.n_heads, self.hidden_dim)
        projected = projected.permute(0, 2, 1, 3)

        h_i = projected.unsqueeze(3).expand(-1, -1, -1, n_agents, -1)
        h_j = projected.unsqueeze(2).expand(-1, -1, n_agents, -1, -1)
        pair_features = torch.cat([h_i, h_j], dim=-1)

        scores = (pair_features * self.attn.unsqueeze(2).unsqueeze(3)).sum(dim=-1)
        scores = self.leaky_relu(scores)
        weights = F.softmax(scores, dim=-1)
        graph = torch.matmul(weights, projected)
        return F.elu(graph)


class DVDMixer(nn.Module):
    """DVD-style QMIX mixer using agent recurrent hidden states for credit weights."""

    requires_hidden_states = True

    def __init__(
        self,
        n_agents,
        state_dim,
        rnn_hidden_dim,
        embed_dim=64,
        dvd_heads=4,
        gat_embed_dim=32,
        hypernet_layers=1,
        hypernet_embed=64,
        abs_weights=True,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.embed_dim = int(embed_dim)
        self.n_heads = int(dvd_heads)
        self.gat_dim = int(gat_embed_dim)
        self.abs_weights = bool(abs_weights)

        self.gat = MultiHeadGAT(self.rnn_hidden_dim, self.gat_dim, self.n_heads)
        self.hyper_w_1_state = nn.Linear(self.state_dim, self.n_heads * self.embed_dim * self.gat_dim)
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        if int(hypernet_layers) == 1:
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        else:
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, int(hypernet_embed)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hypernet_embed), self.embed_dim),
            )

        self.v = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states, hidden_states):
        # agent_qs: [bs, t, n_agents]
        # states: [bs, t, state_dim]
        # hidden_states: [bs, t, n_agents, rnn_hidden_dim]
        batch_size = agent_qs.shape[0]

        flat_agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        flat_states = states.reshape(-1, self.state_dim)
        flat_hidden = hidden_states.reshape(-1, self.n_agents, self.rnn_hidden_dim)

        graph = self.gat(flat_hidden)
        graph_t = graph.permute(0, 1, 3, 2)

        w1_state = self.hyper_w_1_state(flat_states)
        w1_state = w1_state.view(-1, self.n_heads, self.embed_dim, self.gat_dim)
        w1_heads = torch.matmul(w1_state, graph_t)
        if self.abs_weights:
            w1_heads = torch.abs(w1_heads)
        w1 = w1_heads.mean(dim=1).permute(0, 2, 1)

        b1 = self.hyper_b_1(flat_states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(flat_agent_qs, w1) + b1)

        w_final = self.hyper_w_final(flat_states)
        if self.abs_weights:
            w_final = torch.abs(w_final)
        w_final = w_final.view(-1, self.embed_dim, 1)

        value = self.v(flat_states).view(-1, 1, 1)
        q_tot = torch.bmm(hidden, w_final) + value
        return q_tot.view(batch_size, -1, 1)
