import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=128, input_hidden_dim=None, input_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        input_hidden_dim = int(input_hidden_dim) if input_hidden_dim is not None else int(hidden_dim)
        input_layers = max(1, int(input_layers))

        layers = []
        in_dim = int(input_shape)
        for _ in range(input_layers):
            layers.append(nn.Linear(in_dim, input_hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = input_hidden_dim
        self.input_encoder = nn.Sequential(*layers)
        self.rnn_in = nn.Linear(input_hidden_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.rnn_in(self.input_encoder(inputs)))
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q, h
