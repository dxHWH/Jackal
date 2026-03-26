import torch
import random

class DQNAgent:
    def __init__(self, action_dim, policy_net, device):
        self.action_dim = action_dim
        self.policy_net = policy_net  # 传入网络引用
        self.device = device
        
        # 探索策略超参数
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997

    def select_action(self, state):
        """Epsilon-Greedy 动作选择"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)