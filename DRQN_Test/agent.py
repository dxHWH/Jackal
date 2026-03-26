import torch
import random

class DRQNAgent:
    def __init__(self, action_dim, policy_net, device):
        self.action_dim = action_dim
        self.policy_net = policy_net
        self.device = device
        
        # 探索策略超参数
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997

    def init_hidden(self):
        """
        在新的一局游戏 (Episode) 开始时调用。
        向网络索要一个干净的、全零的初始隐状态。
        由于是单步与环境交互，这里的 batch_size 固定为 1。
        """
        return self.policy_net.init_hidden(batch_size=1, device=self.device)

    def select_action(self, state, hidden_state):
        """
        带有记忆的 Epsilon-Greedy 动作选择。
        :param state: 当前环境的观测状态
        :param hidden_state: 智能体脑海中上一刻的记忆
        :return: 选择的动作 (action) 以及更新后的最新记忆 (new_hidden_state)
        """
        # 将 numpy array 转换为 PyTorch Tensor，并增加 batch 维度 (1, State_Dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 【核心差异】：前向传播时必须带上历史记忆，同时接收新记忆
        with torch.no_grad():
            q_values, new_hidden_state = self.policy_net(state_tensor, hidden_state)
            
        # Epsilon-Greedy 探索机制
        if random.random() < self.epsilon:
            # 随机探索
            action = random.randint(0, self.action_dim - 1)
        else:
            # 利用经验：选择 Q 值最大的动作
            action = q_values.argmax().item()
            
        # 必须把 new_hidden_state 一起返回给外层的 while 循环
        return action, new_hidden_state

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)