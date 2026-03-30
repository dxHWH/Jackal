import torch
import random

from DRQN_Test.action_factorization import compose_action

class DRQNAgent:
    def __init__(self, action_dim, policy_net, device, chassis_dim=9, turret_dim=3, fire_action_id=27, fire_explore_bias=0.25):
        self.action_dim = action_dim
        self.policy_net = policy_net
        self.device = device
        self.chassis_dim = chassis_dim
        self.turret_dim = turret_dim
        self.fire_action_id = fire_action_id
        self.fire_explore_bias = fire_explore_bias
        
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
            q_chassis, q_turret, q_fire, new_hidden_state = self.policy_net(state_tensor, hidden_state)
            
        # Epsilon-Greedy 探索机制
        if random.random() < self.epsilon:
            chassis_action = random.randint(0, self.chassis_dim - 1)
            turret_action = random.randint(0, self.turret_dim - 1)
            dynamic_fire_bias = self.fire_explore_bias * (self.epsilon / 1.0)
            fire_action = 1 if random.random() < dynamic_fire_bias else 0
            action = compose_action(
                chassis_action,
                turret_action,
                fire_action,
                chassis_dim=self.chassis_dim,
                fire_action_id=self.fire_action_id,
            )
        else:
            chassis_action = q_chassis.argmax(dim=1).item()
            turret_action = q_turret.argmax(dim=1).item()
            fire_action = q_fire.argmax(dim=1).item()
            action = compose_action(
                chassis_action,
                turret_action,
                fire_action,
                chassis_dim=self.chassis_dim,
                fire_action_id=self.fire_action_id,
            )
            
        # 必须把 new_hidden_state 一起返回给外层的 while 循环
        return action, new_hidden_state

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)