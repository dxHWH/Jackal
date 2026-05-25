import torch
import torch.nn as nn
import torch.optim as optim

class DQNLearner:
    def __init__(self, policy_net, target_net, device, lr=1e-4, gamma=0.99):
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device
        self.gamma = gamma
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        #self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()

    def train_step(self, batch):
        """执行单步梯度下降"""
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 获取当前动作的 Q 值
        q_values = self.policy_net(states).gather(1, actions)
        
        # 计算 Target Q 值
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            
        # 梯度更新
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        """同步 Target 网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())