import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DRQNNetwork, self).__init__()
        self.hidden_dim = hidden_dim

        # 1. 特征提取层 (将几十维的状态升维到隐藏层空间)
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # 2. 核心记忆层：GRUCell 
        # 它接收 fc1 的输出特征，以及上一个时间步的隐状态 (Hidden State)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        # 3. 动作输出层 (输出每个动作的 Q 值)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def init_hidden(self, batch_size=1, device="cpu"):
        """
        初始化全零的隐状态 (Hidden State)。
        在每次新开一局游戏 (Episode) 时，或者在 Learner 开始处理一个新 Batch 时调用。
        """
        # 返回形状为 (Batch_Size, Hidden_Dim) 的零张量
        return torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=device)

    def forward(self, x, hidden_state):
        """
        前向传播
        :param x: 当前的输入状态，形状 (Batch_Size, State_Dim)
        :param hidden_state: 上一刻的隐状态，形状 (Batch_Size, Hidden_Dim)
        :return: 当前的 Q 值和更新后的隐状态
        """
        # 1. 提取当前状态的特征
        x = F.relu(self.fc1(x))
        
        # 2. 将特征和历史记忆送入 GRU 单元，得到新的记忆
        # 注意：GRUCell 的输入必须是 2D 张量 (Batch_Size, Feature_Dim)
        h_out = self.rnn(x, hidden_state)
        
        # 3. 基于最新的记忆，计算当前应该采取的动作 Q 值
        q_values = self.fc2(h_out)
        
        # 【关键】：必须把 h_out 连同 q_values 一起返回出去
        # 因为外层的 Agent 需要把这个 h_out 存下来，留给下一个 Step 使用
        return q_values, h_out