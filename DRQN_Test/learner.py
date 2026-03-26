import torch
import torch.nn.functional as F
import torch.optim as optim

class DRQNLearner:
    def __init__(self, policy_net, target_net, device, lr=1e-4, gamma=0.99):
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device
        self.gamma = gamma
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def train_step(self, batch):
        """
        执行带有掩码的沿时间反向传播 (BPTT)
        """
        # 1. 解析 Batch 数据 (注意现在的形状是 3D: [Batch_Size, Seq_Len, Dim])
        states, actions, rewards, next_states, dones, masks = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        
        batch_size = states.size(0)
        seq_len = states.size(1)
        
        # 2. 初始化全零的隐状态 (Batch_Size, Hidden_Dim)
        # Policy Network 和 Target Network 都需要各自的“空白记忆”起点
        h_policy = self.policy_net.init_hidden(batch_size, self.device)
        h_target = self.target_net.init_hidden(batch_size, self.device)
        
        q_values_list = []
        target_q_values_list = []
        
        # ==========================================
        # 3. 沿时间轴展开网络 (Unroll over time)
        # ==========================================
        for t in range(seq_len):
            # --- 当前动作的 Q 值评估 ---
            # 喂入第 t 步的状态，和 t-1 步的记忆
            q_t, h_policy = self.policy_net(states[:, t, :], h_policy)
            # 挑出实际执行的那个动作的 Q 值
            q_action_t = q_t.gather(1, actions[:, t, :])
            q_values_list.append(q_action_t)
            
           # --- 目标 Q 值评估 (Target) ---
            with torch.no_grad():
                # 【极其关键的修复】：让 Target 网络先处理 current state，以获取正确的 h_{t+1}
                _, next_h_target = self.target_net(states[:, t, :], h_target)
                
                # 然后，使用正确的记忆去评估 next_state
                q_target_next, _ = self.target_net(next_states[:, t, :], next_h_target)
                max_q_target_next = q_target_next.max(1)[0].unsqueeze(1)
                
                # 计算 TD Target
                target_q_t = rewards[:, t, :] + (1 - dones[:, t, :]) * self.gamma * max_q_target_next
                target_q_values_list.append(target_q_t)
                
                # 将隐状态滚动到下一步
                h_target = next_h_target
                
        # 4. 将每一步的计算结果重新拼接成 3D 张量 [Batch_Size, Seq_Len, 1]
        q_values = torch.stack(q_values_list, dim=1).squeeze(-1)
        target_q_values = torch.stack(target_q_values_list, dim=1).squeeze(-1)
        masks = masks.squeeze(-1)
        
        # ==========================================
        # 5. 计算带掩码的损失函数 (Masked Loss)
        # ==========================================
        # 先计算所有位置的 Huber Loss (reduction='none' 保证它不自动求均值)
        loss = F.smooth_l1_loss(q_values, target_q_values, reduction='none')
        
        # 【核心操作】：用 Mask 矩阵乘以 Loss。
        # 那些补零出来的无效步 (Mask=0) 的 Loss 会瞬间变成 0！
        masked_loss = loss * masks
        
        # 计算有效步数的平均 Loss (不能直接 sum() / (Batch * Seq_Len) !)
        # 我们只除以真实发生的总步数 (masks.sum())
        final_loss = masked_loss.sum() / masks.sum()
        
        # 6. 反向传播与梯度裁剪
        self.optimizer.zero_grad()
        final_loss.backward()
        
        # RNN 极其容易发生梯度爆炸，这一步 clip_grad_norm_ 是保命符！
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        
        return final_loss.item()

    def update_target_network(self):
        """
        硬更新：把 Policy 网络的权重完全拷贝给 Target 网络
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())