import torch
import torch.nn.functional as F
import torch.optim as optim

from training.DRQN_Test.action_factorization import decompose_action_tensor

class DRQNLearner:
    def __init__(self, policy_net, target_net, device, lr=1e-4, gamma=0.99, tau=0.01, chassis_dim=9, fire_action_id=27):
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.chassis_dim = chassis_dim
        self.fire_action_id = fire_action_id
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def train_step(self, batch, burn_in=0):
        """
        执行带有掩码的截断序列训练 (Truncated BPTT + Double DQN)
        """
        states, actions, rewards, next_states, dones, masks = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        
        batch_size = states.size(0)
        seq_len = states.size(1)
        burn_in = max(0, min(burn_in, seq_len))
        
        chassis_actions, turret_actions, fire_actions = decompose_action_tensor(
            actions,
            chassis_dim=self.chassis_dim,
            fire_action_id=self.fire_action_id,
        )

        h_policy = self.policy_net.init_hidden(batch_size, self.device)
        h_target = self.target_net.init_hidden(batch_size, self.device)
        h_policy_for_next = self.policy_net.init_hidden(batch_size, self.device)

        if burn_in > 0:
            with torch.no_grad():
                for time_index in range(burn_in):
                    _, _, _, h_policy = self.policy_net(states[:, time_index, :], h_policy)
                    _, _, _, h_policy_for_next = self.policy_net(states[:, time_index, :], h_policy_for_next)
                    _, _, _, h_target = self.target_net(states[:, time_index, :], h_target)
            h_policy = h_policy.detach()
            h_policy_for_next = h_policy_for_next.detach()
            h_target = h_target.detach()
        
        q_values_list = []
        target_q_values_list = []
        valid_masks = []
        
        for time_index in range(burn_in, seq_len):
            q_chassis_t, q_turret_t, q_fire_t, h_policy = self.policy_net(states[:, time_index, :], h_policy)
            q_chassis_selected = q_chassis_t.gather(1, chassis_actions[:, time_index, :])
            q_turret_selected = q_turret_t.gather(1, turret_actions[:, time_index, :])
            q_fire_selected = q_fire_t.gather(1, fire_actions[:, time_index, :])

            q_action_t = q_chassis_selected + q_turret_selected + q_fire_selected
            q_values_list.append(q_action_t)
            valid_masks.append(masks[:, time_index, :])
            
            with torch.no_grad():
                _, _, _, next_h_target = self.target_net(states[:, time_index, :], h_target)
                _, _, _, next_h_policy_for_next = self.policy_net(states[:, time_index, :], h_policy_for_next)

                q_next_online_chassis, q_next_online_turret, q_next_online_fire, _ = self.policy_net(next_states[:, time_index, :], next_h_policy_for_next)
                next_chassis = q_next_online_chassis.argmax(dim=1, keepdim=True)
                next_turret = q_next_online_turret.argmax(dim=1, keepdim=True)
                next_fire = q_next_online_fire.argmax(dim=1, keepdim=True)

                q_next_target_chassis, q_next_target_turret, q_next_target_fire, _ = self.target_net(next_states[:, time_index, :], next_h_target)
                q_next_target_selected = (
                    q_next_target_chassis.gather(1, next_chassis)
                    + q_next_target_turret.gather(1, next_turret)
                    + q_next_target_fire.gather(1, next_fire)
                )

                target_q_t = rewards[:, time_index, :] + (1 - dones[:, time_index, :]) * self.gamma * q_next_target_selected
                target_q_values_list.append(target_q_t)

                h_target = next_h_target
                h_policy_for_next = next_h_policy_for_next

        if len(q_values_list) == 0:
            return 0.0

        q_values = torch.stack(q_values_list, dim=1).squeeze(-1)
        target_q_values = torch.stack(target_q_values_list, dim=1).squeeze(-1)
        valid_masks = torch.stack(valid_masks, dim=1).squeeze(-1)
        
        loss = F.smooth_l1_loss(q_values, target_q_values, reduction='none')
        masked_loss = loss * valid_masks
        valid_count = valid_masks.sum().clamp(min=1.0)
        final_loss = masked_loss.sum() / valid_count

        self.optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.soft_update_target_network()
        
        return final_loss.item()

    def soft_update_target_network(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def update_target_network(self):
        """
        硬更新：把 Policy 网络的权重完全拷贝给 Target 网络
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())