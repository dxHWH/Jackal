import torch
from tqdm import tqdm
from JackalEnv import JackalEnv

# 导入我们刚刚写好的 DRQN 四大金刚
from DRQN_Test.network import DRQNNetwork
from DRQN_Test.agent import DRQNAgent
from DRQN_Test.buffer import EpisodeBuffer
from DRQN_Test.learner import DRQNLearner

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"初始化 JackalEnv (DRQN 序列化架构) | 当前计算设备: {device.type.upper()}")
    
    # ==========================================
    # 1. 实例化环境 (你可以随时把 n_enemies 改为 2 开启 1v2 挑战)
    # ==========================================
    env = JackalEnv(headless=True, use_video=False, auto_aim=False)
    _, initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.n_actions
    
    # ==========================================
    # 2. 实例化带 GRU 记忆的神经网络
    # ==========================================
    policy_net = DRQNNetwork(state_dim, action_dim, hidden_dim=128).to(device)
    target_net = DRQNNetwork(state_dim, action_dim, hidden_dim=128).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # ==========================================
    # 3. 实例化 DRQN 组件 
    # ==========================================
    # 容量 2000 局 (如果显存吃紧，可以适当调小)
    buffer = EpisodeBuffer(capacity=2000) 
    agent = DRQNAgent(action_dim, policy_net, device)
    learner = DRQNLearner(policy_net, target_net, device, lr=0.0001)
    
    # DRQN 超参数：
    # 因为一条轨迹可能长达 500 步，Batch Size 不能像 DQN 那样设为 128
    # 设为 32 条轨迹 (32 * 500 = 16000 帧)，对显存和 BPTT 来说比较健康
    batch_size = 32
    num_episodes = 6000 # 序列训练需要更久的探索时间
    target_update_freq = 10
    
    # ==========================================
    # 4. 主干交互循环 
    # ==========================================
    pbar = tqdm(range(1, num_episodes + 1), desc="DRQN 训练进度", unit="ep")
    
    for episode in pbar:
        _, state = env.reset()
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        done = False
        
        current_episode_trajectory = [] 
        
        # 【DRQN 核心 1】：开局获取全零的空白记忆
        hidden_state = agent.init_hidden()
        
        while not done:
            # 【DRQN 核心 2】：带着历史记忆做决策，并接收新记忆
            action, next_hidden_state = agent.select_action(state, hidden_state)
            
            _, next_state, reward, done, info = env.step([action])
            
            # 记录这一步的经验
            current_episode_trajectory.append((state, action, reward, next_state, float(done)))
            
            # 状态转移 & 记忆流转
            state = next_state
            hidden_state = next_hidden_state
            
            episode_reward += reward
            step_count += 1
            
        # 【DRQN 核心 3】：游戏结束，将一整条完整的时序轨迹压入 Buffer
        buffer.push_episode(current_episode_trajectory)
        
        # 训练更新逻辑：当 Buffer 里的完整轨迹数量达到 Batch Size 时才开始训练
        if len(buffer) >= batch_size:
            # 从 Buffer 抽取对齐 (Padding) 并带有掩码 (Mask) 的 3D 张量
            batch_data = buffer.sample_batch(batch_size)
            
            # 执行沿时间反向传播 (BPTT)
            loss = learner.train_step(batch_data)
            episode_loss += loss
            
        # 回合结束处理
        agent.decay_epsilon()
        if episode % target_update_freq == 0:
            learner.update_target_network()
            
        # ==========================================
        # 5. 动态更新进度条
        # ==========================================
        # 由于我们每个 Episode 只在最后统一 train 一次，Loss 就是那一批的 Loss
        avg_loss = episode_loss 
        battle_result = "Win" if info.get("battle_won", False) else "Loss"
        
        pbar.set_postfix({
            'Step': step_count,
            'Rwd': f"{episode_reward:.1f}",
            'Res': battle_result,
            'Eps': f"{agent.epsilon:.3f}",
            'Loss': f"{avg_loss:.4f}"
        })
        
        # 定期保存模型权重
        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), f"DRQN_Test/models/drqn_model_ep{episode}.pth")
            pbar.write(f"--> [检查点] 模型已保存至 DRQN_Test/models/drqn_model_ep{episode}.pth")

    torch.save(policy_net.state_dict(), "DRQN_Test/models/drqn_model_final.pth")
    print("\nDRQN 训练结束！最终模型已保存。")
    env.close()

if __name__ == "__main__":
    main()