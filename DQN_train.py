import torch
import json
import os
from tqdm import tqdm  
from JackalEnv import JackalEnv
from DQN_Test.network import QNetwork
from DQN_Test.buffer import EpisodeBuffer 
from DQN_Test.agent import DQNAgent
from DQN_Test.learner import DQNLearner

def main():
    # 明确打印出当前使用的是 CPU 还是 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"初始化 JackalEnv (解耦架构 1v1 DQN) | 当前计算设备: {device.type.upper()}")
    
    # 1. 实例化环境
    env = JackalEnv(headless=True, use_video=False)
    _, initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.n_actions

    run_config = {
        "algo": "dqn",
        "auto_aim": env.auto_aim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_agents": env.n_agents,
        "n_enemies": env.n_enemies,
    }
    os.makedirs("DQN_Test/model", exist_ok=True)
    with open("DQN_Test/model/dqn_run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    
    # 2. 实例化共享网络并放入设备
    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # 3. 实例化组件 
    buffer = EpisodeBuffer(capacity=2000)
    agent = DQNAgent(action_dim, policy_net, device)
    learner = DQNLearner(policy_net, target_net, device)
    
    batch_size = 128
    num_episodes = 2000
    target_update_freq = 2000
    
    # ==========================================
    # 4. 主干交互循环 (使用 tqdm 包装)
    # ==========================================
    # 创建一个进度条对象
    pbar = tqdm(range(1, num_episodes + 1), desc="训练进度", unit="ep")
    total_step_count = 0
    for episode in pbar:
        _, state = env.reset()
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        done = False
        
        current_episode_trajectory = [] 
        
        while not done:
            action = agent.select_action(state)
            _, next_state, reward, done, info = env.step([action])
            
            current_episode_trajectory.append((state, action, reward, next_state, float(done)))
            
            if len(buffer) >= 10:  
                batch = buffer.sample_transitions(batch_size)
                loss = learner.train_step(batch)
                episode_loss += loss
            
            state = next_state
            episode_reward += reward
            step_count += 1
            total_step_count += 1
            
        buffer.push_episode(current_episode_trajectory)
        
        # 回合结束处理
        agent.decay_epsilon()
        if total_step_count % target_update_freq == 0:
            learner.update_target_network()
            
        # ==========================================
        # 5. 动态更新进度条后缀
        # ==========================================
        avg_loss = episode_loss / step_count if step_count > 0 else 0
        battle_result = "Win" if info.get("battle_won", False) else "Loss"
        
        # 使用 set_postfix 实时更新当前回合的数据，替代原本刷屏的 print
        pbar.set_postfix({
            'Step': step_count,
            'Rwd': f"{episode_reward:.1f}",
            'Res': battle_result,
            'Eps': f"{agent.epsilon:.3f}",
            'Loss': f"{avg_loss:.4f}"
        })
        
        # 定期保存模型 (使用 pbar.write 防止打断进度条渲染)
        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), f"DQN_Test/model/dqn_model_ep{episode}.pth")
            pbar.write(f"--> [检查点] 模型已保存至 DQN_Test/model/dqn_model_ep{episode}.pth")

    torch.save(policy_net.state_dict(), "DQN_Test/model/dqn_model_final.pth")
    print("\n训练结束！最终模型已保存。")
    env.close()

if __name__ == "__main__":
    main()