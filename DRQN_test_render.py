import torch
import os
from JackalEnv import JackalEnv

# 导入 DRQN 网络
from DRQN_Test.network import DRQNNetwork

def test_drqn_model(model_path, episodes=3):
    print(f"正在加载 DRQN 模型并准备录制视频: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==========================================
    # 1. 初始化环境 (开启视频录制)
    # ==========================================
    # 注意：这里的 auto_aim 和 n_enemies 必须与你训练该模型时完全一致！
    env = JackalEnv(headless=True, use_video=True, video_dir="eval_videos_drqn", auto_aim=True)
    
    _, initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.n_actions
    
    # ==========================================
    # 2. 实例化网络并加载权重
    # ==========================================
    # 这里的 hidden_dim=128 必须与训练时保持一致
    policy_net = DRQNNetwork(state_dim, action_dim, hidden_dim=128).to(device)
    
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        print("--> 模型权重加载成功！")
    else:
        print(f"--> 找不到模型文件: {model_path}，请检查路径。")
        return
        
    policy_net.eval()  # 设置为评估模式，关闭 Dropout/BatchNorm 等
    
    # ==========================================
    # 3. 开启测试循环 (带记忆流转的纯贪婪策略)
    # ==========================================
    for episode in range(1, episodes + 1):
        _, state = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        # 【DRQN 测试核心】：开局获取全零的空白记忆
        hidden_state = policy_net.init_hidden(batch_size=1, device=device)
        
        while not done:
            # 执行纯粹的最优策略 (Greedy)，不带任何 Epsilon 随机探索
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # 网络必须同时吃下当前状态和历史记忆，并吐出 Q 值和新记忆
                q_values, hidden_state = policy_net(state_tensor, hidden_state)
                
                action = q_values.argmax().item()  # 永远选择 Q 值最大的动作
                
            _, next_state, reward, done, info = env.step([action])
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
        battle_result = "Win" if info.get("battle_won", False) else "Loss"
        print(f"测试局 {episode}/{episodes} | 存活步数: {step_count} | 总奖励: {episode_reward:.1f} | 结果: {battle_result}")

    # ==========================================
    # 4. 释放资源并保存视频
    # ==========================================
    env.close()
    print(f"\n测试完成！录制的视频已保存在 {os.path.abspath('eval_videos_drqn')} 目录下。")

if __name__ == "__main__":
    # 指定你要测试的权重文件路径。
    # 建议先跑 final，如果效果不好，可以跑跑 ep900, ep800 看看是不是后期过拟合了
    target_model_path = "DRQN_Test/models/drqn_model_final.pth" 
    
    test_drqn_model(target_model_path, episodes=3)