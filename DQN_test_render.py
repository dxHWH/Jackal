import torch
import os
from JackalEnv import JackalEnv
from DQN_Test.network import QNetwork

def test_model(model_path, episodes=3):
    print(f"正在加载模型并准备录制视频: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==========================================
    # 1. 初始化环境 (开启视频录制)
    # ==========================================
    # 注意：这里的 auto_aim 必须与你训练时保存的模型保持一致！
    # 如果你在没有 GUI 的服务器上跑，保持 headless=True；
    # use_video=True 会自动将每一帧渲染并保存为 mp4 文件。
    env = JackalEnv(headless=True, use_video=True, video_dir="eval_videos", auto_aim=True)
    
    _, initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.n_actions
    
    # ==========================================
    # 2. 实例化网络并加载权重
    # ==========================================
    policy_net = QNetwork(state_dim, action_dim).to(device)
    
    # 使用 map_location 确保即使在没有 GPU 的机器上也能加载 GPU 训练出的模型
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        print("模型权重加载成功！")
    else:
        print(f"找不到模型文件: {model_path}，请检查路径。")
        return
        
    policy_net.eval()  # 设置为评估模式
    
    # ==========================================
    # 3. 开启测试循环
    # ==========================================
    for episode in range(1, episodes + 1):
        _, state = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # 【核心差异】：完全抛弃探索，执行纯粹的最优策略 (Greedy)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
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
    print(f"\n测试完成！录制的视频已保存在 {os.path.abspath('eval_videos')} 目录下。")

if __name__ == "__main__":
    # 指定你要测试的权重文件路径。
    # 如果你训练到一半保存了 checkpoint，也可以改成了 dqn_model_ep200.pth 看看前期多菜
    target_model_path = "DQN_Test/model/dqn_model_final.pth" 
    
    test_model(target_model_path, episodes=5)