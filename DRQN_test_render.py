import torch
import os
import json
from JackalEnv import JackalEnv

# 导入 DRQN 网络
from DRQN_Test.network import DRQNNetwork
from DRQN_Test.action_factorization import compose_action

def test_drqn_model(model_path, episodes=3):
    print(f"正在加载 DRQN 模型并准备录制视频: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(os.path.dirname(model_path), "drqn_run_config.json")
    run_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)
        print(f"检测到训练配置: {config_path}")
    
    # ==========================================
    # 1. 初始化环境 (开启视频录制)
    # ==========================================
    # 注意：这里的 auto_aim 和 n_enemies 必须与你训练该模型时完全一致！
    env = JackalEnv(
        headless=True,
        use_video=True,
        video_dir="eval_videos_drqn",
        auto_aim=run_config.get("auto_aim", False)
    )
    
    _, initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.n_actions
    hidden_dim = run_config.get("hidden_dim", 128)
    chassis_dim = run_config.get("chassis_dim", 9)
    turret_dim = run_config.get("turret_dim", 3)
    fire_dim = run_config.get("fire_dim", 2)
    fire_action_id = run_config.get("fire_action_id", 27)

    if run_config:
        expected_state_dim = run_config.get("state_dim")
        expected_action_dim = run_config.get("action_dim")
        if expected_state_dim is not None and state_dim != expected_state_dim:
            print(f"状态维度不匹配: env={state_dim}, model={expected_state_dim}")
            env.close()
            return
        if expected_action_dim is not None and action_dim != expected_action_dim:
            print(f"动作维度不匹配: env={action_dim}, model={expected_action_dim}")
            env.close()
            return
    
    # ==========================================
    # 2. 实例化网络并加载权重
    # ==========================================
    # hidden_dim 必须与训练时保持一致（优先读取配置）
    policy_net = DRQNNetwork(
        state_dim,
        action_dim,
        hidden_dim=hidden_dim,
        chassis_dim=chassis_dim,
        turret_dim=turret_dim,
        fire_dim=fire_dim,
    ).to(device)
    
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
                
                q_chassis, q_turret, q_fire, hidden_state = policy_net(state_tensor, hidden_state)

                if run_config.get("factorized_action", True):
                    chassis_action = q_chassis.argmax(dim=1).item()
                    turret_action = q_turret.argmax(dim=1).item()
                    fire_action = q_fire.argmax(dim=1).item()
                    action = compose_action(
                        chassis_action,
                        turret_action,
                        fire_action,
                        chassis_dim=chassis_dim,
                        fire_action_id=fire_action_id,
                    )
                else:
                    raise RuntimeError("当前测试脚本仅支持 factorized_action=True 的 DRQN 权重。")
                
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