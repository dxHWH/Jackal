import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import random
from JackalEnv import JackalEnv

def test_marl_env():
    print("初始化 MARL 标准环境测试 (附带视频录制)...")
    
    # 在这里开启 use_video=True，并可以指定输出文件夹名称
    env = JackalEnv(headless=True, fixed_delta_time=0.1, use_video=True, video_dir="jackal_records")
    obs, state = env.reset()
    
    print(f"动作空间大小: {env.n_actions}")
    print("开始模拟运行 100 步...\n")
    
    for step in range(300):
        avail_actions = env.get_avail_actions()[0]
        
        #valid_action_indices = [i for i, is_available in enumerate(avail_actions) if is_available == 1]
        #chosen_action = random.choice(valid_action_indices)
        #actions = [chosen_action]
        actions = [0]  # 默认动作：不移动也不转向
        obs, state, reward, done, info = env.step(actions)
        if done:
            print(f"\n>>> [Step {step}] Episode 结束！<<<")
            break
        if step % 10 == 0 or step == 15 or step == 16:
            print(f"\n--- [Step {step}] 状态报告 (场上子弹: {len(env.bullet_manager.bullets)}) ---")
            for agent in env.agents:
                info = agent.get_info()
                print(f"  [玩家] 位置: ({info['position'][0]:.1f}, {info['position'][1]:.1f}) | 朝向: {info['direction']:>5.1f}")

    print("\n测试完成。正在保存录像文件...")
    env.close()  # 必须调用 close 以正确释放视频文件写入锁
    print("录像保存成功！请查看 'jackal_records' 文件夹。")

if __name__ == "__main__":
    test_marl_env()