import os
import time
import random

# 强制无头模式（屏蔽图像和声音硬件要求）
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"  # 屏蔽声卡警告

import pygame

from Parameter import *
from Unit.Tank.Tank import create_tank, create_enemy_tank
from Map.Map import create_border_map
from Bullet.BulletManager import BulletManager
from Bullet.NormalShell.NormalShell import NormalShell

def test_headless_logic():
    print("初始化无头测试环境...")
    pygame.init()
    # 即使是无头模式，某些 Pygame 内部机制也需要一个 Surface
    screen = pygame.display.set_mode((960, 640)) 
    
    # 1. 实例化游戏对象
    game_map = create_border_map()
    tank = create_tank(1, Team.PLAYER, position=(400, 300))
    bullet_manager = BulletManager()
    enemy = create_enemy_tank(100, position=(500, 300))
    
    fixed_delta_time = 0.1  # 固定的离散时间步
    total_steps = 50        # 测试 50 个 step
    
    print(f"开始模拟运行 {total_steps} 步...\n")
    
    for step in range(total_steps):
        # ---------------------------------------------------
        # A. 模拟动作输入 (替代键盘和鼠标)
        # ---------------------------------------------------
        if step < 10:
            # 前 10 步：模拟按下 W 键 (前进) 和 D 键 (右转)
            tank.set_movement(forward=True, backward=False)
            tank.set_turning(left=False, right=True)
        elif step == 15:
            # 第 15 步：模拟点击鼠标开火
            tank.set_movement(forward=False, backward=False)
            tank.set_turning(left=False, right=False)
            # 假设鼠标在坦克的右侧，模拟炮塔转向并开火
            dummy_mouse_pos = (tank.position[0] + 100, tank.position[1])
            tank.set_turret_target_to_mouse(dummy_mouse_pos, (0, 0))
            
            bullet = tank.fire(NormalShell)
            if bullet:
                bullet_manager.add_bullet(bullet)
                print(f"\n>>> [Step {step}] 玩家坦克开火了！ <<<")
        else:
            # 停止操作
            tank.set_movement(forward=False, backward=False)
            tank.set_turning(left=False, right=False)

        # ---------------------------------------------------
        # B. 物理与逻辑更新
        # ---------------------------------------------------
        tank.update(fixed_delta_time, game_map.obstacles)
        enemy.update(fixed_delta_time, game_map.obstacles)
        all_units = [tank, enemy]
        bullet_manager.update(fixed_delta_time, all_units, game_map.obstacles)
        
        # ---------------------------------------------------
        # C. 状态监控与日志 (打印所有单位的信息)
        # ---------------------------------------------------
        if step % 10 == 0 or step == 15 or step == 16:
            print(f"\n--- [Step {step}] 状态报告 (当前场上子弹数: {len(bullet_manager.bullets)}) ---")
            for unit in all_units:
                info = unit.get_info()
                # 区分是玩家还是敌人
                if info['team'].name == 'PLAYER':
                    role = "[玩家坦克]"
                else:
                    role = f"[敌方坦克 ID:{info['id']}]"
                
                print(f"  {role} 位置: ({info['position'][0]:.1f}, {info['position'][1]:.1f}) | "
                      f"血量: {info['health']}/{info['max_health']} | "
                      f"速度: {info['speed']:>5.1f} | 朝向: {info['direction']:>5.1f}")
            print("-" * 55)

    print("\n测试完成。如果没有报错，说明核心逻辑在服务器上运行正常。")
    pygame.quit()

if __name__ == "__main__":
    test_headless_logic()