import pygame
import os
import random
import numpy as np
import math
import cv2  # 新增：用于视频录制
import datetime

from Parameter import *
from Unit.Tank.Tank import create_tank, create_enemy_tank
from Map.Map import create_border_map
from Bullet.BulletManager import BulletManager
from Unit.EnemyAI import EnemyAI
from Bullet.NormalShell.NormalShell import NormalShell

class JackalEnv:
    def __init__(self, headless=True, fixed_delta_time=0.1, use_video=False, video_dir="videos"):
        self.headless = headless
        self.delta_time = fixed_delta_time
        
        # 视频录制参数
        self.use_video = use_video
        self.video_dir = video_dir
        self.video_writer = None
        
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
            
        pygame.init()
        
        self.screen_width, self.screen_height = 960, 640
        
        # 即使在无头模式下，我们也需要一个 Surface 用于内存绘图
        if not self.headless:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Jackal MARL Environment")
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            
        self.n_agents = 1
        self.n_enemies = 5
        self.max_steps = 500
        
        if self.use_video:
            os.makedirs(self.video_dir, exist_ok=True)

    def reset(self):
        self.steps = 0
        self.game_map = create_border_map()
        self.bullet_manager = BulletManager()
        self.agents = [create_tank(1, Team.PLAYER, position=(400, 300))]
        
        self.enemies = []
        self.enemy_ais = []
        for i in range(self.n_enemies):
            enemy = create_enemy_tank(100 + i, position=(500, 300))
            self.enemies.append(enemy)
            ai = EnemyAI(enemy, self.agents[0], self.bullet_manager, self.game_map.obstacles)
            self.enemy_ais.append(ai)
            
        # 如果开启了视频录制，在每个回合重置时创建一个新的视频文件
        if self.use_video:
            if self.video_writer is not None:
                self.video_writer.release()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.video_dir, f"episode_{timestamp}.mp4")
            # 采用 mp4v 编码，FPS 取决于 delta_time (例如 0.1s 就是 10 FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(1.0 / self.delta_time)
            self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (self.screen_width, self.screen_height))
            
            # 录制初始帧
            self._render_to_video()
            
        return self.get_obs(), self.get_state()

    @property
    def n_actions(self):
        return 9 + self.n_enemies

    def get_avail_agent_actions(self, agent_id):
        avail_actions = [0] * self.n_actions
        agent = self.agents[agent_id]
        if not agent.is_alive:
            avail_actions[0] = 1 
            return avail_actions
            
        avail_actions[0:9] = [1] * 9
        
        for i, enemy in enumerate(self.enemies):
            if enemy.is_alive:
                avail_actions[9 + i] = 1
                
        return avail_actions

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def step(self, actions):
        self.steps += 1
        
        # --- 1. 动作解析 ---
        for agent_id, agent in enumerate(self.agents):
            if not agent.is_alive:
                continue
                
            action = actions[agent_id]
            agent.set_movement(forward=False, backward=False)
            agent.set_turning(left=False, right=False)
            
            if action == 1: agent.set_movement(forward=True, backward=False)
            elif action == 2: agent.set_movement(forward=False, backward=True)
            elif action == 3: agent.set_turning(left=True, right=False)
            elif action == 4: agent.set_turning(left=False, right=True)
            elif action == 5:
                agent.set_movement(forward=True, backward=False)
                agent.set_turning(left=True, right=False)
            elif action == 6:
                agent.set_movement(forward=True, backward=False)
                agent.set_turning(left=False, right=True)
            elif action == 7:
                agent.set_movement(forward=False, backward=True)
                agent.set_turning(left=True, right=False)
            elif action == 8:
                agent.set_movement(forward=False, backward=True)
                agent.set_turning(left=False, right=True)
            elif action >= 9:
                enemy_idx = action - 9
                if enemy_idx < len(self.enemies) and self.enemies[enemy_idx].is_alive:
                    target_enemy = self.enemies[enemy_idx]
                    dx = target_enemy.position[0] - agent.position[0]
                    dy = target_enemy.position[1] - agent.position[1]
                    target_angle = math.degrees(math.atan2(dy, dx)) + 90
                    target_angle = target_angle % 360
                    if target_angle < 0: target_angle += 360
                        
                    agent.turret_target_angle = target_angle
                    bullet = agent.fire(NormalShell)
                    if bullet:
                        self.bullet_manager.add_bullet(bullet)

        # --- 2. 物理更新 ---
        for agent in self.agents: agent.update(self.delta_time, self.game_map.obstacles)
        for ai in self.enemy_ais: ai.update(self.delta_time)
            
        all_units = self.agents + self.enemies
        self.bullet_manager.update(self.delta_time, all_units, self.game_map.obstacles)
        
        # --- 3. 视频录制 ---
        if self.use_video:
            self._render_to_video()
        
        return self.get_obs(), self.get_state(), 0, False, {}

    def _render_to_video(self):
        """将当前游戏状态绘制到内存中的 Surface，并转换为 OpenCV 视频帧"""
        self.screen.fill((50, 50, 70))
        camera_offset = [0, 0]
        
        # 调用原有的绘制逻辑
        self.game_map.draw(self.screen, camera_offset)
        for enemy in self.enemies:
            if enemy.is_alive:
                enemy.draw(self.screen, camera_offset)
        for agent in self.agents:
            if agent.is_alive:
                agent.draw(self.screen, camera_offset)
        self.bullet_manager.draw(self.screen, camera_offset)
        
        # 提取像素数据：Pygame 提取的是 (Width, Height, RGB)
        frame = pygame.surfarray.array3d(self.screen)
        # 转置矩阵：OpenCV 需要的是 (Height, Width, RGB)
        frame = np.transpose(frame, (1, 0, 2))
        # 转换颜色通道：OpenCV 使用 BGR 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self.video_writer.write(frame)

    def get_obs(self): return [np.zeros(10)] * self.n_agents
    def get_state(self): return np.zeros(20)
    
    def close(self):
        if self.use_video and self.video_writer is not None:
            self.video_writer.release()
        pygame.quit()