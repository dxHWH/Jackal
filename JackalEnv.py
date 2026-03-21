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

#最大可视半径：
UNIT_SIGHT_RANGE = 400.0

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
        self.n_enemies = 10
        self.max_steps = 500
        
        # 新增：维护每个智能体的开火冷却时间
        self.fire_cooldown_max = 0.5  # 500毫秒冷却
        self.agent_fire_cooldowns = {i: 0.0 for i in range(self.n_agents)}
        
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
        """动作空间：9个组合机动动作 + 16个定向射击动作 = 25"""
        return 9 + 16

    def get_avail_agent_actions(self, agent_id):
        avail_actions = [0] * self.n_actions
        agent = self.agents[agent_id]
        if not agent.is_alive:
            avail_actions[0] = 1 
            return avail_actions
            
        # 9种机动总是可用
        avail_actions[0:9] = [1] * 9  
        
        # 16个射击方向也始终可用（智能体可以随意向空地开火）
        # 就算在冷却中也可以输出攻击指令（环境内部会拦截无效开火）
        avail_actions[9:25] = [1] * 16
                
        return avail_actions
    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def step(self, actions):
        self.steps += 1
        
        # --- 0. 更新开火冷却时间 ---
        for i in range(self.n_agents):
            if self.agent_fire_cooldowns[i] > 0:
                self.agent_fire_cooldowns[i] -= self.delta_time

        # --- 1. 动作解析 ---
        for agent_id, agent in enumerate(self.agents):
            if not agent.is_alive:
                continue
                
            action = actions[agent_id]
            agent.set_movement(forward=False, backward=False)
            agent.set_turning(left=False, right=False)
            
            # 解析 1-8 的机动动作
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
                
            # 解析 9-24 的 16 维定向射击动作
            elif action >= 9:
                dir_idx = action - 9
                target_angle = dir_idx * (360.0 / 16.0)  # 计算对应的绝对角度
                
                # 设置炮塔目标角度
                agent.turret_target_angle = target_angle
                
                # 检查冷却时间是否允许开火
                if self.agent_fire_cooldowns[agent_id] <= 0:
                    bullet = agent.fire(NormalShell)
                    if bullet:
                        self.bullet_manager.add_bullet(bullet)
                        # 重置该智能体的冷却时间
                        self.agent_fire_cooldowns[agent_id] = self.fire_cooldown_max

        # --- 2. 物理更新 ---
        for agent in self.agents: agent.update(self.delta_time, self.game_map.obstacles)
        for ai in self.enemy_ais: ai.update(self.delta_time)
            
        all_units = self.agents + self.enemies
        self.bullet_manager.update(self.delta_time, all_units, self.game_map.obstacles)
        
        # --- 3. 视频录制 ---
        if self.use_video:
            self._render_to_video()

        # --- 4. 收集反馈信息 (新增) ---
        done = self._check_done()
        reward = 0.0  # 我们会在后续设计具体的奖励函数
        info = {
            "battle_won": all(not enemy.is_alive for enemy in self.enemies)
        }
        
        return self.get_obs(), self.get_state(), 0, done, {}
    
    def _check_done(self):
        """检查当前回合是否结束"""
        # 1. 达到最大步数限制（超时）
        if self.steps >= self.max_steps:
            return True
            
        # 2. 玩家队伍全灭
        agents_dead = all(not agent.is_alive for agent in self.agents)
        if agents_dead:
            return True
            
        # 3. 敌方队伍全灭
        enemies_dead = all(not enemy.is_alive for enemy in self.enemies)
        if enemies_dead:
            return True
            
        return False

    def check_visibility(self, observer, target, sight_range=400.0):
        """
        判断 target 是否在 observer 的视野范围内且未被障碍物遮挡
        """
        if not observer.is_alive or not target.is_alive:
            return False
            
        # 1. 计算距离
        dx = target.position[0] - observer.position[0]
        dy = target.position[1] - observer.position[1]
        distance = math.hypot(dx, dy)
        
        # 超过视野限制
        if distance > sight_range:
            return False
            
        # 2. 障碍物遮挡检测 (Line of Sight Raycasting)
        line_start = observer.position
        line_end = target.position
        
        for obstacle_rect in self.game_map.obstacles:
            # clipline 如果返回非空元组，说明视线穿过了这个矩形障碍物
            if obstacle_rect.clipline(line_start, line_end):
                return False
                
        return True
    
    def get_obs(self):
        """
        获取所有智能体的局部观测 (Observation)
        返回: list[np.ndarray]，长度为 n_agents
        """
        obs_list = []
        sight_range = 400.0
        
        for agent_id, agent in enumerate(self.agents):
            if not agent.is_alive:
                # 阵亡智能体返回全 0 观测，维度需与存活时保持一致
                obs_dim = 8 + (self.n_agents - 1) * 5 + self.n_enemies * 5
                obs_list.append(np.zeros(obs_dim, dtype=np.float32))
                continue
                
            obs_features = []
            # --- 1. 自身特征 (8维) ---
            hp_ratio = agent.health / agent.max_health
            v_x = agent.speed * math.cos(math.radians(agent.direction_angle - 90))
            v_y = agent.speed * math.sin(math.radians(agent.direction_angle - 90))
            cos_dir = math.cos(math.radians(agent.direction_angle))
            sin_dir = math.sin(math.radians(agent.direction_angle))
            cos_turret = math.cos(math.radians(agent.turret_direction_angle))
            sin_turret = math.sin(math.radians(agent.turret_direction_angle))
            
            # 冷却比例：让智能体知道自己现在能不能开火
            cooldown_ratio = self.agent_fire_cooldowns[agent_id] / self.fire_cooldown_max
            
            obs_features.extend([hp_ratio, v_x, v_y, cos_dir, sin_dir, cos_turret, sin_turret, cooldown_ratio])
            
            # --- 2. 友军特征 ---
            for other_id, other_agent in enumerate(self.agents):
                if other_id == agent_id:
                    continue
                if self.check_visibility(agent, other_agent, sight_range):
                    rel_x = other_agent.position[0] - agent.position[0]
                    rel_y = other_agent.position[1] - agent.position[1]
                    dist = math.hypot(rel_x, rel_y) / sight_range  # 归一化距离
                    other_hp = other_agent.health / other_agent.max_health
                    obs_features.extend([1.0, rel_x/100.0, rel_y/100.0, dist, other_hp])
                else:
                    # 不可见时，强行切断信息源
                    obs_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    
            # --- 3. 敌军特征 ---
            for enemy in self.enemies:
                if self.check_visibility(agent, enemy, sight_range):
                    rel_x = enemy.position[0] - agent.position[0]
                    rel_y = enemy.position[1] - agent.position[1]
                    dist = math.hypot(rel_x, rel_y) / sight_range
                    enemy_hp = enemy.health / enemy.max_health
                    obs_features.extend([1.0, rel_x/100.0, rel_y/100.0, dist, enemy_hp])
                else:
                    # 视野外被限制信息
                    obs_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    
            obs_list.append(np.array(obs_features, dtype=np.float32))
            
        return obs_list

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

    def get_state(self): return np.zeros(20)
    
    def close(self):
        if self.use_video and self.video_writer is not None:
            self.video_writer.release()
        pygame.quit()
        #