import pygame
import os
import random
import numpy as np
import math
import cv2
import datetime

from Parameter import *
from Unit.Tank.Tank import create_tank, create_enemy_tank
from Map.GameMap import create_border_map
from Bullet.BulletManager import BulletManager
from Bullet.NormalShell.NormalShell import NormalShell

# 引入单位管理器
from Unit.UnitManager import UnitManager
# from GameMode import Team


class JackalEnv:
    def __init__(self, headless=True, fixed_delta_time=0.03, use_video=False, video_dir="videos", auto_aim=True, reward_config=None):
        self.headless = headless
        self.delta_time = fixed_delta_time
        
        self.use_video = use_video
        self.video_dir = video_dir
        self.video_writer = None

        self.auto_aim = auto_aim
        self.reward_config = {
            "auto_aim": {
                "enemy_limit_scale": 0.1,
                "agent_limit_scale": 0.1,
                "enemy_kill_bonus": 20.0,
                "agent_killed_penalty": 10.0,
                "win_bonus": 30.0,
                "lose_penalty": 30.0,
                "timeout_penalty": 30.0,
            },
            "manual_aim": {
                "enemy_limit_scale": 0.12,
                "agent_limit_scale": 0.12,
                "enemy_kill_bonus": 20.0,
                "agent_killed_penalty": 12.0,
                "aim_good_angle": 8.0,
                "aim_ok_angle": 20.0,
                "aim_good_reward": 0.04,
                "aim_ok_reward": 0.015,
                "aim_bad_penalty": 0.01,
                "fire_good_angle": 12.0,
                "fire_good_reward": 0.08,
                "fire_bad_penalty": 0.06,
                "step_penalty": 0.01,
                "fire_action_id": 27,
                "win_bonus": 30.0,
                "lose_penalty": 30.0,
                "timeout_penalty": 30.0,
            }
        }
        if reward_config:
            self._deep_update(self.reward_config, reward_config)
        
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
            
        pygame.init()
        
        self.screen_width, self.screen_height = 960, 640
        
        if not self.headless:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Jackal MARL Environment")
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            
        self.n_agents = 1
        # ==========================================
        # 【修改】：动态设置敌人数量和初始坐标
        self.n_enemies = 1
        enemy_positions = [(600,300)]
        if enemy_positions is None:
            # 如果不传坐标，默认让敌人在右半场随机或者排开
            self.enemy_positions = [(600, 300 + i * 100) for i in range(self.n_enemies)]
        else:
            self.enemy_positions = enemy_positions
        self.max_steps = 500
        
        self.fire_cooldown_max = 2
        self.agent_fire_cooldowns = {i: 0.0 for i in range(self.n_agents)}
        
        if self.use_video:
            os.makedirs(self.video_dir, exist_ok=True)

    def _deep_update(self, base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and isinstance(base_dict.get(key), dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def set_reward_config(self, reward_config):
        if reward_config:
            self._deep_update(self.reward_config, reward_config)

    def reset(self):
        self.steps = 0
        self.game_map = create_border_map()
        self.bullet_manager = BulletManager()
        self.unit_manager = UnitManager()  # 实例化新的 UnitManager
        
        self.agents = []
        # 创建玩家: RL网络控制，不使用内置AI (usingAI=False)
        player = create_tank(1, Team.PLAYER, position=(400, 300))
        player.usingAI = False  
        self.unit_manager.add_unit(player, self.bullet_manager, self.game_map)
        self.agents.append(player)
        
        self.enemies = []
        for i in range(self.n_enemies):
            # 【修改】：安全地读取坐标，防止传入的坐标数量不够
            pos = self.enemy_positions[i] if i < len(self.enemy_positions) else (600, 200 + i * 100)
            
            # 创建敌人: 开启内置AI控制
            enemy = create_enemy_tank(100 + i, position=pos)
            enemy.usingAI = True
            self.unit_manager.add_unit(enemy, self.bullet_manager, self.game_map)
            self.enemies.append(enemy)
            
        if self.use_video:
            if self.video_writer is not None:
                self.video_writer.release()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.video_dir, f"episode_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(1.0 / self.delta_time)
            self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (self.screen_width, self.screen_height))
            self._render_to_video()
            
        return self.get_obs(), self.get_state()

    @property
    def n_actions(self):
        """
        动作空间动态调整:
        - auto_aim=True : 10 维 (9种机身走位 + 1种停步开火 炮塔自动追踪)
        - auto_aim=False: 28 维 (9种机身 * 3种炮塔旋转 + 1种停步开火)
        """
        return 10 if self.auto_aim else 28
    
    def get_avail_agent_actions(self, agent_id):
        avail_actions = [0] * self.n_actions
        agent = self.agents[agent_id]
        if not agent.is_alive:
            avail_actions[0] = 1 
            return avail_actions
        if self.auto_aim:
            # auto_aim: 0~8 为机动，9 为开火
            avail_actions[0:10] = [1] * 10
        else:
            # manual_aim: 0~26 为机动+炮塔，27 为开火
            avail_actions[0:28] = [1] * 28
        return avail_actions
        
    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def step(self, actions):
        self.steps += 1
        
        for i in range(self.n_agents):
            if self.agent_fire_cooldowns[i] > 0:
                self.agent_fire_cooldowns[i] -= self.delta_time

       # --- 1. 动作解析与执行 ---
        for agent_id, agent in enumerate(self.agents):
            if not agent.is_alive: continue
                
            action = actions[agent_id]
            
            # 每次解析前先重置机身指令
            agent.set_movement(forward=False, backward=False)
            agent.set_turning(left=False, right=False)
            
            # 根据当前的瞄准模式，将动作分发给对应的成员函数处理
            if self.auto_aim:
                self._parse_action_auto_aim(agent_id, agent, action)
            else:
                self._parse_action_manual(agent_id, agent, action)

        # ==========================================
        # 1. 物理步进前：采集环境快照
        pre_stats = self._get_battle_stats()
        # ==========================================

        # --- 2. 物理更新 (完全适配新的 UnitManager 架构) ---
        # 现在所有单位更新和AI结算全部由 UnitManager 自动在内部循环完成
        self.unit_manager.update(self.delta_time, self.unit_manager, self.bullet_manager, self.game_map)
        self.bullet_manager.update(self.delta_time, self.unit_manager, self.game_map)
        
        # --- 3. 视频录制 ---
        if self.use_video:
            self._render_to_video()

        # ==========================================
        # 2. 物理步进后：采集新快照并结算奖励
        post_stats = self._get_battle_stats()
        reward, info = self._calculate_reward(pre_stats, post_stats, actions)
        done = self._check_done()
        # ==========================================
        
        return self.get_obs(), self.get_state(), reward, done, info
    
    def _check_done(self):
        if self.steps >= self.max_steps: return True
        if all(not agent.is_alive for agent in self.agents): return True
        if all(not enemy.is_alive for enemy in self.enemies): return True
        return False

    def check_raycast_unblocked(self, observer, target):
        """仅做纯粹的物理射线遮挡检测 (Raycasting)"""
        line_start = observer.position
        line_end = target.position
        for obstacle_rect in self.game_map.bullet_obstacles:
            if obstacle_rect.clipline(line_start, line_end):
                return False
        return True

    def is_visible_to_agent(self, agent, target):
        """
        结合底层水滴视野与环境层物理遮挡的综合判定
        """
        if not agent.is_alive or not target.is_alive:
            return False
        # 1. 底层查表：检查目标是否在原作者实现的可见列表中（包含水滴视野和距离限制）
        in_underlying_vision = any(u.id == target.id for u in agent.visible_units.units)
        if not in_underlying_vision:
            return False
        # 2. 环境层过滤：如果底层认为可见，叠加一次严格的物理防透视遮挡检测
        return self.check_raycast_unblocked(agent, target)
            
        return True

    def _get_battle_stats(self):
        """获取当前战局的统计信息快照"""
        return {
            "enemy_health": sum([e.health for e in self.enemies]),
            "agent_health": sum([a.health for a in self.agents]),
            "enemy_alive": sum([1 for e in self.enemies if e.is_alive]),
            "agent_alive": sum([1 for a in self.agents if a.is_alive])
        }

    def _calculate_reward(self, pre_stats, post_stats, actions):
        """
        根据当前瞄准模式自动切换奖励函数：
        - auto_aim=True  -> 偏向交战结果
        - auto_aim=False -> 增加手动瞄准过程奖励
        """
        if self.auto_aim:
            return self._calculate_reward_auto_aim(pre_stats, post_stats)
        return self._calculate_reward_manual_aim(pre_stats, post_stats, actions)

    def _calculate_reward_auto_aim(self, pre_stats, post_stats):
        cfg = self.reward_config["auto_aim"]
        enemy_limit_dealt = pre_stats["enemy_health"] - post_stats["enemy_health"]
        agent_limit_received = pre_stats["agent_health"] - post_stats["agent_health"]

        enemies_killed = pre_stats["enemy_alive"] - post_stats["enemy_alive"]
        agents_killed = pre_stats["agent_alive"] - post_stats["agent_alive"]

        reward = 0.0
        reward += (enemy_limit_dealt * cfg["enemy_limit_scale"])
        reward -= (agent_limit_received * cfg["agent_limit_scale"])
        reward += (enemies_killed * cfg["enemy_kill_bonus"])
        reward -= (agents_killed * cfg["agent_killed_penalty"])

        info = {"battle_won": False, "reward_mode": "auto_aim"}
        if post_stats["enemy_alive"] == 0:
            reward += cfg["win_bonus"]
            info["battle_won"] = True
        elif post_stats["agent_alive"] == 0:
            reward -= cfg["lose_penalty"]
        elif self.steps >= self.max_steps:
            reward -= cfg["timeout_penalty"]

        return reward, info

    def _calculate_reward_manual_aim(self, pre_stats, post_stats, actions):
        cfg = self.reward_config["manual_aim"]
        enemy_limit_dealt = pre_stats["enemy_health"] - post_stats["enemy_health"]
        agent_limit_received = pre_stats["agent_health"] - post_stats["agent_health"]

        enemies_killed = pre_stats["enemy_alive"] - post_stats["enemy_alive"]
        agents_killed = pre_stats["agent_alive"] - post_stats["agent_alive"]

        reward = 0.0
        reward += (enemy_limit_dealt * cfg["enemy_limit_scale"])
        reward -= (agent_limit_received * cfg["agent_limit_scale"])
        reward += (enemies_killed * cfg["enemy_kill_bonus"])
        reward -= (agents_killed * cfg["agent_killed_penalty"])

        aim_shaping = 0.0
        fire_shaping = 0.0
        for agent_id, agent in enumerate(self.agents):
            if not agent.is_alive:
                continue

            visible_enemies = [
                enemy for enemy in self.enemies
                if enemy.is_alive and self.is_visible_to_agent(agent, enemy)
            ]
            if not visible_enemies:
                continue

            closest_enemy = min(
                visible_enemies,
                key=lambda enemy: math.hypot(enemy.position[0] - agent.position[0], enemy.position[1] - agent.position[1])
            )

            dx = closest_enemy.position[0] - agent.position[0]
            dy = closest_enemy.position[1] - agent.position[1]
            target_angle = (math.degrees(math.atan2(dy, dx)) + 90) % 360
            angle_diff = abs(agent.get_angle_difference(agent.turret_direction_angle, target_angle))

            if angle_diff <= cfg["aim_good_angle"]:
                aim_shaping += cfg["aim_good_reward"]
            elif angle_diff <= cfg["aim_ok_angle"]:
                aim_shaping += cfg["aim_ok_reward"]
            else:
                aim_shaping -= cfg["aim_bad_penalty"]

            if agent_id < len(actions) and actions[agent_id] == cfg["fire_action_id"]:
                if angle_diff <= cfg["fire_good_angle"]:
                    fire_shaping += cfg["fire_good_reward"]
                else:
                    fire_shaping -= cfg["fire_bad_penalty"]

        reward += aim_shaping
        reward += fire_shaping
        reward -= cfg["step_penalty"]

        info = {
            "battle_won": False,
            "reward_mode": "manual_aim",
            "aim_shaping": round(float(aim_shaping), 4),
            "fire_shaping": round(float(fire_shaping), 4),
        }
        if post_stats["enemy_alive"] == 0:
            reward += cfg["win_bonus"]
            info["battle_won"] = True
        elif post_stats["agent_alive"] == 0:
            reward -= cfg["lose_penalty"]
        elif self.steps >= self.max_steps:
            reward -= cfg["timeout_penalty"]

        return reward, info

    def get_obs(self):
        """
        获取所有智能体的局部观测 (Observation)
        包含自身、友军、敌军、视野内最近子弹以及时间进度。
        """
        obs_list = []
        sight_range = 400.0
        time_ratio = self.steps / self.max_steps
        
        for agent_id, agent in enumerate(self.agents):
            if not agent.is_alive:
                # 自身(6) + 友军((N-1)*5) + 敌军(M*5) + 最近子弹(5) + 时间(1)
                obs_dim = 6 + (self.n_agents - 1) * 5 + self.n_enemies * 5 + 5 + 1
                obs_list.append(np.zeros(obs_dim, dtype=np.float32))
                continue
                
            obs_features = []
            # --- 1. 自身特征 (6维) ---
            hp_ratio = agent.health / agent.max_health
            cos_dir = math.cos(math.radians(agent.direction_angle))
            sin_dir = math.sin(math.radians(agent.direction_angle))
            cos_turret = math.cos(math.radians(agent.turret_direction_angle))
            sin_turret = math.sin(math.radians(agent.turret_direction_angle))
            cooldown_ratio = self.agent_fire_cooldowns[agent_id] / self.fire_cooldown_max
            
            obs_features.extend([hp_ratio, cos_dir, sin_dir, cos_turret, sin_turret, cooldown_ratio])
            
            # --- 2. 友军特征 ---
            for other_id, other_agent in enumerate(self.agents):
                if other_id == agent_id: continue
                if self.is_visible_to_agent(agent, other_agent):
                    rel_x = other_agent.position[0] - agent.position[0]
                    rel_y = other_agent.position[1] - agent.position[1]
                    dist = math.hypot(rel_x, rel_y) / sight_range
                    other_hp = other_agent.health / other_agent.max_health
                    obs_features.extend([1.0, rel_x/100.0, rel_y/100.0, dist, other_hp])
                else:
                    obs_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    
            # --- 3. 敌军特征 ---
            for enemy in self.enemies:
                if self.is_visible_to_agent(agent, enemy):
                    rel_x = enemy.position[0] - agent.position[0]
                    rel_y = enemy.position[1] - agent.position[1]
                    dist = math.hypot(rel_x, rel_y) / sight_range
                    enemy_hp = enemy.health / enemy.max_health
                    obs_features.extend([1.0, rel_x/100.0, rel_y/100.0, dist, enemy_hp])
                else:
                    obs_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    
            # --- 4. 视野内最近的子弹特征 (新增 5 维) ---
            closest_bullet = None
            min_dist = float('inf')
            
            # 遍历底层引擎提供的可见子弹列表
            for bullet in agent.visible_bullets.bullets:
                # 环境层二次过滤：物理防透视遮挡检测（隔着墙飞行的子弹看不到）
                if not self.check_raycast_unblocked(agent, bullet):
                    continue
                    
                dx = bullet.position[0] - agent.position[0]
                dy = bullet.position[1] - agent.position[1]
                dist = math.hypot(dx, dy)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_bullet = bullet
            
            # 如果视野内有符合条件的子弹，提取特征
            if closest_bullet is not None:
                rel_x = closest_bullet.position[0] - agent.position[0]
                rel_y = closest_bullet.position[1] - agent.position[1]
                dist_norm = min_dist / sight_range
                # 安全获取子弹的阵营属性
                team_flag = 1.0 if hasattr(closest_bullet, 'shooter_team') and closest_bullet.shooter_team.name == 'PLAYER' else -1.0
                
                obs_features.extend([1.0, rel_x/100.0, rel_y/100.0, dist_norm, team_flag])
            else:
                # 视野内绝对安全，用 0 填充
                obs_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # --- 5. 时间进度特征 (1维) ---
            obs_features.append(time_ratio)
                    
            obs_list.append(np.array(obs_features, dtype=np.float32))
        return obs_list

    def get_state(self):
        """
        获取全局绝对状态 (Global State)
        """
        state_features = []
        
        # 1. 玩家智能体绝对状态 (由 10 维降为 8 维)
        for agent_id, agent in enumerate(self.agents):
            if agent.is_alive:
                norm_x = agent.position[0] / self.screen_width
                norm_y = agent.position[1] / self.screen_height
                hp_ratio = agent.health / agent.max_health
                
                # --- 暂时注释掉玩家速度 ---
                # v_x = (agent.speed * math.cos(math.radians(agent.direction_angle - 90))) / 100.0
                # v_y = (agent.speed * math.sin(math.radians(agent.direction_angle - 90))) / 100.0
                
                cos_dir = math.cos(math.radians(agent.direction_angle))
                sin_dir = math.sin(math.radians(agent.direction_angle))
                cos_turret = math.cos(math.radians(agent.turret_direction_angle))
                sin_turret = math.sin(math.radians(agent.turret_direction_angle))
                cooldown_ratio = self.agent_fire_cooldowns[agent_id] / self.fire_cooldown_max
                
                # 移除了 v_x, v_y
                state_features.extend([norm_x, norm_y, hp_ratio, cos_dir, sin_dir, cos_turret, sin_turret, cooldown_ratio])
            else:
                # 阵亡填充改为 8 维
                state_features.extend([0.0] * 8)
                
        # 2. 敌方坦克绝对状态 (由 9 维降为 7 维)
        for enemy in self.enemies:
            if enemy.is_alive:
                norm_x = enemy.position[0] / self.screen_width
                norm_y = enemy.position[1] / self.screen_height
                hp_ratio = enemy.health / enemy.max_health
                
                # --- 暂时注释掉敌人速度 ---
                # v_x = (enemy.speed * math.cos(math.radians(enemy.direction_angle - 90))) / 100.0
                # v_y = (enemy.speed * math.sin(math.radians(enemy.direction_angle - 90))) / 100.0
                
                cos_dir = math.cos(math.radians(enemy.direction_angle))
                sin_dir = math.sin(math.radians(enemy.direction_angle))
                cos_turret = math.cos(math.radians(enemy.turret_direction_angle))
                sin_turret = math.sin(math.radians(enemy.turret_direction_angle))
                
                # 移除了 v_x, v_y
                state_features.extend([norm_x, norm_y, hp_ratio, cos_dir, sin_dir, cos_turret, sin_turret])
            else:
                # 阵亡填充改为 7 维
                state_features.extend([0.0] * 7)
                
        # 3. 场上动态子弹特征 (由 5 维降为 3 维)
        MAX_BULLETS = 10
        bullets = self.bullet_manager.bullets
        
        for i in range(MAX_BULLETS):
            if i < len(bullets):
                b = bullets[i]
                norm_x = b.position[0] / self.screen_width
                norm_y = b.position[1] / self.screen_height
                
                # --- 暂时注释掉子弹速度 ---
                # v_x = b.velocity[0] / 100.0 if hasattr(b, 'velocity') else 0.0
                # v_y = b.velocity[1] / 100.0 if hasattr(b, 'velocity') else 0.0
                
                # 修复后的新代码
                team_flag = 1.0 if hasattr(b, 'shooter_team') and b.shooter_team.name == 'PLAYER' else -1.0
                
                # 移除了 v_x, v_y
                state_features.extend([norm_x, norm_y, team_flag])
            else:
                # 空位填充改为 3 维
                state_features.extend([0.0] * 3)

        time_ratio = self.steps / self.max_steps
        state_features.append(time_ratio)
        return np.array(state_features, dtype=np.float32)

    def _render_to_video(self):
        self.screen.fill((50, 50, 70))
        camera_offset = [0, 0]
        
        # 渲染逻辑适配：交由 UnitManager 统一绘制
        self.game_map.draw(self.screen, camera_offset)
        self.unit_manager.draw(self.screen, camera_offset)
        self.bullet_manager.draw(self.screen, camera_offset)
        
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)

    def _parse_action_auto_aim(self, agent_id, agent, action):
        """
        模式 A: 开启辅助瞄准时的动作解析 (10 维)
        """
        # 1. 走位与开火判定
        if action < 9:
            chassis_action = action
            if chassis_action == 1: agent.set_movement(forward=True, backward=False)
            elif chassis_action == 2: agent.set_movement(forward=False, backward=True)
            elif chassis_action == 3: agent.set_turning(left=True, right=False)
            elif chassis_action == 4: agent.set_turning(left=False, right=True)
            elif chassis_action == 5:
                agent.set_movement(forward=True, backward=False)
                agent.set_turning(left=True, right=False)
            elif chassis_action == 6:
                agent.set_movement(forward=True, backward=False)
                agent.set_turning(left=False, right=True)
            elif chassis_action == 7:
                agent.set_movement(forward=False, backward=True)
                agent.set_turning(left=True, right=False)
            elif chassis_action == 8:
                agent.set_movement(forward=False, backward=True)
                agent.set_turning(left=False, right=True)
                
        elif action == 9:
            # 停步开火
            if self.agent_fire_cooldowns[agent_id] <= 0:
                bullet = agent.fire(NormalShell)
                if bullet:
                    self.bullet_manager.add_bullet(bullet)
                    self.agent_fire_cooldowns[agent_id] = self.fire_cooldown_max

        # 2. 环境层接管炮塔的“辅助瞄准”
        closest_enemy = None
        min_dist = float('inf')
        for enemy in self.enemies:
            if enemy.is_alive and self.is_visible_to_agent(agent, enemy):
                dx = enemy.position[0] - agent.position[0]
                dy = enemy.position[1] - agent.position[1]
                dist = math.hypot(dx, dy)
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy = enemy
        
        if closest_enemy is not None:
            dx = closest_enemy.position[0] - agent.position[0]
            dy = closest_enemy.position[1] - agent.position[1]
            target_angle = math.degrees(math.atan2(dy, dx)) + 90
            target_angle = target_angle % 360
            if target_angle < 0: target_angle += 360
            agent.turret_target_angle = target_angle
        else:
            agent.turret_target_angle = agent.direction_angle

    def _parse_action_manual(self, agent_id, agent, action):
        """
        模式 B: 关闭辅助瞄准，完全手动操作时的动作解析 (28 维)
        """
        if action < 27:
            chassis_action = action % 9
            turret_action = action // 9
            
            if chassis_action == 1: agent.set_movement(forward=True, backward=False)
            elif chassis_action == 2: agent.set_movement(forward=False, backward=True)
            elif chassis_action == 3: agent.set_turning(left=True, right=False)
            elif chassis_action == 4: agent.set_turning(left=False, right=True)
            elif chassis_action == 5:
                agent.set_movement(forward=True, backward=False)
                agent.set_turning(left=True, right=False)
            elif chassis_action == 6:
                agent.set_movement(forward=True, backward=False)
                agent.set_turning(left=False, right=True)
            elif chassis_action == 7:
                agent.set_movement(forward=False, backward=True)
                agent.set_turning(left=True, right=False)
            elif chassis_action == 8:
                agent.set_movement(forward=False, backward=True)
                agent.set_turning(left=False, right=True)
                
            # 炮塔手动旋转
            if turret_action == 1:
                agent.turret_target_angle = agent.turret_direction_angle - 15.0
            elif turret_action == 2:
                agent.turret_target_angle = agent.turret_direction_angle + 15.0
            else:
                agent.turret_target_angle = agent.turret_direction_angle
                
        elif action == 27:
            # 停步锁定并开火
            agent.turret_target_angle = agent.turret_direction_angle
            if self.agent_fire_cooldowns[agent_id] <= 0:
                bullet = agent.fire(NormalShell)
                if bullet:
                    self.bullet_manager.add_bullet(bullet)
                    self.agent_fire_cooldowns[agent_id] = self.fire_cooldown_max

    def close(self):
        if self.use_video and self.video_writer is not None:
            self.video_writer.release()
        pygame.quit()