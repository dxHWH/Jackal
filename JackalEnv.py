import pygame
import os
import random
import numpy as np

# 导入游戏原有模块
from Parameter import *
from Unit.Tank.Tank import create_tank, create_enemy_tank
from Map.Map import create_border_map
from Bullet.BulletManager import BulletManager
from Unit.EnemyAI import EnemyAI
from GameMode import Team

class JackalEnv:
    def __init__(self, headless=True, fixed_delta_time=0.1):
        """
        初始化多智能体环境
        :param headless: 是否开启无头模式（关闭渲染以加速训练）
        :param fixed_delta_time: 固定的时间步长，确保环境转移的绝对确定性
        """
        self.headless = headless
        self.delta_time = fixed_delta_time
        
        # 开启无头模式：设置 SDL 视频驱动为 dummy，这样 pygame 可以在没有显示器的情况下运行，且不会消耗 GPU/屏幕渲染资源
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            
        pygame.init()
        
        # 即使在无头模式下，Pygame 也需要一个隐藏的 Surface 来处理内部的图像加载逻辑
        self.screen_width, self.screen_height = 960, 640
        if not self.headless:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Jackal MARL Environment")
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            
        # RL 相关的基本配置
        self.n_agents = 1      # 当前玩家数量（后续可扩展为多个）
        self.n_enemies = 3     # 敌人数量
        
        self.game_map = None
        self.agents = []
        self.enemies = []
        self.enemy_ais = []
        self.bullet_manager = None
        
        self.steps = 0
        self.max_steps = 500   # 每个 Episode 的最大步数

    def reset(self):
        """
        重置环境，返回初始观测和全局状态
        """
        self.steps = 0
        
        # 重置地图与子弹管理器
        self.game_map = create_border_map()
        self.bullet_manager = BulletManager()
        
        # 重置智能体 (当前以单智能体为主，后续扩展为 list)
        self.agents = [create_tank(1, Team.PLAYER, position=(400, 300))]
        
        # 重置敌人与基于规则的 EnemyAI
        self.enemies = []
        self.enemy_ais = []
        for i in range(self.n_enemies):
            enemy_x = random.randint(100, 700)
            enemy_y = random.randint(100, 500)
            enemy = create_enemy_tank(100 + i, position=(enemy_x, enemy_y))
            self.enemies.append(enemy)
            
            # EnemyAI 需要一个目标，目前先默认追踪第一个 agent
            ai = EnemyAI(enemy, self.agents[0], self.bullet_manager, self.game_map.obstacles)
            self.enemy_ais.append(ai)
            
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """
        执行环境步进
        :param actions: 智能体的动作指令集合
        :return: obs, state, reward, done, info
        """
        self.steps += 1
        
        # ---------------------------------------------------------
        # 1. 动作解析与应用 (阶段二的重点，此处预留接口)
        # ---------------------------------------------------------
        # TODO: 解析离散/连续动作，转换为 agent.set_movement() 和 agent.set_turning()
        
        # ---------------------------------------------------------
        # 2. 物理与逻辑更新 (固定 delta_time 确保确定性)
        # ---------------------------------------------------------
        # 更新玩家智能体
        for agent in self.agents:
            agent.update(self.delta_time, self.game_map.obstacles)
            
        # 更新环境内的敌方 AI
        for ai in self.enemy_ais:
            ai.update(self.delta_time)
            
        # 更新所有子弹及碰撞检测
        all_units = self.agents + self.enemies
        self.bullet_manager.update(self.delta_time, all_units, self.game_map.obstacles)
        
        # ---------------------------------------------------------
        # 3. 收集反馈信息
        # ---------------------------------------------------------
        reward = self._calculate_reward()
        done = self._check_done()
        info = {}
        
        return self.get_obs(), self.get_state(), reward, done, info

    def render(self):
        """
        渲染画面（仅用于可视化测试，训练时应保持 headless=True）
        """
        if self.headless:
            return
            
        self.screen.fill((50, 50, 70))
        camera_offset = [0, 0] # 暂时固定视角
        
        self.game_map.draw(self.screen, camera_offset)
        for enemy in self.enemies:
            if enemy.is_alive:
                enemy.draw(self.screen, camera_offset)
        for agent in self.agents:
            if agent.is_alive:
                agent.draw(self.screen, camera_offset)
        self.bullet_manager.draw(self.screen, camera_offset)
        
        pygame.display.flip()

    # ---------------- 预留的核心方法（将在后续阶段实现） ----------------

    def get_obs(self):
        """获取各智能体的局部观测 (Local Observation)"""
        # TODO: 实现射线检测或网格感受野
        return [np.zeros(10)] * self.n_agents

    def get_state(self):
        """获取全局状态 (Global State) 供价值分解网络的 Mixer 使用"""
        # TODO: 提取包含精确空间位置、血量的全局矩阵
        return np.zeros(20)

    def _calculate_reward(self):
        """拦截伤害事件计算奖励"""
        # TODO: 根据敌我双方血量变化计算密集奖励
        return [0.0] * self.n_agents

    def _check_done(self):
        """检查回合是否结束"""
        if self.steps >= self.max_steps:
            return True
        # 如果所有 agent 阵亡，或所有 enemy 阵亡，也应结束
        agents_dead = all(not agent.is_alive for agent in self.agents)
        enemies_dead = all(not enemy.is_alive for enemy in self.enemies)
        return agents_dead or enemies_dead

    def close(self):
        pygame.quit()