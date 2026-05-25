import random
import numpy as np

class EpisodeBuffer:
    def __init__(self, capacity=1000):
        # 容量现在指的是可以装载多少个完整的 Episode
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push_episode(self, episode_trajectory):
        """
        存入一条完整的轨迹
        episode_trajectory 格式: [(s, a, r, s', done), (s, a, r, s', done), ...]
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = episode_trajectory
        self.position = (self.position + 1) % self.capacity

    def sample_transitions(self, batch_size):
        """
        【过渡期采样逻辑】：为了适配当前无 RNN 的 DQN，
        我们从存储的轨迹中随机抽出 Batch Size 数量的单步 Transition。
        这兼顾了轨迹存储的结构和 DQN 对打乱相关性的需求。
        """
        # 1. 随机挑选几个 Episode
        sampled_episodes = random.sample(self.buffer, min(len(self.buffer), batch_size // 10 + 1))
        
        # 2. 从这些 Episode 中打平并随机抽取所需数量的单步帧
        all_transitions = []
        for ep in sampled_episodes:
            all_transitions.extend(ep)
            
        batch = random.sample(all_transitions, min(len(all_transitions), batch_size))
        
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)