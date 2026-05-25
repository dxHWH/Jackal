import random

import numpy as np


class EpisodeReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = int(buffer_size)
        self.episodes = []

    def __len__(self):
        return len(self.episodes)

    def clear(self):
        self.episodes.clear()

    def insert_episode_batch(self, episode_batch):
        if len(self.episodes) >= self.buffer_size:
            self.episodes.pop(0)
        self.episodes.append(episode_batch)

    def can_sample(self, batch_size):
        return len(self.episodes) >= int(batch_size)

    def sample(self, batch_size):
        selected = random.sample(self.episodes, int(batch_size))
        keys = selected[0].keys()
        batch = {}
        for key in keys:
            batch[key] = np.stack([ep[key] for ep in selected], axis=0)
        return batch
