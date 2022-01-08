"""Image dataset for MCTS."""

import os
import pickle

import numpy as np
from ravens import dataset, tasks
from ravens.tasks import cameras
import tensorflow as tf

# See transporter.py, regression.py, dummy.py, task.py, etc.
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.25, 0.25], [0, 0.28]])
# BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

# Names as strings, REVERSE-sorted so longer (more specific) names are first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]


class DatasetMCTS:
  """A simple image dataset class for loading
      different tasks for training MCTS agent.
  """

  def __init__(self, path_list):
    """A simple RGB-D image dataset."""
    
    self.path_list = path_list
    self.dataset_num = len(self.path_list)
    self.sample_set_list = []
    for i in range(len(self.path_list)):
      self.sample_set_list.append([])
    self.n_episodes_list = [0] * len(self.path_list)

    # Track existing dataset if it exists.
    for i, path in enumerate(self.path_list):
      color_path = os.path.join(path, 'action')
      max_seed = -1
      if tf.io.gfile.exists(color_path):
        for fname in sorted(tf.io.gfile.listdir(color_path)):
          if '.pkl' in fname:
            seed = int(fname[(fname.find('-') + 1):-4])
            self.n_episodes_list[i] += 1
            max_seed = max(max_seed, seed)
      print(f'[Dataset Loaded] Path: {path} N_Episodes: {self.n_episodes_list[i]}')

    # Primitive index
    # 0 - put a block on a base
    # 1 - stack a block on another block
    # 2 - stack a block on top of two blocks
    self.primitive_idx_set = [0, 1, 2]

    self._cache = {}

  def set(self, dataset_idx, episodes):
    """Limit random samples to specific fixed set for a dataset."""

    self.sample_set_list[dataset_idx] = episodes
    print(f'Dataset: {self.path_list[dataset_idx]}')
    print(f'Dataset Episode: {self.sample_set_list[dataset_idx]}')

  def load(self, dataset_id, episode_id, images=True, cache=False):
    """Load data from a saved episode.

    Args:
      dataset_id: the ID of the dataset to be loaded.
      episode_id: the ID of the episode to be loaded.
      images: load image data if True.
      cache: load data from memory if True.

    Returns:
      episode: list of (obs, act, reward, info) tuples.
      seed: random seed used to initialize the episode.
    """
    
    def load_field(dataset_id, episode_id, field, fname):

      # Check if sample is in cache.
      if cache:
        if episode_id in self._cache:
          if field in self._cache[episode_id]:
            return self._cache[episode_id][field]
        else:
          self._cache[episode_id] = {}

      # Load sample from files.
      path = os.path.join(self.path_list[dataset_id], field)
      with open(os.path.join(path, fname), 'rb') as f:
        try:
          data = pickle.load(f)
        except EOFError:
          data = None
      if cache:
        self._cache[episode_id][field] = data
      return data

    # Get filename and random seed used to initialize episode.
    seed = None
    path = os.path.join(self.path_list[dataset_id], 'action')

    for fname in sorted(tf.io.gfile.listdir(path)):
      if f'{episode_id:06d}' in fname:
        seed = int(fname[(fname.find('-') + 1):-4])

        # Load data.
        # pick_color = load_field(dataset_id, episode_id, 'pick_color', fname)
        # pick_depth = load_field(dataset_id, episode_id, 'pick_depth', fname)
        color = load_field(dataset_id, episode_id, 'color', fname)
        depth = load_field(dataset_id, episode_id, 'depth', fname)
        action = load_field(dataset_id, episode_id, 'action', fname)
        reward = load_field(dataset_id, episode_id, 'reward', fname)
        info = load_field(dataset_id, episode_id, 'info', fname)

        # Reconstruct episode.
        episode = []
        for i in range(len(action)):
          # pick_obs  = {'color': pick_color[i], 'depth': pick_depth[i]} if images else {}
          obs = {'color': color[i], 'depth': depth[i]} if images else {}
          # episode.append((pick_obs, obs, action[i], reward[i], info[i]))
          episode.append((obs, action[i], reward[i], info[i]))
        return episode, seed

  def sample(self, primitive_idx, images=True, cache=False):
    """Uniformly sample from the dataset.

    Args:
      primitive_idx: index of the primitive
      images: load image data if True.
      cache: load data from memory if True.

    Returns:
      sample: randomly sampled (obs, act, reward, info) tuple.
      goal: the last (obs, act, reward, info) tuple in the episode.
    """

    assert primitive_idx in self.primitive_idx_set, "[Dataset MCTS] primitive_idx should be in self.primitive_idx_set"

    # Train the permissible action.
    permissible_flag = 0
    if primitive_idx == 2:
        permissible_flag = np.random.choice([1, 0, 0, 0, 0, 0, 0, 0])
    # if primitive_idx == 1:
    #     permissible_flag = np.random.choice([1, 0, 0, 0, 0, 0, 0, 0])

    episode_primitive = []
    while True:
      # Choose a random dataset.
      dataset_id = np.random.choice(range(len(self.n_episodes_list)))
      
      # Choose a random episode.
      if len(self.sample_set_list[dataset_id]) > 0:
        episode_id = np.random.choice(self.sample_set_list[dataset_id])
      else:
        episode_id = np.random.choice(range(self.n_episodes_list[dataset_id]))

      # Load the episode
      episode, _ = self.load(dataset_id, episode_id, images, cache)

      # Choose the transition acting with the motion primitive.
      for i in range(len(episode)):
        if ('primitive' in episode[i][3]) and (episode[i][3]['primitive'] == primitive_idx):
          episode_primitive.append(episode[i])

      if len(episode_primitive) > 0:
        break

    # Return random observation action pair (and goal) from episode.
    if permissible_flag == 0:
      i = np.random.choice(range(len(episode_primitive)))
      sample = episode_primitive[i]
    else:
      if primitive_idx == 2:
        # For primitive 2, it should not start with first two actions.
        i = np.random.choice(range(2))
      # elif primitive_idx == 1:
      #   # For primitive, it should not start with the first actions.
      #   i = 0
      sample = episode[i]
      sample[1]['pose1'] = sample[1]['pose0']
      sample[3]['primitive'] = primitive_idx

    print(f'dataset: {self.path_list[dataset_id]} dataset_id: {dataset_id} episode_id: {episode_id} permissible_flag: {permissible_flag}')
    
    # The sample must not be random actions.
    if permissible_flag == 0:
      assert not sample[3]['random']

    return sample, permissible_flag