#!/usr/bin/env python
import os
import cv2
import numpy as np
from ravens import utils as U
from ravens import tasks, cameras

# See transporter.py, regression.py, dummy.py, task.py, load.py, etc.
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.25, 0.25], [0, 0.28]])

# Task names as strings, REVERSE-sorted so longer (more specific) names come first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]


def get_max_episode_len(path):
    """A somewhat more scalable way to get the max episode lengths."""
    path = path.replace('data/', '')
    path = path.replace('goals/', '')
    task = tasks.names[path]()
    max_steps = task.max_steps - 1  # Remember, subtract one!
    return max_steps


def process_depth(img, cutoff=10):
    # Turn to three channels and zero-out values beyond cutoff.
    w,h = img.shape
    d_img = np.zeros([w,h,3])
    img = img.flatten()
    img[img > cutoff] = 0.0
    img = img.reshape([w,h])
    for i in range(3):
        d_img[:,:,i] = img

    # Scale values into [0,255) and make type uint8.
    assert np.max(d_img) > 0.0
    d_img = 255.0 / np.max(d_img) * d_img
    d_img = np.array(d_img, dtype=np.uint8)
    for i in range(3):
        d_img[:,:,i] = cv2.equalizeHist(d_img[:,:,i])
    return d_img


def get_heightmap(obs):
    """Following same implementation as in transporter.py."""
    heightmaps, colormaps = U.reconstruct_heightmaps(
        obs['color'], obs['depth'], CAMERA_CONFIG, BOUNDS, PIXEL_SIZE)
    colormaps = np.float32(colormaps)
    heightmaps = np.float32(heightmaps)

    # Fuse maps from different views.
    valid = np.sum(colormaps, axis=3) > 0
    repeat = np.sum(valid, axis=0)
    repeat[repeat == 0] = 1
    colormap = np.sum(colormaps, axis=0) / repeat[..., None]
    colormap = np.uint8(np.round(colormap))
    heightmap = np.max(heightmaps, axis=0)
    return colormap, heightmap


class DatasetReal:

    def __init__(self, path_list):
        """A simple RGB-D image dataset to work with real data."""
        self.path_list = path_list
        self.dataset_num = len(self.path_list)
        self.sample_set_list = []
        for _ in range(len(self.path_list)):
            self.sample_set_list.append([])

        self.n_episodes_list = [0] * len(self.path_list)

        # Track existing dataset if it exists.
        for i, path in enumerate(self.path_list):
            episode_num_list = os.listdir(path)
            self.n_episodes_list[i] = len(episode_num_list)
            print(f'[Dataset Loaded] Path: {path} N_Episodes: {self.n_episodes_list[i]}')

        self._cache = dict()

        # Only for goal-conditioned Transporters, if we want more goal images.
        self.subsample_goals = False

    def set(self, dataset_idx, episodes):
        """Limit random samples to specific fixed set."""
        
        self.sample_set_list[dataset_idx] = episodes
        print(f'Dataset: {self.path_list[dataset_idx]}')
        print(f'Dataset Episode: {self.sample_set_list[dataset_idx]}')

    def load(self, dataset_id, episode_id):
        """Load data from a saved episode.

        Args:
          dataset_id: the ID of the dataset to be loaded.
          episode_id: the ID of the episode to be loaded.

        Returns:
          episode: list of (obs, act) tuples. obs contains 
            colormap and heightmap.
        """

        task_dir = self.path_list[dataset_id]
        data_dir = os.path.join(task_dir, str(episode_id))

        file_list = os.listdir(data_dir)
        assert (len(file_list) + 1) % 3 == 0
        num_steps = int((len(file_list) + 1) / 3) # The last observation does not have an action

        episode = []
        for step_i in range(num_steps):
            # Load the colormap.
            cmap_file = os.path.join(data_dir, f'cmap_{step_i}.npy')
            cmap = np.load(cmap_file)

            # Load the heightmap.
            hmap_file = os.path.join(data_dir, f'hmap_{step_i}.npy')
            hmap = np.load(hmap_file)

            if step_i < (num_steps - 1):
                # Load the action.
                action_file = os.path.join(data_dir, f'action_{step_i}.txt')
                with open(action_file, 'r') as f:
                    lines = f.readlines()
                pick_action = [float(i) for i in lines[0].split()]
                place_action = [float(i) for i in lines[1].split()]
                random = float(lines[2][0])
                assert pick_action[2] == 0.0
                action = {}
                action['pose0'] = pick_action
                action['pose1'] = place_action
                if random > 0:
                    action['random'] = True
                else:
                    action['random'] = False
            else:
                action = None

            episode.append([cmap, hmap, action])

        return episode

    def random_sample(self, goal_images=False):
        """Randomly sample from the dataset uniformly.

        Daniel: The 'cached_load' will use the load (from pickle file) to
        load the list, and then extract the time step `i` within it as the
        data point. I'm also adding a `goal_images` feature to load the last
        information. The last information isn't in a list, so we don't need
        to extract an index. That is, if loading a 1-length time step, we
        should see this:

        In [11]: data = pickle.load( open('last_color/000099-1.pkl', 'rb') )
        In [12]: data.shape
        Out[12]: (3, 480, 640, 3)

        In [13]: data = pickle.load( open('color/000099-1.pkl', 'rb') )
        In [14]: data.shape
        Out[14]: (1, 3, 480, 640, 3)

        Update: now using goal_images for gt_state, but here we should interpret
        `goal_images` as just giving the 'info' portion to the agent.
        """
        # Randomly select a dataset.
        dataset_id = np.random.choice(range(len(self.n_episodes_list)))
        
        # Randomly select an episode.
        if len(self.sample_set_list[dataset_id]) > 0:
            iepisode = np.random.choice(self.sample_set_list[dataset_id])
        else:
            iepisode = np.random.choice(range(self.n_episodes_list[dataset_id]))

        print(f'{self.path_list[dataset_id]} -- {iepisode}')
        
        # Load the episode.
        episode = self.load(dataset_id, iepisode)
        
        # Pick a step in the episode that is not random action.
        while True:
            i = np.random.choice(range(len(episode)-1))
            random = episode[i][2]['random']
            if not random:
                break
    
        cmap = episode[i][0]
        hmap = episode[i][1]
        act  = episode[i][2]

        if goal_images:
            cmap_g = episode[-1][0]
            hmap_g = episode[-1][1]

            return cmap, hmap, act, cmap_g, hmap_g
        else:
            return cmap, hmap, act