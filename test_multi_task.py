"""Testing code for GCTN."""

import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ravens import agents
import ravens.tasks_multi as tasks
from ravens.environment_mcts import EnvironmentMCTS
from ravens.dataset_test import Dataset

task_list = [
  'put-block-base-mcts',
  'stack-square-mcts',
  'stack-t-mcts',
  'stack-tower-mcts',
  'stack-pyramid-mcts',
  'stack-palace-mcts']

random_seed_list = [0, 1, 2]
n_demo = 10

model_root_dir = '/home/hongtao/src/deformable-ravens'
data_dir = '/home/hongtao/Dropbox/RavensTAMP/data_test'
assets_root = '/home/hongtao/src/ravens_tamp/ravens/environments/assests/'
disp = True

for random_seed in random_seed_list:
  attention_model_path = '/home/hongtao/src/deformable-ravens/checkpoints/GCTN-Multi-transporter-goal-10-0-rots-36-fin_g/attention-ckpt-40000.h5'
  transport_model_path = '/home/hongtao/src/deformable-ravens/checkpoints/GCTN-Multi-transporter-goal-10-0-rots-36-fin_g/transport-ckpt-40000.h5'

  for task_name in task_list:
    env = EnvironmentMCTS(
      assets_root,
      disp,
      shared_memory=False,
      hz=480)
    task = tasks.name[task_name](pp=True)