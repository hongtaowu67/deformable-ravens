#!/usr/bin/env python
"""
The main training script, and testing (if using a non-custom task).

A note about random seeds: the seed we set before drawing the episode's
starting state will determine the initial configuration of objects, for both
standard ravens and any custom envs (so far). Currently, if we ask to set 100
demos, the code will run through seeds 0 through 99. Then, for _test_ demos,
we offset by 10**max_order, so that with 100 demos (max_order=2) we'd start
with 10**2=100 and proceed from there. This way, if doing 20 test episodes,
then ALL snapshots are evaluated on seeds {100, 101, ..., 119}. If we change
max_order=3, which we should for an actual paper, then this simply means the
training is done on {0, 1, ..., 999} and testing starts at seed 1000.

With the custom deformable tasks, I have a separate load.py script. That one
also had max_order=3 (so when doing 100 demos, I was actually starting at
seed 1000, no big deal). However, I now have max_order=4 to start at 10K,
because (a) we will want to use 1000 demos eventually, and (b) for the
deformable tasks, sometimes the initial state might already be 'done', so I
ignore that data and re-sample the starting state with the next seed. Having
load.py start at seed 10K will give us a 'buffer zone' of seeds to protect
the train and test from overlapping.

With goal-conditioning, IF the goals are drawn by sampling from a similar
same starting state distribution (as in the case with insertion-goal) then
use generate_goals.py and set max_order=5 so that there's virtually no chance
of random seed overlap.

When training on a new machine, we can run this script with "1000 demos" for
deformable tasks. It will generate data (but not necessarily "1000 demos"
because max_order determines the actual amount), but exit before we can do
any training, and then we can use subsequent scripts with {1, 10, 100} demos
for actual training.
"""
import datetime
import os
import time
import argparse
import sys
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ravens import Dataset, Environment, agents, tasks
from ravens.dataset_real import DatasetReal

# Of critical importance! Do 2 for max of 100 demos, 3 for max of 1000 demos.
MAX_ORDER = 3

data_dir = '/home/hongtao/Dropbox/RavensTAMP/real/train'

task_list = [
  'row',
  'tower',  
  'square'
]

h_only = False

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',            default='0')
    parser.add_argument('--disp',           action='store_true')
    parser.add_argument('--task',           default='put-block-base-mcts')
    parser.add_argument('--agent',          default='transporter-goal')
    parser.add_argument('--num_demos',      default='1')
    parser.add_argument('--num_rots',       default=36, type=int)
    parser.add_argument('--hz',             default=240.0, type=float)
    parser.add_argument('--gpu_mem_limit',  default=None)
    parser.add_argument('--subsamp_g',      action='store_true')
    
    args = parser.parse_args()

    # Configure which GPU to use.
    cfg = tf.config.experimental
    gpus = cfg.list_physical_devices('GPU')
    if len(gpus) == 0:
        print('No GPUs detected. Running with CPU.')
    else:
        cfg.set_visible_devices(gpus[int(args.gpu)], 'GPU')

    # Configure how much GPU to use.
    if args.gpu_mem_limit is not None:
        MEM_LIMIT = 1024 * int(args.gpu_mem_limit)
        print(args.gpu_mem_limit)
        dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=MEM_LIMIT)]
        cfg.set_virtual_device_configuration(gpus[0], dev_cfg)

    # Initialize task. Later, initialize Environment if necessary.
    dataset_dir_list = [os.path.join(data_dir, f'{task}') for task in task_list]
    dataset = DatasetReal(dataset_dir_list)
    if args.subsamp_g:
        dataset.subsample_goals = True  

    # Evaluate on increasing orders of magnitude of demonstrations.
    num_train_runs = 3  # to measure variance over random initialization
    num_train_iters = 40000
    test_interval = 4000

    # Check if it's goal-conditioned.
    goal_conditioned = True

    # Do multiple training runs from scratch with TensorFlow random initialization.
    for train_run in range(num_train_runs):

        # Set up tensorboard logger.
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join('logs', args.agent, current_time, 'train')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Set the beginning of the agent name.
        name = f'GCTN-Multi-{args.agent}-{args.num_demos}-{train_run}-transfer'

        # Initialize agent and limit random dataset sampling to fixed set.
        tf.random.set_seed(train_run)
        
        assert 'transporter-goal' in args.agent
        assert goal_conditioned
        name = f'{name}-rots-{args.num_rots}'
        if args.subsamp_g:
            name += '-sub_g'
        else:
            name += '-fin_g'
        agent = agents.names[args.agent](name,
                                         args.task,
                                         num_rotations=args.num_rots,
                                         h_only=h_only)

        # Limit random data sampling to fixed set.
        np.random.seed(train_run)
        num_demos = int(args.num_demos)

        episodes_list = []
        for i in range(len(task_list)):
            max_demos = dataset.n_episodes_list[i]
            assert max_demos >= num_demos
            episodes = np.random.choice(range(max_demos), num_demos, False)
            episodes_list.append(episodes)
            dataset.set(i, episodes_list[i])

        performance = []
        while agent.total_iter < num_train_iters:
            # Train agent.
            tf.keras.backend.set_learning_phase(1)
            agent.train(dataset, num_iter=test_interval, writer=train_summary_writer, real=True)
            tf.keras.backend.set_learning_phase(0)
