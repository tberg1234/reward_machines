"""
This code add event detectors to the Ant3 Environment
"""
import gym
import numpy as np
from demo import *
from reward_machines.rm_environment import RewardMachineEnv

class BoxmanEnv(gym.Wrapper):
    def __init__(self):
        # Note that the current position is key for our tasks
        # super().__init__(HalfCheetahEnv(exclude_current_positions_from_observation=False))

        #Start state 1
        player_starting_pos = (5,6)
        sprite_starting_pos = None
        tasks = ['blue', 'square']
        env, collect_env, goals = create_env(tasks, negate=False, include_purple_square=False, goal_condition=None, shift_up=False, player_starting_pos=player_starting_pos, sprite_starting_pos=sprite_starting_pos)
        self.collect_env = collect_env
        self.collected = []
        self.goals = goals
        self.total_reward=0
        super().__init__(env)


    def step(self, action):
        # executing the action in the environment
        obs, reward_dict, done_dict, info = self.env.step(action)
        self.info = info
        reward = max([reward_dict[l] for l in info['matching_labels']])
        done = any([done_dict[l] for l in info['matching_labels']])
        self.total_reward += reward

        return obs, reward, done, info
    
    def get_events(self):
        events = ''
        step_collected = self.collect_env.collected.sprites()
        if step_collected != self.collected:
            sprite_symbols = get_sprite_symbols(step_collected, self.collected)
            for ss in sprite_symbols:
                events += get_sprite_to_alphabet_symbol(ss)
            self.collected = step_collected
        return events

    def reset(self):
        self.env.reset()
        self.collect_env.reset()
        self.collected = []
        #return self.env.get_features()


class MyBoxmanEnvRM1(RewardMachineEnv):
    def __init__(self):
        env = BoxmanEnv()
        rm_files = ["./envs/boxman/reward_machines/t1.txt"]
        super().__init__(env, rm_files)