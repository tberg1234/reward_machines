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
        env, collect_env, goals = create_env(tasks, negate=False, include_purple_square=True, goal_condition=None, shift_up=False, player_starting_pos=player_starting_pos, sprite_starting_pos=sprite_starting_pos)
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
        rm_files = ["./envs/boxman/reward_machines/videogameworld1.txt"]
        super().__init__(env, rm_files)

class MyBoxmanEnvRM2(RewardMachineEnv):
    def __init__(self):
        env = BoxmanEnv()
        rm_files = ["./envs/boxman/reward_machines/videogameworld2.txt"]
        super().__init__(env, rm_files)

class MyBoxmanEnvRM3(RewardMachineEnv):
    def __init__(self):
        env = BoxmanEnv()
        rm_files = ["./envs/boxman/reward_machines/videogameworld3.txt"]
        super().__init__(env, rm_files)

class MyBoxmanEnvRM4(RewardMachineEnv):
    def __init__(self):
        env = BoxmanEnv()
        rm_files = ["./envs/boxman/reward_machines/videogameworld4.txt"]
        super().__init__(env, rm_files)

class MyBoxmanEnvRM5(RewardMachineEnv):
    def __init__(self):
        env = BoxmanEnv()
        rm_files = ["./envs/boxman/reward_machines/videogameworld5.txt"]
        super().__init__(env, rm_files)

class MyBoxmanEnvRM6(RewardMachineEnv):
    """F(b&e) & G!(c|a|d) - Eventually blue AND square, always avoid purple/white/circle"""
    def __init__(self):
        env = BoxmanEnv()
        rm_files = ["./envs/boxman/reward_machines/videogameworld6.txt"]
        super().__init__(env, rm_files)

class MyBoxmanEnvRM7(RewardMachineEnv):
    """F(c&d) & G!(b|a|e) - Eventually purple AND circle, always avoid blue/white/square"""
    def __init__(self):
        env = BoxmanEnv()
        rm_files = ["./envs/boxman/reward_machines/videogameworld7.txt"]
        super().__init__(env, rm_files)

class MyBoxmanEnvRM8(RewardMachineEnv):
    """F(b) & G!(c|a|e) - Eventually blue, always avoid purple/white/square"""
    def __init__(self):
        env = BoxmanEnv()
        rm_files = ["./envs/boxman/reward_machines/videogameworld8.txt"]
        super().__init__(env, rm_files)

class MyBoxmanEnvRM9(RewardMachineEnv):
    """G(F(b&e)) & G(F(c&d)) & G!a - Patrol both, always avoid white"""
    def __init__(self):
        env = BoxmanEnv()
        rm_files = ["./envs/boxman/reward_machines/videogameworld9.txt"]
        super().__init__(env, rm_files)

class MyBoxmanEnvRM10(RewardMachineEnv):
    """F(G((b|c)&e)) & G!(a|d) - Eventually always (blue OR purple) AND square, avoid white/circle"""
    def __init__(self):
        env = BoxmanEnv()
        rm_files = ["./envs/boxman/reward_machines/videogameworld10.txt"]
        super().__init__(env, rm_files)