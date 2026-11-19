# Implementation Plan: New LTL Specifications for Reward Machines

## Overview

This plan adds 5 new LTL specifications for evaluation in the reward machines framework.

## Specifications to Implement

| # | LTL Formula | Short Name | Description |
|---|-------------|------------|-------------|
| 1 | `F(b & e) & G!(c \| a \| d)` | BlueSquareSafe | Eventually blue AND square, always avoid purple/white/circle |
| 2 | `F(c & d) & G!(b \| a \| e)` | PurpleCircleSafe | Eventually purple AND circle, always avoid blue/white/square |
| 3 | `F(b) & G!(c \| a \| e)` | BlueSafe | Eventually blue, always avoid purple/white/square |
| 4 | `G(F(b & e)) & G(F(c & d)) & G!a` | PatrolBoth | Always eventually (blue AND square), always eventually (purple AND circle), always avoid white |
| 5 | `F(G((b \| c) & e)) & G!(a \| d)` | StayBlueOrPurpleSquare | Eventually always ((blue OR purple) AND square), always avoid white/circle |

### Symbol Mapping (from existing codebase)
- `a` = white
- `b` = blue
- `c` = purple
- `d` = circle
- `e` = square

---

## Files to Create/Modify

### 1. Reward Machine Specification Files (5 new .txt files)

**Location:** `reward_machines/reward_machines/envs/boxman/reward_machines/`

#### File 1: `videogameworld6.txt` - F(b&e) & G!(c|a|d)
```
0 # initial state
[-1] # terminal state (trap for safety violation), [1] for success
(0,0,'!b&!e&!c&!a&!d',ConstantRewardFunction(0))
(0,1,'b&e&!c&!a&!d',ConstantRewardFunction(1))
(0,-1,'c|a|d',ConstantRewardFunction(-1))
(1,1,'True',ConstantRewardFunction(0))
(-1,-1,'True',ConstantRewardFunction(0))
# F(b&e) & G!(c|a|d) - Eventually (blue AND square), always avoid (purple OR white OR circle)
```

#### File 2: `videogameworld7.txt` - F(c&d) & G!(b|a|e)
```
0 # initial state
[-1] # terminal state
(0,0,'!c&!d&!b&!a&!e',ConstantRewardFunction(0))
(0,1,'c&d&!b&!a&!e',ConstantRewardFunction(1))
(0,-1,'b|a|e',ConstantRewardFunction(-1))
(1,1,'True',ConstantRewardFunction(0))
(-1,-1,'True',ConstantRewardFunction(0))
# F(c&d) & G!(b|a|e) - Eventually (purple AND circle), always avoid (blue OR white OR square)
```

#### File 3: `videogameworld8.txt` - F(b) & G!(c|a|e)
```
0 # initial state
[-1] # terminal state
(0,0,'!b&!c&!a&!e',ConstantRewardFunction(0))
(0,1,'b&!c&!a&!e',ConstantRewardFunction(1))
(0,-1,'c|a|e',ConstantRewardFunction(-1))
(1,1,'True',ConstantRewardFunction(0))
(-1,-1,'True',ConstantRewardFunction(0))
# F(b) & G!(c|a|e) - Eventually blue, always avoid (purple OR white OR square)
```

#### File 4: `videogameworld9.txt` - G(F(b&e)) & G(F(c&d)) & G!a
```
0 # initial state
[4] # terminal state (trap for white)
(0,0,'!(b&e)&!(c&d)&!a',ConstantRewardFunction(0))
(0,1,'(b&e)&!(c&d)&!a',ConstantRewardFunction(0))
(0,2,'!(b&e)&(c&d)&!a',ConstantRewardFunction(0))
(0,3,'(b&e)&(c&d)&!a',ConstantRewardFunction(1))
(1,0,'!(b&e)&(c&d)&!a',ConstantRewardFunction(1))
(1,1,'!(b&e)&!(c&d)&!a',ConstantRewardFunction(0))
(1,3,'(b&e)&(c&d)&!a',ConstantRewardFunction(1))
(2,0,'(b&e)&!(c&d)&!a',ConstantRewardFunction(1))
(2,2,'!(b&e)&!(c&d)&!a',ConstantRewardFunction(0))
(2,3,'(b&e)&(c&d)&!a',ConstantRewardFunction(1))
(3,1,'(b&e)&!(c&d)&!a',ConstantRewardFunction(0))
(3,2,'!(b&e)&(c&d)&!a',ConstantRewardFunction(0))
(3,0,'!(b&e)&!(c&d)&!a',ConstantRewardFunction(0))
(3,3,'(b&e)&(c&d)&!a',ConstantRewardFunction(1))
(0,4,'a',ConstantRewardFunction(-1))
(1,4,'a',ConstantRewardFunction(-1))
(2,4,'a',ConstantRewardFunction(-1))
(3,4,'a',ConstantRewardFunction(-1))
(4,4,'True',ConstantRewardFunction(0))
# G(F(b&e)) & G(F(c&d)) & G!a - Always eventually (blue&square), always eventually (purple&circle), always avoid white
```

#### File 5: `videogameworld10.txt` - F(G((b|c)&e)) & G!(a|d)
```
0 # initial state
[-1] # terminal state
(0,0,'!((b|c)&e)&!a&!d',ConstantRewardFunction(0))
(0,1,'(b|c)&e&!a&!d',ConstantRewardFunction(1))
(0,-1,'a|d',ConstantRewardFunction(-1))
(1,1,'(b|c)&e&!a&!d',ConstantRewardFunction(1))
(1,0,'!((b|c)&e)&!a&!d',ConstantRewardFunction(0))
(1,-1,'a|d',ConstantRewardFunction(-1))
(-1,-1,'True',ConstantRewardFunction(0))
# F(G((b|c)&e)) & G!(a|d) - Eventually always ((blue OR purple) AND square), always avoid (white OR circle)
```

---

### 2. Environment Classes (Add to boxman_environment.py)

**Location:** `reward_machines/reward_machines/envs/boxman/boxman_environment.py`

Add the following classes after `MyBoxmanEnvRM5`:

```python
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
```

---

### 3. Environment Registration (Modify envs/__init__.py)

**Location:** `reward_machines/reward_machines/envs/__init__.py`

Add after line 33 (after `Boxman-BlueOrPurpleAndSquareTask-v0`):

```python
register(
    id='Boxman-BlueSquareSafe-v0',
    entry_point='envs.boxman.boxman_environment:MyBoxmanEnvRM6',
    max_episode_steps=1000,
)

register(
    id='Boxman-PurpleCircleSafe-v0',
    entry_point='envs.boxman.boxman_environment:MyBoxmanEnvRM7',
    max_episode_steps=1000,
)

register(
    id='Boxman-BlueSafe-v0',
    entry_point='envs.boxman.boxman_environment:MyBoxmanEnvRM8',
    max_episode_steps=1000,
)

register(
    id='Boxman-PatrolBothAvoidWhite-v0',
    entry_point='envs.boxman.boxman_environment:MyBoxmanEnvRM9',
    max_episode_steps=1000,
)

register(
    id='Boxman-StayBlueOrPurpleSquare-v0',
    entry_point='envs.boxman.boxman_environment:MyBoxmanEnvRM10',
    max_episode_steps=1000,
)
```

---

### 4. Training Script (Modify train_comp-LTL_specs.sh)

**Location:** `reward_machines/scripts/train_comp-LTL_specs.sh`

Add the following training commands inside the loop:

```bash
# Spec 1: F(b&e) & G!(c|a|d) - Eventually blue AND square, avoid purple/white/circle
python run.py --env=Boxman-BlueSquareSafe-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/Boxman-BlueSquareSafe-v0/$i/
python run.py --env=Boxman-BlueSquareSafe-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/Boxman-BlueSquareSafe-v0/$i/

# Spec 2: F(c&d) & G!(b|a|e) - Eventually purple AND circle, avoid blue/white/square
python run.py --env=Boxman-PurpleCircleSafe-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/Boxman-PurpleCircleSafe-v0/$i/
python run.py --env=Boxman-PurpleCircleSafe-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/Boxman-PurpleCircleSafe-v0/$i/

# Spec 3: F(b) & G!(c|a|e) - Eventually blue, avoid purple/white/square
python run.py --env=Boxman-BlueSafe-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/Boxman-BlueSafe-v0/$i/
python run.py --env=Boxman-BlueSafe-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/Boxman-BlueSafe-v0/$i/

# Spec 4: G(F(b&e)) & G(F(c&d)) & G!a - Patrol both, avoid white
python run.py --env=Boxman-PatrolBothAvoidWhite-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/Boxman-PatrolBothAvoidWhite-v0/$i/
python run.py --env=Boxman-PatrolBothAvoidWhite-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/Boxman-PatrolBothAvoidWhite-v0/$i/

# Spec 5: F(G((b|c)&e)) & G!(a|d) - Eventually always (blue OR purple) AND square, avoid white/circle
python run.py --env=Boxman-StayBlueOrPurpleSquare-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/Boxman-StayBlueOrPurpleSquare-v0/$i/
python run.py --env=Boxman-StayBlueOrPurpleSquare-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/Boxman-StayBlueOrPurpleSquare-v0/$i/
```

---

### 5. Evaluation/Run Script (Modify run_comp-LTL_specs.sh)

**Location:** `reward_machines/scripts/run_comp-LTL_specs.sh`

Add corresponding run commands for evaluation.

---

## Implementation Order

1. **Create RM specification files** (videogameworld6-10.txt)
2. **Add environment classes** to boxman_environment.py
3. **Register environments** in envs/__init__.py
4. **Update training script** train_comp-LTL_specs.sh
5. **Update run script** run_comp-LTL_specs.sh
6. **Test** - verify each environment loads and runs

---

## Notes

- The symbol mapping follows existing conventions: a=white, b=blue, c=purple, d=circle, e=square
- Safety specifications use -1 as a trap state for violations
- Liveness specifications (GF patterns) require more complex automata with multiple states
- The `ConstantRewardFunction` values: 0 for neutral, 1 for goal achievement, -1 for safety violation
