from gym.envs.registration import register

#------------------------------------------ Boxman

register(
    id='Boxman-BlueSquareTask-v0',
    entry_point='envs.boxman.boxman_environment:MyBoxmanEnvRM1',
    max_episode_steps=1000,
)

register(
    id='Boxman-PurpleCircleTask-v0',
    entry_point='envs.boxman.boxman_environment:MyBoxmanEnvRM2',
    max_episode_steps=1000,
)

register(
    id='Boxman-BlueNotSquareTask-v0',
    entry_point='envs.boxman.boxman_environment:MyBoxmanEnvRM3',
    max_episode_steps=1000,
)

register(
    id='Boxman-BlueSquareAndPurpleCircleTask-v0',
    entry_point='envs.boxman.boxman_environment:MyBoxmanEnvRM4',
    max_episode_steps=1000,
)

register(
    id='Boxman-BlueOrPurpleAndSquareTask-v0',
    entry_point='envs.boxman.boxman_environment:MyBoxmanEnvRM5',
    max_episode_steps=1000,
)

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

# ----------------------------------------- Half-Cheetah

register(
    id='Half-Cheetah-RM1-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM1',
    max_episode_steps=1000,
)

register(
    id='Half-Cheetah-RM2-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM2',
    max_episode_steps=1000,
)

# ----------------------------------------- WATER
for i in range(11):
    w_id = 'Water-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

for i in range(11):
    w_id = 'Water-single-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRM10EnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

# ----------------------------------------- OFFICE
register(
    id='OfficeGetCoffeeTask-v0',
    entry_point='envs.grids.grid_environment:OfficeRMEnv',
    max_episode_steps=1000
)

register(
    id='OfficeGoAandDTask-v0',
    entry_point='envs.grids.grid_environment:OfficeRM2Env',
    max_episode_steps=1000
)

register(
    id='OfficeAvoidBAndIn2TasksGetDecorator-v0',
    entry_point='envs.grids.grid_environment:OfficeRM3Env',
    max_episode_steps=1000
)

# ----------------------------------------- CRAFT
for i in range(11):
    w_id = 'Craft-M%d-v0'%i
    w_en = 'envs.grids.grid_environment:CraftRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=1000
    )

for i in range(11):
    w_id = 'Craft-single-M%d-v0'%i
    w_en = 'envs.grids.grid_environment:CraftRM10EnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=1000
    )
