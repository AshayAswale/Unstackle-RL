from gymnasium.envs.registration import register

register(
    id="Stackle/GridWorld-v0",
    entry_point="Stackle.envs:GridWorldEnv",
)
