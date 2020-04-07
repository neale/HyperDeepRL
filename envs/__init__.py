from gym.envs.registration import register
register(
        id='NChain-v3',
        entry_point='envs.nchain:NChainEnv',
)
