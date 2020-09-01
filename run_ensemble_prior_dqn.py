from deep_rl import *
import envs

def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.tag = config.tb_tag
    config.generate_log_handles()
    config.task_fn = lambda: Task(
            config.game,
            video=False,
            gif=False,
            log_dir=config.tf_log_handle,
            special_args=('NChain', config.chain_len))

    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.Adam(params, 1e-3)
    config.network_fn = lambda: DuelingNet(
            config.action_dim,
            FCBody(
                config.state_dim,
                hidden_units=(256, 256)))

    config.replay_fn = lambda: Replay(memory_size=int(1e5), batch_size=100)

    config.random_action_prob = LinearSchedule(0.0, 0.0, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 4
    config.exploration_steps = 128
    config.ensemble_len = 10
    config.reselect_steps = 100
    config.replay_mask_prob = 0.5
    config.double_q = True
    config.prior_scale = 3.
    config.sgd_update_frequency = 1
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.max_steps = 2000 * (config.chain_len+9)
    config.async_actor = False
    run_steps(DQNRPAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(0)

    tag = 'test_new'
    game = 'NChain-v3'
    for i in range(4, 50, 2):
        tag = 'ensemble-prior-chain/checkout1_{}'.format(i)
        dqn_feature(game=game, tb_tag=tag, chain_len=i)
