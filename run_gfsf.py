######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import itertools
import pprint
import envs

def product_dict(kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def sweep(game, tag, model_fn, trials=50, manual=True, chain_len=4):
    hyperparams = {
        'lr': [2e-4, 1e-4],
        'freq' : [10, 100],
        'grad_clip': [None, 5],
        'hidden': [256, 128, 512],
        'replay_size': [int(1e5)],
        'replay_bs': [128, 64],
        'sgd_freq': [1, 4, 10],
        'dist': ['softmax', 'multinomial', 'normal', 'uniform']
        # 'dist': ['multinomial']
    }
    # manually define
    if manual:
        print ('=========================================================')
        print ('Running Manually Defined Single Trial, [1/1]')
        setting = {
            'game': game,
            'tb_tag': tag,
            'lr': 1e-3,
            'freq': 5,
            'grad_clip': None,
            'hidden': 50,
            'replay_size': int(1e5),
            'replay_bs': 128,
            'dist': 'uniform'
        }
        setting['chain_len'] = chain_len
        print ('Running Config: ')
        for (k, v) in setting.items():
            print ('{} : {}'.format(k, v))
        model_fn(**setting)
        return

    search_space = list(product_dict(hyperparams))
    ordering = list(range(len(search_space)))
    np.random.shuffle(ordering)
    for i, idx in enumerate(ordering):
        setting = search_space[idx]
        setting['game'] = game
        setting['chain_len'] = chain_len
        tag_append='_ai{}-af{}-lr{}-f{}-gc{}-h{}-{}-bs{}'.format(
                setting['alpha_i'],
                setting['alpha_f'],
                setting['lr'],
                setting['freq'],
                setting['grad_clip'],
                setting['hidden'],
                setting['dist'],
                setting['replay_bs'])

        setting['tb_tag'] = tag+tag_append
        print ('=========================================================')
        print ('Search Space Contains {} Trials, Running [{}/{}] ---- ({}%)'.format(
            len(search_space), i+1, trials, int(float(i+1)/trials*100.)))
        print ('Running Config: ')
        for (k, v) in setting.items():
            print ('{} : {}'.format(k, v))
        dqn_feature(**setting)
    
def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True
    config.tag = config.tb_tag
    config.generate_log_handles()
    config.particles = 24
    config.task_fn = lambda: Task(config.game,
            video=False,
            gif=False,
            log_dir=config.tf_log_handle,
            special_args=('NChain', config.chain_len))

    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, config.lr)
    
    config.network_fn = lambda: DuelingHyperHead(
            config.action_dim,
            FCBody(
                config.state_dim,
                hidden_units=(config.hidden, config.hidden)),
            hidden=config.hidden,
            dist=config.dist,
            particles=config.particles,
            critic_hidden=1)

    config.prior_fn = lambda: DuelingNet(
            config.action_dim,
            FCBody(
                config.state_dim,
                hidden_units=(config.hidden, config.hidden)))

    config.replay_fn = lambda: Replay(
            memory_size=config.replay_size,
            batch_size=config.replay_bs)
    
    config.prior_scale = 1.
    config.render = True  # Render environment at every train step
    config.random_action_prob = LinearSchedule(0, 0, 1e4)#1e-1, 1e-7, 1e4)  # eps greedy params
    config.discount = 0.99  # horizon
    config.target_network_update_freq = config.freq  # hard update to target network
    config.exploration_steps = config.replay_bs  # random actions taken at the beginning to fill the replay buffer
    config.double_q = True  # use double q update
    config.sgd_update_frequency = 1  # how often to do learning
    config.gradient_clip = config.grad_clip  # max gradient norm
    config.eval_interval = int(5e7) 
    config.max_steps = 500 * (config.chain_len+9)
    config.async_actor = False
    config.update = 'thompson'
    config.use_pushforward = False

    if config.update == 'sgd':
        if config.prior_scale > 0.:
            run_steps(DQN_GFSF_Prior_Agent(config))
        else:
            run_steps(DQN_GFSF_Agent(config))
    elif config.update == 'thompson':
        run_steps(DQN_GFSF_Prior_Thompson_Agent(config))
    else:
        raise ValueError

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(0)

    tag = 'test_new'
    game = 'NChain-v3'
    for i in range(14, 50, 2):
        tag = 'gfsf-prior-chain/checkout1_{}'.format(i)
        sweep(game, tag, dqn_feature, manual=True, trials=50, chain_len=i)

