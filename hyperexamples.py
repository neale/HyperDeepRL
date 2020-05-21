######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import itertools
import pprint

def product_dict(kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def sweep(game, tag, model_fn, trials=50, manual=True):
    hyperparams = {
        'alpha_i': [1, 10, 100],
        'alpha_f': [.1, 0.01],
        'anneal': [500e3],
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
            'alpha_i': 10,
            'alpha_f': 0.1,
            'anneal': 500e3,
            'lr': 1e-4,
            'freq': 100,
            'grad_clip': None,
            'hidden': 256,
            'replay_size': int(1e5),
            'replay_bs': 128,
            'dist': 'softmax'
        }
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
    config.task_fn = lambda: Task(config.game, video=False, gif=False, log_dir=config.tf_log_handle)
    config.eval_env = config.task_fn()
    config.particles = 24

    config.optimizer_fn = lambda params: torch.optim.Adam(params, config.lr)
    config.network_fn = lambda: DuelingHyperNet(config.action_dim,
                                    CartFCHyperBody(config.state_dim, hidden=config.hidden),
                                hidden=config.hidden, dist=config.dist, particles=config.particles)
    config.replay_fn = lambda: Replay(memory_size=config.replay_size, batch_size=config.replay_bs)
    # config.replay_fn = lambda: AsyncReplay(memory_size=config.replay_size, batch_size=config.replay_bs)
    config.render = True  # Render environment at every train step
    config.random_action_prob = LinearSchedule(0, 0, 1e4)#1e-1, 1e-7, 1e4)  # eps greedy params
    config.max_random_action_prob = LinearSchedule(1e-1, 1e-7, 1e4)#1e-1, 1e-7, 1e4)  # eps greedy params
    config.aux_noise_prob = LinearSchedule(1e-1, 1e-7, 1e4)#1e-1, 1e-7, 1e4)  # eps greedy params
    #config.log_random_action_prob = 0.05
    config.discount = 0.99  # horizon
    config.target_network_update_freq = config.freq  # hard update to target network
    config.exploration_steps = 0#config.replay_bs  # random actions taken at the beginning to fill the replay buffer
    config.double_q = True  # use double q update
    config.sgd_update_frequency = 1  # how often to do learning
    config.gradient_clip = config.grad_clip  # max gradient norm
    config.eval_interval = int(5e3) 
    config.max_steps = 500e3
    config.async_actor = False
    config.alpha_anneal = config.anneal  # how long to anneal SVGD alpha from init to final
    config.alpha_init = config.alpha_i  # SVGD alpha strating value
    config.alpha_final = config.alpha_f  # SVGD alpha end value
    config.svgd_q = 'sample'
    config.update = 'sgd'
    config.max_rand = True
    config.aux_rand = False

    #run_steps(DQN_Param_SVGD_Agent(config))
    if config.update == 'sgd':
        run_steps(DQN_SGD_Agent(config))
    elif config.update == 'thompson':
        run_steps(DQN_Thompson_Agent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)

    tag = 'verify_rand_randexp/trial1'
    game = 'bsuite-cartpole_swingup/0'
    sweep(game, tag, dqn_feature, manual=True, trials=50)

