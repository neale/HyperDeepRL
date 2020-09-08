######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import itertools
import pprint
import envs
import copy

def product_dict(kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def sweep(game, tag, model_fn, trials=50, manual=True, chain_len=4):
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
            'alpha_f': 0.0,
            'lr': 1e-3,
            'freq': 10,
            'grad_clip': None,
            'hidden': 256,
            'replay_size': int(1e5),
            'replay_bs': 128,
            'dist': 'softmax'
        }
        print ('Running Config: ')
        setting['chain_len'] = chain_len
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

        setting['chain_len'] = chain_len
        setting['tb_tag'] = tag+tag_append
        print ('=========================================================')
        print ('Search Space Contains {} Trials, Running [{}/{}] ---- ({}%)'.format(
            len(search_space), i+1, trials, int(float(i+1)/trials*100.)))
        print ('Running Config: ')
        for (k, v) in setting.items():
            print ('{} : {}'.format(k, v))
        dqn_feature(**setting)

class Trainer(object):
    def __init__(
            self,
            game,
            chain_low,
            chain_high,
            use_init_network=True,):

        self.settings = {
            'game': game,
            'use_init_network': use_init_network,
            'alpha_init': 10.,
            'alpha_final': 0.,
            'particles': 24,
            'prior_scale': 0.,
            'lr': 1e-4,
            'target_network_update_freq': 5,
            'gradient_clip' : None,
            'hidden' : 64,
            'batch_size': 128,
            'replay_size': int(1e5),
            'discount': 0.99,
            'dist': 'uniform',
            'use_pushforward': True,
            'sgd_update_frequency': 1,}

        self.game = game
        self._use_init_network = use_init_network
        self._chain_low = chain_low
        self._chain_high = chain_high
        self._network = None
        self._target_network = None

    def run(self):
        print ('=========================================================')
        print ('Running Manually Defined Single Trial, [1/1]')
        print ('Config: ')
        print (self.__dict__)
        for i in range(self._chain_low, self._chain_high, 2):
            self.settings['chain_len'] = i
            self.settings['alpha_anneal'] = 500 * (i + 9)
            self.settings['max_steps'] = 500 * (i + 9)
            self.settings['tb_tag'] = '-{}'.format(i)
            self._dqn_runner()
    
    def _set_init_networks(self, net, target_net):
        self._network = net
        self._target_network = target_net
    
    def _get_init_networks(self):
        return copy.deepcopy(self._network), copy.deepcopy(self._target_network)
    
    def _print_init_weight_norms(self):
        for name, p in self._network.named_parameters():
            print ('Param Group', name)
            print ('mean: ', p.mean().item())
            print ('var: ', p.var().item())
            print ('norm: ', p.norm().item())
            print ()

    def _dqn_runner(self):
        generate_tag(self.settings)
        self.settings.setdefault('log_level', 0)
        config = Config()
        config.merge(self.settings)
        config.tag = config.tb_tag
        config.generate_log_handles()
        config.task_fn = lambda: Task(
            config.game,
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
                batch_size=config.batch_size)
        
        config.render = True  # Render environment at every train step
        config.random_action_prob = LinearSchedule(0., 0., 1e4)#1e-1, 1e-7, 1e4)  # eps greedy params
        config.exploration_steps = config.batch_size  # random actions taken at the beginning to fill the replay buffer
        config.double_q = True  # use double q update
        config.eval_interval = int(5e7) 
        config.async_actor = False
        config.update = 'sgd'
        
        if self._use_init_network:
            if self._network is None:
                net = config.network_fn()
                target_net = config.network_fn()
                target_net.load_state_dict(net.state_dict())
                self._set_init_networks(net, target_net)
                self._print_init_weight_norms()
            
            network, target_network = self._get_init_networks()
            config._init_network = network
            config._init_target_network = target_network
        
        if config.update == 'sgd':
            run_steps(DQN_SVGD_Agent(config))
        elif config.update == 'thompson':
            run_steps(DQN_SVGD_Thompson_Agent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)
    game = 'NChain-v3'
    trainer = Trainer(game, chain_low=50, chain_high=100, use_init_network=False)
    trainer.run()

