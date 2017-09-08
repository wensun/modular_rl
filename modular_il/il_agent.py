"""
*in this code base, the "Agent" is a container with the policy, 
baseline (value function), and a pre-trained expert's value function (V^e)

* all the options (MLPs, Filters, PG) are directly from modular_rl.agentzoo
"""
from gym.spaces import Box, Discrete
from collections import OrderedDict
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from learn_vf_from_demo import *
from IPython import embed
from modular_rl import *
from modular_rl.agentzoo import *


class Tr_AggreVaTeD_Agent(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + TrpoUpdater.options + FILTER_OPTIONS
    def __init__(self, ob_space, ac_space, expert_vf, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = TrpoUpdater(policy, cfg)
        self.expert_vf = expert_vf
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)
        #embed()

class iter_Tr_AggreVaTeD_Agent(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + TrpoUpdater.options + FILTER_OPTIONS
    def __init__(self, ob_space, ac_space, expert_vf, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = TrpoUpdater(policy, cfg)
        self.expert_vf = expert_vf
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)
        #build a freeze net
        net2 = Sequential()
        for i in xrange(len(self.baseline.reg.net.layers)-1):
            input_shape = self.baseline.reg.net.layers[i].input_shape
            output_shape = self.baseline.reg.net.layers[i].output_shape
            inshp = dict(input_shape = (input_shape[1], )) if i == 0 else {}
            net2.add(Dense(output_shape[1],activation=cfg["activation"],**inshp))
        net2.add(Dense(1))
        #do not mix with the old preded_y, as here we are explicitly build a freezen network.
        self.baseline_f = NnVf(net2, cfg["timestep_limit"], dict(mixfrac=1.)) 

    def update_baseline_f(self):
        net = self.baseline.reg.net
        net_f = self.baseline_f.reg.net
        assert len(net.layers) == len(net_f.layers)
        for i in xrange(len(net.layers)):
            w_i = net.layers[i].get_weights()
            net_f.layers[i].set_weights(w_i)
        






