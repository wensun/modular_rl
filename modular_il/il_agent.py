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


class TrpoAgent_il(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + TrpoUpdater.options + FILTER_OPTIONS
    def __init__(self, ob_space, ac_space, expert_vf, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = TrpoUpdater(policy, cfg)
        self.expert_vf = expert_vf
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)

    






