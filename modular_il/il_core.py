import numpy as np, time, itertools
from collections import OrderedDict
from modular_rl.misc_utils import *
from modular_rl. import distributions
concat = np.concatenate
from IPython import embed


def compute_advantage_il(baseline, expert_vf, paths, gamma, lam):

    for path in paths:
        #expert_vf predict:
        ve_val = expert_vf.predict(path)
        ve_val = np.append(ve_val, 0 if path["terminated"] else ve_val[-1])
        path["reshp_reward"] = path["reward"] + gamma*ve_val[1:] - ve_val[:-1]
        #baseline predict:
        b = path["baseline"] = baseline.predict(path)
        b1 = np.append(b, 0 if path["terminated"] else b[-1])
        deltas = path["reshp_reward"] + gamma*b1[1:] - b1[:-1]
        path["advantage"] = discount(deltas, gamma*lam)
    
    alladv = concat([path["advantage"] for path in paths])
    std = alladv.std()
    mean = alladv.mean()
    for path in paths:
        path["advantage"] = (path["advantage"] - mean) / std




