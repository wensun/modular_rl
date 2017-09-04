import numpy as np, time, itertools
from collections import OrderedDict
from modular_rl.misc_utils import *
from modular_rl import *
from modular_rl.agentzoo import *
concat = np.concatenate
from IPython import embed


#lam = 0 ==> Regular AggreVaTeD (simply uses A^e(s,a))
#otherwise the planning horizon ~ 1/(1-lam)log(...)
def compute_advantage_il(baseline, expert_vf, paths, cfg):
    for path in paths:
        #expert_vf predict:
        path["return"] = discount(path["reward"], cfg["gamma"])
        ve_val = expert_vf.predict(path) #it supposes to use the raw observation
        ve_val = np.append(ve_val, 0 if path["terminated"] else ve_val[-1])
        path["reshp_reward"] = path["reward"] + cfg["gamma"]*ve_val[1:] - ve_val[:-1]
        path["reshp_return"] = discount(path["reshp_reward"],cfg["gamma"])
        #baseline predict:
        b = path["baseline"] = baseline.predict(path) #it supposes to use filtered observations.
        b1 = np.append(b, 0 if path["terminated"] else b[-1])
        deltas = path["reshp_reward"] + cfg["gamma"]*b1[1:] - b1[:-1]
        if cfg['truncate_k'] < 0 and cfg['lam'] >= 0:
            path["advantage"] = discount(deltas, cfg["gamma"]*cfg["lam"])
        elif cfg['truncate_k'] > 0 and cfg['lam'] < 0:
            path["advantage"] = sum_over_k_steps(deltas, cfg['truncate_k'], cfg['gamma_mat'])
        
    alladv = concat([path["advantage"] for path in paths])
    std = alladv.std()
    mean = alladv.mean()
    for path in paths:
        path["advantage"] = (path["advantage"] - mean) / std


def AggreVaTeD(env, agent, usercfg=None, callback=None):
    cfg = update_default_config(PG_OPTIONS, usercfg)
    cfg.update(usercfg)
    print "AggreVaTeD Config: {}".format(cfg)

    tstart = time.time()
    seed_iter = itertools.count()

    for _ in xrange(cfg["n_iter"]):
        #roll-out with current policy
        paths = get_paths(env, agent, cfg, seed_iter)
        compute_advantage_il(agent.baseline, agent.expert_vf, paths, cfg)
        #VF baseline update
        vf_stats = agent.baseline.fit_with_reshp_rew(paths)
        #Policy update
        pol_stats = agent.updater(paths)
        #stats
        stats = OrderedDict()
        add_episode_stats(stats,paths)
        add_prefixed_stats(stats, "vf", vf_stats)
        add_prefixed_stats(stats, "pol", pol_stats)
        stats["TimeEllapsed"] = time.time() - tstart
        if callback: callback(stats)
    

def compute_advantage_il_2(vf, paths, expert_vf, cfg, curr_iter):
    for path in paths:
        path["return"] = discount(path["reward"], gamma)
        if curr_iter == 0:
            b = path["baseline"] = expert_vf.predict(path)
        else:
            b = path["baseline"] = vf.predict(path)
        b1 = np.append(b, 0 if path["terminated"] else b[-1])
        #compute advantage:
        path["advantage"] = path["reward"] + cfg["gamma"]*b1[1:] - b1[:-1]
    
    alladv = np.concatenate([path["advantage"] for path in paths])
    std = alladv.std()
    mean = alladv.mean()
    for path in paths:
        path["advantage"] = (path["advantage"] - mean)/std
    

def iter_AggreVaTeD(env, agent, usercfg, callback=None):
    cfg = update_default_config(PG_OPTIONS, usercfg)
    cfg.update(usercfg)
    print "Iter-AggreVaTeD Config: {}".format(cfg)

    tstart = time.time()
    seed_iter = itertools.count()

    curr_batch_iter = 0
    for i in xrange(1,cfg["n_iter"]):
        #roll-out with current policy
        paths = get_paths(env, agent, cfg, seed_iter)
        compute_advantage_il_2(agent.baseline, paths, agent.expert_vf, cfg, 
                curr_batch_iter)

        #every m_iteration, we update the baseline. 
        if np.mod(i, cfg["every_m"]) == 0:
            vf_stats = agent.baseline.fit(paths)
            curr_batch_iter += 1
            add_prefixed_stats(stats,"vf", vf_stats)
        
        pol_stats = agent.updater(paths)

        stats = OrderedDict()
        add_episode_stats(stats,paths)
        add_prefixed_stats(stats,"pol", pol_stats)
        stats["TimeElapsed"] = time.time()-tstart
        if callback:callback(stats)
     
        
