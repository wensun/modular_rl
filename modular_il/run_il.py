#!/usr/bin/env python
"""
This script runs a policy gradient algorithm for IL
"""

from IPython import embed
from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym
from il_core import AggreVaTeD
#from modular_il import *
#from modular_il import Expert_Vf
#from run_pg import callback

def callback(stats, iters):
        #global COUNTER
        #COUNTER += 1
        # Print stats
    print "*********** Iteration %i ****************" % iters
    print tabulate(filter(lambda (k,v) : np.asarray(v).size==1, stats.items())) #pylint: disable=W0110
        

def run_experiment(args):
    tmp_seed = args.seed
    env = make(args.env)
    

    env_spec = env.spec
    args.video = 0
    agent_ctor = get_agent_cls(args.agent)
    
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    args.seed = tmp_seed
    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit
    cfg = args.__dict__
    cfg["lam"] = float(cfg["lam"])
    cfg["truncate_k"] = int(cfg["truncate_k"])

    if cfg["truncate_k"] > 0:
        cfg["gamma_mat"] = construct_gamma_mat(cfg["gamma"], cfg["timestep_limit"], cfg["truncate_k"])

    np.random.seed(args.seed)

    if cfg["env"] == "Swimmer-v1" or cfg["env"] == "Hopper-v1" or cfg["env"] =="Walker2d-v1":
        cfg['timesteps_per_batch'] = 25000

    #cfg["truncate_k"] = -1
    #cfg["n_iter"] = 100

    filename = "expert_models/Est_{}_Vstar_995".format(cfg["env"])
    expertvf = cPickle.load(open(filename, "rb"))
    agent = agent_ctor(env.observation_space, env.action_space, expertvf, cfg)
    run_stats = AggreVaTeD(env, agent, usercfg = cfg, callback=callback)
    return run_stats,cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--truncate_k", required=True)
    parser.add_argument("--lam", required=True)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])

    all_trials_stats = []
    for trial in range(0,15):
        args.seed = trial*10
        run_stats,cfg = run_experiment(args)
        all_trials_stats.append(run_stats)

    results_file_name = "results/{}_{}_{}_{}".format(cfg["env"], 
        cfg["agent"], cfg["lam"], cfg["truncate_k"])
    
    cPickle.dump([all_trials_stats, cfg], open(results_file_name,"wb"))

