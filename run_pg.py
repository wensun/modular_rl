#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""

from IPython import embed
from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym

def callback(stats, iters):
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
    np.random.seed(args.seed)
    env.seed(args.seed)

    if cfg["env"] == "Swimmer-v1" or cfg["env"] == "Hopper-v1" or cfg["env"] =="Walker2d-v1":
        cfg['timesteps_per_batch'] = 25000
    elif cfg["env"] == "MountainCar-v0" or cfg["env"] == "Acrobot-v0":
        cfg['timesteps_per_batch'] = 5000

    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    run_stats = run_policy_gradient_algorithm(env, agent, callback=callback, usercfg = cfg)
    return run_stats, cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    
    np.random.seed(0)
    seeds = np.random.randint(0, 2**32, size = 25)
    print "all seeds {}".format(seeds)

    all_trials_stats = []
    for seed in seeds:
        args.seed = seed
        run_stats,cfg = run_experiment(args)
        all_trials_stats.append(run_stats)

    results_file_name = "results/{}_{}".format(cfg["env"], 
        cfg["agent"])
    
    cPickle.dump([all_trials_stats, cfg], open(results_file_name,"wb"))


    