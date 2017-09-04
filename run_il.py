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
from modular_il import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--truncate_k", required=True)
    parser.add_argument("--lam", required=True)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    env = make(args.env)
    env_spec = env.spec
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)
    args.video = 0
    env = gym.wrappers.Monitor(env, mondir, video_callable=None if args.video else VIDEO_NEVER)
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit
    cfg = args.__dict__
    np.random.seed(args.seed)
    
    tmp_v_e = Expert_Vf()
    agent = agent_ctor(env.observation_space, env.action_space, tmp_v_e, cfg)
    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)

    #cfg: a dictionary stores all the parameters with their names. 
    embed()
