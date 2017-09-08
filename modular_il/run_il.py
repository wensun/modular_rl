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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    #parser.add_argument("--truncate_k", required=True)
    #parser.add_argument("--lam", required=True)
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
    
    if cfg["env"] == "Swimmer-v1" or cfg["env"] == "Hopper-v1" or cfg["env"] =="Walker2d-v1":
        cfg['timesteps_per_batch'] = 25000;

    cfg["truncate_k"] = -1
    cfg["lam"] = 0.9

    filename = "expert_models/Est_{}_Vstar".format(cfg["env"])
    expertvf = cPickle.load(open(filename, "rb"))
    agent = agent_ctor(env.observation_space, env.action_space, expertvf, cfg)
    

    COUNTER = 0
    def callback(stats):
        global COUNTER
        COUNTER += 1
        # Print stats
        print "*********** Iteration %i ****************" % COUNTER
        print tabulate(filter(lambda (k,v) : np.asarray(v).size==1, stats.items())) #pylint: disable=W0110
        # Store to hdf5
        if args.use_hdf:
            for (stat,val) in stats.items():
                if np.asarray(val).ndim==0:
                    diagnostics[stat].append(val)
                else:
                    assert val.ndim == 1
                    diagnostics[stat].extend(val)
            if args.snapshot_every and ((COUNTER % args.snapshot_every==0) or (COUNTER==args.n_iter)):
                hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))
        # Plot
        if args.plot:
            animate_rollout(env, agent, min(500, args.timestep_limit))



    AggreVaTeD(env, agent, usercfg = cfg, callback=callback)
        
    #if args.use_hdf:
    #    hdf, diagnostics = prepare_h5_file(args)
    #gym.logger.setLevel(logging.WARN)



    #cfg: a dictionary stores all the parameters with their names. 

