from modular_rl import *
from modular_rl import do_rollouts_serial
from gym.envs import make
import argparse, sys, cPickle
import shutil, os, logging
import gym
import itertools
from learn_vf_from_demo import *
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from modular_rl import NnVf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    env = make(args.env)
    env_spec = env.spec
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)
    args.video = False
    env = gym.wrappers.Monitor(env, mondir, video_callable=None if args.video else VIDEO_NEVER)
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit
    cfg = args.__dict__
    np.random.seed(args.seed)
    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)

    #np.random.seed(100)
    if cfg["env"] == "Swimmer-v1" or cfg["env"] == "Hopper-v1" or cfg["env"] == "Walker2d-v1":
        cfg["timesteps_per_batch"] = 25000


    model_name = "expert_models/Perfect_ExpertAgent_{}_{}".format(cfg['env'], cfg['agent'])
    loaded_model = cPickle.load(open(model_name, 'rb'))

    #get demonstration by rolling out:
    paths = do_rollouts_serial(env, loaded_model, cfg['timestep_limit'],
        cfg["timesteps_per_batch"], itertools.count(), raw_obs = True) #raw data, no filter is used.
    
    #initailize a keras neural network:
    v_func = Sequential()
    v_func.add(Dense(64, activation='tanh', 
                    input_shape = (paths[0]['observation'].shape[1]+1,),
                    kernel_regularizer=regularizers.l2(1e-5)))

    v_func.add(Dense(1))
    v_func.compile(optimizer = 'adam', loss = 'mse')

    scaler = fit_value_function(v_func, cfg['gamma'], paths = paths);
    scaler.predict(np.random.rand(11),2)
    

    
    

    
