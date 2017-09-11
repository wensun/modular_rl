import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.pyplot import quiver
from matplotlib.backends.backend_pdf import PdfPages
import cPickle
from IPython import embed

def plot_mean_std(lstm_avg, lstm_std, color, label):
    ymean = plt.plot(lstm_avg, alpha=1, linewidth = 3, color = color, label = label );
    yplus = plt.plot(lstm_avg + lstm_std, label=None,alpha = 1., color = color, linewidth=0)
    yminus = plt.plot(lstm_avg - lstm_std,label=None,alpha = 1.,color = color, linewidth=0)
    plt.gca().fill_between(np.arange(lstm_avg.size), lstm_avg, lstm_avg+lstm_std, alpha=1./3., color = color)
    plt.gca().fill_between(np.arange(lstm_avg.size), lstm_avg, lstm_avg-lstm_std, alpha=1./3., color = color);


def process_result(stats):
    num_trials = len(stats)
    T = len(stats[0])
    trial_rewards = np.zeros((num_trials, T))
    for i in xrange(num_trials):
        trial_rewards[i] = np.array([stats[i][j][0] for j in range(T)])
    
    mean_run = np.mean(trial_rewards,axis = 0)
    std_run = np.std(trial_rewards, axis = 0)
    return mean_run, std_run




#env = "Swimmer-v1"
env = "MountainCar-v0"
filename = '{}_comparison.pdf'.format(env);
pp = PdfPages(filename);

#TRPO:
#file_trpo = "{}_modular_rl.agentzoo.TrpoAgent".format(env)
#trpo_stats = cPickle.load(open(file_trpo, "rb"))
#trpo_m, trpo_std = process_result(trpo_stats[0])
#plot_mean_std(trpo_m, trpo_std, 'C0', label = "TRPO")

#T = trpo_stats[1]["n_iter"]

#AggreVaTeD
k = [1,10]
agg_stats_lists = [];
for i in xrange(len(k)):
    file_agg = "{}_il_agent.Tr_AggreVaTeD_Agent_1.0_{}".format(env, k[i])
    agg_stats = cPickle.load(open(file_agg, "rb"))
    agg_stats_lists.append(agg_stats)
    agg_m,agg_std = process_result(agg_stats[0])
    plot_mean_std(agg_m, agg_std, color = 'C{}'.format(i+1), label = "K={}".format(k[i]))


fontsize = 10
plt.legend(bbox_to_anchor=(0.00, 0.30, 1., .00), loc=0,
           ncol=1, borderaxespad=0., fontsize = fontsize)
plt.xlabel('Batch Iteration', fontsize = fontsize);
plt.ylabel('Return', fontsize = fontsize);
facecolor = 0.8
plt.gca().set_facecolor((facecolor, facecolor, facecolor))
plt.grid('on',linewidth=2)
plt.ylim(-200,-90);
plt.xlim(0, 200);
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.title(env, fontsize = fontsize);
pp.savefig();
pp.close();