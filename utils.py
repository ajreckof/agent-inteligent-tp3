import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_sumrwdperepi(sum_rewards):
    "trace courbe de somme des rec par episodes"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(sum_rewards)), sum_rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    
    
def plot_sumrwd_mean_perepi(sum_rewards,avgs, file_name = ""):
    "trace courbe de somme des rec et moyenne glissante par episodes"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.grid(visible= True, which='both')
    plt.plot(np.arange(len(sum_rewards)), sum_rewards, label='sum_rwd')
    plt.plot(np.arange(len(avgs)), avgs, c='r', label='average')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc='upper left');
    if file_name :
        plt.savefig(f'plot/{file_name}.pdf')
    