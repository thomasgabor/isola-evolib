import sys
import os
import time
import dill
import __main__
import explib
from evobasics import *
import matplotlib.pyplot as plt

def lower(avgs, stds):
    return [avgs[x]-stds[x] for x in range(min(len(avgs), len(stds)))]
def upper(avgs, stds):
    return [avgs[x]+stds[x] for x in range(min(len(avgs), len(stds)))]


experiment_dill = sys.argv[1]
exp = dill.load(open(sys.argv[1], 'rb'))
# explib.Experiment.plot(exp, separate_figures=True, separate_subplots=False, no_show=True, save_file=True, plot_std=True, plot_log=False, show_legend=False)

for t,test in enumerate(exp.tests):
    
    if test.name == 'best fitness':
        plotter = plt
        plt.figure(t)
        xs = [0.1 * i for i in range(11)]
        ys = []
        stds =  []
        for e,x in enumerate(xs):
            y =      max([avg(exp.unfolded_results[e][t][gen]) for gen in range(len(exp.unfolded_results[e][t]))])
            std = np.std([avg(exp.unfolded_results[e][t][gen]) for gen in range(len(exp.unfolded_results[e][t]))])
            ys += [y]
            stds += [std]
        params = {'color': 'k'}
        #plot = plotter.semilogy(xs, ys, **params)[0]
        plot = plotter.plot(xs, ys, **params)[0]
        plotter.plot(xs, upper(ys, stds), xs, lower(ys, stds), alpha=0.1, **params)
        plt.xlabel('diversity weight')
        plt.ylabel('best fitness reached')
        plt.show()
        
    if test.name == '|af - fpf|':
        plotter = plt
        plt.figure(t)
        xs = [0.1 * i for i in range(11)]
        ys = []
        stds =  []
        for e,x in enumerate(xs):
            y =      max([avg(exp.unfolded_results[e][t][gen]) for gen in range(len(exp.unfolded_results[e][t]))]) - min([avg(exp.unfolded_results[e][t][gen]) for gen in range(len(exp.unfolded_results[e][t]))])
            std = np.std([avg(exp.unfolded_results[e][t][gen]) for gen in range(len(exp.unfolded_results[e][t]))])
            ys += [y]
            stds += [std]
        params = {'color': 'k'}
        #plot = plotter.semilogy(xs, ys, **params)[0]
        plot = plotter.plot(xs, ys, **params)[0]
        plotter.plot(xs, upper(ys, stds), xs, lower(ys, stds), alpha=0.1, **params)
        plt.xlabel('diversity weight')
        plt.ylabel('max |af - fpf| - min |af - fpf|')
        plt.show()
    
    # stds = [np.std(self.unfolded_results[e][t][gen]) for gen in range(0, len(self.results[e][t]))]
#     print(evolution_args.get('label') + str(' stds'))
#     print(str('min ') + str(min(stds)))
#     print(str('max ') + str(max(stds)))
#     def lower(avgs, stds):
#         return [avgs[x]-stds[x] for x in range(min(len(avgs), len(stds)))]
#     def upper(avgs, stds):
#         return [avgs[x]+stds[x] for x in range(min(len(avgs), len(stds)))]
#     if args.get('plot_log', False):
#         plotter.semilogy(time, upper(self.results[e][t], stds), lower(self.results[e][t], stds), alpha=0.1, **evolution_args)
#     else:
#         plotter.plot(time, upper(self.results[e][t], stds), lower(self.results[e][t], stds), alpha=0.1, **evolution_args)
#
# xs = max([exp.results[0]