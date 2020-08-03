import sys
import os
import time
import dill
import math
import __main__
import explib
import problib

from evobasics import *
from evolib import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# experiment_dill = sys.argv[1]
# exp = dill.load(open(sys.argv[1], 'rb'))
# HistoricalPopulation.historical_individuals  = exp.historical_individuals
# HistoricalPopulation.historical_ancestors  = exp.historical_ancestors
exp = explib.Experiment(problib.NormalizedSchwefelProblem(2))

def measure_problem_fitness(experiment, evolution, run, gen, individual):
    return experiment.problem_instance.evaluate(individual.genome)
    
def measure_final_productive_fitness(experiment, evolution, run, gen, individual):
    return experiment.all_populations[evolution][run][-1].compute_pf_here(individual) or 1.0

def get_max_dist(problem_instance):
    max_dist = (problem_instance.min(0) - problem_instance.max(0)) ** 2 + (problem_instance.min(1) - problem_instance.max(1)) ** 2
    return math.sqrt(max_dist)

def measure_quadratic_error(experiment, evolution, run, gen, individual):
    return (individual.genome[0] - experiment.problem_instance.best()[0]) ** 2 + (individual.genome[1] - experiment.problem_instance.best()[1]) ** 2
    
def measure_distance_to_global_optimum(experiment, evolution, run, gen, individual):
    # print(individual.genome[0], experiment.problem_instance.best()[0])
    # print(measure_quadratic_error(experiment, evolution, run, gen, individual))
    return math.sqrt(measure_quadratic_error(experiment, evolution, run, gen, individual)) / get_max_dist(experiment.problem_instance)

measurements = dict(abs=measure_distance_to_global_optimum, fpf=measure_final_productive_fitness, f=measure_problem_fitness)

def hplot(self, **kwargs):
    args = cp(self.global_args)
    args.update(kwargs)
    evolution = 0
    fig = plt.figure()
    ax = Axes3D(fig, azim = 130, elev = 20)
    ax.set_xlim3d(self.problem_instance.min(0), self.problem_instance.max(0))
    ax.set_ylim3d(self.problem_instance.min(1), self.problem_instance.max(1))
    ax.set_zlim3d(0, 1.0/args.get('zoom', 1.0))
    if args.get('plot_base_fitness', False):
        X = np.arange(self.problem_instance.min(0), self.problem_instance.max(0), 10)
        Y = np.arange(self.problem_instance.min(1), self.problem_instance.max(1), 10)
        X, Y = np.meshgrid(X, Y)
        def base_fitness(sol):
            return self.problem_instance.evaluate(sol)
        Z = np.fromiter(map(base_fitness, zip(X.flat,Y.flat)), dtype=np.float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
        # ax.scatter(X, Y, Z, c=(0,0,0,0.1), marker='o')

            # approx_resolution = args.get('approx_resolution', abs(self.problem_instance.min(0) - self.problem_instance.max(0)) / 10.0)
#             approx_surface = {}
#             for run in range(0, len(self.all_populations[evolution])):
#                 XS = []
#                 YS = []
#                 ZS = []
#                 for individual in self.all_populations[evolution][run][gen].individuals:
#                     m = measure(self, evolution, run, gen, individual)
#                     XS.append(individual.genome[0])
#                     YS.append(individual.genome[1])
#                     ZS.append(m)
#                     # print(individual.genome[0], individual.genome[1], m)
#                     approx_x = int(individual.genome[0]/approx_resolution)
#                     approx_y = int(individual.genome[1]/approx_resolution)
#                     if not approx_x in approx_surface:
#                         approx_surface[approx_x] = {}
#                     if not approx_y in approx_surface[approx_x]:
#                         approx_surface[approx_x][approx_y] = []
#                     if m < 1.0 or True:
#                         approx_surface[approx_x][approx_y].append(m)
#                 # ax.scatter(XS, YS, ZS, c=(1.0,0,0,0.1), marker='o')
#
#             if args.get('approx_surface', False):
#                 if False:
#                     for x in approx_surface.keys():
#                         for y in approx_surface[x].keys():
#                             if len(approx_surface[x][y]) > 0:
#                                 # print(x*approx_resolution, y*approx_resolution, avg(approx_surface[x][y]))
#                                 ax.scatter([x*approx_resolution], [y*approx_resolution], avg(approx_surface[x][y]), c=(0,0,1.0,0.1), marker='o')
#                 X = np.arange(self.problem_instance.min(0)+1, self.problem_instance.max(0), 10)
#                 Y = np.arange(self.problem_instance.min(1)+1, self.problem_instance.max(1), 10)
#                 X, Y = np.meshgrid(X, Y)
#                 def surface_fitness(sol):
#                     approx_x = int(sol[0]/approx_resolution)
#                     approx_y = int(sol[1]/approx_resolution)
#                     if not approx_x in approx_surface:
#                         return 1.0
#                     if not approx_y in approx_surface[approx_x]:
#                         return 1.0
#                     around = 0.0
#
#                     if approx_x in approx_surface and approx_y - 1 in approx_surface[approx_x]:
#                         around += avg(approx_surface[approx_x][approx_y-1])
#                     if approx_x in approx_surface and approx_y + 1 in approx_surface[approx_x]:
#                         around += avg(approx_surface[approx_x][approx_y+1])
#                     if approx_x - 1 in approx_surface and approx_y  in approx_surface[approx_x-1]:
#                         around += avg(approx_surface[approx_x-1][approx_y])
#                     if approx_x + 1 in approx_surface and approx_y  in approx_surface[approx_x+1]:
#                         around += avg(approx_surface[approx_x+1][approx_y])
#
#
#                     if args.get('smooth_diagonal', False):
#                         if approx_x - 1 in approx_surface and approx_y - 1 in approx_surface[approx_x-1]:
#                             around += avg(approx_surface[approx_x-1][approx_y-1])
#                         if approx_x - 1 in approx_surface and approx_y + 1 in approx_surface[approx_x-1]:
#                             around += avg(approx_surface[approx_x-1][approx_y+1])
#                         if approx_x + 1 in approx_surface and approx_y - 1 in approx_surface[approx_x+1]:
#                             around += avg(approx_surface[approx_x+1][approx_y-1])
#                         if approx_x + 1 in approx_surface and approx_y + 1 in approx_surface[approx_x+1]:
#                             around += avg(approx_surface[approx_x+1][approx_y+1])
#
#                     if args.get('smooth_diagonal', False):
#                         around /= 8.0
#                     else:
#                         around /= 4.0
#
#                     return 0.6 * avg(approx_surface[approx_x][approx_y]) + 0.4 * around
#                 Z = np.fromiter(map(surface_fitness, zip(X.flat,Y.flat)), dtype=np.float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
#                 ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1)
    
                
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('base fitness')

    plt.savefig('hplot.png')
    if not args.get('no_show', False):
        plt.show()



explib.Experiment.hplot = hplot
explib.Experiment.hplot(exp, plot_base_fitness=True, no_show=True, zoom=1, approx_surface=True, approx_resolution=10)