from evobasics import *
from evolib import *
import time
import dill
import matplotlib.pyplot as plt

#this is for GECCO compatibility of the PDFs
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


from tqdm import tqdm
from scipy.stats import pearsonr

import __main__ as main
import os 

RESULT_FILES_DIR = './results/'

class Test:
    def __init__(self, name=None, test=None):
        self.name = name or "best fitness"
        self.test = test or (lambda pop: pop.best().compute_problem_fitness())
        self.style = {'color': 'k'}
        self.live = True
    def with_name(self, name):
        self.name = name
        return self
    def with_style(self, **kwargs):
        self.style = kwargs
        return self
    def with_function(self, test):
        self.test = test
        return self
    def with_live(self, live):
        self.live = live
        return self
    def is_live(self):
        return self.live
    def has_context(self):
        return False
    def run(self, pop):
        return self.test(pop)
    def __call__(self, pop):
        return self.run(pop)
        
class ContextTest(Test):
    def __init__(self, name=None, test=None):
        super().__init__(name, test or (lambda pop, _: pop.best().compute_problem_fitness()))
    def has_context(self):
        return True
    def run(self, pop, context):
        return self.test(pop, context)
    def __call__(self, pop, context):
        return self.run(pop, context)

class AbstractExperiment:
    def __init__(self, problem_instance, **kwargs):
        self.problem_instance = problem_instance
        self.global_args = kwargs
        self.real_time = time.time()
        self.tests = []
        self.initialize_more()
    def initialize_more(self):
        pass
    def experiment_filename(self):
        import __main__ as main
        import os
        return RESULT_FILES_DIR + os.path.splitext(os.path.basename(main.__file__))[0] + '.' + str(self.real_time)
    def add_test(self, test=None):
        new_test = type(test) == Test and test or Test(None, test)
        self.tests.append(new_test)
        return new_test
    def save_dill(self, **kwargs):
        with open(self.experiment_filename() + '.pickle', 'wb') as pickle_file:
            dill.dump(self, pickle_file)

class ParameterExperiment(AbstractExperiment):
    def initialize_more(self):
        self.settings = []
        self.evolutions = []
        self.results = []
        self.plotted = 0
    def for_parameter(self, parameter, settings):
        self.parameter = parameter
        self.settings = settings
    def perform(self, **kwargs):
        args = cp(self.global_args)
        args.update(kwargs)
        show_progress = args.get('show_progress', True)
        args['show_progress'] = False
        settings_iterator = range(len(self.settings))
        if show_progress:
            print('Evaluating ' + str(len(self.settings)) + ' settings of ' + str(self.parameter) + ' for ' + args.get('label', 'unlabeled setup') + ' with ' + str(args.get('num_runs', 'an unknown amount of')) + ' runs each')
            settings_iterator = tqdm(settings_iterator)
        current_results = []
        for s in settings_iterator:
            setting = self.settings[s]
            args[self.parameter] = setting
            base_experiment = Experiment(self.problem_instance, **args)
            for t,test in enumerate(self.tests):
                base_experiment.add_test(test)
            base_experiment.perform(**args)
            current_results.append([base_experiment.results[-1][t][-1] for t,_ in enumerate(self.tests)])
        self.evolutions.append(cp(kwargs))
        self.results.append(cp(current_results))
    def plot(self, **kwargs):
        args = cp(self.global_args)
        args.update(kwargs)
        figuring = args.get('separate_figures', False) and len(self.tests) > 1
        subplotting = args.get('separate_subplots', False) and len(self.tests) > 1
        if subplotting:
            f, axarr = plt.subplots(len(self.tests), sharex=True)
        else:
            axarr = None
        for t,test in enumerate(self.tests):
            style_args = cp(test.style)
            style_args.update(self.global_args.get('style', {}))
            style_args.update(kwargs.get('style', {}))
            if figuring:
                plt.figure(t)
            if subplotting:
                plotter = axarr[t]
            else:
                plotter = plt
            plots = []
            data = {}
            for e,evolution in enumerate(self.evolutions):
                evolution_args = cp(style_args)
                evolution_args.update({'label': evolution.get('label', 'unkown setup')})
                evolution_args.update(evolution.get('style', {}))
                if args.get('xlim', False):
                    plotter.xlim(args.get('xlim'))
                if args.get('ylim', False):
                    plotter.ylim(args.get('ylim'))
                if args.get('plot_log', False):
                    plot = plotter.semilogy(self.settings, [self.results[e][s][t] for s,setting in enumerate(self.settings)], **evolution_args)[0]
                else:
                    plot = plotter.plot(self.settings, [self.results[e][s][t] for s,setting in enumerate(self.settings)], **evolution_args)[0]
                plots.append(plot)
            if subplotting:
                plotter.set(xlabel=str(style_args.get('xlabel', self.parameter)), ylabel=str(style_args.get('ylabel', hasattr(test, 'name') and test.name or 'result')))
            else:
                plt.ylabel(str(style_args.get('ylabel', hasattr(test, 'name') and test.name or 'result')))
                plt.xlabel(str(style_args.get('xlabel', self.parameter)))
            plotter.legend(handles=plots)
            if figuring and args.get('save_file', False):
                plt.savefig(self.experiment_filename() + '.figure' + str(t) + '.pplot' + str(self.plotted) + '.png')
                plt.savefig(self.experiment_filename() + '.figure' + str(t) + '.pplot' + str(self.plotted) + '.pdf')
        if (not figuring) and args.get('save_file', False):
            plt.savefig(self.experiment_filename() + '.pplot' + str(self.plotted) + '.png')
            plt.savefig(self.experiment_filename() + '.pplot' + str(self.plotted) + '.pdf')
            with open(self.experiment_filename() + '.pplot' + str(self.plotted) + '.txt', 'w') as text_file:
                print(str(data), file=text_file)
        if not args.get('no_show', False):
            plt.show()
        self.plotted += 1

            

class Experiment:
    def __init__(self, problem_instance, **kwargs):
        self.problem_instance = problem_instance
        self.global_args = kwargs
        self.tests = []
        self.evolutions = []
        self.final_populations = []
        self.all_populations = []
        self.results = []
        self.unfolded_results = []
        self.styles = []
        self.real_time = time.time()
        self.plotted = 0
        self.filename = RESULT_FILES_DIR + os.path.splitext(os.path.basename(main.__file__))[0] + '.' + str(self.real_time)
    def change_problem(self, new_problem):
        self.problem_instance = new_problem
    def experiment_filename(self):
        return self.filename
    def add_test(self, test=None):
        new_test = isinstance(test, Test) and test or Test(None, test)
        self.tests.append(new_test)
        return new_test
    def perform(self, **kwargs):
        args = cp(self.global_args)
        args.update(kwargs)
        num_runs = args.get('num_runs', 100)
        num_gens = args.get('num_gens', 100)
        pop_size = args.get('pop_size', 100)
        show_progress = args.get('show_progress', True)
        population_class  = args.get('population_class', Population)
        save_final_populations = args.get('save_final_populations', True)
        save_all_populations = args.get('save_populations', True)
        folding = kwargs.get('folding', lambda runs: sum(runs)/float(len(runs)))
        result = [[[None for _ in range(0, num_runs)] for _ in range(0, num_gens)] for _ in self.tests]
        final_populations = []
        all_populations = []
        run_iterator = range(0, num_runs)
        if show_progress:
            print('Performing ' + str(num_runs) + ' runs of ' + args.get('label', 'unlabeled setup'))
            run_iterator = tqdm(run_iterator)
        for run in run_iterator:
            pop = population_class(cp(self.problem_instance), pop_size, **kwargs)
            pops = []
            for gen in range(0, num_gens):
                pop = pop.evolve()
                pops += [pop]
                for t,test in enumerate(self.tests):
                    if test.is_live():
                        if test.has_context():
                            result[t][gen][run] = test(pop, dict(current_run=run, current_gen=gen, all_populations=pops))
                        else:
                            result[t][gen][run] = test(pop)
            for gen in range(0, num_gens):
                for t,test in enumerate(self.tests):
                    if not test.is_live():
                        if test.has_context():
                            result[t][gen][run] = test(pops[gen], dict(current_run=run, current_gen=gen, all_populations=pops))
                        else:
                            result[t][gen][run] = test(pops[gen])
            if save_final_populations:
                final_populations.append(pop)
            if save_all_populations:
                all_populations.append(pops)
        folded_result = [[folding(result[t][gen]) for gen in range(0, num_gens)] for t,_ in enumerate(self.tests)]
        self.evolutions.append(cp(kwargs))
        self.results.append(cp(folded_result))
        self.unfolded_results.append(cp(result))
        self.final_populations.append(cp(final_populations))
        self.all_populations.append(cp(all_populations))
        return len(self.evolutions) - 1
    def plot(self, **kwargs):
        args = cp(self.global_args)
        args.update(kwargs)
        figuring = args.get('separate_figures', False) and len(self.tests) > 1
        subplotting = args.get('separate_subplots', False) and len(self.tests) > 1
        correlate_with = args.get('correlate_with', False)
        if subplotting:
            f, axarr = plt.subplots(len(self.tests), sharex=True)
        else:
            axarr = None
        for t,test in enumerate(self.tests):
            if not t in args.get('select_tests', [t]):
                continue
            style_args = cp(test.style)
            style_args.update(self.global_args.get('style', {}))
            style_args.update(kwargs.get('style', {}))
            if figuring:
                plt.figure(t)
            if subplotting:
                plotter = axarr[t]
            else:
                plotter = plt
            plots = []
            data = {}
            for e,evolution in enumerate(self.evolutions):
                evolution_args = cp(style_args)
                evolution_args.update({'label': evolution.get('label', 'unkown setup')})
                if correlate_with:
                    evolution_args.update({'label': str(evolution.get('label', 'unkown setup')) + ' corr. ' + str(pearsonr(self.results[e][t], self.results[e][self.tests.index(correlate_with)])[0])})
                evolution_args.update(evolution.get('style', {}))
                time = [gen for gen in range(0, len(self.results[e][t]))]
                if args.get('xlim', False):
                    plotter.xlim(args.get('xlim'))
                if args.get('ylim', False):
                    plotter.ylim(args.get('ylim'))
                if args.get('plot_log', False):
                    plot = plotter.semilogy(time, self.results[e][t], **evolution_args)[0]
                else:
                    plot = plotter.plot(time, self.results[e][t], **evolution_args)[0]
                if args.get('plot_std', False):
                    stds = [np.std(self.unfolded_results[e][t][gen]) for gen in range(0, len(self.results[e][t]))]
                    print(evolution_args.get('label') + str(' stds'))
                    print(str('min ') + str(min(stds)))
                    print(str('max ') + str(max(stds)))
                    def lower(avgs, stds):
                        return [avgs[x]-stds[x] for x in range(min(len(avgs), len(stds)))]
                    def upper(avgs, stds):
                        return [avgs[x]+stds[x] for x in range(min(len(avgs), len(stds)))]
                    if args.get('plot_log', False):
                        plotter.semilogy(time, upper(self.results[e][t], stds), lower(self.results[e][t], stds), alpha=0.1, **evolution_args)
                    else:
                        plotter.plot(time, upper(self.results[e][t], stds), lower(self.results[e][t], stds), alpha=0.1, **evolution_args)
                    
                plots.append(plot)
                data[evolution_args.get('label')] = self.results[e][t]
            if args.get('plot_baseline', False) and self.problem_instance.best_result() != None:
                baseline_args = cp(style_args)
                baseline_args.update({'label': kwargs.get('baseline_label', 'best possible result')})
                baseline_args.update(kwargs.get('baseline_style', {}))
                plot = plotter.axhline(self.problem_instance.best_result(), **baseline_args)
                plots.append(plot)
            if subplotting:
                plotter.set(xlabel=str(style_args.get('xlabel', 'time')), ylabel=str(style_args.get('ylabel', hasattr(test, 'name') and test.name or 'result')))
            else:
                plt.ylabel(str(style_args.get('ylabel', hasattr(test, 'name') and test.name or 'result')))
                plt.xlabel(str(style_args.get('xlabel', 'time')))
            if args.get('show_legend', True):
                plotter.legend(handles=plots)
            if figuring and args.get('save_file', False):
                plt.savefig(self.experiment_filename() + '.figure' + str(t) + '.plot' + str(self.plotted) + '.png')
                plt.savefig(self.experiment_filename() + '.figure' + str(t) + '.plot' + str(self.plotted) + '.pdf')
        if (not figuring) and args.get('save_file', False):
            plt.savefig(self.experiment_filename() + '.plot' + str(self.plotted) + '.png')
            plt.savefig(self.experiment_filename() + '.plot' + str(self.plotted) + '.pdf')
            with open(self.experiment_filename() + '.plot' + str(self.plotted) + '.txt', 'w') as text_file:
                print(str(data), file=text_file)
        if not args.get('no_show', False):
            plt.show()
        self.plotted += 1
    def print_verdict(self, **kwargs):
        args = cp(self.global_args)
        args.update(kwargs)
        description = ''
        description += 'problem:                 ' + str(self.problem_instance.__class__.__name__) + '\n'
        description += 'problem parameters:      ' + str(self.problem_instance.args) + '\n'
        description += 'global settings:         ' + str(self.global_args) + '\n'
        for t,test in enumerate(self.tests):
            description += 'test for ' + str(hasattr(test, 'name') and test.name or 'result') + '\n'
            for e,evolution in enumerate(self.evolutions):
                # 1 = non diverse, 0 = diverse
                description += '    ' + str(evolution.get('label', 'unkown setup')) + ' evolution\n'
                description += '        settings:        ' + str(evolution) + '\n'
                description += '        final result:    ' + str(self.results[e][t][-1]) + '\n'
                if len(self.populations) > e and len(self.populations[e]) == 1:
                    description += '        best individual: ' + str(self.populations[e][0].best()) + '\n'
        if args.get('save_file', False):
            with open(self.experiment_filename() + '.verdict.txt', 'w') as text_file:
                print(description, file=text_file)
        if not args.get('no_show', False):
            print(description)
    def print_examples(self, **kwargs):
        args = cp(self.global_args)
        args.update(kwargs)
        amount = args.get('amount', 5)
        description = ''
        if not len(self.populations) > 0:
            description += 'No populations were saved. No sampling possible.\n'
        else:
            description += 'Examples\n'
            for e,evolution in enumerate(self.evolutions):
                description += '    ' + str(evolution.get('label', 'unkown setup')) + ' evolution\n'
                description += '        settings:                \t' + str(evolution) + '\n'
                for run,population in enumerate(randsel(self.populations[e], amount)):
                    description += '        best individual from run ' + str(run) + ':\t' + str(population.best()) + '\n'
                    description += '        ... with problem fitness:  \t' + str(population.best().compute_problem_fitness()) + '\n'
        if args.get('save_file', False):
            with open(self.experiment_filename() + '.examples.txt',  'w') as text_file:
                print(description, file=text_file)
        if not args.get('no_show', False):
            print(description)
    def save_all_results(self, **kwargs):
        args = cp(self.global_args)
        args.update(kwargs)
        with open(self.experiment_filename() + '.results.py', 'w') as text_file:
            for t,test in enumerate(self.tests):
                print('# test      ' + str(t) + ': ' + str(test.name))
            for e,evolution in enumerate(self.evolutions):
                print('# evolution ' + str(e) + ': ' + str(evolution.get('label', 'unkown setup')))
            print('results = ' + str(self.results), file=text_file)
            print('unfolded_results = ' + str(self.unfolded_results), file=text_file)
    def save_dill(self, **kwargs):
        with open(self.experiment_filename() + '.pickle', 'wb') as pickle_file:
            dill.dump(self, pickle_file)
    
    
     