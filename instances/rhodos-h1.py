import sys
sys.path += ['../']
from evolib import *
from explib import *
from divlib import *
from optlib import *
from problib import *

#this is for GECCO compatibility of the PDFs
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


from scipy.stats import pearsonr

# Pearson = Test("pearson per generation").with_function(lambda pop: pearsonr([pop.compute_pf(individual) for individual in pop.individuals], [individual.evaluate(pop) for individual in pop.individuals])[0]).with_live(False)


worst = 1.0

tests = []

#tests += [Test("virtual best fitness")]
tests += [Test("best fitness").with_function(lambda pop: pop.best_f()).with_live(True)]
tests += [Test("secondary fitness").with_function(lambda pop: pop.best().compute_secondary_fitness(pop)).with_live(True)]
tests += [Test("augmented fitness").with_function(lambda pop: pop.best().evaluate(pop)).with_live(True)]
tests += [Test("pf").with_function(lambda pop: pop.best_pf()).with_live(False)]
# tests += [Test("opt pf").with_function(lambda pop: pop.best_opf()).with_live(False)]

f_increase  = lambda pop, context: pop.best_f()  - context['all_populations'][context['current_gen']-1].best_f()
pf_increase = lambda pop, context: pop.best_pf() - context['all_populations'][context['current_gen']-1].best_pf()
# tests += [ContextTest(" f increase").with_function(f_increase).with_live(True)]
# tests += [ContextTest("pf increase").with_function(pf_increase).with_live(True)]
# tests+= [ContextTest("increase diff").with_function(lambda pop, context: f_increase(pop, context) - pf_increase(pop, context)).with_live(True)]

f_full   = lambda pop, context: (context['all_populations'][0].best_f()  - pop.best_f())  / \
                                (context['all_populations'][0].best_f()  - context['all_populations'][-1].best_f())
pf_full  = lambda pop, context: (context['all_populations'][0].best_pf() - pop.best_pf())  / \
                                (context['all_populations'][0].best_pf() - context['all_populations'][-1].best_pf())
# tests += [ContextTest(" f progress").with_function(f_full).with_live(False)]
# tests += [ContextTest("pf progress").with_function(pf_full).with_live(False)]
# tests += [ContextTest("progress diff").with_function(lambda pop, context: f_full(pop, context) - pf_full(pop, context)).with_live(False)]


final_descendants = lambda pop, con: avg([len(con["all_populations"][-1].filter_descendants(individual)) for individual in pop.individuals])
tests += [ContextTest("descendants survive").with_function(final_descendants).with_live(False)]

def get_pop(con, offset=0):
    return con["all_populations"][max(0, min(con["current_gen"]+offset, len(con["all_populations"])-1))]

pf_for = lambda individual, con, x: con["all_populations"][min(con["current_gen"]+x, len(con["all_populations"])-1)].compute_pf_here(individual) or worst

onepf = lambda pop, con: avg([pf_for(individual, con, 1) for individual in pop.individuals])
tenpf = lambda pop, con: avg([pf_for(individual, con, 10) for individual in pop.individuals])
tests += [ContextTest(" 1-pf").with_function(onepf).with_live(False)]
tests += [ContextTest("10-pf").with_function(tenpf).with_live(False)]


fpf_for = lambda individual, con: con["all_populations"][-1].compute_pf_here(individual) or worst

fpf = lambda pop, con: avgex([fpf_for(individual, con) for individual in pop.individuals])
fpf_norm = lambda pop, con: fpf(pop, con) / con["all_populations"][-1].best_f()
fpf_incr = lambda pop, con: fpf(pop, con) - fpf(con["all_populations"][max(con['current_gen']-1, 0)], con)
tests += [ContextTest("fpf").with_function(fpf).with_live(False)]
# tests += [ContextTest("fpf norm").with_function(fpf_norm).with_live(False)]
# tests += [ContextTest("fpf incr").with_function(fpf_incr).with_live(False)]

ffpf  = lambda pop, con: avg([abs(individual.compute_problem_fitness() - fpf_for(individual, con)) for individual in pop.individuals])
affpf = lambda pop, con: avg([abs(individual.get_fitness()             - fpf_for(individual, con)) for individual in pop.individuals])

tests += [ContextTest(" |f - fpf|").with_function(ffpf).with_live(False)]
tests += [ContextTest("|af - fpf|").with_function(affpf).with_live(False)]
# tests += [ContextTest("xxx").with_function(lambda pop, con: abs(affpf(pop, con) - ffpf(pop, con))).with_live(False)]
# tests += [ContextTest("potential gain").with_function(lambda pop, con: abs(affpf(pop, con) - ffpf(pop, con))).with_live(False)]
tests += [ContextTest("delta |af - fpf|").with_function(lambda pop, con: affpf(get_pop(con, -1), con) - affpf(pop, con)).with_live(False)]



problem_size = 2
exp = Experiment(InverseNormalizedH1Problem(),
    population_class=HistoricalPopulation,
    pop_size=25,
    num_gens=50,
    num_runs=500,
    diversity_sample=5,
    save_file=True,
    save_all_populations=True)
for test in tests:
    exp.add_test(test).with_style(ls="-")

exp.historical_individuals = HistoricalPopulation.historical_individuals
exp.historical_ancestors = HistoricalPopulation.historical_ancestors


# exp.perform(individual_class=NormalizedManhattanDiversityIndividual,
#     diversity_weight=0.5,
#     # selection=select_by_graceful_cutoff,
#     # selection_args={},
#     style=dict(color='b', ls='-'),
#     label='manhattan diverse')
# exp.perform(individual_class=NormalizedGenealogicalDiversityIndividual,
#     tbits=32,
#     diversity_weight=0.15,
#     style=dict(color='r', ls='-'),
#     label='genealogical diverse')
# exp.perform(individual_class=OptimalIndividual,
#     style=dict(color='gray'),
#     label='optimal baseline')
# exp.perform(individual_class=PFIndividual,
#     selection=select_by_graceful_cutoff,
#     selection_args={},
#     pf_weight=0.25,
#     style=dict(color='g'),
#     label='pf-based')
exp.perform(individual_class=Individual,
    # selection=select_by_graceful_cutoff,
    # selection_args={},
    style=dict(color='k'),
    label='non-diverse')
exp.save_dill()
exp.save_all_results()
# exp.print_verdict()
# exp.plot(show_legend=False)
# exp.print_examples()

# print(HistoricalPopulation.historical_individuals)