from evobasics import *

class Individual:
    next_id = 0
    def __init__(self, problem_instance, genome=None, children=[], **kwargs):
        self.args = kwargs
        self.problem_instance = problem_instance
        self.genome = genome or problem_instance.generate()
        self.children = []
        self.parents = []
        self.initialize_more(**self.args)
        self.fitness = None
        self.id = Individual.next_id
        self.age = 0
        Individual.next_id += 1
        HistoricalPopulation.historical_individuals[self.id] = self
    
    def problem(self):
        return self.problem_instance
    
    def initialize_more(self, **kwargs):
        pass
    
    def clone(self):
        new_individual = type(self)(self.problem_instance, cp(self.genome), **self.args)
        return new_individual
    
    def mutate(self):
        new_individual = type(self)(self.problem_instance, self.problem_instance.mutate(self.genome), **self.args)
        new_individual.parents = [self.id]
        self.children += [new_individual.id]
        return new_individual
    
    def recombine(self, other):
        new_individual = type(self)(self.problem_instance, self.problem_instance.recombine(self.genome, other.genome), **self.args)
        new_individual.parents = [self.id, other.id]
        self.children += [new_individual.id]
        other.children += [new_individual.id]
        return new_individual
    
    def evaluate(self, population):
        self.fitness = self.compute_fitness(population)
        return self.fitness

    def get_fitness(self):
        return self.fitness

    def compute_fitness(self, population):
        return self.compute_problem_fitness()
        
    def compute_secondary_fitness(self, population):
        return 0

    def compute_problem_fitness(self):
        return self.problem_instance.evaluate(self.genome)
    
    def to_str(self):
        result = ''
        result += '<' + str(self.__class__.__name__) 
        result += ' ' + str(self.genome)
        result += str(self.to_str_more() or '')
        result += ' @ ' + str(self.compute_problem_fitness()) 
        result += ' (' + str(self.get_fitness())
        result += ')>'
        return result
        
    def to_str_more(self):
        return ''
    
    def __str__(self):
        return self.to_str()


class Population:
    def __init__(self, problem_instance, individuals=None, **kwargs):
        self.args = kwargs
        self.problem_instance = problem_instance
        if type(individuals) == list:
            self.individual_class = kwargs.get('individual_class', (len(individuals) > 0) and type(individuals[0]) or Individual)
            self.individuals = individuals
        elif type(individuals) == int:
            self.individual_class = kwargs.get('individual_class', Individual)
            self.individuals = self.generate_all(individuals)
        else:
            self.individuals = []
        self.size = len(self.individuals)
        self.age = 0
        self.mig_rate = self.args.get('mig', 0.1)
        self.mut_rate = self.args.get('mut', 0.1)
        self.rec_rate = self.args.get('rec', 0.3)
        self.par_rate = self.args.get('par', 0.3)
        self.selection = self.args.get('selection', select_by_cutoff)
        self.selection_args = self.args.get('selection_args', {})
        self.selection_args.update(dict(problem_instance=self.problem_instance))
        self.initialize_more(**self.args)

    def initialize_more(self, **kwargs):
        pass

    def generate(self):
        return self.individual_class(self.problem_instance, **self.args)
        
    def generate_all(self, size):
        return [self.generate() for _ in range(size)]

    def best(self):
        return self.individuals[0]

    def evaluate(self):
        current_population = cp(self)
        self.individuals.sort(reverse=self.problem_instance.maximizing(), key=lambda individual: individual.evaluate(current_population))
        return self

    def select(self, size=None):
        size = size or self.size
        self.evaluate()
        self.individuals = self.selection(self.individuals, size, **self.selection_args)
        return self
    
    def choose_mate(self, individual):
        return self.individuals[randi(max=int(self.par_rate * self.size))]
        
    def choose_random(self):
        return self.individuals[randi(max=self.size)]

    def invoke_mutation(self, individual):
        return individual.mutate()

    def mutate(self):
        # self.individuals = [randb(self.mut_rate) and self.invoke_mutation(individual) or individual for individual in self.individuals]
        self.individuals += [self.invoke_mutation(individual) for individual in self.individuals if randb(self.mut_rate)]
        return self
    
    def invoke_recombination(self, individual):
        return individual.recombine(self.choose_mate(individual))
    
    def recombine(self):
        self.evaluate()
        self.individuals += [self.invoke_recombination(individual) for individual in self.individuals if randb(self.rec_rate)]
        return self
        
    def augment(self):
        self.individuals += [self.generate() for _ in range(int(self.mig_rate * self.size))]
        return self
        
    def evolve(self):
        new_population = type(self)(self.problem_instance, self.individuals, **self.args)
        new_population.age += 1
        new_population.problem_instance.update(self.age)
        for individual in new_population.individuals:
            individual.age += 1
        new_population.recombine()        
        new_population.mutate()        
        new_population.augment()        
        new_population.select()
        return new_population



class HistoricalPopulation(Population):
    historical_avg_fitnesses = {}
    historical_opt_fitnesses = {}
    historical_individuals = {}
    historical_ancestors = {}
    def initialize_more(self, **kwargs):
        pass
    def compute_pf(self, individual, depth=0):
        if individual.id in HistoricalPopulation.historical_avg_fitnesses:
            return HistoricalPopulation.historical_avg_fitnesses[individual.id]
        if len(individual.children) == 0:
            return individual.compute_problem_fitness()
        else:
            children_pf = []
            for child_id in individual.children:
                children_pf += [self.compute_pf(HistoricalPopulation.historical_individuals[child_id], depth+1)]
            avg_productive_fitness = avg(children_pf)
            HistoricalPopulation.historical_avg_fitnesses[individual.id] = avg_productive_fitness
            return avg_productive_fitness
    def compute_opf(self, individual, depth=0):
        if individual.id in HistoricalPopulation.historical_opt_fitnesses:
            return HistoricalPopulation.historical_opt_fitnesses[individual.id]
        if len(individual.children) == 0:
            return individual.compute_problem_fitness()
        else:
            children_pf = []
            for child_id in individual.children:
                children_pf += [self.compute_opf(HistoricalPopulation.historical_individuals[child_id], depth+1)]
            opt_productive_fitness = self.problem_instance.maximizing() and max(children_pf) or min(children_pf)
            HistoricalPopulation.historical_opt_fitnesses[individual.id] = opt_productive_fitness
            return opt_productive_fitness
    def compute_pf_here(self, ancestor):
        descendant_count = 0
        descendant_total_fitness = 0
        for individual in self.individuals:
            if self.is_ancestor(ancestor, individual):
                descendant_count += 1
                descendant_total_fitness += individual.compute_problem_fitness()
        if descendant_count > 0:
            return float(descendant_total_fitness) / float(descendant_count)
        else:
            return None
    def compute_opf_here(self, ancestor):
        descendant_count = 0
        descendant_best_fitness = self.problem_instance.maximizing() and -9999999 or 9999999
        for individual in self.individuals:
            if self.is_ancestor(ancestor, individual):
                descendant_fitness = individual.compute_problem_fitness()
                if self.problem_instance.maximizing():
                    if descendant_fitness > descendant_best_fitness:
                        descendant_best_fitness = descendant_fitness
                else:
                    if descendant_fitness < descendant_best_fitness:
                        descendant_best_fitness = descendant_fitness
                descendant_count += 1
        if descendant_count > 0:
            return float(descendant_best_fitness)
        else:
            return None
    def is_ancestor(self, ancestor, descendant):
        if str(descendant.id) + " " + str(ancestor.id) in HistoricalPopulation.historical_ancestors:
            return HistoricalPopulation.historical_ancestors[str(descendant.id) + " " + str(ancestor.id)]
        if descendant.id in ancestor.children:
            return True
        else:
            for child_id in ancestor.children:
                if self.is_ancestor(HistoricalPopulation.historical_individuals[child_id], descendant):
                    HistoricalPopulation.historical_ancestors[str(descendant.id) + " " + str(ancestor.id)] = True
                    return True
                HistoricalPopulation.historical_ancestors[str(descendant.id) + " " + str(ancestor.id)] = False
            return False
    def filter_descendants(self, ancestor):
        return [individual for individual in self.individuals if self.is_ancestor(ancestor, individual) ]
    def best_pf(self):
        best_pf = self.problem_instance.maximizing() and -9999999 or 9999999
        pfs = []
        best = None
        for individual in self.individuals:
            pf = self.compute_pf(individual)
            pfs += [pf]
            if (self.problem_instance.maximizing() and pf > best_pf) or (not self.problem_instance.maximizing() and pf < best_pf):
                best_pf = pf
                best = individual
        return best_pf
    def best_opf(self):
        best_pf = self.problem_instance.maximizing() and -9999999 or 9999999
        pfs = []
        best = None
        for individual in self.individuals:
            pf = self.compute_opf(individual)
            pfs += [pf]
            if self.problem_instance.maximizing() and pf > best_pf or not self.problem_instance.maximizing() and pf < best_pf:
                best_pf = pf
                best = individual
        return best_pf
    def best_f(self):
        if self.problem_instance.maximizing():
            return max([individual.compute_problem_fitness() for individual in self.individuals])
        else:
            return min([individual.compute_problem_fitness() for individual in self.individuals])

class TestPopulation(Population):
    def initialize_more(self, **kwargs):
        self.subjects = self.args.get('subjects', [])
    def select(self, size=None):
        size = size or self.size
        self.evaluate()
        self.individuals = self.individuals[0:(size-len(self.subjects))]
        self.individuals += self.subjects
        return self
