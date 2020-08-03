from evobasics import *
from evolib import *
from divlib import *

class OptimalIndividual(Individual):
    def initialize_more(self, **kwargs):
        self.distance_function = kwargs.get('distance_function', manhattan)
    def compute_fitness(self, population):
        return self.distance_function(self.genome, self.problem_instance.best())

class FuturisticIndividual(Individual):
    def initialize_more(self, **kwargs):
        self.mut_sample = kwargs.get('mut_sample', 30)
        self.rec_sample = kwargs.get('rec_sample', 70)
    def compute_fitness(self, population):
        children = []
        children += [self.problem_instance.mutate(self.genome) for _ in range(self.mut_sample)]
        children += [self.problem_instance.recombine(self.genome, population.choose_mate(self).genome) for _ in range(self.rec_sample)]
        children_fitnesses = [self.problem_instance.evaluate(child) for child in children]
        l = 0.5
        fitness = -abs(avg(children_fitnesses) - self.problem_instance.evaluate(self.genome))
        #fitness = l * min(children_fitnesses) + (1-l) * avg(children_fitnesses)
        #fitness = min(children_fitnesses + [self.problem_instance.evaluate(self.genome)])
        return fitness
        
class ProductiveIndividual(Individual):
    def initialize_more(self, **kwargs):
        self.mut_sample = kwargs.get('mut_sample', 30)
        self.rec_sample = kwargs.get('rec_sample', 70)
    def compute_fitness(self, population):
        children = []
        children += [self.problem_instance.mutate(self.genome) for _ in range(self.mut_sample)]
        children += [self.problem_instance.recombine(self.genome, population.choose_mate(self).genome) for _ in range(self.rec_sample)]
        children_fitnesses = [self.problem_instance.evaluate(child) for child in children]
        l = 0.5
        fitness = -abs(avg(children_fitnesses) - self.problem_instance.evaluate(self.genome))
        #fitness = l * min(children_fitnesses) + (1-l) * avg(children_fitnesses)
        #fitness = min(children_fitnesses + [self.problem_instance.evaluate(self.genome)])
        return fitness
