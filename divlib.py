from evobasics import *
from evolib import *

def manhattan(point1, point2):
    distance = 0
    for (value1, value2) in zip(point1, point2):
        distance += abs(value1 - value2)
    return distance

def hamming(point1, point2):
    distance = 0
    for (value1, value2) in zip(point1, point2):
        distance += (value1 != value2) and 1 or 0
    return distance

def normalized(distance_function, point1, point2):
    return distance_function(point1, point2) / float(max(len(point1), len(point2)))

class ManhattanDiversityIndividual(Individual):
    def initialize_more(self, **kwargs):
        self.as_bonus = kwargs.get('diversity_as_bonus', True)
        self.weight = kwargs.get('diversity_weight', 1.0)
        self.sample = kwargs.get('diversity_sample', 5)
        self.max_distance = kwargs.get('max_distance', sum([abs(self.problem().max(d) - self.problem().min(d)) for d in self.problem().dims()]))
    def compute_secondary_fitness(self, population):
        others = [population.choose_random() for _ in range(self.sample)]
        distances = [manhattan(self.genome, other.genome) / float(self.max_distance) for other in others]
        average_distance = sum(distances) / float(self.sample)
        return average_distance
    def compute_fitness(self, population):
        sign = self.problem().maximizing() and 1 or -1
        if self.as_bonus:
            return super().compute_fitness(population) + sign * self.weight * self.compute_secondary_fitness(population)
        else: #as_penalty
            return super().compute_fitness(population) - sign * self.weight * (1 - self.compute_secondary_fitness(population))

class NormalizedManhattanDiversityIndividual(ManhattanDiversityIndividual):
    def initialize_more(self, **kwargs):
        self.weight = kwargs.get('diversity_weight', 0.5)
        self.sample = kwargs.get('diversity_sample', 5)
        self.max_distance = kwargs.get('max_distance', sum([abs(self.problem().max(d) - self.problem().min(d)) for d in self.problem().dims()]))
    def compute_fitness(self, population):
        if self.problem().maximizing():
            return (1 - self.weight) * super().compute_problem_fitness() + self.weight * self.compute_secondary_fitness(population)
        else:
            return (1 - self.weight) * super().compute_problem_fitness() + self.weight * (1 - self.compute_secondary_fitness(population))

class NormalizedDynamicManhattanDiversityIndividual(ManhattanDiversityIndividual):
    def initialize_more(self, **kwargs):
        self.weight = kwargs.get('diversity_weight', 0.5)
        self.sample = kwargs.get('diversity_sample', 5)
    def compute_fitness(self, population):
        others = [population.choose_random() for _ in range(self.sample)]
        distances = [manhattan(self.genome, other.genome) for other in others]
        average_distance = (sum(distances) / float(self.sample)) / max(distances)
        if self.problem().maximizing():
            return (1 - self.weight) * super().compute_problem_fitness() + self.weight * average_distance
        else:
            return (1 - self.weight) * super().compute_problem_fitness() + self.weight * (1 - average_distance)

class HammingDiversityIndividual(Individual):
    def initialize_more(self, **kwargs):
        self.as_bonus = kwargs.get('diversity_as_bonus', True)
        self.weight = kwargs.get('diversity_weight', 1.0)
        self.sample = kwargs.get('diversity_sample', 5)
        #self.max_distance = kwargs.get('max_distance', self.problem_instance.dim())
    def compute_fitness(self, population):
        others = [population.choose_random() for _ in range(self.sample)]
        distances = [normalized(hamming, self.genome, other.genome) for other in others]
        average_distance = sum(distances) / float(self.sample)
        sign = self.problem().maximizing() and 1 or -1
        if self.as_bonus:
            return super().compute_fitness(population) + sign * self.weight * average_distance
        else: #as_penalty
            return super().compute_fitness(population) - sign * self.weight * (1 - average_distance)   
            
class NormalizedHammingDiversityIndividual(Individual):
    def initialize_more(self, **kwargs):
        self.as_bonus = kwargs.get('diversity_as_bonus', True)
        self.weight = kwargs.get('diversity_weight', 1.0)
        self.sample = kwargs.get('diversity_sample', 5)
        #self.max_distance = kwargs.get('max_distance', self.problem_instance.dim())
    def compute_secondary_fitness(self, population):
        others = [population.choose_random() for _ in range(self.sample)]
        distances = [normalized(hamming, self.genome, other.genome) for other in others]
        average_distance = sum(distances) / float(self.sample)
        return average_distance
    def compute_fitness(self, population):
        if self.problem().maximizing():
            return (1 - self.weight) * super().compute_problem_fitness() + self.weight * self.compute_secondary_fitness(population)
        else:
            return (1 - self.weight) * super().compute_problem_fitness() + self.weight * (1 - self.compute_secondary_fitness(population))

class GenealogicalDiversityIndividual(Individual):
    def initialize_more(self, **kwargs):
        self.as_bonus = kwargs.get('diversity_as_bonus', True)
        self.weight = kwargs.get('diversity_weight', 1.0)
        self.sample = kwargs.get('diversity_sample', 5)
        self.tbits = kwargs.get('diversity_tbits', 16)
        if type(self.tbits) == int:
            self.tbits = [randi(min=0, max=2) for _ in range(self.tbits)]
    def compute_fitness(self, population):
        sign = self.problem().maximizing() and 1 or -1
        if self.as_bonus:
            return super().compute_fitness(population) + sign * self.weight * self.compute_secondary_fitness(population)
        else: #as_penalty
            return super().compute_fitness(population) - sign * self.weight * (1 - self.compute_secondary_fitness(population))
    def compute_secondary_fitness(self, population):
        others = [population.choose_random() for _ in range(self.sample)]
        distances = [normalized(hamming, self.tbits, other.tbits) for other in others]
        average_distance = sum(distances) / float(self.sample)
        return average_distance
    def mutate(self):
        child = super().mutate()
        child.tbits = cp(self.tbits)
        d = randi(max=len(child.tbits))
        child.tbits[d] = (child.tbits[d] == 0) and 1 or 0
        return child
    def recombine(self, other):
        child = super().recombine(other)
        child.tbits = [rando(cp(self.tbits[d]), cp(other.tbits[d])) for d in range(len(self.tbits))]
        return child
    def to_str_more(self):
        return ' ' + str(self.tbits)

class NormalizedGenealogicalDiversityIndividual(GenealogicalDiversityIndividual):
    def initialize_more(self, **kwargs):
        self.weight = kwargs.get('diversity_weight', 0.5)
        self.sample = kwargs.get('diversity_sample', 5)
        self.tbits = kwargs.get('diversity_tbits', 16)
        if type(self.tbits) == int:
            self.tbits = [randi(min=0, max=2) for _ in range(self.tbits)]
    def compute_fitness(self, population):
        if self.problem().maximizing():
            return (1 - self.weight) * self.compute_problem_fitness() + self.weight * self.compute_secondary_fitness(population)
        else:
            return (1 - self.weight) * self.compute_problem_fitness() + self.weight * (1 - self.compute_secondary_fitness(population))
    
class TrueGenealogicalDistanceIndividual(Individual):
    def initialize_more(self, **kwargs):
        self.as_bonus = kwargs.get('diversity_as_bonus', True)
        self.weight = kwargs.get('diversity_weight', 1.0)
        self.sample = kwargs.get('diversity_sample', 5)
        self.mut_distance = kwargs.get('single_mut_distance', 2.0/self.problem().dim())
        self.rec_distance = kwargs.get('single_rec_distance', 1.0)
        self.max_distance = kwargs.get('max_distance', 16)
        self.distances = kwargs.get('distances', {})
    def get_distance_to(self, other):
        return min(self.distances.get(other.id, self.max_distance), other.distances.get(self.id, self.max_distance))/float(self.max_distance)
    def update_distances(self, parents, distance):
        for other_id in self.distances.keys():
            self.distances[other_id] += distance
            if self.distances[other_id] >= self.max_distance:
                del self.distances[other_id]
        for parent in parents:
            self.distances[parent.id] = distance
        return self
    def compute_fitness(self, population):
        others = [population.choose_random() for _ in range(self.sample)]
        distances = [self.get_distance_to(other) for other in others]
        average_distance = sum(distances) / float(self.sample)
        sign = self.problem().maximizing() and 1 or -1
        if self.as_bonus:
            return super().compute_fitness(population) + sign * self.weight * average_distance
        else: #as_penalty
            return super().compute_fitness(population) - sign * self.weight * (1 - average_distance)
    def mutate(self):
        child = super().mutate()
        child.update_distances([self], self.mut_distance)
        return child
    def recombine(self, other):
        child = super().recombine(other)
        child.update_distances([self,other], self.rec_distance)
        return child

class InheritedFitnessIndividual(Individual):
    def initialize_more(self, **kwargs):
        self.inheritance_weight = kwargs.get('inheritance_weight', 0.5)
        self.inherited_fitness = 0.0
        self.last_fitness = 0.0
    def mutate(self):
        child = super().mutate()
        child.inherited_fitness = self.last_fitness
        return child
    def recombine(self, other):
        child = super().recombine(other)
        child.inherited_fitness = 0.5 * self.last_fitness + 0.5 * other.last_fitness
        return child
    def to_str_more(self):
        return ' inherited ' + str(self.inherited_fitness)
    def compute_fitness(self, population):
        fitness = (1-self.inheritance_weight) * super().compute_fitness(population) + self.inheritance_weight * self.inherited_fitness
        self.last_fitness = fitness
        return fitness

class FitnessSharingIndividual(Individual):
    def initialize_more(self, **kwargs):
        self.distance_function = kwargs.get('distance_function', 'manhattan')
        self.threshold = kwargs.get('threshold', None)
        if self.distance_function == 'manhattan':
            self.distance_function = manhattan
            self.threshold = kwargs.get('threshold', sum([abs(self.problem().max(d) - self.problem().min(d)) for d in self.problem().dims()]))
        if self.distance_function == 'hamming':
            self.distance_function = hamming
            self.threshold = kwargs.get('threshold', self.problem_instance.dim())
        self.alpha = kwargs.get('alpha', 1.0)
    def sharing_factor_with(self, other):
        d = self.distance_function(self.genome, other.genome)
        if d < self.threshold:
            return 1 - (float(d) / float(self.threshold)) ** self.alpha
        else:
            return 0
    def compute_fitness(self, population):
        return float(super().compute_fitness(population)) / sum([self.sharing_factor_with(individual) for individual in population.individuals])

class EnsemblePopulation(HistoricalPopulation):
    def __init__(self, problem_instance, individuals=None, **kwargs):
        super().__init__(problem_instance, individuals, **kwargs)
        self.subpops = self.args.get('subpops', 2)
        self.subpop_size = int(self.size / self.subpops)
        #self.subpops_map = self.args.get('subpops_map', {individual:i%self.subpops for i,individual in enumerate(self.individuals)})
        self.between_mig_rate = self.args.get('between_mig', 0.2)
        #self.between_rec_rate = self.args.get('between_rec', 0.2)
        for i,individual in enumerate(self.individuals):
            if not hasattr(individual, 'subpop'):
                individual.subpop = i%self.subpops
        
    def split_subpopulations(self):
        subpopulations = [[] for _ in range(self.subpops)]
        for individual in self.individuals:
            subpopulations[individual.subpop].append(individual)
        return subpopulations
            
    def join_subpopulations(self, subpopulations):
        individuals = []
        for subpopulation_index,subpopulation in enumerate(subpopulations):
            for individual in subpopulation:
                individuals.append(individual)
                individual.subpop = subpopulation_index
        #print(len(individuals))
        self.individuals = individuals
    
    def get_subpopulation(self, individual):
        subpopulations = self.split_subpopulations()
        return subpopulations[individual.subpop]
    
    def select(self, size=None):
        size = size or self.size
        subpop_size = int(size / self.subpops)
        self.evaluate()
        subpopulations = self.split_subpopulations()
        for subpopulation_index,subpopulation in enumerate(subpopulations):
            subpopulations[subpopulation_index] = subpopulations[subpopulation_index][0:subpop_size]
        self.join_subpopulations(subpopulations)
        return self

    def choose_mate_from_subpopulation(self, individual):
        subpopulation = self.get_subpopulation(individual)
        return subpopulation[randi(max=int(self.par_rate * len(subpopulation)))]
    
    def invoke_mutation(self, individual):
        child = individual.mutate()
        child.subpop = individual.subpop
        return child

    def invoke_recombination(self, individual):
        child = individual.recombine(self.choose_mate_from_subpopulation(individual))
        child.subpop = individual.subpop
        return child
        
    def augment(self):
        immigrants = [self.generate() for _ in range(int(self.mig_rate * self.size))]
        movers = [randelem(self.individuals).clone() for _ in range(int(self.between_mig_rate * self.size))]
        for individual in immigrants + movers:
            individual.subpop = randi(max=self.subpops)
        self.individuals += immigrants + movers
        return self
        
        
class PFIndividual(Individual):
    def initialize_more(self, **kwargs):
        self.weight = kwargs.get('pf_weight', 0.5)
    def compute_secondary_fitness(self, population):
        return population.compute_pf_here(self) or 0
    def compute_fitness(self, population):
        #return population.compute_pf_here(self) or super().compute_fitness(population)
        if self.problem().maximizing():
            return (1 - self.weight) * super().compute_problem_fitness() + self.weight * self.compute_secondary_fitness(population)
        else:
            return (1 - self.weight) * super().compute_problem_fitness() + self.weight * (1 - self.compute_secondary_fitness(population))
