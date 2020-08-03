import copy
import numpy as np

def randf(**kwargs):
    dim = ('dim' in kwargs) and kwargs['dim'] or 1
    min_val = ('min' in kwargs) and kwargs['min'] or 0
    max_val = ('max' in kwargs) and kwargs['max'] or 1
    choice = np.random.random_sample()
    return (max_val - min_val) * choice + min_val

def randi(**kwargs):
    dim = ('dim' in kwargs) and kwargs['dim'] or 1
    min_val = ('min' in kwargs) and kwargs['min'] or 0
    max_val = ('max' in kwargs) and kwargs['max'] or 1
    choice = np.random.randint(min_val, max_val)
    return choice
    
def randb(chance=0.5):
    return randf() < chance

def rando(option1, option2, chance=0.5):
    if randb(chance):
        return option1
    else:
        return option2

def randelem(xs):
    return xs[randi(max=len(xs))]
    
def randsel(xs, amount=1):
    return np.random.choice(xs, min(len(xs), amount), replace=False)

def cp(object):
    return copy.deepcopy(object)
    
def compact(some_list):
    new_list = []
    for item in some_list:
        if item:
            new_list.append(item)
    return new_list

def ifthenelse(condition, consequence, antisequence):
    if condition:
        return consequence
    else:
        return antisequence

def avg(numbers):
    return float(sum(numbers)) / float(max(len(numbers), 1))
    
def avgex(numbers):
    total = 0
    count = 0
    for number in numbers:
        if number:
            total += number
            count += 1
    return float(total) / float(max(count, 1))
    
def setor(values):
    for value in values:
        if value:
            return True
    return False
    
def setand(values):
    for value in values:
        if not value:
            return False
    return True
    
def select_by_cutoff(pop, size, **kwargs):
    return pop[0:size]
    
def select_by_graceful_cutoff(pop, size, **kwargs):
    grace_period = kwargs.get('grace_period', 5)
    new_pop = []
    for individual in pop:
        if individual.age >= 1 and individual.age <= grace_period:
            new_pop.append(individual)
    for new_individual in new_pop:
        pop.remove(new_individual) 
    new_pop += pop
    return new_pop[0:size]
    
def select_by_roulette(pop, size, problem_instance=None, **kwargs):
    new_pop = []
    while len(new_pop) < size:
        if len(new_pop) > 1:
            if new_pop[-1] in pop:
                pop.remove(new_pop[-1])
        total_fitness = sum([individual.get_fitness() for individual in pop])
        ball = randf() * total_fitness
        current_fitness  = 0
        for individual in pop:
            current_fitness += individual.get_fitness()
            if ball < current_fitness:
                new_pop.append(individual)
                break
    return new_pop
    
def select_by_walk(pop, size, walk_rate=0.5, **kwargs):
    new_pop = []
    while len(new_pop) < size:
        if len(new_pop) > 1:
            if new_pop[-1] in pop:
                pop.remove(new_pop[-1])
        for individual in pop:
            if randf() < walk_rate:
                new_pop.append(individual)
                break
    return new_pop
                