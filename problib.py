from evobasics import *
from deap import benchmarks
# fromdeap.benchmarks import movingpeaks
import math

class Problem:
    #abstract members: self.objective_maximizing
    def maximizing(self):
        return self.objective_maximizing
    def minimizing(self):
        return not self.objective_maximizing
    def best(self):
        return None
    def best_result(self):
        return None
    def update(self, time):
        return self

class Environment:
    def __init__(self):
        self.time = 0
    def update(self, time=1):
        self.time += time


class VectorProblem(Problem):
    #abstract members: self.dimensionality
    def dim(self):
        return self.dimensionality
    def dims(self):
        return range(0, self.dim())

class FloatsProblem(VectorProblem):
    #abstract methods: min, max
    def generate(self):
        return [randf(min=self.min(d), max=self.max(d)) for d in self.dims()]
    def mutate(self, candidate, **kwargs):
        candidate = cp(candidate)
        if hasattr(self, 'mutation_continuous') and self.mutation_continuous:
            strength = hasattr(self, 'mutation_strength') and self.mutation_strength or kwargs.get('strength', 0.1)
            variation = [randf(min=strength*self.min(d), max=strength*self.max(d)) for d in self.dims()]
            candidate = [np.clip(variation + candidate[d], self.min(d), self.max(d)) for d in self.dims()]
        else:
            d = randi(max=self.dim())
            strength = hasattr(self, 'mutation_strength') and self.mutation_strength or kwargs.get('strength', 0.1 * abs(self.min(d) - self.max(d)))
            candidate[d] += randf(min=-strength, max=strength)
            candidate[d] = np.clip(candidate[d], self.min(d), self.max(d))
        return candidate
    def recombine(self, candidate1, candidate2):
        if hasattr(self, 'recombine_average') and self.recombine_average:
            pass #TODO
        else:
            return [rando(cp(candidate1[d]), cp(candidate2[d])) for d in self.dims()]

class BohachevskyProblem(FloatsProblem):
    def __init__(self, dimensionality):
        self.args = {'dimensionality': dimensionality}
        self.dimensionality = dimensionality
        self.objective_maximizing = False
    def min(self, d):
        return -100
    def max(self, d):
        return 100
    def best(self):
        return [0 for _ in range(self.dimensionality)]
    def best_result(self):
        return 0
    def evaluate(self, candidate):
        return benchmarks.bohachevsky(candidate)[0]
        
class NormalizedBohachevskyProblem(BohachevskyProblem):
    def evaluate(self, candidate):
        return benchmarks.bohachevsky(candidate)[0] / 50000.0

class H1Problem(FloatsProblem):
    def __init__(self):
        self.args = {}
        self.dimensionality = 2
        self.objective_maximizing = True
    def min(self, d):
        return -100
    def max(self, d):
        return 100
    def best(self):
        return [8.6998, 6.7665]
    def best_result(self):
        return 2
    def evaluate(self, candidate):
        return benchmarks.h1(candidate)[0]

class NormalizedH1Problem(H1Problem):
    def evaluate(self, candidate):
        return benchmarks.h1(candidate)[0] / 2.0
        
class InverseNormalizedH1Problem(H1Problem):
    def __init__(self):
        self.args = {}
        self.dimensionality = 2
        self.objective_maximizing = False
    def evaluate(self, candidate):
        return 1.0 - benchmarks.h1(candidate)[0] / 2.0
        
class SchafferProblem(FloatsProblem):
    def __init__(self, dimensionality):
        self.args = {'dimensionality': dimensionality}
        self.dimensionality = dimensionality
        self.objective_maximizing = False
    def min(self, d):
        return -100
    def max(self, d):
        return 100
    def best(self):
        return [0.0 for _ in self.dims()]
    def best_result(self):
        return 0
    def evaluate(self, candidate):
        return benchmarks.schaffer(candidate)[0]

class NormalizedSchafferProblem(SchafferProblem):
    def evaluate(self, candidate):
        return benchmarks.schaffer(candidate)[0] / 120.0

class RastriginProblem(FloatsProblem):
    def __init__(self, dimensionality):
        self.args = {'dimensionality': dimensionality}
        self.dimensionality = dimensionality
        self.objective_maximizing = False
    def min(self, d):
        return -5.12
    def max(self, d):
        return 5.12
    def best(self):
        return [0.0 for _ in self.dims()]
    def best_result(self):
        return 0
    def evaluate(self, candidate):
        return benchmarks.rastrigin(candidate)[0]

class NormalizedRastriginProblem(RastriginProblem):
    def evaluate(self, candidate):
        return benchmarks.rastrigin(candidate)[0] / 200.0
        
class RosenbrockProblem(FloatsProblem):
    def __init__(self, dimensionality):
        self.args = {'dimensionality': dimensionality}
        self.dimensionality = dimensionality
        self.objective_maximizing = False
    def min(self, d):
        return -100
    def max(self, d):
        return 100
    def best(self):
        return [1.0 for _ in self.dims()]
    def best_result(self):
        return 0
    def evaluate(self, candidate):
        return benchmarks.rosenbrock(candidate)[0]    
   
class SchwefelProblem(FloatsProblem):
    def __init__(self, dimensionality):
        self.args = {'dimensionality': dimensionality}
        self.dimensionality = dimensionality
        self.objective_maximizing = False
    def min(self, d):
        return -500
    def max(self, d):
        return 500
    def best(self):
        return [420.96874636 for _ in self.dims()]
    def best_result(self):
        return 0
    def evaluate(self, candidate):
        return 418.9828872724339*self.dimensionality-sum(x*math.sin(math.sqrt(abs(x))) for x in candidate)
        # return benchmarks.schwefel(candidate)[0]
        
class NormalizedSchwefelProblem(SchwefelProblem):
    def evaluate(self, candidate):
        return (418.9828872724339*self.dimensionality-sum(x*math.sin(math.sqrt(abs(x))) for x in candidate)) / 4000.0
        # return benchmarks.schwefel(candidate)[0] / 4000.0

class GriewankProblem(FloatsProblem):
    def __init__(self, dimensionality):
        self.args = {'dimensionality': dimensionality}
        self.dimensionality = dimensionality
        self.objective_maximizing = False
    def min(self, d):
        return -600
    def max(self, d):
        return 600
    def best(self):
        return [0.0 for _ in self.dims()]
    def best_result(self):
        return 0
    def evaluate(self, candidate):
        return benchmarks.griewank(candidate)[0]
        
class NormalizedGriewankProblem(GriewankProblem):
    def evaluate(self, candidate):
        return benchmarks.griewank(candidate)[0] / 300.0
        
class RoomWalk(FloatsProblem):
    def __init__(self, **kwargs):
        self.args = kwargs
        self.step_count = kwargs.get('step_count', 10)
        self.step_size = kwargs.get('step_size', 0.3)
        self.target_area = kwargs.get('target_area', ((0.4, 0.8), (0.6, 1.0)))
        self.obstacle = kwargs.get('obstacle', ((0.2, 0.2), (0.8, 0.8)))
        self.start = kwargs.get('start', (0.5, 0.1))
        self.penalty = kwargs.get('penality', -0.1)
        self.reward = kwargs.get('reward', 1.0)
        self.dimensionality = 2 * self.step_count
        self.objective_maximizing = True
    def min(self, d):
        return -self.step_size
    def max(self, d):
        return self.step_size
    def evaluate(self, candidate):
        current_position = cp(self.start)
        current_reward = 0
        for i in range(0, len(candidate), 2):
            step = (candidate[i], candidate[i+1])
            next_position = (current_position[0]+step[0], current_position[1]+step[1])
            if next_position[0] < 0.0 or next_position[0] > 1.0 or next_position[1] < 0.0 or next_position[1] > 1.0:
                current_reward += self.penalty
            elif next_position[0] > self.obstacle[0][0] and next_position[0] < self.obstacle[1][0] and next_position[1] > self.obstacle[0][1] and next_position[1] < self.obstacle[1][1]:
                current_reward += self.penalty
            else:
                current_position = next_position
                if current_position[0] > self.target_area[0][0] and current_position[0] < self.target_area[1][0] and current_position[1] > self.target_area[0][1] and current_position[1] < self.target_area[1][1]:
                    current_reward += self.reward
        return current_reward

class NormalizedRoomWalk(RoomWalk):
    def evaluate(self, candidate):
        return super().evaluate(candidate) / float(self.step_count)

class FactoryRobotRoute(VectorProblem):
    task_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a']
    def __init__(self, **kwargs):
        self.args = kwargs
        self.area_size = kwargs.get('area_size', 1000)
        self.tasks = kwargs.get('tasks', 8)
        if type(self.tasks) == int:
            self.tasks = type(self).task_names[0:self.tasks]
            self.args['actual_tasks'] = self.tasks
        self.stations = kwargs.get('stations', 5)
        if type(self.stations) == int:
            self.stations = {task:[(randf(max=self.area_size),randf(max=self.area_size)) for _ in range(self.stations)] for task in self.tasks}
            self.args['actual_stations'] = self.stations
        self.start = kwargs.get('start', (0.0, 0.0))
        self.manhattan_distance = kwargs.get('manhattan_distance', False)
        self.return_to_start = kwargs.get('return_to_start', False)
        self.dimensionality = len(self.tasks)
        self.objective_maximizing = False
    def generate(self):
        return [randi(max=len(self.stations[task])) for task in self.tasks]
    def mutate(self, candidate, **kwargs):
        candidate = cp(candidate)
        d = randi(max=len(candidate))
        candidate[d] = randi(max=len(self.stations[self.tasks[d]]))
        return candidate
    def recombine(self, candidate1, candidate2):
        return [rando(cp(candidate1[d]), cp(candidate2[d])) for d in self.dims()]
    def evaluate(self, candidate):
        current_position = cp(self.start)
        total_cost = 0.0
        path = [self.stations[task][candidate[t]] for t,task in enumerate(self.tasks)]
        if self.return_to_start:
            path.append(self.start)
        for target_position in path:
            if self.manhattan_distance:
                total_cost += abs(current_position[0] - target_position[0]) + abs(current_position[1] - target_position[1])
            else:
                total_cost += math.sqrt((current_position[0] - target_position[0])**2 + (current_position[1] - target_position[1])**2)
            current_position = cp(target_position)
        return total_cost

class NonEuclideanFactoryRobotRoute(VectorProblem):
    task_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a']
    def __init__(self, **kwargs):
        self.args = kwargs
        self.min_cost = kwargs.get('min_cost', 0)
        self.max_cost = kwargs.get('max_cost', 100)
        self.tasks = kwargs.get('tasks', 8)
        if type(self.tasks) == int:
            self.tasks = type(self).task_names[0:self.tasks]
            self.args['actual_tasks'] = self.tasks
        self.stations = kwargs.get('stations', 5)
        if type(self.stations) == int:
            self.stations = {task:[[randf(min=self.min_cost,max=self.max_cost) for other_station in range(self.stations)] for _ in range(self.stations)] for task in self.tasks}
            self.args['actual_stations'] = self.stations
        self.start = kwargs.get('start', (0.0, 0.0))
        self.return_to_start = kwargs.get('return_to_start', False)
        self.dimensionality = len(self.tasks)
        self.objective_maximizing = False
    def min(self, d):
        return self.min_cost
    def max(self, d):
        return self.max_cost
    def generate(self):
        return [randi(max=len(self.stations[task])) for task in self.tasks]
    def mutate(self, candidate, **kwargs):
        candidate = cp(candidate)
        d = randi(max=len(candidate))
        candidate[d] = randi(max=len(self.stations[self.tasks[d]]))
        return candidate
    def recombine(self, candidate1, candidate2):
        return [rando(cp(candidate1[d]), cp(candidate2[d])) for d in self.dims()]
    def evaluate(self, candidate):
        total_cost = 0.0
        path = [candidate[t] for t,task in enumerate(self.tasks)]
        if self.return_to_start:
            path.append(0)
        current_station = 0
        for current_task,target in enumerate(path):
            current_task_name = type(self).task_names[current_task]
            total_cost += self.stations[current_task_name][current_station][target]
            current_station = target
        return total_cost / (self.max_cost * len(self.tasks))

