import neat
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import math


BEST_SCENARIO = None
BEST_FITNESS = float('-inf')
BEST_FITNESS_LIST = []

# sensor class
class Sensor:
    def __init__(self, x, y, sensing_range, com_range):
        self.x = x
        self.y = y
        self.sensing_range = sensing_range
        self.com_range = com_range
        self.active = True

    # moves sensor to its position + dx and dy
    def move(self, dx, dy):
        # change sensors location
        self.x += dx
        self.y += dy

    # determines if a point is in sensors sensing range
    def is_point_covered(self, px, py):
        distance = ((self.x - px)**2 + (self.y - py)**2)**0.5
        return distance <= self.sensing_range

    # determines if a sensor is in sensors communication range
    def is_neighbor(self, other_sensor_x, other_sensor_y):
        distance = ((self.x - other_sensor_x)**2 + (self.y - other_sensor_y)**2)**0.5
        return distance <= self.com_range


#environment class
class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.sensor_list = []

    # adds sensor to environment
    def add_sensor(self, sensor):
        # increment the value of the grid cells within a sensors sensing range
        for i in range(int(max(0, sensor.y - sensor.sensing_range)), int(min(self.height, sensor.y + sensor.sensing_range))):
            for j in range(int(max(0, sensor.x - sensor.sensing_range)), int(min(self.width, sensor.x + sensor.sensing_range))):
                if sensor.is_point_covered(j, i):
                    self.grid[i][j] += 1


# intialize scenario
def init_scenario(num_sensors, sensing_range, com_range, desired_coverage, width, height):
    scenario = Environment(width, height)
    sensor_positions = []

    for _ in range(num_sensors):
        x = random.randint(0, scenario.width)
        y = random.randint(0, scenario.height)
        if (x,y) in sensor_positions:
            while (x,y) in sensor_positions:
                x = random.randint(0, scenario.width)
                y = random.randint(0, scenario.height)
            sensor_positions.append((x,y))
        else:
            sensor_positions.append((x,y))

        scenario.add_sensor(Sensor(x, y, sensing_range, com_range))
        scenario.sensor_list.append(Sensor(x, y, sensing_range, com_range))

    return scenario


# control sensor based on outputs from neural network
def control_sensor(sensor, outputs):
    dx, dy = outputs[:2]
    sensor.move(dx, dy)
    sensor.active = outputs[2] > 0.5


# return the local coverage inside of a sensors sensing radius.
# If coverage is more than k, coverage will just be k
def local_coverage(environment, sensor, k):
    total_coverage = 0

    for i in range(int(max(0, sensor.y - sensor.sensing_range)), int(min(environment.height, sensor.y + sensor.sensing_range))):
            for j in range(int(max(0, sensor.x - sensor.sensing_range)), int(min(environment.width, sensor.x + sensor.sensing_range))):
                if sensor.is_point_covered(j, i):
                    if environment.grid[i][j] >= k:
                        total_coverage += k
                    else:
                        total_coverage += environment.grid[i][j]

    return total_coverage


# Same as local_coverage except returns the position of the 
# closest point that doesn't have k-coverage
def local_coverage2(environment, sensor, k):
    total_coverage = 0
    lowest_coverage = float('inf')
    lowest_coverage_point = (0,0)

    for i in range(int(max(0, sensor.y - sensor.sensing_range)), int(min(environment.height, sensor.y + sensor.sensing_range))):
            for j in range(int(max(0, sensor.x - sensor.sensing_range)), int(min(environment.width, sensor.x + sensor.sensing_range))):
                if sensor.is_point_covered(j, i):
                    if environment.grid[i][j] >= k:
                        total_coverage += k
                    else:
                        if environment.grid[i][j] < lowest_coverage:
                            lowest_coverage = environment.grid[i][j]

                        total_coverage += environment.grid[i][j]

    # if no lowest coverage point 
    if lowest_coverage_point == (0,0):
        return (total_coverage, sensor.x, sensor.y)
    else:
        return (total_coverage, lowest_coverage_point[0], lowest_coverage_point[1])
    

# finds the global coverage of the environment
# if greater than k then point is just k
def global_coverage(environment, k):
    total_coverage = 0

    for row in range(environment.height):
        for col in range(environment.width):
            if environment.grid[row][col] >= k:
                total_coverage += k
            else:
                total_coverage += environment.grid[row][col]
    
    return total_coverage


# k-coverage inputs into neural network. Finds gloabl coverage,
# local coverage, and the closest point in sensors sensing range
# that isn't k-covered
def input_coverage(environment, sensor, k):
    local_coverage = 0
    global_coverage = 0
    closest_distance = float('inf')
    closest_point = None

    for row in range(environment.height):
        for col in range(environment.width):
            if environment.grid[row][col] >= k:
                global_coverage += k
                if sensor.is_point_covered(col, row):
                    local_coverage += k
            else:
                point_coverage = environment.grid[row][col]
                global_coverage += point_coverage
                if sensor.is_point_covered(col, row):
                    local_coverage += point_coverage

                distance = math.sqrt((row - sensor.y)**2) + math.sqrt((col - sensor.x)**2)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = (col, row)

    # if all points in sensors range k-covered return its own position
    if closest_point == None:
        closest_point = (sensor.x, sensor.y)
                
    return (local_coverage, global_coverage, closest_point)


# finds closest uncovered point in sensors sensing range
def find_closest_uncovered_point(sensor, environment, k):
    closest_distance = float('inf')
    closest_point = None

    for y in range(environment.height):
        for x in range(environment.width):
            if environment.grid[y][x] < k:
                distance = math.sqrt((y - sensor.y)**2) + math.sqrt((x - sensor.x)**2)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = (x, y)

     # if all points in sensors range k-covered return its own position
    if closest_point == None:
        closest_point = (sensor.x, sensor.y)

    return closest_point


# finds distance of sensors closest communication neighbor
def closest_com_neighbor(environment, sensor):
    #closest_location = ()
    min_distance = float('inf')

    for other_sensor in environment.sensor_list:
        if other_sensor != sensor:
            distance = math.sqrt((sensor.x - other_sensor.x)**2 + (sensor.y - other_sensor.y)**2)
            if distance < min_distance:
                #closest_sensor = (other_sensor.x, other_sensor.y)
                min_distance = distance
    
    return min_distance


# finds number of sensors in sensor's communication range.
# aka finds sensors communication neighbors
def num_com_neighbors(environment, sensor):
    total_com_neighbors = 0

    for other_sensor in environment.sensor_list:
        if other_sensor != sensor:
            if sensor.is_neighbor(other_sensor.x, other_sensor.y):
                total_com_neighbors += 1
    
    return total_com_neighbors


# calculates connectivity. Divides number of connected 
# components by the number of active sensors to calculate
# a connectivity score. Best score is when active sensors
# = # of connected sensors, which will be 1.0
def calculate_connectivity_score(sensors, com_range):
    num_active_sensors = 0
    active_sensors = []

    for sensor in sensors:
        if sensor.active:
            active_sensors.append(sensor)
            num_active_sensors += 1
    
    if num_active_sensors == 0:
        return 0
    
    # build graph
    graph = {sensor: [] for sensor in active_sensors}
    
    for i in range(num_active_sensors):
        for j in range(i + 1, num_active_sensors):
            sensor = active_sensors[i]
            other_sensor = active_sensors[j]
            # find distance between two sensors
            distance = math.sqrt((sensor.x - other_sensor.x)**2 + (sensor.y - other_sensor.y)**2)
            # if in eachothers communication range connect them
            if distance <= com_range:
                graph[sensor].append(other_sensor)
                graph[other_sensor].append(sensor)
    
    visited = {sensor: False for sensor in active_sensors}
    
    # DFS
    def dfs(sensor):
        stack = [sensor]
        while stack:
            current = stack.pop()
            if not visited[current]:
                visited[current] = True
                for neighbor in graph[current]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
    
    # run DFS
    dfs(active_sensors[0])
    
    connected_sensors = sum(1 for sensor in active_sensors if visited[sensor])
    connectivity_score = connected_sensors / num_active_sensors
    
    return connectivity_score


# creates scenario, runs genomes on scenario, finds best genomes
# and best scenario, repeats the process until threshold/generation limit is met
def eval_genomes(genomes, config):
    global BEST_FITNESS
    global BEST_SCENARIO
    global BEST_FITNESS_LIST
    k = 2

    current_best_fitness = float('-inf')
    sensor_positions = []

    # if there is no univesal best scenario, make one
    if BEST_SCENARIO == None:
        BEST_SCENARIO = Environment(50, 50)
        current_best_scenario = Environment(50, 50)
        environment = Environment(50, 50)

        # populate universal best
        for _ in range(100):
            x = random.randint(0, environment.width)
            y = random.randint(0, environment.height)
            if (x,y) in sensor_positions:
                while (x,y) in sensor_positions:
                    x = random.randint(0, environment.width)
                    y = random.randint(0, environment.height)
                sensor_positions.append((x,y))
            else:
                sensor_positions.append((x,y))

            sensing_range = 10
            com_range = 20
            BEST_SCENARIO.sensor_list.append(Sensor(x, y, sensing_range, com_range))

    for genome_id, genome in genomes:

        # copy the universal best solution to an environment for the neural network
        environment = Environment(BEST_SCENARIO.height, BEST_SCENARIO.width)
        for sensor in BEST_SCENARIO.sensor_list:
            new_sensor = Sensor(sensor.x, sensor.y, sensor.sensing_range, sensor.com_range)
            new_sensor.active = sensor.active
            environment.sensor_list.append(new_sensor)
        
        # run neural network
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # do changes to current environment
        for sensor in environment.sensor_list:
            nn_inputs = input_coverage(environment, sensor, k)
            inputs = (sensor.x, sensor.y, 
                      nn_inputs[0], nn_inputs[1], nn_inputs[2][0], nn_inputs[2][1], 
                      num_com_neighbors(environment, sensor))
            outputs = net.activate(inputs)
            control_sensor(sensor, outputs)

            if sensor.active:
                environment.add_sensor(sensor)

        # compare fitness of current to best current
        genome.fitness = calculate_fitness(environment.sensor_list, environment, k)

        if genome.fitness > current_best_fitness:
            current_best_scenario = environment
            current_best_fitness = genome.fitness

    # at the end of iterating through genomes compare best current to universal best
    if current_best_fitness > BEST_FITNESS:
        BEST_FITNESS = current_best_fitness
        BEST_SCENARIO = current_best_scenario
    
    BEST_FITNESS_LIST.append(BEST_FITNESS)


# caclulates fitness of scenarios after 
# neural network made changes to them
def calculate_fitness(sensors, environment, desired_coverage):
    k_covered_points = 0
    active_sensors = 0
    total_coverage = 0
    inactive_sensors = 0

    # calculate total k-coverage and coverage
    for row in range(environment.height):
        for col in range(environment.width):
            if environment.grid[row][col] >= desired_coverage:
                k_covered_points += 1
                total_coverage += 1
            else:
                if environment.grid[row][col] > 0:
                    #total_coverage += environment.grid[row][col]
                    total_coverage += 1
    
    connectivity_score = calculate_connectivity_score(sensors, sensors[0].com_range)

    k_coverage_rate = k_covered_points / (environment.height * environment.width)
    coverage_rate = total_coverage / (environment.height * environment.width)


    for sensor in sensors:
        if not sensor.active:
            inactive_sensors += 1
        else:
            active_sensors +=1
    
    inactivity = inactive_sensors / len(sensors)

    if connectivity_score == 1.0:
        connectivity_score *= 100
    if k_coverage_rate == 1.0:
        k_coverage_rate *= 100
    if coverage_rate == 1.0:
        coverage_rate *= 100

    fitness_score = (
        connectivity_score * 0.33 +
        k_coverage_rate * 0.35 +
        coverage_rate * 0.31 +
        inactivity * 1
    )

    print(connectivity_score, k_coverage_rate, coverage_rate, active_sensors, fitness_score)
    return fitness_score

# old fitness function. Not in use
def calculate_fitness_old(sensors, environment, desired_coverage):
    k_covered_points = 0
    active_sensors = 0
    total_coverage = 0

    # calculate total k-coverage
    for row in range(environment.height):
        for col in range(environment.width):
            #if environment.grid[row][col] >= desired_coverage:
                #k_covered_points += 1
            if environment.grid[row][col] >= desired_coverage:
                total_coverage += desired_coverage
            else:
                total_coverage += environment.grid[row][col]
    
    # calulate number of active sensors
    for sensor in sensors:
        if sensor.active:
            active_sensors += 1
    
    if total_coverage < 5000:
        print(active_sensors, total_coverage, -100)
        return -100
    else:
        print(active_sensors, total_coverage, total_coverage - active_sensors*20)
        return total_coverage - active_sensors*20


def main():

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 300)

    gen_nums = [i for i in range(300)]
    plt.plot(gen_nums, BEST_FITNESS_LIST)
    
    plt.title('Best Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.show()


main()