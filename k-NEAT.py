import neat
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import math

BEST_SCENARIO = None
BEST_FITNESS = float('-inf')

class Sensor:
    def __init__(self, x, y, sensing_range, com_range):
        self.x = x
        self.y = y
        self.sensing_range = sensing_range
        self.com_range = com_range
        self.active = True

    def move(self, dx, dy):
        # change sensors location
        self.x += dx
        self.y += dy

    def is_point_covered(self, px, py):
        # check if a point is in sensors sensing range
        distance = ((self.x - px)**2 + (self.y - py)**2)**0.5

        return distance <= self.sensing_range

    def is_neighbor(self, other_sensor_x, other_sensor_y):
        distance = ((self.x - other_sensor_x)**2 + (self.y - other_sensor_y)**2)**0.5

        return distance <= self.com_range


class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.sensor_list = []

    def add_sensor(self, sensor):
        # increment the value of the grid cells within a sensors sensing range
        for i in range(int(max(0, sensor.y - sensor.sensing_range)), int(min(self.height, sensor.y + sensor.sensing_range))):
            for j in range(int(max(0, sensor.x - sensor.sensing_range)), int(min(self.width, sensor.x + sensor.sensing_range))):
                if sensor.is_point_covered(j, i):
                    self.grid[i][j] += 1


# intialize scenario
# only called once
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
    

def control_sensor(sensor, outputs):
    # use first two outputs of the neural network to control the sensor's movement
    #dx, dy = outputs[:2]
    #sensor.move(dx, dy)
    # displacement = outputs[1]
    # sensor.move(displacement//2, displacement//2)

    # # use third output to turn sensor on/off
    # active_output = 1 / (1 + np.exp(-outputs[0]))
    # sensor.active = active_output > 0.5
    # sensor.move(random.randint(1, 20), random.randint(1, 20))
    # sensor.active = random.randint(0, 1) == 1
    dx, dy = outputs[:2]
    sensor.move(dx, dy)
    
    sensor.active = outputs[2] > 0.5


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

def global_coverage(environment, k):
    total_coverage = 0

    for row in range(environment.height):
        for col in range(environment.width):
            if environment.grid[row][col] >= k:
                total_coverage += k
            else:
                total_coverage += environment.grid[row][col]
    
    return total_coverage

def input_coverage(environment, sensor, k):
    local_coverage = 0
    global_coverage = 0
    closest_distance = float('inf')
    closest_point = (0, 0)

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
                
    return (local_coverage, global_coverage, closest_point)


def find_closest_uncovered_point(sensor, environment, k):
    closest_distance = float('inf')
    closest_point = (0, 0)

    for y in range(environment.height):
        for x in range(environment.width):
            if environment.grid[y][x] < k:
                distance = math.sqrt((y - sensor.y)**2) + math.sqrt((x - sensor.x)**2)
                # Update closest point if this one is closer
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = (x, y)

    return closest_point


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


def num_com_neighbors(environment, sensor):
    total_com_neighbors = 0

    for other_sensor in environment.sensor_list:
        if other_sensor != sensor:
            if sensor.is_neighbor(other_sensor.x, other_sensor.y):
                total_com_neighbors += 1
    
    return total_com_neighbors


def eval_genomes(genomes, config):
    global BEST_FITNESS
    global BEST_SCENARIO
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

        for _ in range(50):

            new_environment = Environment(BEST_SCENARIO.height, BEST_SCENARIO.width)

            for sensor in environment.sensor_list:
                new_sensor = Sensor(sensor.x, sensor.y, sensor.sensing_range, sensor.com_range)
                new_sensor.active = sensor.active
                new_environment.sensor_list.append(new_sensor)

            # do changes to current environment
            for sensor in new_environment.sensor_list:
                closest_uncovered_point = find_closest_uncovered_point(sensor, environment, k)
                coverage_inputs = input_coverage(environment, sensor, k)
                #inputs = (existing_coverage_in_sensing_range(new_environment, sensor), closest_com_neighbor(new_environment, sensor))
                inputs = (sensor.x, sensor.y, coverage_inputs[0], coverage_inputs[1], 
                            coverage_inputs[2][0], coverage_inputs[2][1], num_com_neighbors(environment, sensor))
                outputs = net.activate(inputs)
                control_sensor(sensor, outputs)

                if sensor.active:
                    new_environment.add_sensor(sensor)

        # compare fitness of current to best current
        genome.fitness = calculate_fitness(new_environment.sensor_list, new_environment, k)

        if genome.fitness > current_best_fitness:
            current_best_scenario = environment
            current_best_fitness = genome.fitness

    # at the end of iterating through genomes compare best current to universal best
    if current_best_fitness > BEST_FITNESS:
        BEST_FITNESS = current_best_fitness
        BEST_SCENARIO = current_best_scenario



def calculate_fitness(sensors, environment, desired_coverage):
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
    
    # calulate # of active sensors
    for sensor in sensors:
        if sensor.active:
            active_sensors += 1
    
    # return fitness score
    #return k_covered_points - active_sensors*100
    #print((total_coverage, active_sensors))
    if total_coverage < 5000:
        print(active_sensors, total_coverage, -100)
        return -100
    else:
        print(active_sensors, total_coverage, total_coverage - active_sensors*20)
        return total_coverage - active_sensors*20
    # print(active_sensors, total_coverage, total_coverage - active_sensors*50)
    # return total_coverage - active_sensors*50


# environment = init_scenario(10, 10, 20, 2, 100, 100)
# sensor = Sensor(3, 4, 20, 40)
# environment.add_sensor(sensor)
# plt.imshow(environment.grid)
# plt.show()

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


def main():
    pass
    #                           SENSOR MOVEMENT TEST
    # sensor = Sensor(10, 10, 20)
    # sensor.move(5, 0)
    # if sensor.is_point_covered(15, 15):
    #     print("The point is covered by the sensor.")
    # else:
    #     print("The point is not covered by the sensor.")


    #                           ENVIRONMENT TEST
    # environment = Environment(200, 200)
    # sensor = Sensor(0, 0, 20)
    # environment.add_sensor(sensor)
    # print(environment.grid)


    #                       MATPLOTLIB TEST
    # environment = Environment(200, 200)
    # sensor = Sensor(3, 3, 20)
    # environment.add_sensor(sensor)
    # plt.imshow(environment.grid)
    # plt.show()




#main()