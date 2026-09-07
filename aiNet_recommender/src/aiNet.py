from .artificial_lymphocytes import ARB

import numpy as np

# affinity methods

def dot_prod_affinity(antigen: list[float], antibody: list[float]) -> float:
    return np.dot(antigen, antibody)

def euclid_distance_affinity(antigen: list[float], antibody: list[float]) -> float:
    squared_diff = 0

    for x, y in zip(antigen, antibody):
        squared_diff += (x-y)**2

    distance = np.sqrt(squared_diff)

    return 1 / (1 + distance)

def norm(alc) -> float:
    squared_sum = sum([x**2 for x in alc])

    return np.sqrt(squared_sum)


def cosine_sim_affinity(antigen: list[float], antibody: list[float]) -> float:
    return np.dot(antigen, antibody)/(norm(antigen)*norm(antibody))



# the aiNet functions 

def normalized_distance(vector_a: list[float], vector_b: list[float]) -> float:
    squared_diff = 0

    for x, y in zip(vector_a, vector_b):
        squared_diff += (x - y)**2

    distance = np.sqrt(squared_diff)

    # bound the distance between 0 and 1, so it can be compared against a threshold
    max_distance = np.sqrt(len(vector_a))

    return distance / max_distance


def build_network(arbs: list[ARB], affinity_threshold: float) -> None:
    for arb in arbs:
        arb.neighbours = []

    for i in range(len(arbs)):
        for j in range(i + 1, len(arbs)):
            distance = normalized_distance(arbs[i].vector, arbs[j].vector)

            if distance < affinity_threshold:
                arbs[i].neighbours.append(arbs[j])
                arbs[j].neighbours.append(arbs[i])


def calculate_antigen_stimulation(arb, antigens, affinity_threshold):
    stimulation = 0.0

    for antigen in antigens:
        distance = normalized_distance(antigen, arb.vector)

        if distance < affinity_threshold:
            stimulation += 1.0 - distance

    arb.antigen_stimulation = stimulation


def calculate_network_stimulation(arb):
    stimulation = 0.0

    for neighbour in arb.neighbours:
        distance = normalized_distance(arb.vector, neighbour.vector)
        stimulation += 1.0 - distance

    arb.network_stimulation = stimulation


def calculate_network_suppression(arb):
    suppression = 0.0

    for neighbour in arb.neighbours:
        distance = normalized_distance(arb.vector, neighbour.vector)
        suppression -= distance

    arb.network_suppression = suppression


def calculate_total_stimulation(arb):
    arb.total_stimulation = arb.antigen_stimulation + arb.network_stimulation + arb.network_suppression


def calculate_stimulation(arbs, antigens, affinity_threshold):
    for arb in arbs:
        calculate_antigen_stimulation(arb, antigens, affinity_threshold)
        calculate_network_stimulation(arb)
        calculate_network_suppression(arb)
        calculate_total_stimulation(arb)


def calculate_resources(arb, alpha):
    arb.resources = alpha * (arb.total_stimulation ** 2)


def allocate_resources(arbs, maximum_resources, alpha):
    total_resources = 0.0

    for arb in arbs:
        calculate_resources(arb, alpha)
        total_resources += arb.resources

    if total_resources <= maximum_resources:
        return

    # not enough resources to go around, take resources away from the weakest arbs first
    arbs.sort(key=lambda arb: arb.resources)

    excess = total_resources - maximum_resources

    for arb in arbs:
        if arb.resources == 0:
            continue

        if arb.resources <= excess:
            excess -= arb.resources
            arb.resources = 0.0
        else:
            arb.resources -= excess
            excess = 0.0
            break


def remove_zero_resource_arbs(arbs):
    survivors = []

    for arb in arbs:
        if arb.resources > 0:
            survivors.append(arb)

    return survivors


def mutate_vector(vector, mutation_amount):
    mutated = []

    for i in range(len(vector)):
        if i % 2 == 0:
            change = mutation_amount
        else:
            change = -mutation_amount

        new_value = vector[i] + change

        # clamp the mutated value between 0 and 1
        if new_value < 0:
            new_value = 0.0
        if new_value > 1:
            new_value = 1.0

        mutated.append(new_value)

    return mutated


def clone_and_mutate(arbs, stimulation_threshold):
    clones = []

    for arb in arbs:
        if arb.total_stimulation > stimulation_threshold:
            number_of_clones = int(arb.resources)

            if number_of_clones < 1:
                number_of_clones = 1

            mutation_amount = 1.0 / (1.0 + arb.total_stimulation)

            for i in range(number_of_clones):
                new_vector = mutate_vector(arb.vector, mutation_amount)
                clones.append(ARB(new_vector))

    return clones


def integrate_clones(arbs, clones):
    arbs.extend(clones)


def initialise_arbs(training_data, number_of_arbs):
    arbs = []

    for i in range(number_of_arbs):
        arbs.append(ARB(training_data[i]))

    return arbs


def create_antigen_set(training_data, number_of_arbs):
    antigens = []

    for i in range(number_of_arbs, len(training_data)):
        antigens.append(training_data[i])

    return antigens


def calculate_affinity_threshold(training_data):
    total_distance = 0.0
    number_of_pairs = 0.0

    for i in range(len(training_data)):
        for j in range(i + 1, len(training_data)):
            total_distance += normalized_distance(training_data[i], training_data[j])
            number_of_pairs += 1

    return total_distance / number_of_pairs


def train_aiNet(training_data, number_of_initial_arbs, maximum_resources, alpha, stimulation_threshold, maximum_iterations, verbose=True):
    affinity_threshold = calculate_affinity_threshold(training_data)

    if verbose:
        print('Affinity threshold', affinity_threshold)

    arbs = initialise_arbs(training_data, number_of_initial_arbs)
    antigens = create_antigen_set(training_data, number_of_initial_arbs)

    build_network(arbs, affinity_threshold)

    for iteration in range(maximum_iterations):
        calculate_stimulation(arbs, antigens, affinity_threshold)
        allocate_resources(arbs, maximum_resources, alpha)

        arbs = remove_zero_resource_arbs(arbs)

        clones = clone_and_mutate(arbs, stimulation_threshold)
        integrate_clones(arbs, clones)

        build_network(arbs, affinity_threshold)

        if verbose:
            print(f'ITERATION {iteration + 1}: ARBs={len(arbs)}, clones={len(clones)}')

    return arbs, affinity_threshold
