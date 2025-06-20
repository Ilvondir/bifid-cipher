"""
Michal Komsa
Bifid Cipher
https://mattomatti.com/pl/a35bk

Evolutionary attack with:
    - Diversity Injection
    When the number of duplicated keys exceeds half the population size, half of the population is replaced with random keys.
    - Entire Mutation
    When the best key's score stagnates for 2 epochs, a portion of the population undergoes random mutation.
    - Individual Learning
    Six keys are improved using hill climbing during each epoch.
"""


from ngram import NGramScorer
import numpy as np
import random
from copy import deepcopy
import time
import math

BIFID_ALPHABET = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'
NGRAM_SCORER = NGramScorer('english_bigrams/en_bigrams.csv')

def preprocess_plaintext(plaintext):
    plaintext = plaintext.upper()
    # In english bifid cipher we must parse J to I
    return ''.join([c for c in plaintext if c in BIFID_ALPHABET]).replace('J', 'I').strip()



def generate_random_key():
    random_key = random.sample(BIFID_ALPHABET, len(BIFID_ALPHABET))
    return np.reshape(np.array(random_key), (5, 5))



def encrypt(plaintext, key, verbose=False):

    coords = []
    for c in plaintext:
        result = np.where(key == c)
        coords.append( str(result[0][0]+1) + str(result[1][0]+1) )

    if verbose: print(f'Real coords:\n{coords}')
    
    new_letter_coords = []

    current_value = ''
    for dimension in [0, 1]:
        for coord in coords:
            current_value += str(coord[dimension])
            if len(current_value) == 2:
                new_letter_coords.append(current_value)
                current_value = ''

    if verbose: print(f'Encryption coords:\n{new_letter_coords}')

    return ''.join([key[int(coord[0])-1][int(coord[1])-1] for coord in new_letter_coords])



def decrypt(encrypted_text, key, verbose=False):

    char_to_coords = {key[i, j]: f"{i+1}{j+1}" for i in range(key.shape[0]) for j in range(key.shape[1])}
    coords = [char_to_coords[c] for c in encrypted_text]

    if verbose: print(f'Encryption coords:\n{coords}')

    new_letter_coords = []
    total_len = len(coords) * 2

    for i in range(total_len):
        original_index = i // 2
        coord_part = i % 2
        if i < total_len - len(coords):  
            new_letter_coords.append(coords[original_index][coord_part])
        else:  
            new_letter_coords[i - (total_len - len(coords))] += coords[original_index][coord_part]

    if verbose:
        print(f'Decryption coords:\n{new_letter_coords}')

    return ''.join(key[int(coord[0])-1, int(coord[1])-1] for coord in new_letter_coords)



def crossover(key1, key2, verbose=False):
    new_key = np.empty_like(key1[1])

    random_axis = random.randint(0, 1) # 0-rows 1-columns
    lines_to_copy = random.sample(range(new_key.shape[0]), random.randint(1, 3))
    remaining_lines = [i for i in range(new_key.shape[0]) if i not in lines_to_copy]
    random_parent_num = random.randint(0, 1)

    base_parent = key1[1] if random_parent_num == 0 else key2[1]
    second_parent = key1[1] if random_parent_num == 1 else key2[1]

    if random_axis == 1:
        new_key[:, lines_to_copy] = base_parent[:, lines_to_copy]
        new_key[:, remaining_lines] = second_parent[:, remaining_lines]
    else:
        new_key[lines_to_copy, :] = base_parent[lines_to_copy, :]
        new_key[remaining_lines, :] = second_parent[remaining_lines, :]

    
    missing_letters = missing = np.setdiff1d(base_parent, new_key)

    flat_key = new_key.flatten()
    unique_elements, counts = np.unique(flat_key, return_counts=True)
    duplicates = unique_elements[counts > 1]

    duplicates_positions = {}
    for letter in duplicates:
        duplicates_positions[letter] = list(zip(*np.where(new_key == letter)))

    for letter, positions in duplicates_positions.items():
        new_key[positions[1][0], positions[1][1]] = missing_letters[0]
        missing_letters = missing_letters[1:]

    if verbose:
        print(f'Key1:\n{key1[1]}')
        print(f'Key2:\n{key2[1]}')
        print(f'Axis: {random_axis}')
        print(f'To copy from base: {lines_to_copy}')
        print(f'Remaining: {remaining_lines}')
        print(f'Missing: {missing}')
        print(f'Base parent: {random_parent_num}')
        print(f'New key:\n{new_key}')

    return new_key


def commit_key(encrypted_text, key):
    return ( NGRAM_SCORER.score( decrypt(encrypted_text, key) ) , key)



def born1(population):
    r1, r2 = random.sample(population, 2)
    return crossover(r1, r2)



def born2(population1, population2):
    r1 = random.choice(population1)
    r2 = random.choice(population2)
    return crossover(r1, r2)



def remove_duplicates(population, verbose=True):
    seen = set()
    new_population = []
    k = 0

    for elem in population:
        key_tuple = tuple(elem[1].flatten())
        
        if key_tuple not in seen:
            seen.add(key_tuple)
            new_population.append(elem)
        else:
            k += 1

    new_population = sorted(new_population, key=lambda x: x[0], reverse=True)

    if verbose: print(f"Removed duplicates: {k}")
    return k, new_population



def generate_population(encrypted_text, population_length):
    population = []

    for _ in range(population_length):
        key = generate_random_key()
        population.append(commit_key(encrypted_text, key))

    return sorted(population, key=lambda x: x[0], reverse=True)


last_best_values = []

def evolve(encrypted_text, population, population_length, verbose=True):
    global last_best_values

    elite = population[:population_length//20]
    commons = population[population_length//20:]

    for _ in range(len(population)):
        child1 = born1(elite)
        population.append(commit_key(encrypted_text, child1))
        child2 = born2(elite, commons)
        population.append(commit_key(encrypted_text, child2))

    if verbose: print('Childs created')

    k, population = remove_duplicates(population, verbose)
    injection = False

    # Diversity injection
    if k >= population_length//2:
        population = population[:population_length // 2]
        population += generate_population(encrypted_text, population_length // 2)
        population = sorted(population, key=lambda x: x[0], reverse=True)
        injection = True
        if verbose: print('Diversity injected')


    last_best_values.append(population[0][0])

    # Entire mutation
    if len(last_best_values) >= 2:
        if np.all([x == last_best_values[-2] for x in last_best_values[-2:]]):
            population = population[:population_length]

            for i in range(population_length // 50, population_length // 2 if injection else population_length):
                population[i] = commit_key(encrypted_text, change_key(population[i][1], probs=[0, 0.4, 0.7, 1]))

            for i in range(0, population_length // 50):
                population.append(commit_key(encrypted_text, change_key(population[0][1], probs=[0, 0.4, 0.7, 1])))
    
            if verbose: print('Population mutated')
            last_best_values = [last_best_values[-1]]

            population = sorted(population, key=lambda x: x[0], reverse=True)
    

    # Individual learning
    for i in range(3):
        population[i] = individual_learning_hill_climbing(encrypted_text, population[i][1])
        rand_index_1 = random.randint(3, population_length // 50)
        population[rand_index_1] = individual_learning_hill_climbing(encrypted_text, population[rand_index_1][1])

    population = sorted(population, key=lambda x: x[0], reverse=True)

    if verbose: print( population[0] )

    return population[:population_length], population_length



def individual_learning_hill_climbing(encrypted_text, key, wait_to_progress=.025):
    old_key = np.copy(key) 
    old_value = NGRAM_SCORER.score( decrypt(encrypted_text, old_key) )

    deadline = time.time() + wait_to_progress

    while time.time() < deadline:
        new_key = change_key(old_key)
        new_value = NGRAM_SCORER.score( decrypt(encrypted_text, new_key) )

        if old_value <= new_value:
            old_key, old_value = new_key, new_value
            deadline = time.time() + wait_to_progress

    return commit_key(encrypted_text, old_key)



def individual_learning_simulated_annealing(encrypted_text, key, initial_temperature=10, cooling_rate=0.96):
    old_key = np.copy(key) 
    old_value = NGRAM_SCORER.score( decrypt(encrypted_text, old_key) )
    temperature = initial_temperature

    best_key = old_key
    best_value = old_value

    while temperature > .3:

        new_key = change_key(old_key)
        new_value = NGRAM_SCORER.score( decrypt(encrypted_text, new_key) )

        if new_value > old_value:
            old_key = new_key
            old_value = new_value
        else:
            if random.random() < math.exp((new_value - old_value) / temperature):
                old_key = new_key
                old_value = new_value

        if old_value > best_value:
            best_key = old_key
            best_value = old_value

        temperature *= cooling_rate

    return commit_key(encrypted_text, best_key)



def swap_letters(key, num_swaps=1):
    new_key = deepcopy(key)
    all_indices = [(i, j) for i in range(new_key.shape[0]) for j in range(new_key.shape[1])]
    
    selected_indices = random.sample(all_indices, 2 * num_swaps)
    
    for i in range(num_swaps):
        idx1, idx2 = 2 * i, 2 * i + 1
        new_key[selected_indices[idx1]], new_key[selected_indices[idx2]] = new_key[selected_indices[idx2]], new_key[selected_indices[idx1]]
    
    return new_key



def swap_lines(key):
    new_key = deepcopy(key)

    if random.random() >= 0.5:
        cols = list(range(key.shape[1]))
        col1, col2 = random.sample(cols, 2)
        new_key[:, [col1, col2]] = new_key[:, [col2, col1]]
    else:
        rows = list(range(key.shape[0]))
        row1, row2 = random.sample(rows, 2)
        new_key[:, [row1, row2]] = new_key[:, [row2, row1]]

    return new_key

def shiftline(key):
    new_key = deepcopy(key)

    if random.random() < 0.5:
        row_idx = random.randint(0, key.shape[0] - 1)
        shift_amount = random.randint(1, 4)
        new_key[row_idx] = np.roll(new_key[row_idx], shift_amount)
    else:
        col_idx = random.randint(0, key.shape[1] - 1)
        shift_amount = random.randint(1, 4)
        new_key[:, col_idx] = np.roll(new_key[:, col_idx], shift_amount)

    return new_key



def change_key(key, probs=[0.7, 0.8, 0.9, 1]):
    r = random.random()

    if r <= probs[0]:
        return swap_letters(key, 1)
    elif r <= probs[1]:
        return swap_letters(key, 3)
    elif r <= probs[2]:
        return shiftline(key)
    else: 
        return swap_lines(key)
    

def evolutionary_attack(encrypted_text, population_length, max_iters=100, verbose=True):

    population = generate_population(encrypted_text, population_length)

    i = 0
    while i < max_iters:
        if verbose: print(f'\nEPOCH {i+1} ')
        population, population_length = evolve(encrypted_text, population, population_length, verbose)
        i += 1

    return population[0][0], population[0][1], decrypt(encrypted_text, population[0][1])


# with open('./datasets/english_tests/test.txt', encoding='UTF-8') as f:
#         plaintext = f.read()
# plaintext = preprocess_plaintext(plaintext)
# key0 = generate_random_key()
# encrypted_text = encrypt(plaintext, key0)
# plaintext_score = NGRAM_SCORER.score(plaintext)
# print(len(plaintext))
# print(plaintext_score)

# evolutionary_attack(encrypted_text, 500, 100)