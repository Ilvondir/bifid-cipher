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


with open('./datasets/english_tests/test3.txt', encoding='UTF-8') as f:
    plaintext = f.read()

print('Plaintext:')
plaintext = preprocess_plaintext(plaintext)
print(plaintext)



def generate_random_key():
    random_key = random.sample(BIFID_ALPHABET, len(BIFID_ALPHABET))
    return np.reshape(np.array(random_key), (5, 5))



def encrypt(plaintext, key, debug=False):

    coords = []
    for c in plaintext:
        result = np.where(key == c)
        coords.append( str(result[0][0]+1) + str(result[1][0]+1) )

    if debug: print(f'Real coords:\n{coords}')
    
    new_letter_coords = []

    current_value = ''
    for dimension in [0, 1]:
        for coord in coords:
            current_value += str(coord[dimension])
            if len(current_value) == 2:
                new_letter_coords.append(current_value)
                current_value = ''

    if debug: print(f'Encryption coords:\n{new_letter_coords}')

    return ''.join([key[int(coord[0])-1][int(coord[1])-1] for coord in new_letter_coords])



def decrypt(encrypted_text, key, debug=False):

    char_to_coords = {key[i, j]: f"{i+1}{j+1}" for i in range(key.shape[0]) for j in range(key.shape[1])}
    coords = [char_to_coords[c] for c in encrypted_text]

    if debug: print(f'Encryption coords:\n{coords}')

    new_letter_coords = []
    total_len = len(coords) * 2

    for i in range(total_len):
        original_index = i // 2
        coord_part = i % 2
        if i < total_len - len(coords):  
            new_letter_coords.append(coords[original_index][coord_part])
        else:  
            new_letter_coords[i - (total_len - len(coords))] += coords[original_index][coord_part]

    if debug:
        print(f'Decryption coords:\n{new_letter_coords}')

    return ''.join(key[int(coord[0])-1, int(coord[1])-1] for coord in new_letter_coords)



def crossover(key1, key2, debug=False):
    new_key = np.empty_like(key1[1])
    new_key.fill('')

    random_axis = random.randint(0, 1) # 0-rows 1-columns
    random_size = random.randint(1, 3) # Maks 3 linie
    random_start = random.randint(0, new_key.shape[0]-random_size)

    random_parent_num = random.randint(0, 1)
    base_parent = key1[1] if random_parent_num == 0 else key2[1]

    if random_axis == 1:
        new_key[:, random_start:random_start+random_size+1] = base_parent[:, random_start:random_start+random_size+1]
    else:
        new_key[random_start:random_start+random_size+1, :] = base_parent[random_start:random_start+random_size+1, :]


    missing_letters = np.setdiff1d(key1[1], new_key)
    random.shuffle(missing_letters)

    missing_idx = np.where(new_key == '')
    new_key[missing_idx] = missing_letters[:len(missing_idx[0])]


    if debug:
        print(f'Key1:\n{key1[1]}')
        print(f'Key2:\n{key2[1]}')
        print(f'Axis: {random_axis}')
        print(f'Size: {random_size}')
        print(f'Start: {random_start}')
        print(f'Base parent: {random_parent_num}')
        print(f'New key:\n{new_key}')

    return new_key


def commit_key(key):
    return ( NGRAM_SCORER.score( decrypt(ENCRYPTED_TEXT, key) ) , key)



def born1(population):
    r1, r2 = random.sample(population, 2)
    return crossover(r1, r2)



def born2(population1, population2):
    r1 = random.choice(population1)
    r2 = random.choice(population2)
    return crossover(r1, r2)



def remove_duplicates(population):
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

    new_population = sorted(population, key=lambda x: x[0], reverse=True)

    print(f"Removed duplicates: {k}")
    return k, new_population



def generate_population(population_length):
    population = []

    for _ in range(population_length):
        key = generate_random_key()
        population.append(commit_key(key))

    return sorted(population, key=lambda x: x[0], reverse=True)



is_even = False

def evolve(population, population_length):
    global is_even

    elite = population[:population_length//50]
    commons = population[population_length//50:]

    for _ in range(len(population)):
        child1 = born1(elite)
        population.append(commit_key(child1))
        child2 = born2(elite, commons)
        population.append(commit_key(child2))
        child3 = born2(population[0:1], elite)
        population.append( commit_key(child3) )

    print('Childs created')

    k, population = remove_duplicates(population)

    # Diversity injection
    if k >= population_length // 2:
        population = population[:population_length // 2]
        population += generate_population(population_length // 2)
        population = sorted(population, key=lambda x: x[0], reverse=True)
        print('Diversity injected')

    # Individual learning
    is_even = not is_even
    
    print(f'Individual learning method: {'Simulated Annealing' if is_even else 'Hill Climbing'}')

    for i in range(3):
        if is_even:
            population[i] = individual_learning_simulated_annealing(population[i][1])
            rand_index_1 = random.randint(4, 15)
            population[rand_index_1] = individual_learning_simulated_annealing(population[rand_index_1][1])
        else:
            population[i] = individual_learning_hill_climbing(population[i][1])
            rand_index_1 = random.randint(4, 15)
            population[rand_index_1] = individual_learning_hill_climbing(population[rand_index_1][1])

    population = sorted(population, key=lambda x: x[0], reverse=True)

    print( population[0] )

    return population[:population_length], population_length



def individual_learning_hill_climbing(key, wait_to_progress=.02):
    old_key = np.copy(key) 
    old_value = NGRAM_SCORER.score( decrypt(ENCRYPTED_TEXT, old_key) )

    deadline = time.time() + wait_to_progress

    while time.time() < deadline:
        new_key = change_key(old_key)
        new_value = NGRAM_SCORER.score( decrypt(ENCRYPTED_TEXT, new_key) )

        if old_value < new_value:
            old_key, old_value = new_key, new_value
            deadline = time.time() + wait_to_progress

    return commit_key(old_key)



def individual_learning_simulated_annealing(key, initial_temperature=10, cooling_rate=0.96):
    old_key = np.copy(key) 
    old_value = NGRAM_SCORER.score( decrypt(ENCRYPTED_TEXT, old_key) )
    temperature = initial_temperature

    best_key = old_key
    best_value = old_value

    while temperature > .3:

        new_key = change_key(old_key)
        new_value = NGRAM_SCORER.score( decrypt(ENCRYPTED_TEXT, new_key) )

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

    return commit_key(best_key)



def swap_letters(key, num_swaps=1):
    new_key = key.copy()
    all_indices = [(i, j) for i in range(key.shape[0]) for j in range(key.shape[1])]
    
    selected_indices = random.sample(all_indices, 2 * num_swaps)
    
    for i in range(num_swaps):
        idx1, idx2 = 2 * i, 2 * i + 1
        new_key[selected_indices[idx1]], new_key[selected_indices[idx2]] = new_key[selected_indices[idx2]], new_key[selected_indices[idx1]]
    
    return new_key



def swap_lines(key):
    new_key = key.copy()

    if random.random() >= 0.5:
        cols = list(range(key.shape[1]))
        col1, col2 = random.sample(cols, 2)
        new_key[:, [col1, col2]] = new_key[:, [col2, col1]]
    else:
        rows = list(range(key.shape[0]))
        row1, row2 = random.sample(rows, 2)
        new_key[:, [row1, row2]] = new_key[:, [row2, row1]]

    return new_key



def change_key(key):
    return swap_letters(key, 1)
    

def evolutionary_attack(population_length, max_iters=100):
    population = generate_population(population_length)

    i = 0
    while i < max_iters:
        print(f'\nEPOCH {i+1} ')
        population, population_length = evolve(population, population_length)
        i += 1
        if population[0][0] >= plaintext_score: break

    return population[0][0], population[0][1], decrypt(ENCRYPTED_TEXT, population[0][1])

key0 = generate_random_key()
print('Original key:')
print(key0)

ENCRYPTED_TEXT = encrypt(plaintext, key0)
print('Encrypted text:')
print(ENCRYPTED_TEXT)

print('Decrypted text:')
print(decrypt(ENCRYPTED_TEXT, key0))

plaintext_score = NGRAM_SCORER.score(plaintext)
print('Plaintext NGram score:')
print(plaintext_score)

print('\nResult:')
print(evolutionary_attack(1000, 200))

print('Original key:')
print(key0)


