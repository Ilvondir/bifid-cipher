from ngram import NGramScorer
import numpy as np
import random
from copy import deepcopy
import time
import math

BIFID_ALPHABET = 'AĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSŚTUVWXYZŹŻ?'
NGRAM_SCORER = NGramScorer('polish_bigrams/pl_bigrams.csv')


def preprocess_plaintext(plaintext):
    plaintext = plaintext.upper().replace('?', '')
    return ''.join([c for c in plaintext if c in BIFID_ALPHABET]).strip()



with open('./datasets/polish_tests/test1.txt', 'r', encoding='UTF-8') as f:
    plaintext = f.read()



print('Plaintext:')
plaintext = preprocess_plaintext(plaintext)
print(plaintext)



def generate_random_key():
    random_key = random.sample(BIFID_ALPHABET, len(BIFID_ALPHABET))
    return np.reshape(random_key, (6, 6))



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



def inheritance(key1, key2, debug=False):
    if debug: print(f'Key1:\n{key1[1]}')
    if debug: print(f'Key2:\n{key2[1]}')

    new_key = np.empty_like(key1[1], dtype=key1[1].dtype)

    indexes_of_same_letters = np.where(key1[1] == key2[1])

    if debug: print(f'Indexes of same letters:\n{indexes_of_same_letters}')

    new_key.fill('')
    new_key[indexes_of_same_letters] = key2[1][indexes_of_same_letters]

    missing_letters = np.setdiff1d(key1[1], new_key)
    random.shuffle(missing_letters)

    missing_idx = np.where(new_key == '')
    new_key[missing_idx] = missing_letters[:len(missing_idx[0])]

    if debug: print(f'New key:\n{new_key}')

    return new_key




def commit_key(key):
    return ( round(NGRAM_SCORER.score( decrypt(ENCRYPTED_TEXT, key) ), 1) , key)



def born1(population):
    r1, r2 = random.sample(population, 2)
    return inheritance(r1, r2)



def born2(population1, population2):
    r1 = random.choice(population1)
    r2 = random.choice(population2)
    return inheritance(r1, r2)



def remove_duplicates(population):
    temp_population = deepcopy(population)

    sorted_ = sorted(temp_population, key=lambda x: x[0], reverse=True)
    new_population = [sorted_[0]]

    k = 0
    for i in range(1, len(sorted_)):
        if sorted_[i][0] != sorted_[i-1][0]:
            new_population.append(sorted_[i])
        else:
            k += 1

    print(f'Removed duplicates: {k}')

    return k, new_population



def generate_population(population_length):
    population = []

    for _ in range(population_length):
        key = generate_random_key()
        population.append(commit_key(key))

    return sorted(population, key=lambda x: x[0], reverse=True)



is_even = True

def evolve(population, population_length):
    global is_even

    elite = population[:population_length//50]
    commons = population[population_length//50:]

    for _ in range(len(population)):
        child1 = born1(elite)
        population.append(commit_key(child1))
        child2 = born2(elite, commons)
        population.append(commit_key(child2))

    k, population = remove_duplicates(population)

    # Diversity injection
    if k >= 100:
        population = population[:population_length // 2]
        population += generate_population(population_length // 2)
        population = sorted(population, key=lambda x: x[0], reverse=True)
        print('Diversity injected')

    # Individual learning
    is_even = not is_even
    
    print(f'Individual learning method: {'Simulated Annealing' if is_even else 'Hill Climbing'}')

    for i in range(10):
        if is_even:
            population[i] = individual_learning_simulated_annealing(population[i][1])
            rand_index_1 = random.randint(11, 100)
            population[rand_index_1] = individual_learning_simulated_annealing(population[rand_index_1][1])
            rand_index_2 = random.randint(101, population_length-1)
            population[rand_index_2] = individual_learning_simulated_annealing(population[rand_index_2][1])
        else:
            population[i] = shotgun_hill_climbing(population[i][1])
            rand_index_1 = random.randint(11, population_length-1)
            population[rand_index_1] = shotgun_hill_climbing(population[rand_index_1][1])
            rand_index_2 = random.randint(101, population_length-1)
            population[rand_index_2] = shotgun_hill_climbing(population[rand_index_2][1])

    population = sorted(population, key=lambda x: x[0], reverse=True)

    print( population[0] )

    return population[:population_length], population_length



def shotgun_hill_climbing(key, wait_to_progress=0.03, timelimit=0.4):
    best_key = np.copy(key)
    best_value = NGRAM_SCORER.score(decrypt(ENCRYPTED_TEXT, best_key))
    t0 = time.time()

    while time.time() - t0 < timelimit:
        old_key = generate_random_key()
        old_value = NGRAM_SCORER.score(decrypt(ENCRYPTED_TEXT, old_key))
        time_to_progress = time.time()

        while time.time() - time_to_progress < wait_to_progress:
            new_key = change_key(old_key)
            new_value = NGRAM_SCORER.score(decrypt(ENCRYPTED_TEXT, new_key))

            if old_value < new_value:
                old_key, old_value = new_key, new_value
                time_to_progress = time.time()

                # if best_value < new_value:
                #     best_key, best_value = new_key, new_value
                #     print([best_value, best_key, decrypt(ENCRYPTED_TEXT, best_key)])

        # print()

    return commit_key(old_key)



def individual_learning_hill_climbing(key, wait_to_progress=.012):
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



def individual_learning_simulated_annealing(key, initial_temperature=10, cooling_rate=0.97):
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
    # return swap_letters(key, 1)
    rand_num = random.random()

    if 0 <= rand_num < 0.5:
        return swap_letters(key, 1)
    elif 0.5 <= rand_num < 0.75:
        return swap_letters(key, 2)
    elif 0.75 <= rand_num < 0.9:
        return swap_letters(key, 4)
    else:
        return swap_lines(key)

    

def evolutionary_attack(population_length, max_iters=100):
    population = generate_population(population_length)

    i = 0
    while i < max_iters:
        print(f'\nEPOCH {i+1} ')
        population, population_length = evolve(population, population_length)
        i += 1
        if population[0][0] >= plaintext_score: break

    return population[0][0], population[0][1], decrypt(ENCRYPTED_TEXT, population[0][1])



key = generate_random_key()
print('Key:')
print(key)

ENCRYPTED_TEXT = encrypt(plaintext, key)
print('Encrypted text:')
print(ENCRYPTED_TEXT)

print('Decrypted text:')
print(decrypt(ENCRYPTED_TEXT, key))

plaintext_score = round(NGRAM_SCORER.score(plaintext), 2)
print('Plaintext NGram score:')
print(plaintext_score)

print('Result:')
print(evolutionary_attack(2500, 300))

# print(key)


# print(shotgun_hill_climbing(ENCRYPTED_TEXT))

print(key)