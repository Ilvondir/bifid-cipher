from ngram import NGramScorer
import string
import numpy as np
import random
from copy import deepcopy
import time

BIFID_ALPHABET = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'
NGRAM_SCORER = NGramScorer('english_bigrams/english_2grams.csv')

def preprocess_plaintext(plaintext):
    plaintext = plaintext.upper()
    # In bifid cipher we must parse J to I
    return ''.join([c for c in plaintext if c in BIFID_ALPHABET]).replace('J', 'I').strip()


with open('./datasets/english_tests/test1.txt') as f:
    plaintext = f.read()

print('Plaintext:')
plaintext = preprocess_plaintext(plaintext)
print(plaintext)



def generate_random_key():
    random_key = random.sample(BIFID_ALPHABET, len(BIFID_ALPHABET))
    return np.reshape(random_key, (5, 5))



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
    return ( round(NGRAM_SCORER.score( decrypt(ENCRYPTED_TEXT, key) ), 2) , key)



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



def evolve(population, population_length):
    elite = population[:population_length//50]
    commons = population[population_length//50:]

    for _ in range(len(population)):
        child1 = born1(elite)
        population.append(commit_key(child1))
        child2 = born2(elite, commons)
        population.append(commit_key(child2))

    k, population = remove_duplicates(population)

    # Diversity injection
    if k >= population_length // 2:
        population = population[:population_length // 2]
        diversity = generate_population(population_length // 2)
        population += diversity
        population = sorted(population, key=lambda x: x[0], reverse=True)
        print('Diversity injected')

    # Individual learning
    for i in range(5):
        population[i] = individual_learning(population[i][1])

    population = sorted(population, key=lambda x: x[0], reverse=True)

    print( population[0] )

    return population[:population_length], population_length



def individual_learning(key, wait_to_progress=.01):
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


def change_key(key):
    new_key = key.copy()
    (i1, j1), (i2, j2) = random.sample([(i, j) for i in range(key.shape[0]) for j in range(key.shape[1])], 2)
    new_key[i1, j1], new_key[i2, j2] = new_key[i2, j2], new_key[i1, j1]
    return new_key
    

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

plaintext_score = NGRAM_SCORER.score(plaintext)
print('Plaintext NGram score:')
print(plaintext_score)

print(evolutionary_attack(1500, 200))