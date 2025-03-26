from ngram import NGramScorer
import string
import numpy as np
import random

BIFID_ALPHABET = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'

def preprocess_plaintext(plaintext):
    # In bifid cipher we must parse J to I
    return ''.join([c for c in plaintext if c not in string.punctuation and c != ' ']).upper().replace('J', 'I')


with open('./datasets/english_tests/test1.txt') as f:
    plaintext = f.read()

print('Plaintext:')
plaintext = preprocess_plaintext(plaintext)
plaintext = preprocess_plaintext('Tajna informacja')
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
    
    coords = []
    for c in encrypted_text:
        result = np.where(key == c)
        coords.append( str(result[0][0]+1) + str(result[1][0]+1) )

    if debug: print(f'Encryption coords:\n{coords}')
    
    new_letter_coords = []

    for i in range(len(coords)):
        for coord_i in range(2):
            if 2*i + coord_i < len(coords):
                new_letter_coords.append(coords[i][coord_i])
            else:
                new_letter_coords[2*i + coord_i-len(coords)] += coords[i][coord_i]
            

    if debug: print(f'Decryption coords:\n{new_letter_coords}')

    return ''.join([key[int(coord[0])-1][int(coord[1])-1] for coord in new_letter_coords])



# key = generate_random_key()
key = np.array([
    ['T', 'A', 'I', 'N', 'E'],
    ['M', 'P', 'Q', 'R', 'H'],
    ['K', 'Y', 'Z', 'U', 'S'],
    ['G', 'X', 'W', 'V', 'L'],
    ['F', 'D', 'C', 'B', 'O'],
])
print('Key:')
print(key)

encrypted_text = encrypt(plaintext, key)
print('Encrypted text:')
print(encrypted_text)

print('Decrypted text:')
print(decrypt(encrypted_text, key))