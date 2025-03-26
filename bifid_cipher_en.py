from ngram import NGramScorer
import string

bifid_alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'

def preprocess_plaintext(plaintext):
    # In bifid cipher we must parse J to I
    return ''.join([c for c in plaintext if c not in string.punctuation and c != ' ']).upper().replace('J', 'I')

with open('./datasets/english_tests/test1.txt') as f:
    plaintext = f.read()

print('Plaintext:')
print(preprocess_plaintext(plaintext))