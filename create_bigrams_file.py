from pathlib import Path
import re

LANG = 'en'
LENGTH_OF_NGRAMS = 2

if LANG == 'en':
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    base_folder = 'datasets/english_bigrams'
else:
    alphabet = 'AÀBCÇDEÉÈFGHIÍÏJKLMNOÓÒPQRSTUÚÜVWXYZ'
    base_folder = 'datasets/catala_bigrams'

path = Path(base_folder + '/books')

ngrams = {}

for filename in list(path.glob('*.txt')):
    
    with open(filename, 'r', encoding='utf-8') as file:
        content = re.sub(rf'[^{alphabet}]', '', file.read().upper(), flags=re.UNICODE)

    for i in range(LENGTH_OF_NGRAMS-1, len(content)):
        ngram = content[i-LENGTH_OF_NGRAMS : i]

        if ngram != '':
            if ngram in ngrams.keys():
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1
    
    print(f'{filename} finished!')

ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)

with open(f'{base_folder}/{LANG}_bigrams.csv', 'w+', encoding='UTF-8') as file:
    for ngram in ngrams:
        file.write(f'{ngram[0]},{ngram[1]}\n')