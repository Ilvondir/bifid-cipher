"""
Michal Komsa
Bifid Cipher

File with tests of attacks.
"""


from bifid_cipher_ca import evolutionary_attack, preprocess_plaintext, generate_random_key, encrypt, NGRAM_SCORER
from time import time
from multiprocessing import Pool, cpu_count
import os

def test_process(number):

    with open('./datasets/catala_tests/test1.txt', 'r', encoding='UTF-8') as f:
        plaintext = f.read()
    plaintext = preprocess_plaintext(plaintext)

    key0 = generate_random_key()
    encrypted_text = encrypt(plaintext, key0)

    plaintext_score = NGRAM_SCORER.score(plaintext)
    print(len(plaintext))
    print(plaintext_score)

    start = time()
    res = evolutionary_attack(encrypted_text, 1000, 200, verbose=True)
    end = time() - start

    with open(f'test_results/{number}.txt', 'w+', encoding='UTF-8') as file:
        file.write(str(res))
        file.write(str(end))
        file.close()
    print(f'saved by {os.getpid()}')

if __name__ == '__main__':

    with Pool(processes=cpu_count()-1) as pool:
        try:
            pool.map(test_process, range(cpu_count()-1))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()


"""
Tests have shown that the effectiveness of attacks increases significantly for preprocessed plaintexts containing more than 1000 characters. For shorter texts, attacks often stall and considerably prolong the decryption process.


EN: 
    Plaintext length: 1247
    Population length: 400
    Epochs: 150
    Attempts: 15
    Percentage of successes: 14/15 (93.33%)
    Mean time of success: ~43s

    Plaintext length: 1563
    Population length: 500
    Epochs: 150
    Attempts: 15
    Percentage of successes: 14/15 (93.33%)
    Mean time of success: ~101s

    Plaintext length: 2061
    Population length: 500
    Epochs: 150
    Attempts: 15
    Percentage of successes: 14/15 (93.33%)
    Mean time of success: ~108s

    Plaintext length: 2531
    Population length: 500
    Epochs: 150
    Attempts: 15
    Percentage of successes: 14/15 (93.33%)
    Mean time of success: ~81s

CA (bigger alphabet, harder problem):
    Plaintext length: 1033
    Population length: 1000
    Epochs: 300
    Attempts: 15
    Percentage of successes: 8/15 (53.33%)
    Mean time of success: ~148s

    Plaintext length: 1565
    Population length: 1500
    Epochs: 200
    Attempts: 11
    Percentage of successes: 6/11 (54.54%)
    Mean time of success: 681s (multiprocessing)

    Plaintext length: 1349
    Population length: 1000
    Epochs: 300
    Attempts: 11
    Percentage of successes: 7/11 (63.63%)
    Mean time of success: 728s (multiprocessing)
"""