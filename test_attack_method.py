from bifid_cipher_ca import evolutionary_attack
from time import time
from multiprocessing import Pool
import os

def test_process(number):

    start = time()
    res = evolutionary_attack(1000, 200, verbose=False)
    end = time() - start

    with open(f'test_results/{number}.txt', 'w+', encoding='UTF-8') as file:
        file.write(str(res))
        file.write(str(end))
        file.close()
    print(f'saved by {os.getpid()}')

if __name__ == '__main__':

    with Pool(processes=11) as pool:
        try:
            pool.map(test_process, range(11))
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

CA: 
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