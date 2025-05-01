from bifid_cipher_ca import evolutionary_attack
from time import time

results = []
times = []

for _ in range(15):
    start = time()
    res = evolutionary_attack(800, 400) #1157 1508 2215
    results.append(res)
    times.append(time() - start)

print('\n\n')

for i in range(len(results)):
    print(results[i])
    print(times[i])
    print()

print(sum(times) / len(times))


"""
Tests have shown that the effectiveness of attacks increases significantly for preprocessed plaintexts containing more than 1000 characters. For shorter texts, attacks often stall and considerably prolong the decryption process.

Plaintext length: 1247
Population length: 400
Attempts: 15
Percentage of successes: 14/15 (93.33%)
Mean time of success: ~43s

Plaintext length: 1563
Population length: 500
Attempts: 15
Percentage of successes: 14/15 (93.33%)
Mean time of success: ~101s

Plaintext length: 2061
Population length: 500
Attempts: 15
Percentage of successes: 14/15 (93.33%)
Mean time of success: ~108s

Plaintext length: 2531
Population length: 500
Attempts: 15
Percentage of successes: 14/15 (93.33%)
Mean time of success: ~81s
"""