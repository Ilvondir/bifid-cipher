# Bifid Cipher

Python scripts implementing encryption and decryption methods based on the Bifid cipher for both English and Catalan languages. Given that Bifid cipher keys are represented as matrices, the NumPy library is utilized for efficient key handling and manipulation. Beyond basic encryption and decryption functionalities, the scripts include an implementation of a cryptanalytic attack on the Bifid cipher designed to recover the encryption key from the ciphertext alone.

This attack relies on a scoring metric to algorithmically evaluate and optimize candidate keys. For this purpose, bigram frequency analysis was employed. Frequency data for all letter pairs (bigrams) were compiled from multiple books in both English and Catalan, resulting in language-specific frequency files. Based on these datasets, a statistical scorer was implemented to assess the linguistic plausibility of decrypted text.

The attack was implemented using evolutionary algorithms, with crossover operations based on two parent solutions. The algorithm includes mechanisms such as diversity injection and entire mutation. Additionally, the attack employs individual learning based on the hill climbing algorithm, making the overall approach a memetic algorithm.

The project was created to fulfill the requirements of the university course Practical Cryptography in the Computer Science degree. The project received a grade of 5.0 on a scale from 2 to 5.

## Used Tools

- Python 3.13.2
- Numpy 2.2.4

## Requirements

For running the application you need:

- [Anaconda or Miniconda](https://www.anaconda.com/download/success)

## How to run

1. Execute command `git clone https://github.com/Ilvondir/bifid-cipher`.
2. Install Python and required packages by conda `conda env create -f environment.yml`.
