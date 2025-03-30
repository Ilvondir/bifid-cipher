from math import log10

class NGramScorer:
    def __init__(self, ngramfile='en_bigrams.csv', sep=','):
        self.ngrams = {}

        with open(f'datasets/{ngramfile}', 'r', encoding='UTF-8') as file:
            lines = file.readlines()

        for line in lines:
            key, count = line.split(sep)
            self.ngrams[key] = int(count)

        self.L = len(key)
        self.N = sum(self.ngrams.values())

        for key in self.ngrams.keys():
            self.ngrams[key] = log10(float(self.ngrams[key]) / self.N)

        self.floor = log10(0.01 / self.N)


    def score(self, text):
        ngrams = self.ngrams
        floor = self.floor
        L = self.L

        return sum(ngrams.get(text[i:i + L], floor) for i in range(len(text) - L + 1))