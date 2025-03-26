from math import log10

class NGramScorer:
    def __init__(self, ngramfile='english_2grams.csv', sep=','):
        self.ngrams = {}

        with open(f'datasets/{ngramfile}', 'r') as file:
            lines = file.readlines()

        for line in lines:
            key, count = line.split(sep)
            self.ngrams[key.upper()] = int(count)

        self.L = len(key)
        self.N = sum(self.ngrams.values())

        for key in self.ngrams.keys():
            self.ngrams[key] = log10(float(self.ngrams[key]) / self.N)

        self.floor = log10(0.01 / self.N)


    def score(self, text):
        score = 0
        ngrams = self.ngrams.__getitem__

        for i in range(len(text) - self.L + 1):
            if text[i:i + self.L] in self.ngrams:
                score += ngrams(text[i:i + self.L])
            else:
                score += self.floor
        return score