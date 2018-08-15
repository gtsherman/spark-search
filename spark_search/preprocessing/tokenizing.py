import re
import string

import stemming.porter2


class Tokenizer(object):
    def __init__(self, delimiters=(), transformations=()):
        if not delimiters:
            delimiters = ['\s'] + [p for p in string.punctuation.replace('-', '')]
        self._split_regex = '[{}]+'.format(''.join(delimiters))
        self._transformations = (self.lower, self.strip) if not transformations else transformations

    def tokenize(self, text):
        return re.split(self._split_regex, text.strip())

    def transform(self, tokens):
        for transformation in self._transformations:
            tokens = [transformation(token) for token in tokens if token]
        return tokens

    def process(self, text):
        return self.transform(self.tokenize(text))

    @staticmethod
    def lower(token):
        return token.lower()

    @staticmethod
    def strip(token):
        return token.strip()

    @staticmethod
    def stem(token, stemmer=stemming.porter2):
        return stemmer.stem(token)

    @staticmethod
    def stop(token, stoplist):
        return token if stoplist.is_stopword(token) else ''
