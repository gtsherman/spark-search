import csv
import importlib
import os
import re
import string
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict

from bs4 import BeautifulSoup


class Output(object):
    def __init__(self, location):
        self._location = location


class CSVOutput(Output):
    def __init__(self, location):
        super().__init__(location)
        self._out_file = open(self._location, 'w')
        self._writer = csv.writer(self._out_file)

    def write(self, documents):
        for document in documents:
            for field in document.postings:
                for term in document.postings[field]:
                    count = document.postings[field][term]
                    doc_id = document.id
                    self._writer.writerow([term, field, count, doc_id])

    def close(self):
        self._out_file.close()


class Document(object):
    def __init__(self, id, stoplist=None):
        self.id = id
        self.postings = defaultdict(Counter)
        self.stoplist = stoplist

    def record(self, term, field=None):
        if term and (self.stoplist is None or (self.stoplist is not None and term not in self.stoplist)):
            self.postings[field][term] += 1


class Collection(object):
    def __init__(self, stoplist=None):
        self.docs = 0
        self.stoplist = stoplist
        if self.stoplist is not None:
            self.stoplist = stoplist.stopwords.strip().split()  # assume stoplist is a module with stopwords = """words"""

    def document(self):
        doc = Document(self.docs, self.stoplist)
        self.docs += 1
        return doc


class Indexer(object):
    def __init__(self, config):
        # Config
        try:
            stoplist = importlib.import_module(config.stoplist)
        except ImportError:
            sys.exit('There was a problem importing the stoplist. Please double check your path.')
        except AttributeError:
            stoplist = None
        self._collection = Collection(stoplist)
        try:
            self._tokenizer = config.tokenizer
        except AttributeError:
            self._tokenizer = Tokenizer()
        self._parser = config.parser(self._collection, self._tokenizer)
        self._data_locations = config.data_locations
        self._index_location = config.index_location
        self._index_out = config.output_format(self._index_location)

    def _parse_all(self, *paths):
        files = []
        for path in paths:
            if os.path.isdir(path):
                files += [os.path.join(root, file) for root, _, files in os.walk(path) for file in files]
            elif os.path.isfile(path):
                files += [path]

        # ensure no duplicate files
        files = set(files)

        for file in files:
            docs = self._parser.parse(file)
            self._index_out.write(docs)
        self._index_out.close()

    def index(self):
        self._parse_all(*self._data_locations)


class Tokenizer(object):
    def __init__(self, delimiters=[], transformations=[]):
        if not delimiters:
            delimiters = ['\s'] + [p for p in string.punctuation.replace('-', '')]
        self._split_regex = '[{}]+'.format(''.join(delimiters))

        if not transformations:
            self._transformations = [self.lower]

    def tokenize(self, text):
        return re.split(self._split_regex, text)

    def transform(self, tokens):
        for transformation in self._transformations:
            tokens = [transformation(token) for token in tokens]
        return tokens

    def process(self, text):
        return self.transform(self.tokenize(text))

    @staticmethod
    def lower(token):
        return token.lower()

    @staticmethod
    def stem(token):
        """
        TO DO
        :param token:
        :return: A stemmed token
        """
        return token


class Parser(object):
    def __init__(self, collection, tokenizer):
        self._collection = collection
        self._tokenizer = tokenizer


class TrecTextParser(Parser):
    def parse(self, file):
        with open(file, encoding='latin-1') as f:
            docs = BeautifulSoup('<docs>' + f.read() + '</docs>', 'html.parser')

        documents = []
        for doc in docs.find_all('doc'):
            document = self._collection.document()
            for field in doc.find_all():
                for token in self._tokenizer.process(field.text):
                    document.record(token, field=field.name)
            documents.append(document)

        return documents


class HtmlParser(Parser):
    pass


if __name__ == '__main__':
    options = ArgumentParser(description='Index raw data')
    required = options.add_argument_group('required arguments')
    required.add_argument('-c', '--config', help='Configuration file location.', default='config')
    options.add_argument('-d', '--data-locations', action='append', help='Locations of raw text data.')
    options.add_argument('-p', '--parser', choices=['trec', 'html'], help='The type of parser to use.')
    options.add_argument('-f', '--index-format', choices=['csv'], help='The index format to use.')
    options.add_argument('-i', '--index-location', help='The index location.')
    options.add_argument('-v', '--verbose', action='store_true', help='Print verbose output.')
    options.add_argument('-s', '--stoplist', help='A python module containing a stoplist.')
    args = options.parse_args()

    # Load the configuration file
    try:
        config = importlib.import_module(args.config.replace('.py', ''))
    except ImportError:
        sys.exit('You must specify a configuration file.')

    # Update the config with command line arguments, if relevant
    for option in vars(args):
        if getattr(args, option) is not None:
            setattr(config, option, getattr(args, option))

    indexer = Indexer(config)
    indexer.index()
