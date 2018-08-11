import importlib
import os
import re
import string
import sys
from argparse import ArgumentParser
from collections import Counter
from operator import add

import stemming.porter2
from bs4 import BeautifulSoup
from pyspark import SparkContext, SparkConf

FLATTENED_DOCS = 'flat'
COLLECTION_TERM_COUNTS = 'collection-term-counts'
COLLECTION_STATS = 'collection-stats'
DOC_FORMAT = 'format'

TOTAL_TERMS = 'total_terms'
NUM_DOCS = 'num_docs'
UNIQUE_TERMS = 'unique_terms'


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


class Parser(object):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def split(self, all_documents):
        """
        Since this basic parser assumes that each file contains the entirety of a single document, we simply need to
        return the original (file_name, file_contents) RDD.
        :param all_documents: The (file_name, file_contents) RDD returned by wholeTextFiles
        :return: The same (file_name, file_contents) RDD
        """
        return all_documents

    def parse(self, document):
        """
        A plain text parser that assumes the contents of document is the literal text of the document.
        :param document: A string of document text.
        :return: The tokenized and processed document text.
        """
        tokenizer = self._tokenizer

        return tokenizer.process(document)


class TrecTextParser(Parser):
    def split(self, all_documents):
        """
        :param all_documents: The (file_name, file_contents) RDD returned by wholeTextFiles
        :return: A (docno, doctext) RDD
        """
        return all_documents \
            .flatMap(lambda whole_doc: BeautifulSoup(whole_doc[1], 'html.parser').find_all('doc')) \
            .map(lambda doc: (doc.docno.text.strip().upper(), str(doc).replace('\n', ' ')))

    @staticmethod
    def extract(document):
        return BeautifulSoup(document, 'html.parser').find_all('text')

    def parse(self, document):
        """
        :param document: A raw trectext document string for a single document.
        :return: The tokenized and processed document text.
        """
        tokenizer = self._tokenizer

        tokens = []
        for extent in self.extract(document):
            try:
                tokens += tokenizer.process(extent.text)
            except AttributeError:  # is not a Tag object (e.g. NavigableString)
                pass
        return tokens


class TrecWebParser(TrecTextParser):
    @staticmethod
    def extract(document):
        return BeautifulSoup(document, 'html.parser').dochdr.next_siblings


parsers = {
    'trectext': TrecTextParser,
    'trecweb': TrecWebParser,
    'warc': None,
    'text': Parser,
}


def arguments():
    required = ['data_dir', 'index', 'doc_format']
    options = ArgumentParser(description='Preprocess a dataset before querying.')
    options.add_argument('config')
    options.add_argument('-d', '--data-dir', help='The data directory containing all document files.')
    options.add_argument('-i', '--index', help='The output directory for all index files.')
    options.add_argument('-f', '--doc-format', choices=parsers.keys(), help='The format of the input document.')
    options.add_argument('-p', '--partitions', type=int, help='The minimum number of partitions to use.', default=None)
    args = options.parse_args()

    # Load the configuration file
    try:
        conf = importlib.import_module(args.config.replace('.py', ''))
    except ImportError:
        sys.exit('You must specify a configuration file.')

    # Update the config with command line arguments, if relevant
    for option in vars(args):
        if getattr(args, option) is not None:
            setattr(conf, option, getattr(args, option))
        setattr(conf, 'partitions', args.partitions)  # because None is a valid option

    for required_setting in required:
        val = getattr(conf, required_setting)
        if val is None:
            raise RuntimeError('You must provide a value for {} in your configuration or command line '
                               'parameters.'.format(required_setting))

    return conf


if __name__ == '__main__':
    config = arguments()

    parser = parsers[config.doc_format](Tokenizer())

    sc = SparkContext(conf=SparkConf().setAppName('indexer'))

    # Break the whole text files into individual document objects
    data_files = sc.wholeTextFiles(config.data_dir, minPartitions=config.partitions)
    docs = parser.split(data_files).cache()

    # Save the document as a flattened (docno, doctext) SequenceFile for more efficient reading later
    docs.saveAsSequenceFile(os.path.join(config.index, FLATTENED_DOCS))

    # Count the frequency of each term in the full collection
    term_counts = docs \
        .map(lambda docno_doc: docno_doc[1]) \
        .map(parser.parse) \
        .map(lambda terms: Counter(terms)) \
        .flatMap(lambda term_counts: [(term, term_counts[term]) for term in term_counts]) \
        .reduceByKey(add) \
        .cache()

    term_counts.saveAsSequenceFile(os.path.join(config.index, COLLECTION_TERM_COUNTS))

    total_terms = term_counts \
        .map(lambda tc: tc[1]) \
        .reduce(lambda a, b: a + b)
    num_docs = docs.count()
    unique_terms = term_counts.count()

    sc.parallelize([
        (TOTAL_TERMS, total_terms),
        (NUM_DOCS, num_docs),
        (UNIQUE_TERMS, unique_terms),
    ]).saveAsSequenceFile(os.path.join(config.index, COLLECTION_STATS))

    sc.parallelize([('format', config.doc_format)]).saveAsSequenceFile(os.path.join(config.index, DOC_FORMAT))
