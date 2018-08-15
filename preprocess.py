import importlib
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from operator import add

from pyspark import SparkContext, SparkConf

from spark_search.preprocessing.parsing import parsers
from spark_search.preprocessing.tokenizing import Tokenizer


FLATTENED_DOCS = 'flat'
COLLECTION_TERM_COUNTS = 'collection-term-counts'
COLLECTION_STATS = 'collection-stats'
DOC_FORMAT = 'format'

TOTAL_TERMS = 'total_terms'
NUM_DOCS = 'num_docs'
UNIQUE_TERMS = 'unique_terms'


def arguments():
    required = ['data_dir', 'index', 'doc_format']
    options = ArgumentParser(description='Preprocess a dataset before querying.')
    options.add_argument('config')
    options.add_argument('-d', '--data-dir', help='The data directory containing all document files.')
    options.add_argument('-i', '--index', help='The output directory for all index files.')
    options.add_argument('-f', '--doc-format', choices=parsers.keys(), help='The format of the input document.')
    options.add_argument('-p', '--partitions', type=int, help='The minimum number of partitions to use.', default=None)
    options.add_argument('--full-docs', action='store_true', help='Stores the full document representation. The '
                                                                  'default behavior is reduce documents to a '
                                                                  'dictionary of term counts.')
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

    if config.full_docs:
        # Save the document as a flattened (docno, doctext) SequenceFile for more efficient reading later
        docs.saveAsSequenceFile(os.path.join(config.index, FLATTENED_DOCS))

    # Count the frequency of each term in the full collection
    doc_vectors = docs \
        .mapValues(parser.parse) \
        .mapValues(lambda terms: dict(Counter(terms))) \
        .cache()

    if not config.full_docs:
        # Save the document as a bag of words (docno, {term: count, term2: count2, ...}) SequenceFile.
        # This option can lead to significantly faster searches, but throws away structural information about the
        # document.
        doc_vectors.saveAsSequenceFile(os.path.join(config.index, FLATTENED_DOCS))

        # We also need to adjust the format to store for this index, since the documents are now pre-parsed rather
        # than needing parsing at search time.
        config.doc_format = 'preparsed'

    term_counts = doc_vectors \
        .map(lambda doc: doc[1]) \
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
