import os
from argparse import ArgumentParser
from collections import Counter
from operator import add

from bs4 import BeautifulSoup
from pyspark import SparkContext, SparkConf

from index import Tokenizer


def arguments():
    options = ArgumentParser(description='Preprocess a dataset before querying.')
    required = options.add_argument_group('required arguments')
    required.add_argument('-d', '--data-dir', help='The data directory containing all document files.', required=True)
    required.add_argument('-o', '--out-dir', help='The output directory for all index files.', required=True)
    options.add_argument('-f', '--field', action='append', help='A field to include in the index.', default=('text',))
    options.add_argument('-p', '--partitions', type=int, help='The minimum number of partitions to use.', default=None)
    return options.parse_args()


FLATTENED_DOCS = 'flat'
COLLECTION_TERM_COUNTS = 'collection-term-counts'
COLLECTION_STATS = 'collection-stats'

TOTAL_TERMS = 'total_terms'
NUM_DOCS = 'num_docs'
UNIQUE_TERMS = 'unique_terms'

if __name__ == '__main__':
    args = arguments()
    sc = SparkContext(conf=SparkConf().setAppName('indexer'))

    data_files = sc.wholeTextFiles(args.data_dir, minPartitions=args.partitions)
    data_files \
        .flatMap(lambda whole_doc: BeautifulSoup(whole_doc[1], 'html.parser').find_all('doc')) \
        .map(lambda doc: str(doc).replace('\n', ' ')) \
        .saveAsTextFile(os.path.join(args.out_dir, FLATTENED_DOCS))

    tokenizer = Tokenizer()

    def _doc_terms(doc):
        counter = Counter()
        for field in args.field:
            for section in doc.find_all(field):
                tokens = tokenizer.process(section.text.strip())
                counter.update(tokens)
        return [(token, counter[token]) for token in counter]

    docs = sc.textFile(os.path.join(args.out_dir, FLATTENED_DOCS))

    term_counts = docs \
        .map(lambda doc: BeautifulSoup(doc, 'html.parser')) \
        .flatMap(_doc_terms) \
        .reduceByKey(add) \

    term_counts.saveAsSequenceFile(os.path.join(args.out_dir, COLLECTION_TERM_COUNTS))

    total_terms = term_counts \
        .map(lambda tc: tc[1]) \
        .reduce(lambda a, b: a + b)
    num_docs = docs.count()
    unique_terms = term_counts.count()

    sc.parallelize([
        (TOTAL_TERMS, total_terms),
        (NUM_DOCS, num_docs),
        (UNIQUE_TERMS, unique_terms),
    ]).saveAsSequenceFile(os.path.join(args.out_dir, COLLECTION_STATS))
