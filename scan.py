import importlib
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

from pyspark import SparkConf, SparkContext

import preprocess
from spark_search.querying.doc_handling import DirichletDocumentHandler
from spark_search.querying.queries import load_queries


def arguments():
    required = ['index', 'queries', 'out']

    options = ArgumentParser(description='Run a sequential scan style search of documents')
    options.add_argument('config', help='The configuration file.')
    options.add_argument('-i', '--index', help='The pseudo-index created with preprocess.py.')
    options.add_argument('-q', '--queries', help='A file containing all queries to be run.')
    options.add_argument('-l', '--limit', type=int, help='The number of documents to return scores for.', default=1000)
    options.add_argument('-o', '--out', help='The file (local filesystem) to record search results in.')
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

    for required_setting in required:
        val = getattr(conf, required_setting)
        if val is None:
            raise RuntimeError('You must provide a value for {} in your configuration file or command line '
                               'parameters.'.format(required_setting))

    return conf


def trec_format(results, run_name='submitted'):
    """
    Convert the results to a very long string in TREC format.
    :param run_name: The run name. Default: "submitted"
    :param results: A {query: [(docno, score), ...]} dictionary.
    :return: A string representation of the results in TREC format.
    """
    trec = []
    for query in results:
        query_list = sorted(results[query], key=lambda doc_score: -doc_score[1])
        for i, doc_score in enumerate(query_list):
            trec.append('{query} Q0 {docno} {rank} {score} {run}'.format(query=query, docno=doc_score[0],
                                                                         rank=str(i + 1), score=str(doc_score[1]),
                                                                         run=run_name))
    return '\n'.join(trec)


if __name__ == '__main__':
    config = arguments()

    queries = load_queries(config.queries)
    limit = config.limit

    sc = SparkContext(conf=SparkConf().setAppName('convert'))

    try:
        doc_handler = config.doc_handler
    except AttributeError:
        doc_handler = DirichletDocumentHandler

    # Provide the document handler with any information we can glean from the index as well as the queries etc.
    doc_handler = doc_handler(sc, queries, config)

    process_document = doc_handler.document_processing()
    scorer = doc_handler.scorer()

    docs = sc.sequenceFile(os.path.join(config.index, preprocess.FLATTENED_DOCS))

    scored_docs = docs \
        .mapValues(process_document) \
        .flatMapValues(scorer.score) \
        .map(lambda x: (x[1][0], x[0], x[1][1])) \
        .groupBy(lambda x: x[0]) \
        .mapValues(lambda document_scores: sorted(document_scores, key=lambda ds: -ds[2])[:limit]) \
        .flatMapValues(lambda ds: [(t[0], t[1], t[2]) for t in ds]) \
        .map(lambda x: x[1]) \
        .collect()

    r = defaultdict(list)
    for result in scored_docs:
        query_title, returned = result[0], result[1:]
        r[query_title].append(returned)

    with open(config.out, 'w') as out:
        print(trec_format(r), file=out)
