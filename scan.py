import json
import math
import os
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pprint import pprint

from bs4 import BeautifulSoup
from pyspark import SparkConf, SparkContext

import preprocess
from index import Tokenizer


def arguments():
    options = ArgumentParser(description='Run a sequential scan style search of documents')
    required = options.add_argument_group('required arguments')
    required.add_argument('-i', '--index', help='The pseudo-index created with preprocess.py.', required=True)
    required.add_argument('-q', '--queries', help='A file containing all queries to be run.', required=True)
    options.add_argument('-f', '--field', action='append', help='A field to include in the index.', default=('text',))
    options.add_argument('-l', '--limit', type=int, help='The number of documents to return scores for.', default=1000)
    return options.parse_args()


def load_queries(queries_file, file_format=None):
    """
    Load a raw queries file, inferring file type if necessary.
    :param queries_file: The path to the raw queries file.
    :param file_format: The format of the queries, either "json" or "title".
    :return: A dictionary of {title: [term1, ..., termN]}
    """
    with open(queries_file) as f:
        queries_raw = f.read().strip()

    if file_format is None:
        if queries_file.lower().endswith('.json'):
            file_format = 'json'
        else:
            file_format = 'title'

    return _parse_json_queries(queries_raw) if file_format == 'json' else _parse_title_queries(queries_raw)


def _parse_json_queries(queries_raw):
    queries_json = json.loads(queries_raw)
    return {query['title']: query['text'].strip().lower().split() for query in queries_json['queries']}


def _parse_title_queries(queries_raw):
    queries_soup = BeautifulSoup(queries_raw, 'html.parser')  # not actually HTML, but it works fine
    return {query.number.text.strip(): query.find('text').text.strip().lower().split() for query in
            queries_soup.find_all('query')}


def all_query_terms(queries):
    return set([term for q in queries for term in queries[q]])


def dirichlet_scorer(queries, term_collection_probabilities, mu=2500):
    def _with_document_vector(document):
        document_vector = document

        # Document length is the sum of the counts of the individual terms in the document
        document_length = sum(document_vector.values())

        # We will collect the probability of each query given this document
        query_doc_scores = []
        for query in queries:
            query_terms = queries[query]

            # Compute the score for this query
            q_norm = 1 / len(query_terms)
            query_prob = q_norm * sum([math.log((document_vector[term] + mu * term_collection_probabilities[term]) /
                                                (document_length + mu)) for term in query_terms])

            query_doc_scores.append((query, query_prob))

        return query_doc_scores

    return _with_document_vector


def trec_format(results, run_name='submitted'):
    """
    Convert the results to a very long string in TREC format.
    :param results: A {query: [(docno, score), ...]} dictionary.
    :return: A string representation of the results in TREC format.
    """
    trec = []
    for query in results:
        query_list = sorted(results[query], key=lambda doc_score: -doc_score[1])
        for i, doc_score in enumerate(query_list):
            trec.append('{query} Q0 {docno} {rank} {score} {run}'.format(query=query, docno=doc_score[0],
                                                                         rank=str(i+1), score=str(doc_score[1]),
                                                                         run=run_name))
    return '\n'.join(trec)


if __name__ == '__main__':
    args = arguments()

    queries = load_queries(args.queries)

    sc = SparkContext(conf=SparkConf().setAppName('convert'))

    collection_stats = sc.sequenceFile(os.path.join(args.index, preprocess.COLLECTION_STATS))
    collection_total_terms = collection_stats.collectAsMap()[preprocess.TOTAL_TERMS]

    collection_term_counts = sc.sequenceFile(os.path.join(args.index, preprocess.COLLECTION_TERM_COUNTS))

    def _default_prob():
        return 1 / collection_total_terms

    # Assume that the total number of query terms is relatively manageable since we need just one value for each term
    query_term_collection_probabilities = defaultdict(_default_prob)
    query_term_collection_probabilities.update(collection_term_counts
                                               .filter(lambda term_count: term_count[0] in all_query_terms(queries))
                                               .map(lambda term_count: (term_count[0], term_count[1] /
                                                                        collection_total_terms))
                                               .collectAsMap())

    scorer = dirichlet_scorer(queries, query_term_collection_probabilities)

    tokenizer = Tokenizer()

    def _field_terms(doc):
        docno = doc.docno.text.upper()
        counter = Counter()
        for field in args.field:
            for section in doc.find_all(field):
                tokens = tokenizer.process(section.text.strip())
                counter.update(tokens)
        return docno, counter

    docs = sc.textFile(os.path.join(args.index, preprocess.FLATTENED_DOCS))

    # groupByKey throwing error
    results = docs \
        .map(lambda doc: BeautifulSoup(doc, 'html.parser')) \
        .map(_field_terms) \
        .flatMapValues(scorer) \
        .map(lambda x: (x[1][0], x[0], x[1][1])) \
        .groupBy(lambda x: x[0]) \
        .mapValues(lambda document_scores: sorted(document_scores, key=lambda ds: -ds[1])[:args.limit]) \
        .flatMapValues(lambda ds: (ds.docno, ds.score)) \
        .collect()
        #.takeOrdered(args.limit, key=lambda d: -d[1])

    trec_results = trec_format(results)

    sc.parallelize(trec_results.split('\n')).saveAsTextFile('ap-results')
