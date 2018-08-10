import importlib
import json
import math
import os
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict

from bs4 import BeautifulSoup
from pyspark import SparkConf, SparkContext

import preprocess


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


def document_process_pipeline(*transformations):
    def _transform(document):
        for transformation in transformations:
            document = transformation(document)
        return document
    return _transform


class DocumentHandler(object):
    def __init__(self, spark_context, queries, config, **kwargs):
        self.spark_context = spark_context
        self.queries = queries
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

        self.collection_stats = spark_context.sequenceFile(os.path.join(config.index, preprocess.COLLECTION_STATS))
        self.collection_total_terms = self.collection_stats.collectAsMap()[preprocess.TOTAL_TERMS]

        self.collection_term_counts = spark_context.sequenceFile(os.path.join(config.index, preprocess.COLLECTION_TERM_COUNTS))

        self.document_format = spark_context.sequenceFile(os.path.join(config.index, preprocess.DOC_FORMAT)).collectAsMap()[
            'format']

    @staticmethod
    def all_query_terms(queries):
        return set([term for q in queries for term in queries[q]])


class LanguageModelDocumentHandler(DocumentHandler):
    def document_processing(self):
        return document_process_pipeline(preprocess.parsers[self.document_format](preprocess.Tokenizer()).parse, Counter)

    def _collection_probs(self):
        collection_total_terms = self.collection_total_terms
        all_query_terms = self.all_query_terms
        queries = self.queries

        return self.collection_term_counts \
            .filter(lambda term_count: term_count[0] in all_query_terms(queries)) \
            .map(lambda term_count: (term_count[0], term_count[1] / collection_total_terms)) \
            .collectAsMap()


class DirichletDocumentHandler(LanguageModelDocumentHandler):
    def scorer(self):
        query_term_collection_probabilities = self._collection_probs()
        return DirichletScorer(self.queries, self.collection_total_terms, query_term_collection_probabilities)


class JelinekMercerDocumentHandler(LanguageModelDocumentHandler):
    def scorer(self):
        query_term_collection_probabilities = self._collection_probs()
        return JelinekMercerScorer(self.queries, self.collection_total_terms, query_term_collection_probabilities)


class Scorer(object):
    def __init__(self, queries):
        self.queries = queries


class LanguageModelScorer(Scorer):
    def __init__(self, queries, collection_total_terms=1, term_collection_probabilities=None):
        super().__init__(queries)
        self.term_collection_probabilities = term_collection_probabilities
        self.collection_total_terms = collection_total_terms

    def _default_prob(self):
        collection_total_terms = self.collection_total_terms
        return 1 / collection_total_terms


class JelinekMercerScorer(LanguageModelScorer):
    def __init__(self, queries, collection_total_terms=1, term_collection_probabilities=None, orig_weight=0.5):
        super().__init__(queries, collection_total_terms, term_collection_probabilities)
        self.orig_weight = orig_weight

    def score(self, document):
        queries = self.queries
        term_collection_probabilities = self.term_collection_probabilities
        orig_weight = self.orig_weight

        document_vector = document

        # Term probabilities default to 0 if none are provided
        if term_collection_probabilities is None:
            term_collection_probabilities = defaultdict(int)
        else:
            term_collection_probabilities = defaultdict(self._default_prob, term_collection_probabilities)

        # Document length is the sum of the counts of the individual terms in the document
        document_length = sum(document_vector.values())

        # We will collect the probability of each query given this document
        query_doc_scores = []
        for query in queries:
            query_terms = queries[query]

            # Compute the score for this query
            q_norm = 1 / len(query_terms)
            query_prob = q_norm * sum([math.log(orig_weight * ((document_vector[term]) / (document_length + 1)) +
                                                (1 - orig_weight) * term_collection_probabilities[term])
                                       for term in query_terms])

            query_doc_scores.append((query, query_prob))

        return query_doc_scores


class DirichletScorer(LanguageModelScorer):
    def __init__(self, queries, collection_total_terms=1, term_collection_probabilities=None, mu=2500):
        super().__init__(queries, collection_total_terms, term_collection_probabilities)
        self.mu = mu

    def score(self, document):
        """
        Scores a list of queries against a document using Dirichlet-smoothed query likelihood scoring.
        :param document: A dictionary of type {term: term_count}
        :return: A list of (query_title, query_probability) tuples for each query provided.
        """

        # A weird little hack that makes Spark processing more efficient by allowing the function to be shipped
        # without the remainder of the scorer object.
        queries = self.queries
        term_collection_probabilities = self.term_collection_probabilities
        mu = self.mu

        document_vector = document

        # Term probabilities default to 0 if none are provided
        if term_collection_probabilities is None:
            term_collection_probabilities = defaultdict(int)
        else:
            term_collection_probabilities = defaultdict(self._default_prob, term_collection_probabilities)

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
