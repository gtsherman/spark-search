import math
from collections import defaultdict


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