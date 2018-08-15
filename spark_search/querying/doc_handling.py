import os
from collections import Counter

import preprocess
from spark_search.querying.scoring import JelinekMercerScorer, DirichletScorer


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
        self.config = config
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

        self.collection_stats = spark_context.sequenceFile(os.path.join(config.index, preprocess.COLLECTION_STATS))
        self.collection_total_terms = self.collection_stats.collectAsMap()[preprocess.TOTAL_TERMS]

        self.collection_term_counts = spark_context.sequenceFile(os.path.join(config.index,
                                                                              preprocess.COLLECTION_TERM_COUNTS))

        self.document_format = spark_context.sequenceFile(os.path.join(config.index, preprocess.DOC_FORMAT)) \
            .collectAsMap()['format']

        self.prepare_data()

    def prepare_data(self):
        """
        A general utility method to make data operations easier. When defining custom behavior, a class inheriting
        from DocumentHandler can override this method to do any preparation necessary before documents are processed.
        """
        pass

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