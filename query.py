import importlib
from argparse import ArgumentParser

import sys
from pyspark.sql.functions import col, log, sum, upper
from pyspark.sql import SparkSession


class QueryResults(object):
    def __init__(self, results, query_name, run_name):
        """
        :param results: An iterable of (docno, score) tuples.
        """
        self._results = results
        self._query_name = query_name
        self._run_name = run_name

    def print(self, output_format=None):
        if output_format is None:
            output_format = self.trec

        for i, result in enumerate(self._results):
            print(output_format(result, i+1))

    def trec(self, result, rank):
        return '{query} Q0 {docno} {rank} {score} {run}'.format(query=self._query_name, docno=result[0],
                                                                rank=str(rank), score=str(result[1]),
                                                                run=self._run_name)

class QueryInterface(object):
    def __init__(self, index, fields=('text',)):
        spark = SparkSession.builder.appName('query').getOrCreate()

        self.index = spark.read.parquet(index)
        self.fields = fields

        self.doc_lengths = self.index \
            .filter(self.index.field.isin(*self.fields)) \
            .select('docid', 'count') \
            .groupBy('docid') \
            .sum('count') \
            .withColumnRenamed('sum(count)', 'length') \
            .cache()

        self.num_index_terms = self.doc_lengths \
            .select(sum(col('length')).alias('total')) \
            .collect()[0]['total']

        self.index_term_freqs = self.index \
            .filter(self.index.field.isin(*self.fields)) \
            .groupBy('term') \
            .sum('count') \
            .withColumnRenamed('sum(count)', 'freq')

        self.collection_probabilities = self.index_term_freqs \
            .withColumn('colProb', self.index_term_freqs.freq / self.num_index_terms) \
            .cache()

        self.docnos = self.index \
            .filter(self.index.field == 'docno') \
            .select(upper(col('term')).alias('docno'), 'docid') \
            .cache()

        self.num_docs = self.index \
            .select('docid') \
            .distinct() \
            .count()

    def query(self, *terms, mu=2500, limit=1000, query_name=None, run_name=None):
        query_col_probs = self.collection_probabilities \
            .filter(self.collection_probabilities.term.isin(*terms))

        term_document_grid = self.index \
            .filter(self.index.field.isin(*self.fields)) \
            .filter(self.index.term.isin(*terms)) \
            .select('docid') \
            .distinct() \
            .crossJoin(query_col_probs)

        ranked_documents = self.index \
            .filter(self.index.field.isin(*self.fields)) \
            .filter(self.index.term.isin(*terms)) \
            .join(term_document_grid, ['term', 'docid'], 'right_outer') \
            .fillna('text', 'field') \
            .fillna(0, 'count') \
            .join(self.doc_lengths, 'docid') \
            .withColumn('termProb', log((col('count') + mu * col('colProb')) / (col('length') + mu))) \
            .groupBy('docid') \
            .sum('termProb') \
            .withColumnRenamed('sum(termProb)', 'docProb') \
            .join(self.docnos, 'docid') \
            .select('docno', 'docProb') \
            .orderBy(col('docProb'), ascending=False) \
            .take(limit)

        return QueryResults([(result.docno, result.docProb) for result in ranked_documents], query_name, run_name)

    def query_batch(self, queries, mu=2500, limit=1000, run_name=None):
        return [self.query(queries[q].split(), mu=mu, limit=limit, query_name=q, run_name=run_name) for q in queries]


if __name__ == '__main__':
    options = ArgumentParser(description='Run queries.')
    options.add_argument('-c', '--config', help='Configuration file location.', default='config')
    options.add_argument('-i', '--index', help='Specify the index location in HDFS.')
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

    query_interface = QueryInterface(args.index)

    batch_results = query_interface.query_batch(config.queries, run_name='test')
    for results in batch_results:
        results.print()
