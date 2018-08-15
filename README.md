# Spark Search

An exercise in Apache Spark and information retrieval.

## Overview

Spark Search uses Apache Spark to conduct information retrieval research experiments. Instead of the traditional querying-an-index approach, the system performs some preprocessing of documents and then scores a set of queries against each document individually and in parallel. This ensures that each document in the collection will receive a score. It also simplifies thinking about the scoring process in a MapReduce environment.

## Configuration

Spark Search can be configured using configuration files. These are python files containing any settings that need to be passed to the preprocess or scan scripts. For example, a configuration file for preprocessing might look like the following:

```python
data_dir = '/path/to/data'
index = '/index/location'
doc_format = 'trectext'
```

Some options are also available on the command line. Use the `-h` option to see which are available. Note that if an argument is specified in both the configuration file and the command line, the command line option will be used. However, a configuration file is always required, even if all options are passed via command line.

## Preprocessing

Instead of indexing documents, Spark Search performs a few simple preprocessing steps designed to speed up the later querying stage. These steps include:

- Constructing a list of `(term, count)` tuples where `count` is the frequency of `term` across all documents in the collection.
- Computing collection-level statistics such as the total number of words, the total number of unique words, and the total number of documents.
- Splitting files that contain multiple documents into individual documents and collapsing those documents to fit onto one line, which enables use of Spark's `textFile()`.
- Optionally, storing documents as `{term: count}` dictionaries to reduce the parsing required at query time, at the cost of losing document structure.

To preprocess a document collection, you might use the following command:

```
spark-submit --py-files python.zip preprocess.py config.py -p 30
```

Spark Search supports document collections in plain text, TrecText, and TrecWeb formats at present. If you will want access to the full document structure at query time, you should pass the `--full-docs` flag as part of the command above. TrecText documents will be stored as the full `<DOC>` element. TrecWeb documents will be stored only as the portion of the document containing the contents (i.e., the portion after the `<DOCHDR>` element).

You should consider `spark-submit` options that are most appropriate for your system and data.

## Querying

Spark Search distributes the queries across jobs and computes document scores for each query, which are later grouped by query and ranked.

Querying is performed with the `scan.py` script. The default behavior is to run a Dirichlet-smoothed query likelihood model, but it is fairly straightforward to customize the scoring process. First, let's see how to run a default query batch:

```
spark-submit --py-files python.zip scan.py config.py -o results
```

This assumes that the index and query files are specified in `config.py`. Note that query files may be in json or TREC title formats.

### Custom Querying

The querying process is managed by two classes: `DocumentHandler` and `Scorer`.

#### `DocumentHandler`

The `DocumentHandler` class is responsible for connecting the raw document data with the `Scorer`. By default, all `DocumentHandler` classes receive a `SparkContext` object, the queries, and the user configuration. They automatically load collection statistics and make them available as instance variables.

Because the `DocumentHandler` is responsible for fetching data, processing the document, and assigning a scorer, it is an important class for custom systems to override. A breakdown of the most useful methods follows.

- `prepare_data()`

The `prepare_data()` method is called automatically during the initialization of any `DocumentHandler`. By default, this method does nothing, but it is a useful method to use in fetching data (perhaps using `self.spark_context`) or pre-computing figures. For example, one method for removing stop words from a query is to override the `prepare_data()` method:

```python
class StoppedQueryDirichletDocumentHandler(DirichletDocumentHandler):
  def prepare_data(self):
    stoplist = IndriStoplist()
    stopped_queries = {}
    for query in self.queries:
      stopped_queries[query] = [term for term in self.queries[query] if not stoplist.is_stopword(term)]
    self.queries = stopped_queries
```

- `document_processing()`

A required method for each `DocumentHandler` is the `document_processing()` method. This method should return the result of `document_process_pipeline()`, called with a variable number of arguments, each of which is a function that will be applied in turn to the document as read from the "index." For example, the default `LanguageModelDocumentHandler` will call the `parse()` method of the appropriate `Parser`, and apply `Counter` to the resulting token list.

- `scorer()`

The `scorer()` method returns an instance of the `Scorer` to be used to score documents. This method should be overriden when a custom `Scorer` is required or when inheriting from a subclass of `DocumentHandler` that does not implement the `scorer()` method.

#### `Scorer`

All `Scorer` implementations are instantiated with a dictionary of form `{query_name: [query_term1, ..., query_termN]}`. Each implementation is required to define a `score()` function that takes a document argument and returns a list of `(query_name, query_doc_probability)` tuples, one per query.

Some scorers may require different non-document data, and all scorers will likely expect a specific document format. It is the job of the `DocumentHandler` to ensure that these requirements are met. For example, the `LanguageModelDocumentHandler` ensures that term collection probabilities, P(w|C), are made available to any scorers; similarly, its definition of `document_processing()` ensures that documents are provided as dictionaries of term counts, which are expected by the default `DirichletScorer` and `JelinekMercerScorer` implementations.

It is therefore usually the case that a custom `DocumentHandler` and custom `Scorer` will need to be developed in conjunction with one another. The former will prepare the document and compute and provide any necessary data to the latter.

### Configuration Revisited

Since the required configuration file is actually just python code, custom `DocumentHandler`s and `Scorer`s can be defined within the configuration file if desired. This may aid in rapid prototyping. Similarly, because the configuration file is python code and its properties are passed to the `DocumentHandler`, this is a good place for creatively incorporating extra data, new function definitions, etc. which can then be accessed within your custom `DocumentHandler` definition via its `self.config` property.

If classes are defined in a separate file, be sure to import them into your configuration file, since the `DocumentHandler` may only be specified in the configuration file (not on the command line). A simple example follows:

```python
from preprocess import JelinekMercerDocumentHandler

doc_handler = JelinekMercerDocumentHandler
```

Notice that the `doc_handler` is specified as the class name rather than assigned an instance of the class. This is because the `SparkContext` needs to be passed within `scan.py` to prevent issues with nested `SparkContext`s. In contrast, a `Stoplist` is self-contained and does not need to be instantiated within `scan.py`. It should therefore be instantiated as part of the configuration file:

```python
stoplist = IndriStoplist()
```

Remember to import `IndriStoplist` into your configuration file as well.
