import json

from bs4 import BeautifulSoup


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
    return {query['title']: [term for term in query['text'].strip().lower().split() if term] for query in queries_json[
        'queries']}


def _parse_title_queries(queries_raw):
    queries_soup = BeautifulSoup(queries_raw, 'html.parser')  # not actually HTML, but it works fine
    return {query.number.text.strip(): [term for term in query.find('text').text.strip().lower().split() if term] for
                                        query in queries_soup.find_all('query')}