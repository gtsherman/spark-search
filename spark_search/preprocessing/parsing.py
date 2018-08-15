from bs4 import BeautifulSoup


class Parser(object):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def split(self, all_documents):
        """
        Since this basic parser assumes that each file contains the entirety of a single document, we simply need to
        return the original (file_name, file_contents) RDD.
        :param all_documents: The (file_name, file_contents) RDD returned by wholeTextFiles
        :return: The same (file_name, file_contents) RDD
        """
        return all_documents

    def parse(self, document):
        """
        A plain text parser that assumes the contents of document is the literal text of the document.
        :param document: A string of document text.
        :return: The tokenized and processed document text.
        """
        tokenizer = self._tokenizer

        return tokenizer.process(document)


class TrecTextParser(Parser):
    def split(self, all_documents):
        """
        :param all_documents: The (file_name, file_contents) RDD returned by wholeTextFiles
        :return: A (docno, doctext) RDD

        """
        return all_documents \
            .flatMap(lambda whole_doc: BeautifulSoup(whole_doc[1], 'html.parser').find_all('doc')) \
            .map(lambda doc: (doc.docno.text.strip().upper(), str(doc).replace('\n', ' ')))

    @staticmethod
    def extract(document):
        return BeautifulSoup(document, 'html.parser').find_all('text')

    def parse(self, document):
        """
        :param document: A raw trectext document string for a single document.
        :return: The tokenized and processed document text.
        """
        tokenizer = self._tokenizer

        tokens = []
        for extent in self.extract(document):
            try:
                tokens += tokenizer.process(extent.text)
            except AttributeError:  # is not a Tag object (e.g. NavigableString)
                pass
        return tokens


class TrecWebParser(TrecTextParser):
    def _docno(self, doc):
        docno_start_tag = '<DOCNO>'
        docno_end_tag = '</DOCNO>'

        docno_start = doc.index(docno_start_tag) + len(docno_start_tag)
        docno_end = doc.index(docno_end_tag)

        return doc[docno_start:docno_end].strip().upper()

    def _contents(self, doc):
        dochdr_end_tag = '</DOCHDR>'

        contents_start = doc.index(dochdr_end_tag) + len(dochdr_end_tag)

        return doc[contents_start:]

    def _flattened(self, doc):
        return self._docno(doc), self._contents(doc)

    def split(self, all_documents):
        """
        :param all_documents: The (file_name, file_contents) RDD returned by wholeTextFiles
        :return: A (docno, doctext) RDD

        """
        return all_documents \
            .flatMap(lambda whole_doc: whole_doc[1].replace('</DOC>', '').split('<DOC>')[1:]) \
            .map(self._flattened)

    @staticmethod
    def extract(document):
        return BeautifulSoup(document, 'lxml')


class ParsedDocumentParser(Parser):
    """
    Exists to alleviate the reading of documents that have been parsed as part of preprocessing. Rather than having
    to override the DocumentHandler, this Parser simply returns the preparsed document (most likely a dictionary of
    term counts, though other preprocessing techniques could certainly be implemented).
    """
    def parse(self, document):
        return document


parsers = {
    'trectext': TrecTextParser,
    'trecweb': TrecWebParser,
    'text': Parser,
    'preparsed': ParsedDocumentParser,
}
