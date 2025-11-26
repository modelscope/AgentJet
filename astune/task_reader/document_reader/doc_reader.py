import uuid
from typing import Dict, List
from astune.schema.document import Document
from astune.task_reader.document_reader.document_reader_base import DocReaderBase
from unstructured.partition.auto import partition

class DocReader(DocReaderBase):
    """
    Document reader parses them into Document objects, and optionally returns raw text content.
    """
    def __init__(self, config):
        super().__init__(config)


    def load_document(self, source: str, languages=['eng']) -> str:
        """
        Load text from a file.

        Args:
            source (str): Path to the file.

        Returns:
            str: Merged text from all pages.
        """
        text_pages = []
        print(f"reading document from {source} using languages={languages}...")
        text_pages = partition(source, languages=languages)

        if not text_pages:
            raise ValueError(f"No extractable text found in file: {source}")

        text = "\n\n".join([str(sub) for sub in text_pages])
        return text


    def _parser_document(self, raw_document: str) -> Document:

        return Document(
            doc_id=str(uuid.uuid4()),
            content=raw_document,
            metadata={}
        )


    def get_document(self) -> Document:
        file_path = self.config.astune.document_reader.document_path
        raw_doc = self.load_document(file_path, languages=list(self.config.astune.document_reader.languages))
        return self._parser_document(raw_doc)

