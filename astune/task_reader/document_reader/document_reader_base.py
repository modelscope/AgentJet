import json
import uuid
import datasets
from typing import List, Dict, Optional
from astune.schema.document import Document


class DocReaderBase:
    def __init__(self, config):
        self.config = config

    def get_document(self) -> Document:
        raise NotImplementedError
