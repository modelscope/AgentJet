from typing import List, Optional

from agentscope_tuner.schema.document import Document


class DocReaderBase:
    def __init__(self, config):
        self.config = config

    def get_document(self) -> Optional[List[Document]]:
        raise NotImplementedError
