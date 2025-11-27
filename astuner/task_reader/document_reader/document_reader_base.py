from astuner.schema.document import Document


class DocReaderBase:
    def __init__(self, config):
        self.config = config

    def get_document(self) -> Document:
        raise NotImplementedError
