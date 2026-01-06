import json
import unittest

from ajet.utils.dynamic_import import dynamic_import


class TestDynamicImport(unittest.TestCase):
    def test_successful_import(self):
        cls = dynamic_import("json.decoder->JSONDecoder")
        self.assertIs(cls, json.decoder.JSONDecoder)
