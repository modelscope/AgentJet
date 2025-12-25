import json
import unittest

from agentscope_tuner.utils.dynamic_import import dynamic_import


class TestDynamicImport(unittest.TestCase):
    def test_successful_import(self):
        cls = dynamic_import("json.decoder->JSONDecoder")
        self.assertIs(cls, json.decoder.JSONDecoder)

    def test_successful_import2(self):
        cls = dynamic_import("tutorial.example_math_agent.math_agent->ExampleMathLearn")
        self.assertEqual(
            str(cls), "<class 'tutorial.example_math_agent.math_agent.ExampleMathLearn'>"
        )

    def test_successful_import3(self):
        cls = dynamic_import("tutorial/example_math_agent/math_agent.py->ExampleMathLearn")
        self.assertEqual(str(cls), "<class 'math_agent.ExampleMathLearn'>")

    def test_successful_import4(self):
        cls = dynamic_import("tests/data/example_cls.py->ExampleClass")
        self.assertEqual(str(cls), "<class 'example_cls.ExampleClass'>")
        aid = str(cls().value)
        bid = str(cls().value)
        self.assertEqual(aid, bid)
        cls = dynamic_import("tests/data/example_cls.py->ExampleClass")
        cid = str(cls().value)
        self.assertEqual(aid, cid)

    def test_successful_import5(self):
        # if this fails, remove venv site-package `tests` from sys.path and try again
        cls = dynamic_import("tests.data.example_cls->ExampleClass")
        self.assertEqual(str(cls), "<class 'tests.data.example_cls.ExampleClass'>")
        aid = str(cls().value)
        bid = str(cls().value)
        self.assertEqual(aid, bid)
        cls = dynamic_import("tests.data.example_cls->ExampleClass")
        cid = str(cls().value)
        self.assertEqual(aid, cid)

    def test_invalid_format_raises_value_error(self):
        with self.assertRaises(ValueError):
            dynamic_import("json.decoder.JSONDecoder")

    def test_missing_class_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            dynamic_import("json.decoder->MissingClass")
