import unittest

from astuner.utils.config_utils import prepare_experiment_config


class TestConfigUtils(unittest.TestCase):
    def test_load_config(self):
        """A simple test to check if the configuration file is loaded without errors."""
        yaml_backup_dst, exp_base, exp_name, config = prepare_experiment_config(
            "tests/data/config.yaml", "tests/temp", backbone="debug"
        )
        self.assertEqual(exp_name, "sample")
        self.assertEqual(exp_base, "tests/temp/sample")
        self.assertEqual(yaml_backup_dst, "tests/temp/sample/yaml_backup.yaml")
        self.assertIn("astuner", config)
        self.assertIn("project_name", config["astuner"])
        self.assertEqual(config["astuner"]["project_name"], "unittest")
        self.assertIn("experiment_name", config["astuner"])
        self.assertEqual(config["astuner"]["experiment_name"], "sample")
        self.assertIn("task_reader", config["astuner"])
