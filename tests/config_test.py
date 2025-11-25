import unittest

from astune.utils.config_utils import prepare_experiment_config

class TestConfigUtils(unittest.TestCase):

    def test_load_config(self):
        """A simple test to check if the configuration file is loaded without errors."""
        yaml_backup_dst, exp_base, exp_name, config = prepare_experiment_config('tests/data/config.yaml', "tests/temp", backbone="debug")
        self.assertEqual(exp_name, "unittest")
        self.assertEqual(exp_base, "tests/temp/unittest")
        self.assertEqual(yaml_backup_dst, "tests/temp/unittest/yaml_backup.yaml")
        self.assertIn("astune", config)
        self.assertIn("experiment_name", config["astune"])
        self.assertEqual(config["astune"]["experiment_name"], "unittest")
        self.assertIn("task_reader", config["astune"])
