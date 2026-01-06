import tempfile
import unittest

import yaml

from ajet.utils.config_utils import (
    align_parameters,
    expand_astune_hierarchical_config,
    prepare_experiment_config,
    read_astune_hierarchical_config,
)


class TestConfigUtils(unittest.TestCase):
    def test_load_config(self):
        """A simple test to check if the configuration file is loaded without errors."""
        yaml_backup_dst, exp_base, exp_name, config = prepare_experiment_config(
            "tests/data/config.yaml", "tests/temp", backbone="debug"
        )
        self.assertEqual(exp_name, "sample")
        self.assertEqual(exp_base, "tests/temp/sample")
        self.assertEqual(yaml_backup_dst, "tests/temp/sample/yaml_backup.yaml")
        self.assertIn("ajet", config)
        self.assertIn("project_name", config["ajet"])
        self.assertEqual(config["ajet"]["project_name"], "unittest")
        self.assertIn("experiment_name", config["ajet"])
        self.assertEqual(config["ajet"]["experiment_name"], "sample")
        self.assertIn("task_reader", config["ajet"])

    def test_config_alignment_trinity(self):
        """Test configuration alignment based on conversion JSON."""
        from_config_fp = "tests/data/config.yaml"
        # Fixed config asset locations
        TRINITY_CONFIG_AUTO_CONVERSION = (
            "ajet/default_config/trinity/config_auto_convertion_trinity.jsonc"
        )

        with tempfile.NamedTemporaryFile(mode="r", suffix=".yaml") as temp_yaml1:
            config = read_astune_hierarchical_config(
                from_config_fp,
                "dummy_exp_name",
                backbone="trinity",
                write_to=temp_yaml1.name,
                exp_dir="tests/temp",
            )
            expand_astune_hierarchical_config(config, write_to=temp_yaml1.name)
            align_parameters(
                temp_yaml1.name, temp_yaml1.name, TRINITY_CONFIG_AUTO_CONVERSION, "trinity"
            )
            with open(temp_yaml1.name, "r") as file:
                to_config = yaml.safe_load(file)
            self.assertEqual(to_config["checkpoint_root_dir"], "/wow/astuner_checkpoints")
            self.assertEqual(to_config["buffer"]["batch_size"], 120)
            self.assertEqual(to_config["buffer"]["train_batch_size"], 1920)
            # Test simple field mappings
            self.assertEqual(to_config["project"], "unittest")
            self.assertEqual(to_config["name"], "dummy_exp_name")
            self.assertEqual(to_config["model"]["model_path"], "")
            # Test trainer common mappings
            self.assertEqual(to_config["trainer"]["save_interval"], 99999)
            self.assertEqual(to_config["buffer"]["total_epochs"], 99999)
            self.assertEqual(to_config["explorer"]["eval_interval"], 99999)
            # Test algorithm mappings
            self.assertEqual(to_config["algorithm"]["repeat_times"], 8)
            # Test explorer/rollout mappings
            self.assertEqual(to_config["explorer"]["rollout_model"]["tensor_parallel_size"], 4)
            # Test computed values
            self.assertEqual(to_config["buffer"]["batch_size"], 120)
            # (ajet.rollout.max_env_worker // ajet.rollout.n_vllm_engine) = 256 // 2 = 128
            self.assertEqual(to_config["explorer"]["runner_per_model"], 128)

    def test_config_alignment_verl(self):
        """Test configuration alignment based on conversion JSON."""
        from_config_fp = "tests/data/config.yaml"
        # Fixed config asset locations
        TRINITY_CONFIG_AUTO_CONVERSION = (
            "ajet/default_config/verl/config_auto_convertion_verl.jsonc"
        )

        with tempfile.NamedTemporaryFile(mode="r", suffix=".yaml") as temp_yaml1:
            config = read_astune_hierarchical_config(
                from_config_fp,
                "dummy_exp_name",
                backbone="verl",
                write_to=temp_yaml1.name,
                exp_dir="tests/temp",
            )
            expand_astune_hierarchical_config(config, write_to=temp_yaml1.name)
            align_parameters(
                temp_yaml1.name, temp_yaml1.name, TRINITY_CONFIG_AUTO_CONVERSION, "trinity"
            )
            with open(temp_yaml1.name, "r") as file:
                to_config = yaml.safe_load(file)
            self.assertEqual(
                to_config["trainer"]["checkpoint_base_dir"], "/wow/astuner_checkpoints"
            )
            self.assertEqual(
                to_config["trainer"]["default_local_dir"],
                r"${checkpoint_base_dir}/${trainer.project_name}/${trainer.experiment_name}",
            )
