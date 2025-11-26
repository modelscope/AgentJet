import unittest
import os
from typing import Any
from unittest.mock import patch, MagicMock

from astune.utils.config_utils import prepare_experiment_config

try:
    from dotmap import DotMap  # type: ignore
    has_dotmap = True
except ImportError:
    DotMap = None  # type: ignore
    has_dotmap = False


class TestDataGenerator(unittest.TestCase):

    def test_load_config(self):
        """Test if the data generator configuration file is loaded without errors."""
        yaml_backup_dst, exp_base, exp_name, config = prepare_experiment_config(
            'tests/data/data_gen.yaml', 
            "tests/temp", 
            backbone="debug"
        )
        
        # Test basic experiment configuration
        self.assertEqual(exp_name, "data_gen_test")
        self.assertEqual(exp_base, "tests/temp/data_gen_test")
        self.assertEqual(yaml_backup_dst, "tests/temp/data_gen_test/yaml_backup.yaml")
        
        # Test astune configuration
        self.assertIn("astune", config)
        self.assertIn("project_name", config["astune"])
        self.assertEqual(config["astune"]["project_name"], "unittest")
        self.assertIn("experiment_name", config["astune"])
        self.assertEqual(config["astune"]["experiment_name"], "data_gen_test")

    def test_task_reader_config(self):
        """Test task reader configuration."""
        _, _, _, config = prepare_experiment_config(
            'tests/data/data_gen.yaml', 
            "tests/temp", 
            backbone="debug"
        )
        
        # Test task_reader configuration
        self.assertIn("task_reader", config["astune"])
        self.assertEqual(config["astune"]["task_reader"]["type"], "huggingface_dat_repo")
        self.assertIn("huggingface_dat_repo", config["astune"]["task_reader"])
        
        hf_config = config["astune"]["task_reader"]["huggingface_dat_repo"]
        self.assertIn("dataset_path", hf_config)
        self.assertIn("training_split", hf_config)
        self.assertIn("validation_split", hf_config)
        self.assertEqual(hf_config["training_split"], "train")
        self.assertEqual(hf_config["validation_split"], "test")

    def test_document_reader_config(self):
        """Test document reader configuration."""
        _, _, _, config = prepare_experiment_config(
            'tests/data/data_gen.yaml', 
            "tests/temp", 
            backbone="debug"
        )
        
        # Test document_reader configuration
        self.assertIn("document_reader", config["astune"])
        doc_reader_config = config["astune"]["document_reader"]
        self.assertIn("document_path", doc_reader_config)
        self.assertIn("languages", doc_reader_config)
        self.assertIn("eng", doc_reader_config["languages"])

    def test_data_generator_config(self):
        """Test data generator configuration."""
        _, _, _, config = prepare_experiment_config(
            'tests/data/data_gen.yaml', 
            "tests/temp", 
            backbone="debug"
        )
        
        # Test data_generator configuration
        self.assertIn("data_generator", config["astune"])
        data_gen_config = config["astune"]["data_generator"]
        
        # Test LLM model configuration
        self.assertIn("llm_model", data_gen_config)
        self.assertEqual(data_gen_config["llm_model"], "qwen-turbo")
        
        # Test LLM response length
        self.assertIn("llm_response_length", data_gen_config)
        self.assertEqual(data_gen_config["llm_response_length"], 2048)
        
        # Test sampling parameters
        self.assertIn("sampling_params", data_gen_config)
        sampling_params = data_gen_config["sampling_params"]
        self.assertIn("temperature", sampling_params)
        self.assertEqual(sampling_params["temperature"], 0.7)
        self.assertIn("top_p", sampling_params)
        self.assertEqual(sampling_params["top_p"], 0.9)

    def test_task_augmentor_config(self):
        """Test task augmentor configuration."""
        _, _, _, config = prepare_experiment_config(
            'tests/data/data_gen.yaml', 
            "tests/temp", 
            backbone="debug"
        )
        
        # Test task_augmentor configuration
        data_gen_config = config["astune"]["data_generator"]
        self.assertIn("task_augmentor", data_gen_config)
        task_aug_config = data_gen_config["task_augmentor"]
        self.assertIn("enabled", task_aug_config)
        self.assertTrue(task_aug_config["enabled"])

    def test_knowledge_augmentor_config(self):
        """Test knowledge augmentor configuration."""
        _, _, _, config = prepare_experiment_config(
            'tests/data/data_gen.yaml', 
            "tests/temp", 
            backbone="debug"
        )
        
        # Test knowledge_augmentor configuration
        data_gen_config = config["astune"]["data_generator"]
        self.assertIn("knowledge_augmentor", data_gen_config)
        knowledge_aug_config = data_gen_config["knowledge_augmentor"]
        
        self.assertIn("n", knowledge_aug_config)
        self.assertEqual(knowledge_aug_config["n"], 5)
        self.assertIn("enabled", knowledge_aug_config)
        self.assertTrue(knowledge_aug_config["enabled"])

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"})
    @unittest.skipIf(not has_dotmap, "dotmap not installed")
    def test_data_generator_initialization(self):
        """Test if data generator components can be initialized with config."""
        from astune.data_generator.task_augmentation import TaskAugmentor
        from astune.data_generator.knowledge_augmentation import KnowledgeAugmentor
        
        _, _, _, config = prepare_experiment_config(
            'tests/data/data_gen.yaml', 
            "tests/temp", 
            backbone="debug"
        )
        
        config_obj = DotMap(config)  # type: ignore
        
        # Test TaskAugmentor initialization
        task_augmentor = TaskAugmentor(config_obj)
        self.assertIsNotNone(task_augmentor)
        self.assertEqual(task_augmentor.config, config_obj)
        
        # Test KnowledgeAugmentor initialization
        knowledge_augmentor = KnowledgeAugmentor(config_obj)
        self.assertIsNotNone(knowledge_augmentor)
        self.assertEqual(knowledge_augmentor.config, config_obj)


if __name__ == "__main__":
    unittest.main()
