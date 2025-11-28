import unittest

import dotenv

dotenv.load_dotenv()
from loguru import logger

from astuner.data_generator.knowledge_augmentation import KnowledgeAugmentor
from astuner.data_generator.task_augmentation import TaskAugmentor
from astuner.task_reader import TaskReaderRouterV2
from astuner.task_reader.document_reader.doc_reader import DocReader
from astuner.utils.config_utils import read_astune_config


class TestConfigUtils(unittest.TestCase):
    def test_data_gen_main(self):
        try:
            config = read_astune_config("tests/data_gen.yaml")

            task_reader = TaskReaderRouterV2(
                reader_type=config.astuner.data_generator.reader.type,
                reader_config=config.astuner.data_generator.reader,
            )
            Tasks = task_reader.get_training_tasks()
            Tasks = Tasks[:5]
            document_reader = DocReader(config)
            doc = document_reader.get_document()

            gene_tasks = []
            # generate task
            # 1. Task Augmentation
            task_augmentor = TaskAugmentor(config)
            print("-Task Augmentation Start")
            for i, task in enumerate(Tasks):
                result = task_augmentor.generate_task(source_task=task, document=doc)
                gene_tasks.extend([result] if not isinstance(result, list) else result)
            print("-Task Augmentation End")
            # 2. Knowledge Augmentation
            knowledge_augmentor = KnowledgeAugmentor(config)
            print("-Knowledge Augmentation Start")
            gene_tasks.extend(knowledge_augmentor.generate_task(source_task=None, document=doc))
            print("-Knowledge Augmentation End")
        except Exception as e:
            logger.exception("Data generation failed.")
            raise e
