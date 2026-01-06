import random
import unittest

import dotenv
from loguru import logger

from ajet.data_generator.knowledge_augmentation import KnowledgeAugmentor
from ajet.data_generator.task_augmentation import TaskAugmentor
from ajet.task_reader import RouterTaskReader
from ajet.task_reader.document_reader.doc_reader import DocReader
from ajet.utils.config_utils import read_astune_config

dotenv.load_dotenv()


class TestConfigUtils(unittest.TestCase):
    def test_data_gen_main(self):
        try:
            config = read_astune_config("tests/data_gen.yaml")

            task_reader = RouterTaskReader(
                reader_type=config.task_reader.data_generation.query_reader.type,
                reader_config=config.task_reader.data_generation.query_reader,
            )
            Tasks = task_reader.get_training_tasks()
            task_num = config.task_reader.data_generation.task_num
            document_reader = DocReader(config)
            doc = document_reader.get_document()

            gen_tasks = []
            # generate task
            # 1. Task Augmentation
            task_augmentor = TaskAugmentor(config)
            print("-Task Augmentation Start")
            for _ in range(task_num):
                source_task = random.choice(Tasks)
                result = task_augmentor.generate_task(source_task=source_task, document=doc)
                gen_tasks.extend([result] if not isinstance(result, list) else result)
            print("-Task Augmentation End")
            # 2. Knowledge Augmentation
            knowledge_augmentor = KnowledgeAugmentor(config)
            print("-Knowledge Augmentation Start")
            gen_tasks.extend(knowledge_augmentor.generate_task(source_task=None, document=doc))
            print("-Knowledge Augmentation End")
        except Exception as e:
            logger.exception("Data generation failed.")
            raise e
