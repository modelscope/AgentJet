import json
import os
import random
from typing import List

import dotenv

from astuner.data_generator.task_augmentation import TaskAugmentor
from astuner.schema.task import Task
from astuner.task_reader.document_reader.doc_reader import DocReader
from astuner.task_reader.task_reader_base import TaskReaderBase

dotenv.load_dotenv()


class TaskReaderDataGenerator(TaskReaderBase):
    def __init__(self, reader_config):
        super().__init__(reader_config)
        self.reader_config = reader_config
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        dataset_dir = os.path.join(project_root, "dataset/jsonl")
        os.makedirs(dataset_dir, exist_ok=True)

        self.generated_train_file = os.path.join(dataset_dir, "generated_train_tasks.jsonl")

        # Initialize
        from astuner.task_reader import TaskReaderRouter

        task_reader = TaskReaderRouter(
            reader_type=self.reader_config.data_generation.query_reader.type,
            reader_config=self.reader_config.data_generation.query_reader,
        )
        self.original_tasks = task_reader.get_training_tasks()

        # Check if cache file exists
        if os.path.exists(self.generated_train_file):
            try:
                print(f"Cache file found: {self.generated_train_file}")
                print("Loading tasks from cache...")
                self.new_tasks = self._read_jsonl_file(self.generated_train_file)
                if not self.new_tasks:
                    raise ValueError("Empty cache file")
            except Exception as e:
                print(f"Error loading cache: {e}")
                print("Regenerating tasks...")
                self._generate_and_save_tasks()
        else:
            print(f"Cache file not found: {self.generated_train_file}")
            print("Generating new tasks...")
            self._generate_and_save_tasks()

    def _generate_and_save_tasks(self):
        document_reader = DocReader(self.reader_config)
        task_augmentor = TaskAugmentor(self.reader_config)

        document = document_reader.get_document()
        task_num = self.reader_config.data_generation.task_num

        self.new_tasks = []
        for i in range(task_num):
            try:
                source_task = random.choice(self.original_tasks)
                new_task = task_augmentor.generate_task(source_task=source_task, document=document)
                self.new_tasks.append(new_task)

            except Exception as e:
                print(f"Error generating task {i + 1}: {e}")
                continue

        # Save tasks
        if self.new_tasks:
            print(f"Saving {len(self.new_tasks)} tasks to cache: {self.generated_train_file}")
            self._save_tasks_to_jsonl(self.new_tasks, self.generated_train_file)
        else:
            print("Warning: No tasks generated successfully")

    def _read_jsonl_file(self, file_path):
        """
        Read tasks from a JSONL file.

        Args:
            file_path (str): Path to the JSONL file.

        Returns:
            List[Task]: List of Task objects.
        """
        tasks = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        task_data = json.loads(line)
                        # Create a Task object from the JSON data
                        task = Task(
                            main_query=task_data.get("main_query", "[not defined]"),
                            init_messages=task_data.get("init_messages", []),
                            task_id=task_data.get("task_id", ""),
                            env_type=task_data.get("env_type", "no_env"),
                            metadata=task_data.get("metadata", task_data),
                        )
                        tasks.append(task)
        except FileNotFoundError:
            raise ValueError(f"JSONL file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {str(e)}")

        if len(tasks) == 0:
            raise ValueError(f"No tasks found in file: {file_path}")

        return tasks

    def _save_tasks_to_jsonl(self, tasks: List[Task], file_path: str):
        """
        Save tasks to a JSONL file.

        Args:
            tasks (List[Task]): List of Task objects to save.
            file_path (str): Path to the output JSONL file.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                for task in tasks:
                    # Convert Task object to dictionary
                    task_data = {
                        "main_query": task.main_query,
                        "init_messages": task.init_messages,
                        "task_id": task.task_id,
                        "env_type": task.env_type,
                        "metadata": task.metadata,
                    }
                    # Write as JSON line
                    f.write(json.dumps(task_data, ensure_ascii=False) + "\n")

        except Exception as e:
            raise ValueError(f"Error saving tasks to {file_path}: {str(e)}")

    def get_training_tasks(self) -> List[Task]:
        """
        Get training tasks from data generation.

        Returns:
            List[Task]: List of training Task objects.
        """
        return self.new_tasks

    def get_validation_tasks(self) -> List[Task]:
        """
        Get validation tasks from data generation.

        Returns:
            List[Task]: List of validation Task objects.
        """
        return self.original_tasks
