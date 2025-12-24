import hashlib
import json
import math
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import dotenv
from loguru import logger
from tqdm import tqdm

from astuner.data_generator.filters.deduplication_filter import DeduplicationFilter
from astuner.data_generator.knowledge_augmentation import KnowledgeAugmentor
from astuner.data_generator.task_augmentation import TaskAugmentor
from astuner.schema.task import Task
from astuner.task_reader.document_reader.doc_reader import DocReader
from astuner.task_reader.task_reader_base import BaseTaskReader

dotenv.load_dotenv()


class DataGeneratorTaskReader(BaseTaskReader):
    """
    Enhanced version of TaskReaderDataGenerator with multi-threading support,
    progress bars, and improved batch calculation.
    """

    def __init__(self, reader_config):
        super().__init__(reader_config)
        self.reader_config = reader_config
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        dataset_dir = os.path.join(project_root, "dataset/jsonl")
        os.makedirs(dataset_dir, exist_ok=True)

        # Build a cache key based on generation-related config to avoid rigid filenames
        document_path = getattr(
            reader_config.data_generation.document_reader, "document_path", None
        )
        # Convert document_path to a hashable string representation
        if isinstance(document_path, (list, tuple)):
            document_path_str = ",".join(sorted(str(p) for p in document_path))
        elif document_path is not None:
            document_path_str = str(document_path)
        else:
            document_path_str = ""

        cache_config = {
            "task_num": reader_config.data_generation.task_num,
            "num_workers": getattr(reader_config.data_generation, "num_workers", 32),
            "query_reader_type": getattr(reader_config.data_generation.query_reader, "type", None),
            "document_reader": document_path_str,
            "deduplication_filter": {
                "similarity_threshold": getattr(
                    reader_config.data_generation.deduplication_filter.params, "similarity_threshold", None
                ),
                "db_path": getattr(reader_config.data_generation.deduplication_filter.params, "db_path", None),
                "model": getattr(reader_config.data_generation.deduplication_filter.params, "model", None),
            },
        }
        cache_key_str = json.dumps(cache_config, sort_keys=True, ensure_ascii=False)
        cache_key = hashlib.md5(cache_key_str.encode("utf-8")).hexdigest()[:8]

        self.generated_train_file = os.path.join(
            dataset_dir, f"generated_train_tasks_{cache_key}.jsonl"
        )
        self.generated_valid_file = os.path.join(
            dataset_dir, f"generated_valid_tasks_{cache_key}.jsonl"
        )

        # Get number of workers from config, default to 32
        self.num_workers = getattr(reader_config.data_generation, "num_workers", 32)

        # Thread-safe lock for shared resources
        self.lock = threading.Lock()

        # Initialize duplicate filter
        if self.reader_config.data_generation.deduplication_filter.enabled:
            self.duplicate_filter = DeduplicationFilter(
                similarity_threshold=self.reader_config.data_generation.deduplication_filter.params.similarity_threshold,
                db_path=self.reader_config.data_generation.deduplication_filter.params.db_path,
                model=self.reader_config.data_generation.deduplication_filter.params.model,
                api_key=self.reader_config.data_generation.deduplication_filter.params.api_key,
                base_url=self.reader_config.data_generation.deduplication_filter.params.base_url,
            )
        else:
            self.duplicate_filter = None
        # Initialize task reader
        from astuner.task_reader import RouterTaskReader

        task_reader = RouterTaskReader(
            reader_type=self.reader_config.data_generation.query_reader.type,
            reader_config=self.reader_config.data_generation.query_reader,
        )
        self.original_tasks = task_reader.get_training_tasks()

        # Check cache files and load/generate accordingly
        train_cache_exists = os.path.exists(self.generated_train_file)
        valid_cache_exists = os.path.exists(self.generated_valid_file)

        # Load validation tasks from cache if available
        if valid_cache_exists:
            try:
                logger.info(f"Validation cache found: {self.generated_valid_file}")
                self.doc_tasks = self._read_jsonl_file(self.generated_valid_file)
                logger.info(f"Loaded {len(self.doc_tasks)} validation tasks from cache")
            except Exception as e:
                logger.error(f"Error loading validation cache: {e}")
                self.doc_tasks = None
        else:
            self.doc_tasks = None

        # Load training tasks from cache if available
        if train_cache_exists:
            try:
                logger.info(f"Training cache found: {self.generated_train_file}")
                self.new_tasks = self._read_jsonl_file(self.generated_train_file)
                logger.info(f"Loaded {len(self.new_tasks)} training tasks from cache")
            except Exception as e:
                logger.error(f"Error loading training cache: {e}")
                self.new_tasks = None
        else:
            self.new_tasks = None

        # Generate missing tasks
        self._generate_and_save_tasks()

    def _generate_document_tasks_worker(self, args):
        """
        Worker function for generating document-based tasks.

        Args:
            args: Tuple containing (batch_index, document, knowledge_augmentor)

        Returns:
            Tuple of (batch_index, generated_tasks, error_message)
        """
        batch_index, document, knowledge_augmentor = args
        try:
            tasks = knowledge_augmentor.generate_task(source_task=None, document=document)
            return batch_index, tasks, None
        except Exception as e:
            error_msg = f"Error generating document batch {batch_index}: {e}"
            return batch_index, [], error_msg

    def _generate_augmented_tasks_worker(self, args):
        """
        Worker function for generating augmented tasks.

        Args:
            args: Tuple containing (task_index, source_task, document, task_augmentor)

        Returns:
            Tuple of (task_index, generated_task, error_message)
        """
        task_index, source_task, document, task_augmentor = args
        try:
            new_task = task_augmentor.generate_task(source_task=source_task, document=document)
            return task_index, new_task, None
        except Exception as e:
            error_msg = f"Error generating task {task_index}: {e}"
            return task_index, None, error_msg

    def _generate_and_save_tasks(self):
        """
        Enhanced version with selective generation based on cache availability.
        """
        logger.info(f"Using {self.num_workers} workers for task generation")

        document_reader = DocReader(self.reader_config)
        documents = document_reader.get_document()
        task_num = self.reader_config.data_generation.task_num

        # Phase 1: Generate document-based tasks only if not cached
        if self.doc_tasks is None and documents is not None:
            logger.info("Phase 1: Generating document-based tasks for validation...")

            task_augmentor = TaskAugmentor(self.reader_config)
            knowledge_augmentor = KnowledgeAugmentor(self.reader_config)

            # Calculate batches using ceiling division
            N = 10  # 10 is the hyperparameter we found that produces relatively stable outputs, same with knowledge_augmentation
            doc_task_rounds = math.ceil(task_num / N)
            logger.info(
                f"Generating {doc_task_rounds} document-based task batches (ceil({task_num}/10))"
            )

            self.doc_tasks = []

            # Prepare arguments for workers
            doc_worker_args = []
            for i in range(doc_task_rounds):
                document = documents[i % len(documents)]
                doc_worker_args.append((i, document, knowledge_augmentor))

            # Execute document task generation with progress bar
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_batch = {
                    executor.submit(self._generate_document_tasks_worker, args): args[0]
                    for args in doc_worker_args
                }

                # Process results with progress bar
                with tqdm(total=doc_task_rounds, desc="Document tasks", unit="batch") as pbar:
                    for future in as_completed(future_to_batch):
                        batch_index, tasks, error_msg = future.result()

                        if error_msg:
                            logger.error(f"\n{error_msg}")
                        else:
                            with self.lock:
                                self.doc_tasks.extend(tasks)

                        pbar.update(1)
            if self.duplicate_filter is not None:
                self.doc_tasks = self.duplicate_filter.filter(self.doc_tasks)
            logger.info(f"Generated {len(self.doc_tasks)} document-based tasks")

            # Save doc_tasks as validation tasks cache
            if self.doc_tasks:
                logger.info(
                    f"Saving {len(self.doc_tasks)} validation tasks to cache: {self.generated_valid_file}"
                )
                self._save_tasks_to_jsonl(self.doc_tasks, self.generated_valid_file)
        else:
            logger.info(
                "Phase 1: Skipping document task generation (using cached validation tasks)"
            )

        # Phase 2: Generate augmented tasks only if not cached
        if self.new_tasks is None:
            logger.info("Phase 2: Generating augmented tasks using original + document tasks...")

            task_augmentor = TaskAugmentor(self.reader_config)

            self.new_tasks = []

            # Combine original tasks and doc tasks for source task selection
            if not self.original_tasks:
                self.original_tasks = []
            if not self.doc_tasks:
                self.doc_tasks = []
            combined_source_tasks = self.original_tasks + self.doc_tasks
            logger.info(
                f"Using {len(combined_source_tasks)} source tasks ({len(self.original_tasks)} original + {len(self.doc_tasks)} document-based)"
            )

            # Prepare arguments for workers
            aug_worker_args = []
            for i in range(task_num):
                source_task = random.choice(combined_source_tasks)
                # Use a random document for augmentation
                document = random.choice(documents) if documents else None
                aug_worker_args.append((i, source_task, document, task_augmentor))

            # Execute augmented task generation with progress bar
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._generate_augmented_tasks_worker, args): args[0]
                    for args in aug_worker_args
                }

                # Process results with progress bar
                with tqdm(total=task_num, desc="Augmented tasks", unit="task") as pbar:
                    for future in as_completed(future_to_task):
                        task_index, new_task, error_msg = future.result()

                        if error_msg:
                            logger.error(f"\n{error_msg}")
                        elif new_task:
                            with self.lock:
                                self.new_tasks.append(new_task)

                        pbar.update(1)
            if self.duplicate_filter is not None:
                self.new_tasks = self.duplicate_filter.filter(self.new_tasks)
            logger.info(f"Generated {len(self.new_tasks)} augmented tasks")

            # Save training tasks
            if self.new_tasks:
                logger.info(
                    f"Saving {len(self.new_tasks)} training tasks to cache: {self.generated_train_file}"
                )
                self._save_tasks_to_jsonl(self.new_tasks, self.generated_train_file)
            else:
                logger.warning("No training tasks generated successfully")
        else:
            logger.info("Phase 2: Skipping training task generation (using cached training tasks)")

        logger.info(
            f"Task generation complete: {len(self.new_tasks)} training tasks, {len(self.doc_tasks)} validation tasks"
        )

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
        Save tasks to a JSONL file with progress bar.

        Args:
            tasks (List[Task]): List of Task objects to save.
            file_path (str): Path to the output JSONL file.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                with tqdm(total=len(tasks), desc="Saving tasks", unit="task") as pbar:
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
                        pbar.update(1)

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
        Now returns document-based tasks as validation tasks.

        Returns:
            List[Task]: List of validation Task objects (doc_tasks).
        """
        return getattr(self, "doc_tasks", []) + self.original_tasks

    def get_generation_stats(self) -> dict:
        """
        Get statistics about the task generation process.

        Returns:
            dict: Statistics including worker count, batch info, and task counts
        """
        task_num = self.reader_config.data_generation.task_num
        doc_task_rounds = math.ceil(task_num / 10)

        return {
            "num_workers": self.num_workers,
            "target_task_num": task_num,
            "calculated_batches": doc_task_rounds,
            "doc_tasks_generated": len(getattr(self, "doc_tasks", [])),
            "augmented_tasks_generated": len(getattr(self, "new_tasks", [])),
            "original_tasks_count": len(self.original_tasks),
            "validation_tasks_count": len(getattr(self, "doc_tasks", [])),
            "combined_source_tasks_count": len(self.original_tasks)
            + len(getattr(self, "doc_tasks", [])),
        }
