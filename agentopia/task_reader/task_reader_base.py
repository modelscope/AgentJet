import json
import uuid
import datasets
from typing import List, Dict, Optional
from agentopia.schema.task import Task
from agentopia.utils.process_dataset import create_rl_dataset, create_rl_sampler
from agentopia.client.env_client_ng import EnvClient


class TaskReaderBase:
    def __init__(self, config):
        self.config = config

    def get_training_tasks(self)->List[Task]:
        raise NotImplementedError

    def get_validation_tasks(self)->List[Task]:
        raise NotImplementedError


class TaskReaderAppWorld(TaskReaderBase):
    def __init__(self, config):
        super().__init__(config)

    def get_tasks(self, split):
        env_url = self.config.astune.task_reader.env_service.env_url
        env_type = self.config.astune.task_reader.env_service.env_type
        env_service_client = EnvClient(base_url=env_url)
        task_id_array = env_service_client.get_env_profile(env_type, split=split)
        if len(task_id_array) == 0:
            raise ValueError(f"No task_id found for env_type: {env_type}, split: {split}, Please check connection to {env_url}")
        tasks = [
            Task(
                main_query='[not defined]',
                init_messages=[],
                task_id=str(task_id),
                env_type=env_type,
                metadata={},
            ) for task_id in task_id_array]
        return tasks

    def get_validation_tasks(self):
        split = self.config.astune.task_reader.env_service.validation_split
        return self.get_tasks(split=split)

    def get_training_tasks(self):
        split = self.config.astune.task_reader.env_service.training_split
        return self.get_tasks(split=split)


class TaskReaderJsonl(TaskReaderBase):
    def __init__(self, config):
        super().__init__(config)

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
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        task_data = json.loads(line)
                        # Create a Task object from the JSON data
                        task = Task(
                            main_query=task_data.get('main_query', '[not defined]'),
                            init_messages=task_data.get('init_messages', []),
                            task_id=task_data.get('task_id', ''),
                            env_type=task_data.get('env_type', 'no_env'),
                            metadata=task_data.get('metadata', {})
                        )
                        tasks.append(task)
        except FileNotFoundError:
            raise ValueError(f"JSONL file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {str(e)}")

        if len(tasks) == 0:
            raise ValueError(f"No tasks found in file: {file_path}")

        return tasks

    def get_training_tasks(self) -> List[Task]:
        """
        Get training tasks from the JSONL file specified in the config.

        Returns:
            List[Task]: List of training Task objects.
        """
        file_path = self.config.astune.task_reader.dataset_file.training.file_path
        return self._read_jsonl_file(file_path)

    def get_validation_tasks(self) -> List[Task]:
        """
        Get validation tasks from the JSONL file specified in the config.

        Returns:
            List[Task]: List of validation Task objects.
        """
        file_path = self.config.astune.task_reader.dataset_file.validation.file_path
        return self._read_jsonl_file(file_path)


class TaskReaderHuggingFace(TaskReaderBase):
    """
    Task reader that reads tasks from Hugging Face datasets.

    This class allows loading tasks directly from Hugging Face dataset repositories.
    It supports configuring the dataset name and split names for training and validation.
    """

    def __init__(self, config):
        super().__init__(config)


    def _load_dataset_split(self, dataset_name: str, split: str) -> List[Task]:
        """
        Load a dataset split from Hugging Face datasets.

        Args:
            dataset_name: Name of the dataset in Hugging Face format (e.g., 'gsm8k')
            split: Name of the split to load (e.g., 'train', 'validation')

        Returns:
            List[Task]: List of Task objects created from the dataset.
        """
        try:
            dataset = datasets.load_dataset(dataset_name, split=split)
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{dataset_name}' with split '{split}': {str(e)}")

        # if len(dataset) == 0:
        #     raise ValueError(f"No examples found in dataset '{dataset_name}' with split '{split}'")

        tasks = []
        for idx, example in enumerate(dataset):
            # Try to extract relevant information from the dataset example
            # The structure might vary between datasets, so we handle common cases

            # Extract main query - try common field names for questions/instructions
            main_query = None
            for field in ['question', 'input', 'instruction', 'text', 'prompt']:
                if field in example:
                    main_query = example[field]
                    break

            if main_query is None:
                # Use the first string field as main_query if no known field is found
                for key, value in example.items():
                    if isinstance(value, str) and value.strip():
                        main_query = value
                        break

            # If still no main query, use a generic placeholder
            if main_query is None:
                main_query = f"Example {idx} from {dataset_name}"

            # Create a unique task_id if not provided
            task_id = str(example.get('task_id', f"{dataset_name}-{split}-{idx}-{uuid.uuid4().hex[:8]}"))

            # Extract any answer/target/label as metadata
            metadata = {}
            for field in ['answer', 'output', 'target', 'label', 'solution']:
                if field in example:
                    metadata[field] = example[field]

            # Include all original fields in metadata for reference
            metadata['original_data'] = {k: v for k, v in example.items()}

            # Create Task object
            task = Task(
                main_query=main_query,
                init_messages=[],  # Dataset examples typically don't have init messages
                task_id=task_id,
                env_type=f"huggingface/{dataset_name}",
                metadata=metadata,
            )
            tasks.append(task)

        return tasks

    def get_training_tasks(self) -> List[Task]:
        """
        Get training tasks from the Hugging Face dataset specified in the config.

        Returns:
            List[Task]: List of training Task objects.
        """
        dataset_name = self.config.astune.task_reader.huggingface_dat_repo.dataset_name
        split = self.config.astune.task_reader.huggingface_dat_repo.training_split
        return self._load_dataset_split(dataset_name, split)

    def get_validation_tasks(self) -> List[Task]:
        """
        Get validation tasks from the Hugging Face dataset specified in the config.

        Returns:
            List[Task]: List of validation Task objects.
        """
        dataset_name = self.config.astune.task_reader.huggingface_dat_repo.dataset_name
        split = self.config.astune.task_reader.huggingface_dat_repo.validation_split
        return self._load_dataset_split(dataset_name, split)


class TaskReaderRouter(TaskReaderBase):
    def __init__(self, config):
        super().__init__(config)
        self.task_reader_type = self.config.astune.task_reader.type
        if self.task_reader_type == 'env_service':
            self.task_reader = TaskReaderAppWorld(config)
        elif self.task_reader_type == 'dataset_file':
            self.task_reader = TaskReaderJsonl(config)
        elif self.task_reader_type == 'huggingface_dat_repo':
            self.task_reader = TaskReaderHuggingFace(config)
        else:
            raise ValueError(f"Unsupported task reader type: {self.task_reader_type}")

    def get_training_tasks(self) -> List[Task]:
        return self.task_reader.get_training_tasks()

    def get_validation_tasks(self) -> List[Task]:
        return self.task_reader.get_validation_tasks()

def task_to_standard_dataset(tasks: List[Task]) -> datasets.Dataset:
    """
    Convert a list of Task objects to a standard Hugging Face Dataset.

    Args:
        tasks (List[Task]): List of Task objects.

    Returns:
        datasets.Dataset: Hugging Face Dataset containing the tasks.
    """
    data = {
        'task_id': [],
        'main_query': [],
        'init_messages': [],
        'env_type': [],
        'metadata': [],
    }

    for task in tasks:
        data['task_id'].append(task.task_id)
        data['main_query'].append(task.main_query)
        data['init_messages'].append(task.init_messages)
        data['env_type'].append(task.env_type)
        data['metadata'].append(task.metadata)

    return datasets.Dataset.from_dict(data)