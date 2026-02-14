
import datasets

from ajet.schema.task import Task
from ajet.task_reader.task_reader_base import BaseTaskReader


class HuggingFaceTaskReader(BaseTaskReader):
    """
    Task reader that reads tasks from Hugging Face datasets.

    This class allows loading tasks directly from Hugging Face dataset repositories.
    It supports configuring the dataset name and split names for training and validation.
    """

    def __init__(self, reader_config):
        super().__init__(reader_config)
        self.reader_config = reader_config
        self.as_generator = False
        self.dataset_name = self.reader_config.huggingface_dat_repo.dataset_path

    def _load_dataset_split(self, split: str):
        """
        Load a dataset split from Hugging Face datasets.

        Args:
            split: Name of the split to load (e.g., 'train', 'validation')

        Returns:
            Generator: List of Task objects created from the dataset.
        """
        try:
            if self.dataset_name.endswith(".parquet"):
                # Load from local parquet file
                dataset = datasets.load_dataset("parquet", data_files=self.dataset_name, split=split)
            else:
                # Load from Hugging Face hub
                dataset = datasets.load_dataset(self.dataset_name, split=split)
            # shuffle dataset
            dataset = dataset.map(lambda example, idx: {"original_idx": idx}, with_indices=True)
            dataset = dataset.shuffle()
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset '{self.dataset_name}' with split '{split}': {str(e)}"
            )

        if len(dataset) == 0:
            raise ValueError(f"No examples found in dataset '{self.dataset_name}' with split '{split}'")

        self.as_generator = True

        for _, example in enumerate(dataset):
            # Create Task object
            task = Task(
                main_query=example.get("question", "Empty"),    # type: ignore
                init_messages=[],  # Dataset examples typically don't have init messages
                task_id=str(example["original_idx"]),           # type: ignore
                env_type="no_env",
                metadata=example,                               # type: ignore
            )
            yield task

        return

    def generate_training_tasks(self):
        """
        Get training tasks from the Hugging Face dataset specified in the config.

        Returns:
            A generator of training Task objects.
        """
        split = self.reader_config.huggingface_dat_repo.training_split
        return self._load_dataset_split(split)

    def generate_validation_tasks(self):
        """
        Get validation tasks from the Hugging Face dataset specified in the config.

        Returns:
            A generator of validation Task objects.
        """
        split = self.reader_config.huggingface_dat_repo.validation_split
        return self._load_dataset_split(split)

    def get_training_tasks(self):
        return list(self.generate_training_tasks())

    def get_validation_tasks(self):
        return list(self.generate_validation_tasks())
