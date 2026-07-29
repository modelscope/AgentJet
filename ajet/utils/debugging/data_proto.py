# Copyright 2025 Alibaba Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pickle

from loguru import logger
from verl import DataProto


def import_or_export_data_proto(batch: DataProto, direction: str = "export", file: str = "./tmp.pkl") -> DataProto:
    """Import or export a DataProto batch to/from a pickle file.

    Args:
        batch: The DataProto batch object. Used when direction is "export";
               ignored (can be None) when direction is "import".
        direction: "import" to load a batch from file, "export" to save the batch to file.
        file: Path to the pickle file. Defaults to "./tmp.pkl".

    Returns:
        The DataProto batch — either the one just loaded (import) or the one just saved (export).

    Raises:
        ValueError: If direction is not "import" or "export".
        FileNotFoundError: If direction is "import" and the file does not exist.
    """
    if direction == "export":
        with open(file, "wb") as f:
            pickle.dump(batch, f)
        logger.info(f"[import_or_export_data_proto] Exported batch to {file}")
        return batch
    elif direction == "import":
        with open(file, "rb") as f:
            batch = pickle.load(f)
        logger.info(f"[import_or_export_data_proto] Imported batch from {file}")
        return batch
    else:
        raise ValueError(f"direction must be 'import' or 'export', got '{direction}'")
