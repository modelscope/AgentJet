import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime


DEFAULT_SYNC_TRAIN_CODE_ROOT = "./ajet_temp"


def extract_ajet_zip(
    zip_bytes: bytes,
    sync_train_code_root: str = DEFAULT_SYNC_TRAIN_CODE_ROOT,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_code_base_path = os.path.abspath(os.path.join(sync_train_code_root, timestamp))
    temp_ajet_code_path = os.path.join(temp_code_base_path, "ajet")

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
            members = zip_file.infolist()
            if not members:
                raise ValueError("Uploaded zip file is empty.")

            valid_members = []
            for member in members:
                normalized_name = member.filename.replace("\\", "/")
                normalized_path = os.path.normpath(normalized_name)
                if normalized_path in (".", ""):
                    continue
                if normalized_path.startswith("../") or os.path.isabs(normalized_path):
                    raise ValueError(f"Unsafe zip member path: {member.filename}")
                if normalized_path != "ajet" and not normalized_path.startswith("ajet/"):
                    raise ValueError(f"Zip member must be under ajet/: {member.filename}")
                valid_members.append((member, normalized_path))

            if not valid_members:
                raise ValueError("Uploaded zip file does not contain an ajet/ directory.")

            os.makedirs(temp_code_base_path, exist_ok=False)
            for member, normalized_path in valid_members:
                target_path = os.path.join(temp_code_base_path, normalized_path)
                if member.is_dir():
                    os.makedirs(target_path, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zip_file.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

        if not os.path.isdir(temp_ajet_code_path):
            raise ValueError("Uploaded zip file does not contain an ajet/ directory.")
        return temp_ajet_code_path
    except Exception:
        if os.path.exists(temp_code_base_path):
            shutil.rmtree(temp_code_base_path, ignore_errors=True)
        raise


def create_tracked_ajet_zip_from_dir(directory_path: str) -> tuple[str, int]:
    code_base_dir = os.path.abspath(os.path.expanduser(directory_path))
    ajet_dir = os.path.join(code_base_dir, "ajet")
    if not os.path.isdir(ajet_dir):
        raise FileNotFoundError(f"Directory must contain an ajet/ folder: {code_base_dir}")

    try:
        result = subprocess.run(
            ["git", "-C", code_base_dir, "ls-files", "-z", "--", "ajet"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Unable to list Git-controlled files under {ajet_dir}: {error_message}"
        ) from e

    tracked_files = [
        path.decode("utf-8")
        for path in result.stdout.split(b"\0")
        if path
    ]
    if not tracked_files:
        raise RuntimeError(f"No Git-controlled files found under {ajet_dir}")

    with tempfile.NamedTemporaryFile(
        prefix="ajet_train_code_",
        suffix=".zip",
        dir="/tmp",
        delete=False,
    ) as temp_zip:
        zip_file_path = temp_zip.name

    file_count = 0
    with zipfile.ZipFile(
        zip_file_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:
        for relative_path in tracked_files:
            normalized_path = os.path.normpath(relative_path)
            if normalized_path == "ajet" or not normalized_path.startswith("ajet/"):
                continue
            source_path = os.path.join(code_base_dir, normalized_path)
            if not os.path.isfile(source_path):
                continue
            zip_file.write(source_path, arcname=normalized_path)
            file_count += 1

    if file_count == 0:
        os.remove(zip_file_path)
        raise RuntimeError(f"No existing Git-controlled files found under {ajet_dir}")

    return zip_file_path, file_count
