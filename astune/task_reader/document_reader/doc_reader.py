import uuid
import hashlib
import json
import os
from pathlib import Path
from typing import List
from astune.schema.document import Document
from astune.task_reader.document_reader.document_reader_base import DocReaderBase
from unstructured.partition.auto import partition

class DocReader(DocReaderBase):
    """
    Document reader with file hash caching support.
    """
    def __init__(self, config):
        super().__init__(config)
        self.cache_enabled = getattr(config.astune.document_reader, 'cache_enabled', True)

    def load_document(self, source: str, languages=['eng']) -> str:
        """
        Load text from a file with caching support.
        """
        if not self.cache_enabled:
            return self._parse_document(source, languages)
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(source)
        if not file_hash:
            return self._parse_document(source, languages)
        
        # Generate cache path
        cache_path = self._get_cache_path(source, file_hash, languages)
        
        # Try to load from cache
        cached_content = self._load_from_cache(cache_path)
        if cached_content:
            print(f"Cache hit: {Path(cache_path).name}")
            return cached_content
        
        # Cache miss, parse document
        print(f"Cache miss: Parsing {Path(source).name}...")
        text = self._parse_document(source, languages)
        
        # Save to cache
        self._save_to_cache(cache_path, text)
        print(f"Cached to: {Path(cache_path).name}")
        
        return text

    def _parse_document(self, source: str, languages: List[str]) -> str:
        """Parse document using unstructured."""
        text_pages = partition(source, languages=languages)
        if not text_pages:
            raise ValueError(f"No extractable text found in file: {source}")
        return "\n\n".join([str(sub) for sub in text_pages])

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""

    def _get_cache_path(self, source_path: str, file_hash: str, languages: List[str]) -> str:
        """Generate cache file path."""
        source_path = Path(source_path)
        lang_suffix = "_" + "_".join(sorted(languages)) if languages else ""
        cache_filename = f"{source_path.stem}.{file_hash[:16]}{lang_suffix}.cache.json"
        return str(source_path.parent / cache_filename)

    def _load_from_cache(self, cache_path: str) -> str:
        """Load cached content."""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    return cache_data.get('content', '')
        except Exception:
            pass
        return None

    def _save_to_cache(self, cache_path: str, content: str) -> bool:
        """Save content to cache."""
        try:
            cache_data = {'content': content}
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def get_document(self) -> Document:
        """Get document with caching."""
        file_path = self.config.astune.document_reader.document_path
        languages = list(self.config.astune.document_reader.languages)
        raw_doc = self.load_document(file_path, languages)
        return Document(
            doc_id=str(uuid.uuid4()),
            content=raw_doc,
            metadata={}
        )

