import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import List, Union

from loguru import logger

try:
    from unstructured.partition.auto import partition
except Exception:
    logger.warning("Cannot import dependency `unstructured`")

from astuner.schema.document import Document
from astuner.task_reader.document_reader.document_reader_base import DocReaderBase


class DocReader(DocReaderBase):
    """
    Enhanced document reader with file hash caching support and document chunking capabilities.
    """

    def __init__(self, config):
        super().__init__(config)
        self.cache_enabled = getattr(config.data_generation.document_reader, "cache_enabled", True)
        self.chunk_size = getattr(config.data_generation.document_reader, "chunk_size", 10240)
        self.split_by = getattr(config.data_generation.document_reader, "split_by", "sentence")

    def load_document(self, source: str, languages=["eng"]) -> str:
        """
        Load text from a file with caching support.
        """
        if not self.cache_enabled:
            return self._parse_document(source, languages)

        # Calculate file hash
        file_hash = self._calculate_file_hash(source)
        if not file_hash:
            return self._parse_document(source, languages)

        # Generate cache path (include chunking parameters in cache key)
        cache_path = self._get_cache_path(source, file_hash, languages)

        # Try to load from cache
        cached_content = self._load_from_cache(cache_path)
        if cached_content:
            logger.info(f"Cache hit: {Path(cache_path).name}")
            return cached_content

        # Cache miss, parse document
        logger.info(f"Cache miss: Parsing {Path(source).name}...")
        text = self._parse_document(source, languages)

        # Save to cache
        self._save_to_cache(cache_path, text)
        logger.info(f"Cached to: {Path(cache_path).name}")

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
        """Generate cache file path with chunking parameters."""
        source_path = Path(source_path)
        lang_suffix = "_" + "_".join(sorted(languages)) if languages else ""

        # Include chunking parameters in cache filename
        chunk_suffix = ""
        if self.chunk_size:
            chunk_suffix = f"_chunk{self.chunk_size}_{self.split_by}"

        cache_filename = (
            f"{source_path.stem}.{file_hash[:16]}{lang_suffix}{chunk_suffix}.cache.json"
        )
        return str(source_path.parent / cache_filename)

    def _load_from_cache(self, cache_path: str) -> Union[str, None]:
        """Load cached content."""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                    return cache_data.get("content", "")
        except Exception:
            pass
        return None

    def _save_to_cache(self, cache_path: str, content: str) -> bool:
        """Save content to cache."""
        try:
            cache_data = {"content": content}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def _split_text_by_sentences(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text by sentences with specified chunk size.
        """
        sentence_endings = r"[.!?。！？]+"
        sentences = re.split(f"({sentence_endings})", text)
        combined_sentences = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            if sentence:
                combined_sentences.append(sentence)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in combined_sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_text_by_paragraphs(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text by paragraphs with specified chunk size.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)

            if current_length + paragraph_length > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _split_text_by_characters(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text by characters with specified chunk size.
        """
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _chunk_document(self, text: str) -> List[str]:
        """
        Split document into chunks based on chunk_size and split_by parameters.
        """
        if not self.chunk_size or self.chunk_size <= 0:
            return [text]

        if self.split_by == "sentence":
            return self._split_text_by_sentences(text, self.chunk_size)
        elif self.split_by == "paragraph":
            return self._split_text_by_paragraphs(text, self.chunk_size)
        elif self.split_by == "character":
            return self._split_text_by_characters(text, self.chunk_size)
        else:
            logger.warning(f"Unknown split_by value '{self.split_by}', using 'sentence' as default")
            return self._split_text_by_sentences(text, self.chunk_size)

    def _parser_document(self, raw_document: str, source_path: str = None) -> List[Document]:
        """
        Parse raw document into Document objects, with chunking support.
        Each chunk from the same source document will have the same group_id.
        """
        chunks = self._chunk_document(raw_document)
        documents = []

        # Generate a unique group_id for all chunks from the same source document
        group_id = str(uuid.uuid4())
        source_name = Path(source_path).name if source_path else "unknown"

        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            metadata = {
                "group_id": group_id,
                "source_file": source_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": self.chunk_size,
                "split_by": self.split_by,
            }
            documents.append(Document(doc_id=doc_id, content=chunk, metadata=metadata))

        return documents

    def get_document(self) -> Union[List[Document], None]:
        """
        Get all documents with chunking support.
        Each source file will generate chunks with the same group_id.
        """
        # Safely retrieve document_path from config. If it's missing or falsy, return None.
        file_paths = getattr(self.config.data_generation.document_reader, "document_path", None)
        if not file_paths:
            return None

        # Normalize single string/Path into a list of paths
        if isinstance(file_paths, (str, Path)):
            file_paths = [str(file_paths)]

        # Ensure we have a concrete list (in case it's a generator or other iterable)
        try:
            file_paths = list(file_paths)
        except Exception:
            file_paths = [file_paths]

        all_documents = []

        for file_path in file_paths:
            raw_doc = self.load_document(
                file_path, languages=list(self.config.data_generation.document_reader.languages)
            )
            # _parser_document now returns a list of documents (chunks) with group_id
            documents = self._parser_document(raw_doc, source_path=file_path)
            all_documents.extend(documents)

        return all_documents
