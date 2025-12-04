import os
from typing import List, Optional, cast
import uuid

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Union, Sequence, Dict, Any
from chromadb.config import Settings
import chromadb
import httpx

class OpenAIEmbeddingClient:
    """
    Client class for OpenAI Embedding API.
    Supports calling embedding APIs in OpenAI format with rate limiting.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", 
                 model_name: str = "text-embedding-ada-002",
                 rate_limit_calls: int = 60, rate_limit_window: int = 60):
        """
        Initializes the OpenAI Embedding API client.

        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL for the API, defaulting to the official OpenAI address.
            model_name (str): The name of the model to use, defaulting to text-embedding-ada-002.
            rate_limit_calls (int): The number of allowed calls within the rate limit window, defaulting to 60.
            rate_limit_window (int): The time window in seconds for the rate limit, defaulting to 60 seconds.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')  # ⭐ Ensures the base URL does not end with a trailing slash
        self.model_name = model_name
        # Set up the request headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"  # ⭐ Constructs the authorization header using the provided API key
        }
        
        logger.info(f"init OpenAI Embedding client, quota: {rate_limit_calls} times/{rate_limit_window}s")

    @retry(stop=stop_after_attempt(4),wait=wait_exponential(multiplier=1, min=4, max=60))
    def get_embeddings(self, texts: Union[str, Sequence[str]], 
                      model: Optional[str] = None,
                      encoding_format: str = "float",
                      dimensions: Optional[int] = None,
                      user: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches the embedding vectors for the provided texts with rate limiting.

        Args:
            texts (Union[str, Sequence[str]]): Text(s) for which to fetch the embeddings, can be a single string or a list of strings.
            model (Optional[str]): Name of the model to use; if not specified, the model set during initialization is used.
            encoding_format (str): Encoding format for the embeddings, default is "float".
            dimensions (Optional[int]): Output dimensionality (supported by some models).
            user (Optional[str]): User identifier.

        Returns:
            Dict[str, Any]: The API response as a dictionary.

        Raises:
            requests.RequestException: If there is an issue with the request.
            ValueError: If the input parameters are invalid.
        """
        # Rate limiting control
        # self.rate_limiter.acquire()  # ⭐ Acquires a token from the rate limiter to ensure the request does not exceed the allowed rate
        
        # Parameter validation
        if not texts:
            raise ValueError("texts cannot be empty")
        
        # Construct the request payload
        payload = {
            "input": texts,
            "model": model or self.model_name,
            "encoding_format": encoding_format
        }
        
        # Add optional parameters
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if user is not None:
            payload["user"] = user
        
        # Send the request
        url = f"{self.base_url}/embeddings"
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    url,
                    headers=self.headers,
                    json=payload,
                )
            if not response.is_success:
                logger.error(f"failed to request embedding: {response.status_code} {response.reason_phrase}")
                try:
                    logger.error(f"err json: {response.json()}")
                except Exception:
                    logger.error("err json: <failed to decode response body>")
                response.raise_for_status()
            
            return response.json()
            
        except httpx.RequestError as e:
            raise httpx.RequestError(f"failed to request embedding: {e}")

    def get_single_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Retrieves the embedding vector for a single piece of text. This is a simplified method that wraps around the `get_embeddings` method.

        Args:
            text (str): The text for which to retrieve the embedding vector.
            **kwargs: Additional arguments to pass to the `get_embeddings` method.

        Returns:
            List[float]: The embedding vector for the provided text.
        """
        result = self.get_embeddings(text, **kwargs)  # ⭐ Calls the get_embeddings method with the given text and additional arguments
        return result['data'][0]['embedding']

    def get_multiple_embeddings(self, texts: Sequence[str], **kwargs) -> List[List[float]]:
        """
        Retrieves the embedding vectors for multiple texts (simplified method).

        Args:
            texts (Sequence[str]): A list of texts to get the embedding vectors for.
            **kwargs: Additional arguments to pass to the `get_embeddings` method.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        result = self.get_embeddings(texts, **kwargs)  # ⭐ Calls the `get_embeddings` method with provided texts and additional arguments
        return [item['embedding'] for item in result['data']]  # ⭐ Extracts the 'embedding' field from each item in the returned data

    def set_model(self, model_name: str):
        """
        Sets the default model name for the API client.

        Args:
            model_name (str): The name of the model to be used.
        """
        self.model_name = model_name  # ⭐ Set the model name

    def set_base_url(self, base_url: str):
        """
        Sets the base URL for the API, ensuring it does not end with a trailing slash.

        Args:
            base_url (str): The base URL for the API.
        """
        self.base_url = base_url.rstrip('/')  # ⭐ Remove trailing slash if present

    def set_api_key(self, api_key: str):
        """
        Sets the API key and updates the authorization header for the API requests.

        Args:
            api_key (str): The API key for authentication.
        """
        self.api_key = api_key
        self.headers["Authorization"] = f"Bearer {self.api_key}"  # ⭐ Update the authorization header

class EmbeddingClient:
    def __init__(self, similarity_threshold: float, base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                 api_key: Optional[str] = None, model: str = "text-embedding-v4",
                 chroma_db_path: str = "./chroma_db", collection_name: str = "trajectories"):
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        assert api_key is not None, "DASHSCOPE_API_KEY is required"

        self._client = OpenAIEmbeddingClient(api_key=api_key, base_url=base_url, model_name=model)
        self.similarity_threshold = similarity_threshold

        self._chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add(self, text: str, id: int):
        """
        Add text and ID to ChromaDB
        """
        embedding = self._client.get_single_embedding(text)

        chroma_id = f"doc_{id}_{uuid.uuid4().hex[:8]}"

        self._collection.add(
            embeddings=[embedding],
            documents=[text],
            ids=[chroma_id],
            metadatas=[{"original_id": id, "text_length": len(text)}]
        )
    
    def find_by_text(self, text: str) -> Optional[int]:
        """
        Find a similar text in ChromaDB, return the corresponding ID
        """
        if self._collection.count() == 0:
            return None

        query_embedding = self._client.get_single_embedding(text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=1,  # only the top result
            include=["documents", "metadatas", "distances"]
        )

        if not results["ids"] or not results["ids"][0]:
            return None

        distance = results["distances"][0][0] # type: ignore
        similarity = 1 - distance

        if similarity >= self.similarity_threshold:
            # Get the original_id from metadata instead of using _reverse_id_mapping
            metadata = results["metadatas"][0][0] # type: ignore
            return cast(int|None,metadata.get("original_id"))
        else:
            return None
    
    def find_top_k_by_text(self, text: str, k: int = 5) -> list[tuple[int, float, str]]:
        """
        Find the top k similar documents
        """
        if self._collection.count() == 0:
            return []

        query_embedding = self._client.get_single_embedding(text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        result_list = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i] # type: ignore
            similarity = 1 - distance
            document = results["documents"][0][i] # type: ignore
            # Get the original_id from metadata instead of using _reverse_id_mapping
            metadata = results["metadatas"][0][i] # type: ignore
            original_id = metadata.get("original_id")

            if original_id is not None:
                result_list.append((original_id, similarity, document))

        return result_list
    
    def _embedding(self, texts: Sequence[str], bs=10) -> list[list[float]]:
        """
        Get the embedding of texts
        """
        res: list[list[float]] = []
        for i in range(0, len(texts), bs):
            res.extend(self._client.get_multiple_embeddings(texts[i:i+bs]))
        
        return res
    
    def get_all_stored_texts(self) -> dict[int, str]:
        """
        Get all stored texts
        """
        all_data = self._collection.get(include=["documents", "metadatas"])
        result = {}

        if all_data["ids"]:
            for i in range(len(all_data["ids"])):
                # Get the original_id from metadata instead of using _reverse_id_mapping
                metadata = all_data["metadatas"][i] # type: ignore
                original_id = metadata.get("original_id")
                if original_id is not None:
                    result[original_id] = all_data["documents"][i] # type: ignore

        return result
    
    def exists(self, id: int) -> bool:
        """
        Check if the ID exists
        """
        results = self._collection.get(
            where={"original_id": id},
            include=[]
        )
        return bool(results["ids"])

    def remove(self, id: int) -> bool:
        """
        Remove the text and embedding vector of the specified ID
        """
        # Find the chroma_id by querying for the document with the specified original_id
        results = self._collection.get(
            where={"original_id": id},
            include=["metadatas"]
        )

        if not results["ids"] or not results["ids"][0]:
            return False

        chroma_id = results["ids"][0]

        try:
            self._collection.delete(ids=[chroma_id])
            return True
        except Exception:
            return False
    
    def clear(self):
        """clear all stored texts and embeddings"""
        try:
            self._chroma_client.delete_collection(self._collection.name)
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._collection.name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"failed to clear stores: {e}")
    
    def size(self) -> int:
        """get the number of stored texts"""
        return self._collection.count()
    
    def get_collection_info(self) -> dict:
        """get the collection info of ChromaDB"""
        return {
            "name": self._collection.name,
            "count": self._collection.count(),
            "metadata": self._collection.metadata
        }