from pydantic import BaseModel, Field
from typing import List, Dict, Any


class Document(BaseModel):
    doc_id: str = Field(default="")
    content: str = Field(default="")
    metadata: dict = Field(default_factory=dict)