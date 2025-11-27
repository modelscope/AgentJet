from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str = Field(default="")
    content: str = Field(default="")
    metadata: dict = Field(default_factory=dict)
