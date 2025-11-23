"""
Pydantic schemas for NER responses

Used for:
1. Structure validation
2. Generating JSON schemas for structured output
"""

from pydantic import BaseModel, Field
from typing import List


class NEREntity(BaseModel):
    """Single entity in NER response"""
    entity: str = Field(..., description="Entity text")
    types: List[str] = Field(..., description="Entity types/labels")

    class Config:
        schema_extra = {
            "example": {
                "entity": "Star Wars",
                "types": ["title"]
            }
        }


class NERResponse(BaseModel):
    """Complete NER response structure"""
    text: str = Field(..., description="Original input text")
    entities: List[NEREntity] = Field(
        default_factory=list,
        description="Extracted entities"
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "I love Star Wars",
                "entities": [
                    {"entity": "Star Wars", "types": ["title"]}
                ]
            }
        }

    @classmethod
    def get_json_schema(cls) -> dict:
        """Generate JSON schema for structured output APIs"""
        return cls.schema()
