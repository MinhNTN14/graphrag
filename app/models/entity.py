"""Pydantic models for the knowledge-graph exploration API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EntityResponse(BaseModel):
    """A single entity node."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique entity name.")
    type: str = Field(description="Entity category, e.g. PERSON or TECHNOLOGY.")
    description: str = Field(description="Short description of the entity.")


class RelationResponse(BaseModel):
    """A single relationship between two entities."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Source entity name.")
    relation: str = Field(description="Relationship label, e.g. USES.")
    target: str = Field(description="Target entity name.")
    description: str = Field(description="Short description of the relationship.")


class SubgraphResponse(BaseModel):
    """A subgraph centred on a single entity."""

    model_config = ConfigDict(extra="forbid")

    entity: EntityResponse = Field(description="The centre entity.")
    neighbors: list[EntityResponse] = Field(
        default_factory=list, description="Directly and transitively connected entities."
    )
    relationships: list[RelationResponse] = Field(
        default_factory=list, description="Relationships within the subgraph."
    )
