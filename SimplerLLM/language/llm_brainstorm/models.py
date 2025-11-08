"""
Pydantic models for the Recursive Brainstorm feature.

This module defines the data structures used to represent brainstorming
sessions, individual ideas, and results in a structured format.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BrainstormIdea(BaseModel):
    """
    Represents a single brainstormed idea with metadata.

    Attributes:
        id: Unique identifier for the idea
        text: The idea content
        reasoning: Explanation of why this idea was generated
        quality_score: Score from 1-10 evaluating the idea's potential
        depth: Depth level in the brainstorm tree (0 = root level)
        parent_id: ID of the parent idea (None for root-level ideas)
        iteration: Iteration number when this idea was generated
        criteria_scores: Scores for specific evaluation criteria
        metadata: Additional custom metadata
    """
    id: str
    text: str
    reasoning: str = ""
    quality_score: float = Field(default=0.0, ge=0.0, le=10.0)
    depth: int = Field(default=0, ge=0)
    parent_id: Optional[str] = None
    iteration: int = 0
    criteria_scores: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "idea_0_1",
                "text": "Use AI to optimize energy consumption in smart homes",
                "reasoning": "Combines AI capabilities with practical sustainability goals",
                "quality_score": 8.5,
                "depth": 1,
                "parent_id": "idea_0",
                "iteration": 1,
                "criteria_scores": {
                    "feasibility": 8.0,
                    "impact": 9.0,
                    "novelty": 8.0
                }
            }
        }


class IdeaGeneration(BaseModel):
    """
    Structured output from LLM for generating multiple ideas.

    This model is used with generate_pydantic_json_model to ensure
    the LLM returns properly structured brainstorming output.
    """
    ideas: List[str] = Field(description="List of generated ideas")
    reasoning_per_idea: List[str] = Field(
        description="Reasoning for each idea in the same order"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ideas": [
                    "Smart home energy optimization using AI",
                    "Carbon footprint tracking app",
                    "Community solar panel sharing platform"
                ],
                "reasoning_per_idea": [
                    "AI can learn patterns and optimize automatically",
                    "Awareness is the first step to reduction",
                    "Makes renewable energy more accessible"
                ]
            }
        }


class IdeaEvaluation(BaseModel):
    """
    Structured output from LLM for evaluating a single idea.

    Used to score and critique ideas against specific criteria.
    """
    quality_score: float = Field(
        description="Overall quality score from 1-10",
        ge=1.0,
        le=10.0
    )
    strengths: List[str] = Field(description="Key strengths of the idea")
    weaknesses: List[str] = Field(description="Potential weaknesses or challenges")
    criteria_scores: Dict[str, float] = Field(
        description="Scores for specific evaluation criteria",
        default_factory=dict
    )
    should_expand: bool = Field(
        description="Whether this idea is worth expanding further",
        default=True
    )
    reasoning: str = Field(
        description="Overall reasoning for the evaluation",
        default=""
    )

    class Config:
        json_schema_extra = {
            "example": {
                "quality_score": 8.5,
                "strengths": [
                    "Addresses real problem",
                    "Technology is mature",
                    "Clear value proposition"
                ],
                "weaknesses": [
                    "Requires user engagement",
                    "Privacy concerns with data collection"
                ],
                "criteria_scores": {
                    "feasibility": 8.0,
                    "impact": 9.0,
                    "novelty": 8.0
                },
                "should_expand": True,
                "reasoning": "Strong potential with manageable challenges"
            }
        }


class BrainstormLevel(BaseModel):
    """
    Represents all ideas generated at a specific depth level.

    Attributes:
        depth: The depth level (0 = initial prompt level)
        ideas: All ideas generated at this depth
        total_ideas: Number of ideas at this level
        average_score: Average quality score of all ideas
        best_idea: The highest-scoring idea at this level
        execution_time: Time taken to generate this level (seconds)
    """
    depth: int
    ideas: List[BrainstormIdea]
    total_ideas: int
    average_score: float = 0.0
    best_idea: Optional[BrainstormIdea] = None
    execution_time: float = 0.0

    def __init__(self, **data):
        super().__init__(**data)
        if self.ideas:
            self.total_ideas = len(self.ideas)
            scores = [idea.quality_score for idea in self.ideas if idea.quality_score > 0]
            self.average_score = sum(scores) / len(scores) if scores else 0.0
            self.best_idea = max(self.ideas, key=lambda x: x.quality_score, default=None)


class BrainstormIteration(BaseModel):
    """
    Represents a single iteration in the brainstorming process.

    Tracks what happened during one step of idea generation and evaluation.
    """
    iteration_number: int
    depth: int
    parent_idea: Optional[BrainstormIdea] = None
    generated_ideas: List[BrainstormIdea]
    prompt_used: str
    mode_used: str  # "tree", "linear", or "hybrid"
    execution_time: float
    provider_used: str = ""
    model_used: str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "iteration_number": 1,
                "depth": 1,
                "parent_idea": None,
                "generated_ideas": [],
                "prompt_used": "Generate 5 ideas about reducing carbon emissions",
                "mode_used": "tree",
                "execution_time": 2.5,
                "provider_used": "openai",
                "model_used": "gpt-4o"
            }
        }


class BrainstormResult(BaseModel):
    """
    Complete result of a recursive brainstorming session.

    Contains all generated ideas organized by level, execution metadata,
    and analysis of the brainstorming session.

    Attributes:
        initial_prompt: The original prompt that started the brainstorm
        mode: The generation mode used ("tree", "linear", or "hybrid")
        total_ideas: Total number of ideas generated
        total_iterations: Total number of LLM calls made
        max_depth_reached: Maximum depth level achieved
        levels: Ideas organized by depth level
        all_ideas: Flat list of all ideas for easy access
        best_ideas_per_level: Best idea from each depth level
        overall_best_idea: Highest-scoring idea across all levels
        all_iterations: Detailed log of each iteration
        execution_time: Total time taken (seconds)
        stopped_reason: Why the brainstorming stopped
        tree_structure: Hierarchical representation of ideas
        config_used: Configuration parameters used
        timestamp: When the brainstorming session occurred
    """
    initial_prompt: str
    mode: str  # "tree", "linear", or "hybrid"
    total_ideas: int
    total_iterations: int
    max_depth_reached: int
    levels: List[BrainstormLevel] = Field(default_factory=list)
    all_ideas: List[BrainstormIdea] = Field(default_factory=list)
    best_ideas_per_level: List[BrainstormIdea] = Field(default_factory=list)
    overall_best_idea: Optional[BrainstormIdea] = None
    all_iterations: List[BrainstormIteration] = Field(default_factory=list)
    execution_time: float
    stopped_reason: str
    tree_structure: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Parent ID -> List of child IDs mapping"
    )
    config_used: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    def get_children(self, idea_id: str) -> List[BrainstormIdea]:
        """Get all child ideas of a given idea."""
        child_ids = self.tree_structure.get(idea_id, [])
        return [idea for idea in self.all_ideas if idea.id in child_ids]

    def get_ideas_at_depth(self, depth: int) -> List[BrainstormIdea]:
        """Get all ideas at a specific depth level."""
        return [idea for idea in self.all_ideas if idea.depth == depth]

    def get_path_to_best(self) -> List[BrainstormIdea]:
        """Get the path from root to the overall best idea."""
        if not self.overall_best_idea:
            return []

        path = [self.overall_best_idea]
        current = self.overall_best_idea

        while current.parent_id:
            parent = next((idea for idea in self.all_ideas if idea.id == current.parent_id), None)
            if parent:
                path.insert(0, parent)
                current = parent
            else:
                break

        return path

    def to_tree_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a nested dictionary representation.

        Returns a hierarchical dictionary that can be easily serialized to JSON
        or visualized as a tree.
        """
        def build_tree(idea: BrainstormIdea) -> Dict[str, Any]:
            children = self.get_children(idea.id)
            return {
                "id": idea.id,
                "text": idea.text,
                "score": idea.quality_score,
                "depth": idea.depth,
                "reasoning": idea.reasoning,
                "children": [build_tree(child) for child in children]
            }

        root_ideas = self.get_ideas_at_depth(0)
        return {
            "prompt": self.initial_prompt,
            "mode": self.mode,
            "total_ideas": self.total_ideas,
            "roots": [build_tree(idea) for idea in root_ideas]
        }

    def to_csv(self, file_path: str, expand_criteria: bool = True) -> None:
        """
        Export all brainstormed ideas to a CSV file.

        Creates a flat table with all ideas and their metadata. Criteria scores
        can be expanded into separate columns or kept as a JSON string.

        Args:
            file_path: Path to the CSV file to create
            expand_criteria: If True, creates separate columns for each criterion
                           (e.g., feasibility_score, impact_score). If False, stores
                           all criteria as a single JSON string column.

        Example:
            >>> result = brainstorm.brainstorm("Generate ideas")
            >>> result.to_csv("ideas.csv")  # With expanded criteria
            >>> result.to_csv("ideas.csv", expand_criteria=False)  # Compact format

        CSV Columns (with expand_criteria=True):
            - id: Unique idea identifier
            - text: The idea text
            - quality_score: Overall quality score (1-10)
            - depth: Depth level in the tree
            - parent_id: ID of parent idea (empty for root ideas)
            - iteration: Iteration number when generated
            - reasoning: Why this idea was generated
            - [criterion]_score: One column per evaluation criterion

        CSV Columns (with expand_criteria=False):
            - Same as above, but criteria_scores is a single JSON string column
        """
        import csv
        import json

        # Collect all unique criteria across all ideas
        all_criteria = set()
        if expand_criteria:
            for idea in self.all_ideas:
                all_criteria.update(idea.criteria_scores.keys())
            all_criteria = sorted(all_criteria)  # For consistent column order

        # Define CSV columns
        base_columns = [
            "id",
            "text",
            "quality_score",
            "depth",
            "parent_id",
            "iteration",
            "reasoning",
        ]

        if expand_criteria and all_criteria:
            # Add a column for each criterion
            criteria_columns = [f"{criterion}_score" for criterion in all_criteria]
            columns = base_columns + criteria_columns
        else:
            # Single column for all criteria
            columns = base_columns + ["criteria_scores"]

        # Write CSV file
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()

            for idea in self.all_ideas:
                row = {
                    "id": idea.id,
                    "text": idea.text,
                    "quality_score": idea.quality_score,
                    "depth": idea.depth,
                    "parent_id": idea.parent_id or "",
                    "iteration": idea.iteration,
                    "reasoning": idea.reasoning,
                }

                if expand_criteria and all_criteria:
                    # Add each criterion as a separate column
                    for criterion in all_criteria:
                        score = idea.criteria_scores.get(criterion, "")
                        row[f"{criterion}_score"] = score
                else:
                    # Add criteria as JSON string
                    row["criteria_scores"] = json.dumps(idea.criteria_scores) if idea.criteria_scores else ""

                writer.writerow(row)

    class Config:
        json_schema_extra = {
            "example": {
                "initial_prompt": "Generate ideas to reduce carbon emissions",
                "mode": "tree",
                "total_ideas": 15,
                "total_iterations": 3,
                "max_depth_reached": 2,
                "execution_time": 12.5,
                "stopped_reason": "max_depth_reached"
            }
        }
