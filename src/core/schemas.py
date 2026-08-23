
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

class SourcePassage(BaseModel):
    text: str = Field(description="Content text snippet of the chunk.")
    page_number: int = Field(default=0, description="PDF page number of the source chunk.")
    section_title: str = Field(default="General", description="Section or chapter heading.")
    score: float = Field(default=0.0, description="Composite/primary score.")
    vector_score: float = Field(default=0.0, description="Similarity score from Qdrant vector search.")
    rerank_score: float = Field(default=0.0, description="Relevance score from LLM cross-encoder reranker.")


class ResearchOutput(BaseModel):
    draft_answer: str = Field(description="Drafted text answer based on context passages.")
    sources: List[SourcePassage] = Field(default_factory=list, description="List of source passages referenced.")
    is_refusal: bool = Field(default=False, description="True if question could not be answered from context.")


class ReviewerOutput(BaseModel):
    verdict: Literal["APPROVED", "REJECTED"] = Field(description="Review verdict ('APPROVED' or 'REJECTED').")
    reasoning: str = Field(default="", description="Auditor reasoning explaining the verdict.")
    unsupported_claims: List[str] = Field(default_factory=list, description="List of any unsupported statements.")
    feedback_for_researcher: str = Field(default="", description="Feedback for revision if REJECTED.")


class AgentState(BaseModel):
    question: str = Field(description="Original user question.")
    passages: List[SourcePassage] = Field(default_factory=list, description="Retrieved Qdrant passages.")
    draft_answer: str = Field(default="", description="Latest draft answer from Researcher.")
    reviewer_verdict: str = Field(default="PENDING", description="Verdict from Reviewer ('APPROVED' / 'REJECTED' / 'PENDING').")
    reviewer_feedback: str = Field(default="", description="Feedback from Reviewer if rejected.")
    iteration_count: int = Field(default=0, description="Number of handoff iterations executed.")
    max_iterations: int = Field(default=2, description="Maximum allowed feedback handoff iterations.")
    final_answer: str = Field(default="", description="Final verified answer presented to user.")
    is_refusal: bool = Field(default=False, description="True if final response is a refusal.")
