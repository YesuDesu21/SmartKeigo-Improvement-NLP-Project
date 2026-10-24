from typing import List, Literal
from pydantic import BaseModel, Field

Register = Literal["teineigo", "sonkeigo", "kenjogo", "no_change", "unknown"]


class ActorAction(BaseModel):
    actor: Literal["speaker", "in_group", "addressee", "out_group_third_party", "unknown"]
    action: str = Field(description="The verb phrase from the input this actor performs")
    needed_register: Literal["sonkeigo", "kenjogo", "teineigo", "none"] = Field(
        description="sonkeigo raises the actor; kenjogo lowers the speaker/in-group; teineigo is addressee-directed")


class AnalysisResult(BaseModel):
    addressee: Literal["client", "superior", "coworker", "subordinate", "unknown"]
    intent: str = Field(description="Communicative goal in one short sentence")
    tone: str = Field(description="Current tone of the input, e.g. casual, blunt, neutral")
    actions: List[ActorAction]
    overall_register: Register
    unknowns: List[str] = Field(default_factory=list, description="Facts that could not be determined from the input")


class DraftResult(BaseModel):
    keigo_text: str = Field(description="ONLY the converted Japanese text, no commentary")
    explanation: str = Field(description="1-3 short sentences explaining the main honorific choices")


class Issue(BaseModel):
    category: Literal["wrong_direction", "double_honorific", "meaning_change", "register_mismatch", "unnatural", "other"]
    description: str
    suggested_fix: str = ""


class ReviewResult(BaseModel):
    approved: bool
    issues: List[Issue] = Field(default_factory=list)
