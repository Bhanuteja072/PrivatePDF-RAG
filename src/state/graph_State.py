from typing import TypedDict, List
from pydantic import BaseModel, Field


class GraphState(TypedDict):
    """ Represents the state of our graph.

        Attributes:
                question: question
                generation: LLM generation
                documents: list of documents
                failures: how many times generation was judged not useful/not supported
    """
    question:str
    generation:str
    documents:List[str]
    failures:int


   
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


### Hallucination Grader

class HallucinationCheck(BaseModel):
    """ Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
  )



# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
