"""HTTP API for serving the complete clinical decision tree."""

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from decision_tree import (
    DecisionTree,
    DiagnosisNode,
    InvalidChoiceError,
    Node,
    NodeNotFoundError,
    evaluate_step,
)
from rag.response_generator import AnswerRequest, PlanResponse, generate_plan
from rag.retriever import CONDITION_QUERIES

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
logger = logging.getLogger(__name__)


class DecisionTreeRequest(BaseModel):
    """Current node and selected answer; omit both to start at the root."""

    model_config = ConfigDict(extra="forbid")

    node_id: str | None = Field(default=None, alias="nodeId")
    answer: str | None = None
    fabq_answers: dict[int, int] | None = Field(default=None, alias="fabqAnswers")
    quickdash_answers: dict[int, int] | None = Field(default=None, alias="quickdashAnswers")


class DecisionTreeRoute(BaseModel):
    """The edge followed by the selected answer or diagnosis bridge."""

    model_config = ConfigDict(populate_by_name=True)

    source_node_id: str = Field(alias="sourceNodeId")
    answer: str | None
    target_node_id: str = Field(alias="targetNodeId")


class DecisionTreeResponse(BaseModel):
    """Current decision node and any resulting route."""

    model_config = ConfigDict(populate_by_name=True)

    node: Node
    next_node: Node | None = Field(alias="nextNode")
    route: DecisionTreeRoute | None
    diagnosis: DiagnosisNode | None
    complete: bool


app = FastAPI(title="OrthoSight API")
decision_tree = DecisionTree()


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Return a lightweight health response without calling external services."""
    return {"status": "ok"}


@app.post("/api/decision-tree", response_model=DecisionTreeResponse)
def get_decision_tree(request: DecisionTreeRequest) -> DecisionTreeResponse:
    """Return a decision node and advance the tree when an answer is provided."""
    if request.node_id is None:
        if request.answer is not None:
            raise HTTPException(status_code=422, detail="nodeId is required with answer")
        node = decision_tree.get_root_node()
    else:
        try:
            node = decision_tree.get_node(request.node_id)
        except NodeNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        next_node = evaluate_step(
            decision_tree,
            node.id,
            answer=request.answer,
            fabq_answers=request.fabq_answers,
            quickdash_answers=request.quickdash_answers,
        )
    except (InvalidChoiceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    route = (
        DecisionTreeRoute(
            source_node_id=node.id,
            answer=request.answer,
            target_node_id=next_node.id,
        )
        if next_node is not None
        else None
    )
    diagnosis: DiagnosisNode | None = None
    if isinstance(node, DiagnosisNode) and next_node is None:
        diagnosis = node
    elif isinstance(next_node, DiagnosisNode):
        if decision_tree.get_next_node(next_node.id) is None:
            diagnosis = next_node

    return DecisionTreeResponse(
        node=node,
        next_node=next_node,
        route=route,
        diagnosis=diagnosis,
        complete=diagnosis is not None,
    )


@app.post("/api/exercise-plan", response_model=PlanResponse)
def get_exercise_plan(request: AnswerRequest) -> PlanResponse:
    """Return evidence-backed exercises for diagnoses with a home-plan route."""
    if request.diagnosis.id not in CONDITION_QUERIES:
        raise HTTPException(
            status_code=422,
            detail=f"Diagnosis {request.diagnosis.id} is not eligible for a home exercise plan",
        )
    try:
        return generate_plan(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Exercise-plan generation failed for diagnosis %s", request.diagnosis.id)
        raise HTTPException(
            status_code=500,
            detail="Exercise-plan generation failed. Check backend logs for details.",
        ) from exc
