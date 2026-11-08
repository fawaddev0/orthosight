"""HTTP API for serving the complete clinical decision tree."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from decision_tree import (
    DecisionTree,
    DiagnosisNode,
    InvalidChoiceError,
    Node,
    NodeNotFoundError,
)


class DecisionTreeRequest(BaseModel):
    """Current node and selected answer; omit both to start at the root."""

    model_config = ConfigDict(extra="forbid")

    node_id: str | None = Field(default=None, alias="nodeId")
    answer: str | None = None


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
        next_node = decision_tree.get_next_node(node.id, request.answer)
    except InvalidChoiceError as exc:
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
