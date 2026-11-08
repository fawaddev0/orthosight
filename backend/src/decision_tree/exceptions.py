"""Domain errors raised by the standalone decision tree engine."""


class DecisionTreeError(Exception):
    """Base class for decision tree errors."""


class InvalidTreeError(DecisionTreeError):
    """Layer data or cross-layer routing is invalid."""


class InvalidLayerError(DecisionTreeError):
    """A requested layer does not exist."""

    def __init__(self, layer_id: int) -> None:
        super().__init__(f"Layer {layer_id} does not exist")


class NodeNotFoundError(DecisionTreeError):
    """A requested node does not exist."""

    def __init__(self, node_id: str) -> None:
        super().__init__(f"Node {node_id!r} does not exist")


class InvalidChoiceError(DecisionTreeError):
    """A selected answer is not valid for the current node."""

    def __init__(self, node_id: str, answer: str) -> None:
        super().__init__(f"Answer {answer!r} is not valid for node {node_id!r}")
