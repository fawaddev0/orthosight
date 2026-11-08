"""Read-only loader and lookup interface for the four decision tree layers."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from pydantic import ValidationError

from .exceptions import (
    InvalidChoiceError,
    InvalidLayerError,
    InvalidTreeError,
    NodeNotFoundError,
)
from .models import DiagnosisNode, Layer, Node, QuestionNode, ScreeningNode


class DecisionTree:
    """Load and validate all layers without retaining patient state."""

    _BRIDGES: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "400": "2000",
            "2910": "3000",
            "2920": "3000",
            "2930": "3000",
            "2940": "3000",
            "2950": "3000",
            "2960": "3000",
            "3910": "4100",
            "3920": "4400",
            "3930": "4200",
            "3940": "4200",
            "3950": "4300",
        }
    )

    def __init__(self, data_dir: Path | None = None) -> None:
        directory = data_dir or Path(__file__).resolve().parents[3] / "data"
        layers: dict[int, Layer] = {}
        nodes: dict[str, Node] = {}
        node_layers: dict[str, int] = {}

        for layer_id in range(1, 5):
            path = directory / f"layer{layer_id}.json"
            try:
                layer = Layer.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValidationError) as exc:
                raise InvalidTreeError(f"Cannot load valid layer {layer_id} from {path}") from exc
            if layer.id != str(layer_id):
                raise InvalidTreeError(f"Layer {layer_id} has ID {layer.id!r}")
            if layer.root_node_id not in layer.nodes:
                raise InvalidTreeError(f"Layer {layer_id} root {layer.root_node_id!r} does not exist")

            for node_id, node in layer.nodes.items():
                if node_id != node.id:
                    raise InvalidTreeError(f"Node key {node_id!r} differs from node ID {node.id!r}")
                if node_id in nodes:
                    raise InvalidTreeError(f"Duplicate node ID {node_id!r}")
                nodes[node_id] = node
                node_layers[node_id] = layer_id
            layers[layer_id] = layer

        self._layers = layers
        self._nodes = nodes
        self._node_layers = node_layers
        self._validate_graph()

    def get_node(self, node_id: str) -> Node:
        """Return a validated node or raise ``NodeNotFoundError``."""
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise NodeNotFoundError(node_id) from exc

    def get_layer(self, layer_id: int) -> Layer:
        """Return a copy of a validated layer or raise ``InvalidLayerError``."""
        try:
            return self._layers[layer_id].model_copy(deep=True)
        except KeyError as exc:
            raise InvalidLayerError(layer_id) from exc

    def get_root_node(self, layer_id: int = 1) -> Node:
        """Return the root node for a layer."""
        layer = self.get_layer(layer_id)
        return self.get_node(layer.root_node_id)

    def get_next_node(self, node_id: str, answer: str | None = None) -> Node | None:
        """Resolve an answer or diagnosis bridge to the next node."""
        node = self.get_node(node_id)
        if isinstance(node, ScreeningNode):
            if answer is None:
                return None
            if answer == "none":
                target_node_id = node.default_next_node_id
            else:
                flag = next((flag for flag in node.red_flags if flag.id == answer), None)
                if flag is None:
                    raise InvalidChoiceError(node_id, answer)
                target_node_id = flag.next_node_id
        elif isinstance(node, QuestionNode):
            if answer is None:
                return None
            option = next((option for option in node.options if option.value == answer), None)
            if option is None:
                raise InvalidChoiceError(node_id, answer)
            target_node_id = option.next_node_id
        else:
            if answer is not None:
                raise InvalidChoiceError(node_id, answer)
            target_node_id = self._BRIDGES.get(node.id)
            if target_node_id is None:
                return None
        return self.get_node(target_node_id)

    def _validate_graph(self) -> None:
        for layer_id, layer in self._layers.items():
            for node in layer.nodes.values():
                if isinstance(node, ScreeningNode):
                    targets = [node.default_next_node_id, *(flag.next_node_id for flag in node.red_flags)]
                    choices = [flag.id for flag in node.red_flags]
                elif isinstance(node, QuestionNode):
                    targets = [option.next_node_id for option in node.options]
                    choices = [option.value for option in node.options]
                else:
                    continue

                if len(choices) != len(set(choices)):
                    raise InvalidTreeError(f"Node {node.id!r} has duplicate choices")
                for target in targets:
                    if target not in self._nodes:
                        raise InvalidTreeError(f"Node {node.id!r} references missing node {target!r}")
                    if self._node_layers[target] != layer_id:
                        raise InvalidTreeError(f"Node {node.id!r} crosses layers without a bridge")

        for source, target in self._BRIDGES.items():
            source_node = self._nodes.get(source)
            target_node = self._nodes.get(target)
            if not isinstance(source_node, DiagnosisNode):
                raise InvalidTreeError(f"Bridge source {source!r} is not a diagnosis node")
            if not isinstance(target_node, QuestionNode):
                raise InvalidTreeError(f"Bridge target {target!r} is not a question node")
            if self._node_layers[target] != self._node_layers[source] + 1:
                raise InvalidTreeError(f"Bridge {source!r} to {target!r} skips a layer")

        for layer_id in (2, 3):
            for node in self._layers[layer_id].nodes.values():
                if isinstance(node, DiagnosisNode) and node.id not in self._BRIDGES:
                    raise InvalidTreeError(f"Layer {layer_id} diagnosis {node.id!r} has no bridge")

        dispatch = self._nodes.get("4000")
        if not isinstance(dispatch, QuestionNode):
            raise InvalidTreeError("Layer 4 dispatch node '4000' is missing")
        branch_targets = {target for source, target in self._BRIDGES.items() if self._node_layers[source] == 3}
        if branch_targets != {option.next_node_id for option in dispatch.options}:
            raise InvalidTreeError("Layer 3 bridges disagree with Layer 4 dispatch options")
