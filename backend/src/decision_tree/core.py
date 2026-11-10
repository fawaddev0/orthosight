"""Read-only loader and lookup interface for the four decision tree layers."""

import re
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
from .models import (
    DiagnosisNode,
    Layer,
    Node,
    QuestionnaireEvalNode,
    QuestionNode,
    ScreeningNode,
)
from .scoring import calculate_fabq_scores, calculate_quickdash_score


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
            "3910": "4100",  # Adhesive capsulitis -> FABQ evaluation
            "3920": "4200",  # Full-thickness tear -> QuickDASH evaluation
            "3930": "4200",  # Partial tear -> QuickDASH evaluation
            "3940": "4200",  # Impingement -> QuickDASH evaluation
            "3950": "4100",  # Indeterminate -> FABQ evaluation
            "3960": "4200",  # AC joint -> QuickDASH evaluation
            "3970": "4200",  # Labral tear -> QuickDASH evaluation
            "3980": "4200",  # Instability -> QuickDASH evaluation
            "3990": "4100",  # GH OA -> FABQ evaluation
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

    def get_next_node(
        self,
        node_id: str,
        answer: str | None = None,
        eval_scores: dict[str, float] | None = None,
    ) -> Node | None:
        """Resolve an answer or diagnosis bridge to the next node."""
        node = self.get_node(node_id)
        target_node_id: str | None
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
        elif isinstance(node, QuestionnaireEvalNode):
            if node.rules is not None:
                target_node_id = self._resolve_questionnaire_rule(node, answer, eval_scores)
            else:
                target_node_id = self._resolve_questionnaire_options(node, answer, eval_scores)
            if target_node_id is None:
                return None
        else:
            if answer is not None or eval_scores is not None:
                raise InvalidChoiceError(node_id, answer or "unexpected evaluation scores")
            target_node_id = self._BRIDGES.get(node.id)
            if target_node_id is None:
                return None
        return self.get_node(target_node_id)

    @staticmethod
    def _resolve_questionnaire_rule(
        node: QuestionnaireEvalNode,
        answer: str | None,
        eval_scores: dict[str, float] | None,
    ) -> str | None:
        """Resolve an explicit condition or evaluate score comparisons."""
        rules = node.rules
        if rules is None:
            raise InvalidChoiceError(node.id, "missing questionnaire rules")
        default_target: str | None = None
        for rule in rules:
            if rule.condition == "default":
                default_target = rule.next_node_id
                continue
            if answer is not None and rule.condition == answer:
                return rule.next_node_id

        if eval_scores is None:
            for rule in rules:
                if rule.condition in {"fabq_pa", "fabq_w", "score"}:
                    raise InvalidChoiceError(node.id, f"missing score {rule.condition!r}")
                match = re.fullmatch(
                    r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|!=|>|<)\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*",
                    rule.condition,
                )
                if match is not None:
                    raise InvalidChoiceError(node.id, f"missing score {match.group(1)!r}")
            if default_target is not None:
                return default_target
            raise InvalidChoiceError(node.id, answer or "missing evaluation scores")

        for rule in rules:
            if rule.condition == "default":
                continue
            match = re.fullmatch(
                r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|!=|>|<)\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*",
                rule.condition,
            )
            if match is None:
                if rule.condition in eval_scores:
                    return rule.next_node_id
                if rule.condition in {"fabq_pa", "fabq_w", "score"}:
                    raise InvalidChoiceError(node.id, f"missing score {rule.condition!r}")
                continue

            score_name, operator, threshold_text = match.groups()
            if score_name not in eval_scores:
                raise InvalidChoiceError(node.id, f"missing score {score_name!r}")
            score = eval_scores[score_name]
            threshold = float(threshold_text)
            comparisons = {
                ">=": score >= threshold,
                "<=": score <= threshold,
                "==": score == threshold,
                "!=": score != threshold,
                ">": score > threshold,
                "<": score < threshold,
            }
            if comparisons[operator]:
                return rule.next_node_id

        if default_target is not None:
            return default_target
        raise InvalidChoiceError(node.id, "no questionnaire rule matched")

    @staticmethod
    def _resolve_questionnaire_options(
        node: QuestionnaireEvalNode,
        answer: str | None,
        eval_scores: dict[str, float] | None,
    ) -> str:
        """Resolve questionnaire options using answer or scores described by labels."""
        options = node.options
        if options is None or node.question_text is None:
            raise InvalidChoiceError(node.id, "invalid questionnaire definition")
        if answer is not None:
            for option in options:
                if option.value == answer:
                    return option.next_node_id
            raise InvalidChoiceError(node.id, answer)
        if eval_scores is None:
            raise InvalidChoiceError(node.id, "missing evaluation scores")

        physical_match = re.search(r"physical activity score above (\d+)", node.question_text)
        work_match = re.search(r"work score above (\d+)", node.question_text)
        if physical_match is not None and work_match is not None:
            if len(options) < 2:
                raise InvalidChoiceError(node.id, "questionnaire requires two result options")
            for score_name in ("fabq_pa", "fabq_w"):
                if score_name not in eval_scores:
                    raise InvalidChoiceError(node.id, f"missing score {score_name!r}")
            is_high = (
                eval_scores["fabq_pa"] > float(physical_match.group(1))
                or eval_scores["fabq_w"] > float(work_match.group(1))
            )
            selected_option = options[0] if is_high else options[1]
            return selected_option.next_node_id

        if "score" not in eval_scores:
            raise InvalidChoiceError(node.id, "missing score 'score'")
        score = eval_scores["score"]
        for option in options:
            below_match = re.search(r"score below (\d+(?:\.\d+)?)", option.label)
            above_match = re.search(r"score above (\d+(?:\.\d+)?)", option.label)
            range_match = re.search(
                r"score (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?)", option.label
            )
            if below_match is not None and score < float(below_match.group(1)):
                return option.next_node_id
            if above_match is not None and score > float(above_match.group(1)):
                return option.next_node_id
            if range_match is not None and float(range_match.group(1)) <= score <= float(
                range_match.group(2)
            ):
                return option.next_node_id
        raise InvalidChoiceError(node.id, f"score {score}")

    def _validate_graph(self) -> None:
        for layer_id, layer in self._layers.items():
            for node in layer.nodes.values():
                if isinstance(node, ScreeningNode):
                    targets = [node.default_next_node_id, *(flag.next_node_id for flag in node.red_flags)]
                    choices = [flag.id for flag in node.red_flags]
                elif isinstance(node, QuestionNode):
                    targets = [option.next_node_id for option in node.options]
                    choices = [option.value for option in node.options]
                elif isinstance(node, QuestionnaireEvalNode):
                    if node.rules is not None:
                        targets = [rule.next_node_id for rule in node.rules]
                        choices = [rule.condition for rule in node.rules]
                    elif node.options is not None:
                        targets = [option.next_node_id for option in node.options]
                        choices = [option.value for option in node.options]
                    else:
                        raise InvalidTreeError(f"Node {node.id!r} has no routing rules")
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
            if not isinstance(target_node, (QuestionNode, QuestionnaireEvalNode)):
                raise InvalidTreeError(f"Bridge target {target!r} is not a routable node")
            if self._node_layers[target] != self._node_layers[source] + 1:
                raise InvalidTreeError(f"Bridge {source!r} to {target!r} skips a layer")

        for layer_id in (2, 3):
            for node in self._layers[layer_id].nodes.values():
                if isinstance(node, DiagnosisNode) and node.id not in self._BRIDGES:
                    raise InvalidTreeError(f"Layer {layer_id} diagnosis {node.id!r} has no bridge")

def evaluate_step(
    tree: DecisionTree,
    current_node_id: str,
    answer: str | None = None,
    fabq_answers: dict[int, int] | None = None,
    quickdash_answers: dict[int, int] | None = None,
    session_state: dict[str, str | bool] | None = None,
) -> Node | None:
    """Evaluate one decision tree step, including questionnaire scores."""
    current_node = tree.get_node(current_node_id)
    if isinstance(current_node, QuestionnaireEvalNode):
        questionnaire_id = (current_node.questionnaire_id or "").upper()
        # The current tree data omits questionnaireId on its questionnaire
        # nodes. In that case, infer the calculator from the one answer set
        # the client submitted; explicit IDs still take precedence.
        if questionnaire_id == "FABQ" or (
            not questionnaire_id
            and fabq_answers is not None
            and quickdash_answers is None
        ):
            if fabq_answers is None:
                raise InvalidChoiceError(current_node_id, "missing FABQ answers")
            eval_scores = calculate_fabq_scores(fabq_answers)
        elif questionnaire_id == "QUICKDASH" or (
            not questionnaire_id
            and quickdash_answers is not None
            and fabq_answers is None
        ):
            if quickdash_answers is None:
                raise InvalidChoiceError(current_node_id, "missing QuickDASH answers")
            eval_scores = calculate_quickdash_score(quickdash_answers)
        elif not questionnaire_id and fabq_answers is not None and quickdash_answers is not None:
            raise InvalidChoiceError(current_node_id, "submit only the answers for this questionnaire")
        elif not questionnaire_id:
            raise InvalidChoiceError(current_node_id, "missing questionnaire answers")
        else:
            raise InvalidChoiceError(
                current_node_id,
                f"unsupported questionnaire {questionnaire_id!r}",
            )
        return tree.get_next_node(current_node_id, eval_scores=eval_scores)

    active_session_state = session_state if session_state is not None else {}
    if answer is not None:
        if current_node_id == "2010":
            active_session_state["age_group"] = answer
        elif current_node_id == "2020":
            active_session_state["prior_dislocation"] = answer == "1"
        elif current_node_id == "2030":
            active_session_state["overhead_activity"] = answer == "1"

    next_node = tree.get_next_node(current_node_id, answer=answer)
    if isinstance(next_node, DiagnosisNode):
        bridge_target = tree.get_next_node(next_node.id)
        if bridge_target is not None:
            if bridge_target.id == "3000":
                return tree.get_node(get_layer3_entry_node(active_session_state))
            return bridge_target
    return next_node


def get_layer3_entry_node(session_state: dict[str, str | bool]) -> str:
    """Select Layer 3 entry using recorded Layer 2 history."""
    if session_state.get("prior_dislocation") is True and session_state.get("age_group") == "1":
        return "3520"
    return "3000"
