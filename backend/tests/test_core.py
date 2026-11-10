import hashlib
import json
import shutil
from pathlib import Path

import pytest

from decision_tree.core import DecisionTree
from decision_tree.exceptions import (
    InvalidChoiceError,
    InvalidLayerError,
    InvalidTreeError,
    NodeNotFoundError,
)
from decision_tree.models import (
    DiagnosisNode,
    QuestionnaireEvalNode,
    QuestionNode,
    ScreeningNode,
)

SOURCE_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _copy_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for source in SOURCE_DATA_DIR.glob("layer*.json"):
        shutil.copyfile(source, data_dir / source.name)
    return data_dir


def _change_node(data_dir: Path, layer: int, node_id: str, change: dict[str, object]) -> None:
    path = data_dir / f"layer{layer}.json"
    payload = json.loads(path.read_text())
    payload["nodes"][node_id].update(change)
    path.write_text(json.dumps(payload))


def test_loads_all_layers_without_changing_source_data() -> None:
    paths = sorted(SOURCE_DATA_DIR.glob("layer*.json"))
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    tree = DecisionTree()

    assert sum(len(tree.get_layer(layer_id).nodes) for layer_id in range(1, 5)) == 44
    assert isinstance(tree.get_node("100"), ScreeningNode)
    assert isinstance(tree.get_node("200"), QuestionNode)
    assert isinstance(tree.get_node("999"), DiagnosisNode)
    assert tree.get_layer(4).root_node_id == "4000"
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_lookup_errors_are_specific() -> None:
    tree = DecisionTree()

    with pytest.raises(NodeNotFoundError):
        tree.get_node("missing")
    with pytest.raises(InvalidLayerError):
        tree.get_layer(5)


def test_layer_lookup_cannot_change_loaded_tree() -> None:
    tree = DecisionTree()
    layer = tree.get_layer(1)
    layer.nodes.pop("100")
    assert tree.get_node("100").id == "100"


def test_rejects_malformed_node(tmp_path: Path) -> None:
    data_dir = _copy_data(tmp_path)
    _change_node(data_dir, 1, "200", {"options": "invalid"})

    with pytest.raises(InvalidTreeError):
        DecisionTree(data_dir)


def test_rejects_missing_target(tmp_path: Path) -> None:
    data_dir = _copy_data(tmp_path)
    path = data_dir / "layer1.json"
    payload = json.loads(path.read_text())
    payload["nodes"]["200"]["options"][0]["nextNodeId"] = "missing"
    path.write_text(json.dumps(payload))

    with pytest.raises(InvalidTreeError, match="missing"):
        DecisionTree(data_dir)


def test_rejects_duplicate_node_ids(tmp_path: Path) -> None:
    data_dir = _copy_data(tmp_path)
    path = data_dir / "layer2.json"
    payload = json.loads(path.read_text())
    payload["nodes"]["100"] = json.loads((data_dir / "layer1.json").read_text())["nodes"]["100"]
    path.write_text(json.dumps(payload))

    with pytest.raises(InvalidTreeError, match="100"):
        DecisionTree(data_dir)


def test_rejects_invalid_bridge_source(tmp_path: Path) -> None:
    data_dir = _copy_data(tmp_path)
    path = data_dir / "layer1.json"
    payload = json.loads(path.read_text())
    payload["nodes"]["400"] = payload["nodes"]["200"] | {"id": "400"}
    path.write_text(json.dumps(payload))

    with pytest.raises(InvalidTreeError, match="400"):
        DecisionTree(data_dir)


def test_questionnaire_eval_node_routing() -> None:
    tree = DecisionTree()

    assert isinstance(tree.get_node("2410"), QuestionnaireEvalNode)
    assert tree.get_next_node("2410", eval_scores={"fabq_pa": 16, "fabq_w": 10}).id == "2960"
    assert tree.get_next_node("2410", eval_scores={"fabq_pa": 5, "fabq_w": 10}).id == "2950"


def test_questionnaire_eval_raises_on_missing_or_invalid_scores() -> None:
    tree = DecisionTree()

    with pytest.raises(InvalidChoiceError):
        tree.get_next_node("2410", eval_scores={})
    with pytest.raises(InvalidChoiceError):
        tree.get_next_node("2410", answer="invalid")
